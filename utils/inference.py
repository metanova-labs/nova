import atexit
import collections
import math
import multiprocessing as mp
import os
import shutil
import statistics
import subprocess
import sys
import time
from multiprocessing import connection as mp_connection
from typing import NamedTuple

NOVA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(NOVA_DIR)

_thread_vars = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
if not any(_v in os.environ for _v in _thread_vars):
    _threads = os.environ.get("NOVA_BOLTZ_THREADS", "1")
    for _var in _thread_vars:
        os.environ[_var] = _threads

import torch
import bittensor as bt


# How often the parent re-checks the wall deadline while waiting on workers.
_POLL_INTERVAL_S = 5.0
# Grace given to a worker to exit once it has reported (or been killed).
_TEARDOWN_JOIN_S = 30.0
# Subtensor block time, for deriving the deadline from epoch_length.
_BLOCK_TIME_S = 12.0
# Fraction of an epoch inference may consume before workers are killed. The
# remainder has to cover score sharing, ranking, weight setting and payouts.
_EPOCH_DEADLINE_FRACTION = 0.7
# Extra attempts per worker after the first. One is deliberate: a second failure
# is usually the same failure, and each attempt costs a shard's worth of wall
# clock that the epoch does not get back.
_DEFAULT_MAX_RETRIES = 2
# A retry is only started if the remaining budget exceeds this multiple of the
# median observed runtime for that kind of worker. Above 1.0 so a retry that is
# certain to be killed by the deadline is never started.
_RETRY_TIME_MARGIN = 1.5
# Estimate used before any worker of that kind has finished, as a fraction of
# the whole budget. Nothing has been observed yet, so this is what bounds a
# blind retry: it permits one early on and refuses one near the deadline.
_RETRY_BLIND_FRACTION = 0.5
# VRAM a single worker is assumed to need, in MiB. Used two ways: to warn up
# front when the shard layout cannot fit, and to hold a retry back until its
# target GPU actually has room. 6000 comes from observed boltz shards, which
# settle around 5.0-5.3 GiB once the model is resident. Set
# NOVA_WORKER_VRAM_MIB=0 to disable both checks and retry immediately.
_DEFAULT_WORKER_VRAM_MIB = 6000
# How long after spawning a worker its VRAM is treated as claimed even though
# nvidia-smi cannot see it yet. A boltz shard takes tens of seconds to reach
# Lightning's model_to_device, and until it gets there the memory it is about
# to take still reads as free -- so a burst of retries admitted against that
# same reading throws an OOM.
_VRAM_SETTLE_S = 90.0


def _worker_vram_mib() -> int:
    """Assumed per-worker VRAM footprint in MiB. 0 disables capacity checks."""
    raw = os.environ.get("NOVA_WORKER_VRAM_MIB")
    if not raw:
        return _DEFAULT_WORKER_VRAM_MIB
    try:
        return max(0, int(raw))
    except ValueError:
        bt.logging.warning(
            f"Ignoring non-integer NOVA_WORKER_VRAM_MIB={raw!r}; "
            f"using {_DEFAULT_WORKER_VRAM_MIB}."
        )
        return _DEFAULT_WORKER_VRAM_MIB


def _gpu_free_mib() -> dict:
    """Free VRAM per GPU id, or {} if it cannot be read.

    Uses nvidia-smi rather than torch.cuda.mem_get_info on purpose: the parent
    process has no CUDA context and must not create one.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        bt.logging.warning(
            f"Could not read GPU free memory ({type(e).__name__}: {e}); "
            f"falling back to worker-count placement."
        )
        return {}
    free = {}
    for line in out.strip().splitlines():
        idx, _, mib = line.partition(",")
        try:
            free[int(idx)] = int(mib)
        except ValueError:
            continue
    return free


class BoltzResult:
    """stand-in for BoltzWrapper when inference runs in subprocesses"""

    __slots__ = ("per_molecule_components", "unique_molecules")

    def __init__(self, per_molecule_components: dict | None = None, unique_molecules: dict | None = None):
        self.per_molecule_components = per_molecule_components or {}
        self.unique_molecules = unique_molecules or {}

    @property
    def per_molecule_metric(self) -> dict:
        """for backwards compatibility with save_data / apply_external_scores"""
        return self.per_molecule_components


class BoltzgenResult(NamedTuple):
    """stand-in for BoltzgenWrapper when inference runs in subprocesses.

    ``final_boltzgen_scores`` is always None here; the validator merges ranked
    nanobody scores into ``score_dict`` after score sharing.
    """

    per_nanobody_components: dict
    final_boltzgen_scores: dict | None


class InferenceResult(NamedTuple):
    """Result of inference.main(). Merges molecule scores in-place; nanobody ranking/merge is done by the validator."""

    boltz: BoltzResult | None
    boltzgen: BoltzgenResult | None


def _log_inference_summary(results: list, num_shards: int) -> None:
    """Emit one machine-parseable line describing the parallel inference span."""
    stamped = [r for r in results if "t_start" in r and "t_end" in r]
    if not stamped:
        return
    t0 = min(r["t_start"] for r in stamped)
    t1 = max(r["t_end"] for r in stamped)
    span = t1 - t0
    parts = [f"STAGE_SUMMARY inference_parallel span_s={span:.1f} shards={num_shards}"]
    b = [r for r in stamped if "per_molecule_components" in r]
    g = [r for r in stamped if "per_nanobody_components" in r]
    if b:
        each = ",".join(f"{r['t_end'] - r['t_start']:.1f}" for r in sorted(b, key=lambda r: r.get("shard_id", 0)))
        parts.append(f"boltz_total_s={max(r['t_end'] for r in b) - min(r['t_start'] for r in b):.1f}")
        parts.append(f"boltz_shard_s=[{each}]")
    if g:
        parts.append(f"boltzgen_s={g[0]['t_end'] - g[0]['t_start']:.1f}")
    if b and g:
        idle = abs(max(r["t_end"] for r in b) - g[0]["t_end"])
        parts.append(f"gpu_idle_tail_s={idle:.1f}")
    bt.logging.info(" ".join(parts))


def assemble_molecule_scores(score_dict: dict, valid_molecules_by_uid: dict,
                             final_boltz_scores: dict, subnet_config: dict) -> None:
    """Write molecule_scores into score_dict from (possibly merged) shard results.

    Lives here rather than in boltz_wrapper so the parent process can assemble
    once from the union of all shards without importing boltz.
    """
    sentinel = math.inf if subnet_config['boltz_mode'] == "min" else -math.inf
    for uid, data in score_dict.items():
        smiles_list = (valid_molecules_by_uid.get(uid) or {}).get('smiles') or []
        by_target = final_boltz_scores.get(uid) or {}
        data['molecule_scores'] = [
            [(by_target.get(target) or {}).get(s, sentinel) for s in smiles_list]
            for target in subnet_config['small_molecule_target']
        ]


def _stop_bt_log_listener() -> None:
    """Stop bittensor's QueueListener before multiprocessing tears the queue down.
    """
    listener = getattr(bt.logging, "_listener", None)
    if listener is None or getattr(listener, "_thread", None) is None:
        return
    try:
        atexit.unregister(listener.stop)  # stop() is not idempotent
        listener.stop()
    except Exception:
        pass


def _proc_entry(conn, gpu_id, payload, inference_type, shard_id, num_shards):
    """Entry point for a non-daemonic worker process.

    multiprocessing.Pool marks its workers daemonic, and daemonic processes may
    not have children -- which is what DataLoader(num_workers>0) needs. Using
    ctx.Process(daemon=False) lifts that restriction.

    Exactly one message is sent on ``conn``, then it is closed. The parent
    relies on that close (or on the one the kernel performs when the process
    dies) to tell success from death -- see _run_workers.
    """
    try:
        conn.send(infer_worker(gpu_id, payload, inference_type, shard_id, num_shards))
    except Exception as e:
        import traceback
        try:
            conn.send({"gpu": gpu_id, "ok": False, "shard_id": shard_id,
                       "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()})
        except Exception:
            # Nothing left to report through; the parent will see EOF.
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
        _stop_bt_log_listener()


def _failed_result(info: dict, error: str) -> dict:
    """Stand-in result for a worker that never reported one of its own.

    Deliberately omits per_molecule_components / per_nanobody_components so the
    merge and summary paths in main() skip it the same way they skip a worker
    that reported ok=False.
    """
    return {"gpu": info["gpu"], "ok": False, "shard_id": info["shard_id"],
            "kind": info["kind"], "error": error}


def _reap(proc, timeout: float = _TEARDOWN_JOIN_S) -> None:
    """Join a worker, escalating to kill if it will not exit."""
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=timeout)


def _max_retries() -> int:
    """Extra attempts allowed per worker. 0 disables retrying entirely."""
    raw = os.environ.get("NOVA_INFERENCE_MAX_RETRIES")
    if not raw:
        return _DEFAULT_MAX_RETRIES
    try:
        return max(0, int(raw))
    except ValueError:
        bt.logging.warning(
            f"Ignoring non-integer NOVA_INFERENCE_MAX_RETRIES={raw!r}; "
            f"using {_DEFAULT_MAX_RETRIES}."
        )
        return _DEFAULT_MAX_RETRIES


def _worker_tmp_dir(kind: str, shard_id: int, num_shards: int) -> str | None:
    """Scratch directory a worker of this kind owns, or None if it has none.

    Recomputed here rather than imported because importing
    either wrapper in the parent pulls in the whole model package. Both are
    per-worker -- boltz shards get their own subdirectory.
    """
    if kind == "boltzgen":
        return os.path.join(NOVA_DIR, "external_tools", "boltzgen", "boltzgen_tmp_files")
    if kind != "boltz":
        return None
    d = os.path.join(NOVA_DIR, "external_tools", "boltz", "boltz_tmp_files")
    if num_shards > 1:
        d = os.path.join(d, f"shard{shard_id}")
    return d


def _run_workers(ctx, specs: list, deadline_s: float | None = None,
                 gpu_pools: dict | None = None,
                 max_retries: int | None = None) -> list:
    """Run (gpu_id, payload, kind, shard_id, num_shards) specs concurrently.

    Every worker gets its own one-way pipe rather than sharing one Queue. A
    worker killed by a signal -- the OOM killer, a CUDA fault, a segfault in a
    native extension -- never reaches _proc_entry's except clause, so with a
    shared queue the parent blocked on get() forever. A pipe closes when its writer dies, so
    connection.wait() reports it readable and recv() raises EOFError, turning an
    indefinite hang into a definite failure result.

    Per-worker pipes also contain two smaller hazards. A child killed midway
    through writing a result leaves a partial pickle on its own pipe instead of
    corrupting a queue every other worker still has to read; and a result that
    fails to pickle raises in the child (where _proc_entry can report it) rather
    than in a Queue feeder thread that only logs to stderr.

    ``deadline_s`` bounds the whole wave. Workers still pending when it expires
    are killed and reported as failures.

    A failed worker is relaunched while budget remains. 

    ``gpu_pools`` maps kind -> candidate GPU ids; a retry goes to the least
    loaded of them rather than back to the GPU that just failed.
    """
    gpu_pools = gpu_pools or {}
    if max_retries is None:
        max_retries = _max_retries()

    all_procs = []
    pending = {}  # recv conn -> worker info
    deferred = []  # retries waiting for GPU capacity
    results = []
    # Successful wall times per kind, used to price a retry against the budget.
    durations: dict[str, list[float]] = {}
    retried = 0
    t_deadline = None if deadline_s is None else time.monotonic() + deadline_s

    def _spawn(gpu_id, payload, kind, shard_id, num_shards, attempt):
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_proc_entry,
                        args=(send_conn, gpu_id, payload, kind, shard_id, num_shards),
                        daemon=False)
        p.start()
        # The parent's copy of the write end must go, or the pipe never reports
        # EOF when the child dies
        send_conn.close()
        all_procs.append(p)
        pending[recv_conn] = {"proc": p, "gpu": gpu_id, "shard_id": shard_id,
                              "kind": kind, "payload": payload,
                              "num_shards": num_shards, "attempt": attempt,
                              "t_start": time.monotonic()}

    def _retry_cost(kind) -> float:
        """Wall time to assume a fresh worker of this kind will need."""
        seen = durations.get(kind)
        if seen:
            return statistics.median(seen) * _RETRY_TIME_MARGIN
        return (deadline_s or 0.0) * _RETRY_BLIND_FRACTION

    def _reserved_mib(need: int) -> dict:
        """VRAM promised to workers that nvidia-smi cannot see yet, per GPU.
        """
        if not need:
            return {}
        now = time.monotonic()
        held: dict = {}
        for other in pending.values():
            if now - other["t_start"] < _VRAM_SETTLE_S:
                held[other["gpu"]] = held.get(other["gpu"], 0) + need
        return held

    def _pick_gpu(kind, fallback, free=None):
        """Emptiest GPU in the kind's pool, and how much it has free.
        Ranks by measured free VRAM.
        Falls back to counting workers when nvidia-smi is unavailable.
        Returns (gpu_id, free_mib) with free_mib None when memory is unreadable.
        """
        pool = gpu_pools.get(kind) or [fallback]
        seen = list(dict.fromkeys(pool))  # dedupes a weighted pool like 0,0,1,0,1
        if free is None:
            free = _gpu_free_mib()
        known = {g: free[g] for g in seen if g in free}
        if known:
            # Charge not-yet-visible workers before ranking, so a GPU that has
            # just been handed a retry stops looking empty.
            held = _reserved_mib(_worker_vram_mib())
            known = {g: v - held.get(g, 0) for g, v in known.items()}
            gpu = max(known, key=lambda g: (known[g], -seen.index(g)))
            return gpu, known[gpu]
        load = {g: 0 for g in seen}
        for other in pending.values():
            if other["gpu"] in load:
                load[other["gpu"]] += 1
        return min(load, key=load.get), None

    def _maybe_retry(info, failure) -> bool:
        nonlocal retried
        kind, shard_id = info["kind"], info["shard_id"]
        tag = f"{kind} shard={shard_id}/{info['num_shards']}"
        err = failure.get("error")

        if max_retries <= 0:
            return False
        if info["attempt"] > max_retries:
            # Distinguish "gave up" from "never tried"
            bt.logging.error(
                f"{tag} failed on gpu={info['gpu']} ({err}) after all "
                f"{max_retries + 1} attempts; giving up on it."
            )
            return False

        cost = _retry_cost(kind)
        remaining = None if t_deadline is None else t_deadline - time.monotonic()
        if remaining is not None and remaining <= cost:
            bt.logging.error(
                f"{tag} failed on gpu={info['gpu']} ({err}) and will NOT be retried: "
                f"{remaining:.0f}s left before the inference deadline, a retry needs "
                f"about {cost:.0f}s."
            )
            return False

        # Reap before respawning so the dead worker's GPU memory is released
        # before its replacement tries to allocate.
        _reap(info["proc"])

        tmp = _worker_tmp_dir(kind, shard_id, info["num_shards"])
        if tmp and os.path.isdir(tmp):
            try:
                shutil.rmtree(tmp)
            except OSError as e:
                # Leftover predictions would be skipped by boltz, so a retry on
                # a directory we could not clear would come back short.
                bt.logging.error(
                    f"{tag}: could not clear {tmp} ({e}); not retrying, because a "
                    f"retry over stale predictions would silently skip molecules."
                )
                return False

        budget = "no deadline" if remaining is None else f"{remaining:.0f}s left"
        bt.logging.warning(
            f"{tag} failed on gpu={info['gpu']} (attempt {info['attempt']} of "
            f"{max_retries + 1}): {err}\n{failure.get('tb', '')}\n"
            f"Queued for retry ({budget}, ~{cost:.0f}s needed)."
        )
        deferred.append({"info": info, "failure": failure, "cost": cost,
                         "tag": tag, "t_queued": time.monotonic(),
                         "logged_wait": False})
        return True

    def _admit_deferred() -> None:
        """Launch queued retries whose target GPU now has room.

        Retries are queued rather than respawned on the spot because the failure
        that most often needs one is CUDA OOM, and at the instant a shard OOMs
        its siblings are still holding the memory it needs. Respawning straight
        back into a saturated GPU just reproduces the OOM and burns the attempt
        """
        nonlocal retried
        if not deferred:
            return
        need = _worker_vram_mib()
        free_now = _gpu_free_mib()  # one reading for the whole pass
        for entry in list(deferred):
            info, tag = entry["info"], entry["tag"]
            remaining = (None if t_deadline is None
                         else t_deadline - time.monotonic())
            if remaining is not None and remaining <= entry["cost"]:
                bt.logging.error(
                    f"{tag}: gave up waiting for GPU capacity after "
                    f"{time.monotonic() - entry['t_queued']:.0f}s -- "
                    f"{remaining:.0f}s left, a retry needs about "
                    f"{entry['cost']:.0f}s. Not retrying."
                )
                deferred.remove(entry)
                results.append(entry["failure"])
                continue

            gpu, free = _pick_gpu(info["kind"], info["gpu"], free=free_now)
            if need and free is not None and free < need:
                if not entry["logged_wait"]:
                    bt.logging.info(
                        f"{tag}: holding retry until a GPU has room -- emptiest "
                        f"is gpu={gpu} with {free} MiB uncommitted, need {need} "
                        f"MiB. Waiting for running workers to finish."
                    )
                    entry["logged_wait"] = True
                continue

            waited = time.monotonic() - entry["t_queued"]
            free_txt = "unknown" if free is None else f"{free} MiB uncommitted"
            bt.logging.warning(
                f"{tag}: retrying on gpu={gpu} ({free_txt}"
                f"{f', after waiting {waited:.0f}s' if waited >= 1 else ''})."
            )
            deferred.remove(entry)
            retried += 1
            _spawn(gpu, info["payload"], info["kind"], info["shard_id"],
                   info["num_shards"], info["attempt"] + 1)

    for gpu_id, payload, kind, shard_id, num_shards in specs:
        _spawn(gpu_id, payload, kind, shard_id, num_shards, attempt=1)

    while pending or deferred:
        timeout = _POLL_INTERVAL_S
        if t_deadline is not None:
            remaining = t_deadline - time.monotonic()
            if remaining <= 0:
                break
            timeout = min(timeout, remaining)

        _admit_deferred()

        if not pending:
            # Nothing running, but a retry is still waiting on capacity. Sleep
            # instead of spinning; the deadline check above bounds the wait.
            if deferred:
                time.sleep(min(timeout, _POLL_INTERVAL_S))
            continue

        for conn in mp_connection.wait(list(pending), timeout=timeout):
            info = pending.pop(conn)
            try:
                r = conn.recv()
            except EOFError:
                _reap(info["proc"])
                r = _failed_result(
                    info,
                    f"worker died without reporting a result "
                    f"(exitcode={info['proc'].exitcode})",
                )
            except Exception as e:
                _reap(info["proc"])
                r = _failed_result(
                    info, f"unreadable result from worker: {type(e).__name__}: {e}"
                )
            finally:
                conn.close()

            if r.get("ok"):
                durations.setdefault(info["kind"], []).append(
                    time.monotonic() - info["t_start"]
                )
            elif _maybe_retry(info, r):
                # Superseded: only the last attempt for a shard reaches results,
                # so callers still see exactly one entry per spec.
                continue
            results.append(r)

    for entry in deferred:
        # Queued but never admitted: the deadline expired while it waited for a
        # GPU with room. Its original failure is the honest result to report.
        bt.logging.error(
            f"{entry['tag']}: never retried -- waited "
            f"{time.monotonic() - entry['t_queued']:.0f}s for GPU capacity and "
            f"the inference deadline expired first."
        )
        results.append(entry["failure"])
    deferred.clear()

    for conn, info in pending.items():
        # No retry here: the budget that would pay for one is what just expired.
        bt.logging.error(
            f"Worker {info['kind']} shard={info['shard_id']} gpu={info['gpu']} "
            f"still running after {deadline_s:.0f}s deadline; killing it."
        )
        info["proc"].kill()
        conn.close()
        results.append(_failed_result(info, f"timed out after {deadline_s:.0f}s"))

    for proc in all_procs:
        _reap(proc)

    for r in results:
        if not r.get("ok"):
            bt.logging.error(f"Worker failed: {r.get('error')}\n{r.get('tb', '')}")
    if retried:
        bt.logging.info(f"{retried} worker(s) were relaunched after failing.")
    return results


def _inference_deadline_s(config) -> float | None:
    """Wall clock budget for one inference wave, or None to wait indefinitely.

    Defaults to a fraction of the epoch rather than a constant: a timeout longer
    than an epoch cannot help, and a hardcoded one rots when epoch_length or the
    molecule count changes. NOVA_INFERENCE_TIMEOUT_S overrides it; 0 disables
    the deadline entirely (the old hang-forever behaviour).
    """
    raw = os.environ.get("NOVA_INFERENCE_TIMEOUT_S")
    if raw:
        try:
            override = float(raw)
        except ValueError:
            bt.logging.warning(
                f"Ignoring non-numeric NOVA_INFERENCE_TIMEOUT_S={raw!r}"
            )
        else:
            return override if override > 0 else None

    epoch_length = getattr(config, "epoch_length", None)
    if epoch_length is None and isinstance(config, dict):
        epoch_length = config.get("epoch_length")
    try:
        epoch_length = float(epoch_length)
    except (TypeError, ValueError):
        bt.logging.warning(
            "No usable epoch_length in config; inference workers will run without a deadline."
        )
        return None
    return epoch_length * _BLOCK_TIME_S * _EPOCH_DEADLINE_FRACTION


def _merge_boltz_shards(results: list) -> tuple:
    """Union the per-shard boltz outputs. Shards hold disjoint SMILES."""
    unique_molecules, per_molecule_components, final_boltz_scores = {}, {}, {}
    for r in results:
        if not r.get("ok"):
            continue
        unique_molecules.update(r.get("unique_molecules") or {})
        for uid, by_smiles in (r.get("per_molecule_components") or {}).items():
            per_molecule_components.setdefault(uid, {}).update(by_smiles)
        for uid, by_target in (r.get("final_boltz_scores") or {}).items():
            dst = final_boltz_scores.setdefault(uid, {})
            for target, by_smiles in by_target.items():
                dst.setdefault(target, {}).update(by_smiles)
    return unique_molecules, per_molecule_components, final_boltz_scores


def _log_runtime_config(gpu_id, inference_type, shard_id, num_shards, pid) -> None:
    """Emit the EFFECTIVE runtime config, read back from the libraries themselves.
    """
    try:
        devices = torch.cuda.device_count()
    except Exception as e:
        devices = f"err:{type(e).__name__}"
    print(
        f"RUNTIME_CONFIG worker_{inference_type} pid={pid} "
        f"shard={shard_id}/{num_shards} gpu={gpu_id} "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} cuda_devices={devices} "
        f"torch_threads={torch.get_num_threads()} "
        f"torch_interop={torch.get_num_interop_threads()} "
        f"omp_env={os.environ.get('OMP_NUM_THREADS')} "
        f"mkl_env={os.environ.get('MKL_NUM_THREADS')}",
        flush=True,
    )


def infer_worker(gpu_id: int, payload: dict, inference_type: str,
                 shard_id: int = 0, num_shards: int = 1) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import time as _time
    _t0 = _time.time()
    _pid = os.getpid()
    print(f"STAGE_BEGIN worker_{inference_type} t={_t0:.3f} gpu={gpu_id} pid={_pid}", flush=True)
    _log_runtime_config(gpu_id, inference_type, shard_id, num_shards, _pid)

    def _stamp(d):
        d["t_start"], d["t_end"], d["pid"] = _t0, _time.time(), _pid
        print(
            f"STAGE_END worker_{inference_type} t={d['t_end']:.3f} "
            f"elapsed_s={d['t_end'] - _t0:.2f} gpu={gpu_id} pid={_pid}",
            flush=True,
        )
        return d

    if inference_type == "boltz":
        from external_tools.boltz.boltz_wrapper import BoltzWrapper
        boltz = BoltzWrapper(shard_id=shard_id, num_shards=num_shards)
        boltz.score_molecules(payload["molecules"], payload["score_dict"], payload["config"])
        assigned = getattr(boltz, "unique_molecules", {}) or {}
        produced = getattr(boltz, "final_boltz_scores", {}) or {}
        if assigned and not produced:
            # boltz can fail without raising. Its predict loop catches CUDA OOM
            # per batch, logs "ran out of memory, skipping batch", reports
            # "Number of failed examples: N" and returns normally having scored
            # nothing. Partial loss (some batches skipped) still reports ok=True; catching
            # that needs per-molecule coverage accounting, which this is not.
            return _stamp({
                "gpu": gpu_id,
                "ok": False,
                "shard_id": shard_id,
                "kind": "boltz",
                "error": (f"boltz scored none of its {len(assigned)} assigned "
                          f"molecules and raised nothing; check the worker log "
                          f"for 'ran out of memory, skipping batch'"),
            })
        score_dict_updates = {
            uid: {"molecule_scores": payload["score_dict"][uid]["molecule_scores"]}
            for uid in payload["score_dict"]
        }
        return _stamp({
            "gpu": gpu_id,
            "ok": True,
            "shard_id": shard_id,
            "boltz": score_dict_updates,
            "per_molecule_components": getattr(boltz, "per_molecule_components", {}),
            "unique_molecules": getattr(boltz, "unique_molecules", {}),
            "final_boltz_scores": getattr(boltz, "final_boltz_scores", {}),
        })

    elif inference_type == "boltzgen":
        from boltzgen.boltzgen_wrapper import BoltzgenWrapper
        boltzgen = BoltzgenWrapper()
        per_nanobody_components = boltzgen.run_nanobody_inference(
            payload["nanobodies"], payload["config"]
        )
        return _stamp({
            "gpu": gpu_id,
            "ok": True,
            "per_nanobody_components": per_nanobody_components,
        })
    else:
        return {"gpu": gpu_id, "ok": False, "error": f"unknown inference_type={inference_type}"}

def _merge_boltzgen_into_score_dict(
    score_dict: dict,
    final_boltzgen_scores: dict | None,
    valid_nanobodies_by_uid: dict,
    config,
    rank_mode: str = "min",
) -> None:
    """Write nanobody_scores into score_dict"""
    if not final_boltzgen_scores:
        bt.logging.warning("_merge_boltzgen_into_score_dict: final_boltzgen_scores is None or empty")
        return
    nanobody_target = getattr(config, "nanobody_target", None) or config.get("nanobody_target", [])
    sentinel = math.inf if rank_mode == "min" else -math.inf
    for uid in final_boltzgen_scores:
        if uid not in score_dict:
            bt.logging.warning(f"_merge_boltzgen_into_score_dict: UID {uid} not in score_dict, skipping")
            continue
        sequences = list(valid_nanobodies_by_uid.get(uid, {}).get("sequences", []))
        if not sequences:
            bt.logging.warning(
                f"_merge_boltzgen_into_score_dict: UID {uid} has inference results but no "
                f"validated sequences; skipping."
            )
            continue
        rows = []
        for target in nanobody_target:
            row = [
                final_boltzgen_scores[uid].get(seq, {}).get(target, sentinel).item()
                if hasattr(final_boltzgen_scores[uid].get(seq, {}).get(target, sentinel), 'item')
                else final_boltzgen_scores[uid].get(seq, {}).get(target, sentinel)
                for seq in sequences
            ]
            rows.append(row)
        score_dict[uid]["nanobody_scores"] = rows


def main(valid_molecules_by_uid: dict, valid_nanobodies_by_uid: dict, score_dict: dict, config) -> InferenceResult:
    """
    Run Boltz and/or Boltzgen inference, each on its own GPU when available.

    Updates ``score_dict['molecule_scores']`` in-place. Nanobody inference fills
    ``per_nanobody_components`` only; ranked ``nanobody_scores`` are applied later
    by the caller (validator) after external score sharing.
    """
    run_boltz = bool(valid_molecules_by_uid)
    run_boltzgen = bool(valid_nanobodies_by_uid)
    if not run_boltz and not run_boltzgen:
        return InferenceResult(boltz=None, boltzgen=None)

    num_gpus = torch.cuda.device_count()
    bt.logging.info(f"Detected GPUs: {num_gpus}. Boltz={run_boltz}, Boltzgen={run_boltzgen}")

    plain_config = config.to_dict()
    payload_boltz = {"molecules": valid_molecules_by_uid, "score_dict": score_dict, "config": plain_config}
    payload_boltzgen = {"nanobodies": valid_nanobodies_by_uid, "config": plain_config}

    per_nanobody_components = None
    per_molecule_components = None
    unique_molecules = None
    ctx = mp.get_context("spawn")

    # GPU allocation. Defaults preserve the previous behaviour (boltz on 0,
    # boltzgen on 1 when both run and 2+ GPUs exist); override via env for
    # benchmarking or asymmetric boxes, e.g. NOVA_BOLTZ_GPUS=0,1,2.
    #
    # Shards are assigned round-robin (boltz_gpus[s % len(boltz_gpus)]), so
    # repeating a GPU id in the list weights it. That is the knob for the
    # asymmetry created by boltzgen sitting on GPU1: co-resident shards there
    # ran ~13% slower than GPU0's at 1280 molecules. To split 12 shards 7/5,
    # give a 12-element list, e.g.
    #   NOVA_BOLTZ_GPUS=0,0,1,0,0,1,0,1,0,1,0,1
    # A short list like 0,0,1 only yields the intended ratio when num_shards is
    # a multiple of its length (at K=12 it gives 8/4).
    def _gpu_list(env_name, default):
        raw = os.environ.get(env_name)
        if not raw:
            return default
        return [int(x) for x in raw.split(",") if x.strip() != ""]

    num_shards = int(os.environ.get("NOVA_BOLTZ_SHARDS", "1"))
    num_shards = max(1, num_shards)

    # The default has to depend on the shard count.When sharding, spread by default and
    # let NOVA_BOLTZ_GPUS override for a deliberate weighting.
    if num_gpus >= 2 and run_boltz and run_boltzgen:
        default_boltz = list(range(num_gpus)) if num_shards > 1 else [0]
        boltz_gpus = _gpu_list("NOVA_BOLTZ_GPUS", default_boltz)
        boltzgen_gpus = _gpu_list("NOVA_BOLTZGEN_GPUS", [1])
    else:
        boltz_gpus = _gpu_list("NOVA_BOLTZ_GPUS", [0])
        boltzgen_gpus = _gpu_list("NOVA_BOLTZGEN_GPUS", [0])

    if run_boltz and num_shards > 1:
        # Say up front whether the layout can physically fit.
        per_gpu = collections.Counter(
            boltz_gpus[s % len(boltz_gpus)] for s in range(num_shards)
        )
        if run_boltzgen:
            per_gpu[boltzgen_gpus[0]] += 1
        bt.logging.info(
            f"Worker layout by GPU: "
            f"{ {g: n for g, n in sorted(per_gpu.items())} }"
        )
        need_each = _worker_vram_mib()
        free_now = _gpu_free_mib() if need_each else {}
        for g, n in sorted(per_gpu.items()):
            have = free_now.get(g)
            if have is not None and n * need_each > have:
                bt.logging.warning(
                    f"GPU {g} is assigned {n} concurrent workers needing about "
                    f"{n * need_each} MiB but only {have} MiB is free. Expect "
                    f"CUDA OOM. Lower NOVA_BOLTZ_SHARDS, or spread the load with "
                    f"NOVA_BOLTZ_GPUS (e.g. "
                    f"{','.join(str(x) for x in range(num_gpus))})."
                )

    specs = []
    if run_boltz:
        for s in range(num_shards):
            specs.append((boltz_gpus[s % len(boltz_gpus)], payload_boltz, "boltz", s, num_shards))
    if run_boltzgen:
        specs.append((boltzgen_gpus[0], payload_boltzgen, "boltzgen", 0, 1))

    deadline_s = _inference_deadline_s(config)
    retries = _max_retries()
    gpu_pools = {"boltz": boltz_gpus, "boltzgen": boltzgen_gpus}
    bt.logging.info(
        f"Launching {len(specs)} workers: boltz shards={num_shards if run_boltz else 0} "
        f"on GPUs {boltz_gpus if run_boltz else []}, "
        f"boltzgen on GPUs {boltzgen_gpus[:1] if run_boltzgen else []}, "
        f"deadline={f'{deadline_s:.0f}s' if deadline_s else 'none'}, "
        f"max_retries={retries}"
    )

    results = _run_workers(ctx, specs, deadline_s=deadline_s,
                           gpu_pools=gpu_pools, max_retries=retries)
    boltz_results = [r for r in results if r.get("shard_id") is not None
                     and r.get("gpu") is not None and "per_molecule_components" in r]
    boltzgen_results = [r for r in results if "per_nanobody_components" in r]

    if run_boltz:
        # Count shards that came back with scores, not shards that came back.
        produced = [r for r in boltz_results if r.get("final_boltz_scores")]
        lost = num_shards - len(produced)
        if lost > 0:
            bt.logging.error(
                f"{lost} of {num_shards} boltz shards produced no results. "
                f"Every molecule assigned to them scores as a sentinel, which "
                f"makes the affected UIDs unrankable."
            )

    if run_boltz and boltz_results:
        unique_molecules, per_molecule_components, final_boltz_scores = \
            _merge_boltz_shards(boltz_results)
        assemble_molecule_scores(score_dict, valid_molecules_by_uid, final_boltz_scores, plain_config)

    if run_boltzgen and boltzgen_results and boltzgen_results[0].get("ok"):
        per_nanobody_components = boltzgen_results[0].get("per_nanobody_components")

    _log_inference_summary(results, num_shards)

    boltz = BoltzResult(per_molecule_components, unique_molecules) if run_boltz else None
    boltzgen = BoltzgenResult(per_nanobody_components or {}, None) if run_boltzgen else None
    return InferenceResult(boltz=boltz, boltzgen=boltzgen)
