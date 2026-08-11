import atexit
import math
import multiprocessing as mp
import os
import sys
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


def assemble_molecule_scores(score_dict: dict, unique_molecules: dict,
                             final_boltz_scores: dict, subnet_config: dict) -> None:
    """Write molecule_scores into score_dict from (possibly merged) shard results.

    Lives here rather than in boltz_wrapper so the parent process can assemble
    once from the union of all shards without importing boltz.
    """
    sentinel = math.inf if subnet_config['boltz_mode'] == "min" else -math.inf
    for uid, data in score_dict.items():
        if uid in final_boltz_scores:
            smiles_list = [s for s, id_list in unique_molecules.items()
                           if any(u == uid for u, _ in id_list)]
            data['molecule_scores'] = [
                [final_boltz_scores[uid].get(target, {}).get(s, sentinel) for s in smiles_list]
                for target in subnet_config['small_molecule_target']
            ]
        else:
            target_count = len(subnet_config['small_molecule_target'])
            data['molecule_scores'] = [[sentinel] for _ in range(target_count)]


def _stop_bt_log_listener() -> None:
    """Stop bittensor's QueueListener before multiprocessing tears the queue down.

    Each spawn child builds its own LoggingMachine, with an mp.Queue and a
    _monitor thread blocked on queue.get(). BaseProcess._bootstrap runs
    multiprocessing's _exit_function in a finally block, which closes that
    queue's pipe while _monitor is still reading it -> OSError(EBADF). bittensor
    stops the listener via atexit, which only fires later at interpreter
    shutdown -- too late in a child. Stopping here also flushes records that
    would otherwise be dropped from the tail of the worker's log.
    """
    listener = getattr(bt.logging, "_listener", None)
    if listener is None or getattr(listener, "_thread", None) is None:
        return
    try:
        atexit.unregister(listener.stop)  # stop() is not idempotent
        listener.stop()
    except Exception:
        pass


def _proc_entry(queue, gpu_id, payload, inference_type, shard_id, num_shards):
    """Entry point for a non-daemonic worker process.

    multiprocessing.Pool marks its workers daemonic, and daemonic processes may
    not have children -- which is what DataLoader(num_workers>0) needs. Using
    ctx.Process(daemon=False) lifts that restriction.
    """
    try:
        queue.put(infer_worker(gpu_id, payload, inference_type, shard_id, num_shards))
    except Exception as e:
        import traceback
        queue.put({"gpu": gpu_id, "ok": False, "shard_id": shard_id,
                   "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()})
    finally:
        _stop_bt_log_listener()


def _run_workers(ctx, specs: list) -> list:
    """Run (gpu_id, payload, kind, shard_id, num_shards) specs concurrently."""
    queue = ctx.Queue()
    procs = []
    for gpu_id, payload, kind, shard_id, num_shards in specs:
        p = ctx.Process(target=_proc_entry,
                        args=(queue, gpu_id, payload, kind, shard_id, num_shards),
                        daemon=False)
        p.start()
        procs.append(p)
    results = [queue.get() for _ in specs]
    for p in procs:
        p.join()
    for r in results:
        if not r.get("ok"):
            bt.logging.error(f"Worker failed: {r.get('error')}\n{r.get('tb', '')}")
    return results


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

    Deliberately reports torch.get_num_threads() rather than only the env var:
    the two have disagreed in a way that cost real benchmark runs.
    boltz_wrapper.py set OMP_NUM_THREADS=1 while every shard actually ran 26
    threads, because torch was already imported by the time it ran. Separately,
    .env silently overrode an exported NOVA_BOLTZ_SHARDS. In both cases a
    setting that failed to apply looked identical to one that worked.

    Both the read-back and the env var are printed: their *disagreement* is the
    signal. print() rather than bt.logging, for the same reason as STAGE_BEGIN
    -- a child can lose queued log records on exit.
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


def _merge_boltz_into_score_dict(score_dict: dict, boltz_result: dict) -> None:
    if not boltz_result or "boltz" not in boltz_result:
        return
    for uid, data in boltz_result["boltz"].items():
        if uid in score_dict and "molecule_scores" in data:
            score_dict[uid]["molecule_scores"] = data["molecule_scores"]


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
            sequences = list(final_boltzgen_scores[uid].keys())
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

    if num_gpus >= 2 and run_boltz and run_boltzgen:
        boltz_gpus = _gpu_list("NOVA_BOLTZ_GPUS", [0])
        boltzgen_gpus = _gpu_list("NOVA_BOLTZGEN_GPUS", [1])
    else:
        boltz_gpus = _gpu_list("NOVA_BOLTZ_GPUS", [0])
        boltzgen_gpus = _gpu_list("NOVA_BOLTZGEN_GPUS", [0])

    num_shards = int(os.environ.get("NOVA_BOLTZ_SHARDS", "1"))
    num_shards = max(1, num_shards)

    specs = []
    if run_boltz:
        for s in range(num_shards):
            specs.append((boltz_gpus[s % len(boltz_gpus)], payload_boltz, "boltz", s, num_shards))
    if run_boltzgen:
        specs.append((boltzgen_gpus[0], payload_boltzgen, "boltzgen", 0, 1))

    bt.logging.info(
        f"Launching {len(specs)} workers: boltz shards={num_shards if run_boltz else 0} "
        f"on GPUs {boltz_gpus if run_boltz else []}, "
        f"boltzgen on GPUs {boltzgen_gpus[:1] if run_boltzgen else []}"
    )

    results = _run_workers(ctx, specs)
    boltz_results = [r for r in results if r.get("shard_id") is not None
                     and r.get("gpu") is not None and "per_molecule_components" in r]
    boltzgen_results = [r for r in results if "per_nanobody_components" in r]

    if run_boltz and boltz_results:
        unique_molecules, per_molecule_components, final_boltz_scores = \
            _merge_boltz_shards(boltz_results)
        assemble_molecule_scores(score_dict, unique_molecules, final_boltz_scores, plain_config)

    if run_boltzgen and boltzgen_results and boltzgen_results[0].get("ok"):
        per_nanobody_components = boltzgen_results[0].get("per_nanobody_components")

    _log_inference_summary(results, num_shards)

    boltz = BoltzResult(per_molecule_components, unique_molecules) if run_boltz else None
    boltzgen = BoltzgenResult(per_nanobody_components or {}, None) if run_boltzgen else None
    return InferenceResult(boltz=boltz, boltzgen=boltzgen)
