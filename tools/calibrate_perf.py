#!/usr/bin/env python3
"""Offline performance calibration for the NOVA validator.

Finds the best shard count, thread count and GPU layout for THIS machine and
THIS target combination, then writes the result to a standalone config file for
you to review and apply by hand. It never touches the validator's runtime path
and never sets weights -- every run uses --test_mode and --local_input_file.

Why this exists: the tuned values do not transfer. They depend on GPU count and
VRAM, CPU cores, and the size of the clipped target proteins in
`config/config.yaml` (`protein_selection`). Change any of those and the previous
numbers are stale.

    python3 tools/calibrate_perf.py --molecules 120 --out config/perf_tuning.yaml

Phases (each skippable, see --skip):
    noise    two identical runs -> the run-to-run envelope. Everything else is
             judged against this; on the reference box it was ~7%, which is
             larger than several differences that look meaningful.
    shards   ladder over K, capped by measured VRAM per shard
    threads  intra-op threads at the winning K
    layout   boltz on one GPU vs spread across all, boltzgen co-resident
    epoch    one full run with sequences at the recommended config, to get the
             real epoch time and the molecule ceiling

Selection rule: the SMALLEST setting within the noise envelope of the best, not
the fastest sample. Picking the raw minimum over-fits to noise and tends to
choose high shard counts whose only real effect is more VRAM pressure.

Every run is verified against the RUNTIME_CONFIG line the worker prints, which
reports the EFFECTIVE config read back from torch. A run whose effective config
does not match what was requested is discarded, not averaged in. This matters:
`.env` silently overrides exported variables (load_dotenv(override=True) in
utils/molecules.py and utils/files.py), and thread caps set after torch is
imported are no-ops. Both have invalidated benchmark runs here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV_FILE = REPO / ".env"
VALIDATOR = REPO / "neurons" / "validator" / "validator.py"


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def sh(cmd: list[str], **kw) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, **kw).stdout.strip()


def probe_environment() -> dict:
    gpus = []
    out = sh(["nvidia-smi", "--query-gpu=index,name,memory.total",
              "--format=csv,noheader,nounits"])
    for line in out.splitlines():
        idx, name, total = [p.strip() for p in line.split(",")]
        gpus.append({"index": int(idx), "name": name, "memory_total_mib": int(total)})

    cores = os.cpu_count()
    try:
        import yaml
        cfg = yaml.safe_load((REPO / "config" / "config.yaml").read_text())
        sel = cfg["protein_selection"]

        def describe(block):
            targets = block["target"].split(",")
            clips = block.get("clip_interval") or [None] * len(targets)
            out = []
            for t, c in zip(targets, clips):
                length = (c[1] - c[0]) if c else None
                out.append({"target": t, "clip": c, "clipped_length": length})
            return out

        targets = {"small_molecule": describe(sel["small_molecule"]),
                   "nanobody": describe(sel["nanobody"])}
    except Exception as e:                      # pragma: no cover
        targets = {"error": f"{type(e).__name__}: {e}"}

    return {
        "hostname": sh(["hostname"]),
        "gpus": gpus,
        "cpu_cores": cores,
        "targets": targets,
        "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def ensure_quiet() -> None:
    """Refuse to measure while anything else is on the GPUs.

    A live validator competing for the same GPUs silently corrupted an entire
    shard benchmark here once; the numbers looked plausible throughout.
    """
    busy = sh(["nvidia-smi", "--query-compute-apps=pid,used_memory",
               "--format=csv,noheader"])
    if busy:
        sys.exit(f"ABORT: GPU processes already running -- machine is not quiet:\n{busy}")
    stray = sh(["pgrep", "-af", "validator.py"])
    if stray:
        sys.exit(f"ABORT: validator already running:\n{stray}")


# --------------------------------------------------------------------------
# running the validator
# --------------------------------------------------------------------------

class MemorySampler(threading.Thread):
    """Poll per-GPU memory so the shard ladder can be capped by real VRAM use."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak: dict[int, int] = {}
        # NOT self._stop -- that name shadows threading.Thread._stop(), which
        # join() calls internally, and the override is not callable.
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            out = sh(["nvidia-smi", "--query-gpu=index,memory.used",
                      "--format=csv,noheader,nounits"])
            for line in out.splitlines():
                try:
                    idx, used = [int(p.strip()) for p in line.split(",")]
                except ValueError:
                    continue
                self.peak[idx] = max(self.peak.get(idx, 0), used)
            self._halt.wait(self.interval)

    def stop(self) -> dict[int, int]:
        self._halt.set()
        self.join(timeout=10)
        return self.peak


@dataclass
class RunResult:
    label: str
    shards: int
    threads: int
    boltz_gpus: str
    boltzgen_gpus: str
    ok: bool
    wall_s: float
    boltz_s: float | None = None
    boltzgen_s: float | None = None
    shard_times: list[float] = field(default_factory=list)
    peak_mib: dict[int, int] = field(default_factory=dict)
    stages: dict[str, float] = field(default_factory=dict)
    effective: list[dict] = field(default_factory=list)
    problem: str | None = None


RUNTIME_RE = re.compile(
    r"RUNTIME_CONFIG worker_(\w+) pid=(\d+) shard=(\d+)/(\d+) gpu=(\d+) .*?"
    r"torch_threads=(\d+)")
SUMMARY_RE = re.compile(r"STAGE_SUMMARY inference_parallel ([^\n|]*)")
STAGE_RE = re.compile(r"STAGE_END (\w+) t=[\d.]+ elapsed_s=([\d.]+)")


def parse_log(text: str) -> dict:
    out: dict = {"effective": [], "stages": {}}
    for m in RUNTIME_RE.finditer(text):
        out["effective"].append({
            "kind": m.group(1), "pid": int(m.group(2)),
            "shard": int(m.group(3)), "num_shards": int(m.group(4)),
            "gpu": int(m.group(5)), "torch_threads": int(m.group(6)),
        })
    for m in STAGE_RE.finditer(text):
        out["stages"][m.group(1)] = float(m.group(2))
    s = SUMMARY_RE.findall(text)
    if s:
        fields = dict(
            kv.split("=", 1) for kv in s[-1].split() if "=" in kv
        )
        if "boltz_total_s" in fields:
            out["boltz_s"] = float(fields["boltz_total_s"])
        if "boltzgen_s" in fields:
            out["boltzgen_s"] = float(fields["boltzgen_s"])
        if "boltz_shard_s" in fields:
            out["shard_times"] = [
                float(x) for x in fields["boltz_shard_s"].strip("[]").split(",") if x
            ]
    return out


class ShardSetting:
    """Set NOVA_BOLTZ_SHARDS where it will actually take effect.

    utils/molecules.py and utils/files.py call load_dotenv(override=True), so a
    key present in .env beats anything exported. If the key is there, it must be
    edited in place; the original file is restored on exit.
    """

    def __init__(self, shards: int):
        self.shards = shards
        self.backup: Path | None = None

    def __enter__(self) -> dict:
        env = dict(os.environ)
        env["NOVA_BOLTZ_SHARDS"] = str(self.shards)
        if ENV_FILE.exists() and re.search(r"^NOVA_BOLTZ_SHARDS=", ENV_FILE.read_text(),
                                           re.M):
            fd, path = tempfile.mkstemp(prefix="env-backup-")
            os.close(fd)
            self.backup = Path(path)
            shutil.copy2(ENV_FILE, self.backup)
            ENV_FILE.write_text(re.sub(r"^NOVA_BOLTZ_SHARDS=.*$",
                                       f"NOVA_BOLTZ_SHARDS={self.shards}",
                                       ENV_FILE.read_text(), flags=re.M))
        return env

    def __exit__(self, *exc) -> None:
        if self.backup:
            shutil.copy2(self.backup, ENV_FILE)
            self.backup.unlink(missing_ok=True)


def run_once(label: str, fixture: Path, shards: int, threads: int,
             boltz_gpus: str, boltzgen_gpus: str, outdir: Path,
             timeout_s: int) -> RunResult:
    ensure_quiet()
    shutil.rmtree(REPO / "external_tools/boltz/boltz_tmp_files", ignore_errors=True)
    shutil.rmtree(REPO / "external_tools/boltzgen/boltzgen_tmp_files", ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / f"{label}.log"

    res = RunResult(label=label, shards=shards, threads=threads,
                    boltz_gpus=boltz_gpus, boltzgen_gpus=boltzgen_gpus,
                    ok=False, wall_s=0.0)

    with ShardSetting(shards) as env:
        env["NOVA_BOLTZ_THREADS"] = str(threads)
        env["NOVA_BOLTZ_GPUS"] = boltz_gpus
        env["NOVA_BOLTZGEN_GPUS"] = boltzgen_gpus
        env["NOVA_SKIP_HISTORICAL_CHECKS"] = "1"
        # never inherit a stale explicit cap -- it would beat NOVA_BOLTZ_THREADS
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env.pop(v, None)

        sampler = MemorySampler()
        sampler.start()
        t0 = time.time()
        try:
            with log_path.open("w") as fh:
                proc = subprocess.run(
                    [sys.executable, "-u", str(VALIDATOR), "--logging.debug",
                     "--test_mode", "--local_input_file", str(fixture)],
                    cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT,
                    timeout=timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -1
            res.problem = f"timed out after {timeout_s}s"
        res.wall_s = time.time() - t0
        res.peak_mib = sampler.stop()

    text = log_path.read_text(errors="replace")
    parsed = parse_log(text)
    res.boltz_s = parsed.get("boltz_s")
    res.boltzgen_s = parsed.get("boltzgen_s")
    res.shard_times = parsed.get("shard_times", [])
    res.stages = parsed["stages"]
    res.effective = parsed["effective"]

    if rc != 0 and res.problem is None:
        res.problem = f"exit code {rc}"
    if "CUDA out of memory" in text:
        res.problem = "CUDA out of memory"
    if res.problem is None and res.boltz_s is None:
        res.problem = "no STAGE_SUMMARY -- run produced no boltz timing"

    # verify the config that actually applied
    boltz_workers = [e for e in res.effective if e["kind"] == "boltz"]
    if res.problem is None:
        if not boltz_workers:
            res.problem = "no RUNTIME_CONFIG lines -- cannot verify effective config"
        else:
            got_shards = boltz_workers[0]["num_shards"]
            got_threads = {e["torch_threads"] for e in boltz_workers}
            if got_shards != shards:
                res.problem = (f"requested {shards} shards, got {got_shards} "
                               f"(is NOVA_BOLTZ_SHARDS pinned in .env?)")
            elif got_threads != {threads}:
                res.problem = (f"requested {threads} threads, got {sorted(got_threads)}")
            elif len(boltz_workers) != shards:
                res.problem = f"expected {shards} workers, saw {len(boltz_workers)}"

    res.ok = res.problem is None
    status = "ok" if res.ok else f"INVALID: {res.problem}"
    print(f"  [{label}] K={shards} threads={threads} gpus={boltz_gpus} -> "
          f"boltz={res.boltz_s}s wall={res.wall_s:.0f}s  {status}", flush=True)
    return res


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------

def pick_smallest_within(results: list[RunResult], key, noise_pct: float):
    """Smallest `key` whose time is within the noise envelope of the best.

    Deliberately not argmin: with a ~7% envelope the fastest sample is often
    noise, and the larger setting costs VRAM for nothing.
    """
    good = [r for r in results if r.ok and r.boltz_s]
    if not good:
        return None, []
    best = min(r.boltz_s for r in good)
    threshold = best * (1 + noise_pct / 100.0)
    eligible = sorted([r for r in good if r.boltz_s <= threshold], key=key)
    return eligible[0], good


def phase_noise(fixture, outdir, shards, threads, boltz_gpus, timeout_s) -> tuple[float, list]:
    print("\n[noise] two identical runs to size the envelope")
    runs = [run_once(f"noise{i}", fixture, shards, threads, boltz_gpus, boltz_gpus,
                     outdir, timeout_s) for i in (1, 2)]
    good = [r for r in runs if r.ok and r.boltz_s]
    if len(good) < 2:
        print("  could not measure noise; assuming 7% (reference-box value)")
        return 7.0, runs
    a, b = good[0].boltz_s, good[1].boltz_s
    pct = abs(a - b) / min(a, b) * 100
    print(f"  {a:.1f}s vs {b:.1f}s -> envelope {pct:.1f}%")
    return max(pct, 2.0), runs


def phase_shards(fixture, outdir, ladder, threads, boltz_gpus, gpu_total_mib,
                 noise_pct, timeout_s) -> tuple[int, list]:
    print(f"\n[shards] ladder {ladder}")
    runs = []
    for k in ladder:
        r = run_once(f"k{k}", fixture, k, threads, boltz_gpus, boltz_gpus,
                     outdir, timeout_s)
        runs.append(r)
        if r.problem == "CUDA out of memory":
            print(f"  OOM at K={k}; stopping the ladder here")
            break
        if r.ok and r.peak_mib:
            worst = max(r.peak_mib.values())
            if worst > 0.80 * gpu_total_mib:
                print(f"  peak {worst} MiB is >80% of {gpu_total_mib} MiB; "
                      f"stopping before OOM risk")
                break
    best, good = pick_smallest_within(runs, lambda r: r.shards, noise_pct)
    if best is None:
        sys.exit("ABORT: no valid shard runs -- check the logs in " + str(outdir))
    print(f"  chosen K={best.shards} ({best.boltz_s:.1f}s); smallest within "
          f"{noise_pct:.1f}% of best {min(r.boltz_s for r in good):.1f}s")
    return best.shards, runs


def phase_threads(fixture, outdir, shards, cores, boltz_gpus, noise_pct,
                  timeout_s) -> tuple[int, list]:
    candidates = sorted({1, max(1, cores // max(shards, 1))})
    if len(candidates) == 1:
        print(f"\n[threads] only one candidate ({candidates[0]}); skipping")
        return candidates[0], []
    print(f"\n[threads] candidates {candidates} at K={shards}")
    runs = [run_once(f"t{t}", fixture, shards, t, boltz_gpus, boltz_gpus,
                     outdir, timeout_s) for t in candidates]
    best, _ = pick_smallest_within(runs, lambda r: r.threads, noise_pct)
    if best is None:
        print("  no valid runs; defaulting to 1")
        return 1, runs
    print(f"  chosen threads={best.threads}")
    return best.threads, runs


def phase_layout(fixture, outdir, shards, threads, gpu_indices, noise_pct,
                 timeout_s) -> tuple[str, list]:
    if len(gpu_indices) < 2:
        print("\n[layout] single GPU; nothing to compare")
        return str(gpu_indices[0]), []
    spread = ",".join(str(i) for i in gpu_indices)
    single = str(gpu_indices[0])
    print(f"\n[layout] boltz on '{single}' vs '{spread}'")
    runs = [
        run_once("layout_single", fixture, shards, threads, single, single,
                 outdir, timeout_s),
        run_once("layout_spread", fixture, shards, threads, spread, spread,
                 outdir, timeout_s),
    ]
    good = [r for r in runs if r.ok and r.boltz_s]
    if len(good) < 2:
        print("  incomplete; defaulting to spreading across all GPUs")
        return spread, runs
    a, b = good[0], good[1]
    gain = (a.boltz_s - b.boltz_s) / a.boltz_s * 100
    if gain > noise_pct:
        print(f"  spreading wins by {gain:.1f}% (> {noise_pct:.1f}% noise)")
        return spread, runs
    print(f"  difference {gain:.1f}% is within noise; keeping boltz on one GPU "
          f"so the other stays free for boltzgen")
    return single, runs


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def write_config(path: Path, env_info: dict, rec: dict, evidence: dict) -> None:
    lines = [
        "# NOVA validator performance tuning -- GENERATED, review before use.",
        "#",
        f"# Produced by tools/calibrate_perf.py on {env_info['calibrated_at']}.",
        "# These values are specific to this machine AND these targets. Re-run",
        "# the calibration if you change GPUs, CPU count, or the targets/clip",
        "# intervals in config/config.yaml -- they do not transfer.",
        "#",
        "# Apply by exporting these before launching the validator, e.g.:",
        f"#   NOVA_BOLTZ_SHARDS={rec['boltz_shards']} "
        f"NOVA_BOLTZ_THREADS={rec['boltz_threads']} \\",
        f"#   NOVA_BOLTZ_GPUS={rec['boltz_gpus']} "
        f"NOVA_BOLTZGEN_GPUS={rec['boltzgen_gpus']} python3 neurons/validator/validator.py",
        "#",
        "# NOTE: if NOVA_BOLTZ_SHARDS is present in .env it will OVERRIDE the",
        "# exported value (load_dotenv(override=True)). Set it there instead,",
        "# and confirm via the RUNTIME_CONFIG line in the logs.",
        "",
        "tuning:",
    ]
    for k, v in rec.items():
        lines.append(f"  {k}: {v!r}" if isinstance(v, str) else f"  {k}: {v}")
    lines += ["", "environment:"]
    for line in json.dumps(env_info, indent=2).splitlines():
        lines.append("  " + line)
    lines += ["", "evidence:"]
    for line in json.dumps(evidence, indent=2, default=str).splitlines():
        lines.append("  " + line)
    path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path}")


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixture", type=Path, required=True,
                   help="molecules-only fixture (uid|molecule|~ per line) for the sweeps")
    p.add_argument("--epoch-fixture", type=Path,
                   help="full fixture WITH sequences, for the final epoch phase")
    p.add_argument("--out", type=Path, default=REPO / "config" / "perf_tuning.yaml")
    p.add_argument("--workdir", type=Path, default=REPO / ".calibration")
    p.add_argument("--ladder", type=str, default="4,8,12,16",
                   help="shard counts to try (comma separated)")
    p.add_argument("--deadline-s", type=float, default=3600.0,
                   help="epoch budget used for the capacity estimate")
    p.add_argument("--run-timeout-s", type=int, default=5400)
    p.add_argument("--skip", type=str, default="",
                   help="comma separated: noise,shards,threads,layout,epoch")
    args = p.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    env_info = probe_environment()
    gpu_indices = [g["index"] for g in env_info["gpus"]]
    if not gpu_indices:
        sys.exit("ABORT: no GPUs detected")
    gpu_total = min(g["memory_total_mib"] for g in env_info["gpus"])
    cores = env_info["cpu_cores"]
    all_gpus = ",".join(str(i) for i in gpu_indices)

    print(f"machine: {len(gpu_indices)}x {env_info['gpus'][0]['name']}, "
          f"{gpu_total} MiB each, {cores} cores")
    print(f"targets: {json.dumps(env_info['targets'])}")

    args.workdir.mkdir(parents=True, exist_ok=True)
    evidence: dict = {}
    ladder = [int(x) for x in args.ladder.split(",") if x.strip()]

    noise = 7.0
    if "noise" not in skip:
        noise, runs = phase_noise(args.fixture, args.workdir, ladder[0], 1,
                                  all_gpus, args.run_timeout_s)
        evidence["noise"] = [asdict(r) for r in runs]
    else:
        print(f"\n[noise] skipped; assuming {noise}%")
    evidence["noise_pct"] = noise

    shards = ladder[0]
    if "shards" not in skip:
        shards, runs = phase_shards(args.fixture, args.workdir, ladder, 1, all_gpus,
                                    gpu_total, noise, args.run_timeout_s)
        evidence["shards"] = [asdict(r) for r in runs]

    threads = 1
    if "threads" not in skip:
        threads, runs = phase_threads(args.fixture, args.workdir, shards, cores,
                                      all_gpus, noise, args.run_timeout_s)
        evidence["threads"] = [asdict(r) for r in runs]

    boltz_gpus = all_gpus
    if "layout" not in skip:
        boltz_gpus, runs = phase_layout(args.fixture, args.workdir, shards, threads,
                                        gpu_indices, noise, args.run_timeout_s)
        evidence["layout"] = [asdict(r) for r in runs]

    boltzgen_gpus = str(gpu_indices[-1])

    rec = {
        "boltz_shards": shards,
        "boltz_threads": threads,
        "boltz_gpus": boltz_gpus,
        "boltzgen_gpus": boltzgen_gpus,
    }

    if "epoch" not in skip and args.epoch_fixture:
        print("\n[epoch] full run at the recommended config")
        r = run_once("epoch", args.epoch_fixture, shards, threads, boltz_gpus,
                     boltzgen_gpus, args.workdir, args.run_timeout_s)
        evidence["epoch"] = asdict(r)
        if r.ok:
            serial = r.wall_s - max(r.boltz_s or 0, r.boltzgen_s or 0)
            headroom = args.deadline_s - r.wall_s
            rec["measured_epoch_s"] = round(r.wall_s, 1)
            rec["measured_boltz_s"] = round(r.boltz_s or 0, 1)
            rec["measured_boltzgen_s"] = round(r.boltzgen_s or 0, 1)
            rec["serial_overhead_s"] = round(serial, 1)
            rec["deadline_headroom_s"] = round(headroom, 1)
            rec["critical_path"] = ("boltz" if (r.boltz_s or 0) > (r.boltzgen_s or 0)
                                    else "boltzgen")
            print(f"  epoch {r.wall_s:.0f}s, headroom {headroom:.0f}s, "
                  f"critical path = {rec['critical_path']}")
            if headroom < 0:
                print("  WARNING: over the deadline at this molecule count")
    elif "epoch" not in skip:
        print("\n[epoch] skipped (--epoch-fixture not given)")

    write_config(args.out, env_info, rec, evidence)
    print("\nRecommendation:")
    for k, v in rec.items():
        print(f"  {k} = {v}")
    print("\nReview the file, then set these yourself -- nothing was applied.")


if __name__ == "__main__":
    main()
