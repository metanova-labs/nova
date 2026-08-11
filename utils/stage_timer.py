"""Lightweight wall-clock stage timing for epoch profiling.

Purely additive: emits STAGE log lines that can be aligned against a
per-GPU utilization trace. Remove this module and its call sites to revert.
"""

import time
from contextlib import contextmanager

import bittensor as bt


@contextmanager
def stage(name: str, **extra):
    detail = " ".join(f"{k}={v}" for k, v in extra.items())
    bt.logging.info(f"STAGE_BEGIN {name} t={time.time():.3f} {detail}".rstrip())
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        bt.logging.info(
            f"STAGE_END {name} t={time.time():.3f} elapsed_s={elapsed:.2f} {detail}".rstrip()
        )
