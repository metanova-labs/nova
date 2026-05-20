import math
import multiprocessing as mp
import os
import sys
from typing import NamedTuple

NOVA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(NOVA_DIR)

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


def infer_worker(gpu_id: int, payload: dict, inference_type: str) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if inference_type == "boltzgen":
        from boltzgen.boltzgen_wrapper import BoltzgenWrapper
        boltzgen = BoltzgenWrapper()
        per_nanobody_components = boltzgen.run_nanobody_inference(
            payload["nanobodies"], payload["config"]
        )
        return {
            "gpu": gpu_id,
            "ok": True,
            "per_nanobody_components": per_nanobody_components,
        }
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


def run_nanobody_inference(valid_nanobodies_by_uid: dict, config) -> BoltzgenResult | None:
    """Run Boltzgen inference for nanobodies in a spawn subprocess.

    Args:
        valid_nanobodies_by_uid: Dict mapping UID to nanobody data.
        config: Subnet configuration.

    Returns:
        BoltzgenResult with per_nanobody_components, or None if no nanobodies.
    """
    if not valid_nanobodies_by_uid:
        return None

    bt.logging.info("Running Boltzgen inference for nanobodies...")
    payload = {"nanobodies": valid_nanobodies_by_uid, "config": config}

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        out = pool.apply_async(infer_worker, (0, payload, "boltzgen")).get()

    if not out.get("ok"):
        bt.logging.error(f"Nanobody inference failed: {out.get('error')}")
        return BoltzgenResult({}, None)

    per_nanobody_components = out.get("per_nanobody_components") or {}
    return BoltzgenResult(per_nanobody_components, None)
