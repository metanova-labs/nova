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
    """stand-in for BoltzgenWrapper when inference runs in subprocesses"""

    per_nanobody_components: dict
    final_boltzgen_scores: dict | None


class InferenceResult(NamedTuple):
    """Result of inference.main(). score_dict is updated in-place; use .boltz and .boltzgen to access wrapper-like attributes."""

    boltz: BoltzResult | None
    boltzgen: BoltzgenResult | None


def infer_worker(gpu_id: int, payload: dict, inference_type: str) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if inference_type == "boltz":
        from boltz.boltz_wrapper import BoltzWrapper
        boltz = BoltzWrapper()
        boltz.score_molecules(payload["molecules"], payload["score_dict"], payload["config"])
        score_dict_updates = {
            uid: {"molecule_scores": payload["score_dict"][uid]["molecule_scores"]}
            for uid in payload["score_dict"]
        }
        return {
            "gpu": gpu_id,
            "ok": True,
            "boltz": score_dict_updates,
            "per_molecule_components": getattr(boltz, "per_molecule_components", {}),
            "unique_molecules": getattr(boltz, "unique_molecules", {}),
        }

    elif inference_type == "boltzgen":
        from boltzgen.boltzgen_wrapper import BoltzgenWrapper
        boltzgen = BoltzgenWrapper()
        final_boltzgen_scores, per_nanobody_components = boltzgen.score_nanobodies(
            payload["nanobodies"], payload["config"]
        )
        return {
            "gpu": gpu_id,
            "ok": True,
            "boltzgen": final_boltzgen_scores,
            "per_nanobody_components": per_nanobody_components,
        }
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
        bt.logging.debug(f"_merge_boltzgen_into_score_dict: Set nanobody_scores for UID {uid}: {len(rows)} targets, {len(rows[0]) if rows else 0} sequences")


def main(valid_molecules_by_uid: dict, valid_nanobodies_by_uid: dict, score_dict: dict, config) -> InferenceResult:
    """
    Run Boltz and/or Boltzgen inference, each on its own GPU when available.

    - score_dict is updated in-place: molecule_scores (Boltz), nanobody_scores (Boltzgen).
    - Returns InferenceResult(boltz=..., boltzgen=...) so the validator can use result.boltz / result.boltzgen
      like the wrapper instances (e.g. result.boltz.per_molecule_components, etc).
    """
    run_boltz = bool(valid_molecules_by_uid)
    run_boltzgen = bool(valid_nanobodies_by_uid)
    if not run_boltz and not run_boltzgen:
        return InferenceResult(boltz=None, boltzgen=None)

    num_gpus = torch.cuda.device_count()
    bt.logging.info(f"Detected GPUs: {num_gpus}. Boltz={run_boltz}, Boltzgen={run_boltzgen}")

    payload_boltz = {"molecules": valid_molecules_by_uid, "score_dict": score_dict, "config": config}
    payload_boltzgen = {"nanobodies": valid_nanobodies_by_uid, "config": config}

    final_boltzgen_scores = None
    per_nanobody_components = None
    per_molecule_components = None
    unique_molecules = None
    ctx = mp.get_context("spawn")
    rank_mode = getattr(config, "boltzgen_rank_mode", None) or getattr(config, "rank_mode", "min")

    if num_gpus >= 2 and run_boltz and run_boltzgen:
        with ctx.Pool(processes=2) as pool:
            r_boltz = pool.apply_async(infer_worker, (0, payload_boltz, "boltz"))
            r_boltzgen = pool.apply_async(infer_worker, (1, payload_boltzgen, "boltzgen"))
            out_boltz = r_boltz.get()
            out_boltzgen = r_boltzgen.get()
        _merge_boltz_into_score_dict(score_dict, out_boltz)
        per_molecule_components = out_boltz.get("per_molecule_components") or {}
        unique_molecules = out_boltz.get("unique_molecules") or {}
        if out_boltzgen.get("ok"):
            final_boltzgen_scores = out_boltzgen.get("boltzgen")
            per_nanobody_components = out_boltzgen.get("per_nanobody_components")
    else:
        with ctx.Pool(processes=1) as pool:
            if run_boltz:
                out = pool.apply_async(infer_worker, (0, payload_boltz, "boltz")).get()
                _merge_boltz_into_score_dict(score_dict, out)
                per_molecule_components = out.get("per_molecule_components") or {}
                unique_molecules = out.get("unique_molecules") or {}
            if run_boltzgen:
                out = pool.apply_async(infer_worker, (0, payload_boltzgen, "boltzgen")).get()
                if out.get("ok"):
                    final_boltzgen_scores = out.get("boltzgen")
                    per_nanobody_components = out.get("per_nanobody_components")

    _merge_boltzgen_into_score_dict(score_dict, final_boltzgen_scores, valid_nanobodies_by_uid, config, rank_mode=rank_mode)

    boltz = BoltzResult(per_molecule_components, unique_molecules) if run_boltz else None
    boltzgen = BoltzgenResult(per_nanobody_components or {}, final_boltzgen_scores) if run_boltzgen else None
    return InferenceResult(boltz=boltz, boltzgen=boltzgen)
