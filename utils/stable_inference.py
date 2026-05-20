"""
Stable inference module for deterministic Boltz2 scoring.

Provides process-isolated molecular scoring to eliminate cross-molecule
interference caused by shared model state and RNG pollution.

Design: Physical isolation via spawn processes, each with independent
CUDA context and fresh model loading.
"""

import multiprocessing as mp
import torch
import gc
import os
import sys
import logging
from typing import Optional

NOVA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, NOVA_DIR)
sys.path.insert(0, os.path.join(NOVA_DIR, "external_tools", "boltz", "src"))

logger = logging.getLogger(__name__)


def _worker_process(
    smiles_list: list[str],
    target: str,
    protein_sequence: str,
    base_seed: int,
    result_queue: mp.Queue,
    gpu_id: str = "0",
) -> None:
    """Worker process: runs predictions in complete isolation.

    Each call creates a fresh BoltzWrapper instance with independent
    CUDA context. Model is deleted and GPU memory freed after scoring.
    
    Args:
        smiles_list: SMILES strings to score.
        target: Protein target code.
        protein_sequence: Protein amino acid sequence.
        base_seed: Base random seed.
        result_queue: Queue to return results.
        gpu_id: GPU device ID to use (default "0").
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        from external_tools.boltz.boltz_wrapper import BoltzWrapper, _get_molecule_seed

        model = BoltzWrapper()
        scores: dict[str, float] = {}
        per_molecule_components: dict = {}

        for smiles in smiles_list:
            try:
                mol_idx = _get_molecule_seed(smiles, base_seed)
                mol_seed = _get_molecule_seed(mol_idx, base_seed)

                result = model.predict_single_molecule(
                    smiles, target, protein_sequence, mol_seed
                )

                # Extract final score
                pred_value = result.get("affinity_pred_value")
                if isinstance(pred_value, list):
                    pred_value = pred_value[0]
                scores[smiles] = float(pred_value) if pred_value is not None else 0.0

                # Extract full metrics for per_molecule_components
                metrics = model._extract_metrics(result)
                per_molecule_components[smiles] = {target: metrics}

            except Exception as e:
                logger.error(f"Failed to score molecule {smiles}: {e}")
                scores[smiles] = 0.0
                per_molecule_components[smiles] = {target: {}}

        result_queue.put(("SUCCESS", {
            "scores": scores,
            "per_molecule_components": per_molecule_components,
        }))
    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"Worker process failed: {error_msg}")
        result_queue.put(("ERROR", error_msg))
    finally:
        if "model" in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()


def _split_for_gpus(smiles_list: list[str], num_workers: int) -> list[list[str]]:
    """Split smiles list into chunks for each worker.
    
    Args:
        smiles_list: List of SMILES strings.
        num_workers: Number of workers to split across.
        
    Returns:
        List of chunks, each containing a subset of smiles.
    """
    if num_workers <= 1:
        return [smiles_list]
    
    chunks = []
    n = len(smiles_list)
    base = n // num_workers
    extra = n % num_workers
    
    start = 0
    for i in range(num_workers):
        size = base + (1 if i < extra else 0)
        if start < n:
            chunks.append(smiles_list[start:start + size])
            start += size
    
    return chunks


def _run_single_worker(
    smiles_list: list[str],
    target: str,
    protein_sequence: str,
    base_seed: int,
    gpu_id: str,
    timeout_seconds: int,
) -> tuple[dict, dict]:
    """Run a single worker process and return results.
    
    Args:
        smiles_list: SMILES strings to score.
        target: Protein target code.
        protein_sequence: Protein amino acid sequence.
        base_seed: Base random seed.
        gpu_id: GPU device ID.
        timeout_seconds: Maximum time to wait.
        
    Returns:
        Tuple of (scores dict, per_molecule_components dict).
        
    Raises:
        RuntimeError: If worker fails or times out.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    p = ctx.Process(
        target=_worker_process,
        args=(smiles_list, target, protein_sequence, base_seed, result_queue, gpu_id),
    )
    p.start()

    try:
        status, data = result_queue.get(timeout=timeout_seconds)
    except Exception as e:
        p.terminate()
        p.join()
        raise RuntimeError(
            f"Worker process on GPU {gpu_id} timed out or failed after {timeout_seconds}s: {e}"
        )

    p.join()

    if status == "ERROR":
        raise RuntimeError(f"Worker process on GPU {gpu_id} error: {data}")

    return data["scores"], data.get("per_molecule_components", {})


def _run_multi_worker(
    smiles_list: list[str],
    target: str,
    protein_sequence: str,
    base_seed: int,
    gpu_ids: list[str],
    timeout_seconds: int,
) -> tuple[dict, dict]:
    """Run multiple worker processes across GPUs and aggregate results.
    
    Args:
        smiles_list: SMILES strings to score.
        target: Protein target code.
        protein_sequence: Protein amino acid sequence.
        base_seed: Base random seed.
        gpu_ids: List of GPU device IDs.
        timeout_seconds: Maximum time to wait per worker.
        
    Returns:
        Tuple of (scores dict, per_molecule_components dict).
        
    Raises:
        RuntimeError: If any worker fails or times out.
    """
    num_workers = len(gpu_ids)
    chunks = _split_for_gpus(smiles_list, num_workers)
    
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = ctx.Process(
            target=_worker_process,
            args=(chunk, target, protein_sequence, base_seed, result_queue, gpu_ids[i]),
        )
        p.start()
        processes.append(p)
    
    if not processes:
        return {}, {}
    
    # Collect results from all workers
    all_scores = {}
    all_components = {}
    errors = []
    
    for _ in range(len(processes)):
        try:
            status, data = result_queue.get(timeout=timeout_seconds)
            if status == "ERROR":
                errors.append(data)
            else:
                all_scores.update(data["scores"])
                all_components.update(data.get("per_molecule_components", {}))
        except Exception as e:
            errors.append(str(e))
    
    # Ensure all processes are terminated
    for p in processes:
        if p.is_alive():
            p.terminate()
        p.join()
    
    if errors:
        raise RuntimeError(f"Multiprocessing inference errors: {errors}")
    
    # Verify all molecules were scored
    missing = set(smiles_list) - set(all_scores.keys())
    if missing:
        raise RuntimeError(f"Missing scores for molecules: {missing}")
    
    return all_scores, all_components


def get_stable_scores(
    smiles_list: list[str],
    target: str,
    protein_sequence: str,
    base_seed: int = 68,
    gpu_ids: str = "all",
    timeout_seconds: int = 3600,
) -> tuple[dict[str, float], dict]:
    """Score molecules with deterministic, isolated predictions.

    Each molecule is scored in a separate spawn process with independent
    CUDA context and fresh model loading. This guarantees that:
    - Same molecule + same seed = same score (regardless of other molecules)
    - No cross-molecule RNG interference
    - No model state accumulation

    Args:
        smiles_list: List of SMILES strings to score.
        target: Protein target code (e.g., "Q92769").
        protein_sequence: Full protein amino acid sequence.
        base_seed: Base random seed for deriving per-molecule seeds.
        gpu_ids: GPU device ID(s) to use. Default "all" for all GPUs.
            Supports "all" for all available GPUs, or comma-separated like "0,1,2,3".
        timeout_seconds: Maximum time to wait for each worker subprocess (default 3600s).

    Returns:
        Tuple of (scores dict, per_molecule_components dict).
        scores: {smiles: final_score}
        per_molecule_components: {smiles: {target: {metric: value}}}

    Raises:
        RuntimeError: If subprocess fails or times out.
    """
    if not smiles_list:
        return {}, {}

    unique_smiles = sorted(set(smiles_list))
    
    # Parse GPU configuration
    if gpu_ids.lower() == "all":
        num_gpus = torch.cuda.device_count()
        gpu_ids = [str(i) for i in range(num_gpus)]
    else:
        gpu_ids = [g.strip() for g in gpu_ids.split(",")]
    
    # Single GPU: use original behavior
    if len(gpu_ids) <= 1:
        return _run_single_worker(
            unique_smiles, target, protein_sequence, base_seed, gpu_ids[0], timeout_seconds
        )
    
    # Multi-GPU: split work across GPUs
    return _run_multi_worker(
        unique_smiles, target, protein_sequence, base_seed, gpu_ids, timeout_seconds
    )
