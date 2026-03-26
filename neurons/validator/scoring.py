"""
PSICHIC-based molecular scoring functionality for the validator
"""

import math
from typing import Any, Optional

import bittensor as bt

from neurons.downloads import download_model_file
from utils import get_sequence_from_protein_code

# Global variable to store PSICHIC instance - will be set by validator.py
psichic = None
BASE_DIR = None


def _apply_failed_initialization_scores(
    score_dict: dict[int, dict[str, Any]],
    valid_molecules_by_uid: dict[int, dict[str, list[str]]],
    uid_to_data: Optional[dict[int, dict[str, Any]]],
    score_key: str,
    col_idx: int,
) -> list[int]:
    affected_uids = []
    for uid in score_dict:
        num_molecules = len(valid_molecules_by_uid.get(uid, {}).get('smiles', []))
        if num_molecules == 0 and uid_to_data:
            num_molecules = len(uid_to_data.get(uid, {}).get("molecules", []))
        score_dict[uid][score_key][col_idx] = [-math.inf] * num_molecules
        affected_uids.append(uid)

    return affected_uids


def score_all_proteins_psichic(
    target_proteins: list[str],
    antitarget_proteins: list[str],
    score_dict: dict[int, dict[str, Any]],
    valid_molecules_by_uid: dict[int, dict[str, list[str]]],
    uid_to_data: Optional[dict[int, dict[str, Any]]] = None,
    batch_size: int = 32,
) -> None:
    """
    Score all molecules against all proteins using efficient batching.
    This replaces the need to call score_protein_for_all_uids multiple times.
    
    Args:
        target_proteins: List of target protein codes
        antitarget_proteins: List of antitarget protein codes
        score_dict: Dictionary to store scores
        valid_molecules_by_uid: Dictionary of valid molecules by UID
        uid_to_data: Original UID data (for fallback molecule counts)
        batch_size: Number of molecules to process in each batch
    """
    global psichic, BASE_DIR
    
    # Ensure psichic is initialized
    if psichic is None:
        bt.logging.error("PSICHIC model not initialized.")
        return
    
    all_proteins = target_proteins + antitarget_proteins
    
    # Process each protein
    for protein_idx, protein in enumerate(all_proteins):
        is_target = protein_idx < len(target_proteins)
        col_idx = protein_idx if is_target else protein_idx - len(target_proteins)
        
        # Initialize PSICHIC for this protein
        bt.logging.info(f'Initializing model for protein code: {protein}')
        protein_sequence = get_sequence_from_protein_code(protein)
        
        try:
            psichic.run_challenge_start(protein_sequence)
            bt.logging.info('Model initialized successfully.')
        except Exception as e:
            bt.logging.warning(f"Model initialization failed for protein {protein}, retrying after re-download: {e}")
            try:
                if BASE_DIR:
                    download_model_file(
                        f"{BASE_DIR}/PSICHIC/trained_weights/TREAT1/model.pt",
                        "https://huggingface.co/Metanova/TREAT-1/resolve/main/model.pt",
                    )
                psichic.run_challenge_start(protein_sequence)
                bt.logging.info('Model initialized successfully.')
            except Exception as retry_error:
                score_key = "target_scores" if is_target else "antitarget_scores"
                affected_uids = _apply_failed_initialization_scores(
                    score_dict=score_dict,
                    valid_molecules_by_uid=valid_molecules_by_uid,
                    uid_to_data=uid_to_data,
                    score_key=score_key,
                    col_idx=col_idx,
                )
                bt.logging.error(
                    f"Error initializing model for protein {protein}: {retry_error}. "
                    f"Setting -inf for UIDs {affected_uids}."
                )
                continue
        
        # Collect all unique molecules across all UIDs
        unique_molecules = {}  # {smiles: [(uid, mol_idx), ...]}
        
        for uid, valid_molecules in valid_molecules_by_uid.items():
            if not valid_molecules.get('smiles'):
                # Set -inf scores for UIDs with no valid molecules
                num_molecules = 0
                if uid_to_data:
                    num_molecules = len(uid_to_data.get(uid, {}).get("molecules", []))
                score_dict[uid]["target_scores" if is_target else "antitarget_scores"][col_idx] = [-math.inf] * num_molecules
                continue
            
            for mol_idx, smiles in enumerate(valid_molecules['smiles']):
                if smiles not in unique_molecules:
                    unique_molecules[smiles] = []
                unique_molecules[smiles].append((uid, mol_idx))
        
        # Process unique molecules in batches
        unique_smiles_list = list(unique_molecules.keys())
        molecule_scores = {}  # {smiles: score}
        
        for batch_start in range(0, len(unique_smiles_list), batch_size):
            batch_end = min(batch_start + batch_size, len(unique_smiles_list))
            batch_molecules = unique_smiles_list[batch_start:batch_end]
            
            try:
                # Score the batch
                results_df = psichic.run_validation(batch_molecules)
                
                if not results_df.empty and len(results_df) == len(batch_molecules):
                    for idx, smiles in enumerate(batch_molecules):
                        val = results_df.iloc[idx].get('predicted_binding_affinity')
                        score_value = float(val) if val is not None else -math.inf
                        molecule_scores[smiles] = score_value
                else:
                    bt.logging.warning(f"Unexpected results for batch, falling back to individual scoring")
                    for smiles in batch_molecules:
                        molecule_scores[smiles] = score_molecule_individually(smiles)
            except Exception as e:
                bt.logging.error(f"Error scoring batch: {e}")
                for smiles in batch_molecules:
                    molecule_scores[smiles] = score_molecule_individually(smiles)
        
        # Distribute scores to all UIDs
        for uid, valid_molecules in valid_molecules_by_uid.items():
            if not valid_molecules.get('smiles'):
                continue
            
            uid_scores = []
            for smiles in valid_molecules['smiles']:
                score = molecule_scores.get(smiles, -math.inf)
                uid_scores.append(score)
            
            if is_target:
                score_dict[uid]["target_scores"][col_idx] = uid_scores
            else:
                score_dict[uid]["antitarget_scores"][col_idx] = uid_scores
        
        bt.logging.info(f"Completed scoring for protein {protein}: {len(unique_molecules)} unique molecules")


def score_molecule_individually(smiles: str) -> float:
    """Helper function to score a single molecule."""
    global psichic
    
    if psichic is None:
        bt.logging.error("PSICHIC model not initialized.")
        return -math.inf
    
    try:
        results_df = psichic.run_validation([smiles])
        if not results_df.empty:
            val = results_df.iloc[0].get('predicted_binding_affinity')
            return float(val) if val is not None else -math.inf
        else:
            return -math.inf
    except Exception as e:
        bt.logging.error(f"Error scoring molecule {smiles}: {e}")
        return -math.inf

