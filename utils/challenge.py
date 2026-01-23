import random
import time

import bittensor as bt
from datasets import load_dataset
from huggingface_hub import hf_hub_url, hf_hub_download, get_hf_file_metadata
from huggingface_hub.errors import EntryNotFoundError
import pandas as pd
from rdkit import Chem

def get_challenge_params_from_blockhash(block_hash: str, small_molecule_target: str, nanobody_target: str, num_antitargets: int = 0, include_reaction: bool = False) -> dict:
    """
    Use block_hash as a seed to pick 'random_valid_reaction' and/or 'antitargets' random entries. 
    Returns {'small_molecule_target': ..., 'nanobody_target': ..., 'random_valid_reaction': ..., 'antitargets': [...], 'allowed_reaction': '...'}.
    """
    if not (isinstance(block_hash, str) and block_hash.startswith("0x")):
        bt.logging.error("block_hash must start with '0x'.")
        return None

    # Convert block hash to an integer seed
    try:
        seed = int(block_hash[2:], 16)
    except ValueError:
        raise ValueError(f"Invalid hex in block_hash: {block_hash}")

    # Initialize random number generator
    rng = random.Random(seed)

    result = {
        "small_molecule_target": small_molecule_target,
        "nanobody_target": nanobody_target,
    }

    if num_antitargets > 0:
        # Load huggingface protein dataset
        try:
            dataset = load_dataset("Metanova/Proteins", split="train")
        except Exception as e:
            bt.logging.error(f"Could not load the 'Metanova/Proteins' dataset: {e}")
            return None

        dataset_size = len(dataset)
        if dataset_size == 0:
            bt.logging.error("Dataset is empty; cannot pick random entries.")
            return None

        # Grab all required indices at once, ensure uniqueness
        unique_indices = rng.sample(range(dataset_size), k=(num_antitargets))

        # Split indices for antitargets
        antitarget_indices = unique_indices[:num_antitargets]

        # Convert indices to protein codes
        antitargets = [dataset[i]["Entry"] for i in antitarget_indices]
        result["antitargets"] = antitargets

    if include_reaction:
        try:
            from .reactions import get_total_reactions
            total_reactions = get_total_reactions()
            allowed_option = seed % total_reactions
            result["allowed_reaction"] = f"rxn:{allowed_option + 1}"
        except Exception as e:
            bt.logging.warning(f"Failed to determine allowed reaction: {e}, defaulting to all reactions allowed")

    return result

def entry_unique_for_protein_hf(protein: str, entry_id: str, entity_type: str = 'molecules') -> bool:
    """
    Check if entry exists in Hugging Face Submission-Archive dataset by comparing InChIKeys or sequence hashes.
    entry_id: smiles for molecules or sequence hash for nanobodies
    entity_type: 'molecules' or 'nanobodies'
    Returns True if unique (not found), False if found.
    """
    
    if entity_type == 'molecules':
        entity_id = 'InChI_Key'
    elif entity_type == 'nanobodies':
        entity_id = 'sequence_hash'
    else:
        bt.logging.error(f"Invalid entity type: {entity_type}")
        return False

    if not hasattr(entry_unique_for_protein_hf, "_CACHE"):
        entry_unique_for_protein_hf._CACHE = (None, None, None, None, 0)
    
    try:
        cached_protein, cached_entity_type, cached_sha, entry_ids_set, last_check_time = entry_unique_for_protein_hf._CACHE
        current_time = time.time()
        metadata_ttl = 60 
        
        if protein != cached_protein:
            bt.logging.debug(f"Switching from protein {cached_protein} to {protein}")
            cached_entity_type = None
            cached_sha = None
            entry_ids_set = None
        
        if cached_entity_type != entity_type:
            bt.logging.debug(f"Switching from entity type {cached_entity_type} to {entity_type}")
            cached_entity_type = entity_type
            cached_sha = None
            entry_ids_set = None
        
        filename = f"{protein}_{entity_type}.csv"
        
        if cached_sha is None or (current_time - last_check_time > metadata_ttl):
            url = hf_hub_url(
                repo_id="Metanova/Submission-Archive",
                filename=filename,
                repo_type="dataset"
            )
            
            metadata = get_hf_file_metadata(url)
            current_sha = metadata.commit_hash
            last_check_time = current_time
            
            if cached_sha != current_sha:
                file_path = hf_hub_download(
                    repo_id="Metanova/Submission-Archive",
                    filename=filename,
                    repo_type="dataset",
                    revision=current_sha
                )
                
                df = pd.read_csv(file_path, usecols=[entity_id])
                entry_ids_set = set(df[entity_id])
                bt.logging.debug(f"Loaded {len(entry_ids_set)} {entity_id} into lookup set for {protein} (commit {current_sha[:7]})")
                
                entry_unique_for_protein_hf._CACHE = (protein, entity_type, current_sha, entry_ids_set, last_check_time)
            else:
                entry_unique_for_protein_hf._CACHE = entry_unique_for_protein_hf._CACHE[:4] + (last_check_time,)
        
        
        if entity_type == 'molecules':
            mol = Chem.MolFromSmiles(entry_id)
            if mol is None:
                bt.logging.warning(f"Could not parse SMILES string: {smiles}")
                return True  # Assume unique if we can't parse the SMILES
                
            inchikey = Chem.MolToInchiKey(mol)
            
            return inchikey not in entry_ids_set

        elif entity_type == 'nanobodies':
            return entry_id not in entry_ids_set
        
    except EntryNotFoundError:
        # File doesn't exist, cache empty set to avoid repeated calls
        entry_ids_set = set()
        entry_unique_for_protein_hf._CACHE = (protein, entity_type, 'not_found', entry_ids_set, time.time())
        bt.logging.debug(f"File {filename} not found on HF, caching empty result")
        return True
    except Exception as e:
        # Assume entry is unique if there's an error
        bt.logging.warning(f"Error checking entry in HF dataset: {e}")
        return True
