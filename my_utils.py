import requests
import os
import json
from dotenv import load_dotenv
import bittensor as bt
from datasets import load_dataset
import random
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np
import math
import pandas as pd
from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata
from huggingface_hub.errors import EntryNotFoundError
import time
import datetime
import sqlite3
from combinatorial_db.reactions import get_smiles_from_reaction

load_dotenv(override=True)

def upload_file_to_github(filename: str, encoded_content: str):
    # Github configs
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')   # example: nova
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH') # example: main
    github_token = os.environ.get('GITHUB_TOKEN')
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER') # example: metanova-labs
    github_repo_path = os.environ.get('GITHUB_REPO_PATH') # example: /data/results or ""

    if not github_repo_name or not github_repo_branch or not github_token or not github_repo_owner:
        raise ValueError("Github environment variables not set. Please set them in your .env file.")

    target_file_path = os.path.join(github_repo_path, f'{filename}.txt')
    url = f"https://api.github.com/repos/{github_repo_owner}/{github_repo_name}/contents/{target_file_path}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        }

    # Check if the file already exists (need its SHA to update)
    existing_file = requests.get(url, headers=headers, params={"ref": github_repo_branch})
    sha = existing_file.json().get("sha") if existing_file.status_code == 200 else None

    payload = {
        "message": f"Encrypted response for {filename}",
        "content": encoded_content,
        "branch": github_repo_branch,
    }
    if sha:
        payload["sha"] = sha  # updating existing file

    response = requests.put(url, headers=headers, json=payload)
    if response.status_code in [200, 201]:
        return True
    else:
        bt.logging.error(f"Failed to upload file for {filename}: {response.status_code} {response.text}")
        return False


def get_smiles(product_name):
    # Remove single and double quotes from product_name if they exist
    if product_name:
        product_name = product_name.replace("'", "").replace('"', "")
    else:
        bt.logging.error("Product name is empty.")
        return None

    if product_name.startswith("rxn:"):
        return get_smiles_from_reaction(product_name)

    api_key = os.environ.get("VALIDATOR_API_KEY")
    if not api_key:
        raise ValueError("validator_api_key environment variable not set.")

    url = f"https://8vzqr9wt22.execute-api.us-east-1.amazonaws.com/dev/smiles/{product_name}"

    headers = {"x-api-key": api_key}
    
    response = requests.get(url, headers=headers)

    data = response.json()

    return data.get("smiles")



def get_sequence_from_protein_code(protein_code:str) -> str:
    """
    Get the amino acid sequence for a protein code.
    First tries to fetch from UniProt API, and if that fails,
    falls back to searching the Hugging Face dataset.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{protein_code}.fasta"
    response = requests.get(url)

    if response.status_code == 200:
        lines = response.text.splitlines()
        sequence_lines = [line.strip() for line in lines if not line.startswith('>')]
        amino_acid_sequence = ''.join(sequence_lines)
        # Check if the sequence is empty
        if not amino_acid_sequence:
            bt.logging.warning(f"Retrieved empty sequence for {protein_code} from UniProt API")
        else:
            return amino_acid_sequence
    
    bt.logging.info(f"Failed to retrieve sequence for {protein_code} from UniProt API. Trying Hugging Face dataset.")
    try:
        dataset = load_dataset("Metanova/Proteins", split="train")
        
        for i in range(len(dataset)):
            if dataset[i]["Entry"] == protein_code:
                sequence = dataset[i]["Sequence"]
                bt.logging.info(f"Found sequence for {protein_code} in Hugging Face dataset")
                return sequence
                
        bt.logging.error(f"Could not find protein {protein_code} in Hugging Face dataset")
        return None
        
    except Exception as e:
        bt.logging.error(f"Error accessing Hugging Face dataset: {e}")
        return None

def get_challenge_proteins_from_blockhash(block_hash: str, weekly_target: str, num_antitargets: int) -> dict:
    """
    Use block_hash as a seed to pick 'num_targets' and 'num_antitargets' random entries
    from the 'Metanova/Proteins' dataset. Returns {'targets': [...], 'antitargets': [...]}.
    """
    if not (isinstance(block_hash, str) and block_hash.startswith("0x")):
        raise ValueError("block_hash must start with '0x'.")
    if not weekly_target or num_antitargets < 0:
        raise ValueError("weekly_target must exist and num_antitargets must be non-negative.")

    # Convert block hash to an integer seed
    try:
        seed = int(block_hash[2:], 16)
    except ValueError:
        raise ValueError(f"Invalid hex in block_hash: {block_hash}")

    # Initialize random number generator
    rng = random.Random(seed)

    # Load huggingface protein dataset
    try:
        dataset = load_dataset("Metanova/Proteins", split="train")
    except Exception as e:
        raise RuntimeError("Could not load the 'Metanova/Proteins' dataset.") from e

    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Dataset is empty; cannot pick random entries.")

    # Grab all required indices at once, ensure uniqueness
    unique_indices = rng.sample(range(dataset_size), k=(num_antitargets))

    # Split indices for antitargets
    antitarget_indices = unique_indices[:num_antitargets]

    # Convert indices to protein codes
    targets = [weekly_target]
    antitargets = [dataset[i]["Entry"] for i in antitarget_indices]

    return {
        "targets": targets,
        "antitargets": antitargets
    }

def get_heavy_atom_count(smiles: str) -> int:
    """
    Calculate the number of heavy atoms in a molecule from its SMILES string.
    """
    count = 0
    i = 0
    while i < len(smiles):
        c = smiles[i]
        
        if c.isalpha() and c.isupper():
            elem_symbol = c
            
            # If the next character is a lowercase letter, include it (e.g., 'Cl', 'Br')
            if i + 1 < len(smiles) and smiles[i + 1].islower():
                elem_symbol += smiles[i + 1]
                i += 1 
            
            # If it's not 'H', count it as a heavy atom
            if elem_symbol != 'H':
                count += 1
        
        i += 1
    
    return count
    
def compute_maccs_entropy(smiles_list: list[str]) -> float:
    """
    Computes fingerprint entropy from MACCS keys for a list of SMILES.

    Parameters:
        smiles_list (list of str): Molecules in SMILES format.

    Returns:
        avg_entropy (float): Average entropy per bit.
    """
    n_bits = 167  # RDKit uses 167 bits (index 0 is always 0)
    bit_counts = np.zeros(n_bits)
    valid_mols = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.array(fp)
            bit_counts += arr
            valid_mols += 1

    if valid_mols == 0:
        raise ValueError("No valid molecules found.")

    probs = bit_counts / valid_mols
    entropy_per_bit = np.array([
        -p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0
        for p in probs
    ])

    avg_entropy = np.mean(entropy_per_bit)

    return avg_entropy

def molecule_unique_for_protein_api(protein: str, molecule: str) -> bool:
    """
    Check if a molecule has been previously submitted for the same target protein in any competition.
    """
    api_key = os.environ.get("VALIDATOR_API_KEY")
    if not api_key:
        raise ValueError("validator_api_key environment variable not set.")
    
    url = f"https://dashboard-backend-multitarget.up.railway.app/api/molecule_seen/{molecule}/{protein}"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            bt.logging.error(f"Failed to check molecule uniqueness: {response.status_code} {response.text}")
            return True
            
        data = response.json()
        return not data.get("seen", False)
        
    except Exception as e:
        bt.logging.error(f"Error checking molecule uniqueness: {e}")
        return True

def molecule_unique_for_protein_hf(protein: str, smiles: str) -> bool:
    """
    Check if molecule exists in Hugging Face Submission-Archive dataset by comparing InChIKeys.
    Returns True if unique (not found), False if found.
    """
    if not hasattr(molecule_unique_for_protein_hf, "_CACHE"):
        molecule_unique_for_protein_hf._CACHE = (None, None, None, 0)
    
    try:
        cached_protein, cached_sha, inchikeys_set, last_check_time = molecule_unique_for_protein_hf._CACHE
        current_time = time.time()
        metadata_ttl = 60 
        
        if protein != cached_protein:
            bt.logging.debug(f"Switching from protein {cached_protein} to {protein}")
            cached_sha = None 
        
        filename = f"{protein}_molecules.csv"
        
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
                
                df = pd.read_csv(file_path, usecols=["InChI_Key"])
                inchikeys_set = set(df["InChI_Key"])
                bt.logging.debug(f"Loaded {len(inchikeys_set)} InChI Keys into lookup set for {protein} (commit {current_sha[:7]})")
                
                molecule_unique_for_protein_hf._CACHE = (protein, current_sha, inchikeys_set, last_check_time)
            else:
                molecule_unique_for_protein_hf._CACHE = molecule_unique_for_protein_hf._CACHE[:3] + (last_check_time,)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            bt.logging.warning(f"Could not parse SMILES string: {smiles}")
            return True  # Assume unique if we can't parse the SMILES
            
        inchikey = Chem.MolToInchiKey(mol)
        
        return inchikey not in inchikeys_set
        
    except EntryNotFoundError:
        # File doesn't exist, cache empty set to avoid repeated calls
        inchikeys_set = set()
        molecule_unique_for_protein_hf._CACHE = (protein, 'not_found', inchikeys_set, time.time())
        bt.logging.debug(f"File {filename} not found on HF, caching empty result")
        return True
    except Exception as e:
        # Assume molecule is unique if there's an error
        bt.logging.warning(f"Error checking molecule in HF dataset: {e}")
        return True
    
def find_chemically_identical(smiles_list: list[str]) -> dict:
    """
    Check for identical molecules in a list of SMILES strings by converting to InChIKeys.
    """
    inchikey_to_indices = {}
    
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                inchikey = Chem.MolToInchiKey(mol)
                if inchikey not in inchikey_to_indices:
                    inchikey_to_indices[inchikey] = []
                inchikey_to_indices[inchikey].append(i)
        except Exception as e:
            bt.logging.warning(f"Error processing SMILES {smiles}: {e}")
    
    duplicates = {k: v for k, v in inchikey_to_indices.items() if len(v) > 1}
    
    return duplicates

def calculate_dynamic_entropy(starting_weight: float, step_size: float, start_epoch: int, current_epoch: int) -> float:
    """
    Calculate entropy weight based on epochs elapsed since start epoch.
    """
    epochs_elapsed = current_epoch - start_epoch
    
    entropy_weight = starting_weight + (epochs_elapsed * step_size)
    entropy_weight = max(0, entropy_weight)
    
    bt.logging.info(f"Epochs elapsed: {epochs_elapsed}, entropy weight: {entropy_weight}")
    return entropy_weight

def monitor_validator(score_dict, metagraph, current_epoch, current_block, validator_hotkey, winning_uid):
    api_key = os.environ.get('VALIDATOR_API_KEY')
    if not api_key:
        return
    
    try:
        import torch
        machine_info = {
            "torch_version": torch.__version__
        }
        if torch.cuda.is_available():
            machine_info["cuda_version"] = torch.version.cuda
            machine_info["gpu_name"] = torch.cuda.get_device_name(0)
        
        best_rounded_score = max([round(d['final_score'], 4) for d in score_dict.values() if 'final_score' in d], default=-math.inf)
        
        winning_group = []
        for uid, data in score_dict.items():
            if 'final_score' in data and round(data['final_score'], 4) == best_rounded_score:
                winning_group.append({
                    "uid": uid,
                    "hotkey": metagraph.hotkeys[uid] if uid < len(metagraph.hotkeys) else "unknown",
                    "final_score": data['final_score'],
                    "blocks_elapsed": (data.get('block_submitted', 0) % 361),
                    "push_time": data.get('push_time', ''),
                    "winner": uid == winning_uid
                })
        
        requests.post("https://valiwatch-production.up.railway.app/weights-info", json={
            "epoch": current_epoch,
            "current_block": current_block,
            "blocks_into_epoch": current_block % 361,
            "validator_hotkey": validator_hotkey,
            "validator_version": 1.3, #deterministic settings
            "winning_group": winning_group,
            "machine_info": machine_info
        }, headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
        
    except Exception as e:
        bt.logging.debug(f"API send failed: {e}")