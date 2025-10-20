import math
import numpy as np
import pandas as pd
import time
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata
from huggingface_hub.errors import EntryNotFoundError
import bittensor as bt
from combinatorial_db.reactions import get_smiles_from_reaction
import requests
import os
from dotenv import load_dotenv

load_dotenv(override=True)


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

def is_boltz_safe_smiles(smiles: str) -> tuple[bool, str | None]:
    """
    Replicates Boltz atom-name generation and enforces <= 4 characters.
    Returns (ok, reason). ok=False means the SMILES should be rejected for Boltz.
    """
    try:
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return False, "RDKit failed to parse SMILES"
        mol = AllChem.AddHs(mol)
        canonical_order = AllChem.CanonicalRankAtoms(mol)
        for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
            atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
            if len(atom_name) > 4:
                return False, f"Atom name would exceed 4 chars: {atom_name}"
        return True, None
    except Exception as e:
        return False, f"Boltz safety check failed: {e}"


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
