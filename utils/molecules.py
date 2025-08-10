import os
import math
import time
import sqlite3
import requests
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import bittensor as bt
from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata
from huggingface_hub.errors import EntryNotFoundError
from combinatorial_db.reactions import get_smiles_from_reaction


def get_smiles(product_name: str):
    """Return SMILES string for a product name or reaction identifier."""
    if product_name:
        product_name = product_name.replace("'", "").replace('"', "")
    else:
        bt.logging.error("Product name is empty.")
        return None
    if product_name.startswith("rxn:"):
        return get_smiles_from_reaction(product_name)
    api_key = os.environ.get('VALIDATOR_API_KEY')
    if not api_key:
        raise ValueError('validator_api_key environment variable not set.')
    url = f"https://8vzqr9wt22.execute-api.us-east-1.amazonaws.com/dev/smiles/{product_name}"
    headers = {"x-api-key": api_key}
    response = requests.get(url, headers=headers)
    data = response.json()
    return data.get('smiles')


def get_heavy_atom_count(smiles: str) -> int:
    """Calculate the number of heavy atoms in a molecule from its SMILES string."""
    count = 0
    i = 0
    while i < len(smiles):
        c = smiles[i]
        if c.isalpha() and c.isupper():
            elem_symbol = c
            if i + 1 < len(smiles) and smiles[i + 1].islower():
                elem_symbol += smiles[i + 1]
                i += 1
            if elem_symbol != 'H':
                count += 1
        i += 1
    return count


def compute_maccs_entropy(smiles_list: list[str]) -> float:
    """Compute fingerprint entropy from MACCS keys for a list of SMILES."""
    n_bits = 167
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
        raise ValueError('No valid molecules found.')
    probs = bit_counts / valid_mols
    entropy_per_bit = np.array([
        -p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0
        for p in probs
    ])
    return float(np.mean(entropy_per_bit))


def molecule_unique_for_protein_api(protein: str, molecule: str) -> bool:
    """Check if a molecule has been previously submitted for the same target protein via API."""
    api_key = os.environ.get('VALIDATOR_API_KEY')
    if not api_key:
        raise ValueError('validator_api_key environment variable not set.')
    url = f"https://dashboard-backend-multitarget.up.railway.app/api/molecule_seen/{molecule}/{protein}"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            bt.logging.error(f"Failed to check molecule uniqueness: {response.status_code} {response.text}")
            return True
        data = response.json()
        return not data.get('seen', False)
    except Exception as e:
        bt.logging.error(f"Error checking molecule uniqueness: {e}")
        return True


def molecule_unique_for_protein_hf(protein: str, smiles: str) -> bool:
    """Check if molecule exists in Hugging Face Submission-Archive dataset by InChIKey."""
    if not hasattr(molecule_unique_for_protein_hf, '_CACHE'):
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
            url = hf_hub_url(repo_id='Metanova/Submission-Archive', filename=filename, repo_type='dataset')
            metadata = get_hf_file_metadata(url)
            current_sha = metadata.commit_hash
            last_check_time = current_time
            if cached_sha != current_sha:
                file_path = hf_hub_download(
                    repo_id='Metanova/Submission-Archive',
                    filename=filename,
                    repo_type='dataset',
                    revision=current_sha
                )
                df = pd.read_csv(file_path, usecols=['InChI_Key'])
                inchikeys_set = set(df['InChI_Key'])
                bt.logging.debug(
                    f"Loaded {len(inchikeys_set)} InChI Keys into lookup set for {protein} (commit {current_sha[:7]})"
                )
                molecule_unique_for_protein_hf._CACHE = (protein, current_sha, inchikeys_set, last_check_time)
            else:
                molecule_unique_for_protein_hf._CACHE = molecule_unique_for_protein_hf._CACHE[:3] + (last_check_time,)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            bt.logging.warning(f"Could not parse SMILES string: {smiles}")
            return True
        inchikey = Chem.MolToInchiKey(mol)
        return inchikey not in inchikeys_set
    except EntryNotFoundError:
        inchikeys_set = set()
        molecule_unique_for_protein_hf._CACHE = (protein, 'not_found', inchikeys_set, time.time())
        bt.logging.debug(f"File {filename} not found on HF, caching empty result")
        return True
    except Exception as e:
        bt.logging.warning(f"Error checking molecule in HF dataset: {e}")
        return True


def find_chemically_identical(smiles_list: list[str]) -> dict:
    """Check for identical molecules in a list of SMILES strings by converting to InChIKeys."""
    inchikey_to_indices = {}
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                inchikey = Chem.MolToInchiKey(mol)
                inchikey_to_indices.setdefault(inchikey, []).append(i)
        except Exception as e:
            bt.logging.warning(f"Error processing SMILES {smiles}: {e}")
    return {k: v for k, v in inchikey_to_indices.items() if len(v) > 1}


def get_total_reactions() -> int:
    """Query database for total number of reactions, add 1 for savi option."""
    try:
        db_path = os.path.join(os.path.dirname(__file__), '..', 'combinatorial_db/molecules.sqlite')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM reactions')
        count = cursor.fetchone()[0]
        conn.close()
        return count + 1
    except Exception as e:
        bt.logging.warning(f"Could not query reaction count: {e}, defaulting to 4")
        return 4


def is_reaction_allowed(molecule: str, allowed_reaction: str | None = None) -> bool:
    """Check if molecule matches the allowed reaction for this epoch."""
    if allowed_reaction is None:
        return True
    if not molecule:
        return False
    if molecule.startswith('rxn:'):
        try:
            parts = molecule.split(':')
            if len(parts) >= 2:
                rxn_id = int(parts[1])
                return allowed_reaction == f'rxn:{rxn_id}'
            return False
        except Exception as e:
            bt.logging.warning(f"Error parsing reaction molecule '{molecule}': {e}")
            return False
    return allowed_reaction == 'savi'

