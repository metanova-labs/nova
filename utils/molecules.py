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
    else:
        return None


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
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        bt.logging.warning(f"Could not parse SMILES string: {smiles}, returning 0")
        return 0
    return mol.GetNumHeavyAtoms()


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
        bt.logging.warning("No valid molecules found.")
        return None

    probs = bit_counts / valid_mols
    entropy_per_bit = np.array([
        -p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0
        for p in probs
    ])

    avg_entropy = np.mean(entropy_per_bit)

    return avg_entropy


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
