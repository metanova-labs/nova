import random
import requests
import bittensor as bt
from datasets import load_dataset


def get_sequence_from_protein_code(protein_code: str) -> str:
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


def get_challenge_params_from_blockhash(block_hash: str, weekly_target: str, num_antitargets: int, include_reaction: bool = False) -> dict:
    """
    Use block_hash as a seed to pick 'num_targets' and 'num_antitargets' random entries
    from the 'Metanova/Proteins' dataset. Optionally also pick allowed reaction.
    Returns {'targets': [...], 'antitargets': [...], 'allowed_reaction': '...'}.
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

    result = {
        "targets": targets,
        "antitargets": antitargets
    }

    if include_reaction:
        try:
            from .reactions import get_total_reactions
            total_reactions = get_total_reactions()
            allowed_option = seed % total_reactions
            if allowed_option == 0:
                result["allowed_reaction"] = "savi"
            else:
                result["allowed_reaction"] = f"rxn:{allowed_option}"
        except Exception as e:
            bt.logging.warning(f"Failed to determine allowed reaction: {e}, defaulting to all reactions allowed")

    return result
