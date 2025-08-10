import random
import math
import bittensor as bt
from datasets import load_dataset

from .molecules import get_total_reactions


def get_challenge_params_from_blockhash(block_hash: str, weekly_target: str, num_antitargets: int, include_reaction: bool = False) -> dict:
    """Use block hash as a seed to pick target and antitarget proteins and optionally an allowed reaction."""
    if not (isinstance(block_hash, str) and block_hash.startswith("0x")):
        raise ValueError("block_hash must start with '0x'.")
    if not weekly_target or num_antitargets < 0:
        raise ValueError("weekly_target must exist and num_antitargets must be non-negative.")

    try:
        seed = int(block_hash[2:], 16)
    except ValueError as e:
        raise ValueError(f"Invalid hex in block_hash: {block_hash}") from e

    rng = random.Random(seed)
    try:
        dataset = load_dataset("Metanova/Proteins", split="train")
    except Exception as e:
        raise RuntimeError("Could not load the 'Metanova/Proteins' dataset.") from e

    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Dataset is empty; cannot pick random entries.")

    unique_indices = rng.sample(range(dataset_size), k=num_antitargets)
    antitarget_indices = unique_indices[:num_antitargets]

    targets = [weekly_target]
    antitargets = [dataset[i]["Entry"] for i in antitarget_indices]

    result = {
        "targets": targets,
        "antitargets": antitargets,
    }

    if include_reaction:
        try:
            total_reactions = get_total_reactions()
            allowed_option = seed % total_reactions
            if allowed_option == 0:
                result["allowed_reaction"] = "savi"
            else:
                result["allowed_reaction"] = f"rxn:{allowed_option}"
        except Exception as e:
            bt.logging.warning(f"Failed to determine allowed reaction: {e}, defaulting to all reactions allowed")

    return result


def get_challenge_proteins_from_blockhash(block_hash: str, weekly_target: str, num_antitargets: int, include_reaction: bool = False) -> dict:
    """Backward-compatible wrapper for get_challenge_params_from_blockhash."""
    return get_challenge_params_from_blockhash(block_hash, weekly_target, num_antitargets, include_reaction)


def calculate_dynamic_entropy(starting_weight: float, step_size: float, start_epoch: int, current_epoch: int) -> float:
    """Calculate entropy weight based on epochs elapsed since start epoch."""
    epochs_elapsed = current_epoch - start_epoch
    entropy_weight = starting_weight + (epochs_elapsed * step_size)
    entropy_weight = max(0, entropy_weight)
    bt.logging.info(f"Epochs elapsed: {epochs_elapsed}, entropy weight: {entropy_weight}")
    return entropy_weight

