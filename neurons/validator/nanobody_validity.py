import bittensor as bt
from utils import (
    entry_unique_for_protein_hf,
    normalize_seq,
    seq_hash,
    max_run_length,
    max_di_repeat_pairs,
    has_plausible_cys_pair,
    looks_like_signal_peptide
)

ALLOWED_AAS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWV")

def validate_nanobodies(
    uid_to_data: dict[int, dict[str, list]],
    score_dict: dict[int, dict[str, list[list[float]]]],
    config: dict,
) -> dict[int, dict[str, list[str]]]:
    """
    Validate nanobodies and return a dictionary of valid nanobodies by uid
    """
    
    valid_nanobodies_by_uid = {}

    for uid, data in uid_to_data.items():

        normalized_sequences = [normalize_seq(seq) for seq in data["sequences"]]
        submission_hashes = [seq_hash(seq) for seq in normalized_sequences]

        if any("~" in seq for seq in normalized_sequences):
            continue

        # check for duplicate sequences
        if len(submission_hashes) != len(set(submission_hashes)):
            bt.logging.warning(f"UID {uid} submission contains duplicate sequences")
            continue

        # check if sequence is in the valid length range
        if any(len(seq) < config["min_sequence_length"] or len(seq) > config["max_sequence_length"] for seq in normalized_sequences):
            bt.logging.warning(f"UID {uid} submission contains sequences outside the valid length range: {normalized_sequences}")
            continue

        # check if sequence contains only valid aas
        if any(set(seq) - ALLOWED_AAS for seq in normalized_sequences):
            bt.logging.warning(f"UID {uid} submission contains sequences with invalid amino acids")
            continue

        # check for homopolymer runs
        if any(max_run_length(seq) > config["max_homopolymer_run"] for seq in normalized_sequences):
            bt.logging.warning(f"UID {uid} contains not allowed homopolymer runs")
            continue

        # check for di-repeat pairs
        if any(max_di_repeat_pairs(seq) > config["max_di_repeat_pairs"] for seq in normalized_sequences):
            bt.logging.warning(f"UID {uid} contains not allowed di-repeat pairs")
            continue

        # cysteine count and pair separation
        cys_counts = [seq.count("C") for seq in normalized_sequences]
        if any(cys_count < config["min_cysteines"] for cys_count in cys_counts):
            bt.logging.warning(f"UID {uid} contains sequences with too few cysteines")
            continue

        if config["min_cysteines"] > 1:
            if not any(has_plausible_cys_pair(seq, config["cys_pair_min_separation"], config["cys_pair_max_separation"]) for seq in normalized_sequences):
                bt.logging.warning(f"UID {uid} contains sequences with no plausible cysteine pairs")
                continue

        # signal peptide heuristic
        if any(looks_like_signal_peptide(seq, config["sp_window"], config["sp_hydro_min_in_window"], config["sp_scan_prefix"]) for seq in normalized_sequences):
            bt.logging.warning(f"UID {uid} contains signal peptide-like sequences")
            continue

        # check if sequence is unique for all targets
        is_unique = True
        for target in config["nanobody_target"]:
            if any(not entry_unique_for_protein_hf(target, h, 'nanobodies') for h in submission_hashes):
                bt.logging.warning(f"UID {uid} contains sequences that are not unique for target {target}")
                is_unique = False
                break
        
        if not is_unique:
            continue

        valid_nanobodies_by_uid[uid] = {
            "sequences": normalized_sequences,
            "hashes": submission_hashes
        }

    return valid_nanobodies_by_uid
        
        