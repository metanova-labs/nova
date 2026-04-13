import os
import bittensor as bt
from utils import (
    entry_unique_for_protein_hf,
    normalize_seq,
    seq_hash,
    max_run_length,
    max_di_repeat_pairs,
    has_plausible_cys_pair,
    looks_like_signal_peptide,
    analyze_developability,
    compute_igblast_nativeness,
    index_top_sequences,
    is_duplicate,
    NOVA_DIR,
)

ALLOWED_AAS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWV")

async def validate_nanobodies(
    uid_to_data: dict[int, dict[str, list]],
    score_dict: dict[int, dict[str, list[list[float]]]],
    config: dict,
) -> dict[int, dict[str, list[str]]]:
    """
    Validate nanobodies and return a dictionary of valid nanobodies by uid
    """
    
    pre_validated = {}

    # index top sequences for all targets
    search_engines = {}
    for target in config["nanobody_target"]:
        search_engines[target] = index_top_sequences(target)

    for uid, data in uid_to_data.items():

        normalized_sequences = [normalize_seq(seq) for seq in data["sequences"]]
        submission_hashes = [seq_hash(seq) for seq in normalized_sequences]

        if any("~" in seq for seq in normalized_sequences):
            continue

        if len(normalized_sequences) > config["num_sequences"]:
            bt.logging.warning(f"UID {uid}: submission contains {len(normalized_sequences)} sequences, considering only first {config['num_sequences']}")
            normalized_sequences = normalized_sequences[:config["num_sequences"]]
            submission_hashes = submission_hashes[:config["num_sequences"]]
        elif len(normalized_sequences) < config["num_sequences"]:
            bt.logging.warning(f"UID {uid}: submission contains {len(normalized_sequences)} sequences, expected {config['num_sequences']}, skipping")
            continue

        # check for duplicate sequences
        if len(submission_hashes) != len(set(submission_hashes)):
            bt.logging.warning(f"UID {uid}: submission contains duplicate sequences")
            continue

        # check if sequence is in the valid length range
        if any(len(seq) < config["min_sequence_length"] or len(seq) > config["max_sequence_length"] for seq in normalized_sequences):
            bt.logging.warning(f"UID {uid}: submission contains sequences outside the valid length range: {normalized_sequences}")
            continue

        # check if sequence contains only valid aas
        if any(set(seq) - ALLOWED_AAS for seq in normalized_sequences):
            bt.logging.warning(f"UID {uid}: submission contains sequences with invalid amino acids")
            continue

        # check for homopolymer runs
        if any(max_run_length(seq) > config["max_homopolymer_run"] for seq in normalized_sequences):
            bt.logging.warning(f"UID {uid}: contains not allowed homopolymer runs")
            continue

        # check for di-repeat pairs
        if any(max_di_repeat_pairs(seq) > config["max_di_repeat_pairs"] for seq in normalized_sequences):
            bt.logging.warning(f"UID {uid}: contains not allowed di-repeat pairs")
            continue

        # cysteine count and pair separation
        cys_counts = [seq.count("C") for seq in normalized_sequences]
        if any(cys_count < config["min_cysteines"] for cys_count in cys_counts):
            bt.logging.warning(f"UID {uid}: contains sequences with too few cysteines")
            continue

        if config["min_cysteines"] > 1:
            if not any(has_plausible_cys_pair(seq, config["cys_pair_min_separation"], config["cys_pair_max_separation"]) for seq in normalized_sequences):
                bt.logging.warning(f"UID {uid}: contains sequences with no plausible cysteine pairs")
                continue

        # signal peptide heuristic
        if any(looks_like_signal_peptide(seq, config["sp_window"], config["sp_hydro_min_in_window"], config["sp_scan_prefix"]) for seq in normalized_sequences):
            bt.logging.warning(f"UID {uid}: contains signal peptide-like sequences")
            continue

        # check for exact duplicates in all previous submissions
        is_unique = True
        for target in config["nanobody_target"]:
            if any(not entry_unique_for_protein_hf(target, h, 'nanobodies') for h in submission_hashes):
                bt.logging.warning(f"UID {uid}: contains sequences that are not unique for target {target}")
                is_unique = False
                break
        
        if not is_unique:
            continue

        # similarity to top sequences search
        uid_invalid = False
        for target in config["nanobody_target"]:
            search_engine = search_engines[target]
            similarity_results = []
            if search_engine is not None:
                similarity_check_failed = False

                for seq in normalized_sequences:
                    try:
                        similarity_result = search_engine.search(
                            seq,
                            include_alignment=True,
                            exclude_ids=None,
                            coarse_min_shared=None,
                            coarse_jaccard=None,
                        )
                        similarity_results.append(similarity_result)
                    except Exception as e:
                        bt.logging.warning(f"UID {uid}: error searching for similarity: {e}")
                        similarity_check_failed = True
                        break

                if similarity_check_failed:
                    bt.logging.warning(f"UID {uid}: contains sequences that failed the similarity check")
                    uid_invalid = True
                    break

                if any(is_duplicate(m) for result in similarity_results for m in result.matches):
                    bt.logging.warning(f"UID {uid}: contains sequences too similar to a top sequence for this target")
                    uid_invalid = True
                    break

        if uid_invalid:
            continue

        pre_validated[uid] = {
            "sequences": normalized_sequences,
            "hashes": submission_hashes,
            "similarity_results": similarity_results if similarity_results else [],
        }

    # --- batch nativeness check across all surviving UIDs ---
    if not pre_validated:
        return {}

    all_nativeness_seqs = {}
    for uid, info in pre_validated.items():
        for h, seq in zip(info["hashes"], info["sequences"]):
            all_nativeness_seqs[h] = seq

    results_by_id = {}
    try:
        all_nativeness_results = compute_igblast_nativeness(all_nativeness_seqs)
        results_by_id = {r.sequence_id: r for r in all_nativeness_results}
    except Exception as e:
        bt.logging.warning(f"Batch IgBLAST failed ({e}), falling back to per-UID")
        for uid, info in pre_validated.items():
            try:
                uid_results = compute_igblast_nativeness(
                    {h: seq for h, seq in zip(info["hashes"], info["sequences"])}
                )
                for r in uid_results:
                    results_by_id[r.sequence_id] = r
            except Exception as e2:
                bt.logging.warning(f"UID {uid}: IgBLAST failed ({e2}), rejecting")

    post_nativeness = {}
    for uid, info in pre_validated.items():
        nativeness_result = [results_by_id.get(h) for h in info["hashes"]]
        if any(r is None for r in nativeness_result):
            bt.logging.warning(f"UID {uid}: missing IgBLAST results, rejecting")
            continue

        bt.logging.debug(f"UID {uid}: nativeness/humanness results: {nativeness_result}")

        if any(r.vhh_nativeness < config["min_nativeness_score"] for r in nativeness_result):
            low = [seq for seq, r in zip(info["sequences"], nativeness_result) if r.vhh_nativeness < config["min_nativeness_score"]]
            bt.logging.warning(f"UID {uid}: contains sequences that are low in nativeness score: {low}")
            continue
        if any(r.human_framework < config["min_human_framework_score"] for r in nativeness_result):
            low = [seq for seq, r in zip(info["sequences"], nativeness_result) if r.human_framework < config["min_human_framework_score"]]
            bt.logging.warning(f"UID {uid}: contains sequences that are low in human framework score: {low}")
            continue

        post_nativeness[uid] = {
            **info,
            "nativeness_result": nativeness_result,
        }

    # --- batch developability check across all nativeness-passing UIDs ---
    if not post_nativeness:
        return {}

    all_dev_sequences = []
    uid_seq_counts = []
    for uid, info in post_nativeness.items():
        seqs = info["sequences"]
        uid_seq_counts.append((uid, len(seqs)))
        all_dev_sequences.extend(seqs)

    try:
        all_dev_results = await analyze_developability(all_dev_sequences)
    except Exception as e:
        bt.logging.warning(f"Batch developability analysis failed: {e}")
        return {}

    valid_nanobodies_by_uid = {}
    offset = 0
    for uid, count in uid_seq_counts:
        info = post_nativeness[uid]
        developability_result = all_dev_results[offset : offset + count]
        offset += count

        bt.logging.debug(f"UID {uid}: developability results: {developability_result}")
        rejected_sequences = [seq for seq, result in zip(info["sequences"], developability_result) if not result["passed"]]
        if rejected_sequences:
            bt.logging.warning(f"UID {uid}: contains sequences that are not developable: {rejected_sequences}")
            continue

        valid_nanobodies_by_uid[uid] = {
            "sequences": info["sequences"],
            "hashes": info["hashes"],
            "developability_result": developability_result,
            "nativeness_result": info["nativeness_result"],
            "similarity_results": info["similarity_results"],
        }

    return valid_nanobodies_by_uid