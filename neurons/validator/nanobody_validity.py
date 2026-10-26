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
    aggrescan_rejection,
    index_top_sequences,
    extract_chains_batch,
    is_duplicate,
    classify_chain,
    extract_regions_from_chains,
    NOVA_DIR,
    human_germline_identity_anarci,
)
from utils.stage_timer import stage

ALLOWED_AAS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWV")

async def validate_nanobodies(
    uid_to_data: dict[int, dict[str, list]],
    score_dict: dict[int, dict[str, list[list[float]]]],
    config: dict,
    allowed_v: list[str] = None,
    allowed_cdr3_length: list[int] = None,
) -> dict[int, dict[str, list[str]]]:
    """
    Validate nanobodies and return a dictionary of valid nanobodies by uid
    """

    pre_validated = {}

    # index top sequences for all targets
    search_engines = {}
    for target in config["nanobody_target"]:
        search_engines[target] = index_top_sequences(target, config["num_top_sequences"])

    for uid, data in uid_to_data.items():

        normalized_sequences = [normalize_seq(seq) for seq in data["sequences"]]
        submission_hashes = [seq_hash(seq) for seq in normalized_sequences]


        # Tier 1: basic sequence validity checks
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

        # signal peptide heuristic
        if any(looks_like_signal_peptide(seq, config["sp_window"], config["sp_hydro_min_in_window"], config["sp_scan_prefix"]) for seq in normalized_sequences):
            bt.logging.warning(f"UID {uid}: contains signal peptide-like sequences")
            continue

        # check for exact duplicates in all previous submissions.
        # NOVA_SKIP_HISTORICAL_CHECKS: benchmark-only bypass for replaying an already-archived epoch
        skip_historical = os.environ.get("NOVA_SKIP_HISTORICAL_CHECKS") == "1" # do not set it in live operation
        is_unique = True
        for target in ([] if skip_historical else config["nanobody_target"]):
            if any(not entry_unique_for_protein_hf(target, h, 'nanobodies') for h in submission_hashes):
                bt.logging.warning(f"UID {uid}: contains sequences that are not unique for target {target}")
                is_unique = False
                break
        
        if not is_unique:
            continue

        # region-aware similarity to historical sequences
        uid_invalid = False
        similarity_results = None
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

                if not skip_historical and any(
                    is_duplicate(m)[0] for result in similarity_results for m in result.matches
                ):
                    bt.logging.warning(f"UID {uid}: contains sequences too similar to a top sequence for this target")
                    uid_invalid = True
                    break

        if uid_invalid:
            continue

        # Tier 2:human germline identity scoring
        germline_results = []
        for seq in normalized_sequences:
            try:
                germline_result = human_germline_identity_anarci(seq)
                bt.logging.debug(f"UID {uid}: germline result: {germline_result}")
                germline_results.append(germline_result)
            except ValueError as e1:
                bt.logging.warning(f"UID {uid}: {e1}")
                uid_invalid = True
                break
            except Exception as e2:
                bt.logging.warning(f"UID {uid}: error scoring germline identity: {e2}")
                uid_invalid = True
                break

        if uid_invalid:
            continue

        if allowed_v:
            passes_v_filter = any(allowed_v in germline_result['V']['genes'] for germline_result in germline_results)
            if not passes_v_filter:
                bt.logging.warning(f"UID {uid}: contains sequences that are not in the allowed V neighborhood ({allowed_v})")
                continue

        if any(germline_result['V']['identity_pct'] < config["min_human_v_identity"] or germline_result['J']['identity_pct'] < config["min_human_j_identity"] for germline_result in germline_results):
            bt.logging.warning(f"UID {uid}: contains sequences that are do not meet human germline identity thresholds")
            continue

        pre_validated[uid] = {
            "sequences": normalized_sequences,
            "hashes": submission_hashes,
            "similarity_results": similarity_results if similarity_results else [],
            "germline_results": germline_results if germline_results else [],
        }

    # Tier 3: validity checks from AbNumber Chain object
    sequences_flat = {f"{uid}-{i}": seq for uid in pre_validated.keys() for i, seq in enumerate(pre_validated[uid]["sequences"])}
    chains = extract_chains_batch(sequences_flat)

    post_chain_validity_checks = {}
    for uid, uid_chains in chains.items():
        # check that all chains are VHH
        all_vhh = all(classify_chain(c, enforce_hallmarks=config["enforce_vhh_hallmarks"]) == "VHH" for c in uid_chains)
        if not all_vhh:
            bt.logging.warning(f"UID {uid}: contains sequences that are not VHH")
            continue

        # check canonical cysteines
        try:
            has_canonical_cysteines = all(c["H23"] == "C" and c["H104"] == "C" for c in uid_chains)
        except KeyError as e:
            bt.logging.warning(f"UID {uid} contains missing IMGT positions: {e}")
            continue
        if not has_canonical_cysteines:
            bt.logging.warning(f"UID {uid}: contains sequences that do not contain the canonical cysteines")
            continue

        # check allowed CDR3 length for this round
        if allowed_cdr3_length:
            cdr3_lengths = [len(c.cdr3_seq) for c in uid_chains]
            print(f"UID {uid}: cdr3_lengths: {cdr3_lengths}")
            if any(cdr3_length != allowed_cdr3_length for cdr3_length in cdr3_lengths):
                bt.logging.warning(f"UID {uid}: contains sequences that do not have the allowed CDR3 length for this round ({allowed_cdr3_length})")
                continue

        regions = extract_regions_from_chains(uid_chains)
        post_chain_validity_checks[uid] = {
            **pre_validated[uid],
            "regions": regions,
            "chains": uid_chains,
        }

    # Tier 4: batch developability check across all surviving UIDs
    if not post_chain_validity_checks:
        return {}

    all_dev_sequences = []
    uid_seq_counts = []
    for uid, info in post_chain_validity_checks.items():
        seqs = info["sequences"]
        uid_seq_counts.append((uid, len(seqs)))
        all_dev_sequences.extend(seqs)

    try:
        with stage("developability_tnp", n=len(all_dev_sequences)):
            all_dev_results = await analyze_developability(all_dev_sequences)
    except Exception as e:
        bt.logging.warning(f"Batch developability analysis failed: {e}")
        return {}

    valid_nanobodies_by_uid = {}
    offset = 0
    for uid, count in uid_seq_counts:
        info = post_chain_validity_checks[uid]
        developability_result = all_dev_results[offset : offset + count]
        offset += count

        bt.logging.debug(f"UID {uid}: developability results: {developability_result}")
        rejected_sequences = [
            seq for seq, result in zip(info["sequences"], developability_result)
            if result.get("red_flags") or result.get("error")
        ]
        if rejected_sequences:
            bt.logging.warning(f"UID {uid}: contains sequences that are not developable: {rejected_sequences}")
            continue

        psh_strict_rejection = [(seq, result["surface_hydrophobic_patches"]) for seq, result in zip(info["sequences"], developability_result) if result["surface_hydrophobic_patches"] > config["psh_strict_rejection_threshold"]]
        if psh_strict_rejection:
            bt.logging.warning(
                f"UID {uid}: contains sequences that are rejected by PSH strict rejection: {psh_strict_rejection}"
            )
            continue

        aggrescan_rejections = [
            (seq, reason)
            for seq, result in zip(info["sequences"], developability_result)
            if (reason := aggrescan_rejection(result))
        ]
        if aggrescan_rejections:
            bt.logging.warning(
                f"UID {uid}: contains sequences rejected by Aggrescan3D: {aggrescan_rejections}"
            )
            continue

        valid_nanobodies_by_uid[uid] = {
            **info,
            "developability_result": developability_result,
        }

    return valid_nanobodies_by_uid