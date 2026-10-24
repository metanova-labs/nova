import atexit
import asyncio
import hashlib
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
import math

NOVA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(NOVA_DIR, "NOVA-nanobody-filter"))


import bittensor as bt
from abnumber import Chain
from metanano.config import SearchConfig
from metanano.search import IndexManager, SearchEngine
from metanano.utils.alignment import AlignmentEngine
from metanano.utils.kmer import generate_kmers

from utils.constants import ALLOWED_AAS, HYDROPHOBIC
from utils.minmax_weighted_rank import rank_binders
from utils.challenge import get_historical_submissions

_CDR_BATCH_SIZE = 512
_MAX_CDR_WORKERS = 8

CDRs = Dict[str, str]
Regions = Dict[str, str]

def normalize_seq(seq: str) -> str:
    return seq.strip().upper()

def seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode("ascii")).hexdigest()

def max_run_length(s: str) -> int:
    """maximum length of a homopolymer run in a sequence"""
    if not s:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best

def max_di_repeat_pairs(s: str) -> int:
    """
    Detect strongest ABABAB... repeats.
    Returns the max number of AB pairs in a contiguous ABAB... region.
    Example: "GSGSGS" -> "GS" repeated 3 times => 3 pairs.
    """
    best = 0
    n = len(s)

    for i in range(n - 3):
        a, b = s[i], s[i + 1]
        if a == b:
            continue  # skip AA repeats; handled by homopolymer
        pairs = 1
        j = i + 2
        while j + 1 < n and s[j] == a and s[j + 1] == b:
            pairs += 1
            j += 2
        if pairs > best:
            best = pairs
    return best

def has_plausible_cys_pair(seq: str, min_sep: int, max_sep: int) -> bool:
    """True if there is a Cys pair with separation within [min_sep, max_sep]."""
    cys_positions = [i for i, aa in enumerate(seq) if aa == "C"]
    
    for i in range(len(cys_positions) - 1):
        for j in range(i + 1, len(cys_positions)):
            sep = cys_positions[j] - cys_positions[i]
            if sep < min_sep:
                continue
            if sep > max_sep:
                break
            return True
    return False

def looks_like_signal_peptide(seq: str, window: int, hydro_min: int, scan_prefix: int) -> bool:
    """
    simple signal peptide heuristic:
    - Scan first scan_prefix aa
    - If any window of length `window` has >= `hydro_min` hydrophobic residues, reject.
    """
    prefix = seq[: min(len(seq), scan_prefix)]
    if len(prefix) < window:
        return False

    # sliding window count of hydrophobic residues
    count = sum(1 for aa in prefix[:window] if aa in HYDROPHOBIC)
    if count >= hydro_min:
        return True

    for i in range(window, len(prefix)):
        if prefix[i - window] in HYDROPHOBIC:
            count -= 1
        if prefix[i] in HYDROPHOBIC:
            count += 1
        if count >= hydro_min:
            return True

    return False

async def analyze_developability(seqs: List[str]) -> bool:
    from metanano.services.async_manager import AsyncServiceManager
    from metanano.services.developability_service import DevelopabilityService
    from utils.aggrescan3d import developability_config
    config = developability_config()

    # TNP concurrency. DevelopabilityService otherwise falls back to the global
    # get_service_manager(), which builds a bare AsyncConfig
    config.async_config.max_concurrent_tnp = max(1, min(16, int(
        os.environ.get("NOVA_MAX_CONCURRENT_TNP", "16"))))
    manager = AsyncServiceManager(config.async_config)

    developability_service = DevelopabilityService(config.developability, manager)
    result = await developability_service.analyze_batch_async(seqs)
    #bt.logging.info(f"Developability analysis result: {result}")
    return result

def index_top_sequences(target: str, n: int | None = None) -> SearchEngine:
    search_config = SearchConfig()
    index_manager = IndexManager(search_config)
    alignment_engine = AlignmentEngine(search_config.fine_alignment)
    search_engine = SearchEngine(search_config, index_manager, alignment_engine)

    # read HF dataset for target
    try:
        historical_submissions = get_historical_submissions(target, "nanobodies")
        if n is None:
            n = len(historical_submissions)
            top_sequences = historical_submissions[['sequence', 'sequence_hash']]
        else:
            top_sequences = rank_binders(historical_submissions, k=n)
            top_sequences = top_sequences.head(n)[['sequence', 'sequence_hash']]

        records = list(top_sequences.itertuples(index=False, name=None))
        cdrs_by_id = _extract_cdrs_for_records(records)

        for seq, seq_id in records:
            kmers = generate_kmers(seq, k=search_config.k)
            index_manager.add_sequence(seq_id, seq, cdrs_by_id.get(seq_id), kmers)

        return search_engine

    except Exception as e:
        bt.logging.warning(f"Could not get historical submissions for target {target}: {e}")
        return None

def is_duplicate(match):
    """
    Decide whether a search match represents a duplicate submission.
    Uses CDR-focused logic rather than whole-sequence identity alone.
    from: NOVA-nanobody-filter/blob/main/metanano/search/search_engine.py#L134
    """
    identity = match.identity
    cdr_sim = match.cdr_similarity  # may be None

    # Tier 1: Whole-sequence near-identity — definitely a duplicate
    if identity >= 0.95:
        bt.logging.debug(f"match is_duplicate on step 1: high whole-sequence identity {identity:.4%}")
        return True, "near-identical sequence"

    # Tier 2: CDR3-focused check (if CDRs available)
    if cdr_sim is not None:
        cdr3 = cdr_sim.get("CDR3", 0.0)

        # CDR3 identity >= 0.85 = same clonotype, likely same epitope
        if cdr3 >= 0.85:
            bt.logging.debug(f"match is_duplicate on step 2.1: high CDR3 identity {cdr3:.4%}")
            return True, f"CDR3 identity {cdr3:.4%} (same clonotype)"

        # CDR3 >= 0.80 with conserved CDR1+CDR2 = functional duplicate
        cdr1 = cdr_sim.get("CDR1", 0.0)
        cdr2 = cdr_sim.get("CDR2", 0.0)
        if cdr3 >= 0.80 and cdr1 >= 0.90 and cdr2 >= 0.90:
            bt.logging.debug(f"match is_duplicate on step 2.2: high CDR3 identity {cdr3:.4%} with conserved CDR1/CDR2")
            return True, f"CDR3={cdr3:.4%} with conserved CDR1/CDR2"

    # Tier 3: High whole-sequence identity without CDR data
    if identity >= 0.90 and cdr_sim is None:
        bt.logging.debug(f"match is_duplicate on step 3: high whole-sequence identity {identity:.4%} with no CDR data")
        return True, f"whole-sequence identity {identity:.4%} (no CDR data)"

    return False, "novel"

def classify_chain(chain, enforce_hallmarks: bool = True):
    if chain.chain_type != "H":
        return "not VHH"

    if chain.scheme != "imgt":
        bt.logging.warning(f"Chain {chain.name} is not IMGT numbered")
        return "not VHH"

    if enforce_hallmarks:
        try:
            pos49 = chain["H49"]
            pos50 = chain["H50"]
        except KeyError as exc:
            bt.logging.warning(f"Chain {chain.name} is missing required IMGT position 49 or 50")
            return "not VHH"
        
        return "VHH" if pos49 in {"E", "Q"} and pos50 == "R" else "not VHH"
    else:
        return "VHH"


def _stop_worker_log_listener() -> None:
    """Stop Bittensor's queue listener before a spawned worker exits."""
    listener = getattr(bt.logging, "_listener", None)
    if listener is None or getattr(listener, "_thread", None) is None:
        return
    try:
        atexit.unregister(listener.stop)
        listener.stop()
    except Exception:
        pass


def _extract_cdrs_worker(
    sequences: Dict[str, str],
) -> Dict[str, Optional[Dict[str, str]]]:
    try:
        return extract_cdrs_batch(sequences)
    finally:
        _stop_worker_log_listener()


def _extract_cdrs_for_records(
    records: List[tuple[str, str]],
) -> Dict[str, Optional[Dict[str, str]]]:
    """Extract CDRs in process-parallel AbNumber batches.

    Each record is ``(sequence, sequence_id)``. Only the small CDR result maps
    cross the process boundary; the search index is built in the parent.
    """
    if not records:
        return {}

    cpu_workers = max(1, (os.cpu_count() or 1) // 2)
    useful_workers = max(1, math.ceil(len(records) / _CDR_BATCH_SIZE))
    max_workers = min(_MAX_CDR_WORKERS, cpu_workers, useful_workers)
    if max_workers == 1:
        return extract_cdrs_batch(
            {seq_id: sequence for sequence, seq_id in records}
        )

    batch_size = math.ceil(len(records) / max_workers)
    batches = [
        {
            seq_id: sequence
            for sequence, seq_id in records[offset : offset + batch_size]
        }
        for offset in range(0, len(records), batch_size)
    ]
    context = mp.get_context("spawn")

    cdrs_by_id: Dict[str, Optional[Dict[str, str]]] = {}
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
    ) as executor:
        for batch_result in executor.map(_extract_cdrs_worker, batches):
            cdrs_by_id.update(batch_result)
    return cdrs_by_id

def _chain_from_sequence(seq_id: str, sequence: str):
    try:
        return Chain(sequence, scheme="imgt", cdr_definition="imgt", use_anarcii=False)
    except Exception as e:
        bt.logging.warning(f"Error extracting chain for {seq_id}: {e}")
        return None

def _chain_to_cdrs(chain: Any) -> CDRs:
    """Convert an AbNumber Chain into the CDR representation used by MetaNano."""
    return {
        "cdr1": str(chain.cdr1_seq) if chain.cdr1_seq else "",
        "cdr2": str(chain.cdr2_seq) if chain.cdr2_seq else "",
        "cdr3": str(chain.cdr3_seq) if chain.cdr3_seq else "",
    }

def extract_chains_batch(sequence_dict: Dict[str, str]):
    chains = {}
    errors = {}
    try:
        chains, errors = Chain.batch(sequence_dict, scheme="imgt", cdr_definition="imgt", use_anarcii=False)
    except Exception as e:
        bt.logging.warning(f"Batch chain extraction failed, falling back to sequential extraction: {e}")
        errors = {seq_id: str(e) for seq_id in sequence_dict}

    for seq_id in errors:
        chain = _chain_from_sequence(seq_id, sequence_dict[seq_id])
        if chain is not None:
            chains[seq_id] = chain
        else:
            bt.logging.warning(f"Failed to extract chain for {seq_id}: {errors[seq_id]}")

    expected_counts: Dict[int, int] = {}
    for seq_id in sequence_dict:
        uid = int(seq_id.split("-")[0])
        expected_counts[uid] = expected_counts.get(uid, 0) + 1

    # revert back to uid: list of chains format for nanobody_validity.py
    chains_dict: Dict[int, list] = {}
    for seq_id, chain in chains.items():
        uid = int(seq_id.split("-")[0])
        chains_dict.setdefault(uid, []).append(chain)
    for uid, expected in expected_counts.items():
        if len(chains_dict.get(uid, [])) != expected:
            bt.logging.warning(f"UID {uid}: chain extraction incomplete, rejecting")
            chains_dict.pop(uid, None)

    return chains_dict


def extract_regions_from_chains(chains: list[Chain]) -> list[dict[str, str]]:
    all_regions = []
    for chain in chains:
        regions = {
            "cdr1": str(chain.cdr1_seq) if chain.cdr1_seq else "",
            "cdr2": str(chain.cdr2_seq) if chain.cdr2_seq else "",
            "cdr3": str(chain.cdr3_seq) if chain.cdr3_seq else "",
            "fr1": str(chain.fr1_seq) if chain.fr1_seq else "",
            "fr2": str(chain.fr2_seq) if chain.fr2_seq else "",
            "fr3": str(chain.fr3_seq) if chain.fr3_seq else "",
        }
        all_regions.append(regions)
    return all_regions

def extract_cdrs_batch(
    sequences: Dict[str, str],
) -> Dict[str, Optional[CDRs]]:
    """
    Extract CDR regions for multiple sequences in one AbNumber invocation.

    Args:
        sequences: Mapping from stable sequence IDs to amino-acid sequences.
        return_FR: Whether to return the FR regions.
            If True, the FR regions will be returned in the CDRs dictionary.
    Returns:
        A result for every input ID. Sequences that AbNumber cannot parse map
        to None, matching the failure behavior of :func:`extract_cdrs`.

    Notes:
        This function is module-level so it can be submitted directly to a
        multiprocessing executor.
    """
    if not sequences:
        return {}

    try:
        from abnumber import Chain

        chains, errors = Chain.batch(sequences, scheme="imgt")
    except Exception as e:
        bt.logging.warning(
            f"Batch CDR extraction failed for {len(sequences)} sequences: {e}"
        )
        return {seq_id: None for seq_id in sequences}

    results: Dict[str, Optional[CDRs]] = {}
    for seq_id in sequences:
        chain = chains.get(seq_id)
        results[seq_id] = _chain_to_cdrs(chain) if chain is not None else None

    if errors:
        bt.logging.warning(
            f"Could not extract CDRs for {len(errors)} of "
            f"{len(sequences)} sequences"
        )

    return results