import hashlib
import asyncio
import os
import sys
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
import math

NOVA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(NOVA_DIR, "NOVA-nanobody-filter"))

import bittensor as bt
from metanano.config import Config, SearchConfig
from metanano.search import IndexManager, SearchEngine
from metanano.utils.alignment import AlignmentEngine
from metanano.utils.kmer import generate_kmers
from metanano.utils.cdr_utils import extract_cdrs
from metanano.utils.igblast_nativeness import features_to_cdrs

from utils.constants import ALLOWED_AAS, HYDROPHOBIC
from utils.minmax_weighted_rank import rank_binders


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
    from metanano.services.developability_service import DevelopabilityService
    config = Config()
    developability_service = DevelopabilityService(config.developability)
    result = await developability_service.analyze_batch_async(seqs)
    #bt.logging.info(f"Developability analysis result: {result}")
    return result

def shannon_entropy_aa(seq: str) -> float:
    # entropy in bits over AA alphabet present in seq (not normalized)
    if not seq:
        return 0.0
    counts: Dict[str, int] = {}
    n = 0
    for ch in seq:
        if ch in ALLOWED_AAS:
            counts[ch] = counts.get(ch, 0) + 1
            n += 1
    if n == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent

def compute_igblast_nativeness(sequences: Dict[str, str]) -> List[Dict[str, Any]]:
    '''
    wrapper for igblast_nativeness.run - supports batch processing
    '''
    from metanano.utils import igblast_nativeness
    result = igblast_nativeness.score_sequences(sequences)
    return result

def index_top_sequences(target: str, n: int = 50) -> SearchEngine:
    search_config = SearchConfig()
    index_manager = IndexManager(search_config)
    alignment_engine = AlignmentEngine(search_config.fine_alignment)
    search_engine = SearchEngine(search_config, index_manager, alignment_engine)

    # read HF dataset for target
    try:
        local_path = hf_hub_download(
            repo_id="Metanova/Submission-Archive",
            filename=f"{target}_nanobodies.csv",
            repo_type='dataset',
            token=os.getenv("HF_TOKEN"),
        )
        top_sequences = pd.read_csv(local_path)
        top_sequences = rank_binders(top_sequences, k=50, max_liability_violations=50)
        top_sequences = top_sequences.head(n)[['sequence', 'sequence_hash']]
    except EntryNotFoundError:
        return None
    except Exception as e:
        bt.logging.warning(
            f"Could not download existing {target}_nanobodies.csv from Metanova/Submission-Archive: {e}"
        )
        return None
        
    for seq, seq_id in top_sequences.values:
        kmers = generate_kmers(seq, k=search_config.k)
        cdrs = extract_cdrs(seq)
        #print(f"CDRs found by abnumber: {cdrs}")

        # fallback to igblast if cdrs are not found by abnumber
        if cdrs is None:
            bt.logging.warning(f"Failed to extract CDRs for sequence {seq_id} using abnumber, falling back to igblast")
            with NamedTemporaryFile(suffix=".fasta") as temp_file:
                with open(temp_file.name, "w") as f:
                    f.write(f">seq_{seq_id}\n{seq}")
                features = compute_igblast_nativeness(temp_file.name)
                #print(features)
            cdrs = features_to_cdrs(features[0]['features'])
            #print(f"CDRs found by igblast: {cdrs}")
            if cdrs is None:
                bt.logging.warning(f"Failed to extract CDRs for sequence {seq_id} using igblast")
                continue

        index_manager.add_sequence(seq_id, seq, cdrs, kmers)

    return search_engine

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
        return True, "near-identical sequence"

    # Tier 2: CDR3-focused check (if CDRs available)
    if cdr_sim is not None:
        cdr3 = cdr_sim.get("CDR3", 0.0)

        # CDR3 identity >= 0.85 = same clonotype, likely same epitope
        if cdr3 >= 0.85:
            return True, f"CDR3 identity {cdr3:.0%} (same clonotype)"

        # CDR3 >= 0.80 with conserved CDR1+CDR2 = functional duplicate
        cdr1 = cdr_sim.get("CDR1", 0.0)
        cdr2 = cdr_sim.get("CDR2", 0.0)
        if cdr3 >= 0.80 and cdr1 >= 0.90 and cdr2 >= 0.90:
            return True, f"CDR3={cdr3:.0%} with conserved CDR1/CDR2"

    # Tier 3: High whole-sequence identity without CDR data
    if identity >= 0.90 and cdr_sim is None:
        return True, f"whole-sequence identity {identity:.0%} (no CDR data)"

    return False, "novel"