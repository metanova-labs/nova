import hashlib

ALLOWED_AAS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWV")

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