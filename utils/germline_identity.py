"""Tie-aware V/J percent identity for single heavy-chain antibody domains.

Identity = matches / non-gap REFERENCE positions. Missing query positions are
mismatches. Query insertions absent from a reference are not in its denominator.
All maximizers of this exact fraction are returned; no secondary similarity
ranking, sequence deduplication, rounding, or arbitrary top-N truncation.

Sources:
https://github.com/oxpig/ANARCI/blob/master/lib/python/anarci/anarci.py
https://github.com/prihoda/AbNumber/blob/master/abnumber/germlines.py
https://github.com/prihoda/AbNumber/blob/master/abnumber/chain.py
"""

from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json


def _normalize(sequence):
    if not isinstance(sequence, str):
        raise TypeError("Provide one amino-acid sequence as a string")
    sequence = "".join(sequence.split()).upper()
    if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError("Expected a nonempty, ungapped sequence of the 20 amino acids")
    return sequence


def _all_top_matches(query, references):
    """query and reference values are {aligned_position: residue} mappings."""
    scored = []
    for allele, reference in references.items():
        positions = [p for p, aa in reference.items() if aa != "-"]
        if not positions:
            raise ValueError(f"Empty reference: {allele}")
        matches = sum(query.get(p, "-") == reference[p] for p in positions)
        covered = sum(query.get(p, "-") != "-" for p in positions)
        row = {
            "allele": allele,
            "matches": matches,
            "reference_length": len(positions),
            "query_covered": covered,
            "coverage_pct": 100 * covered / len(positions),
        }
        scored.append((Fraction(matches, len(positions)), row))
    if not scored:
        raise ValueError("No reference alleles available")
    best = max(score for score, _ in scored)
    hits = sorted((row for score, row in scored if score == best),
                  key=lambda row: row["allele"])
    return {
        "identity_pct": float(100 * best),
        "identity_fraction": str(best),
        "alleles": [row["allele"] for row in hits],
        "genes": sorted({row["allele"].split("*")[0] for row in hits}),
        "hits": hits,
        "reference_count": len(scored),
    }


@lru_cache(maxsize=None)
def _anarci_panel(species):
    from anarci.germlines import all_germlines
    raw = {region: all_germlines[region]["H"][species]
           for region in ("V", "J")}
    if any(len(seq) != 128 for panel in raw.values() for seq in panel.values()):
        raise ValueError("Expected ANARCI's 128-state reference alignment")
    digest = sha256(json.dumps(raw, sort_keys=True,
                              separators=(",", ":")).encode()).hexdigest()
    panels = {region: {allele: dict(enumerate(ref, start=1))
                       for allele, ref in panel.items()}
              for region, panel in raw.items()}
    return panels, digest


@lru_cache(maxsize=1)
def _hmm_digest():
    from anarci.anarci import HMM_path
    return sha256((Path(HMM_path) / "ALL.hmm").read_bytes()).hexdigest()


def human_germline_identity_anarci(sequence, *, species="human",
                                  hmmer_species=None):
    """Score all V and J alleles on ANARCI's underlying HMM match states.

    species selects the reference panel being scored, NOT the alignment model.
    Default hmmer_species=None permits all HMM species, as AbNumber normally
    does. Keep this choice fixed across participants and validators.
    Only one recognized domain is accepted, and it must be a heavy chain.
    """
    from anarci.anarci import run_hmmer, check_for_j

    sequence = _normalize(sequence)
    records = [("query", sequence)]
    alignments = run_hmmer(records, ncpu=1, hmmer_species=hmmer_species)
    # Preserve ANARCI's second pass for unusually long CDR3s / missed J regions.
    check_for_j(records, alignments, "imgt")
    _, state_vectors, details = alignments[0]
    if len(state_vectors) != 1 or len(details) != 1:
        raise ValueError("Expected exactly one recognized antibody domain")
    if details[0]["chain_type"] != "H":
        raise ValueError("Expected a heavy-chain domain")

    states = dict(state_vectors[0])
    query = {}
    for position in range(1, 129):
        index = states.get((position, "m"))
        query[position] = "-" if index is None else sequence[index]

    panels, digest = _anarci_panel(species)
    return {
        "method": "anarci_hmm_reference_identity",
        "species": species,
        "reference_sha256": digest,
        "hmm_sha256": _hmm_digest(),
        "alignment_species_filter": hmmer_species,
        "alignment_model": details[0]["id"],
        "V": _all_top_matches(query, panels["V"]),
        "J": _all_top_matches(query, panels["J"]),
    }

