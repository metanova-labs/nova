"""Pure-stdlib parser for the `--local_input_file` format.

Format: one entry per line, three pipe-separated fields:
    uid|mol_name1,mol_name2,...|protein_sequence1,protein_sequence2,...

Example:
    1|aspirin,caffeine|MVLSPADKTNVKAA...,MEEPQSDPSVEPPLSQETFSDL...

Extracted from utils/files.py.read_local_input_file so the parsing rule is
testable without bittensor/dotenv at import time.
"""

from typing import List, Tuple


def parse_local_input_line(line: str) -> Tuple[int, List[str], List[str]]:
    """Parse one line of the local-input file.

    Raises ValueError if the line does not contain exactly three pipe-separated
    fields or if the uid is not an integer. Empty/whitespace-only lines must
    be filtered by the caller.
    """
    parts = line.strip().split("|")
    if len(parts) != 3:
        raise ValueError(
            f"expected 3 pipe-separated fields, got {len(parts)}: {line!r}"
        )
    uid_str, mol_names, protein_sequences = parts
    return (
        int(uid_str),
        [m for m in mol_names.split(",") if m],
        [s for s in protein_sequences.split(",") if s],
    )
