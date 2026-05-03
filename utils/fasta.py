"""Pure-stdlib FASTA helpers extracted from utils/files.py for unit testing.

Importing utils/files.py pulls bittensor + dotenv at module load. These two
helpers are pure I/O and deserve a dependency-free home so CI can exercise
them without installing the full validator stack.
"""

from typing import List, Tuple


def read_fasta(path: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    name = None
    seq_parts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    items.append((name, "".join(seq_parts)))
                name = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if name is not None:
            items.append((name, "".join(seq_parts)))
    return items


def write_fasta(items: List[Tuple[str, str]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for name, seq in items:
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")
