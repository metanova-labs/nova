"""Parser tests for the `--local_input_file` format used by `make run-local`.

Format: `uid|mol_name1,mol_name2,...|protein_seq1,protein_seq2,...`
See nova/example_local_input for the canonical sample.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module


local_input = load_module("nova_local_input", "utils/local_input.py")


def test_parse_basic_line():
    uid, mols, seqs = local_input.parse_local_input_line(
        "1|aspirin,caffeine|MVLSP,MEEPQSDP"
    )
    assert uid == 1
    assert mols == ["aspirin", "caffeine"]
    assert seqs == ["MVLSP", "MEEPQSDP"]


def test_parse_strips_trailing_newline():
    uid, mols, seqs = local_input.parse_local_input_line("42|x|Y\n")
    assert uid == 42
    assert mols == ["x"]
    assert seqs == ["Y"]


def test_parse_single_molecule_and_sequence():
    uid, mols, seqs = local_input.parse_local_input_line("7|onlyone|MEEPQ")
    assert (uid, mols, seqs) == (7, ["onlyone"], ["MEEPQ"])


def test_parse_drops_empty_csv_entries():
    """Trailing commas should not produce phantom empty strings."""
    uid, mols, seqs = local_input.parse_local_input_line("3|a,b,|p,q,")
    assert mols == ["a", "b"]
    assert seqs == ["p", "q"]


def test_parse_rejects_non_integer_uid():
    with pytest.raises(ValueError):
        local_input.parse_local_input_line("notanint|x|Y")


def test_parse_rejects_wrong_field_count():
    with pytest.raises(ValueError, match="3 pipe-separated fields"):
        local_input.parse_local_input_line("1|onlytwo")
    with pytest.raises(ValueError, match="3 pipe-separated fields"):
        local_input.parse_local_input_line("1|a|b|c")


def test_parse_matches_example_local_input(tmp_path):
    """End-to-end sanity vs. the file actually shipped in the repo."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    example = repo_root / "example_local_input"
    if not example.exists():
        pytest.skip("example_local_input not present")

    parsed = []
    for line in example.read_text().splitlines():
        if not line.strip():
            continue
        parsed.append(local_input.parse_local_input_line(line))

    assert parsed, "example file was empty"
    for uid, mols, seqs in parsed:
        assert isinstance(uid, int)
        assert mols and all(m for m in mols)
        assert seqs and all(s for s in seqs)
