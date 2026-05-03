"""Round-trip and edge cases for utils/fasta.py.

Loaded via importlib to avoid triggering utils/__init__.py (which transitively
imports bittensor). See tests/conftest.py.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module


fasta = load_module("nova_fasta", "utils/fasta.py")


def test_round_trip_single_record(tmp_path):
    path = tmp_path / "single.fasta"
    items = [("seq1", "MVLSPADKTNVKAA")]
    fasta.write_fasta(items, str(path))
    assert fasta.read_fasta(str(path)) == items


def test_round_trip_multiple_records(tmp_path):
    path = tmp_path / "multi.fasta"
    items = [
        ("alpha", "ACDEFGHIKL"),
        ("beta", "MNPQRSTVWY"),
        ("gamma", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMP"),
    ]
    fasta.write_fasta(items, str(path))
    assert fasta.read_fasta(str(path)) == items


def test_long_sequence_is_wrapped_at_80_columns(tmp_path):
    path = tmp_path / "wrapped.fasta"
    seq = "A" * 200
    fasta.write_fasta([("long", seq)], str(path))

    body_lines = [ln for ln in path.read_text().splitlines() if not ln.startswith(">")]
    assert max(len(ln) for ln in body_lines) == 80
    assert len(body_lines) == 3  # 80 + 80 + 40

    [(name, recovered)] = fasta.read_fasta(str(path))
    assert name == "long"
    assert recovered == seq


def test_read_skips_blank_lines(tmp_path):
    path = tmp_path / "blanks.fasta"
    path.write_text(">a\nACGT\n\n\n>b\nTGCA\n")
    assert fasta.read_fasta(str(path)) == [("a", "ACGT"), ("b", "TGCA")]


def test_read_handles_header_with_trailing_metadata(tmp_path):
    """`>name extra info` must keep only the first whitespace-delimited token."""
    path = tmp_path / "header.fasta"
    path.write_text(">seq1 organism=human chain=A\nACGT\n")
    assert fasta.read_fasta(str(path)) == [("seq1", "ACGT")]


def test_read_empty_file(tmp_path):
    path = tmp_path / "empty.fasta"
    path.write_text("")
    assert fasta.read_fasta(str(path)) == []


@pytest.mark.parametrize("seq_len", [0, 1, 79, 80, 81, 159, 160])
def test_round_trip_boundary_lengths(tmp_path, seq_len):
    path = tmp_path / f"len{seq_len}.fasta"
    items = [("x", "A" * seq_len)]
    fasta.write_fasta(items, str(path))
    assert fasta.read_fasta(str(path)) == items
