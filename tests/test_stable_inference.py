#!/usr/bin/env python3
"""
Test suite for stable_inference determinism.

Validates that Boltz2 scoring is deterministic when using spawn process
isolation via utils/stable_inference.py.

Usage:
    cd /root/gits/nova
    source /root/sn68/boltz_service/sn68-boltz-service/core/boltz_service/.venv/bin/activate
    python tests/test_stable_inference.py

Tests:
    1. Small test: 3 molecules x 2 rounds (~10 min)
    2. Large test: 15 molecules x 5 rounds (~90 min)
"""

import os
import sys
import time
import json
import argparse

NOVA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

sys.path.insert(0, NOVA_DIR)
sys.path.insert(0, os.path.join(NOVA_DIR, "external_tools", "boltz", "src"))
sys.path.insert(0, os.path.join(NOVA_DIR, "utils"))

from stable_inference import get_stable_scores

# Test molecules - Real molecules from production validator scoring
# All SMILES are valid and have been verified with RDKit
TEST_SMILES_SMALL = [
    "CS(=O)c1ncc(-c2ccc(C(=O)NCCc3ccc(F)cc3)s2)cn1",
    "NS(=O)(=O)c1ncc(-c2ccc(CNC3CC3)cc2F)cn1",
    "COc1nccc(F)c1-c1ccn2cc(OS(=O)(=O)C(F)(F)F)nc2c1",
    "CCCCNS(=O)(=O)c1ccc(-c2n[nH]c(CCC)n2)cc1",
    "CCc1nc(-c2ccc(S(=O)(=O)Nc3ccc(C)cc3)cc2)n[nH]1",
]

TEST_SMILES_LARGE = [
    "O=C1C=CC(=NS(=O)(=O)c2ccc(-c3ccc4cc[nH]c4c3)cc2)C=C1",
    "CCSc1ncc(-c2ccc(CN(C)Cc3ccccc3)cc2)cn1",
    "CCCc1nc(-c2ccc(C(=O)NCc3ccc(OC)cc3)cc2)n[nH]1",
    "CCc1cccc(-c2ccc(C(=O)COS(=O)(=O)C(F)(F)F)cc2)c1F",
    "CCCc1nc(-c2ccc(CN3CCNCC3)c(Cl)c2)c[nH]1",
    "CCCc1ncc(-c2ccc(S(=O)(=O)Nc3cccc(Cl)c3C)cc2)cn1",
    "CC(NC1CC1)c1ccc(-c2cnc(C(F)(F)C(F)(F)C(F)(F)F)nc2)s1",
    "CCSc1nc(-c2cnc(NCc3ccc(OC)cc3)nc2)n[nH]1",
    "CC(NC1CC1)c1ccc(-c2ccc(C(=O)CC(=O)C(F)F)cc2)s1",
    "CN(C)CCn1cc(-c2ccc(-c3cnc(C4CC4)[nH]3)cc2)cn1",
    "CN1CCC(c2ccc(-c3nc(C4CCC4)n[nH]3)cc2)CC1",
    "CC(C)CNC(=O)c1cc(-c2cc(Cl)c(C3CC3)cn2)ccc1F",
    "CCCc1ncc(-c2cnc(NCc3ccc(OC)cc3)nc2)s1",
    "CCC(=O)c1cc(-c2ccc(C(=O)NC3CCC3)cc2)c[nH]1",
    "Cc1cc(C#N)cc(-c2ccn3cc(OS(=O)(=O)C(F)(F)F)nc3c2)c1",
]

TARGET = "Q92769"
with open(os.path.join(NOVA_DIR, "data", "msa_files", TARGET + ".a3m")) as f:
    lines = f.readlines()
PROTEIN_SEQUENCE = lines[1].strip()

BASE_SEED = 68


def _run_round(smiles_list, round_idx, total_rounds):
    """Run one scoring round and return scores dict."""
    print(f"\n{'='*70}")
    print(f"ROUND {round_idx}/{total_rounds}")
    print(f"{'='*70}")

    t0 = time.time()
    scores, _components = get_stable_scores(smiles_list, TARGET, PROTEIN_SEQUENCE, BASE_SEED, gpu_id="0")
    t1 = time.time()
    elapsed = t1 - t0

    print(f"  Total time: {elapsed:.1f}s = {elapsed/60:.1f}min")
    for i, smi in enumerate(smiles_list):
        score = scores.get(smi)
        print(f"  [{i+1:2d}] {smi[:45]}... -> {score}")

    return scores, elapsed


def _analyze_consistency(all_rounds, smiles_list, round_times):
    """Analyze consistency across rounds."""
    print(f"\n{'='*70}")
    print("CONSISTENCY ANALYSIS")
    print(f"{'='*70}")

    consistency_results = []
    for i, smi in enumerate(smiles_list):
        scores_across_rounds = [r.get(smi) for r in all_rounds]
        all_same = all(s == scores_across_rounds[0] for s in scores_across_rounds)
        status = "MATCH" if all_same else "DIFFER"
        consistency_results.append((smi, all_same, scores_across_rounds))

        print(f"\n  [{i+1:2d}] {smi[:50]}...")
        for r_idx, s in enumerate(scores_across_rounds):
            print(f"       Round {r_idx+1}: {s}")
        print(f"       -> {status}")

    match_count = sum(1 for _, ok, _ in consistency_results if ok)
    differ_count = len(smiles_list) - match_count

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Molecules: {len(smiles_list)}")
    print(f"  Rounds: {len(all_rounds)}")
    print(f"  MATCH: {match_count} ({match_count/len(smiles_list)*100:.1f}%)")
    print(f"  DIFFER: {differ_count} ({differ_count/len(smiles_list)*100:.1f}%)")

    print(f"\n  Timing:")
    for r_idx, t in enumerate(round_times):
        print(f"    Round {r_idx+1}: {t:.1f}s = {t/60:.1f}min")
    avg_time = sum(round_times) / len(round_times)
    print(f"    Average: {avg_time:.1f}s = {avg_time/60:.1f}min")
    total_time = sum(round_times)
    print(f"    Total: {total_time:.1f}s = {total_time/60:.1f}min")

    return match_count, differ_count, consistency_results


def test_small():
    """Small test: 3 molecules x 2 rounds."""
    print("="*70)
    print("SMALL TEST: 3 molecules x 2 rounds")
    print("="*70)

    all_rounds = []
    round_times = []

    for r in range(2):
        scores, elapsed = _run_round(TEST_SMILES_SMALL, r + 1, 2)
        all_rounds.append(scores)
        round_times.append(elapsed)

    match_count, differ_count, _ = _analyze_consistency(all_rounds, TEST_SMILES_SMALL, round_times)
    return differ_count == 0


def test_large():
    """Large test: 15 molecules x 5 rounds."""
    print("="*70)
    print("LARGE TEST: 15 molecules x 5 rounds")
    print("="*70)

    all_rounds = []
    round_times = []

    for r in range(5):
        scores, elapsed = _run_round(TEST_SMILES_LARGE, r + 1, 5)
        all_rounds.append(scores)
        round_times.append(elapsed)

    match_count, differ_count, consistency_results = _analyze_consistency(
        all_rounds, TEST_SMILES_LARGE, round_times
    )

    # Save detailed results
    result_data = {
        "molecules": TEST_SMILES_LARGE,
        "num_rounds": 5,
        "round_times": round_times,
        "results_per_round": [
            {smi: float(score) if score is not None else None for smi, score in r.items()}
            for r in all_rounds
        ],
        "consistency": {
            "match": match_count,
            "differ": differ_count,
            "match_rate": match_count / len(TEST_SMILES_LARGE),
        },
    }

    output_path = os.path.join(NOVA_DIR, "tests", "test_15mols_5rounds_result.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n  Detailed results saved to: {output_path}")

    return differ_count == 0


def main():
    parser = argparse.ArgumentParser(description="Test stable inference determinism")
    parser.add_argument(
        "--test",
        choices=["small", "large", "all"],
        default="all",
        help="Which test to run (default: all)",
    )
    args = parser.parse_args()

    results = []

    if args.test in ("small", "all"):
        try:
            passed = test_small()
            results.append(("small (3x2)", passed))
        except Exception as e:
            print(f"\nSMALL TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(("small (3x2)", False))

    if args.test in ("large", "all"):
        try:
            passed = test_large()
            results.append(("large (15x5)", passed))
        except Exception as e:
            print(f"\nLARGE TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(("large (15x5)", False))

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
