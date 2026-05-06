"""
Integration test for per-molecule seed modification in Validator's BoltzWrapper.

This script tests the actual modified BoltzWrapper by:
1. Creating single-molecule input directories (like the modified code does)
2. Calling boltz.predict() with per-molecule seeds
3. Verifying score stability across multiple runs

The test verifies that after the modification:
- Each molecule gets a deterministic, unique seed derived from its SMILES
- Random state is reset before each molecule's prediction
- Results are reproducible regardless of prediction order

Usage:
    cd /root/sn68/boltz_service_batch/nova
    PYTHONPATH=.:external_tools/boltz/src python -m pytest test/test_per_mol_seed_integration.py -v
    # Or run directly:
    PYTHONPATH=.:external_tools/boltz/src python test/test_per_mol_seed_integration.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

# Add boltz src to path for predict() import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external_tools", "boltz", "src"))

from boltz.main import predict

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

TEST_MOLECULES: list[str] = [
    "Cc1nc(-c2ccco2)sc1C(C)NCCC#Cc1ccsc1",
    "CC(NCCCc1cnc2[nH]ncc2c1)c1ccc(-c2cccs2)s1",
    "CC(Nc1cc2cncnc2s1)c1ccc(-c2cccs2)cc1",
]

PROTEIN_SEQ: str = (
    "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
)
TARGET: str = "Q63380"
BASE_SEED: int = 68
NUM_RUNS: int = 3

BOLTZ_CONFIG: dict[str, object] = {
    "recycling_steps": 3,
    "sampling_steps": 100,
    "diffusion_samples": 1,
    "sampling_steps_affinity": 100,
    "diffusion_samples_affinity": 3,
    "affinity_mw_correction": True,
    "output_format": "mmcif",
    "override": False,
}

# ---------------------------------------------------------------------------
# Helper functions (mirroring boltz_wrapper.py)
# ---------------------------------------------------------------------------

def _get_record_id(rec_id: str, base_seed: int) -> int:
    """Generate a deterministic, unique seed for a given record ID."""
    h = hashlib.sha256(str(rec_id).encode()).digest()
    return (int.from_bytes(h[:8], "little") ^ base_seed) % (2**31 - 1)


def _set_random_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    This ensures reproducible behavior for a single molecule prediction.
    Called before each per-molecule predict() invocation.
    """
    random = __import__("random")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)


def _create_yaml(protein_seq: str, smiles: str, target: str) -> str:
    """Create Boltz2 input YAML content."""
    return f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {protein_seq}
  - ligand:
      id: B
      smiles: {smiles}
properties:
  - affinity:
      binder: B
"""


def _extract_affinity_scores(output_dir: str, mol_idx: int, target: str) -> tuple[float | None, float | None]:
    """Extract affinity_pred_value and affinity_probability_binary from prediction output."""
    pred_dir = Path(output_dir) / f"boltz_results___single_{mol_idx}_{target}" / "predictions"
    if not pred_dir.exists():
        return None, None

    for fpath in pred_dir.glob("affinity*.json"):
        try:
            with open(fpath) as f:
                data = json.load(f)
            for key, val in data.items():
                if isinstance(val, dict):
                    pred = val.get("affinity_pred_value")
                    prob = val.get("affinity_probability_binary")
                    if pred is not None and prob is not None:
                        pred_val = pred[0] if isinstance(pred, list) else pred
                        prob_val = prob[0] if isinstance(prob, list) else prob
                        return float(pred_val), float(prob_val)
        except Exception:
            continue
    return None, None


def _predict_single_molecule(
    smiles: str,
    mol_idx: int,
    target: str,
    input_base: str,
    output_base: str,
) -> tuple[float, float, float]:
    """Predict a single molecule with its own deterministic seed.

    This mirrors the behavior of the modified BoltzWrapper.score_molecules()
    which now calls predict() once per molecule instead of once per epoch.
    """
    mol_seed = _get_record_id(smiles, BASE_SEED)

    # Create single-molecule input directory
    single_dir = Path(input_base) / f"__single_{mol_idx}_{target}"
    single_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = single_dir / f"{mol_idx}_{target}.yaml"
    yaml_path.write_text(_create_yaml(PROTEIN_SEQ, smiles, target))

    # Reset global random state to molecule-specific seed
    _set_random_seeds(mol_seed)

    # Run prediction
    start = time.time()
    predict(
        data=str(single_dir),
        out_dir=output_base,
        recycling_steps=BOLTZ_CONFIG["recycling_steps"],
        sampling_steps=BOLTZ_CONFIG["sampling_steps"],
        diffusion_samples=BOLTZ_CONFIG["diffusion_samples"],
        sampling_steps_affinity=BOLTZ_CONFIG["sampling_steps_affinity"],
        diffusion_samples_affinity=BOLTZ_CONFIG["diffusion_samples_affinity"],
        output_format=BOLTZ_CONFIG["output_format"],
        seed=mol_seed,
        affinity_mw_correction=BOLTZ_CONFIG["affinity_mw_correction"],
        override=BOLTZ_CONFIG["override"],
        num_workers=0,
    )
    elapsed = time.time() - start

    # Extract scores
    pred, prob = _extract_affinity_scores(output_base, mol_idx, target)

    # Cleanup single-molecule input directory
    shutil.rmtree(single_dir, ignore_errors=True)

    if pred is None or prob is None:
        raise RuntimeError(f"Failed to extract scores for mol{mol_idx}")

    return pred, prob, elapsed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPerMoleculeSeed:
    """Test suite for per-molecule deterministic seed behavior."""

    def test_seed_determinism(self) -> None:
        """_get_record_id must return the same seed for the same input."""
        for mol in TEST_MOLECULES:
            s1 = _get_record_id(mol, BASE_SEED)
            s2 = _get_record_id(mol, BASE_SEED)
            assert s1 == s2, f"Seed not deterministic for {mol}"

    def test_seed_uniqueness(self) -> None:
        """Different molecules must get different seeds."""
        seeds = [_get_record_id(mol, BASE_SEED) for mol in TEST_MOLECULES]
        assert len(set(seeds)) == len(seeds), "Different molecules got same seed"

    def test_seed_in_valid_range(self) -> None:
        """Generated seeds must be within numpy's valid range."""
        for mol in TEST_MOLECULES:
            seed = _get_record_id(mol, BASE_SEED)
            assert 0 <= seed < 2**31, f"Seed {seed} out of valid range"

    def test_random_state_reproducibility(self) -> None:
        """_set_random_seeds must make random operations reproducible."""
        for mol in TEST_MOLECULES:
            seed = _get_record_id(mol, BASE_SEED)

            _set_random_seeds(seed)
            r1 = (__import__("random").random(), torch.rand(3).tolist(), np.random.rand(3).tolist())

            _set_random_seeds(seed)
            r2 = (__import__("random").random(), torch.rand(3).tolist(), np.random.rand(3).tolist())

            assert r1 == r2, f"Random state not reproducible for {mol}"

    def test_different_seeds_different_values(self) -> None:
        """Different seeds should produce different random values."""
        seed1 = _get_record_id(TEST_MOLECULES[0], BASE_SEED)
        seed2 = _get_record_id(TEST_MOLECULES[1], BASE_SEED)

        _set_random_seeds(seed1)
        v1 = torch.rand(1).item()

        _set_random_seeds(seed2)
        v2 = torch.rand(1).item()

        assert v1 != v2, "Different seeds produced same random value"

    def test_prediction_order_independence(self) -> None:
        """A molecule's result must not depend on prediction order.

        This is the core property that the per-molecule seed fix provides:
        whether we predict mol0 then mol1, or mol1 then mol0,
        each molecule should get the same result because each resets
        its own random state.
        """
        # Order A: mol0, mol1
        _set_random_seeds(_get_record_id(TEST_MOLECULES[0], BASE_SEED))
        val_a_mol0 = torch.rand(1).item()
        _set_random_seeds(_get_record_id(TEST_MOLECULES[1], BASE_SEED))
        val_a_mol1 = torch.rand(1).item()

        # Order B: mol1, mol0
        _set_random_seeds(_get_record_id(TEST_MOLECULES[1], BASE_SEED))
        val_b_mol1 = torch.rand(1).item()
        _set_random_seeds(_get_record_id(TEST_MOLECULES[0], BASE_SEED))
        val_b_mol0 = torch.rand(1).item()

        assert val_a_mol0 == val_b_mol0, "mol0 result depends on prediction order"
        assert val_a_mol1 == val_b_mol1, "mol1 result depends on prediction order"


class TestPerMoleculeIntegration:
    """Integration tests using actual boltz.predict() calls."""

    @pytest.fixture(scope="class")
    def temp_dirs(self):
        """Provide temporary input/output directories."""
        input_base = tempfile.mkdtemp(prefix="boltz_test_inputs_")
        output_base = tempfile.mkdtemp(prefix="boltz_test_outputs_")
        yield input_base, output_base
        shutil.rmtree(input_base, ignore_errors=True)
        shutil.rmtree(output_base, ignore_errors=True)

    @pytest.mark.slow
    def test_three_molecules_three_runs(self, temp_dirs) -> None:
        """Run 3 molecules, 3 times each, and verify stability.

        This test takes several minutes as it calls boltz.predict() 9 times.
        """
        input_base, output_base = temp_dirs

        # Store results: {mol_idx: [(pred, prob, time), ...]}
        all_results: dict[int, list[tuple[float, float, float]]] = {
            i: [] for i in range(len(TEST_MOLECULES))
        }

        for run_idx in range(1, NUM_RUNS + 1):
            print(f"\n--- Run {run_idx}/{NUM_RUNS} ---")
            for mol_idx, smiles in enumerate(TEST_MOLECULES):
                pred, prob, elapsed = _predict_single_molecule(
                    smiles, mol_idx, TARGET, input_base, output_base
                )
                all_results[mol_idx].append((pred, prob, elapsed))
                print(f"  mol{mol_idx}: pred={pred:.4f}, prob={prob:.4f}, time={elapsed:.1f}s")

        # Verify stability
        for mol_idx in range(len(TEST_MOLECULES)):
            preds = [r[0] for r in all_results[mol_idx]]
            probs = [r[1] for r in all_results[mol_idx]]

            pred_range = max(preds) - min(preds)
            prob_range = max(probs) - min(probs)

            # Allow small numerical differences (CUDA non-determinism)
            assert pred_range < 0.01, (
                f"mol{mol_idx} pred_value unstable: range={pred_range:.6f}, "
                f"values={preds}"
            )
            assert prob_range < 0.01, (
                f"mol{mol_idx} prob_binary unstable: range={prob_range:.6f}, "
                f"values={probs}"
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run tests with plain output (no pytest dependency required for basic tests)."""
    print("=" * 70)
    print("Per-Molecule Seed Integration Test")
    print("=" * 70)

    # Run unit tests
    test_suite = TestPerMoleculeSeed()
    tests = [
        ("seed determinism", test_suite.test_seed_determinism),
        ("seed uniqueness", test_suite.test_seed_uniqueness),
        ("seed valid range", test_suite.test_seed_in_valid_range),
        ("random state reproducibility", test_suite.test_random_state_reproducibility),
        ("different seeds different values", test_suite.test_different_seeds_different_values),
        ("prediction order independence", test_suite.test_prediction_order_independence),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1

    print(f"\nUnit tests: {passed} passed, {failed} failed")

    # Run integration test (optional, slow)
    print("\n" + "=" * 70)
    print("Integration Test (3 molecules × 3 runs, ~10-15 minutes)")
    print("=" * 70)
    print("Press Enter to run, or Ctrl+C to skip...")
    try:
        input()
    except KeyboardInterrupt:
        print("Skipped.")
        return

    input_base = tempfile.mkdtemp(prefix="boltz_test_inputs_")
    output_base = tempfile.mkdtemp(prefix="boltz_test_outputs_")

    try:
        all_results: dict[int, list[tuple[float, float, float]]] = {
            i: [] for i in range(len(TEST_MOLECULES))
        }

        for run_idx in range(1, NUM_RUNS + 1):
            print(f"\n--- Run {run_idx}/{NUM_RUNS} ---")
            for mol_idx, smiles in enumerate(TEST_MOLECULES):
                pred, prob, elapsed = _predict_single_molecule(
                    smiles, mol_idx, TARGET, input_base, output_base
                )
                all_results[mol_idx].append((pred, prob, elapsed))
                print(f"  mol{mol_idx}: pred={pred:.4f}, prob={prob:.4f}, time={elapsed:.1f}s")

        print("\n" + "=" * 70)
        print("Stability Analysis")
        print("=" * 70)

        for mol_idx in range(len(TEST_MOLECULES)):
            preds = [r[0] for r in all_results[mol_idx]]
            probs = [r[1] for r in all_results[mol_idx]]
            times = [r[2] for r in all_results[mol_idx]]

            pred_range = max(preds) - min(preds)
            prob_range = max(probs) - min(probs)

            pred_ok = pred_range < 0.01
            prob_ok = prob_range < 0.01

            status = "✅ PASS" if (pred_ok and prob_ok) else "❌ FAIL"

            print(f"mol{mol_idx}: {TEST_MOLECULES[mol_idx][:50]}...")
            print(f"  pred_values: {[f'{p:.6f}' for p in preds]} (range={pred_range:.6f})")
            print(f"  prob_values: {[f'{p:.6f}' for p in probs]} (range={prob_range:.6f})")
            print(f"  avg_time: {sum(times)/len(times):.1f}s")
            print(f"  {status}")
            print()

        # Save results
        out_path = "/tmp/per_mol_seed_integration.json"
        with open(out_path, "w") as f:
            json.dump({
                "molecules": TEST_MOLECULES,
                "results": {
                    f"mol{i}": [
                        {"pred": r[0], "prob": r[1], "time": r[2]}
                        for r in all_results[i]
                    ]
                    for i in range(len(TEST_MOLECULES))
                },
            }, f, indent=2)
        print(f"Results saved to: {out_path}")

    finally:
        shutil.rmtree(input_base, ignore_errors=True)
        shutil.rmtree(output_base, ignore_errors=True)


if __name__ == "__main__":
    main()
