"""
Test the core per-molecule seed logic without importing nova's full stack.

This script tests:
1. _get_record_id() generates deterministic seeds
2. _set_random_seeds() properly resets global random state
3. Each molecule gets a unique, reproducible seed
"""

import os
import sys
import hashlib
import random

# Set CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch

# Replicate the functions from boltz_wrapper.py
def _get_record_id(rec_id, base_seed):
    """Generate a deterministic, unique seed for a given record ID."""
    h = hashlib.sha256(str(rec_id).encode()).digest()
    return (int.from_bytes(h[:8], "little") ^ base_seed) % (2**31 - 1)

def _set_random_seeds(seed):
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)

# Test molecules
TEST_MOLECULES = [
    "Cc1nc(-c2ccco2)sc1C(C)NCCC#Cc1ccsc1",
    "CC(NCCCc1cnc2[nH]ncc2c1)c1ccc(-c2cccs2)s1",
    "CC(Nc1cc2cncnc2s1)c1ccc(-c2cccs2)cc1",
]

BASE_SEED = 68

print("=" * 70)
print("Core Per-Molecule Seed Logic Test")
print("=" * 70)

# Test 1: Seed determinism
print("\n[Test 1] Seed determinism")
for mol in TEST_MOLECULES:
    s1 = _get_record_id(mol, BASE_SEED)
    s2 = _get_record_id(mol, BASE_SEED)
    assert s1 == s2, f"Seed not deterministic for {mol}"
    print(f"  {mol[:40]}... -> seed={s1}")

# Test 2: Different molecules get different seeds
print("\n[Test 2] Unique seeds for different molecules")
seeds = [_get_record_id(mol, BASE_SEED) for mol in TEST_MOLECULES]
assert len(set(seeds)) == len(seeds), "Different molecules got same seed!"
print(f"  All {len(seeds)} seeds are unique: {seeds}")

# Test 3: _set_random_seeds reproducibility
print("\n[Test 3] Random state reproducibility")
for mol in TEST_MOLECULES:
    seed = _get_record_id(mol, BASE_SEED)
    
    _set_random_seeds(seed)
    r1 = random.random()
    t1 = torch.rand(3).tolist()
    n1 = np.random.rand(3).tolist()
    
    _set_random_seeds(seed)
    r2 = random.random()
    t2 = torch.rand(3).tolist()
    n2 = np.random.rand(3).tolist()
    
    assert r1 == r2, f"random not reproducible for {mol}"
    assert t1 == t2, f"torch not reproducible for {mol}"
    assert n1 == n2, f"numpy not reproducible for {mol}"
    print(f"  {mol[:40]}... -> ✅ reproducible")

# Test 4: Different seeds produce different random values
print("\n[Test 4] Different seeds produce different random values")
seed1 = _get_record_id(TEST_MOLECULES[0], BASE_SEED)
seed2 = _get_record_id(TEST_MOLECULES[1], BASE_SEED)

_set_random_seeds(seed1)
r1 = random.random()

_set_random_seeds(seed2)
r2 = random.random()

assert r1 != r2, "Different seeds produced same random value!"
print(f"  Seed {seed1} -> random={r1:.6f}")
print(f"  Seed {seed2} -> random={r2:.6f}")
print(f"  ✅ Different values")

# Test 5: Simulate per-molecule prediction order independence
print("\n[Test 5] Prediction order independence")
# Simulate: if we predict mol0 then mol1, vs mol1 then mol0,
# each should get the same result (because each resets its own seed)

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

assert val_a_mol0 == val_b_mol0, "mol0 result depends on prediction order!"
assert val_a_mol1 == val_b_mol1, "mol1 result depends on prediction order!"
print(f"  mol0 (order A): {val_a_mol0:.6f} == mol0 (order B): {val_b_mol0:.6f} -> ✅")
print(f"  mol1 (order A): {val_a_mol1:.6f} == mol1 (order B): {val_b_mol1:.6f} -> ✅")

print("\n" + "=" * 70)
print("All tests passed! Per-molecule seed logic is correct.")
print("=" * 70)
