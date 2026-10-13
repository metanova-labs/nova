import sqlite3
import os
import random
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
import bittensor as bt


def get_db_path() -> str:
    """Return the path to the molecules SQLite database."""
    return os.path.join(os.path.dirname(__file__), "molecules.sqlite")


def as_mask(value) -> int:
    """Role mask as an int (BLOB or legacy integer)."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return int.from_bytes(bytes(value), "big")
    return int(value or 0)


def _has_role(role_mask: int, role: int) -> bool:
    """A molecule fills a role only when it carries every bit the role requires."""
    return role != 0 and (role_mask & role) == role


def _column_names(cursor, table: str) -> set:
    return {r[1] for r in cursor.execute(f"PRAGMA table_info({table})")}


def get_reaction_info(rxn_id: int, db_path: str) -> tuple:
    """(smarts, roleA, roleB, roleC)."""
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT smarts, roleA, roleB, roleC FROM reactions WHERE rxn_id = ?",
            (str(rxn_id),)).fetchone()
        conn.close()
        if not row:
            return None
        smarts, a, b, c = row
        return smarts, as_mask(a), as_mask(b), (as_mask(c) if c is not None else None)
    except Exception as e:
        bt.logging.error(f"Error getting reaction info: {e}")
        return None


def get_molecules(mol_ids: list, db_path: str) -> list:
    """[(smiles, role_mask), ...] in the order requested."""
    try:
        conn = sqlite3.connect(db_path)
        molecules = []
        for mol_id in mol_ids:
            row = conn.execute("SELECT smiles, role_mask FROM molecules WHERE mol_id = ?",
                               (mol_id,)).fetchone()
            molecules.append((row[0], as_mask(row[1])) if row else None)
        conn.close()
        return molecules
    except Exception as e:
        bt.logging.error(f"Error getting molecules: {e}")
        return [None] * len(mol_ids)


def get_molecules_by_role(role_mask: int, db_path: str) -> list:
    """[(mol_id, smiles, role_mask), ...] for every molecule filling the role."""
    try:
        conn = sqlite3.connect(db_path)
        indexed = conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='molecule_roles'").fetchone()

        if indexed:
            bits = [i for i in range(role_mask.bit_length()) if role_mask >> i & 1]
            if not bits:
                conn.close()
                return []
            inner = " INTERSECT ".join("SELECT mol_id FROM molecule_roles WHERE bit = ?" for _ in bits)
            rows = conn.execute(
                f"SELECT m.mol_id, m.smiles, m.role_mask FROM molecules m "
                f"JOIN ({inner}) r ON r.mol_id = m.mol_id", bits).fetchall()
        else:
            rows = conn.execute("SELECT mol_id, smiles, role_mask FROM molecules").fetchall()

        conn.close()
        out = [(i, s, as_mask(rm)) for i, s, rm in rows]
        return out if indexed else [r for r in out if (r[2] & role_mask) == role_mask]
    except Exception as e:
        bt.logging.error(f"Error getting molecules by role {role_mask}: {e}")
        return []


def get_available_reactions(db_path: str) -> List[Tuple[int, int, int, Optional[int]]]:
    """
    Return list of (rxn_id, roleA, roleB, roleC) for all reactions.
    roleC is None for 2-component reactions.
    """
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT rxn_id, roleA, roleB, roleC FROM reactions").fetchall()
        conn.close()
        return [(int(r[0]), as_mask(r[1]), as_mask(r[2]), (as_mask(r[3]) if r[3] is not None else None))
                for r in rows]
    except Exception as e:
        bt.logging.error(f"Error listing reactions: {e}")
        return []


def get_molecule_ids_for_role(db_path: str, role: int) -> List[int]:
    """Return mol_ids that can fulfill this role."""
    return [mol_id for mol_id, _, _ in get_molecules_by_role(role, db_path)]


def get_random_reaction_product(
    db_path: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """
    Pick a random reaction and valid reactant IDs, return product name in
    'rxn:rxn_id:mol1_id:mol2_id' or 'rxn:rxn_id:mol1_id:mol2_id:mol3_id' format.
    Returns None if DB is missing, empty, or no valid combination is found.
    """
    db_path = db_path or get_db_path()
    if not os.path.isfile(db_path):
        bt.logging.error(f"Combinatorial DB not found: {db_path}")
        return None
    rng = rng or random
    reactions = get_available_reactions(db_path)
    if not reactions:
        return None
    rxn_id, roleA, roleB, roleC = rng.choice(reactions)
    ids_a = get_molecule_ids_for_role(db_path, roleA)
    ids_b = get_molecule_ids_for_role(db_path, roleB)
    if not ids_a or not ids_b:
        return None
    if roleC is None or roleC == 0:
        mol1_id = rng.choice(ids_a)
        mol2_id = rng.choice(ids_b)
        product_smiles = react_molecules(rxn_id, mol1_id, mol2_id, db_path)
        if product_smiles is None:
            return None
        return f"rxn:{rxn_id}:{mol1_id}:{mol2_id}"
    else:
        ids_c = get_molecule_ids_for_role(db_path, roleC)
        if not ids_c:
            return None
        mol1_id = rng.choice(ids_a)
        mol2_id = rng.choice(ids_b)
        mol3_id = rng.choice(ids_c)
        product_smiles = react_three_components(
            rxn_id, mol1_id, mol2_id, mol3_id, db_path
        )
        if product_smiles is None:
            return None
        return f"rxn:{rxn_id}:{mol1_id}:{mol2_id}:{mol3_id}"


def combine_triazole_synthons(azide_smiles: str, alkyne_smiles: str) -> str:
    """Combine azide and alkyne synthons to form triazole."""
    try:
        m1 = Chem.RWMol(Chem.MolFromSmiles(azide_smiles))   # azide with [1*]
        m2 = Chem.RWMol(Chem.MolFromSmiles(alkyne_smiles))  # alkyne with [2*]

        if not m1 or not m2:
            return None

        a1 = next((i for i, atom in enumerate(m1.GetAtoms()) if atom.GetSymbol() == '*' and atom.GetIsotope() == 1), None)
        a2 = next((i for i, atom in enumerate(m2.GetAtoms()) if atom.GetSymbol() == '*' and atom.GetIsotope() == 2), None)

        if a1 is None or a2 is None:
            return None

        n1 = m1.GetAtomWithIdx(a1).GetNeighbors()[0].GetIdx()
        n2 = m2.GetAtomWithIdx(a2).GetNeighbors()[0].GetIdx()

        combined = Chem.RWMol(m1)
        atom_mapping = {}

        for i, atom in enumerate(m2.GetAtoms()):
            if i != a2:
                atom_mapping[i] = combined.AddAtom(atom)

        for bond in m2.GetBonds():
            begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a2 not in (begin_idx, end_idx):
                combined.AddBond(atom_mapping[begin_idx], atom_mapping[end_idx], bond.GetBondType())

        combined.RemoveAtom(a1)
        n1_adj = n1 - (1 if n1 > a1 else 0)
        n2_adj = atom_mapping[n2] - (1 if atom_mapping[n2] > a1 else 0)
        combined.AddBond(n1_adj, n2_adj, Chem.BondType.SINGLE)

        Chem.SanitizeMol(combined)
        return Chem.MolToSmiles(combined)

    except Exception as e:
        bt.logging.error(f"Error in triazole synthesis: {e}")
        return None


def enumerate_smarts_products(smiles1: str, smiles2: str, smarts: str) -> list:
    """Distinct products in RunReactants order (not sorted, so k stays stable)."""
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if rxn is None or not mol1 or not mol2:
            return []

        found = []
        for product_set in rxn.RunReactants((mol1, mol2)):
            if not product_set:
                continue
            candidate = product_set[0]
            try:
                Chem.SanitizeMol(candidate)
            except Exception:
                continue
            smiles = Chem.MolToSmiles(candidate)
            if smiles not in found:
                found.append(smiles)
        return found

    except Exception as e:
        bt.logging.error(f"Error in SMARTS reaction: {e}")
        return []


def perform_smarts_reaction(smiles1: str, smiles2: str, smarts: str, k: int = 0) -> str:
    """Perform SMARTS-based reaction, returning the k-th distinct product."""
    products = enumerate_smarts_products(smiles1, smiles2, smarts)
    if not products or k >= len(products):
        return None
    return products[k]


def validate_and_order_reactants(smiles1: str, smiles2: str, role_mask1: int, role_mask2: int, roleA: int, roleB: int,
                                smiles3: str = None, role_mask3: int = None, roleC: int = None) -> tuple:
    """Validate reactants can react and return in correct order."""
    try:
        if smiles3 is None:
            can_react = (_has_role(role_mask1, roleA) and _has_role(role_mask2, roleB)) or \
                        (_has_role(role_mask1, roleB) and _has_role(role_mask2, roleA))
            if not can_react:
                return None, None

            if _has_role(role_mask1, roleA) and _has_role(role_mask2, roleB):
                return smiles1, smiles2
            else:
                return smiles2, smiles1

        else:
            can_react_12 = (_has_role(role_mask1, roleA) and _has_role(role_mask2, roleB)) or \
                            (_has_role(role_mask1, roleB) and _has_role(role_mask2, roleA))
            can_react_3 = _has_role(role_mask3, roleC)

            if not can_react_12 or not can_react_3:
                return None, None, None

            if _has_role(role_mask1, roleA) and _has_role(role_mask2, roleB):
                return smiles1, smiles2, smiles3
            else:
                return smiles2, smiles1, smiles3

    except Exception as e:
        bt.logging.error(f"Error validating reactants: {e}")
        return (None, None) if smiles3 is None else (None, None, None)


def react_molecules(rxn_id: int, mol1_id: int, mol2_id: int, db_path: str, k: int = 0) -> str:
    try:
        reaction_info = get_reaction_info(rxn_id, db_path)
        molecules = get_molecules([mol1_id, mol2_id], db_path)

        if not reaction_info or not all(molecules):
            return None

        smarts, roleA, roleB, roleC = reaction_info
        (smiles1, role_mask1), (smiles2, role_mask2) = molecules

        reactant1, reactant2 = validate_and_order_reactants(smiles1, smiles2, role_mask1, role_mask2, roleA, roleB)
        if not reactant1 or not reactant2:
            return None

        if int(rxn_id) == 1:  # Triazole synthesis
            return combine_triazole_synthons(reactant1, reactant2) if k == 0 else None
        else:  # SMARTS-based reactions
            return perform_smarts_reaction(reactant1, reactant2, smarts, k)

    except Exception as e:
        bt.logging.error(f"Error reacting molecules {mol1_id}, {mol2_id}: {e}")
        return None


def react_three_components(rxn_id: int, mol1_id: int, mol2_id: int, mol3_id: int, db_path: str, k: int = 0) -> str:
    try:
        reaction_info = get_reaction_info(rxn_id, db_path)
        molecules = get_molecules([mol1_id, mol2_id, mol3_id], db_path)

        if not reaction_info or not all(molecules):
            return None

        smarts, roleA, roleB, roleC = reaction_info
        (smiles1, role_mask1), (smiles2, role_mask2), (smiles3, role_mask3) = molecules

        validation_result = validate_and_order_reactants(smiles1, smiles2, role_mask1, role_mask2, roleA, roleB,
                                                        smiles3, role_mask3, roleC)
        if not all(validation_result):
            return None

        reactant1, reactant2, reactant3 = validation_result

        if int(rxn_id) == 3:  # click_amide_cascade
            triazole_cooh = combine_triazole_synthons(reactant1, reactant2)
            if not triazole_cooh:
                return None

            amide_smarts = "[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]"
            return perform_smarts_reaction(triazole_cooh, reactant3, amide_smarts, k)

        if int(rxn_id) == 5:  # suzuki_bromide_then_chloride (two-step cascade)
            suzuki_br_smarts = "[#6:1][Br].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"
            suzuki_cl_smarts = "[#6:1][Cl].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"

            intermediate = perform_smarts_reaction(reactant1, reactant2, suzuki_br_smarts)
            if not intermediate:
                return None

            return perform_smarts_reaction(intermediate, reactant3, suzuki_cl_smarts, k)

        return None

    except Exception as e:
        bt.logging.error(f"Error in 3-component reaction {mol1_id}, {mol2_id}, {mol3_id}: {e}")
        return None


def parse_product_name(product_name: str) -> tuple:
    """rxn:<rxn>:<m1>:<m2>[:<m3>][:k<k>] -> (rxn_id, [mol_ids], k), k defaults to 0."""
    parts = product_name.split(":")
    if len(parts) < 4 or parts[0] != "rxn":
        return None

    k = 0
    if parts[-1][:1] == "k":
        if not parts[-1][1:].isdigit():
            return None
        k = int(parts[-1][1:])
        parts = parts[:-1]

    if len(parts) not in (4, 5):
        return None
    try:
        return int(parts[1]), [int(p) for p in parts[2:]], k
    except ValueError:
        return None


def get_smiles_from_reaction(product_name):
    """Resolve a molecule id to its product SMILES."""
    try:
        parsed = parse_product_name(product_name)
        if parsed is None:
            bt.logging.error(f"Invalid reaction format: {product_name}")
            return None

        rxn_id, mol_ids, k = parsed
        db_path = os.path.join(os.path.dirname(__file__), "molecules.sqlite")

        if len(mol_ids) == 2:
            return react_molecules(rxn_id, mol_ids[0], mol_ids[1], db_path, k)
        return react_three_components(rxn_id, mol_ids[0], mol_ids[1], mol_ids[2], db_path, k)

    except Exception as e:
        bt.logging.error(f"Error in combinatorial reaction {product_name}: {e}")
        return None
