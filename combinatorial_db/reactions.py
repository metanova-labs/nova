import sqlite3
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import bittensor as bt

def get_reaction_info(rxn_id: int, db_path: str) -> tuple:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT smarts, roleA, roleB, roleC FROM reactions WHERE rxn_id = ?", (rxn_id,))
        result = cursor.fetchone()
        conn.close()
        return result
    except Exception as e:
        bt.logging.error(f"Error getting reaction info: {e}")
        return None


def get_molecules(mol_ids: list, db_path: str) -> list:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        molecules = []
        for mol_id in mol_ids:
            cursor.execute("SELECT smiles, role_mask FROM molecules WHERE mol_id = ?", (mol_id,))
            result = cursor.fetchone()
            molecules.append(result)
        conn.close()
        return molecules
    except Exception as e:
        bt.logging.error(f"Error getting molecules: {e}")
        return [None] * len(mol_ids)
    
def combine_triazole_synthons(azide_smiles: str, alkyne_smiles: str) -> str:
    """Combine azide and alkyne synthons to form triazole."""
    try:
        m1 = Chem.RWMol(Chem.MolFromSmiles(azide_smiles))  # azide with [1*]
        m2 = Chem.RWMol(Chem.MolFromSmiles(alkyne_smiles))  # alkyne with [2*]
        
        if not m1 or not m2:
            return None
        
        # Find attachment points
        a1 = next((i for i, atom in enumerate(m1.GetAtoms()) if atom.GetSymbol() == '*' and atom.GetIsotope() == 1), None)
        a2 = next((i for i, atom in enumerate(m2.GetAtoms()) if atom.GetSymbol() == '*' and atom.GetIsotope() == 2), None)
        
        if a1 is None or a2 is None:
            return None
        
        # Get neighbors of attachment points
        n1 = m1.GetAtomWithIdx(a1).GetNeighbors()[0].GetIdx()
        n2 = m2.GetAtomWithIdx(a2).GetNeighbors()[0].GetIdx()
        
        # Create combined molecule
        combined = Chem.RWMol(m1)
        atom_mapping = {}
        
        # Add alkyne atoms (except attachment point)
        for i, atom in enumerate(m2.GetAtoms()):
            if i != a2:
                atom_mapping[i] = combined.AddAtom(atom)
        
        # Add alkyne bonds (except those involving attachment point)
        for bond in m2.GetBonds():
            begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a2 not in (begin_idx, end_idx):
                combined.AddBond(atom_mapping[begin_idx], atom_mapping[end_idx], bond.GetBondType())
        
        # Remove azide attachment point and connect
        combined.RemoveAtom(a1)
        n1_adj = n1 - (1 if n1 > a1 else 0)
        n2_adj = atom_mapping[n2] - (1 if atom_mapping[n2] > a1 else 0)
        combined.AddBond(n1_adj, n2_adj, Chem.BondType.SINGLE)
        
        Chem.SanitizeMol(combined)
        return Chem.MolToSmiles(combined)
        
    except Exception as e:
        bt.logging.error(f"Error in triazole synthesis: {e}")
        return None


def perform_smarts_reaction(smiles1: str, smiles2: str, smarts: str) -> str:
    """Perform SMARTS-based reaction."""
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if not mol1 or not mol2:
            return None
        
        products = rxn.RunReactants((mol1, mol2))
        return Chem.MolToSmiles(products[0][0]) if products else None
        
    except Exception as e:
        bt.logging.error(f"Error in SMARTS reaction: {e}")
        return None


def validate_and_order_reactants(smiles1: str, smiles2: str, role_mask1: int, role_mask2: int, roleA: int, roleB: int, 
                                smiles3: str = None, role_mask3: int = None, roleC: int = None) -> tuple:
    """Validate reactants can react and return in correct order."""
    try:
        if smiles3 is None:
            # Require all bits in the role to be present in the molecule's role_mask
            can_react = (((role_mask1 & roleA) == roleA) and ((role_mask2 & roleB) == roleB)) or \
                        (((role_mask1 & roleB) == roleB) and ((role_mask2 & roleA) == roleA))
            if not can_react:
                return None, None
            
            # Order reactants based on roles
            if ((role_mask1 & roleA) == roleA) and ((role_mask2 & roleB) == roleB):
                return smiles1, smiles2
            else:
                return smiles2, smiles1
        
        else:
            # Check if first two molecules are valid for their roles
            can_react_12 = (((role_mask1 & roleA) == roleA) and ((role_mask2 & roleB) == roleB)) or \
                            (((role_mask1 & roleB) == roleB) and ((role_mask2 & roleA) == roleA))
            can_react_3 = (role_mask3 & roleC) == roleC
            
            if not can_react_12 or not can_react_3:
                return None, None, None
            
            # Order first two reactants based on roles
            if (role_mask1 & roleA) and (role_mask2 & roleB):
                return smiles1, smiles2, smiles3
            else:
                return smiles2, smiles1, smiles3
            
    except Exception as e:
        bt.logging.error(f"Error validating reactants: {e}")
        return (None, None) if smiles3 is None else (None, None, None)

def react_molecules(rxn_id: int, mol1_id: int, mol2_id: int, db_path: str) -> str:
    try:
        # Get reaction info and molecules
        reaction_info = get_reaction_info(rxn_id, db_path)
        molecules = get_molecules([mol1_id, mol2_id], db_path)
        
        if not reaction_info or not all(molecules):
            return None
            
        smarts, roleA, roleB, roleC = reaction_info
        (smiles1, role_mask1), (smiles2, role_mask2) = molecules
        
        reactant1, reactant2 = validate_and_order_reactants(smiles1, smiles2, role_mask1, role_mask2, roleA, roleB)
        if not reactant1 or not reactant2:
            return None
            
        if rxn_id == 1:  # Triazole synthesis
            return combine_triazole_synthons(reactant1, reactant2)
        else:  # SMARTS-based reactions
            return perform_smarts_reaction(reactant1, reactant2, smarts)
        
    except Exception as e:
        bt.logging.error(f"Error reacting molecules {mol1_id}, {mol2_id}: {e}")
        return None


def react_three_components(rxn_id: int, mol1_id: int, mol2_id: int, mol3_id: int, db_path: str) -> str:
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
        
        if rxn_id == 3:  # click_amide_cascade
            # Triazole formation
            triazole_cooh = combine_triazole_synthons(reactant1, reactant2)
            if not triazole_cooh:
                return None
            
            # Amide coupling
            amide_smarts = "[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]"
            return perform_smarts_reaction(triazole_cooh, reactant3, amide_smarts)
        
        if rxn_id == 5:  # suzuki_bromide_then_chloride (two-step cascade)
            suzuki_br_smarts = "[#6:1][Br].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"
            suzuki_cl_smarts = "[#6:1][Cl].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"

            # First couple at bromide
            intermediate = perform_smarts_reaction(reactant1, reactant2, suzuki_br_smarts)
            if not intermediate:
                return None

            # Then couple at chloride
            final_product = perform_smarts_reaction(intermediate, reactant3, suzuki_cl_smarts)
            return final_product
        
        return None
        
    except Exception as e:
        bt.logging.error(f"Error in 3-component reaction {mol1_id}, {mol2_id}, {mol3_id}: {e}")
        return None


def get_smiles_from_reaction(product_name):
    """Handle reaction format: rxn:reaction_id:mol1_id:mol2_id or rxn:reaction_id:mol1_id:mol2_id:mol3_id"""
    try:
        parts = product_name.split(":")
        if len(parts) == 4:
            _, rxn_id, mol1_id, mol2_id = parts
            rxn_id, mol1_id, mol2_id = int(rxn_id), int(mol1_id), int(mol2_id)
            
            db_path = os.path.join(os.path.dirname(__file__), "molecules.sqlite")
            return react_molecules(rxn_id, mol1_id, mol2_id, db_path)
            
        elif len(parts) == 5:
            _, rxn_id, mol1_id, mol2_id, mol3_id = parts
            rxn_id, mol1_id, mol2_id, mol3_id = int(rxn_id), int(mol1_id), int(mol2_id), int(mol3_id)
            
            db_path = os.path.join(os.path.dirname(__file__), "molecules.sqlite")
            return react_three_components(rxn_id, mol1_id, mol2_id, mol3_id, db_path)
            
        else:
            bt.logging.error(f"Invalid reaction format: {product_name}")
            return None
        
    except Exception as e:
        bt.logging.error(f"Error in combinatorial reaction {product_name}: {e}")
        return None 