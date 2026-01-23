import bittensor as bt
from rdkit import Chem
from rdkit.Chem import Descriptors

from utils import (
    get_smiles, 
    get_heavy_atom_count, 
    compute_maccs_entropy,
    entry_unique_for_protein_hf,
    find_chemically_identical,
    is_reaction_allowed
)


def validate_molecules_and_calculate_entropy(
    uid_to_data: dict[int, dict[str, list]],
    score_dict: dict[int, dict[str, list[list[float]]]],
    config: dict,
    allowed_reaction: str = None
) -> dict[int, dict[str, list[str]]]:
    """
    Validates molecules for all UIDs and calculates their MACCS entropy.
    Updates the score_dict with entropy values.
    
    Args:
        uid_to_data: Dictionary mapping UIDs to their data including molecules
        score_dict: Dictionary to store scores and entropy
        config: Configuration dictionary containing validation parameters
        allowed_reaction: Optional allowed reaction filter for this epoch
        
    Returns:
        Dictionary mapping UIDs to their list of valid SMILES strings
    """
    valid_molecules_by_uid = {}
    
    for uid, data in uid_to_data.items():
        valid_smiles = []
        valid_names = []

        if any("~" in molecule for molecule in data["molecules"]):
            continue
        
        # Check for duplicate molecules in submission
        if len(data["molecules"]) != len(set(data["molecules"])):
            bt.logging.error(f"UID={uid} submission contains duplicate molecules")
            continue
            
        for molecule in data["molecules"]:
            try:
                # Check if reaction is allowed this epoch (if filtering enabled)
                if config.get('random_valid_reaction') and not is_reaction_allowed(molecule, allowed_reaction):
                    bt.logging.warning(
                        f"UID={uid}, molecule='{molecule}' uses disallowed reaction for this epoch (only {allowed_reaction} allowed)"
                    )
                    break

                # temporary: Always allow reactions 1 and 2, ignore config/random selection
                # allowed_ok = is_reaction_allowed(molecule, "rxn:1") or is_reaction_allowed(molecule, "rxn:3")
                # if not allowed_ok:
                #     bt.logging.warning(
                #         f"UID={uid}, molecule='{molecule}' uses disallowed reaction for this temporary window (only 1 or 3 allowed)"
                #     )
                #     break
                
                smiles = get_smiles(molecule)
                if not smiles:
                    bt.logging.error(f"No valid SMILES found for UID={uid}, molecule='{molecule}'")
                    break
                
                if get_heavy_atom_count(smiles) < config['min_heavy_atoms']:
                    bt.logging.warning(f"UID={uid}, molecule='{molecule}' has insufficient heavy atoms")
                    break

                try:
                    mol = Chem.MolFromSmiles(smiles)
                    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                    if num_rotatable_bonds < config['min_rotatable_bonds'] or num_rotatable_bonds > config['max_rotatable_bonds']:
                        bt.logging.warning(f"UID={uid}, molecule='{molecule}' has an invalid number of rotatable bonds")
                        break
                except Exception as e:
                    bt.logging.error(f"Molecule is not parseable by RDKit for UID={uid}, molecule='{molecule}': {e}")
                    break
                
                # Check if the molecule is unique for all target proteins
                is_unique = True
                for target in config['small_molecule_target']:
                    if not entry_unique_for_protein_hf(target, smiles, 'molecules'):
                        bt.logging.warning(f"UID={uid}, molecule='{molecule}' is not unique for protein '{target}'")
                        is_unique = False
                        break
                
                if not is_unique:
                    break
     
                valid_smiles.append(smiles)
                valid_names.append(molecule)
            except Exception as e:
                bt.logging.error(f"Error validating molecule for UID={uid}, molecule='{molecule}': {e}")
                break
            
        # Check for chemically identical molecules
        if valid_smiles:
            try:
                identical_molecules = find_chemically_identical(valid_smiles)
                if identical_molecules:
                    duplicate_names = []
                    for inchikey, indices in identical_molecules.items():
                        molecule_names = [valid_names[idx] for idx in indices]
                        duplicate_names.append(f"{', '.join(molecule_names)} (same InChIKey: {inchikey})")
                    bt.logging.warning(f"UID={uid} submission contains chemically identical molecules: {'; '.join(duplicate_names)}")
                    continue 
            except Exception as e:
                bt.logging.warning(f"Error checking for chemically identical molecules for UID={uid}: {e}")
                continue

        # Calculate entropy if we have valid molecules
        if valid_smiles:
            if config['num_molecules'] > 1:
                try:
                    entropy = compute_maccs_entropy(valid_smiles)
                    score_dict[uid]["entropy"] = entropy
                    valid_molecules_by_uid[uid] = {"smiles": valid_smiles, "names": valid_names}
                    score_dict[uid]["block_submitted"] = data["block_submitted"]               
                except Exception as e:
                    bt.logging.error(f"Error calculating entropy for UID={uid}: {e}")
                    continue
            else:
                score_dict[uid]["entropy"] = None
                score_dict[uid]["block_submitted"] = data["block_submitted"]
                valid_molecules_by_uid[uid] = {"smiles": valid_smiles, "names": valid_names}
            
    return valid_molecules_by_uid
    
