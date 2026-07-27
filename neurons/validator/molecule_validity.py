import bittensor as bt
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from utils import (
    get_smiles, 
    get_heavy_atom_count, 
    compute_maccs_entropy,
    entry_unique_for_protein_hf,
    find_chemically_identical,
    is_reaction_allowed,
    contains_atom_type,
    get_historical_submissions
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

    # index historical submissions by target (once per epoch)
    historical_submissions = {target: get_historical_submissions(target, 'molecules') for target in config['small_molecule_target']}
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048
        )
    for target, historical_submissions_df in historical_submissions.items():
        if historical_submissions_df is not None:
            mols = [Chem.MolFromSmiles(smi) for smi in historical_submissions_df["SMILES"]]
            fps = morgan_gen.GetFingerprints(
                    mols,
                    numThreads=8,
                )
            historical_submissions_df["fingerprint"] = list(fps)

    valid_molecules_by_uid = {}
    
    for uid, data in uid_to_data.items():
        valid_smiles = []
        valid_names = []

        if any("~" in molecule for molecule in data["molecules"]):
            continue
        
        # Check for duplicate molecules in submission
        if len(data["molecules"]) != len(set(data["molecules"])):
            bt.logging.warning(f"UID={uid}: submission contains duplicate molecules")
            continue
            
        for molecule in data["molecules"]:
            try:
                # Check if reaction is allowed this epoch (if filtering enabled)
                if config.get('random_valid_reaction') and not is_reaction_allowed(molecule, allowed_reaction):
                    bt.logging.warning(
                        f"UID={uid}: molecule='{molecule}' uses disallowed reaction for this epoch (only {allowed_reaction} allowed)"
                    )
                    break

                elif not config.get('random_valid_reaction'):
                    allowed_ok = [is_reaction_allowed(molecule, reaction) for reaction in config.get('allowed_reactions')]
                    if len(allowed_ok) == 0:
                        bt.logging.error(f"random_valid_reaction is false but no allowed reactions are specified. Check config/config.yaml")
                        break
                    if not any(allowed_ok):
                        bt.logging.warning(f"UID={uid}: molecule='{molecule}' uses disallowed reaction for this epoch (only {config.get('allowed_reactions')} allowed)")
                        break
                
                smiles = get_smiles(molecule)
                if not smiles:
                    bt.logging.warning(f"UID={uid}: no valid SMILES found for molecule='{molecule}'")
                    break
                
                if get_heavy_atom_count(smiles) < config['min_heavy_atoms']:
                    bt.logging.warning(f"UID={uid}: molecule='{molecule}' has insufficient heavy atoms")
                    break

                try:
                    mol = Chem.MolFromSmiles(smiles)

                    if contains_atom_type(mol, config['banned_atom_types']):
                        bt.logging.warning(f"UID={uid}: molecule='{molecule}' contains banned atom types")
                        break

                    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                    if num_rotatable_bonds < config['min_rotatable_bonds'] or num_rotatable_bonds > config['max_rotatable_bonds']:
                        bt.logging.warning(f"UID={uid}: molecule='{molecule}' has an invalid number of rotatable bonds")
                        break
                except Exception as e:
                    bt.logging.warning(f"UID={uid}: molecule='{molecule}' is not parseable by RDKit: {e}")
                    break
                
                # Check if the molecule is unique for all target proteins
                is_unique = True
                for target in config['small_molecule_target']:
                    if not entry_unique_for_protein_hf(target, smiles, 'molecules'):
                        bt.logging.warning(f"UID={uid}: molecule='{molecule}' is not unique for protein '{target}'")
                        is_unique = False
                        break
                
                if not is_unique:
                    break
                
                # Check if the molecule is diverse enough compared to historical submissions
                pass_diversity = True
                miner_mol_fp = morgan_gen.GetFingerprint(mol)
                for target in config['small_molecule_target']:
                    if historical_submissions[target] is not None:
                        similarities = DataStructs.BulkTanimotoSimilarity(
                                        miner_mol_fp,
                                        historical_submissions[target]["fingerprint"],
                                    )
                        pass_ = not any(sim >= config['max_similarity_to_historical'] for sim in similarities)
                        if not pass_:
                            bt.logging.warning(f"UID={uid}: molecule='{molecule}' is not diverse enough compared to historical submissions for target '{target}'")
                            pass_diversity = False
                            break
                    else:
                        bt.logging.warning(f"UID={uid}: no historical submissions found for target '{target}'")
                        continue
                if not pass_diversity:
                    continue
    
                valid_smiles.append(smiles)
                valid_names.append(molecule)
            except Exception as e:
                bt.logging.warning(f"UID={uid}: error validating molecule='{molecule}': {e}")
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
                    bt.logging.warning(f"UID={uid}: submission contains chemically identical molecules: {'; '.join(duplicate_names)}")
                    continue 
            except Exception as e:
                bt.logging.warning(f"UID={uid}: error checking for chemically identical molecules: {e}")
                continue

        # Calculate entropy if we have valid molecules
        if valid_smiles:
            if config['num_molecules'] > 1:
                try:
                    entropy = compute_maccs_entropy(valid_smiles)
                    score_dict[uid]["entropy"] = entropy
                    valid_molecules_by_uid[uid] = {"smiles": valid_smiles, "names": valid_names}
                except Exception as e:
                    bt.logging.warning(f"UID={uid}: error calculating entropy: {e}")
                    continue
            else:
                score_dict[uid]["entropy"] = None
                valid_molecules_by_uid[uid] = {"smiles": valid_smiles, "names": valid_names}
            
    return valid_molecules_by_uid
    
