"""
Final scoring and winner determination functionality for the validator
"""

import math
import datetime
from typing import Optional

import bittensor as bt
from utils import calculate_dynamic_entropy


def calculate_final_scores(
    score_dict: dict[int, dict[str, list[list[float]]]],
    valid_molecules_by_uid: dict[int, dict[str, list[str]]],
    molecule_name_counts: dict[str, int],
    config: dict,
    current_epoch: int
) -> dict[int, dict[str, list[list[float]]]]:
    """
    Calculates final scores per molecule for each UID, considering target and antitarget scores.
    Applies entropy bonus and tie-breaking by earliest submission block.
    
    Args:
        score_dict: Dictionary containing scores for each UID
        valid_molecules_by_uid: Dictionary of valid molecules by UID
        molecule_name_counts: Count of molecule name occurrences
        config: Configuration dictionary
        current_epoch: Current epoch number
        
    Returns:
        Updated score_dict with final scores calculated
    """
    
    dynamic_entropy_weight = calculate_dynamic_entropy(
        starting_weight=config['entropy_start_weight'],
        step_size=config['entropy_step_size'],
        start_epoch=config['entropy_start_epoch'],
        current_epoch=current_epoch
    )
    
    # Go through each UID scored
    for uid, data in valid_molecules_by_uid.items():
        targets = score_dict[uid]['target_scores']
        antitargets = score_dict[uid]['antitarget_scores']
        entropy = score_dict[uid]['entropy']
        submission_block = score_dict[uid]['block_submitted']

        # Replace None with -inf
        targets = [[-math.inf if not s else s for s in sublist] for sublist in targets]
        antitargets = [[-math.inf if not s else s for s in sublist] for sublist in antitargets]

        # Get number of molecules (length of any target score list)
        if not targets or not targets[0]:
            continue
        num_molecules = len(targets[0])

        # Calculate scores per molecule
        combined_molecule_scores = []
        molecule_scores_after_repetition = []
        
        for mol_idx in range(num_molecules):
            # Calculate average target score for this molecule
            target_scores_for_mol = [target_list[mol_idx] for target_list in targets]
            if any(score == -math.inf for score in target_scores_for_mol):
                combined_molecule_scores.append(-math.inf)
                molecule_scores_after_repetition.append(-math.inf)
                continue
            avg_target = sum(target_scores_for_mol) / len(target_scores_for_mol)

            # Calculate average antitarget score for this molecule
            antitarget_scores_for_mol = [antitarget_list[mol_idx] for antitarget_list in antitargets]
            if any(score == -math.inf for score in antitarget_scores_for_mol):
                combined_molecule_scores.append(-math.inf)
                molecule_scores_after_repetition.append(-math.inf)
                continue
            avg_antitarget = sum(antitarget_scores_for_mol) / len(antitarget_scores_for_mol)

            # Calculate score after target/antitarget combination
            mol_score = avg_target - (config['antitarget_weight'] * avg_antitarget)
            combined_molecule_scores.append(mol_score)

            # Calculate molecule repetition penalty
            if config['molecule_repetition_weight'] != 0:
                if mol_score > config['molecule_repetition_threshold']:
                    denominator = config['molecule_repetition_weight'] * molecule_name_counts[data['names'][mol_idx]]
                    if denominator == 0:
                        mol_score = mol_score  
                    else:
                        mol_score = mol_score / denominator
                else:
                    mol_score = mol_score * config['molecule_repetition_weight'] * molecule_name_counts[data['names'][mol_idx]]
            
            molecule_scores_after_repetition.append(mol_score)
        
        # Store all score lists in score_dict
        score_dict[uid]['combined_molecule_scores'] = combined_molecule_scores
        score_dict[uid]['molecule_scores_after_repetition'] = molecule_scores_after_repetition
        score_dict[uid]['final_score'] = sum(molecule_scores_after_repetition)
                
        # Apply entropy bonus for scores above threshold - disabled while num_molecules is 1
        # if score_dict[uid]['final_score'] > config['entropy_bonus_threshold'] and entropy is not None:
        #     score_dict[uid]['final_score'] = score_dict[uid]['final_score'] * (1 + (dynamic_entropy_weight * entropy))

        boltz_score = score_dict[uid]['boltz_score']
        entropy_boltz = score_dict[uid]['entropy_boltz']
        threshold_boltz = config.get('entropy_bonus_threshold')

        if (
            boltz_score is not None
            and entropy_boltz is not None
            and math.isfinite(boltz_score)
            and math.isfinite(entropy_boltz)
            and boltz_score > threshold_boltz
            and entropy_boltz > 0
            and config['num_molecules_boltz'] > 1
        ):
            score_dict[uid]['boltz_score'] = boltz_score * (1 + (dynamic_entropy_weight * entropy_boltz))

        # Log details
        # Prepare detailed log info
        smiles_list = data.get('smiles', [])
        names_list = data.get('names', [])
        # Transpose target/antitarget scores to get per-molecule lists
        target_scores_per_mol = list(map(list, zip(*targets))) if targets and targets[0] else []
        antitarget_scores_per_mol = list(map(list, zip(*antitargets))) if antitargets and antitargets[0] else []
        log_lines = [
            f"UID={uid}",
            f"  Molecule names: {names_list}",
            f"  SMILES: {smiles_list}",
            # f"  Target scores per molecule: {target_scores_per_mol}",
            # f"  Antitarget scores per molecule: {antitarget_scores_per_mol}",
            #f"  Entropy: {entropy}",
            f"  Boltz scores: {score_dict[uid]['boltz_score']}",
            #f"  Entropy Boltz: {score_dict[uid]['entropy_boltz'] if score_dict[uid]['entropy_boltz'] is not None else 'None'}",
            #f"  Dynamic entropy weight: {dynamic_entropy_weight}",
            #f"  Final score: {score_dict[uid]['final_score']}"
        ]
        bt.logging.info("\n".join(log_lines))

    return score_dict


def determine_winner(score_dict: dict[int, dict[str, list[list[float]]]]) -> Optional[int]:
    """
    Determines the winning UID based on final score.
    In case of ties, earliest submission time is used as the tiebreaker.
    
    Args:
        score_dict: Dictionary containing final scores for each UID
        
    Returns:
        Optional[int]: Winning UID or None if no valid scores found
    """
    best_score_psichic = -math.inf
    best_score_boltz = -math.inf
    best_uids_psichic = []
    best_uids_boltz = []

    def parse_timestamp(uid):
        ts = score_dict[uid].get('push_time', '')
        try:
            return datetime.datetime.fromisoformat(ts)
        except Exception as e:
            bt.logging.warning(f"Failed to parse timestamp '{ts}' for UID={uid}: {e}")
            return datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)

    def tie_breaker(tied_uids: list[int], best_score: float, model_name: str, print_message: bool = True):
        # Sort by block number first, then push time, then uid to ensure deterministic result
        winner = sorted(tied_uids, key=lambda uid: (
            score_dict[uid].get('block_submitted', float('inf')), 
            parse_timestamp(uid), 
            uid
        ))[0]
        
        winner_block = score_dict[winner].get('block_submitted')
        current_epoch = winner_block // 361 if winner_block else None
        push_time = score_dict[winner].get('push_time', '')
        
        tiebreaker_message = f"Epoch {current_epoch} tiebreaker {model_name} winner: UID={winner}, score={best_score}, block={winner_block}"
        if push_time:
            tiebreaker_message += f", push_time={push_time}"
            
        if print_message:
            bt.logging.info(tiebreaker_message)
            
        return winner
    
    # Find highest final score
    for uid, data in score_dict.items():
        if 'final_score' not in data or 'boltz_score' not in data:
            continue
        
        psichic_score = round(data['final_score'], 4)
        boltz_score = round(data['boltz_score'], 4)
        
        if psichic_score > best_score_psichic:
            best_score_psichic = psichic_score
            best_uids_psichic = [uid]
        elif psichic_score == best_score_psichic:
            best_uids_psichic.append(uid)

        if boltz_score > best_score_boltz:
            best_score_boltz = boltz_score
            best_uids_boltz = [uid]
        elif boltz_score == best_score_boltz:
            best_uids_boltz.append(uid)
    
    if not best_uids_psichic and not best_uids_boltz:
        bt.logging.info("No valid winner found (all scores -inf or no submissions).")
        return None, None

    # Treat all -inf as no valid winners for each model
    if best_score_psichic == -math.inf:
        best_uids_psichic = []
    if best_score_boltz == -math.inf:
        best_uids_boltz = []
    
    # Select winner from each model
    if best_uids_psichic:
        if len(best_uids_psichic) == 1:
            psichic_winner_block = score_dict[best_uids_psichic[0]].get('block_submitted')
            current_epoch = psichic_winner_block // 361 if psichic_winner_block else None
            #bt.logging.info(f"Epoch {current_epoch} PSICHIC winner: UID={best_uids_psichic[0]}, winning_score={best_score_psichic}")
            winner_psichic = best_uids_psichic[0]
        else:
            winner_psichic = tie_breaker(best_uids_psichic, best_score_psichic, "PSICHIC", print_message=False)
    else:
        winner_psichic = None
    
    if best_uids_boltz:
        if len(best_uids_boltz) == 1:
            boltz_winner_block = score_dict[best_uids_boltz[0]].get('block_submitted')
            current_epoch = boltz_winner_block // 361 if boltz_winner_block else None
            bt.logging.info(f"Epoch {current_epoch} BOLTZ winner: UID={best_uids_boltz[0]}, winning_score={best_score_boltz}")
            winner_boltz = best_uids_boltz[0]
        else:
            winner_boltz = tie_breaker(best_uids_boltz, best_score_boltz, "BOLTZ")
    else:
        winner_boltz = None
    
    return winner_psichic, winner_boltz
    
