import math
import datetime
from typing import Optional

import bittensor as bt


def calculate_scores_for_type(
    score_dict: dict[int, dict[str, list[list[float]]]],
    valid_items_by_uid: dict[int, dict[str, list[str]]],
    item_type: str,
    config: dict,
) -> dict[int, dict[str, list[list[float]]]]:
    """
    Calculates scores for a given item type for each UID.
    """
    
    if item_type == "molecule":
        num_items = config['num_molecules']
        score_key = 'molecule_scores'
        replace_value = -math.inf if config['boltz_mode'] == 'max' else math.inf
        id_key = 'names'
        item_key = 'smiles'
    elif item_type == "nanobody":
        num_items = config['num_sequences']
        score_key = 'nanobody_scores'
        replace_value = -math.inf if config['boltzgen_rank_mode'] == 'max' else math.inf
        id_key = 'hashes'
        item_key = 'sequences'
    else:
        bt.logging.error(f"Invalid item type: {item_type}")
        return None

    for uid, data in valid_items_by_uid.items():
        if uid not in score_dict:
            bt.logging.error(f"UID={uid}: not found in score_dict. Skipping.")
            continue
        if score_key not in score_dict[uid]:
            bt.logging.error(f"UID={uid}: {score_key} not found in score_dict. Skipping.")
            continue
        
        targets = score_dict[uid][score_key]
        block_submitted = score_dict[uid]['block_submitted']

        targets = [[replace_value if not s else s for s in sublist] for sublist in targets]
        combined_item_scores = []
        
        for item_idx in range(num_items):
            # Calculate average target score for this molecule
            target_scores_for_item = [target_list[item_idx] for target_list in targets]
            if any(score == -math.inf or score == math.inf for score in target_scores_for_item):
                combined_item_scores.append(replace_value)
                continue
            avg_target = sum(target_scores_for_item) / len(target_scores_for_item)
            combined_item_scores.append(avg_target)

        score_dict[uid][f"combined_{item_type}_scores"] = combined_item_scores
        score_dict[uid][f"final_{item_type}_score"] = sum(combined_item_scores)

        id_list = data.get(id_key, [])
        item_list = data.get(item_key, [])
        log_lines = [
            f"UID={uid}",
            f"  {item_type} {id_key}: {id_list}",
            f"  {item_type} {item_key}: {item_list}",
        ]
        if num_items > 1:
            log_lines.append(f"  {item_type} scores: {combined_item_scores}")
            log_lines.append(f"  {item_type} final score: {score_dict[uid][f'final_{item_type}_score']}")
        else:
            log_lines.append(f"  final {item_type} score: {score_dict[uid][f'final_{item_type}_score']}")
        bt.logging.info("\n".join(log_lines))

    return score_dict

def determine_winner(score_dict: dict[int, dict[str, list[list[float]]]], config: dict, item_type: str) -> Optional[int]:
    """
    Determines the winning UID based on final score for a given item type.
    In case of ties, earliest submission time is used as the tiebreaker.
    
    Args:
        score_dict: Dictionary containing final scores for each UID
        config: subnet config dict
        type: item type to determine winner for (molecule or nanobody)
    Returns:
        Optional[int]: Winning UID or None if no valid scores found
    """

    if item_type == "molecule":
        mode = getattr(config, 'boltz_mode', None) or (config.get('boltz_mode', 'max') if isinstance(config, dict) else 'max')
    elif item_type == "nanobody":
        mode = getattr(config, 'boltzgen_rank_mode', None) or (config.get('boltzgen_rank_mode', 'min') if isinstance(config, dict) else 'min')
    else:
        bt.logging.error(f"Invalid item type: {item_type}")
        return None

    best_uids = []
    best_score = -math.inf if mode == "max" else math.inf

    def parse_timestamp(uid):
        ts = score_dict[uid].get('push_time', '')
        try:
            return datetime.datetime.fromisoformat(ts)
        except Exception as e:
            bt.logging.warning(f"Failed to parse timestamp '{ts}' for UID={uid}: {e}")
            return datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)

    def tie_breaker(tied_uids: list[int], best_score: float, item_type: str, print_message: bool = True):
        # Sort by block number first, then push time, then uid to ensure deterministic result
        winner = sorted(tied_uids, key=lambda uid: (
            score_dict[uid].get('block_submitted', float('inf')), 
            parse_timestamp(uid), 
            uid
        ))[0]
        
        winner_block = score_dict[winner].get('block_submitted')
        current_epoch = winner_block // 361 if winner_block else None
        push_time = score_dict[winner].get('push_time', '')
        
        tiebreaker_message = f"Epoch {current_epoch} {item_type} tiebreaker winner: UID={winner}, score={best_score}, block={winner_block}"
        if push_time:
            tiebreaker_message += f", push_time={push_time}"
            
        if print_message:
            bt.logging.info(tiebreaker_message)
            
        return winner
    
    # Find highest final score
    for uid, data in score_dict.items():
        if f"final_{item_type}_score" not in data:
            continue
        score = data[f'final_{item_type}_score']

        if mode == "max":
            if score > best_score:
                best_score = score
                best_uids = [uid]
            elif score == best_score:
                best_uids.append(uid)
        elif mode == "min":
            if score < best_score:
                best_score = score
                best_uids = [uid]
            elif score == best_score:
                best_uids.append(uid)
    
    if not best_uids:
        bt.logging.debug(f"score_dict: {score_dict}")
        bt.logging.info(f"No valid {item_type} winner found (all scores -inf/inf or no submissions).")
        return None

    # Treat all -inf or inf as no valid winners
    if best_score == -math.inf or best_score == math.inf:
        return None
    
    # Select winner
    if best_uids:
        if len(best_uids) == 1:
            winner_block = score_dict[best_uids[0]].get('block_submitted')
            current_epoch = winner_block // 361 if winner_block else None
            bt.logging.info(f"Epoch {current_epoch} {item_type} winner: UID={best_uids[0]}, winning_score={best_score}")
            winner = best_uids[0]
        else:
            winner = tie_breaker(best_uids, best_score, item_type, print_message=True)
    else:
        winner = None
    
    return winner
    
