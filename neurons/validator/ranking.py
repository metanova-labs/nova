import math
import datetime
from typing import Optional

import bittensor as bt


NANOBODY_TIEBREAK_CATEGORIES = ("confidence", "physical_interaction", "developability")


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

def _parse_timestamp(ts: str, uid: int) -> datetime.datetime:
    try:
        return datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except Exception as e:
        bt.logging.warning(f"Failed to parse timestamp '{ts}' for UID={uid}: {e}")
        return datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)


def _block_or_inf(block) -> float:
    """Treat missing/None block_submitted as +inf so it loses tie-breakers."""
    if block is None:
        return math.inf
    try:
        return float(block)
    except (TypeError, ValueError):
        return math.inf


def _config_get(config, key: str, default):
    """Read a key from either an attribute-style config or a plain dict."""
    val = getattr(config, key, None)
    if val is not None:
        return val
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def _nanobody_category_score(
    per_nanobody_components: Optional[dict],
    uid: int,
    category: str,
) -> float:
    """
    Sum {category}_rank_sum across every (sequence, target) combination the UID
    submitted, lower is better

    Returns +inf if the components are missing so the UID loses category-based
    tie-breakers gracefully.
    """
    if not per_nanobody_components:
        return math.inf

    uid_components = per_nanobody_components.get(uid)
    if not uid_components:
        return math.inf

    metric_key = f"{category}_rank_sum"
    total = 0.0
    found_any = False
    for seq_components in uid_components.values():
        for target_components in seq_components.values():
            value = target_components.get(metric_key)
            if value is None:
                continue
            try:
                total += float(value)
                found_any = True
            except (TypeError, ValueError):
                continue

    return total if found_any else math.inf


def rank_uids(
    score_dict: dict[int, dict[str, list[list[float]]]],
    config,
    item_type: str,
    per_nanobody_components: Optional[dict] = None,
) -> dict[int, int]:
    """
    Rank UIDs that have a valid finite ``final_{item_type}_score``.

    Returns a dict ``{uid: rank}`` where ``rank == 1`` is the best. UIDs without
    a submission for ``item_type`` (or whose final score is missing/+/-inf) are
    omitted from the result.

    Tie-breaking:
      * molecules: ``block_submitted`` -> ``push_time`` -> ``uid``.
      * nanobodies: confidence rank -> physical interaction rank ->
        developability rank (each computed by summing ``{category}_rank_sum``
        from ``per_nanobody_components`` across the UID's seq/target pairs;
        lower is better) -> ``block_submitted`` -> ``push_time`` -> ``uid``.
    """
    if item_type == "molecule":
        mode = _config_get(config, 'boltz_mode', 'max')
    elif item_type == "nanobody":
        mode = _config_get(config, 'boltzgen_rank_mode', 'min')
    else:
        bt.logging.error(f"Invalid item type: {item_type}")
        return {}

    score_key = f"final_{item_type}_score"

    eligible: list[int] = []
    for uid, data in score_dict.items():
        if score_key not in data:
            continue
        score = data[score_key]
        if score is None:
            continue
        if score == math.inf or score == -math.inf:
            continue
        eligible.append(uid)

    if not eligible:
        bt.logging.info(f"No valid {item_type} submissions to rank.")
        return {}

    def sort_key(uid: int):
        data = score_dict[uid]
        score = data[score_key]
        primary = -score if mode == "max" else score

        if item_type == "nanobody":
            category_scores = tuple(
                _nanobody_category_score(per_nanobody_components, uid, cat)
                for cat in NANOBODY_TIEBREAK_CATEGORIES
            )
        else:
            category_scores = ()

        return (
            primary,
            *category_scores,
            _block_or_inf(data.get('block_submitted')),
            _parse_timestamp(data.get('push_time', '') or '', uid),
            uid,
        )

    ordered = sorted(eligible, key=sort_key)
    ranks = {uid: idx + 1 for idx, uid in enumerate(ordered)}

    winner = ordered[0]
    winner_data = score_dict[winner]
    winner_block = winner_data.get('block_submitted')
    winner_score = winner_data[score_key]
    epoch_length = _config_get(config, 'epoch_length', 361)
    current_epoch = winner_block // epoch_length if isinstance(winner_block, int) else None
    bt.logging.info(
        f"Epoch {current_epoch} {item_type} ranking: winner UID={winner}, "
        f"winning_score={winner_score}, block={winner_block}, "
        f"ranked_uids={len(ranks)}"
    )

    return ranks

