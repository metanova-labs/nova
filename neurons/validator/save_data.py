import os
import math
import json
import aiohttp
import bittensor as bt

from utils import calculate_dynamic_entropy


def _safe_num(x: float) -> float:
    return -999.99 if x == -math.inf else x


def _get_dynamic_entropy_weight_for_epoch(config, current_epoch: int) -> float:
    start_weight = getattr(config, 'entropy_start_weight', 1.0)
    step_size = getattr(config, 'entropy_step_size', 0.0)
    start_epoch = getattr(config, 'entropy_start_epoch', 0)
    return calculate_dynamic_entropy(start_weight, step_size, start_epoch, current_epoch)


def _build_competition_payload(config, current_epoch: int, target_proteins: list[str], antitarget_proteins: list[str]) -> dict:
    epoch_number = current_epoch
    entropy_weight = _get_dynamic_entropy_weight_for_epoch(config, current_epoch)
    return {
        "epoch_number": epoch_number,
        "target_proteins": target_proteins,
        "antitarget_proteins": antitarget_proteins,
        "config": {
            "antitarget_weight": getattr(config, 'antitarget_weight', 1.0),
            "entropy_weight": entropy_weight,
            "entropy_bonus_threshold": getattr(config, 'entropy_bonus_threshold', 0.0),
            "molecule_repetition_weight": getattr(config, 'molecule_repetition_weight', 1.0),
            "molecule_repetition_threshold": getattr(config, 'molecule_repetition_threshold', 0.0),
            "min_heavy_atoms": getattr(config, 'min_heavy_atoms', 0),
            "num_molecules": getattr(config, 'num_molecules', 0),
            "min_rotatable_bonds": getattr(config, 'min_rotatable_bonds', 0),
            "max_rotatable_bonds": getattr(config, 'max_rotatable_bonds', 0),
            # Boltz config
            "boltz_weight": getattr(config, 'boltz_weight', None),
            "num_molecules_boltz": getattr(config, 'num_molecules_boltz', None),
            "boltz_metric": getattr(config, 'boltz_metric', None),
        },
    }


def _get_boltz_selected_map_for_uid(boltz, uid: int, smiles_list: list[str]) -> list[bool | None]:
    try:
        if not smiles_list:
            return []
        unique = getattr(boltz, 'unique_molecules', None)
        if not isinstance(unique, dict):
            return [ None for _ in smiles_list ]
        selected = set()
        for smiles, pairs in unique.items():
            if any(pair_uid == uid for pair_uid, _ in pairs):
                selected.add(smiles)
        return [ (s in selected) for s in smiles_list ]
    except Exception:
        return [ None for _ in smiles_list ]


def _build_submissions_payload(config, metagraph, boltz, current_block: int, start_block: int, uid_to_data: dict, valid_molecules_by_uid: dict, molecule_name_counts: dict, score_dict: dict) -> list[dict]:
    submissions = []
    for uid, data in uid_to_data.items():
        valid = valid_molecules_by_uid.get(uid, {})
        smiles_list = valid.get('smiles', [])
        names_list = valid.get('names', [])
        if not smiles_list:
            continue

        hotkey = data.get('hotkey') or (metagraph.hotkeys[uid] if uid < len(metagraph.hotkeys) else "unknown")
        coldkey = metagraph.coldkeys[uid] if hasattr(metagraph, 'coldkeys') and uid < len(metagraph.coldkeys) else "unknown"

        block_submitted = score_dict.get(uid, {}).get('block_submitted', data.get('block_submitted'))
        if isinstance(block_submitted, int):
            blocks_elapsed = block_submitted - start_block
        else:
            blocks_elapsed = (block_submitted or current_block) - start_block

        entropy = score_dict.get(uid, {}).get('entropy')
        boltz_score = score_dict.get(uid, {}).get('boltz_score')
        boltz_entropy = score_dict.get(uid, {}).get('entropy_boltz')
        final_score = score_dict.get(uid, {}).get('final_score', -math.inf)
        final_score_safe = _safe_num(final_score)

        targets = score_dict.get(uid, {}).get('target_scores', [])
        antitargets = score_dict.get(uid, {}).get('antitarget_scores', [])
        combined_scores = score_dict.get(uid, {}).get('combined_molecule_scores', [])
        scores_after_repetition = score_dict.get(uid, {}).get('molecule_scores_after_repetition', [])

        boltz_selected_flags = _get_boltz_selected_map_for_uid(boltz, uid, smiles_list)

        molecule_details = []
        for idx in range(len(smiles_list)):
            per_mol_boltz = None
            try:
                if getattr(boltz, 'per_molecule_metric', None):
                    if boltz_selected_flags and (idx < len(boltz_selected_flags)) and boltz_selected_flags[idx]:
                        per_mol_boltz = boltz.per_molecule_metric.get(uid, {}).get(smiles_list[idx])
            except Exception:
                per_mol_boltz = None

            molecule_details.append({
                "name": names_list[idx],
                "smiles": smiles_list[idx],
                "repetition_count": molecule_name_counts.get(names_list[idx], 1),
                "target_scores": [ _safe_num(score) for score in ([t[idx] for t in targets] if targets else []) ],
                "antitarget_scores": [ _safe_num(score) for score in ([a[idx] for a in antitargets] if antitargets else []) ],
                "combined_score": _safe_num(combined_scores[idx] if idx < len(combined_scores) else -math.inf),
                "score_after_repetition": _safe_num(scores_after_repetition[idx] if idx < len(scores_after_repetition) else -math.inf),
                "boltz_selected": (boltz_selected_flags[idx] if idx < len(boltz_selected_flags) else None),
                "boltz_score": (None if per_mol_boltz is None else _safe_num(float(per_mol_boltz))),
            })

        submissions.append({
            "neuron": {
                "uid": uid,
                "hotkey": hotkey,
                "coldkey": coldkey,
            },
            "blocks_elapsed": blocks_elapsed,
            "molecules": molecule_details,
            "entropy": (entropy if (entropy is not None and final_score_safe > getattr(config, 'entropy_bonus_threshold', 0.0)) else 0),
            "final_score": final_score_safe,
            "final_boltz_score": (None if boltz_score is None else _safe_num(float(boltz_score))),
            "boltz_entropy": (None if boltz_entropy is None else float(boltz_entropy)),
        })

    return submissions


async def submit_epoch_results(submit_url: str, config, metagraph, boltz, current_block: int, start_block: int, current_epoch: int, target_proteins: list[str], antitarget_proteins: list[str], uid_to_data: dict, valid_molecules_by_uid: dict, molecule_name_counts: dict, score_dict: dict) -> None:
    competition = _build_competition_payload(config, current_epoch, target_proteins, antitarget_proteins)
    submissions = _build_submissions_payload(config, metagraph, boltz, current_block, start_block, uid_to_data, valid_molecules_by_uid, molecule_name_counts, score_dict)

    if not submissions:
        bt.logging.info("No submissions to send to dashboard API.")
        return

    payload = {"competition": competition, "submissions": submissions}

    # In test mode, write payload to file and skip POST
    if bool(getattr(config, 'test_mode', False)):
        try:
            results_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(results_dir, exist_ok=True)
            epoch_number = competition.get("epoch_number", 0)
            outfile = os.path.join(results_dir, f"submissions_dryrun_epoch_{epoch_number}.json")
            with open(outfile, "w") as f:
                json.dump(payload, f, indent=2)
            bt.logging.info(f"[DRY-RUN] Saved submissions payload to {outfile}; skipping API POST in test mode.")
        except Exception as e:
            bt.logging.error(f"[DRY-RUN] Failed to write payload: {e}")
        return

    api_key = os.environ.get('SUBMIT_RESULTS_API_KEY')
    if not api_key:
        bt.logging.info("SUBMIT_RESULTS_API_KEY not set; skipping submission.")
        return
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(submit_url, json=payload, headers=headers) as resp:
                if 200 <= resp.status < 300:
                    bt.logging.info(f"Submitted results to dashboard API: {resp.status}")
                else:
                    text = await resp.text()
                    bt.logging.error(f"Dashboard API responded {resp.status}: {text}")
        except Exception as e:
            bt.logging.error(f"Error submitting results to dashboard API: {e}")



