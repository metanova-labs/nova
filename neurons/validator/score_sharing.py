import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt
import requests

from config.config_loader import load_boltzgen_metrics

WAIT_SECONDS = 300
FINALIZATION_BUFFER_BLOCKS = 30

# Per-target molecule metrics to share (scalar only; chains_ptm and
# pair_chains_iptm are nested dicts and heavy_atom_count is invariant)
MOLECULE_METRIC_KEYS = [
    "affinity_probability_binary",
    "affinity_pred_value",
    "affinity_probability_binary1",
    "affinity_pred_value1",
    "affinity_probability_binary2",
    "affinity_pred_value2",
    "confidence_score",
    "ptm",
    "iptm",
    "ligand_iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
]

NANOBODY_METRIC_KEYS = load_boltzgen_metrics()


async def _post_json(url: str, json_body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
    def _do_post():
        resp = requests.post(url, json=json_body, headers=headers, timeout=10)
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, {}

    return await asyncio.to_thread(_do_post)


async def _get_json(url: str, headers: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
    def _do_get():
        resp = requests.get(url, headers=headers, timeout=10)
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, {}

    return await asyncio.to_thread(_do_get)


async def _get_target_averages(
    url: str, headers: Dict[str, str]
) -> Tuple[int, Optional[Dict[str, Dict[str, Optional[float]]]]]:
    """Fetch averaged per-target metrics. Returns {protein_name: {metric: avg_value}}."""
    status, data = await _get_json(url, headers)
    if status >= 400:
        return status, None
    try:
        target_averages = data.get("target_averages")
        if target_averages is None:
            return status, None
        return status, target_averages
    except Exception:
        return status, None


def _safe_float(val) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Generic per-target validation builder
# ---------------------------------------------------------------------------

def _build_target_validations(
    per_components: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]],
    valid_items_by_uid: Dict[int, Dict[str, Any]],
    item_key: str,
    hash_key: str,
    id_field_name: str,
    metric_keys: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, str]]]:
    """
    Build per-target validation entries from a per_*_components dict.

    Works for both molecules (keyed by SMILES → name) and nanobodies
    (keyed by sequence → hash).

    Args:
        per_components:   {uid: {item: {protein: {metric: val}}}}
        valid_items_by_uid: {uid: {item_key: [...], hash_key: [...]}}
        item_key:         Key in valid_items_by_uid for the item list (e.g. "smiles", "sequences")
        hash_key:         Key in valid_items_by_uid for the ID list (e.g. "names", "hashes")
        id_field_name:    Field name in the output entry (e.g. "name", "sequence_hash")
        metric_keys:      List of metric names to include

    Returns:
        (validations_list, uid_to_id_item_map)
        uid_to_id_item_map: {uid: {id_value: item_value}} for reverse lookups
    """
    validations: List[Dict[str, Any]] = []
    uid_to_id_item: Dict[int, Dict[str, str]] = {}

    for uid, item_to_proteins in per_components.items():
        item_data = valid_items_by_uid.get(uid) or {}
        items_list = item_data.get(item_key, []) or []
        ids_list = item_data.get(hash_key, []) or []
        if not items_list or not ids_list:
            continue

        item_to_id = {item: id_val for item, id_val in zip(items_list, ids_list)}

        for item, protein_metrics in item_to_proteins.items():
            id_val = item_to_id.get(item)
            if not id_val:
                continue

            uid_to_id_item.setdefault(uid, {})[id_val] = item

            for protein_name, metrics in protein_metrics.items():
                entry: Dict[str, Any] = {
                    id_field_name: id_val,
                    "target_protein": protein_name,
                }
                for key in metric_keys:
                    entry[key] = _safe_float(metrics.get(key))

                validations.append(entry)

    return validations, uid_to_id_item


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

async def apply_external_scores(
    score_dict: Dict[int, Dict[str, Any]],
    valid_molecules_by_uid: Dict[int, Dict[str, List[str]]],
    valid_nanobodies_by_uid: Dict[int, Dict[str, Any]],
    api_url: str,
    api_key: Optional[str] = None,
    epoch: Optional[int] = None,
    boltz: Optional[Any] = None,
    boltzgen: Optional[Any] = None,
    target_proteins: Optional[List[str]] = None,
    subtensor: Optional[Any] = None,
    epoch_end_block: Optional[int] = None,
    finalization_buffer_blocks: int = FINALIZATION_BUFFER_BLOCKS,
    test_mode: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Share Boltz per-molecule-target scores and BoltzGen per-nanobody-target
    scores with the score-share API and, after a delay, update each UID's
    per-target metrics with the validator-averaged values.
    """
    if not api_url:
        return score_dict

    base_url = api_url.rstrip("/") + "/api/v1"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        if epoch is None:
            bt.logging.warning("Epoch was not provided to score-share step; keeping local scores.")
            return score_dict

        # --- Build molecule per-target validations ---
        molecule_validations = []
        uid_to_mol_id: Dict[int, Dict[str, str]] = {}  # {uid: {name: smiles}}
        per_molecule_components = getattr(boltz, 'per_molecule_components', None) or {}
        if per_molecule_components:
            molecule_validations, uid_to_mol_id = _build_target_validations(
                per_components=per_molecule_components,
                valid_items_by_uid=valid_molecules_by_uid,
                item_key="smiles",
                hash_key="names",
                id_field_name="name",
                metric_keys=MOLECULE_METRIC_KEYS,
            )

        # --- Enrich molecule entries with final calculated score ---
        # score_dict[uid]['target_scores'] is list[list[float]]:
        #   target_scores[target_idx][molecule_idx] where target ordering
        #   matches the target_proteins list.
        if molecule_validations and target_proteins:
            protein_to_idx = {p: i for i, p in enumerate(target_proteins)}

            # Build lookup: (mol_name, protein_name) -> score
            mol_score_lookup: Dict[Tuple[str, str], float] = {}
            for uid in score_dict:
                targets = score_dict[uid].get('target_scores', [])
                mol_data = valid_molecules_by_uid.get(uid, {})
                names_list = mol_data.get('names', []) or []
                for protein_name, target_idx in protein_to_idx.items():
                    if target_idx >= len(targets):
                        continue
                    for mol_idx, name in enumerate(names_list):
                        if mol_idx < len(targets[target_idx]):
                            mol_score_lookup[(name, protein_name)] = targets[target_idx][mol_idx]

            for entry in molecule_validations:
                score = mol_score_lookup.get((entry["name"], entry.get("target_protein")))
                entry["score"] = _safe_float(score)

        # --- Build nanobody per-target validations ---
        nanobody_validations = []
        uid_to_nano_id: Dict[int, Dict[str, str]] = {}  # {uid: {hash: sequence}}
        per_nanobody_components = getattr(boltzgen, 'per_nanobody_components', None) or {}
        if per_nanobody_components:
            nanobody_validations, uid_to_nano_id = _build_target_validations(
                per_components=per_nanobody_components,
                valid_items_by_uid=valid_nanobodies_by_uid,
                item_key="sequences",
                hash_key="hashes",
                id_field_name="sequence_hash",
                metric_keys=NANOBODY_METRIC_KEYS,
            )

        if not molecule_validations and not nanobody_validations:
            return score_dict

        # --- Submit batch ---
        batch_request: Dict[str, Any] = {"epoch": int(epoch)}
        if molecule_validations:
            batch_request["molecules"] = molecule_validations
        if nanobody_validations:
            batch_request["nanobodies"] = nanobody_validations

        # In test mode, write payload to file and skip API calls
        if test_mode:
            try:
                results_dir = os.path.join(os.getcwd(), "results")
                os.makedirs(results_dir, exist_ok=True)
                outfile = os.path.join(results_dir, f"score_share_dryrun_epoch_{int(epoch)}.json")
                with open(outfile, "w") as f:
                    json.dump(batch_request, f, indent=2)
                bt.logging.info(f"[DRY-RUN] Saved score-share payload to {outfile}; skipping API calls in test mode.")
            except Exception as e:
                bt.logging.error(f"[DRY-RUN] Failed to write score-share payload: {e}")
            return score_dict

        status_code, _ = await _post_json(base_url + "/validations/batch", batch_request, headers)
        if status_code >= 400:
            bt.logging.warning(
                f"Score-share POST failed for epoch {epoch} with status {status_code}; using local scores."
            )
            return score_dict

        total_validations = len(molecule_validations) + len(nanobody_validations)

        # --- Wait for other validators ---
        if subtensor is not None and epoch_end_block is not None:
            target_block = max(epoch_end_block - int(finalization_buffer_blocks), 0)
            bt.logging.info(
                f"Submitted {total_validations} validation(s) to score-share API; "
                f"waiting until block {target_block}, {finalization_buffer_blocks} blocks before epoch end, "
                f"before retrieving averages"
            )
            try:
                await subtensor.wait_for_block(target_block)
            except Exception as e:
                bt.logging.warning(
                    f"Error while waiting for target block before score-share GET; using local scores. Error: {e}"
                )
                return score_dict
        else:
            bt.logging.info(
                f"Submitted {total_validations} validation(s) to score-share API, "
                f"waiting {WAIT_SECONDS}s before retrieving averages"
            )
            await asyncio.sleep(WAIT_SECONDS)

        # Fetch all averages before mutating (avoid applying molecule updates if nanobody GET fails).
        name_to_target_avgs: Optional[Dict[str, Dict]] = None
        hash_to_target_avgs: Optional[Dict[str, Dict]] = None

        if uid_to_mol_id:
            all_mol_names: set[str] = set()
            for id_map in uid_to_mol_id.values():
                all_mol_names.update(id_map.keys())

            async def _get_mol_avg(name: str) -> Tuple[str, int, Optional[Dict]]:
                url = base_url + f"/molecule-targets/{name}/{int(epoch)}"
                s, avgs = await _get_target_averages(url, headers)
                return name, s, avgs

            mol_results = await asyncio.gather(*[_get_mol_avg(n) for n in all_mol_names])

            name_to_target_avgs = {}
            for name, status_code_mol, target_avgs in mol_results:
                if status_code_mol >= 400 or target_avgs is None:
                    bt.logging.warning(
                        f"Score-share molecule avg fetch failed for {name}@{epoch} "
                        f"with status {status_code_mol}; using local scores."
                    )
                    return score_dict
                name_to_target_avgs[name] = target_avgs

        if uid_to_nano_id:
            all_hashes: set[str] = set()
            for id_map in uid_to_nano_id.values():
                all_hashes.update(id_map.keys())

            async def _get_nano_avg(seq_hash: str) -> Tuple[str, int, Optional[Dict]]:
                url = base_url + f"/nanobodies/{seq_hash}/{int(epoch)}"
                s, avgs = await _get_target_averages(url, headers)
                return seq_hash, s, avgs

            nano_results = await asyncio.gather(*[_get_nano_avg(h) for h in all_hashes])

            hash_to_target_avgs = {}
            for seq_hash, status_code_nano, target_avgs in nano_results:
                if status_code_nano >= 400 or target_avgs is None:
                    bt.logging.warning(
                        f"Score-share nanobody avg fetch failed for {seq_hash}@{epoch} "
                        f"with status {status_code_nano}; using local scores."
                    )
                    return score_dict
                hash_to_target_avgs[seq_hash] = target_avgs

        if uid_to_mol_id and name_to_target_avgs is not None:
            if per_molecule_components:
                for uid, id_item_map in uid_to_mol_id.items():
                    for name, smiles in id_item_map.items():
                        averaged = name_to_target_avgs.get(name)
                        if not averaged:
                            continue
                        for protein_name, avg_metrics in averaged.items():
                            if uid in per_molecule_components and smiles in per_molecule_components[uid]:
                                if protein_name in per_molecule_components[uid][smiles]:
                                    comp = per_molecule_components[uid][smiles][protein_name]
                                    for key in MOLECULE_METRIC_KEYS:
                                        avg_val = avg_metrics.get(key)
                                        if avg_val is not None:
                                            comp[key] = avg_val
                                    avg_score = avg_metrics.get("score")
                                    if avg_score is not None:
                                        comp["score"] = float(avg_score)

            if target_proteins:
                protein_to_idx = {p: i for i, p in enumerate(target_proteins)}
                for uid, id_item_map in uid_to_mol_id.items():
                    mol_data = valid_molecules_by_uid.get(uid, {})
                    names_list = mol_data.get('names', []) or []
                    name_to_mol_idx = {n: i for i, n in enumerate(names_list)}
                    uid_targets = score_dict.get(uid, {}).get('target_scores', [])

                    for name, smiles in id_item_map.items():
                        averaged = name_to_target_avgs.get(name)
                        if not averaged:
                            continue
                        mol_idx = name_to_mol_idx.get(name)
                        if mol_idx is None:
                            continue
                        for protein_name, avg_metrics in averaged.items():
                            target_idx = protein_to_idx.get(protein_name)
                            if target_idx is None:
                                continue
                            avg_score = avg_metrics.get("score")
                            if avg_score is not None and target_idx < len(uid_targets) and mol_idx < len(uid_targets[target_idx]):
                                uid_targets[target_idx][mol_idx] = float(avg_score)

            bt.logging.info(
                f"Replaced molecule scores with validator averages for {len(name_to_target_avgs)} molecule(s)"
            )

        if uid_to_nano_id and hash_to_target_avgs is not None:
            if per_nanobody_components:
                for uid, id_item_map in uid_to_nano_id.items():
                    for seq_hash, sequence in id_item_map.items():
                        averaged = hash_to_target_avgs.get(seq_hash)
                        if not averaged:
                            continue
                        for protein_name, avg_metrics in averaged.items():
                            if uid in per_nanobody_components and sequence in per_nanobody_components[uid]:
                                if protein_name in per_nanobody_components[uid][sequence]:
                                    for key in NANOBODY_METRIC_KEYS:
                                        avg_val = avg_metrics.get(key)
                                        if avg_val is not None:
                                            per_nanobody_components[uid][sequence][protein_name][key] = avg_val

            bt.logging.info(
                f"Replaced nanobody scores with validator averages for {len(hash_to_target_avgs)} sequence(s)"
            )

        return score_dict

    except Exception as e:
        bt.logging.error(f"Score-share step failed; using local scores. Error: {e}")
        return score_dict