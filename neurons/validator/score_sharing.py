import asyncio
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt
import requests

WAIT_SECONDS = 300   # Delay between POST and GET


async def _post_json(url: str, json_body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
    """Async wrapper around requests.post returning (status_code, json)."""
    def _do_post():
        resp = requests.post(url, json=json_body, headers=headers, timeout=10)
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, {}

    return await asyncio.to_thread(_do_post)


async def _get_json(url: str, headers: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
    """Async wrapper around requests.get returning (status_code, json)."""
    def _do_get():
        resp = requests.get(url, headers=headers, timeout=10)
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, {}

    return await asyncio.to_thread(_do_get)


async def _get_molecule_avg(url: str, headers: Dict[str, str]) -> Tuple[int, Optional[float]]:
    """Fetch a single molecule and return its avg, or None."""
    status, data = await _get_json(url, headers)
    if status >= 400:
        return status, None
    try:
        avg = data.get("avg")
        if avg is None:
            return status, None
        return status, float(avg)
    except Exception:
        return status, None


async def apply_external_scores(
    score_dict: Dict[int, Dict[str, Any]],
    valid_molecules_by_uid: Dict[int, Dict[str, List[str]]],
    api_url: str,
    api_key: Optional[str] = None,
    epoch: Optional[int] = None,
    boltz_per_molecule: Optional[Dict[int, Dict[str, float]]] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Share Boltz per-molecule scores with the score-share API and, after a fixed wait,
    update each UID's boltz_score with the validator-averaged values.
    Falls back to local scores on any error or missing data.
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

        if not boltz_per_molecule:
            return score_dict

        post_items: List[Tuple[str, Dict[str, Any]]] = []
        uid_to_names: Dict[int, List[str]] = {}
        for uid, smiles_to_metric in boltz_per_molecule.items():
            mol_data = valid_molecules_by_uid.get(uid) or {}
            smiles_list = mol_data.get("smiles", []) or []
            names_list = mol_data.get("names", []) or []
            if not smiles_list or not names_list:
                continue
            smiles_to_name = {s: n for s, n in zip(smiles_list, names_list)}

            for smiles, metric in smiles_to_metric.items():
                name = smiles_to_name.get(smiles)
                if not name:
                    continue
                try:
                    score_val = float(metric)
                except Exception:
                    continue

                payload = {
                    "name": name,
                    "epoch": int(epoch),
                    "score": score_val,
                    "target_score": score_val,
                    "antitarget_score": None,
                }
                post_items.append((name, payload))
                uid_to_names.setdefault(uid, []).append(name)

        async def _post_one(name: str, payload: Dict[str, Any]) -> Tuple[str, int]:
            status_code, _ = await _post_json(base_url + "/validations", payload, headers)
            return name, status_code

        total_posts = 0
        if post_items:
            results = await asyncio.gather(
                *[_post_one(name, payload) for name, payload in post_items]
            )
            for name, status_code in results:
                if status_code >= 400:
                    bt.logging.warning(
                        f"Score-share POST failed for {name}@{epoch} with status {status_code}; using local scores."
                    )
                    return score_dict
                total_posts += 1

        if total_posts == 0:
            return score_dict

        bt.logging.info(
            f"Submitted {total_posts} validation(s) to score-share API, "
            f"waiting {WAIT_SECONDS}s before retrieving averages"
        )

        await asyncio.sleep(WAIT_SECONDS)

        unique_names: set[str] = {name for name, _ in post_items}

        if not unique_names:
            return score_dict

        async def _get_one(name: str) -> Tuple[str, int, Optional[float]]:
            url = base_url + f"/molecules/{name}/{int(epoch)}"
            status_code, avg = await _get_molecule_avg(url, headers)
            return name, status_code, avg

        name_to_avg: Dict[str, float] = {}
        results = await asyncio.gather(*[_get_one(name) for name in unique_names])

        for name, status_code, avg in results:
            if status_code >= 400 or avg is None:
                bt.logging.warning(
                    f"Score-share average fetch failed for {name}@{epoch} with status {status_code}; using local scores."
                )
                return score_dict
            name_to_avg[name] = avg

        if not name_to_avg:
            bt.logging.warning("No averages available from score-share API; using local scores.")
            return score_dict

        for uid, names_list in uid_to_names.items():
            if not names_list:
                continue
            values: List[float] = []
            for name in names_list:
                values.append(float(name_to_avg[name]))
            if values:
                score_dict.setdefault(uid, {})["boltz_score"] = float(sum(values) / len(values))

        bt.logging.info(
            f"Replaced Boltz scores with validator averages for {len(name_to_avg)} unique molecule(s)"
        )
        return score_dict

    except Exception as e:
        bt.logging.error(f"Score-share step failed; using local scores. Error: {e}")
        return score_dict



