import os
import math
import json
import aiohttp
import dataclasses
import bittensor as bt
import pandas as pd
from huggingface_hub import hf_hub_download, upload_file, HfApi
from huggingface_hub.utils import EntryNotFoundError
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
NOVA_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(NOVA_PATH)

from utils.molecules import get_heavy_atom_count
from config.config_loader import load_boltzgen_metrics

def _safe_num(x: float) -> float:
    if x == -math.inf:
        return -999.99
    if x == math.inf:
        return 999.99
    return x


def _to_json_safe(obj):
    """
    Convert payload to JSON-serializable form.
    Handles dataclasses (e.g. ScoreResult, SearchResult, SearchMatch) and nested structures.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type):
        return _to_json_safe(dataclasses.asdict(obj))
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        return _to_json_safe(obj.model_dump())
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        return _to_json_safe(obj.dict())
    try:
        return _to_json_safe(vars(obj))
    except (TypeError, ValueError):
        return str(obj)


def _build_competition_payload(config: dict, current_epoch: int) -> dict:
    epoch_number = current_epoch
    nanobody_proportion = config.get('nanobody_weight', 0.0)
    return {
        # competition configs
        "epoch_number": epoch_number,
        "small_molecule_target": config.get('small_molecule_target', []),   # list[str] — protein names
        "nanobody_target": config.get('nanobody_target', []),               # list[str] — protein names
        "incentive_distribution": (1.0 - nanobody_proportion, nanobody_proportion),

        # molecule-specific configs
        "num_molecules": getattr(config, 'num_molecules', 0),
        "min_heavy_atoms": getattr(config, 'min_heavy_atoms', 0),
        "min_rotatable_bonds": getattr(config, 'min_rotatable_bonds', 0),
        "max_rotatable_bonds": getattr(config, 'max_rotatable_bonds', 0),
        "entropy_min_threshold": getattr(config, 'min_entropy', 0.0),
        "boltz_metric": getattr(config, 'combination_strategy', None),

        # nanobody-specific configs
        "num_sequences": getattr(config, 'num_sequences', 0),
        "min_sequence_length": getattr(config, 'min_sequence_length', 0),
        "max_sequence_length": getattr(config, 'max_sequence_length', 0),
        "min_cysteines": getattr(config, 'min_cysteines', 0),
        "cys_pair_min_separation": getattr(config, 'cys_pair_min_separation', 0),
        "cys_pair_max_separation": getattr(config, 'cys_pair_max_separation', 0),
        "max_homopolymer_run": getattr(config, 'max_homopolymer_run', 0),
        "max_di_repeat_pairs": getattr(config, 'max_di_repeat_pairs', 0),
        "reject_signal_peptides": getattr(config, 'reject_signal_peptides', False),
        "sp_window": getattr(config, 'sp_window', 0),
        "sp_hydro_min_in_window": getattr(config, 'sp_hydro_min_in_window', 0),
        "sp_scan_prefix": getattr(config, 'sp_scan_prefix', 0),
        "min_nativeness_score": getattr(config, 'min_nativeness_score', 0.0),
        "min_human_framework_score": getattr(config, 'min_human_framework_score', 0.0),
        "max_similarity_score": getattr(config, 'max_similarity_score', 0.0),
        "num_top_sequences": getattr(config, 'num_top_sequences', 0),
    }


# ---------------------------------------------------------------------------
# Molecule helpers
# ---------------------------------------------------------------------------

def _build_molecule_details(
    uid: int,
    smiles_list: list[str],
    names_list: list[str],
    score_dict: dict,
    boltz,
    target_proteins: list[str],
) -> list[dict]:
    """Build per-molecule detail dicts for a single submission."""
    mol_scores = score_dict.get(uid, {}).get('molecule_scores', [])
    protein_to_idx = {p: i for i, p in enumerate(target_proteins)}

    molecule_details = []
    for idx in range(len(smiles_list)):
        # --- per-molecule Boltz / structural metrics (per target protein) ---
        per_mol_boltz = None
        comp = {}
        try:
            if getattr(boltz, 'per_molecule_metric', None):
                per_mol_boltz = boltz.per_molecule_metric.get(uid, {}).get(smiles_list[idx])

            per_molecule_components = getattr(boltz, 'per_molecule_components', None)
            if per_molecule_components:
                comp = per_molecule_components.get(uid, {}).get(smiles_list[idx], {})
        except Exception:
            per_mol_boltz = None
            comp = {}

        if comp and not isinstance(next(iter(comp.values()), None), dict):
            comp = {"unknown": comp}

        if not comp or all(
            all(v is None for v in metrics.values())
            for metrics in comp.values()
            if isinstance(metrics, dict)
        ):
            continue

        per_protein_entries = []
        for protein_name, metrics in (comp.items() if comp else [("unknown", {})]):
            def _get(key):
                v = metrics.get(key)
                if v is None:
                    return None
                if isinstance(v, (int, float, str)):
                    return _safe_num(float(v))
                if isinstance(v, dict):
                    return _to_json_safe(v)

            target_idx = protein_to_idx.get(protein_name)
            if target_idx is not None and target_idx < len(mol_scores) and idx < len(mol_scores[target_idx]):
                final_score = _safe_num(mol_scores[target_idx][idx])
            else:
                final_score = None

            per_protein_entries.append({
                "protein_name": protein_name,
                "name": names_list[idx],
                "smiles": smiles_list[idx],
                "final_score": final_score,
                # --- structural metrics (per target) ---
                "affinity_probability_binary": _get("affinity_probability_binary"),
                "affinity_pred_value": _get("affinity_pred_value"),
                "affinity_probability_binary1": _get("affinity_probability_binary1"),
                "affinity_pred_value1": _get("affinity_pred_value1"),
                "affinity_probability_binary2": _get("affinity_probability_binary2"),
                "affinity_pred_value2": _get("affinity_pred_value2"),
                "confidence_score": _get("confidence_score"),
                "ptm": _get("ptm"),
                "iptm": _get("iptm"),
                "ligand_iptm": _get("ligand_iptm"),
                "protein_iptm": _get("protein_iptm"),
                "complex_plddt": _get("complex_plddt"),
                "complex_iplddt": _get("complex_iplddt"),
                "complex_pde": _get("complex_pde"),
                "complex_ipde": _get("complex_ipde"),
                "chains_ptm": metrics.get("chains_ptm"),
                "pair_chains_iptm": metrics.get("pair_chains_iptm"),
                "heavy_atom_count": get_heavy_atom_count(smiles_list[idx]),
            })

        molecule_details.extend(per_protein_entries)

    return molecule_details


# ---------------------------------------------------------------------------
# Nanobody helpers
# ---------------------------------------------------------------------------

def _build_nanobody_details(
    uid: int,
    sequences_list: list[str],
    hashes_list: list[str],
    nanobody_data: dict,
    boltzgen,
    boltzgen_rank_by: str,
) -> list[dict]:
    """
    Build per-nanobody detail dicts for a single submission.

    Each nanobody yields ONE entry containing:
      - target_scores: dict[protein_name -> per-target structural metrics]
      - developability, nativeness, similarity: target-invariant metrics

    Per-target metrics are pulled from boltzgen.per_nanobody_components,
    mirroring how molecule metrics are pulled from boltz.per_molecule_components.
    """
    developability_results = nanobody_data.get('developability_result', [])
    nativeness_results = nanobody_data.get('nativeness_result', [])
    similarity_results = nanobody_data.get('similarity_results', [])

    per_nanobody_components = getattr(boltzgen, 'per_nanobody_components', None) or {}
    uid_components = per_nanobody_components.get(uid, {})

    nanobody_details = []
    for idx, sequence in enumerate(sequences_list):
        seq_hash = hashes_list[idx] if idx < len(hashes_list) else None

        # --- Per-target structural metrics ---
        per_target = uid_components.get(sequence, {})  # {protein_name: {metric: val}}

        if not per_target or all(
            all(v is None for v in metrics.values())
            for metrics in per_target.values()
            if isinstance(metrics, dict)
        ):
            continue

        target_scores = {}
        for protein_name, metrics in per_target.items():
            def _get(key, m=metrics):
                v = m.get(key)
                return None if v is None else _safe_num(float(v))
            target_scores[protein_name] = {
                "design_iiptm": _get("design_iiptm"),
                "design_ptm": _get("design_ptm"),
                "design_to_target_iptm": _get("design_to_target_iptm"),
                "min_design_to_target_pae": _get("min_design_to_target_pae"),
                "interaction_pae": _get("interaction_pae"),
                "plip_hbonds_refolded": _get("plip_hbonds_refolded"),
                "plip_saltbridge_refolded": _get("plip_saltbridge_refolded"),
                "delta_sasa_refolded": _get("delta_sasa_refolded"),
                "liability_score": _get("liability_score"),
                "liability_num_violations": _get("liability_num_violations"),
                "confidence_rank": _get(f"confidence_{boltzgen_rank_by}"),
                "physical_interaction_rank": _get(f"physical_interaction_{boltzgen_rank_by}"),
                "developability_rank": _get(f"developability_{boltzgen_rank_by}"),
                "final_score": _get(f"{boltzgen_rank_by}"),
            }

        # --- Target-invariant: developability ---
        dev = developability_results[idx] if idx < len(developability_results) else None
        developability = None
        if dev is not None:
            developability = {
                "passed": dev.get("passed"),
                "total_cdr_length": dev.get("total_cdr_length"),
                "cdr3_length": dev.get("cdr3_length"),
                "cdr3_compactness": dev.get("cdr3_compactness"),
                "surface_hydrophobic_patches": dev.get("surface_hydrophobic_patches"),
                "positive_charge_patches": dev.get("positive_charge_patches"),
                "negative_charge_patches": dev.get("negative_charge_patches"),
                "flags": dev.get("flags"),  # dict of {flag_name: "green"/"red"/...}
            }

        # --- Target-invariant: nativeness ---
        nat = nativeness_results[idx] if idx < len(nativeness_results) else None
        nativeness = None
        if nat is not None:
            nat_dict = _to_json_safe(nat)
            nativeness = {
                "vhh_nativeness": nat_dict.get("vhh_nativeness"),
                "human_framework": nat_dict.get("human_framework"),
                "features": nat_dict.get("features"),  # full features dict (as json object because key can vary)
            }

        # --- Target-invariant: similarity ---
        sim = similarity_results[idx] if idx < len(similarity_results) else None
        similarity = None
        if sim is not None:
            sim_dict = _to_json_safe(sim)
            matches_raw = sim_dict.get("matches", [])
            similarity = {
                "matches": [
                    {
                        "target_id": str(m.get("target_id")) if m.get("target_id") is not None else None,
                        "target_sequence": str(m.get("target_sequence")) if m.get("target_sequence") is not None else None,
                        "identity": _safe_num(float(m.get("identity"))) if m.get("identity") is not None else None,
                        "tier": str(m.get("tier")) if m.get("tier") is not None else None,
                        "cdr_similarity": _to_json_safe(m.get("cdr_similarity")) if m.get("cdr_similarity") is not None else None,
                    }
                    for m in matches_raw
                ]
            }

        nanobody_details.append({
            "sequence": sequence,
            "sequence_hash": seq_hash,
            "target_scores": target_scores,             # per-target (goes to nanobodies table)
            "developability": developability,            # invariant (goes to nanobody_properties)
            "nativeness": nativeness,                    # invariant (goes to nanobody_properties)
            "similarity": similarity,                    # invariant (goes to nanobody_properties)
        })

    return nanobody_details


# ---------------------------------------------------------------------------
# Submission-level builder
# ---------------------------------------------------------------------------

def _build_submissions_payload(
    config,
    metagraph,
    boltz,
    boltzgen,
    current_block: int,
    start_block: int,
    uid_to_data: dict,
    valid_molecules_by_uid: dict,
    valid_nanobodies_by_uid: dict,
    score_dict: dict,
    nanobody_ranks: dict[int, int] | None = None,
) -> list[dict]:
    submissions = []
    nanobody_ranks = nanobody_ranks or {}

    # Collect all UIDs that have either molecules or nanobodies
    all_uids = set(uid_to_data.keys())

    for uid in all_uids:
        data = uid_to_data[uid]

        hotkey = data.get('hotkey') or (metagraph.hotkeys[uid] if uid < len(metagraph.hotkeys) else "unknown")
        coldkey = metagraph.coldkeys[uid] if hasattr(metagraph, 'coldkeys') and uid < len(metagraph.coldkeys) else "unknown"

        uid_scores = score_dict.get(uid, {})
        block_submitted = uid_scores.get('block_submitted', data.get('block_submitted'))
        if isinstance(block_submitted, int):
            blocks_elapsed = block_submitted - start_block
        else:
            blocks_elapsed = (block_submitted or current_block) - start_block

        # --- Molecule data ---
        valid_mol = valid_molecules_by_uid.get(uid, {})
        smiles_list = valid_mol.get('smiles', [])
        names_list = valid_mol.get('names', [])

        molecule_details = []
        if smiles_list:
            molecule_details = _build_molecule_details(
                uid, smiles_list, names_list,
                score_dict, boltz,
                target_proteins=config.get('small_molecule_target', []) if isinstance(config, dict) else getattr(config, 'small_molecule_target', []),
            )

        # --- Nanobody data ---
        valid_nano = valid_nanobodies_by_uid.get(uid, {})
        sequences_list = valid_nano.get('sequences', [])
        hashes_list = valid_nano.get('hashes', [])

        nanobody_details = []
        if sequences_list:
            nanobody_details = _build_nanobody_details(
                uid, sequences_list, hashes_list,
                valid_nano, boltzgen, config.boltzgen_rank_by,
            )

        # Skip if nothing to submit
        if not molecule_details and not nanobody_details:
            continue

        # --- Scores ---
        molecule_final_score = _safe_num(uid_scores.get('final_molecule_score', -math.inf))
        nanobody_final_score = _safe_num(uid_scores.get('final_nanobody_score', -math.inf))

        submissions.append({
            "neuron": {
                "uid": uid,
                "hotkey": hotkey,
                "coldkey": coldkey,
            },
            "blocks_elapsed": blocks_elapsed,
            "molecule_final_score": molecule_final_score,
            "nanobody_final_score": nanobody_final_score,
            "nanobody_leaderboard_rank": nanobody_ranks.get(uid),
            "molecules": molecule_details,
            "nanobodies": nanobody_details,
        })

    return submissions

def _flatten_nanobody_rows(nanobody_details: list[dict], metric_names: list[str]) -> dict[str, list[dict]]:
    """
    Flatten nanobody details into per-target row dicts.
    Returns {target_name: [row_dict, ...]} where each row_dict has keys:
      sequence, sequence_hash, and all metric_names.
    """
    by_target: dict[str, list[dict]] = {}
    for entry in nanobody_details:
        sequence = entry["sequence"]
        seq_hash = entry.get("sequence_hash")
        target_scores = entry.get("target_scores", {})
        for target_name, metrics in target_scores.items():
            row = {
                "sequence": sequence,
                "sequence_hash": seq_hash,
            }
            for key in metric_names:
                row[key] = metrics.get(key)
            by_target.setdefault(target_name, []).append(row)
    return by_target

def _append_and_upload_csv_to_hf(
    *,
    repo_id: str,
    filename: str,
    new_df: pd.DataFrame,
    token: str,
    repo_type: str = "dataset",
    dedup_subset: str | list[str] | None = None,
    commit_message: str | None = None,
    epoch: int | None = None,
) -> None:
    """
    Helper to append rows to an existing CSV in a HF dataset repo, drop duplicates,
    and upload the updated file.
    """
    if new_df is None or new_df.empty:
        bt.logging.warning(f"No new rows to upload to {repo_id}/{filename}.")
        return

    api = HfApi(token=token)

    # Try to fetch the existing file from the repo
    existing_df = None
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            token=token,
        )
        existing_df = pd.read_csv(local_path)
    except EntryNotFoundError:
        # File does not exist yet – we'll create it
        pass
    except Exception as e:
        bt.logging.warning(
            f"Could not download existing {filename} from {repo_id}: {e}"
        )

    if existing_df is not None:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    if dedup_subset:
        combined_df = combined_df.drop_duplicates(subset=dedup_subset, keep="last")

    if commit_message is None:
        n_new = len(new_df)
        epoch_str = f"from epoch {epoch}" if epoch is not None else ""
        commit_message = f"Append {n_new} rows {epoch_str} to {filename}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        combined_df.to_csv(tmp, index=False)
        tmp_path = tmp.name

    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        bt.logging.info(
            f"Uploaded {repo_id}/{filename}: "
            f"added {len(new_df)} rows, {len(combined_df)} total after dedup."
        )
    except Exception as e:
        bt.logging.error(f"Failed to upload {filename} to {repo_id}: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def push_nanobodies_to_hf(
    nanobody_details: list[dict],
    metric_names: list[str],
    repo_id: str,
    repo_type: str = "dataset",
    token: str | None = None,
    epoch: int = 0,
) -> None:
    """
    Feeds the nanobody details to a Metanova/Submission-Archive dataset (per target).
    """
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        bt.logging.warning("HF_TOKEN not set; skipping HF dataset push.")
        return

    rows_by_target = _flatten_nanobody_rows(nanobody_details, metric_names)
    if not rows_by_target:
        bt.logging.info("No nanobody target rows to push to HF.")
        return

    for target_name, rows in rows_by_target.items():
        filename = f"{target_name}_nanobodies.csv"
        new_df = pd.DataFrame(rows)

        _append_and_upload_csv_to_hf(
            repo_id=repo_id,
            filename=filename,
            new_df=new_df,
            token=token,
            repo_type=repo_type,
            dedup_subset="sequence_hash",
            commit_message=(
                f"Added {len(new_df)} nanobody sequences from epoch {epoch} "
                f"to {filename}"
            ),
            epoch=epoch,
        )

async def push_molecules_to_hf(
    molecule_details: list[dict],
    config: dict,
    repo_id: str,
    repo_type: str = "dataset",
    token: str | None = None,
    epoch: int = 0,
    dedup_on_inchi: bool = True,
) -> None:
    """
    Feeds the molecule details to a Metanova/Submission-Archive dataset.
    """
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        bt.logging.warning("HF_TOKEN not set; skipping HF molecule dataset push.")
        return

    if not molecule_details:
        bt.logging.warning("No molecule rows to push to HF.")
        return

    # Construct a DataFrame with only the required fields;
    # deduplication is handled when appending to the existing dataset.
    new_df = pd.DataFrame(molecule_details)[["Molecule_ID", "SMILES", "InChI_Key"]]

    dedup_subset = "InChI_Key" if dedup_on_inchi else "Molecule_ID"

    for target_name in config["small_molecule_target"]:
        filename = f"{target_name}_molecules.csv"
        _append_and_upload_csv_to_hf(
            repo_id=repo_id,
            filename=filename,
            new_df=new_df,
            token=token,
            repo_type=repo_type,
            dedup_subset=dedup_subset,
            commit_message=(
                f"Added {len(new_df)} molecules from epoch {epoch} to {filename}"
            ),
            epoch=epoch,
        )

# ---------------------------------------------------------------------------
# Top-level submit function
# ---------------------------------------------------------------------------

async def submit_epoch_results(
    submit_url: str,
    config,
    metagraph,
    boltz,
    boltzgen,
    current_block: int,
    start_block: int,
    current_epoch: int,
    uid_to_data: dict,
    valid_molecules_by_uid: dict,
    valid_nanobodies_by_uid: dict,
    score_dict: dict,
    nanobody_ranks: dict[int, int] | None = None,
) -> None:
    competition = _build_competition_payload(config, current_epoch)
    submissions = _build_submissions_payload(
        config, metagraph, boltz, boltzgen, current_block, start_block,
        uid_to_data, valid_molecules_by_uid, valid_nanobodies_by_uid,
        score_dict,
        nanobody_ranks=nanobody_ranks,
    )

    if not submissions:
        bt.logging.info("No submissions to send to dashboard API.")
        return

    # Push molecules and nanobodies submissions to public archive dataset
    boltzgen_metric_names = list(load_boltzgen_metrics().keys())
    try:
        all_nanobody_details = []
        for sub in submissions:
            all_nanobody_details.extend(sub.get("nanobodies", []))
        hf_repo = "Metanova/Submission-Archive"
        if hf_repo and all_nanobody_details:
            await push_nanobodies_to_hf(all_nanobody_details, boltzgen_metric_names, repo_id=hf_repo, epoch=current_epoch)
    except Exception as e:
        bt.logging.error(f"Error pushing nanobody details to HF: {e}")

    try:
        all_molecule_details = []
        for sub in submissions:
            names_list = [molecule.get("name") for molecule in sub.get("molecules", [])]
            smiles_list = [molecule.get("smiles") for molecule in sub.get("molecules", [])]

            for name, smiles in zip(names_list, smiles_list):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    inchikey = Chem.MolToInchiKey(mol)
                    all_molecule_details.append({"Molecule_ID": name, "SMILES": smiles, "InChI_Key": inchikey})

        hf_repo = "Metanova/Submission-Archive"
        if hf_repo and all_molecule_details:
            await push_molecules_to_hf(all_molecule_details, config, repo_id=hf_repo, epoch=current_epoch)
    except Exception as e:
        bt.logging.error(f"Error pushing molecule details to HF: {e}")

    payload = _to_json_safe({"competition": competition, "submissions": submissions})

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
    submit_url = os.environ.get('SUBMIT_RESULTS_URL') + "/api/competitions/submission"
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