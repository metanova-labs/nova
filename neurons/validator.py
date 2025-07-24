import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import asyncio
import binascii
import datetime
import hashlib
import math
import os
import subprocess
import sys
import time
from ast import literal_eval
from types import SimpleNamespace
from typing import cast, Optional

import aiohttp
import bittensor as bt
import numpy as np
import requests
from bittensor.core.chain_data.utils import decode_metadata
from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem import Descriptors
from substrateinterface import SubstrateInterface

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from auto_updater import AutoUpdater
from btdr import QuicknetBittensorDrandTimelock
from config.config_loader import load_config
from my_utils import (
    calculate_dynamic_entropy,
    compute_maccs_entropy,
    find_chemically_identical,
    get_challenge_proteins_from_blockhash,
    get_heavy_atom_count,
    get_sequence_from_protein_code,
    get_smiles,
    molecule_unique_for_protein_hf,
    monitor_validator
)
from PSICHIC.wrapper import PsichicWrapper

MAX_RESPONSE_SIZE = 20 * 1024  # 20KB
GITHUB_HEADERS = {}

psichic = PsichicWrapper()
btd = QuicknetBittensorDrandTimelock()

def get_config():
    """
    Parse command-line arguments to set up the configuration for the wallet
    and subtensor client.
    """
    load_dotenv()
    parser = argparse.ArgumentParser('Nova')
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)

    config = bt.config(parser)
    config.netuid = 68
    config.network = os.environ.get("SUBTENSOR_NETWORK")
    node = SubstrateInterface(url=config.network)
    config.epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value + 1

    # Load configuration options
    config.update(load_config())

    return config

def setup_logging(config):
    """
    Configures Bittensor logging to write logs to a file named `validator.log` 
    in the same directory as this Python file
    """
    # Use the directory of this file (so validator log is in the same folder).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bt.logging(config=config, logging_dir=script_dir, record_log=True)

    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.network} with config:")
    bt.logging.info(config)

async def check_registration(wallet, subtensor, netuid):
    """
    Confirm that the wallet hotkey is in the metagraph for the specified netuid.
    Logs an error and exits if it's not registered. Warns if stake is less than 1000.
    """
    metagraph = await subtensor.metagraph(netuid=netuid)
    my_hotkey_ss58 = wallet.hotkey.ss58_address

    if my_hotkey_ss58 not in metagraph.hotkeys:
        bt.logging.error(f"Hotkey {my_hotkey_ss58} is not registered on netuid {netuid}.")
        bt.logging.error("Are you sure you've registered and staked?")
        sys.exit(1) 
    
    uid = metagraph.hotkeys.index(my_hotkey_ss58)
    myStake = metagraph.S[uid]
    bt.logging.info(f"Hotkey {my_hotkey_ss58} found with UID={uid} and stake={myStake}")

    if (myStake < 1000):
        bt.logging.warning(f"Hotkey has less than 1000 stake, unable to validate")

async def get_commitments(subtensor, metagraph, block_hash: str, netuid: int, min_block: int, max_block: int) -> dict:
    """
    Retrieve commitments for all miners on a given subnet (netuid) at a specific block.

    Args:
        subtensor: The subtensor client object.
        netuid (int): The network ID.
        block (int, optional): The block number to query. Defaults to None.

    Returns:
        dict: A mapping from hotkey to a SimpleNamespace containing uid, hotkey,
              block, and decoded commitment data.
    """

    # Gather commitment queries for all validators (hotkeys) concurrently.
    commits = await asyncio.gather(*[
        subtensor.substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey],
            block_hash=block_hash,
        ) for hotkey in metagraph.hotkeys
    ])

    # Process the results and build a dictionary with additional metadata.
    result = {}
    for uid, hotkey in enumerate(metagraph.hotkeys):
        commit = cast(dict, commits[uid])
        if commit and min_block < commit['block'] < max_block:
            result[hotkey] = SimpleNamespace(
                uid=uid,
                hotkey=hotkey,
                block=commit['block'],
                data=decode_metadata(commit)
            )
    return result

def tuple_safe_eval(input_str: str) -> tuple:
    # Limit input size to prevent overly large inputs.
    if len(input_str) > MAX_RESPONSE_SIZE:
        bt.logging.error("Input exceeds allowed size")
        return None
    
    try:
        # Safely evaluate the input string as a Python literal.
        result = literal_eval(input_str)
    except (SyntaxError, ValueError, MemoryError, RecursionError, TypeError) as e:
        bt.logging.error(f"Input is not a valid literal: {e}")
        return None

    # Check that the result is a tuple with exactly two elements.
    if not (isinstance(result, tuple) and len(result) == 2):
        bt.logging.error("Expected a tuple with exactly two elements")
        return None

    # Verify that the first element is an int.
    if not isinstance(result[0], int):
        bt.logging.error("First element must be an int")
        return None
    
    # Verify that the second element is a bytes object.
    if not isinstance(result[1], bytes):
        bt.logging.error("Second element must be a bytes object")
        return None
    
    return result

def decrypt_submissions(current_commitments: dict) -> tuple[dict, dict]:
    """Fetch GitHub submissions and file-specific commit timestamps, then decrypt"""

    file_paths = [commit.data for commit in current_commitments.values() if '/' in commit.data]
    if not file_paths:
        return {}, {}
    
    github_data = {}
    for path in set(file_paths): 
        content_url = f"https://raw.githubusercontent.com/{path}"
        try:
            resp = requests.get(content_url, headers={**GITHUB_HEADERS, "Range": f"bytes=0-{MAX_RESPONSE_SIZE}"})
            content = resp.content if resp.status_code in [200, 206] else None
            if content is None:
                bt.logging.warning(f"Failed to fetch content: {resp.status_code} for https://raw.githubusercontent.com/{path}")
        except Exception as e:
            bt.logging.warning(f"Error fetching content for https://raw.githubusercontent.com/{path}: {e}")
            content = None
        
        # Only fetch timestamp if content was successful
        timestamp = ''
        if content is not None:
            parts = path.split('/')
            if len(parts) >= 4:
                api_url = f"https://api.github.com/repos/{parts[0]}/{parts[1]}/commits"
                try:
                    resp = requests.get(api_url, params={'path': '/'.join(parts[3:]), 'per_page': 1}, headers=GITHUB_HEADERS)
                    commits = resp.json() if resp.status_code == 200 else []
                    timestamp = commits[0]['commit']['committer']['date'] if commits else ''
                    if not timestamp:
                        bt.logging.warning(f"No commit history found for https://github.com/{parts[0]}/{parts[1]}/blob/{parts[2]}/{'/'.join(parts[3:])}")
                except Exception as e:
                    bt.logging.warning(f"Error fetching timestamp for https://github.com/{parts[0]}/{parts[1]}: {e}")
        
        github_data[path] = {'content': content, 'timestamp': timestamp}
    
    encrypted_submissions = {}
    push_timestamps = {}
    
    for commit in current_commitments.values():
        data = github_data.get(commit.data)
        if not data:
            continue
            
        content = data.get('content')
        push_timestamps[commit.uid] = data.get('timestamp', '')
        
        if not content:
            continue
            
        try:
            content_hash = hashlib.sha256(content.decode('utf-8').encode('utf-8')).hexdigest()[:20]
            if commit.data.endswith(f'/{content_hash}.txt'):
                encrypted_content = tuple_safe_eval(content.decode('utf-8', errors='replace'))
                if encrypted_content:
                    encrypted_submissions[commit.uid] = encrypted_content
        except:
            pass
    
    # Decrypt all submissions
    try:
        decrypted_submissions = btd.decrypt_dict(encrypted_submissions)
        decrypted_submissions = {k: v.split(',') for k, v in decrypted_submissions.items() if v is not None}
        # Ensure each UID has the correct number of molecules
        decrypted_submissions = {k: v for k, v in decrypted_submissions.items() if len(v) == config['num_molecules']}
    except Exception as e:
        bt.logging.error(f"Failed to decrypt submissions: {e}")
        decrypted_submissions = {}
    
    bt.logging.info(f"GitHub: {len(file_paths)} paths → {len(decrypted_submissions)} decrypted")
    return decrypted_submissions, push_timestamps

def validate_molecules_and_calculate_entropy(
    uid_to_data: dict[int, dict[str, list]],
    score_dict: dict[int, dict[str, list[list[float]]]],
    config: dict
) -> dict[int, dict[str, list[str]]]:
    """
    Validates molecules for all UIDs and calculates their MACCS entropy.
    Updates the score_dict with entropy values.
    
    Args:
        uid_to_data: Dictionary mapping UIDs to their data including molecules
        score_dict: Dictionary to store scores and entropy
        config: Configuration dictionary containing validation parameters
        
    Returns:
        Dictionary mapping UIDs to their list of valid SMILES strings
    """
    valid_molecules_by_uid = {}
    
    for uid, data in uid_to_data.items():
        valid_smiles = []
        valid_names = []
        
        # Check for duplicate molecules in submission
        if len(data["molecules"]) != len(set(data["molecules"])):
            bt.logging.error(f"UID={uid} submission contains duplicate molecules")
            score_dict[uid]["entropy"] = None
            score_dict[uid]["block_submitted"] = None
            continue
            
        for molecule in data["molecules"]:
            try:
                smiles = get_smiles(molecule)
                if not smiles:
                    bt.logging.error(f"No valid SMILES found for UID={uid}, molecule='{molecule}'")
                    valid_smiles = []
                    valid_names = []
                    break
                
                if get_heavy_atom_count(smiles) < config['min_heavy_atoms']:
                    bt.logging.warning(f"UID={uid}, molecule='{molecule}' has insufficient heavy atoms")
                    valid_smiles = []
                    valid_names = []
                    break

                try:
                    mol = Chem.MolFromSmiles(smiles)
                    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                    if num_rotatable_bonds < config['min_rotatable_bonds'] or num_rotatable_bonds > config['max_rotatable_bonds']:
                        bt.logging.warning(f"UID={uid}, molecule='{molecule}' has an invalid number of rotatable bonds")
                        valid_smiles = []
                        valid_names = []
                        break
                except Exception as e:
                    bt.logging.error(f"Molecule is not parseable by RDKit for UID={uid}, molecule='{molecule}': {e}")
                    valid_smiles = []
                    valid_names = []
                    break
                
                # Check if the molecule is unique for the target protein (weekly_target)
                if not molecule_unique_for_protein_hf(config.weekly_target, smiles):
                    bt.logging.warning(f"UID={uid}, molecule='{molecule}' is not unique for protein '{config.weekly_target}'")
                    valid_smiles = []
                    valid_names = []
                    break
     
                valid_smiles.append(smiles)
                valid_names.append(molecule)
            except Exception as e:
                bt.logging.error(f"Error validating molecule for UID={uid}, molecule='{molecule}': {e}")
                valid_smiles = []
                valid_names = []
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
                    score_dict[uid]["entropy"] = None
                    score_dict[uid]["block_submitted"] = None
                    continue 
            except Exception as e:
                bt.logging.warning(f"Error checking for chemically identical molecules for UID={uid}: {e}")

        
        # Calculate entropy if we have valid molecules
        if valid_smiles:
            try:
                entropy = compute_maccs_entropy(valid_smiles)
                score_dict[uid]["entropy"] = entropy
                valid_molecules_by_uid[uid] = {"smiles": valid_smiles, "names": valid_names}
                score_dict[uid]["block_submitted"] = data["block_submitted"]
            except Exception as e:
                bt.logging.error(f"Error calculating entropy for UID={uid}: {e}")
                score_dict[uid]["entropy"] = None
                score_dict[uid]["block_submitted"] = None
                valid_smiles = []
                valid_names = []
        else:
            score_dict[uid]["entropy"] = None
            score_dict[uid]["block_submitted"] = None
    
    return valid_molecules_by_uid

def count_molecule_names(valid_molecules_by_uid: dict[int, dict[str, list[str]]]) -> dict[str, int]:
    """
    Counts how many times each molecule name occurs across all UIDs in valid_molecules_by_uid.
    
    Args:
        valid_molecules_by_uid: Dictionary mapping UIDs to their valid molecules data,
                              where each UID maps to a dict containing 'names' list
    
    Returns:
        dict: A dictionary mapping molecule names to their occurrence count
    """
    name_counts = {}
    
    for uid_data in valid_molecules_by_uid.values():
        if 'names' in uid_data:
            for name in uid_data['names']:
                name_counts[name] = name_counts.get(name, 0) + 1
                
    return name_counts

def score_all_proteins_batched(
    target_proteins: list[str],
    antitarget_proteins: list[str],
    score_dict: dict[int, dict[str, list[list[float]]]],
    valid_molecules_by_uid: dict[int, dict[str, list[str]]],
    batch_size: int = 32
) -> None:
    """
    Score all molecules against all proteins using efficient batching.
    This replaces the need to call score_protein_for_all_uids multiple times.
    
    Args:
        target_proteins: List of target protein codes
        antitarget_proteins: List of antitarget protein codes
        score_dict: Dictionary to store scores
        valid_molecules_by_uid: Dictionary of valid molecules by UID
        batch_size: Number of molecules to process in each batch
    """
    all_proteins = target_proteins + antitarget_proteins
    
    # Process each protein
    for protein_idx, protein in enumerate(all_proteins):
        is_target = protein_idx < len(target_proteins)
        col_idx = protein_idx if is_target else protein_idx - len(target_proteins)
        
        # Initialize PSICHIC for this protein
        bt.logging.info(f'Initializing model for protein code: {protein}')
        protein_sequence = get_sequence_from_protein_code(protein)
        
        try:
            psichic.run_challenge_start(protein_sequence)
            bt.logging.info('Model initialized successfully.')
        except Exception as e:
            try:
                os.system(f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/TREAT2/model.pt')} https://huggingface.co/Metanova/TREAT-2/resolve/main/model.pt")
                psichic.run_challenge_start(protein_sequence)
                bt.logging.info('Model initialized successfully.')
            except Exception as e:
                bt.logging.error(f'Error initializing model: {e}')
                # Set all scores to -inf for this protein
                for uid in score_dict:
                    num_molecules = len(valid_molecules_by_uid.get(uid, {}).get('smiles', []))
                    if num_molecules == 0:
                        num_molecules = len(uid_to_data.get(uid, {}).get("molecules", []))
                    score_dict[uid]["target_scores" if is_target else "antitarget_scores"][col_idx] = [-math.inf] * num_molecules
                continue
        
        # Collect all unique molecules across all UIDs
        unique_molecules = {}  # {smiles: [(uid, mol_idx), ...]}
        
        for uid, valid_molecules in valid_molecules_by_uid.items():
            if not valid_molecules.get('smiles'):
                # Set -inf scores for UIDs with no valid molecules
                num_molecules = len(uid_to_data.get(uid, {}).get("molecules", []))
                score_dict[uid]["target_scores" if is_target else "antitarget_scores"][col_idx] = [-math.inf] * num_molecules
                continue
            
            for mol_idx, smiles in enumerate(valid_molecules['smiles']):
                if smiles not in unique_molecules:
                    unique_molecules[smiles] = []
                unique_molecules[smiles].append((uid, mol_idx))
        
        # Process unique molecules in batches
        unique_smiles_list = list(unique_molecules.keys())
        molecule_scores = {}  # {smiles: score}
        
        for batch_start in range(0, len(unique_smiles_list), batch_size):
            batch_end = min(batch_start + batch_size, len(unique_smiles_list))
            batch_molecules = unique_smiles_list[batch_start:batch_end]
            
            try:
                # Score the batch
                results_df = psichic.run_validation(batch_molecules)
                
                if not results_df.empty and len(results_df) == len(batch_molecules):
                    for idx, smiles in enumerate(batch_molecules):
                        val = results_df.iloc[idx].get('predicted_binding_affinity')
                        score_value = float(val) if val is not None else -math.inf
                        molecule_scores[smiles] = score_value
                else:
                    bt.logging.warning(f"Unexpected results for batch, falling back to individual scoring")
                    for smiles in batch_molecules:
                        molecule_scores[smiles] = score_molecule_individually(smiles)
            except Exception as e:
                bt.logging.error(f"Error scoring batch: {e}")
                for smiles in batch_molecules:
                    molecule_scores[smiles] = score_molecule_individually(smiles)
        
        # Distribute scores to all UIDs
        for uid, valid_molecules in valid_molecules_by_uid.items():
            if not valid_molecules.get('smiles'):
                continue
            
            uid_scores = []
            for smiles in valid_molecules['smiles']:
                score = molecule_scores.get(smiles, -math.inf)
                uid_scores.append(score)
            
            if is_target:
                score_dict[uid]["target_scores"][col_idx] = uid_scores
            else:
                score_dict[uid]["antitarget_scores"][col_idx] = uid_scores
        
        bt.logging.info(f"Completed scoring for protein {protein}: {len(unique_molecules)} unique molecules")


def score_molecule_individually(smiles: str) -> float:
    """Helper function to score a single molecule."""
    try:
        results_df = psichic.run_validation([smiles])
        if not results_df.empty:
            val = results_df.iloc[0].get('predicted_binding_affinity')
            return float(val) if val is not None else -math.inf
        else:
            return -math.inf
    except Exception as e:
        bt.logging.error(f"Error scoring molecule {smiles}: {e}")
        return -math.inf

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
    Returns the winning UID or None if no valid scores are found.
    """
    best_score = -math.inf
    best_uid = None
    
    dynamic_entropy_weight = calculate_dynamic_entropy(
        starting_weight=config['entropy_start_weight'],
        step_size=config['entropy_step_size'],
        start_epoch=config['entropy_start_epoch'],
        current_epoch=current_epoch
    )
    
    # Go through each UID scored
    for uid, data in valid_molecules_by_uid.items():
        print(score_dict[uid])
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
                molecule_scores.append(-math.inf)
                combined_molecule_scores.append(-math.inf)
                molecule_scores_after_repetition.append(-math.inf)
                continue
            avg_target = sum(target_scores_for_mol) / len(target_scores_for_mol)

            # Calculate average antitarget score for this molecule
            antitarget_scores_for_mol = [antitarget_list[mol_idx] for antitarget_list in antitargets]
            if any(score == -math.inf for score in antitarget_scores_for_mol):
                molecule_scores.append(-math.inf)
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
                
        # Apply entropy bonus for scores above threshold
        if score_dict[uid]['final_score'] > config['entropy_bonus_threshold'] and entropy is not None:
            score_dict[uid]['final_score'] = score_dict[uid]['final_score'] * (1 + (dynamic_entropy_weight * entropy))
        
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
            f"  Target scores per molecule: {target_scores_per_mol}",
            f"  Antitarget scores per molecule: {antitarget_scores_per_mol}",
            f"  Entropy: {entropy}",
            f"  Dynamic entropy weight: {dynamic_entropy_weight}",
            f"  Final score: {score_dict[uid]['final_score']}"
        ]
        bt.logging.info("\n".join(log_lines))

    return score_dict

def determine_winner(score_dict: dict[int, dict[str, list[list[float]]]]) -> Optional[int]:
    """
    Determines the winning UID based on final score.
    In case of ties, earliest submission time is used as the tiebreaker.
    """
    best_score = -math.inf
    best_uids = []
    
    # Find highest final score
    for uid, data in score_dict.items():
        if 'final_score' not in data:
            continue
            
        final_score = round(data['final_score'], 4)
        
        if final_score > best_score:
            best_score = final_score
            best_uids = [uid]
        elif final_score == best_score:
            best_uids.append(uid)
    
    if not best_uids:
        bt.logging.info("No valid winner found (all scores -inf or no submissions).")
        return None
    
    # If only one winner, return it
    if len(best_uids) == 1:
        winner_block = score_dict[best_uids[0]].get('block_submitted')
        current_epoch = winner_block // 361 if winner_block else None
        bt.logging.info(f"Epoch {current_epoch} winner: UID={best_uids[0]}, winning_score={best_score}")
        return best_uids[0]
    
    def parse_timestamp(uid):
        ts = score_dict[uid].get('push_time', '')
        try:
            return datetime.datetime.fromisoformat(ts)
        except Exception as e:
            bt.logging.warning(f"Failed to parse timestamp '{ts}' for UID={uid}: {e}")
            return datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)
    
    # Sort by block number first, then push time, then uid to ensure deterministic result
    winner = sorted(best_uids, key=lambda uid: (
        score_dict[uid].get('block_submitted', float('inf')), 
        parse_timestamp(uid), 
        uid
    ))[0]
    
    winner_block = score_dict[winner].get('block_submitted')
    current_epoch = winner_block // 361 if winner_block else None
    push_time = score_dict[winner].get('push_time', '')
    
    tiebreaker_message = f"Epoch {current_epoch} tiebreaker winner: UID={winner}, score={best_score}, block={winner_block}"
    if push_time:
        tiebreaker_message += f", push_time={push_time}"
        
    bt.logging.info(tiebreaker_message)
        
    return winner



async def main(config):
    """
    Main routine that continuously checks for the end of an epoch to perform:
        - Setting a new commitment.
        - Retrieving past commitments.
        - Selecting the best protein/molecule pairing based on stakes and scores.
        - Setting new weights accordingly.

    Args:
        config: Configuration object for subtensor and wallet.
    """
    wallet = bt.wallet(config=config)

    # Initialize the asynchronous subtensor client.
    subtensor = bt.async_subtensor(network=config.network)
    await subtensor.initialize()

    # Check if the hotkey is registered and has at least 1000 stake.
    await check_registration(wallet, subtensor, config.netuid)

    # Check GitHub token status 
    if os.environ.get('GITHUB_TOKEN'):
        bt.logging.info("GitHub authentication token found")
        GITHUB_HEADERS['Authorization'] = f"token {os.environ['GITHUB_TOKEN']}"
    else:
        bt.logging.warning("No GITHUB_TOKEN found. Using unauthenticated requests. For better performance and higher rate limits, add 'GITHUB_TOKEN=your_personal_access_token' to your .env file.")

    # Initialize auto-updater if enabled via environment variable
    if os.environ.get('AUTO_UPDATE') == '1':
        updater = AutoUpdater(logger=bt.logging)
        asyncio.create_task(updater.start_update_loop())
        bt.logging.info(f"Auto-updater enabled, checking for updates every {updater.UPDATE_INTERVAL} seconds")
    else:
        bt.logging.info("Auto-updater disabled. Set AUTO_UPDATE=1 to enable.")

    while True:
        try:
            # Fetch the current metagraph for the given subnet (netuid 68).
            metagraph = await subtensor.metagraph(config.netuid)
            bt.logging.debug(f'Found {metagraph.n} nodes in network')
            current_block = await subtensor.get_current_block()

            # Check if the current block marks the end of an epoch.
            if current_block % config.epoch_length == 0:

                # Reload config.yaml
                config.update(load_config())

                try:
                    start_block = current_block - config.epoch_length
                    start_block_hash = await subtensor.determine_block_hash(start_block)
                    
                    current_epoch = current_block // config.epoch_length

                    proteins = get_challenge_proteins_from_blockhash(
                        block_hash=start_block_hash,
                        weekly_target=config.weekly_target,
                        num_antitargets=config.num_antitargets
                    )
                    target_proteins = proteins["targets"]
                    antitarget_proteins = proteins["antitargets"]

                    bt.logging.info(f"Scoring using target proteins: {target_proteins}, antitarget proteins: {antitarget_proteins}")

                except Exception as e:
                    bt.logging.error(f"Error generating challenge proteins: {e}")
                    continue

                # Retrieve commitments from current epoch only
                current_block_hash = await subtensor.determine_block_hash(current_block)
                current_commitments = await get_commitments(
                    subtensor, 
                    metagraph, 
                    current_block_hash, 
                    netuid=config.netuid,
                    min_block=start_block,
                    max_block=current_block - config.no_submission_blocks
                )
                bt.logging.debug(f"Current epoch commitments: {len(current_commitments)}")

                # Decrypt submissions
                decrypted_submissions, push_timestamps = decrypt_submissions(current_commitments)

                uid_to_data = {}
                for hotkey, commit in current_commitments.items():
                    uid = commit.uid
                    molecules = decrypted_submissions.get(uid)
                    if molecules is not None:
                        uid_to_data[uid] = {
                            "molecules": molecules,
                            "block_submitted": commit.block,
                            "push_time": push_timestamps.get(uid, '')
                        }
                    else:
                        bt.logging.error(f"No decrypted submission found for UID: {uid}")

                if not uid_to_data:
                    bt.logging.info("No valid submissions found this epoch.")
                    await asyncio.sleep(1)
                    continue

                score_dict = {
                    uid: {
                        "target_scores": [[] for _ in range(len(target_proteins))],
                        "antitarget_scores": [[] for _ in range(len(antitarget_proteins))],
                        "entropy": None,
                        "block_submitted": None,
                        "push_time": uid_to_data[uid].get("push_time", '')
                    }
                    for uid in uid_to_data
                }

                # Validate all molecules and calculate entropy for UIDs with all valid molecules
                valid_molecules_by_uid = validate_molecules_and_calculate_entropy(
                    uid_to_data=uid_to_data,
                    score_dict=score_dict,
                    config=config
                )

                # Count molecule names occurrences
                molecule_name_counts = count_molecule_names(valid_molecules_by_uid)

                # Score all target proteins then all antitarget proteins one protein at a time
                score_all_proteins_batched(
                    target_proteins=target_proteins,
                    antitarget_proteins=antitarget_proteins,
                    score_dict=score_dict,
                    valid_molecules_by_uid=valid_molecules_by_uid,
                    batch_size=32
                    )

                score_dict = calculate_final_scores(score_dict, valid_molecules_by_uid, molecule_name_counts, config, current_epoch)

                winning_uid = determine_winner(score_dict)

                set_weights_call_block = await subtensor.get_current_block()
                monitor_validator(
                    score_dict=score_dict,
                    metagraph=metagraph,
                    current_epoch=current_epoch,
                    current_block=set_weights_call_block,
                    validator_hotkey=wallet.hotkey.ss58_address,
                    winning_uid=winning_uid
                )

                if winning_uid is not None:
                    try:
                        external_script_path =  os.path.abspath(os.path.join(os.path.dirname(__file__), "set_weight_to_uid.py"))
                        cmd = [
                            "python", 
                            external_script_path, 
                            f"--target_uid={winning_uid}",
                            f"--wallet_name={config.wallet.name}",
                            f"--wallet_hotkey={config.wallet.hotkey}",
                        ]
                        bt.logging.info(f"Calling: {' '.join(cmd)}")
                    
                        proc = subprocess.run(cmd, capture_output=True, text=True)
                        bt.logging.info(f"Output from set_weight_to_uid:\n{proc.stdout}")
                        bt.logging.info(f"Errors from set_weight_to_uid:\n{proc.stderr}")
                        if proc.returncode != 0:
                            bt.logging.error(f"Script returned non-zero exit code: {proc.returncode}")

                    except Exception as e:
                        bt.logging.error(f"Error calling set_weight_to_uid script: {e}")
                else:
                    bt.logging.warning("No valid molecule commitment found for current epoch.")
                    await asyncio.sleep(1)
                    continue
                
            # keep validator alive
            elif current_block % (config.epoch_length/2) == 0:
                subtensor = bt.async_subtensor(network=config.network)
                await subtensor.initialize()
                bt.logging.info("Validator reset subtensor connection.")
                await asyncio.sleep(12) # Sleep for 1 block to avoid unncessary re-connection
            
            else:
                bt.logging.info(f"Waiting for epoch to end... {config.epoch_length - (current_block % config.epoch_length)} blocks remaining.")
                await asyncio.sleep(1)
        except Exception as e:
            bt.logging.error(f"Error in main loop: {e}")
            await asyncio.sleep(3)


if __name__ == "__main__":
    config = get_config()
    setup_logging(config)
    asyncio.run(main(config))
