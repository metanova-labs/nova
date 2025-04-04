import asyncio
from ast import literal_eval
import math
import os
import sys
import argparse
import binascii
from typing import cast, Optional
from types import SimpleNamespace
import bittensor as bt
from substrateinterface import SubstrateInterface
import requests
import hashlib
import subprocess
from dotenv import load_dotenv
from bittensor.core.chain_data.utils import decode_metadata

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from config.config_loader import load_config
from my_utils import get_smiles, get_sequence_from_protein_code, get_heavy_atom_count, get_challenge_proteins_from_blockhash
from PSICHIC.wrapper import PsichicWrapper
from btdr import QuicknetBittensorDrandTimelock

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
    config.epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value

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

async def get_commitments(subtensor, metagraph, block_hash: str, netuid: int) -> dict:
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
        if commit:
            result[hotkey] = SimpleNamespace(
                uid=uid,
                hotkey=hotkey,
                block=commit['block'],
                data=decode_metadata(commit)
            )
    return result

def tuple_safe_eval(input_str: str) -> tuple:
    # Limit input size to prevent overly large inputs.
    if len(input_str) > 1024:
        bt.logging.error("Input exceeds allowed size")
        return None
    
    try:
        # Safely evaluate the input string as a Python literal.
        result = literal_eval(input_str)
    except (SyntaxError, ValueError) as e:
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

def decrypt_submissions(current_commitments: dict, headers: dict = {"Range": "bytes=0-1024"}) -> dict:
    """
    Decrypts submissions from validators by fetching encrypted content from GitHub URLs and decrypting them.

    Args:
        current_commitments (dict): A dictionary of miner commitments where each value contains:
            - uid: Miner's unique identifier
            - data: GitHub URL path containing the encrypted submission
            - Other commitment metadata
        headers (dict, optional): HTTP request headers for fetching content. 
            Defaults to {"Range": "bytes=0-1024"} to limit response size.

    Returns:
        dict: A dictionary of decrypted submissions mapped by validator UIDs.
            Empty if no valid submissions were found or decryption failed.

    Note:
        - Only processes commitments where data contains a '/' (indicating a GitHub URL)
        - Uses btd.decrypt_dict for decryption of the fetched submissions
        - Logs errors for failed HTTP requests and submission counts
    """
    encrypted_submissions = {}
    for commit in current_commitments.values():
        if '/' in commit.data: # Filter only url submissions
            try:
                full_url = f"https://raw.githubusercontent.com/{commit.data}"
                response = requests.get(full_url, headers=headers)
                if response.status_code in [200, 206]:
                    encrypted_content = response.content
                    content_hash = hashlib.sha256(encrypted_content.decode('utf-8').encode('utf-8')).hexdigest()[:20]

                    # Disregard any submissions that don't match the expected filename
                    if not full_url.endswith(f'/{content_hash}.txt'):
                        bt.logging.error(f"Filename for {commit.uid} is not compatible with expected content hash")
                        continue
                    encrypted_content = encrypted_content.decode('utf-8', errors='replace')

                    # Safely evaluate the input string as a Python literal.
                    encrypted_content = tuple_safe_eval(encrypted_content)
                    if encrypted_content is None:
                        bt.logging.error(f"Encrypted content for {commit.uid} is not a tuple")
                        continue

                    encrypted_submissions[commit.uid] = (encrypted_content[0], encrypted_content[1])
                else:
                    bt.logging.error(f"Error fetching encrypted submission: {response.status_code}")
                    bt.logging.error(f"uid: {commit.uid}, commited data: {commit.data}")
                    continue
            except Exception as e:
                bt.logging.error(f"Error handling submission for uid {commit.uid}: {e}")
                continue

    bt.logging.info(f"Encrypted submissions: {len(encrypted_submissions)}")
    
    try:
        decrypted_submissions = btd.decrypt_dict(encrypted_submissions)
    except Exception as e:
        bt.logging.error(f"Failed to decrypt submissions: {e}")
        decrypted_submissions = {}

    bt.logging.info(f"Decrypted submissions: {len(decrypted_submissions)}")
            
    return decrypted_submissions

def score_protein_for_all_uids(
    protein: str,
    score_dict: dict[int, dict[str, list[float]]],
    uid_to_data: dict[int, dict[str, str]],
    col_idx: int,
    is_target: bool = True
) -> None:
    """
    Initialize PSICHIC once for 'protein' and score each UID's molecule, 
    storing results in the appropriate (target or antitarget) index of 'score_dict'.
    """
    # Initialize PSICHIC for new protein
    bt.logging.info(f'Initializing model for protein code: {protein}')
    protein_sequence = get_sequence_from_protein_code(protein)
    try:
        psichic.run_challenge_start(protein_sequence)
        bt.logging.info('Model initialized successfully.')
    except Exception as e:
        try:
            os.system(f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt')} https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt")
            psichic.run_challenge_start(protein_sequence)
            bt.logging.info('Model initialized successfully.')
        except Exception as e:
            bt.logging.error(f'Error initializing model: {e}')
            for uid in uid_to_data:
                score_dict[uid]["target_scores" if is_target else "antitarget_scores"][col_idx] = -math.inf
            return # If we can't initialize set all scores to -inf

    # Score each UID's molecule
    for uid, data in uid_to_data.items():
        score_value = -math.inf
        try:
            smiles = get_smiles(data["molecule"])
        except Exception as e:
            bt.logging.error(f"Error getting SMILES for UID={uid}, molecule='{data['molecule']}': {e}")
            smiles = None

        if not smiles:
            bt.logging.debug(f"No SMILES found for UID={uid}, molecule='{data['molecule']}'.")

        elif get_heavy_atom_count(smiles) < config['min_heavy_atoms']:
            bt.logging.info(f"UID: {uid}, SMILES: {smiles} has less than {config['min_heavy_atoms']} heavy atoms, scoring -inf")

        else:
            try:
                results_df = psichic.run_validation([smiles])
                if not results_df.empty:
                    val = results_df.iloc[0].get('predicted_binding_affinity')
                    score_value = float(val) if val is not None else -math.inf
                else:
                    bt.logging.warning(f"PSICHIC returned an empty DataFrame for UID={uid}.")
            except Exception as e:
                bt.logging.error(f"Error scoring UID={uid}, molecule='{data['molecule']}': {e}")

        # Store the score in the correct list
        if is_target:
            score_dict[uid]["target_scores"][col_idx] = score_value
        else:
            score_dict[uid]["antitarget_scores"][col_idx] = score_value


def determine_winner(
    score_dict: dict[int, dict[str, list[float]]],
    uid_to_data: dict[int, dict[str, int]]
) -> list[int]:
    """
    Logs target/antitarget scores for each UID, identifies winning UIDs.
    If multiple UIDs have the same highest score and submitted in the same block,
    they will all be returned as winners to split the reward.
    
    Returns:
        list[int]: List of winning UIDs. Empty if no valid scores are found.
    """
    best_score = -math.inf
    tied_winners = []
    best_block_submitted = math.inf  # Track earliest block in tie-break

    # Go through each UID scored
    for uid, data in uid_to_data.items():
        targets = score_dict[uid]['target_scores']
        antitargets = score_dict[uid]['antitarget_scores']
        submission_block = data["block_submitted"]

        # Replace None with -inf
        targets = [-math.inf if s is None else s for s in targets]
        antitargets = [-math.inf if s is None else s for s in antitargets]

        # Compute final score
        if -math.inf not in targets and -math.inf not in antitargets:
            target_sum = sum(targets)
            target_score = target_sum / len(targets)

            antitarget_sum = sum(antitargets)
            antitarget_score = antitarget_sum / len(antitargets)

            final_score = (config['target_weight'] * target_score) - (config['antitarget_weight'] * antitarget_score)
        else:
            final_score = -math.inf

        # Round the final score to 4 decimal places for comparison
        rounded_final_score = round(final_score, 4) if final_score != -math.inf else -math.inf
        rounded_best_score = round(best_score, 4) if best_score != -math.inf else -math.inf

        # Log details (use original unrounded score for logging)
        bt.logging.info(
            f"UID={uid} -> target_scores={targets}, "
            f"antitarget_scores={antitargets}, "
            f"final_score={final_score} (rounded to {rounded_final_score}), "
            f"block_submitted={submission_block}"
        )

        # Check if this is a new best score (using rounded scores)
        if rounded_final_score > rounded_best_score:
            best_score = final_score  # Keep original precision for future comparisons
            tied_winners = [uid]
            best_block_submitted = submission_block
        # If it's tied with the current best score (using rounded scores)
        elif rounded_final_score == rounded_best_score:
            # If submitted in an earlier block, this one becomes the sole winner
            if submission_block < best_block_submitted:
                tied_winners = [uid]
                best_block_submitted = submission_block
                best_score = final_score  # Update best_score to this one
            # If submitted in the same block as our current best, add to the tie list
            elif submission_block == best_block_submitted:
                tied_winners.append(uid)

    # Log final result
    if tied_winners and best_score != -math.inf:
        rounded_best_score = round(best_score, 4)
        if len(tied_winners) == 1:
            winner_uid = tied_winners[0]
            bt.logging.info(
                f"Winner: UID={winner_uid}, "
                f"molecule={uid_to_data[winner_uid]['molecule']}, "
                f"SMILES={get_smiles(uid_to_data[winner_uid]['molecule'])}, "
                f"block_submitted={best_block_submitted}, "
                f"winning_score={best_score} (rounded to {rounded_best_score})"
            )
        else:
            bt.logging.info(
                f"Tie detected! {len(tied_winners)} UIDs tied with score {best_score} (rounded to {rounded_best_score}) in block {best_block_submitted}: {tied_winners}"
            )
            for winner_uid in tied_winners:
                bt.logging.info(
                    f"Tied Winner: UID={winner_uid}, "
                    f"molecule={uid_to_data[winner_uid]['molecule']}, "
                    f"SMILES={get_smiles(uid_to_data[winner_uid]['molecule'])}"
                )
    else:
        bt.logging.info("No valid winner found (all scores -inf or no submissions).")

    return tied_winners


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

    tolerance = 3 # block tolerance window for validators to commit protein

    while True:
        try:
            # Fetch the current metagraph for the given subnet (netuid 68).
            metagraph = await subtensor.metagraph(config.netuid)
            bt.logging.debug(f'Found {metagraph.n} nodes in network')
            current_block = await subtensor.get_current_block()

            # Check if the current block marks the end of an epoch (using a 360-block interval).
            if current_block % config.epoch_length == 0:

                try:
                    start_block = current_block - config.epoch_length
                    start_block_hash = await subtensor.determine_block_hash(start_block)

                    proteins = get_challenge_proteins_from_blockhash(
                        block_hash=start_block_hash,
                        num_targets=config.num_targets,
                        num_antitargets=config.num_antitargets
                    )
                    target_proteins = proteins["targets"]
                    antitarget_proteins = proteins["antitargets"]

                    bt.logging.info(f"Scoring using target proteins: {target_proteins}, antitarget proteins: {antitarget_proteins}")

                except Exception as e:
                    bt.logging.error(f"Error generating challenge proteins: {e}")
                    continue

                # Retrieve the latest commitments (current epoch).
                current_block_hash = await subtensor.determine_block_hash(current_block)
                current_commitments = await get_commitments(subtensor, metagraph, current_block_hash, netuid=config.netuid)
                bt.logging.debug(f"Current commitments: {len(list(current_commitments.values()))}")

                # Decrypt submissions
                decrypted_submissions = decrypt_submissions(current_commitments)

                uid_to_data = {}
                for hotkey, commit in current_commitments.items():
                    # Ensure submission is from the current epoch
                    if current_block - commit.block <= config.epoch_length:
                        uid = commit.uid
                        molecule = decrypted_submissions.get(uid)
                        if molecule is not None:
                            uid_to_data[uid] = {
                                "molecule": molecule,
                                "block_submitted": commit.block
                            }
                        else:
                            bt.logging.error(f"No decrypted submission found for UID: {uid}")

                if not uid_to_data:
                    bt.logging.info("No valid submissions found this epoch.")
                    await asyncio.sleep(1)
                    continue

                score_dict = {
                    uid: {
                        "target_scores": [None] * len(target_proteins),
                        "antitarget_scores": [None] * len(antitarget_proteins)
                    }
                    for uid in uid_to_data
                }

                # Score all target proteins then all antitarget proteins one protein at a time
                for i, target_protein in enumerate(target_proteins):
                    score_protein_for_all_uids(
                        protein=target_protein,
                        score_dict=score_dict,
                        uid_to_data=uid_to_data,
                        col_idx=i,
                        is_target=True
                    )
                for j, anti_protein in enumerate(antitarget_proteins):
                    score_protein_for_all_uids(
                        protein=anti_protein,
                        score_dict=score_dict,
                        uid_to_data=uid_to_data,
                        col_idx=j,
                        is_target=False
                    )

                winning_uids = determine_winner(score_dict, uid_to_data)

                if winning_uids:
                    try:
                        external_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "set_weight_to_uids.py"))
                        # Join winning UIDs as comma-separated list
                        target_uids_str = ",".join(map(str, winning_uids))
                        
                        cmd = [
                            "python", 
                            external_script_path, 
                            f"--target_uids={target_uids_str}",
                            f"--wallet_name={config.wallet.name}",
                            f"--wallet_hotkey={config.wallet.hotkey}",
                        ]
                        bt.logging.info(f"Calling: {' '.join(cmd)}")
                    
                        proc = subprocess.run(cmd, capture_output=True, text=True)
                        bt.logging.info(f"Output from set_weight_to_uids:\n{proc.stdout}")
                        bt.logging.info(f"Errors from set_weight_to_uids:\n{proc.stderr}")
                        if proc.returncode != 0:
                            bt.logging.error(f"Script returned non-zero exit code: {proc.returncode}")

                    except Exception as e:
                        bt.logging.error(f"Error calling set_weight_to_uids script: {e}")
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
