import os
import sys
import math
import random
import argparse
import asyncio
import datetime
import tempfile
import traceback
import base64
import hashlib

from typing import Any, Dict, List, Optional, Tuple, cast
from types import SimpleNamespace

from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.errors import MetadataError
from substrateinterface import SubstrateInterface
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

from config.config_loader import load_config
from utils.constants import ALLOWED_AAS
from utils import (
    get_sequence_from_protein_code,
    upload_file_to_github,
    get_challenge_params_from_blockhash,
    get_heavy_atom_count,
    compute_maccs_entropy,

)
from utils.btdr import QuicknetBittensorDrandTimelock
from combinatorial_db.reactions import get_random_reaction_product

'''
This example miner contains all logic necessary for submitting a response to the chain
in a format that will be accepted by the validator.

Each miner should implement their own logic for selecting candidate molecules and nanobodies.
The placeholder selection logic here is a random selection of a small molecule from the combinatorial db 
and a random mutation of a base nanobody sequence (therefore not competitive).
'''

# ----------------------------------------------------------------------------
# 1. CONFIG & ARGUMENT PARSING
# ----------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments and merges with config defaults.

    Returns:
        argparse.Namespace: The combined configuration object.
    """
    parser = argparse.ArgumentParser()
    # Add override arguments for network.
    parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK'), help='Network to use')
    # Adds override arguments for netuid.
    parser.add_argument('--netuid', type=int, default=379, help="The chain subnet uid.") #68 for mainnet
    # Bittensor standard argument additions.
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

    # Parse combined config
    config = bt.config(parser)

    # Load protein selection params
    config.update(load_config())

    # Final logging dir
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey_str,
            config.netuid,
            'miner',
        )
    )

    # Ensure the logging directory exists.
    os.makedirs(config.full_path, exist_ok=True)
    return config


def load_github_path() -> str:
    """
    Constructs the path for GitHub operations from environment variables.
    
    Returns:
        str: The fully qualified GitHub path (owner/repo/branch/path).
    Raises:
        ValueError: If the final path exceeds 100 characters.
    """
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')  # e.g., "nova"
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH')  # e.g., "main"
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER')  # e.g., "metanova-labs"
    github_repo_path = os.environ.get('GITHUB_REPO_PATH')  # e.g., "data/results" or ""

    if github_repo_name is None or github_repo_branch is None or github_repo_owner is None:
        raise ValueError("Missing one or more GitHub environment variables (GITHUB_REPO_*)")

    if github_repo_path == "":
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}"
    else:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}/{github_repo_path}"

    if len(github_path) > 100:
        raise ValueError("GitHub path is too long. Please shorten it to 100 characters or less.")

    return github_path


# ----------------------------------------------------------------------------
# 2. LOGGING SETUP
# ----------------------------------------------------------------------------

def setup_logging(config: argparse.Namespace) -> None:
    """
    Sets up Bittensor logging.

    Args:
        config (argparse.Namespace): The miner configuration object.
    """
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running miner for subnet: {config.netuid} on network: {config.subtensor.network} with config:")
    bt.logging.info(config)


# ----------------------------------------------------------------------------
# 3. BITTENSOR & NETWORK SETUP
# ----------------------------------------------------------------------------

async def setup_bittensor_objects(config: argparse.Namespace) -> Tuple[Any, Any, Any, int, int]:
    """
    Initializes wallet, subtensor, and metagraph. Fetches the epoch length
    and calculates the miner UID.

    Args:
        config (argparse.Namespace): The miner configuration object.

    Returns:
        tuple: A 5-element tuple of
            (wallet, subtensor, metagraph, miner_uid, epoch_length).
    """
    bt.logging.info("Setting up Bittensor objects.")

    # Initialize wallet
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # Initialize subtensor (asynchronously)
    try:
        async with bt.async_subtensor(network=config.network) as subtensor:
            bt.logging.info(f"Connected to subtensor network: {config.network}")
            
            # Sync metagraph
            metagraph = await subtensor.metagraph(config.netuid)
            await metagraph.sync()
            bt.logging.info(f"Metagraph synced successfully.")

            bt.logging.info(f"Subtensor: {subtensor}")
            bt.logging.info(f"Metagraph synced: {metagraph}")

            # Get miner UID
            miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            bt.logging.info(f"Miner UID: {miner_uid}")

            # Query epoch length
            node = SubstrateInterface(url=config.network)
            # Set epoch_length to tempo + 1
            epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value + 1
            bt.logging.info(f"Epoch length query successful: {epoch_length} blocks")

        return wallet, subtensor, metagraph, miner_uid, epoch_length
    except Exception as e:
        bt.logging.error(f"Failed to setup Bittensor objects: {e}")
        bt.logging.error("Please check your network connection and the subtensor network status")
        raise

# ----------------------------------------------------------------------------
# 4. ADD YOUR OWN CODE FOR MINING
# ----------------------------------------------------------------------------

def get_random_small_molecule(
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """
    Uses combinatorial_db.reactions to pick a random reaction and valid
    reactant IDs, then returns the product name in the format
    'rxn:rxn_id:mol1_id:mol2_id' or 'rxn:rxn_id:mol1_id:mol2_id:mol3_id'.
    Returns None if the DB is missing or no valid combination is found.
    """
    return get_random_reaction_product(rng=rng)


def mutate_sequence(
    sequence: str,
    num_mutations: int,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Mutates a variable number of positions in a sequence by replacing residues
    with one of the allowed amino acids from utils.constants.ALLOWED_AAS.

    Args:
        sequence: The base amino acid sequence (e.g. nanobody one-letter codes).
        num_mutations: Number of positions to mutate. Clamped to [0, len(sequence)].
        rng: Optional random.Random instance for reproducible randomness.

    Returns:
        A new string with exactly num_mutations positions changed to random
        allowed AAs (always to a different residue at each chosen position).
    """
    if not sequence or num_mutations <= 0:
        return sequence
    rng = rng or random
    allowed = list(ALLOWED_AAS)
    seq_list = list(sequence)
    length = len(seq_list)
    num_mutations = min(num_mutations, length)
    indices = rng.sample(range(length), num_mutations)
    for i in indices:
        current = seq_list[i]
        choices = [a for a in allowed if a != current]
        if not choices:
            continue
        seq_list[i] = rng.choice(choices)
    return "".join(seq_list)


def dummy_get_candidate_products() -> Tuple[str, str]:
    """
    Returns a dummy candidate small molecule and nanobody.
    If n>1, items should be separated by ',' e.g. 'rxn:3:49485:2099:70633,rxn:3:49485:2099:70634'
    """
    try:
        candidate_small_molecules = get_random_small_molecule() #'rxn:3:49485:2099:70633'
    except Exception as e:
        bt.logging.error(f"Error getting candidate small molecule: {e}")
        candidate_small_molecules = "~"

    try:
        base_nanobody = 'EVELLASGGDLVQPGGSLRLSCAASGFTFSTYAMSWVRQAPGKGLERVSRVNQNGGTTTYADAMKGRFTISRDNAKNTLYLQMINVKPEDTAIYYCARWDGGSWSTDPWGRGTLVTVS'
        candidate_nanobodies = mutate_sequence(base_nanobody, num_mutations=random.randint(1, 30))
    except Exception as e:
        bt.logging.error(f"Error getting candidate nanobody: {e}")
        candidate_nanobodies = "~"
    
    return candidate_small_molecules, candidate_nanobodies

# ----------------------------------------------------------------------------
# 5. RESPONSE SUBMISSION LOGIC
# ----------------------------------------------------------------------------

async def submit_response(state: Dict[str, Any]) -> None:
    """
    Encrypts and submits the current candidate product as a chain commitment and uploads
    the encrypted response to GitHub. If the chain accepts the commitment, we finalize it.

    Args:
        state (dict): Shared state dictionary containing references to:
            'bdt', 'miner_uid', 'candidate_product', 'subtensor', 'wallet', 'config',
            'github_path', etc.
    """
    candidate_small_molecules = state['candidate_small_molecules']
    candidate_nanobodies = state['candidate_nanobodies']
    if not candidate_small_molecules and not candidate_nanobodies:
        bt.logging.warning("No candidate small molecules or nanobodies to submit")
        return

    bt.logging.info(f"Starting submission process for small molecules: {candidate_small_molecules} and nanobodies: {candidate_nanobodies}")
    
    # 1) Encrypt the response

    message = f"{candidate_small_molecules}|{candidate_nanobodies}"
    current_block = await state['subtensor'].get_current_block()
    encrypted_response = state['bdt'].encrypt(state['miner_uid'], message, current_block)
    bt.logging.info(f"Encrypted response generated successfully")

    # 2) Create temp file, write content
    tmp_file = tempfile.NamedTemporaryFile(delete=True)
    with open(tmp_file.name, 'w+') as f:
        f.write(str(encrypted_response))
        f.flush()

        # Read, base64-encode
        f.seek(0)
        content_str = f.read()
        encoded_content = base64.b64encode(content_str.encode()).decode()

        # Generate short hash-based filename
        filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
        commit_content = f"{state['github_path']}/{filename}.txt"
        bt.logging.info(f"Prepared commit content: {commit_content}")

    # 3) Attempt chain commitment
    bt.logging.info(f"Attempting chain commitment...")
    try: 
        commitment_status = await state['subtensor'].set_commitment(
            wallet=state['wallet'],
            netuid=state['config'].netuid,
            data=commit_content
        )
        bt.logging.info(f"Chain commitment status: {commitment_status}")
    except MetadataError:
        bt.logging.info("Too soon to commit again. Will keep looking for better candidates.")
        return

    # 4) If chain commitment success, upload to GitHub
    if commitment_status:
        try:
            bt.logging.info(f"Commitment set successfully for {commit_content}")
            bt.logging.info("Attempting GitHub upload...")
            github_status = upload_file_to_github(filename, encoded_content)
            if github_status:
                bt.logging.info(f"File uploaded successfully to {commit_content}")
            else:
                bt.logging.error(f"Failed to upload file to GitHub for {commit_content}")
        except Exception as e:
            bt.logging.error(f"Failed to upload file for {commit_content}: {e}")


# ----------------------------------------------------------------------------
# 6. MAIN MINING LOOP
# ----------------------------------------------------------------------------

async def run_miner(config: argparse.Namespace) -> None:
    """
    The main mining loop, orchestrating:
      - Bittensor objects initialization
      - Model initialization
      - Fetching new proteins each epoch
      - Running inference and submissions
      - Periodically syncing metagraph

    Args:
        config (argparse.Namespace): The miner configuration object.
    """

    # 1) Setup wallet, subtensor, metagraph, etc.
    wallet, subtensor, metagraph, miner_uid, epoch_length = await setup_bittensor_objects(config)

    # 2) Prepare shared state
    state: Dict[str, Any] = {
        # environment / config
        'config': config,
        'submission_interval': 1200,

        # GitHub
        'github_path': load_github_path(),

        # Bittensor
        'wallet': wallet,
        'subtensor': subtensor,
        'metagraph': metagraph,
        'miner_uid': miner_uid,
        'epoch_length': epoch_length,

        # Bittensor Drand Timelock objects
        'bdt': QuicknetBittensorDrandTimelock(),

        # Inference state
        'candidate_small_molecules': None,
        'candidate_nanobodies': None,
        'last_submitted_block': 0,
    }

    bt.logging.info("Entering main miner loop...")

    # 3) If we start mid-epoch, obtain most recent proteins from block hash
    current_block = await subtensor.get_current_block()
    last_boundary = (current_block // epoch_length) * epoch_length
    next_boundary = last_boundary + epoch_length

    # If we start too close to epoch end, wait for next epoch
    if next_boundary - current_block < 20:
        bt.logging.info(f"Too close to epoch end, waiting for next epoch to start...")
        block_to_check = next_boundary
        await asyncio.sleep(12*(next_boundary - current_block))
    else:
        block_to_check = last_boundary

    block_hash = await subtensor.determine_block_hash(block_to_check)
    challenge_params = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        small_molecule_target=config.small_molecule_target,
        nanobody_target=config.nanobody_target,
        include_reaction=config.random_valid_reaction
    )
    bt.logging.info(f"Challenge params: {challenge_params}")

    # Update state with your results and other relevant info here
    candidate_small_molecules, candidate_nanobodies = dummy_get_candidate_products()
    state['candidate_small_molecules'] = candidate_small_molecules
    state['candidate_nanobodies'] = candidate_nanobodies
    state['last_submitted_block'] = current_block
    
    # Submit response
    await submit_response(state)
    await asyncio.sleep(1)

    # 5) Main epoch-based loop
    while True:
        try:
            current_block = await subtensor.get_current_block()

            # If we are at an epoch boundary, fetch new proteins
            if current_block % epoch_length == 0:
                bt.logging.info(f"Found epoch boundary at block {current_block}.")
                
                block_hash = await subtensor.determine_block_hash(current_block)
                
                challenge_params = get_challenge_params_from_blockhash(
                    block_hash=block_hash,
                    small_molecule_target=config.small_molecule_target,
                    nanobody_target=config.nanobody_target,
                    include_reaction=config.random_valid_reaction
                )

                # Update state with your results and other relevant info here
                candidate_small_molecules, candidate_nanobodies = dummy_get_candidate_products()
                state['candidate_small_molecules'] = candidate_small_molecules
                state['candidate_nanobodies'] = candidate_nanobodies
                state['last_submitted_block'] = current_block
                
                # Submit response
                await submit_response(state)
                await asyncio.sleep(1)
                
            else:
                # Waiting for epoch
                blocks_remaining = epoch_length - (current_block % epoch_length)
                if (blocks_remaining > 20) and (current_block - state['last_submitted_block'] > 50):
                    bt.logging.info(f"Waiting for epoch to end... {blocks_remaining} blocks remaining.")
                    candidate_small_molecules, candidate_nanobodies = dummy_get_candidate_products()
                    state['candidate_small_molecules'] = candidate_small_molecules
                    state['candidate_nanobodies'] = candidate_nanobodies
                    state['last_submitted_block'] = current_block
                    await submit_response(state)
                    await asyncio.sleep(1)
                await asyncio.sleep(1)

        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting miner.")
            break

        except asyncio.CancelledError:
            bt.logging.info("Resetting subtensor connection.")
            subtensor = bt.async_subtensor(network=config.network)
            await subtensor.initialize()
            await asyncio.sleep(1)
            continue
        except Exception as e:
            bt.logging.error(f"Error in main loop: {e}")
            await asyncio.sleep(3)


# ----------------------------------------------------------------------------
# 7. ENTRY POINT
# ----------------------------------------------------------------------------

async def main() -> None:
    """
    Main entry point for asynchronous execution of the miner logic.
    """
    config = parse_arguments()
    setup_logging(config)
    await run_miner(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
