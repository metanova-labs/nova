import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import asyncio
import sys
import bittensor as bt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

from auto_updater import AutoUpdater
from btdr import QuicknetBittensorDrandTimelock
from config.config_loader import load_config
from PSICHIC.wrapper import PsichicWrapper

from utils import get_challenge_params_from_blockhash
from .setup import get_config, setup_logging, check_registration, setup_github_auth
from .weights import set_weights
from .commitments import gather_and_decrypt_commitments
from .validity import validate_molecules_and_calculate_entropy, count_molecule_names
from .scoring import score_all_proteins_batched
from .ranking import calculate_final_scores, determine_winner
from .monitoring import monitor_validator

# Initialize global components
psichic = PsichicWrapper()
btd = QuicknetBittensorDrandTimelock()
GITHUB_HEADERS = {}

# Set global variables for scoring module
import neurons.validator.scoring as scoring_module
scoring_module.psichic = psichic
scoring_module.BASE_DIR = BASE_DIR

async def process_epoch(config, current_block, metagraph, subtensor, wallet):
    """
    Process a single epoch end-to-end.
    """
    try:
        start_block = current_block - config.epoch_length
        start_block_hash = await subtensor.determine_block_hash(start_block)
        current_epoch = (current_block // config.epoch_length) - 1

        # Get challenge parameters for this epoch
        challenge_params = get_challenge_params_from_blockhash(
            block_hash=start_block_hash,
            weekly_target=config.weekly_target,
            num_antitargets=config.num_antitargets,
            include_reaction=config.random_valid_reaction
        )
        target_proteins = challenge_params["targets"]
        antitarget_proteins = challenge_params["antitargets"]
        allowed_reaction = challenge_params.get("allowed_reaction")

        if allowed_reaction:
            bt.logging.info(f"Allowed reaction this epoch: {allowed_reaction}")

        bt.logging.info(f"Scoring using target proteins: {target_proteins}, antitarget proteins: {antitarget_proteins}")

        # Gather and decrypt commitments
        uid_to_data, current_commitments, decrypted_submissions, push_timestamps = await gather_and_decrypt_commitments(
            subtensor, metagraph, config.netuid, start_block, current_block, config.no_submission_blocks, GITHUB_HEADERS, btd
        )

        if not uid_to_data:
            bt.logging.info("No valid submissions found this epoch.")
            return None

        # Initialize scoring structure
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

        # Validate molecules and calculate entropy
        valid_molecules_by_uid = validate_molecules_and_calculate_entropy(
            uid_to_data=uid_to_data,
            score_dict=score_dict,
            config=config,
            allowed_reaction=allowed_reaction
        )

        # Count molecule name occurrences
        molecule_name_counts = count_molecule_names(valid_molecules_by_uid)

        # Score all target proteins then all antitarget proteins one protein at a time
        score_all_proteins_batched(
            target_proteins=target_proteins,
            antitarget_proteins=antitarget_proteins,
            score_dict=score_dict,
            valid_molecules_by_uid=valid_molecules_by_uid,
            uid_to_data=uid_to_data,
            batch_size=32
        )

        # Calculate final scores
        score_dict = calculate_final_scores(
            score_dict, valid_molecules_by_uid, molecule_name_counts, config, current_epoch
        )

        # Determine winner
        winning_uid = determine_winner(score_dict)

        # Monitor validator performance
        set_weights_call_block = await subtensor.get_current_block()
        monitor_validator(
            score_dict=score_dict,
            metagraph=metagraph,
            current_epoch=current_epoch,
            current_block=set_weights_call_block,
            validator_hotkey=wallet.hotkey.ss58_address,
            winning_uid=winning_uid
        )

        return winning_uid

    except Exception as e:
        bt.logging.error(f"Error processing epoch: {e}")
        return None

async def main(config):
    """
    Main validator loop
    """
    wallet = bt.wallet(config=config)
    
    # Initialize subtensor client
    subtensor = bt.async_subtensor(network=config.network)
    await subtensor.initialize()

    # Setup and validation
    await check_registration(wallet, subtensor, config.netuid)
    setup_github_auth(GITHUB_HEADERS)

    # Auto-updater setup
    if os.environ.get('AUTO_UPDATE') == '1':
        updater = AutoUpdater(logger=bt.logging)
        asyncio.create_task(updater.start_update_loop())
        bt.logging.info(f"Auto-updater enabled, checking for updates every {updater.UPDATE_INTERVAL} seconds")
    else:
        bt.logging.info("Auto-updater disabled. Set AUTO_UPDATE=1 to enable.")

    # Main validator loop
    while True:
        try:
            metagraph = await subtensor.metagraph(config.netuid)
            bt.logging.debug(f'Found {metagraph.n} nodes in network')
            current_block = await subtensor.get_current_block()

            if current_block % config.epoch_length == 0:
                # Epoch end - process and set weights
                config.update(load_config())
                winning_uid = await process_epoch(config, current_block, metagraph, subtensor, wallet)
                await set_weights(winning_uid, config)
                
            elif current_block % (config.epoch_length/2) == 0:
                # Mid-epoch - refresh connection
                subtensor = bt.async_subtensor(network=config.network)
                await subtensor.initialize()
                bt.logging.info("Validator reset subtensor connection.")
                await asyncio.sleep(12)
            
            else:
                # Waiting for epoch
                blocks_remaining = config.epoch_length - (current_block % config.epoch_length)
                bt.logging.info(f"Waiting for epoch to end... {blocks_remaining} blocks remaining.")
                await asyncio.sleep(1)
                
        except Exception as e:
            bt.logging.error(f"Error in main loop: {e}")
            await asyncio.sleep(3)

if __name__ == "__main__":
    config = get_config()
    setup_logging(config)
    asyncio.run(main(config))