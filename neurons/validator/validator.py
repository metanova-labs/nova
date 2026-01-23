import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import asyncio
import sys
import bittensor as bt
import torch
import gc
import traceback
import multiprocessing as mp
import shutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

from auto_updater import AutoUpdater
from config.config_loader import load_config

from utils import get_challenge_params_from_blockhash, inference, QuicknetBittensorDrandTimelock
from neurons.validator.setup import get_config, setup_logging, check_registration, setup_github_auth
from neurons.validator.weights import set_weights
from neurons.validator.commitments import gather_and_decrypt_commitments
from neurons.validator.molecule_validity import validate_molecules_and_calculate_entropy
from neurons.validator.nanobody_validity import validate_nanobodies
from neurons.validator.ranking import calculate_scores_for_type, determine_winner
from neurons.validator.monitoring import monitor_validator
from neurons.validator.save_data import submit_epoch_results
from neurons.validator.score_sharing import apply_external_scores

# Initialize global components (lazy loading for models)
boltz = None
boltzgen = None
btd = QuicknetBittensorDrandTimelock()
GITHUB_HEADERS = {}

async def process_epoch(config, current_block, metagraph, subtensor, wallet):
    """
    Process a single epoch end-to-end.
    """
    global boltz
    test_mode = bool(getattr(config, 'test_mode', False))
    try:
        start_block = current_block - config.epoch_length
        start_block_hash = await subtensor.determine_block_hash(start_block)
        final_block_hash = await subtensor.determine_block_hash(current_block)
        current_epoch = (current_block // config.epoch_length) - 1

        # Get challenge parameters for this epoch
        challenge_params = get_challenge_params_from_blockhash(
            block_hash=start_block_hash,
            small_molecule_target=config.small_molecule_target,
            nanobody_target=config.nanobody_target,
            include_reaction=config.random_valid_reaction,
        )
        if challenge_params:
            small_molecule_target = challenge_params.get("small_molecule_target")
            nanobody_target = challenge_params.get("nanobody_target")
            allowed_reaction = challenge_params.get("allowed_reaction")
        else:
            bt.logging.error("Failed to get challenge parameters from blockhash.")
            return None

        if allowed_reaction:
            bt.logging.info(f"Allowed reaction this epoch: {allowed_reaction}")

        bt.logging.info(f"Scoring using small molecule target: {small_molecule_target} and nanobody target: {nanobody_target}")

        if not config.local_input_file:
            uid_to_data, current_commitments, decrypted_submissions, push_timestamps = await gather_and_decrypt_commitments(
                subtensor, metagraph, config.netuid, start_block, current_block, config, GITHUB_HEADERS, btd
            )
        else:
            from utils import read_local_input_file
            uid_to_data = await read_local_input_file(config.local_input_file, config, subtensor)

        if not uid_to_data:
            bt.logging.info("No valid submissions found this epoch.")
            return None

        # Initialize scoring structure
        score_dict = {
            uid: {
                "molecule_scores": [[] for _ in range(len(small_molecule_target))],
                "nanobody_scores": [[] for _ in range(len(nanobody_target))],
                "entropy": None,
                "block_submitted": None,
                "push_time": uid_to_data[uid].get("push_time", '')
            }
            for uid in uid_to_data
        }

        # Validate submissions
        valid_molecules_by_uid = validate_molecules_and_calculate_entropy(
            uid_to_data=uid_to_data,
            score_dict=score_dict,
            config=config,
            allowed_reaction=allowed_reaction
        )

        valid_nanobodies_by_uid = validate_nanobodies(
            uid_to_data=uid_to_data,
            score_dict=score_dict,
            config=config,
        )

        result = inference.main(valid_molecules_by_uid, valid_nanobodies_by_uid, score_dict, config)
        boltz = result.boltz
        boltzgen = result.boltzgen
        bt.logging.debug(f"score_dict: {score_dict}")
        if boltzgen:
            bt.logging.debug(f"final_boltzgen_scores: {boltzgen.final_boltzgen_scores}")
            bt.logging.debug(f"per_nanobody_components: {boltzgen.per_nanobody_components}")

        # update score_dict for molecules
        score_dict = calculate_scores_for_type(
            score_dict=score_dict,
            valid_items_by_uid=valid_molecules_by_uid,
            item_type="molecule",
            config=config
        )
        
        score_dict = calculate_scores_for_type(
            score_dict=score_dict,
            valid_items_by_uid=valid_nanobodies_by_uid,
            item_type="nanobody",
            config=config
        )

        # TODO: Add external score sharing for nanobodies
        external_api_url = os.environ.get('SCORE_SHARE_API_URL', 'https://vali-score-share-api.metanova-labs.ai')
        if external_api_url and not test_mode:
            external_api_key = os.environ.get('VALIDATOR_API_KEY')
            score_dict = await apply_external_scores(
                score_dict=score_dict,
                valid_molecules_by_uid=valid_molecules_by_uid,
                api_url=external_api_url,
                api_key=external_api_key,
                epoch=current_epoch,
                boltz_per_molecule=getattr(boltz, "per_molecule_metric", None) if boltz is not None else None,
                subtensor=subtensor,
                epoch_end_block=current_block + config.epoch_length,
            )

        # Determine winner for each model
        winner_molecules = determine_winner(score_dict, config=config, item_type="molecule")
        winner_nanobodies = determine_winner(score_dict, config=config, item_type="nanobody")

        # Yield so ws heartbeats can run before the next RPC
        await asyncio.sleep(0)

        # Submit results to dashboard API if configured
        try:
            submit_url = os.environ.get('SUBMIT_RESULTS_URL')
            if submit_url and not test_mode:
                await submit_epoch_results(
                    submit_url=submit_url,
                    config=config,
                    metagraph=metagraph,
                    boltz=boltz,
                    current_block=current_block,
                    start_block=start_block,
                    current_epoch=current_epoch,
                    target_proteins=target_proteins,
                    antitarget_proteins=antitarget_proteins,
                    uid_to_data=uid_to_data,
                    valid_molecules_by_uid=valid_molecules_by_uid,
                    molecule_name_counts=molecule_name_counts,
                    score_dict=score_dict
                )

        except Exception as e:
            bt.logging.error(f"Failed to submit results to dashboard API: {e}")
        # only our validator keeps structure files
        if not submit_url:
            try:
                shutil.rmtree(os.path.join(BASE_DIR, "boltz", "boltz_tmp_files"))
                shutil.rmtree(os.path.join(BASE_DIR, "boltzgen", "boltzgen_tmp_files"))
            except Exception as e:
                bt.logging.warning(f"Error cleaning up temporary files: {e}")
        
        # Monitor validators
        # TODO: adapt monitoring to current structure
        if not test_mode:
            try:
                set_weights_call_block = await subtensor.get_current_block()
            except asyncio.CancelledError:
                bt.logging.info("Resetting subtensor connection.")
                subtensor = bt.async_subtensor(network=config.network)
                await subtensor.initialize()
                await asyncio.sleep(1)
                set_weights_call_block = await subtensor.get_current_block()
            monitor_validator(
                score_dict=score_dict,
                metagraph=metagraph,
                current_epoch=current_epoch,
                current_block=set_weights_call_block,
                validator_hotkey=wallet.hotkey.ss58_address,
                winning_uid=winner_molecules
            )

        return winner_molecules, winner_nanobodies

    except Exception as e:
        bt.logging.error(f"Error processing epoch: {e}")
        bt.logging.error(traceback.format_exc())
        return None

async def main(config):
    """
    Main validator loop
    """
    test_mode = bool(getattr(config, 'test_mode', False))
    local_input = bool(getattr(config, 'local_input_file', None))
    
    # Initialize subtensor client
    subtensor = bt.async_subtensor(network=config.network)
    await subtensor.initialize()
    
    # Wallet + registration check (skipped in test mode)
    wallet = None
    if test_mode:
        bt.logging.info("TEST MODE: running without setting weights")
    else:
        try:
            wallet = bt.wallet(config=config)
            await check_registration(wallet, subtensor, config.netuid)
        except Exception as e:
            bt.logging.error(f"Wallet/registration check failed: {e}")
            sys.exit(1)

    setup_github_auth(GITHUB_HEADERS)

    # Auto-updater setup
    if os.environ.get('AUTO_UPDATE') == '1':
        updater = AutoUpdater(logger=bt.logging)
        asyncio.create_task(updater.start_update_loop())
        bt.logging.info(f"Auto-updater enabled, checking for updates every {updater.UPDATE_INTERVAL} seconds")
    else:
        bt.logging.info("Auto-updater disabled. Set AUTO_UPDATE=1 to enable.")

    # Main validator loop
    last_logged_blocks_remaining = None
    while True:
        try:
            metagraph = await subtensor.metagraph(config.netuid)
            current_block = await subtensor.get_current_block()

            # Only wait for epoch boundary if not reading from local input
            if local_input or current_block % config.epoch_length == 0:
                # Epoch end - process and set weights
                config.update(load_config())
                winner_molecules, winner_nanobodies = await process_epoch(config, current_block, metagraph, subtensor, wallet)
                if not test_mode:
                    await set_weights(winner_molecules, winner_nanobodies, config)
                
                # If using local input, exit after processing
                if local_input:
                    break
                
            else:
                # Waiting for epoch
                blocks_remaining = config.epoch_length - (current_block % config.epoch_length)
                if (blocks_remaining % 5 == 0) and (blocks_remaining != last_logged_blocks_remaining):
                    bt.logging.info(f"Waiting for epoch to end... {blocks_remaining} blocks remaining.")
                    last_logged_blocks_remaining = blocks_remaining
                await asyncio.sleep(1)
                
 
        except asyncio.CancelledError:
            bt.logging.info("Resetting subtensor connection.")
            subtensor = bt.async_subtensor(network=config.network)
            await subtensor.initialize()
            await asyncio.sleep(1)
            continue
        except Exception as e:
            bt.logging.error(f"Error in main loop: {e}")
            await asyncio.sleep(3)

if __name__ == "__main__":
    config = get_config()
    setup_logging(config)
    asyncio.run(main(config))
