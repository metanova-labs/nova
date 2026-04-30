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
from pathlib import Path

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
from neurons.validator.payouts import dispatch_bounty_payouts
from boltzgen.boltzgen_wrapper import BoltzgenWrapper

try:
    from prot_viz_pkg.run import main as run_protein_viz
except Exception as e:
    bt.logging.info(f"protein viz package not found, skipping")

# Initialize global components (lazy loading for models)
boltz = None
boltzgen = None
btd = QuicknetBittensorDrandTimelock()
GITHUB_HEADERS = {}

async def connect_subtensor(network):
    subtensor = bt.async_subtensor(network=network)
    await subtensor.initialize()
    return subtensor

async def reconnect_subtensor(subtensor, network):
    if subtensor is not None:
        await subtensor.close()
    return await connect_subtensor(network)

async def call_subtensor(subtensor, network, rpc_fn, timeout_s=10):
    try:
        result = await asyncio.wait_for(rpc_fn(subtensor), timeout=timeout_s)
        return result, subtensor
    except Exception as e:
        bt.logging.warning(f"Subtensor RPC reconnect triggered due to {type(e).__name__}: {e}")
        subtensor = await reconnect_subtensor(subtensor, network)
        result = await asyncio.wait_for(rpc_fn(subtensor), timeout=timeout_s)
        return result, subtensor

async def process_epoch(config, current_block, metagraph, subtensor, wallet):
    """
    Process a single epoch end-to-end.
    """
    global boltz
    test_mode = bool(getattr(config, 'test_mode', False))
    try:
        # clean up temp files if there's any left from previous epoch
        for model in ["boltz", "boltzgen"]:
            tmp_files_dir = os.path.join(BASE_DIR, "external_tools", model, f"{model}_tmp_files")
            if os.path.exists(tmp_files_dir):
                shutil.rmtree(tmp_files_dir)

        start_block = current_block - config.epoch_length
        start_block_hash = await subtensor.determine_block_hash(start_block)
        final_block_hash = await subtensor.determine_block_hash(current_block)
        current_epoch = (current_block // config.epoch_length) - 1

        bt.logging.info(f"Epoch {current_epoch} scoring started.")

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

        valid_nanobodies_by_uid = await validate_nanobodies(
            uid_to_data=uid_to_data,
            score_dict=score_dict,
            config=config,
        )

        result = inference.main(valid_molecules_by_uid, valid_nanobodies_by_uid, score_dict, config)
        boltz = result.boltz
        boltzgen = result.boltzgen

        # update score_dict for molecules 
        # (before external score sharing because averaging final scores and components is equivalent)
        # nanobody final scores are calculated after score sharing
        score_dict = calculate_scores_for_type(
            score_dict=score_dict,
            valid_items_by_uid=valid_molecules_by_uid,
            item_type="molecule",
            config=config
        )

        test_mode = bool(getattr(config, 'test_mode', False))
        external_api_url = os.environ.get('SCORE_SHARE_API_URL', 'https://vali-score-share-api.metanova-labs.ai')
        external_api_key = os.environ.get('VALIDATOR_API_KEY')
        if external_api_url and not test_mode:
            score_dict = await apply_external_scores(
                score_dict=score_dict,
                valid_molecules_by_uid=valid_molecules_by_uid,
                valid_nanobodies_by_uid=valid_nanobodies_by_uid,
                api_url=external_api_url,
                api_key=external_api_key,
                epoch=current_epoch,
                boltz=boltz,
                boltzgen=boltzgen,
                subtensor=subtensor,
                epoch_end_block=current_block + config.epoch_length,
                test_mode=test_mode,
                target_proteins=small_molecule_target,
            )
            
        # update scores for nanobodies
        if valid_nanobodies_by_uid and boltzgen and boltzgen.per_nanobody_components:
            rank_mode = getattr(config, "boltzgen_rank_mode", None) or getattr(config, "rank_mode", "min")
            final_boltzgen_scores, per_nanobody_components = BoltzgenWrapper.finalize_from_shared_components(
                boltzgen.per_nanobody_components,
                valid_nanobodies_by_uid,
                config,
            )
            bt.logging.debug(f"Final boltzgen scores: {per_nanobody_components}")
            inference._merge_boltzgen_into_score_dict(
                score_dict,
                final_boltzgen_scores,
                valid_nanobodies_by_uid,
                config,
                rank_mode=rank_mode,
            )
        
        score_dict = calculate_scores_for_type(
            score_dict=score_dict,
            valid_items_by_uid=valid_nanobodies_by_uid,
            item_type="nanobody",
            config=config
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
                    boltzgen=boltzgen,
                    current_block=current_block,
                    start_block=start_block,
                    current_epoch=current_epoch,
                    uid_to_data=uid_to_data,
                    valid_molecules_by_uid=valid_molecules_by_uid,
                    valid_nanobodies_by_uid=valid_nanobodies_by_uid,
                    score_dict=score_dict,
                )
                # only our validator creates protein viz files
                try:
                    mmcif_dir = os.path.join(BASE_DIR, "external_tools", "boltzgen", "boltzgen_tmp_files", "outputs", "intermediate_designs", "refold_cif")
                    for mmcif_file in os.listdir(mmcif_dir):
                        if mmcif_file.endswith(".cif"):
                            mmcif_path = Path(os.path.join(mmcif_dir, mmcif_file))
                            run_protein_viz(mmcif_path=mmcif_path, 
                            chain_a="A", chain_b="B", output_path=Path(os.path.join(mmcif_dir, f"{mmcif_file.replace('.cif', '.html')}")), 
                            upload=True, epoch=current_epoch)
                except Exception as e:
                    bt.logging.error(f"Error running protein viz: {e}")

        except Exception as e:
            bt.logging.error(f"Failed to submit results to dashboard API: {e}")
            bt.logging.error(traceback.format_exc())            

        for model in ["boltz", "boltzgen"]:
            tmp_files_dir = os.path.join(BASE_DIR, "external_tools", model, f"{model}_tmp_files")
            if os.path.exists(tmp_files_dir):
                try:
                    shutil.rmtree(tmp_files_dir)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    bt.logging.error(f"Error cleaning up temporary files for {model}: {e}")

        return winner_molecules, winner_nanobodies, uid_to_data

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
    subtensor = await connect_subtensor(config.network)
    
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
            metagraph, subtensor = await call_subtensor(
                subtensor,
                config.network,
                lambda st: st.metagraph(config.netuid),
            )
            current_block, subtensor = await call_subtensor(
                subtensor,
                config.network,
                lambda st: st.get_current_block(),
            )

            # Only wait for epoch boundary if not reading from local input
            if local_input or current_block % config.epoch_length == 0:
                # Epoch end - process and set weights
                config.update(load_config())
                epoch_result = await process_epoch(config, current_block, metagraph, subtensor, wallet)
                winner_molecules = None
                winner_nanobodies = None
                if epoch_result is None:
                    pass
                else:
                    winner_molecules, winner_nanobodies, uid_to_data = epoch_result

                current_epoch = (current_block // config.epoch_length) - 1
                if not test_mode:
                    payouts = await set_weights(winner_molecules, winner_nanobodies, config)
                    if payouts:
                        try:
                            hotkey_payouts = []
                            for component, uid, proportion in payouts:
                                hotkey = uid_to_data.get(uid, {}).get("hotkey")
                                if not hotkey:
                                    bt.logging.error(
                                        f"Missing hotkey for payout component={component} uid={uid}; skipping."
                                    )
                                    continue
                                hotkey_payouts.append((component, hotkey, proportion))

                            await dispatch_bounty_payouts(
                                payouts=hotkey_payouts,
                                subtensor=subtensor,
                                config=config,
                                epoch=current_epoch,
                            )
                        except Exception as e:
                            bt.logging.error(f"Error dispatching bounty payouts: {e}")
                            bt.logging.error(traceback.format_exc())
                
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
            subtensor = await reconnect_subtensor(subtensor, config.network)
            await asyncio.sleep(1)
            continue
        except Exception as e:
            bt.logging.error(f"Error in main loop: {e}")
            subtensor = await reconnect_subtensor(subtensor, config.network)
            await asyncio.sleep(3)

if __name__ == "__main__":
    config = get_config()
    setup_logging(config)
    asyncio.run(main(config))
