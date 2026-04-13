import sys
import argparse
import bittensor as bt
import os
from dotenv import load_dotenv
import asyncio

async def set_weights(winner_molecules, winner_nanobodies, config):
    if winner_molecules is not None or winner_nanobodies is not None:
        bt.logging.debug(f"Setting weights for winner molecules: {winner_molecules}, Winner nanobodies: {winner_nanobodies}")
        load_dotenv()
        
        burn_rate = 0.722
        
        wallet_name = config.wallet.name
        wallet_hotkey = config.wallet.hotkey

        NETUID = config.netuid #68
        
        wallet = bt.wallet(
            name=wallet_name,  
            hotkey=wallet_hotkey, 
        )

        # Create Subtensor connection using network from .env
        subtensor_network = os.getenv('SUBTENSOR_NETWORK')
        subtensor = bt.subtensor(network=subtensor_network)

        # Download the metagraph for netuid=68
        metagraph = subtensor.metagraph(NETUID)

        # Check registration
        hotkey_ss58 = wallet.hotkey.ss58_address
        if hotkey_ss58 not in metagraph.hotkeys:
            bt.logging.error(f"Hotkey {hotkey_ss58} is not registered on netuid {NETUID}. Exiting.")
            return

        # 2) Build the weight vector
        n = len(metagraph.uids)
        weights = [0.0] * n

        # Validate the user-provided target UIDs
        for target_uid in [winner_molecules, winner_nanobodies]:
            if target_uid is not None:
                if not (0 <= target_uid < n):
                    bt.logging.error(f"Error: target_uid {target_uid} out of range [0, {n-1}]. Exiting.")
                    return

        # Set weights: burn to UID 0, remainder to winner
        weights[0] = burn_rate
        mol_incentive = (1.0 - burn_rate) * (1 - config.nanobody_weight)
        nanobody_incentive = (1.0 - burn_rate) * config.nanobody_weight
        if winner_molecules and winner_nanobodies:
            if winner_molecules == winner_nanobodies:
                weights[winner_molecules] = 1.0 - burn_rate
            else:
                weights[winner_molecules] = mol_incentive
                weights[winner_nanobodies] = nanobody_incentive

        # if no winner in any one category, send its portion of the incentive to the burn rate
        elif winner_molecules and not winner_nanobodies:
            bt.logging.debug(f"No nanobody winner, sending {nanobody_incentive} to burn rate")
            weights[0] += nanobody_incentive
            weights[winner_molecules] = mol_incentive
        elif winner_nanobodies and not winner_molecules:
            bt.logging.debug(f"No molecule winner, sending {mol_incentive} to burn rate")
            weights[0] += mol_incentive
            weights[winner_nanobodies] = nanobody_incentive

        # 3) Send the weights to the chain with retry logic
        max_retries = 10
        delay_between_retries = 12  # seconds
        for attempt in range(max_retries):
            try:
                bt.logging.info(f"Attempt {attempt + 1} to set weights.")
                result = subtensor.set_weights(
                    netuid=NETUID,
                    wallet=wallet,
                    uids=metagraph.uids,
                    weights=weights,
                    wait_for_inclusion=True
                )
                bt.logging.info(f"Result from set_weights: {result}")

                # Only break if result indicates success (result[0] == True).
                if result[0] is True:
                    bt.logging.info("Weights set successfully. Exiting retry loop.")
                    break
                else:
                    bt.logging.info("set_weights returned a non-success response. Will retry if attempts remain.")
                    if attempt < max_retries - 1:
                        bt.logging.info(f"Retrying in {delay_between_retries} seconds...")
                        await asyncio.sleep(delay_between_retries)

            except Exception as e:
                bt.logging.error(f"Error setting weights: {e}")

                if attempt < max_retries - 1:
                    bt.logging.info(f"Retrying in {delay_between_retries} seconds...")
                    await asyncio.sleep(delay_between_retries)
                else:
                    bt.logging.error("Failed to set weights after multiple attempts. Exiting.")
                    return

        bt.logging.info("Done.")
    else:
        bt.logging.warning("No valid molecule commitment found for current epoch.")
