import asyncio
import math 
import os
import sys
import argparse
from typing import cast
from types import SimpleNamespace
import bittensor as bt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from my_utils import get_smiles, get_random_protein, get_sequence_from_protein_code
from PSICHIC.wrapper import PsichicWrapper
from bittensor.core.chain_data.utils import decode_metadata

psichic = PsichicWrapper()

def get_config():
    """
    Parse command-line arguments to set up the configuration for the wallet
    and subtensor client.
    """
    parser = argparse.ArgumentParser('Nova')
    parser.add_argument("--network", default='finney')
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)

    config = bt.config(parser)
    config.netuid = 68
    config.epoch_length = 360

    return config


def run_model(protein: str, molecule: str) -> float:
    """
    Given a protein sequence (protein) and a molecule identifier (molecule),
    retrieves its SMILES string, then uses the PsichicWrapper to produce
    a predicted binding score. Returns 0.0 if SMILES not found or if
    there's any issue with scoring.
    """
    smiles = get_smiles(molecule)
    if not smiles:
        bt.logging.debug(f"Could not retrieve SMILES for '{molecule}', returning score of 0.0.")
        return 0.0

    results_df = psichic.run_validation([smiles])  # returns a DataFrame
    if results_df.empty:
        bt.logging.warning("Psichic returned an empty DataFrame, returning 0.0.")
        return 0.0
    
    predicted_score = results_df.iloc[0]['predicted_binding_affinity']
    if predicted_score is None:
        bt.logging.warning("No 'predicted_binding_affinity' found, returning 0.0.")
        return 0.0

    return float(predicted_score)


async def get_commitments(subtensor, netuid: int, block: int = None) -> dict:
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
    # Use the provided netuid to fetch the corresponding metagraph.
    metagraph = await subtensor.metagraph(netuid)
    # Determine the block hash if a block is specified.
    block_hash = await subtensor.determine_block_hash(block) if block is not None else None

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
    tolerance = 3 # block tolerance window for validators to commit protein

    while True:
        # Initialize the asynchronous subtensor client.
        subtensor = bt.async_subtensor(network=config.network)
        await subtensor.initialize()
        # Fetch the current metagraph for the given subnet (netuid 68).
        metagraph = await subtensor.metagraph(config.netuid)
        bt.logging.debug(f'Found {metagraph.n} nodes in network')
        current_block = await subtensor.get_current_block()

        # Check if the current block marks the end of an epoch (using a 360-block interval).
        if current_block % config.epoch_length == 0:

            try:
                # Set the next commitment target protein.
                protein = get_random_protein()
                await subtensor.set_commitment(
                    wallet=wallet,
                    netuid=config.netuid,
                    data=protein
                )
                bt.logging.info(f'Commited successfully: {protein}')
            except Exception as e:
                bt.logging.error(f'Error: {e}')

            # Retrieve commitments from the previous epoch.
            prev_epoch = current_block - config.epoch_length
            best_stake = -math.inf
            current_protein = None

            for offset in range(tolerance):
                block_to_check = prev_epoch + offset
                epoch_metagraph = await subtensor.metagraph(config.netuid, block=block_to_check)
                epoch_commitments = await get_commitments(subtensor, netuid=config.netuid, block=block_to_check)
                for index, (hotkey, commit) in enumerate(epoch_commitments.items()):
                    if current_block - commit.block <= (config.epoch_length + tolerance):
                        hotkey_stake = epoch_metagraph.S[index]
                        if hotkey_stake > best_stake and commit.data is not None:
                            best_stake = hotkey_stake
                            current_protein = commit.data

            bt.logging.info(f"Current protein: {current_protein}")

            # Initialize psichic on the current protein
            protein_sequence = get_sequence_from_protein_code(current_protein)
            try:
                psichic.run_challenge_start(protein_sequence)
                bt.logging.info(f"Model initialized successfully for protein {current_protein}.")
            except Exception as e:
                bt.logging.error(f"Error initializing model: {e}")

            # Retrieve the latest commitments (current epoch).
            current_commitments = await get_commitments(subtensor, netuid=config.netuid, block=current_block)

            # Identify the best molecule based on the scoring function.
            best_score = -math.inf
            best_molecule = None
            for hotkey, commit in current_commitments.items():
                if current_block - commit.block <= config.epoch_length:
                    # Assuming that 'commit.data' contains the necessary molecule data; adjust if needed.
                    score = run_model(protein=current_protein, molecule=commit.data)
                    # If the score is higher, or equal but the block is earlier, update the best.
                    if (score > best_score) or (score == best_score and best_molecule is not None and commit.block < best_molecule.block):
                        best_score = score
                        best_molecule = commit

            # Ensure a best molecule was found before setting weights.
            if best_molecule is not None:
                try:
                    # Create weights where the best molecule's UID receives full weight.
                    weights = [0.0 for i in range(metagraph.n)]
                    print(weights)
                    weights[best_molecule.uid] = 1.0
                    print(weights)
                    uids = list(range(metagraph.n))
                    await subtensor.set_weights(
                        wallet=wallet,
                        uids=uids,
                        weights=weights,
                        netuid=config.netuid
                        )
                    bt.logging.info(f"Weights set successfully: {weights}.")
                except Exception as e:
                    bt.logging.error(f"Error setting weights: {e}")
            else:
                bt.logging.info("No valid molecule commitment found for current epoch.")

            # Sleep briefly to prevent busy-waiting (adjust sleep time as needed).
            await asyncio.sleep(1)
        else:
            bt.logging.info(f"Waiting for epoch to end... {config.epoch_length - (current_block % config.epoch_length)} blocks remaining.")
            await asyncio.sleep(3)


if __name__ == "__main__":
    config = get_config()
    asyncio.run(main(config))

