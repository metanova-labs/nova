import argparse
import os
import sys
from dotenv import load_dotenv
import bittensor as bt
from substrateinterface import SubstrateInterface

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

from config.config_loader import load_config

def get_config():
    """
    Parse command-line arguments to set up the configuration for the wallet
    and subtensor client.
    """
    load_dotenv()
    parser = argparse.ArgumentParser('Nova')
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    
    parser.add_argument('--test_mode', action='store_true', 
                       help='Run test validator without setting weights')

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

def setup_github_auth(github_headers):
    """Setup GitHub authentication headers"""
    if os.environ.get('GITHUB_TOKEN'):
        bt.logging.info("GitHub authentication token found")
        github_headers['Authorization'] = f"token {os.environ['GITHUB_TOKEN']}"
    else:
        bt.logging.warning("No GITHUB_TOKEN found. Using unauthenticated requests. For better performance and higher rate limits, add 'GITHUB_TOKEN=your_personal_access_token' to your .env file.")
