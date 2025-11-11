"""
Commitment retrieval and decryption functionality for the validator
"""

import asyncio
import hashlib
import requests
from ast import literal_eval
from types import SimpleNamespace
from typing import cast, Optional

import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata

MAX_RESPONSE_SIZE = 20 * 1024  # 20KB


async def get_commitments(subtensor, metagraph, block_hash: str, netuid: int, min_block: int, max_block: int) -> dict:
    """
    Retrieve commitments for all miners on a given subnet (netuid) at a specific block.

    Args:
        subtensor: The subtensor client object.
        netuid (int): The network ID.
        block (int, optional): The block number to query. Defaults to None.

    Returns:
        dict: A mapping from hotkey to a SimpleNamespace containing uid, hotkey,
              data (commitment), and block.
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


def decrypt_submissions(current_commitments: dict, github_headers: dict, btd, config: dict) -> tuple[dict, dict]:
    """Fetch GitHub submissions and file-specific commit timestamps, then decrypt"""

    file_paths = [commit.data for commit in current_commitments.values() if '/' in commit.data]
    if not file_paths:
        return {}, {}
    
    github_data = {}
    for path in set(file_paths): 
        content_url = f"https://raw.githubusercontent.com/{path}"
        try:
            resp = requests.get(content_url, headers={**github_headers, "Range": f"bytes=0-{MAX_RESPONSE_SIZE}"})
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
                    resp = requests.get(api_url, params={'path': '/'.join(parts[3:]), 'per_page': 1}, headers=github_headers)
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

async def gather_and_decrypt_commitments(subtensor, metagraph, netuid, start_block, current_block, no_submission_blocks, github_headers, btd):
    # Get commitments
    current_block_hash = await subtensor.determine_block_hash(current_block)
    current_commitments = await get_commitments(
        subtensor, 
        metagraph, 
        current_block_hash, 
        netuid=netuid,
        min_block=start_block,
        max_block=current_block - no_submission_blocks
    )
    bt.logging.debug(f"Current epoch commitments: {len(current_commitments)}")

    # Decrypt submissions
    decrypted_submissions, push_timestamps = decrypt_submissions(
        current_commitments, github_headers, btd, {"num_molecules": 1}  # Default config
    )

    # Prepare structured data
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

    return uid_to_data, current_commitments, decrypted_submissions, push_timestamps
