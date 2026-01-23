"""
Commitment retrieval and decryption functionality for the validator
"""

import asyncio
import hashlib
import requests
from ast import literal_eval
from types import SimpleNamespace
from typing import cast, Optional, Tuple, List

import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata

MAX_RESPONSE_SIZE = 20 * 1024  # 20KB
DELIMITER = "|" # separates between molecules and sequences
SEPARATOR = "," # separates molecules or sequences among themselves
NULL_TOKEN = "~" # represents an empty list

def normalize_list(part: str) -> list[str]:
    if part == NULL_TOKEN:
        return []
    return [x for x in part.split(SEPARATOR) if x]

def parse_decrypted_submission(
    uid: int,
    decrypted: str,
    num_molecules: int,
    num_sequences: int,
) -> Optional[Tuple[List[str], List[str]]]:
    """
    Expected format:
      mol1,mol2,...|seq1,seq2,...
    For num_molecules and num_sequences both =1, simplest is:
      mol|seq
    """
    if decrypted is None:
        return None

    decrypted = decrypted.strip()
    # hard reject whitespace
    if any(ch.isspace() for ch in decrypted):
        bt.logging.warning(f"UID {uid}: Decrypted submission contains whitespace (not allowed)")
        return None

    # hard reject non-ascii characters
    if not all(ch.isascii() for ch in decrypted):
        bt.logging.warning(f"UID {uid}: Decrypted submission contains non-ascii characters (not allowed)")
        return None

    # must contain exactly one delimiter (|)
    if decrypted.count("|") != 1:
        bt.logging.warning(f"UID {uid}: Expected exactly one '|' delimiter between molecules and sequences")
        return None

    # must contain both parts
    mol_part, seq_part = decrypted.split("|", 1)
    if not mol_part or not seq_part:
        bt.logging.warning(f"UID {uid}: Missing molecules or sequences section")
        return None

    mols = normalize_list(mol_part)
    seqs = normalize_list(seq_part)

    return mols, seqs

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

    # Gather commitment queries for all hotkeys concurrently.
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


def tuple_safe_eval(uid: int, input_str: str) -> tuple:
    # Limit input size to prevent overly large inputs.
    if len(input_str) > MAX_RESPONSE_SIZE:
        bt.logging.warning(f"UID {uid}: Input exceeds allowed size")
        return None
    
    try:
        # Safely evaluate the input string as a Python literal.
        result = literal_eval(input_str)
    except (SyntaxError, ValueError, MemoryError, RecursionError, TypeError) as e:
        bt.logging.warning(f"UID {uid}: Input is not a valid literal: {e}")
        return None

    # Check that the result is a tuple with exactly two elements.
    if not (isinstance(result, tuple) and len(result) == 2):
        bt.logging.warning(f"UID {uid}: Expected a tuple with exactly two elements")
        return None

    # Verify that the first element is an int.
    if not isinstance(result[0], int):
        bt.logging.warning(f"UID {uid}: First element must be an int")
        return None
    
    # Verify that the second element is a bytes object.
    if not isinstance(result[1], bytes):
        bt.logging.warning(f"UID {uid}: Second element must be a bytes object")
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
                        bt.logging.debug(f"No commit history found for https://github.com/{parts[0]}/{parts[1]}/blob/{parts[2]}/{'/'.join(parts[3:])}")
                except Exception as e:
                    bt.logging.warning(f"Error fetching timestamp for https://github.com/{parts[0]}/{parts[1]}: {e}")
        
        github_data[path] = {'content': content, 'timestamp': timestamp}
    
    encrypted_submissions = {}
    push_timestamps = {}
    
    for commit in current_commitments.values():
        uid = commit.uid        
        data = github_data.get(commit.data)
        if not data:
            continue
            
        content = data.get('content')
        push_timestamps[commit.uid] = data.get('timestamp', '')
        
        if not content:
            continue            
        try:
            content_str = content.decode('utf-8', errors='replace')
            content_hash = hashlib.sha256(content_str.encode('utf-8')).hexdigest()[:20]
            expected_suffix = f'/{content_hash}.txt'
            if commit.data.endswith(expected_suffix):
                encrypted_content = tuple_safe_eval(uid, content_str)
                if encrypted_content:
                    encrypted_submissions[commit.uid] = encrypted_content
        except Exception as e:
            bt.logging.warning(f"UID {uid}: Exception during content processing: {e}", exc_info=True)

    # Decrypt all submissions
    try:
        decrypted_raw = btd.decrypt_dict(encrypted_submissions)

        # parse into (molecules, sequences)
        parsed = {}
        for uid, payload in decrypted_raw.items():
            if payload is None:
                bt.logging.warning(f"UID {uid}: Decryption returned None payload")
                continue
            
            parsed_pair = parse_decrypted_submission(
                uid,
                payload,
                num_molecules=config["num_molecules"],
                num_sequences=config["num_sequences"],
            )
            if parsed_pair is None:
                continue
            mols, seqs = parsed_pair
            parsed[uid] = {"molecules": mols, "sequences": seqs}

        decrypted_submissions = parsed

    except Exception as e:
        bt.logging.error(f"Failed to decrypt submissions: {e}", exc_info=True)
        decrypted_submissions = {}

    bt.logging.info(f"GitHub: {len(file_paths)} paths → {len(decrypted_submissions)} decrypted")
    return decrypted_submissions, push_timestamps

async def gather_and_decrypt_commitments(subtensor, metagraph, netuid, start_block, current_block, config, github_headers, btd):
    # Get commitments
    current_block_hash = await subtensor.determine_block_hash(current_block)
    current_commitments = await get_commitments(
        subtensor, 
        metagraph, 
        current_block_hash, 
        netuid=netuid,
        min_block=start_block,
        max_block=current_block - config.no_submission_blocks
    )
    bt.logging.debug(f"Current epoch commitments: {len(current_commitments)}")

    # Decrypt submissions
    decrypted_submissions, push_timestamps = decrypt_submissions(
        current_commitments, github_headers, btd, config
    )
    bt.logging.debug(f"Decrypted submissions: {decrypted_submissions}")

    # Prepare structured data
    uid_to_data = {}
    for hotkey, commit in current_commitments.items():
        uid = commit.uid
        submission = decrypted_submissions.get(uid)

        if submission is not None:
            uid_to_data[uid] = {
                "molecules": submission["molecules"],
                "sequences": submission["sequences"],
                "block_submitted": commit.block,
                "push_time": push_timestamps.get(uid, '')
            }

    return uid_to_data, current_commitments, decrypted_submissions, push_timestamps
