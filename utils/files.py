import asyncio
import os
import random
import requests
from dotenv import load_dotenv
import bittensor as bt

load_dotenv(override=True)

def upload_file_to_github(filename: str, encoded_content: str):
    # Github configs
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')   # example: nova
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH') # example: main
    github_token = os.environ.get('GITHUB_TOKEN')
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER') # example: metanova-labs
    github_repo_path = os.environ.get('GITHUB_REPO_PATH') # example: /data/results or ""

    if not github_repo_name or not github_repo_branch or not github_token or not github_repo_owner:
        raise ValueError("Github environment variables not set. Please set them in your .env file.")

    target_file_path = os.path.join(github_repo_path, f'{filename}.txt')
    url = f"https://api.github.com/repos/{github_repo_owner}/{github_repo_name}/contents/{target_file_path}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        }

    # Check if the file already exists (need its SHA to update)
    existing_file = requests.get(url, headers=headers, params={"ref": github_repo_branch})
    sha = existing_file.json().get("sha") if existing_file.status_code == 200 else None

    payload = {
        "message": f"Encrypted response for {filename}",
        "content": encoded_content,
        "branch": github_repo_branch,
    }
    if sha:
        payload["sha"] = sha  # updating existing file

    response = requests.put(url, headers=headers, json=payload)
    if response.status_code in [200, 201]:
        return True
    else:
        bt.logging.error(f"Failed to upload file for {filename}: {response.status_code} {response.text}")
        return False


async def read_local_input_file(file_path, config, subtensor):
    """
    Loads input data from a local file for testing purposes.
    Expected format:
    uid|mol_name1,mol_name2,...|protein_sequence1,protein_sequence2,...
    Multiple lines supported. Creates random block numbers for each entry.
    """
    if not os.path.exists(file_path):
        bt.logging.error(f"Local input file not found: {file_path}")
        return None

    current_block = await subtensor.get_current_block()
    uid_to_data = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        uid, mol_names, protein_sequences = line.strip().split('|')
        uid = int(uid)
        mol_names = mol_names.split(',')
        protein_sequences = protein_sequences.split(',')
        uid_to_data[uid] = {
            'molecules': mol_names,
            'sequences': protein_sequences,
            'block_submitted': current_block - random.randint(11, 360),
            'push_time': ""
        }
    return uid_to_data