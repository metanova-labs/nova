import os
import requests
import bittensor as bt


def upload_file_to_github(filename: str, encoded_content: str):
    """Upload a base64-encoded file to a GitHub repository using environment-configured credentials."""
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH')
    github_token = os.environ.get('GITHUB_TOKEN')
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER')
    github_repo_path = os.environ.get('GITHUB_REPO_PATH') or ""

    if not github_repo_name or not github_repo_branch or not github_token or not github_repo_owner:
        raise ValueError('Github environment variables not set. Please set them in your .env file.')

    target_file_path = os.path.join(github_repo_path, f'{filename}.txt')
    url = f"https://api.github.com/repos/{github_repo_owner}/{github_repo_name}/contents/{target_file_path}"
    headers = {
        'Authorization': f'Bearer {github_token}',
        'Accept': 'application/vnd.github+json',
    }

    existing_file = requests.get(url, headers=headers, params={'ref': github_repo_branch})
    sha = existing_file.json().get('sha') if existing_file.status_code == 200 else None

    payload = {
        'message': f'Encrypted response for {filename}',
        'content': encoded_content,
        'branch': github_repo_branch,
    }
    if sha:
        payload['sha'] = sha

    response = requests.put(url, headers=headers, json=payload)
    if response.status_code in [200, 201]:
        return True
    bt.logging.error(f"Failed to upload file for {filename}: {response.status_code} {response.text}")
    return False
