# NOVA - SN68

## High-throughput ML-driven drug screening.

### Accelerating drug discovery, powered by Bittensor.

NOVA harnesses global compute and collective intelligence to navigate huge unexplored chemical spaces, uncovering breakthrough compounds at a fraction of the cost and time.

## System Requirements for validators

- Ubuntu 24.04 LTS (recommended)
- Python 3.10 - 3.12
- CUDA 12.6 (for GPU support)
- Sufficient RAM for ML model operations
- 2 GPU devices for parallel inference. If only one is available, inference will run sequentially which may result in delayed/missing scoring rounds.
- Internet connection for network participation

## Installation and Running

1. Clone the repository:
```bash
git clone --recurse-submodules https://github.com/metanova-labs/nova.git
cd nova
```

2. Prepare your .env file as in example.env:
```
# General configs
SUBTENSOR_NETWORK="ws://localhost:9944" # or your chosen node
DEVICE_OVERRIDE="cpu" # None to run on GPU

# GitHub authentication
GITHUB_TOKEN="your_personal_access_token"

# GitHub configs - FOR MINERS
GITHUB_REPO_NAME="repo-name"
GITHUB_REPO_BRANCH="repo-branch"
GITHUB_REPO_OWNER="repo-owner"
GITHUB_REPO_PATH="" # path within repo or ""

# For validators
VALIDATOR_API_KEY="your_api_key"
AUTO_UPDATE="1" # Set to "0" to disable auto-updates (not recommended)
```

3. Install dependencies:
   ```bash
   ./install_deps_cu126.sh
   ```

4. Run:
```bash
# Activate your virtual environment:
source .venv/bin/activate

# Run your script:
# miner:
python3 neurons/miner.py --wallet.name <your_wallet> --wallet.hotkey <your_hotkey> --logging.info

# validator:
python3 neurons/validator/validator.py --wallet.name <your_wallet> --wallet.hotkey <your_hotkey> --logging.debug
```

## Configuration

The project uses several configuration files:
- `.env`: Environment variables and API keys
- `requirements/`: Dependency specifications for different environments
- Command-line arguments for runtime configuration

### GitHub Authentication

Set up GitHub authentication:
1. Create a [Personal Access Token](https://github.com/settings/personal-access-tokens/new) on GitHub
2. For validators: No specific permissions needed (read-only access)
3. For miners: Grant repository access permissions for your submission repository

## For Validators

DM the NOVA team to obtain an API key.


## Support

For support, please open an issue in the repository or contact the NOVA team.
