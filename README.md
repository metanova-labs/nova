# NOVA - SN68

## High-throughput ML-driven drug screening.

### Accelerating drug discovery, powered by Bittensor.

NOVA harnesses global compute and collective intelligence to navigate huge unexplored chemical spaces, uncovering breakthrough compounds at a fraction of the cost and time.

## System Requirements

- Ubuntu 24.04 LTS (recommended)
- Python 3.12
- CUDA 12.6 (for GPU support)
- Sufficient RAM for ML model operations
- Internet connection for network participation

## Installation and Running

1. Clone the repository:
```bash
git clone <repository-url>
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
```

3. Install dependencies:
   - For CPU:
   ```bash
   ./install_deps_cpu.sh
   ```
   - For CUDA 12.6:
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
- `PSICHIC/runtime_config.py`: runtime configurations for PSICHIC model

### GitHub Authentication

Set up GitHub authentication:
1. Create a [Personal Access Token](https://github.com/settings/personal-access-tokens/new) on GitHub
2. For validators: No specific permissions needed (read-only access)
3. For miners: Grant repository access permissions for your submission repository

## For Validators

DM the NOVA team to obtain an API key.


## Support

For support, please open an issue in the repository or contact the NOVA team.
