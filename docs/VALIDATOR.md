# Validator Documentation

## Overview

Validators are a core component of the NOVA subnet that evaluate miner submissions, score them using machine learning models, and set weights on the Bittensor network. Validators process molecular (SMILES) and nanobody (protein sequence) submissions from miners each epoch, validate them, score them against target proteins, and determine winners.

## Architecture

The validator operates in an epoch-based cycle:

1. **Commitment Gathering**: Retrieves encrypted commitments from miners on-chain
2. **Decryption**: Decrypts submissions using timelock encryption
3. **Validation**: Validates molecules and nanobodies according to chemical/biological rules
4. **Scoring**: Uses ML models (Boltz and Boltzgen) to score submissions against target proteins
5. **Ranking**: Calculates final scores and determines winners
6. **Weight Setting**: Sets weights on the Bittensor network to reward winners

## Key Components

### Core Modules

#### `validator.py`
Main entry point and orchestration logic. Contains:
- `main()`: Main validator loop that waits for epoch boundaries
- `process_epoch()`: Processes a complete epoch end-to-end
- Handles epoch timing, metagraph updates, and weight setting

#### `setup.py`
Configuration and initialization:
- `get_config()`: Parses command-line arguments and loads configuration
- `setup_logging()`: Configures Bittensor logging to file
- `check_registration()`: Verifies validator is registered on subnet
- `setup_github_auth()`: Sets up GitHub API authentication

#### `commitments.py`
Handles retrieval and decryption of miner submissions:
- `get_commitments()`: Retrieves commitments from blockchain
- `decrypt_submissions()`: Fetches encrypted data from GitHub and decrypts
- `gather_and_decrypt_commitments()`: Main function orchestrating the process
- `parse_decrypted_submission()`: Parses decrypted submission format

#### `molecule_validity.py`
Validates molecular submissions:
- `validate_molecules_and_calculate_entropy()`: Validates SMILES strings
- Checks for:
  - Valid SMILES format
  - Duplicate molecules
  - Reaction compatibility (if filtering enabled)
  - Chemical validity (RDKit parsing)
  - MACCS entropy calculation

#### `nanobody_validity.py`
Validates nanobody (protein sequence) submissions:
- `validate_nanobodies()`: Validates protein sequences
- Checks for:
  - Valid amino acid composition
  - Sequence length constraints
  - Homopolymer runs
  - Di-repeat pairs
  - Cysteine pairing
  - Signal peptide patterns
  - Duplicate sequences

#### `ranking.py`
Score calculation and winner determination:
- `calculate_scores_for_type()`: Aggregates scores for molecules or nanobodies
- `determine_winner()`: Determines winning UID based on final scores
- Handles tie-breaking using block submission time and push timestamps

#### `weights.py`
Weight setting on Bittensor network:
- `set_weights()`: Sets weights on-chain with retry logic
- Implements burn rate (73.7% to UID 0)
- Distributes remaining weights to winners

#### `monitoring.py`
Validator monitoring and reporting:
- `monitor_validator()`: Sends monitoring data to external service
- Tracks validator performance, scores, and machine info

#### `score_sharing.py`
External score sharing functionality:
- `apply_external_scores()`: Applies scores from external API
- Enables cross-validator score validation

#### `save_data.py`
Data submission to dashboard:
- `submit_epoch_results()`: Submits epoch results to dashboard API
- Includes scores, submissions, and metadata

## Configuration

### Environment Variables

Required in `.env` file:

```bash
# Network configuration
SUBTENSOR_NETWORK="wss://entrypoint-finney.opentensor.ai:443"  # or your local node

# GitHub authentication (optional but recommended)
GITHUB_TOKEN="your_personal_access_token"

# Validator API key (for monitoring and score sharing, not needed in test_mode)
VALIDATOR_API_KEY="your_api_key"

# Score sharing API (averages out results of non-deterministic models)
SCORE_SHARE_API_URL="https://vali-score-share-api.metanova-labs.ai"

# Auto-updater (optional)
AUTO_UPDATE=1  # Enable automatic updates
```

### Command-Line Arguments

```bash
python3 neurons/validator/validator.py \
  --wallet.name <wallet_name> \
  --wallet.hotkey <hotkey_name> \
  --test_mode \                    # Optional: run without setting weights
  --local_input_file <path> \      # Optional: use local file instead of chain - for testing purposes
  --logging.debug                  # Logging level (debug recommended)
```

### Configuration Options

The validator loads configuration from `config/config_loader.py` which includes:
- `num_molecules`: Number of molecules per submission
- `num_sequences`: Number of nanobody sequences per submission
- `boltz_mode`: Ranking mode for molecules ('max' or 'min')
- `boltzgen_rank_mode`: Ranking mode for nanobodies ('max' or 'min')
- `epoch_length`: Blocks per epoch (auto-detected from chain)
- `min_sequence_length`: Minimum nanobody sequence length
- `max_sequence_length`: Maximum nanobody sequence length
- `max_homopolymer_run`: Maximum consecutive identical amino acids
- And more (see nova/config/config.yaml for all configurable values)

For each model, there is also a specific config file that stores configurable values that are only relevant for that model. 

## Running a Validator

### Prerequisites (production)

1. **Registration**: Validator must be registered on subnet 68
2. **Stake**: Minimum 1000 NOVA stake required
3. **Hardware**: Sufficient GPU and RAM for ML model operations. In production, 2 GPUs are required to process all submissions within an epoch (molecules and nanobodies are evaluated in parallel). If a single GPU is provided, molecules and nanobodies will be evaluated sequentially.
4. **Network**: Stable connection to Bittensor network
5. **System**: 
- Ubuntu 24.04 LTS (recommended)
- Python 3.12
- CUDA 12.6 (recommended)

For test_mode, registration and stake checks are skipped.

### Basic Usage

Installation:
```bash
./install_deps_cu126.sh
```

Running:
```bash
# Activate virtual environment
source .venv/bin/activate

# Run validator
python3 neurons/validator/validator.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_hotkey \
  --logging.info
```

### Test Mode

Run validator without setting weights (useful for testing):

```bash
python3 neurons/validator/validator.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_hotkey \
  --test_mode
```

### Local Input Mode

Test with local input file instead of fetching from chain:

```bash
python3 neurons/validator/validator.py \
  --wallet.name my_wallet \
  --wallet.hotkey my_hotkey \
  --local_input_file /path/to/input \
  --test_mode
```

Local input format:
```
  uid1|mol1,mol2...|seq1,seq2...
  uid2|mol3,mol4...|seq3,seq4...
  ...
```

### Accepted input formats:

The input string submitted should follow the format `mol|seq`, using the | character as a separator. If multiple molecules or sequences are required, they should be comma separated.

Note: the character ~ is a null placeholder. It is accepted in miner submissions if they do not wish to participate in either molecules or nanobodies competition.

Examples:
```
  mol|seq   ✅
  ~|seq     ✅
  mol|~     ✅
  ~|~       Not sure why you would do that, but technically ✅

  mol       ❌
  seq       ❌
  mol|      ❌
  |seq      ❌
  seq|mol   ❌  # order must be mol|seq
  mol,seq   ❌  # separator must be `|`
```

## Epoch Processing Flow

### 1. Epoch Detection
- Validator waits for epoch boundary (block % epoch_length == 0)
- Calculates start and end blocks for current epoch

### 2. Challenge Parameters
- Derives the random challenge parameters from block hash:
  - Allowed reaction (if filtering enabled)

### 3. Commitment Gathering
- Retrieves commitments from blockchain for all miners
- Filters by submission block (must be within epoch window)
- Excludes submissions too close to epoch end (< 10 blocks to epoch end)

### 4. Decryption
- Fetches encrypted submissions from GitHub
- Decrypts using timelock encryption (used to prevent miners from copying each other's submissions before epoch ends)
- Parses submission format: `molecule1,molecule2|sequence1,sequence2`
- Character ~ burns submission for either molecules or sequences

### 5. Validation
- **Molecules**: Validates SMILES, checks duplicates, reaction compatibility, minimum required entropy if num_molecules > 1.
- **Nanobodies**: Validates sequences, checks length, composition, patterns

### 6. Scoring
- Runs ML models (Boltz for molecules, Boltzgen for nanobodies)
- Scores each valid submission against the target proteins
- Aggregates scores across multiple targets, if applicable.

### 7. Ranking
- Calculates final scores for each UID
- Determines winners based on scoring mode (max/min)
- Tie-breaking: earliest block submission, then push timestamp

### 8. Weight Setting
- Sets weights on-chain:
  - 73.7% to UID 0 (burn)
  - Remaining to winner(s)
- Retries up to 10 times with 12-second delays

### 9. Monitoring
- Sends monitoring data to external service
- Tracks validator performance and scores

## Scoring System

### Molecule Scoring (Boltz)
- Scores molecules against small molecule target proteins
- Supports multiple targets per epoch
- Final score: `config['boltz_metric']` items combined with `config['combination_strategy']`
- Mode: `max` (higher is better) or `min` (lower is better)

### Nanobody Scoring (Boltzgen)
- Also supports multiple targets per epoch
- Scores nanobody sequences against target proteins
- Ranks each miner's submission on each one of the selected metrics
- Final score: sum of each miner's rank across all metrics
- Mode: typically `min` (ranking higher is better)

### Score Aggregation
1. For each target, calculate average score across all valid items, if applicable
2. Sum averages across all targets to get final score
3. Invalid/missing items get `-inf` (max mode) or `inf` (min mode)

### Score sharing
All validators running share the scores they obtained via a dedicated API. This is done to increase robustness of scoring with nondeterinistic models. When all validators have finished the scoring loop, they retrieve scores of the remaining validators and set weights to the best average score accross all runs.

## Validation Rules

### Molecules
- Must be valid SMILES strings (parsable by RDKit)
- No duplicates within submission
- Must pass reaction filter (if enabled)
- Must have valid chemical structure

### Nanobodies
- Must contain only valid amino acids (ACDEFGHIKLMNPQRSTVWY)
- Length between `min_sequence_length` and `max_sequence_length`
- No homopolymer runs exceeding `max_homopolymer_run`
- Valid cysteine pairing
- No signal peptide patterns
- No duplicate sequences

## Logging

Logs are written to `bittensor.log` in the validator directory. Log levels:
- `--logging.debug`: Detailed debugging information (recommended)
- `--logging.info`: Standard operational information
- `--logging.warning`: Warnings only
- `--logging.error`: Errors only

## Monitoring

Validators can send monitoring data to external services:
- Machine information (GPU, CUDA version, etc.)
- Score distributions
- Winning UIDs and scores
- Epoch statistics

Set `VALIDATOR_API_KEY` environment variable to enable.

## Auto-Updater

Validators can automatically update from GitHub:
- Set `AUTO_UPDATE=1` in environment
- Checks for updates every configured interval
- Automatically pulls and restarts on updates


## Contributing

When modifying validator code:
1. Test changes in test mode first
2. Ensure backward compatibility with existing submissions
3. Update documentation for new features
4. Follow existing code style and patterns
5. Add appropriate error handling

## Support

For issues or questions:
- Open an issue in the repository
- Contact the NOVA team
- Check logs in `bittensor.log` for detailed error information
