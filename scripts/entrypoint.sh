#!/bin/bash
set -e

# Determine which script to run based on RUN_MODE (default to miner if not set)
if [ -z "$RUN_MODE" ]; then
    echo "RUN_MODE not set, defaulting to miner."
    SCRIPT="neurons/miner.py"
elif [ "$RUN_MODE" = "miner" ]; then
    SCRIPT="neurons/miner.py"
elif [ "$RUN_MODE" = "validator" ]; then
    SCRIPT="neurons/validator.py"
else
    echo "Unknown RUN_MODE: ${RUN_MODE}. Should be 'miner' or 'validator'."
    exit 1
fi

# Build the command-line arguments array
ARGS=("$@")

# Optional flags if variables are defined in the environment
if [ -n "$WALLET_NAME" ]; then
    ARGS+=( "--wallet.name" "$WALLET_NAME" )
fi
if [ -n "$WALLET_HOTKEY" ]; then
    ARGS+=( "--wallet.hotkey" "$WALLET_HOTKEY" )
fi
if [ -n "$SUBTENSOR_NETWORK" ]; then
    ARGS+=( "--network" "$SUBTENSOR_NETWORK" )
fi

echo "Running command: python ${SCRIPT} ${ARGS[@]}"
exec python "${SCRIPT}" "${ARGS[@]}"