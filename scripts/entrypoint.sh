#!/bin/bash
set -e

# Download model file from huggingface to avoid errors
wget -O PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt

# Build the command-line arguments array
SCRIPT="neurons/validator.py"
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
