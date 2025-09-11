#!/usr/bin/env bash
set -Eeuo pipefail

# Install uv:
wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install Rust (cargo) with auto-confirmation:
wget -qO- https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Install system build/env tools (Ubuntu/Debian):
sudo apt update && sudo apt install -y build-essential
sudo apt install python3.12-venv

# Clone timelock at specific commit:
git clone https://github.com/ideal-lab5/timelock.git
cd timelock
git checkout 23fe963f17175e413b7434180d2d0d0776722f1f
cd ..


# Create and activate virtual environment
uv venv && source .venv/bin/activate \
        && uv pip install -r requirements/requirements.txt \
        && uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 \
        && uv pip install torch-geometric==2.6.1 \
        && uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu126.html \
        && uv pip install patchelf \
        && uv pip install maturin==1.8.3 \
        && uv pip install -e boltz

# Build timelock Python bindings (WASM)
export PYO3_CROSS_PYTHON_VERSION="3.12" && cd timelock/wasm && ./wasm_build_py.sh && cd ../..

# Build timelock Python package:
cd timelock/py && uv pip install --upgrade build && python3.12 -m build
uv pip install timelock

echo "Installation complete."
