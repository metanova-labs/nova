#!/usr/bin/env bash
set -Eeuo pipefail

# get python version and check if supported
PYTHON_BIN=$(command -v python3)
PYTHON_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

if [[ ! "$PYTHON_VERSION" =~ ^3\.(10|11|12)$ ]]; then
  echo "Error: Python $PYTHON_VERSION is not supported. Requires 3.10, 3.11, or 3.12." >&2
  exit 1
fi


# Install uv:
wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install Rust (cargo) with auto-confirmation:
wget -qO- https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Install system build/env tools (Ubuntu/Debian):
sudo apt update && sudo apt install -y build-essential
sudo apt install -y python"$PYTHON_VERSION"-venv

# install hmmer and mmseqs2
sudo apt install -y hmmer
if [ ! -d external_tools/mmseqs ]; then
  wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O external_tools/mmseqs.tar.gz && tar xzf external_tools/mmseqs.tar.gz -C external_tools/ && rm external_tools/mmseqs.tar.gz && export PATH=$(pwd)/external_tools/mmseqs/bin/:$PATH
fi

# install igblast
if [ ! -d external_tools/igblast ]; then
  wget -q https://ftp.ncbi.nih.gov/blast/executables/igblast/release/LATEST/ncbi-igblast-1.22.0-x64-linux.tar.gz -O external_tools/igblast.tar.gz
  tar -xzf external_tools/igblast.tar.gz -C external_tools/
  mv external_tools/ncbi-igblast-1.22.0 external_tools/igblast
  rm external_tools/igblast.tar.gz
fi

# copy custom data to external_tools
cp -r data/igblast_internal_data/camelid external_tools/igblast/internal_data/
cp -r data/igblast_database/database external_tools/igblast

# download combinatorial db
[ -d combinatorial_db/molecules.sqlite ] && wget -q https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/main/molecules.sqlite?download=true -O combinatorial_db/molecules.seqlite

# Check if .venv and timelock exist and delete them if they do (for reinstalling)
[ -d .venv ] && rm -rf .venv
[ -d external_tools/timelock ] && rm -rf external_tools/timelock

# Clone timelock at specific commit:
cd external_tools
git clone https://github.com/ideal-lab5/timelock.git
cd timelock
git checkout 23fe963f17175e413b7434180d2d0d0776722f1f
cd ../..

# Create and activate virtual environment (for main process)
uv venv --python "$PYTHON_BIN" && source .venv/bin/activate \
        && uv pip install -r requirements/requirements.txt \
        && uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 \
        && uv pip install patchelf \
        && uv pip install maturin==1.8.3 \
        && cd external_tools/boltz && uv pip install -e . \
        && cd ../.. && cd external_tools/boltzgen && uv pip install -e . \
        && cd ../..

# Build timelock Python bindings (WASM)
export PYO3_CROSS_PYTHON_VERSION="$PYTHON_VERSION" && cd external_tools/timelock/wasm && ./wasm_build_py.sh && cd ../../..

# Build timelock Python package:
cd external_tools/timelock/py && uv pip install --upgrade build && python3 -m build
uv pip install timelock

echo "Installation complete."
