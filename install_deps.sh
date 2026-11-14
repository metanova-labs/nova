#!/usr/bin/env bash
# NOVA bootstrap for Ubuntu/Debian x86_64.
#
# Usage: ./install_deps.sh [--cuda cu126|cu128]
set -Eeuo pipefail
shopt -s nullglob
trap 'rc=$?; echo "[ERROR] Failed at line $LINENO (exit $rc)" >&2' ERR

NOVA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$NOVA_ROOT"

log()  { printf '\n[install] %s\n' "$*"; }
warn() { printf '[WARNING] %s\n' "$*" >&2; }
die()  { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

CUDA_VERSION="cu126"
while (( $# > 0 )); do
  case "$1" in
    --cuda)
      (( $# >= 2 )) || die '--cuda requires a value'
      CUDA_VERSION="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--cuda cu126|cu128]"
      exit 0
      ;;
    *) die "Unknown argument: $1" ;;
  esac
done
case "$CUDA_VERSION" in cu126|cu128) ;; *) die "Unsupported CUDA version: $CUDA_VERSION" ;; esac

[[ "$(uname -s):$(uname -m)" == "Linux:x86_64" ]] || \
  die 'This script currently supports Linux x86_64 only (MMseqs2 archive requires AVX2).'
command -v apt-get >/dev/null || die 'This script requires Ubuntu/Debian (apt-get).'
for tool in curl sha256sum tar git; do
  command -v "$tool" >/dev/null || die "Missing bootstrap tool: $tool"
done

PYTHON_BIN="$(command -v python3 || true)"
[[ -n "$PYTHON_BIN" ]] || die 'python3 not found'
PYTHON_BIN="$(readlink -f "$PYTHON_BIN")"
PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$PYTHON_VERSION" =~ ^3\.(10|11|12)$ ]] || \
  die "Python $PYTHON_VERSION is unsupported (expected 3.10, 3.11, or 3.12)"

# Prerequisite: remediate and verify the host OS trust store BEFORE running.
# curl will validate HTTPS against the host OS trust configuration.
fetch() {
  local url="$1" destination="$2"
  curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
    --retry 3 --retry-delay 2 --output "$destination" "$url"
}
verify_sha256() {
  local path="$1" expected="$2"
  printf '%s  %s\n' "$expected" "$path" | sha256sum --check --status || \
    die "SHA-256 mismatch for $path"
}

# ---------------------------------------------------------------------------
# 1. Install independently verified, pinned uv.
# ---------------------------------------------------------------------------
UV_VERSION="0.13.8"
UV_TARGET="uv-x86_64-unknown-linux-gnu"
UV_SHA256="5acfd9177cfecf0bf87578f059728215ccc884388988f55f7a65041b57387d91"
UV_INSTALL_DIR="$NOVA_ROOT/.tools/bootstrap/bin"
UV_BIN="$UV_INSTALL_DIR/uv"

install_verified_uv() {
  local tempdir archive
  tempdir="$(mktemp -d)"
  archive="$tempdir/$UV_TARGET.tar.gz"
  log "Downloading pinned uv $UV_VERSION"
  fetch "https://releases.astral.sh/github/uv/releases/download/$UV_VERSION/$UV_TARGET.tar.gz" "$archive"
  verify_sha256 "$archive" "$UV_SHA256"
  tar -xzf "$archive" -C "$tempdir" "$UV_TARGET/uv"
  mkdir -p "$UV_INSTALL_DIR"
  install -m 0755 "$tempdir/$UV_TARGET/uv" "$UV_BIN"
  rm -rf -- "$tempdir"
  # uv may print build/target metadata, e.g. "uv 0.13.4 (x86_64-unknown-linux-gnu)".
  local uv_output actual_uv_version
  uv_output="$("$UV_BIN" --version)" || die 'Could not run installed uv'
  [[ "$uv_output" =~ ^uv[[:space:]]+([0-9]+\.[0-9]+\.[0-9]+)([[:space:]]|$) ]] || \
    die "Could not parse uv version: $uv_output"
  actual_uv_version="${BASH_REMATCH[1]}"
  [[ "$actual_uv_version" == "$UV_VERSION" ]] || \
    die "Unexpected uv version: $uv_output (expected $UV_VERSION)"
}
install_verified_uv
export PATH="$UV_INSTALL_DIR:$PATH"
export UV_SYSTEM_CERTS=true
export UV_PYTHON_DOWNLOADS=never
log "Using $($UV_BIN --version)"

# Expose the verified uv binary in the user's local bin directory.
mkdir -p "$HOME/.local/bin"
ln -sfn "$UV_BIN" "$HOME/.local/bin/uv"

# ---------------------------------------------------------------------------
# 2. Rust: verified rustup bootstrap and pinned compilation toolchain.
# ---------------------------------------------------------------------------
RUSTUP_VERSION="1.29.0"
RUSTUP_SHA256="4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10"
RUST_HOST="x86_64-unknown-linux-gnu"
RUST_TOOLCHAIN="1.90.0"
if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
if ! command -v rustup >/dev/null; then
  log "Bootstrapping pinned rustup $RUSTUP_VERSION with verified SHA-256"
  RUSTUP_TEMP="$(mktemp -d)"
  fetch "https://static.rust-lang.org/rustup/archive/$RUSTUP_VERSION/$RUST_HOST/rustup-init" \
    "$RUSTUP_TEMP/rustup-init"
  verify_sha256 "$RUSTUP_TEMP/rustup-init" "$RUSTUP_SHA256"
  chmod 0755 "$RUSTUP_TEMP/rustup-init"
  "$RUSTUP_TEMP/rustup-init" -y --no-modify-path --profile minimal \
    --default-host "$RUST_HOST" --default-toolchain none
  rm -rf -- "$RUSTUP_TEMP"
  export PATH="$HOME/.cargo/bin:$PATH"
fi
command -v rustup >/dev/null || die 'Verified rustup bootstrap failed'
log "Installing/using pinned Rust toolchain $RUST_TOOLCHAIN"
rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal
export RUSTUP_TOOLCHAIN="$RUST_TOOLCHAIN"
command -v cargo >/dev/null && command -v rustc >/dev/null || die 'Rust toolchain unavailable'
log "Rust compiler: $(rustc --version)"
export CARGO_NET_GIT_FETCH_WITH_CLI=true

# ---------------------------------------------------------------------------
# 3. OS build tools.
# ---------------------------------------------------------------------------
log 'Installing system build tools'
sudo apt-get update
sudo apt-get install -y build-essential git hmmer "python${PYTHON_VERSION}-venv"

# ---------------------------------------------------------------------------
# 4. Aggrescan3D and FreeSASA.
# ---------------------------------------------------------------------------
log 'Installing Aggrescan3D and FreeSASA'
sh "$NOVA_ROOT/NOVA-nanobody-filter/install/install-aggrescan3d.sh"
test -x "$NOVA_ROOT/NOVA-nanobody-filter/.tools/bin/aggrescan3d-rs" || \
  die 'Aggrescan3D executable was not installed'
test -x "$NOVA_ROOT/NOVA-nanobody-filter/.tools/bin/freesasa" || \
  die 'FreeSASA executable was not installed'

# ---------------------------------------------------------------------------
# 5. Auxiliary tools and combinatorial database.
# The upstream URLs here remain mutable; see security notes after the script.
# ---------------------------------------------------------------------------
mkdir -p external_tools combinatorial_db
if [[ ! -d external_tools/mmseqs ]]; then
  log 'Installing MMseqs2 (mutable latest archive; not hash-pinned)'
  fetch 'https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz' \
    external_tools/mmseqs.tar.gz
  if [[ -n "${MMSEQS_SHA256:-}" ]]; then
    verify_sha256 external_tools/mmseqs.tar.gz "$MMSEQS_SHA256"
  else
    warn 'MMSEQS_SHA256 not set; MMseqs2 archive is not independently hash-verified.'
  fi
  tar -xzf external_tools/mmseqs.tar.gz -C external_tools/
  rm -f external_tools/mmseqs.tar.gz
fi
[[ -d external_tools/mmseqs/bin ]] || die 'MMseqs2 bin directory missing'
export PATH="$NOVA_ROOT/external_tools/mmseqs/bin:$PATH"

if [[ ! -d external_tools/igblast ]]; then
  log 'Installing IgBLAST (LATEST URL; not hash-pinned)'
  fetch 'https://ftp.ncbi.nih.gov/blast/executables/igblast/release/LATEST/ncbi-igblast-1.22.0-x64-linux.tar.gz' \
    external_tools/igblast.tar.gz
  if [[ -n "${IGBLAST_SHA256:-}" ]]; then
    verify_sha256 external_tools/igblast.tar.gz "$IGBLAST_SHA256"
  else
    warn 'IGBLAST_SHA256 not set; IgBLAST archive is not independently hash-verified.'
  fi
  tar -xzf external_tools/igblast.tar.gz -C external_tools/
  test -d external_tools/ncbi-igblast-1.22.0 || die 'Unexpected IgBLAST archive structure'
  mv external_tools/ncbi-igblast-1.22.0 external_tools/igblast
  rm -f external_tools/igblast.tar.gz
fi
mkdir -p external_tools/igblast/internal_data
cp -a data/igblast_internal_data/camelid external_tools/igblast/internal_data/
cp -a data/igblast_database/database external_tools/igblast/

log 'Downloading molecule database (mutable main branch)'
fetch 'https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/main/molecules.sqlite?download=true' \
  combinatorial_db/molecules.sqlite
if [[ -n "${MOLECULES_DB_SHA256:-}" ]]; then
  verify_sha256 combinatorial_db/molecules.sqlite "$MOLECULES_DB_SHA256"
else
  warn 'MOLECULES_DB_SHA256 not set; molecules.sqlite is not authenticated by an independent hash.'
fi

# ---------------------------------------------------------------------------
# 6. Fetch exact timelock source; validate commit before replacing old checkout.
# ---------------------------------------------------------------------------
TIMELOCK_COMMIT="23fe963f17175e413b7434180d2d0d0776722f1f"
TIMELOCK_DIR="$NOVA_ROOT/external_tools/timelock"
TIMELOCK_TEMP="$NOVA_ROOT/external_tools/timelock.clone.$$"
log 'Cloning pinned timelock source'
git clone --quiet https://github.com/ideal-lab5/timelock.git "$TIMELOCK_TEMP"
git -C "$TIMELOCK_TEMP" checkout --quiet --detach "$TIMELOCK_COMMIT"
[[ "$(git -C "$TIMELOCK_TEMP" rev-parse HEAD)" == "$TIMELOCK_COMMIT" ]] || \
  die 'Timelock Git commit mismatch'
rm -rf -- "$TIMELOCK_DIR"
mv -- "$TIMELOCK_TEMP" "$TIMELOCK_DIR"

# ---------------------------------------------------------------------------
# 7. Create virtual environment, then install Python dependencies.
# WARNING: like your previous script, this replaces the existing .venv.
# ---------------------------------------------------------------------------
log 'Creating a fresh NOVA virtual environment'
rm -rf -- "$NOVA_ROOT/.venv"
"$UV_BIN" venv --python "$PYTHON_BIN" "$NOVA_ROOT/.venv"
VENV_PY="$NOVA_ROOT/.venv/bin/python"
VENV_MATURIN="$NOVA_ROOT/.venv/bin/maturin"
export VIRTUAL_ENV="$NOVA_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

log 'Installing main Python requirements'
"$UV_BIN" pip install --python "$VENV_PY" -r requirements/requirements.txt
log "Installing PyTorch CUDA $CUDA_VERSION packages"
"$UV_BIN" pip install --python "$VENV_PY" \
  'torch==2.7.1' 'torchvision==0.22.1' 'torchaudio==2.7.1' \
  --index-url "https://download.pytorch.org/whl/$CUDA_VERSION"

"$UV_BIN" pip install --python "$VENV_PY" 'patchelf' 'maturin==1.8.3'
"$UV_BIN" pip install --python "$VENV_PY" -e "$NOVA_ROOT/external_tools/boltz"
"$UV_BIN" pip install --python "$VENV_PY" -e "$NOVA_ROOT/external_tools/boltzgen"

# Prevent the main requirements / editable installs from restoring an older
# certifi bundle or a Requests version outside Bittensor 10.4.1's constraints.
"$UV_BIN" pip install --python "$VENV_PY" \
  'certifi==2026.10.19' 'requests>=2.33.0,<3.0'

# ---------------------------------------------------------------------------
# 8. Build pinned timelock's native Python wrapper, then install local Python wheel.
# ---------------------------------------------------------------------------
log 'Building timelock wasm Python wrapper'
export PYO3_CROSS_PYTHON_VERSION="$PYTHON_VERSION"
(
  cd "$TIMELOCK_DIR/wasm"
  # Upstream script invokes `cargo build ...` then `maturin develop ...`.
  bash ./wasm_build_py.sh
)
"$VENV_PY" -c 'import timelock_wasm_wrapper; print("timelock wrapper import OK")'

log 'Building and installing local timelock Python wheel'
"$UV_BIN" pip install --python "$VENV_PY" \
  'build==1.3.0' 'setuptools==80.9.0' 'wheel==0.45.1'
(
  cd "$TIMELOCK_DIR/py"
  rm -rf -- dist
  "$VENV_PY" -m build --wheel --no-isolation
  wheels=(dist/*.whl)
  [[ ${#wheels[@]} -eq 1 ]] || die 'Expected exactly one locally built timelock wheel'
  # The native wrapper was already installed by `maturin develop`.
  "$UV_BIN" pip install --python "$VENV_PY" --no-deps "${wheels[0]}"
)
"$VENV_PY" -c 'import timelock; print("Local timelock import OK")'

# ---------------------------------------------------------------------------
# 9. Preserve the deliberate async-substrate-interface cyscale override.
# NOTE: Bittensor 10.4.1 declares cyscale==0.4.0; this is a deliberate conflict.
# ---------------------------------------------------------------------------
log 'Applying intentional cyscale override'
"$UV_BIN" pip uninstall --python "$VENV_PY" scalecodec cyscale
"$UV_BIN" pip install --python "$VENV_PY" --reinstall 'cyscale==0.5.0'

# ---------------------------------------------------------------------------
# 10. Build patched bittensor-drand and replace the regular wheel LAST.
# ---------------------------------------------------------------------------
build_patched_bittensor_drand() (
  set -Eeuo pipefail
  shopt -s nullglob

  local DRAND_VERSION="1.3.0"
  local DRAND_COMMIT="33f74100add8d99e8acb364c35f9ec148021f8de"
  local WEBPKI_VERSION="0.103.16"
  local WHEEL_DIR="$NOVA_ROOT/.tools/wheels/bittensor-drand"
  local REVIEWED_LOCK="$NOVA_ROOT/requirements/bittensor-drand-1.3.0.Cargo.lock"
  local WORK_DIR src actual_commit wheel

  mkdir -p "$WHEEL_DIR"
  WORK_DIR="$(mktemp -d)"
  trap 'rm -rf -- "$WORK_DIR"' EXIT
  src="$WORK_DIR/source"

  log "Building bittensor-drand $DRAND_VERSION with patched rustls-webpki $WEBPKI_VERSION"
  git clone --quiet --depth 1 --branch "v$DRAND_VERSION" \
    https://github.com/RaoFoundation/bittensor-drand.git "$src"
  actual_commit="$(git -C "$src" rev-parse HEAD)"
  [[ "$actual_commit" == "$DRAND_COMMIT" ]] || die 'Unexpected bittensor-drand source commit'
  cd "$src"

  if [[ -f "$REVIEWED_LOCK" ]]; then
    log 'Using reviewed patched Cargo.lock from requirements/'
    cp -- "$REVIEWED_LOCK" Cargo.lock
  else
    warn 'No reviewed drand Cargo.lock checked in; generating a patched lockfile for review.'
    cargo update -p rustls-webpki --precise "$WEBPKI_VERSION"
  fi

  # cargo metadata with --locked proves the exact resolved version; reject duplicates.
  "$VENV_PY" -I - <<'PY'
import json
import subprocess
metadata = json.loads(subprocess.check_output(
    ["cargo", "metadata", "--locked", "--format-version", "1"]
))
versions = sorted({p["version"] for p in metadata["packages"]
                   if p["name"] == "rustls-webpki"})
assert versions == ["0.103.16"], f"Unexpected rustls-webpki versions: {versions}"
print("Verified resolved rustls-webpki:", versions)
PY
  cargo tree --locked -i rustls-webpki

  # Avoid stale wheels from a previous run.
  rm -f -- "$WHEEL_DIR"/bittensor_drand-1.3.0-*.whl
  "$VENV_MATURIN" build --release --locked --interpreter "$VENV_PY" --out "$WHEEL_DIR"
  cp -- Cargo.lock "$WHEEL_DIR/Cargo.lock"

  local wheels=("$WHEEL_DIR"/bittensor_drand-1.3.0-*.whl)
  [[ ${#wheels[@]} -eq 1 ]] || die 'Expected exactly one patched bittensor-drand wheel'
  wheel="${wheels[0]}"
  (cd "$WHEEL_DIR" && sha256sum "$(basename "$wheel")") > "$WHEEL_DIR/SHA256SUMS"
  {
    echo "source=https://github.com/RaoFoundation/bittensor-drand"
    echo "source_commit=$actual_commit"
    echo "rustls_webpki=$WEBPKI_VERSION"
    echo "rust_toolchain=$(rustc --version)"
    echo "maturin=$($VENV_MATURIN --version)"
  } > "$WHEEL_DIR/provenance.txt"

  "$UV_BIN" pip install --python "$VENV_PY" --no-deps --reinstall --no-cache "$wheel"
  "$VENV_PY" -I - <<'PY'
from importlib.metadata import version
import bittensor_drand
assert version("bittensor-drand") == "1.3.0", version("bittensor-drand")
assert callable(bittensor_drand.get_encrypted_commit)
print("Patched bittensor-drand package loaded:", bittensor_drand.__file__)
PY
  log "Patched wheel: $wheel"
)

build_patched_bittensor_drand

# ---------------------------------------------------------------------------
# 11. Final validation and model weights.
# ---------------------------------------------------------------------------
log 'Validating core package versions'
"$VENV_PY" - <<'PY'
from importlib.metadata import version
assert version('bittensor') == '10.4.1', 'Expected bittensor 10.4.1'
assert version('bittensor-drand') == '1.3.0'
print('bittensor:', version('bittensor'))
print('bittensor-drand:', version('bittensor-drand'))
PY

# A strict dependency checker may complain about your intentional cyscale 0.5.0
# override of bittensor's cyscale 0.4.0 requirement; inspect its output.
if ! "$UV_BIN" pip check --python "$VENV_PY"; then
  warn 'uv pip check found dependency conflicts (cyscale override is expected; review others).'
fi

log 'Downloading model weights'
"$VENV_PY" "$NOVA_ROOT/tools/download_model_weights.py"

log 'Installation complete.'
log "uv executable: $UV_BIN"
log "Patched drand wheel and Cargo.lock: $NOVA_ROOT/.tools/wheels/bittensor-drand"
