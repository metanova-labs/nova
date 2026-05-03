# syntax=docker/dockerfile:1
# Minimal NOVA validator image (CUDA 12.6, linux/amd64). For local Apple Silicon users, pass --platform linux/amd64:
#   docker build --platform linux/amd64 -t nova-validator:local .

FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

LABEL org.opencontainers.image.description="NOVA SN68 validator (CUDA 12.6)"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHON_VERSION=3.12 \
    PYO3_CROSS_PYTHON_VERSION=3.12 \
    CUDA_VERSION=cu126

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      pkg-config \
      cmake \
      ca-certificates \
      curl \
      git \
      wget \
      hmmer \
      zlib1g-dev \
      libssl-dev \
      python3.12 \
      python3.12-dev \
      python3.12-venv \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && git config --global url."https://github.com/".insteadOf "http://github.com/"
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV NOVA_HOME=/app/nova
WORKDIR ${NOVA_HOME}
RUN mkdir -p external_tools combinatorial_db

WORKDIR ${NOVA_HOME}/external_tools
RUN wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O mmseqs.tar.gz \
    && tar xzf mmseqs.tar.gz \
    && rm mmseqs.tar.gz \
    && if [ ! -d mmseqs ]; then top=$(find . -maxdepth 1 -mindepth 1 -type d ! -name igblast | head -1); mv "$top" mmseqs; fi

RUN wget -q https://ftp.ncbi.nih.gov/blast/executables/igblast/release/LATEST/ncbi-igblast-1.22.0-x64-linux.tar.gz \
      -O igblast.tar.gz \
    && tar -xzf igblast.tar.gz \
    && rm igblast.tar.gz \
    && mv ncbi-igblast-1.22.0 igblast

WORKDIR ${NOVA_HOME}
COPY data/igblast_internal_data/camelid ${NOVA_HOME}/external_tools/igblast/internal_data/camelid
COPY data/igblast_database/database ${NOVA_HOME}/external_tools/igblast/database

RUN wget -q "https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/main/molecules.sqlite?download=true" \
    -O combinatorial_db/molecules.sqlite

WORKDIR ${NOVA_HOME}/external_tools
RUN git init timelock \
    && cd timelock \
    && git remote add origin https://github.com/ideal-lab5/timelock.git \
    && git fetch --depth 1 origin 23fe963f17175e413b7434180d2d0d0776722f1f \
    && git checkout FETCH_HEAD

WORKDIR ${NOVA_HOME}
COPY requirements/requirements.txt requirements/requirements.txt

RUN uv venv --python python3.12 ${NOVA_HOME}/.venv \
    && . ${NOVA_HOME}/.venv/bin/activate \
    && uv pip install -r requirements/requirements.txt \
    && uv pip install \
         torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
         --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}" \
    && uv pip install patchelf maturin==1.8.3

COPY external_tools/boltz external_tools/boltz/
RUN . ${NOVA_HOME}/.venv/bin/activate \
    && cd external_tools/boltz \
    && uv pip install -e .

COPY external_tools/boltzgen external_tools/boltzgen/
RUN . ${NOVA_HOME}/.venv/bin/activate \
    && cd external_tools/boltzgen \
    && uv pip install -e .

WORKDIR ${NOVA_HOME}/external_tools/timelock/wasm
RUN chmod +x wasm_build_py.sh \
    && export PATH="${NOVA_HOME}/.venv/bin:${PATH}" \
    && bash -eo pipefail ./wasm_build_py.sh

WORKDIR ${NOVA_HOME}/external_tools/timelock/py
RUN . ${NOVA_HOME}/.venv/bin/activate \
    && uv pip install --upgrade build \
    && python3 -m build \
    && bash -eo pipefail -c 'uv pip install dist/*.whl'

WORKDIR ${NOVA_HOME}
COPY . .

ENV PATH="${NOVA_HOME}/.venv/bin:${NOVA_HOME}/external_tools/mmseqs/bin:${NOVA_HOME}/external_tools/igblast/bin:${PATH}"
ENV PYTHONPATH=${NOVA_HOME}

WORKDIR ${NOVA_HOME}

ENTRYPOINT ["/app/nova/.venv/bin/python3", "neurons/validator/validator.py"]
CMD []
