# NOVA validator — local Docker workflows (Dockerfile expects linux/amd64).
#
# Prerequisites: Docker Desktop with buildx (Apple Silicon: expect slow first build).
# Optional: ~/.bittensor/wallets populated; copy example.env → .env
#
# Examples:
#   make build
#   make shell
#   WALLET_NAME=cold WALLET_HOTKEY=hot make run
#   make lint          # phase-1 ruff scope (matches CI; requires: pip install ruff==0.7.4)
#   make lint-fix      # apply safe auto-fixes, then commit
#   make test          # pytest (requires: pip install pytest pyyaml)

# Pinned to match .github/workflows/release.yml (lint job).
RUFF_VERSION     ?= 0.7.4
# Phase 1: same paths as CI `lint` job (not whole repo — see docs/CICD.md).
RUFF_TARGETS     ?= tests/ utils/fasta.py utils/local_input.py utils/files.py neurons/validator/lifecycle.py

# Directory containing this Makefile (so `lint` / `test` work even if invoked
# as `make -f /path/to/nova/Makefile` from another cwd).
THIS_DIR         := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

IMAGE            ?= nova-validator:local
PLATFORM         ?= linux/amd64
CONTAINER_NAME   ?= nova-validator-dev
ENV_FILE         ?= $(CURDIR)/.env

.PHONY: default help build shell run run-test run-local run-bg logs stop rm clean inspect \
	lint lint-fix lint-fix-unsafe test _need-ruff _need-pytest \
	compose-config compose-up compose-down compose-logs compose-ready

COMPOSE_FILE     := $(THIS_DIR)/docker-compose.yml

ENV_FLAGS        := $(shell test -f "$(ENV_FILE)" && echo --env-file "$(ENV_FILE)" || true)

HELP_WALLET_REQ   = WALLET_NAME and WALLET_HOTKEY must be non-empty.

WALLET_NAME      ?=
WALLET_HOTKEY    ?=
EXTRA_ARGS       ?= --logging.debug
LOCAL_INPUT      ?=

WALLET_BIND      ?= $(HOME)/.bittensor/wallets
WALLET_VOL       ?= -v $(WALLET_BIND):/root/.bittensor/wallets:ro

default: help

help:
	@echo "Targets:"
	@echo "  make build           Build $(IMAGE) for $(PLATFORM)"
	@echo "  make shell           Interactive bash (/app)"
	@echo "  make run             Foreground validator (needs $(HELP_WALLET_REQ))"
	@echo "  make run-test        Foreground --test_mode (no wallet, no set_weights)"
	@echo "  make run-local       run-test + --local_input_file (LOCAL_INPUT=path; one epoch then exit)"
	@echo "  make run-bg          Daemonize as $(CONTAINER_NAME)"
	@echo "  make logs            Follow daemon logs"
	@echo "  make stop            docker stop $(CONTAINER_NAME)"
	@echo "  make rm              docker rm -f $(CONTAINER_NAME)"
	@echo "  make clean           docker rmi $(IMAGE)"
	@echo "  make inspect         Smoke: import neuron module in ephemeral container"
	@echo "  make lint            Run ruff on CI scope (phase 1; requires: pip install ruff==$(RUFF_VERSION))"
	@echo "  make lint-fix        ruff --fix on same scope"
	@echo "  make lint-fix-unsafe Run ruff --fix --unsafe-fixes on same scope"
	@echo "  make test            Run pytest tests/ (same suite as CI tests job)"
	@echo "  make compose-config  docker compose config -q"
	@echo "  make compose-up      docker compose up -d --build (see README-DEPLOY.md)"
	@echo "  make compose-down    docker compose down"
	@echo "  make compose-logs    follow validator + watchtower logs"
	@echo "  make compose-ready   curl /ready_to_update inside running validator svc"

_need-ruff:
	@python3 -c "import ruff" 2>/dev/null || \
		(echo "error: pip install ruff==$(RUFF_VERSION) (must match .github/workflows/release.yml)" >&2 && exit 1)

_need-pytest:
	@python3 -c "import pytest" 2>/dev/null || \
		(echo "error: pip install pytest pyyaml" >&2 && exit 1)

lint: _need-ruff
	cd "$(THIS_DIR)" && python3 -m ruff check $(RUFF_TARGETS)

lint-fix: _need-ruff
	cd "$(THIS_DIR)" && python3 -m ruff check $(RUFF_TARGETS) --fix

lint-fix-unsafe: _need-ruff
	cd "$(THIS_DIR)" && python3 -m ruff check $(RUFF_TARGETS) --fix --unsafe-fixes

test: _need-pytest
	python3 -m pytest "$(THIS_DIR)tests/"

compose-config:
	docker compose -f "$(COMPOSE_FILE)" config -q

compose-up:
	docker compose -f "$(COMPOSE_FILE)" up -d --build

compose-down:
	docker compose -f "$(COMPOSE_FILE)" down

compose-logs:
	docker compose -f "$(COMPOSE_FILE)" logs -f validator watchtower

compose-ready:
	docker compose -f "$(COMPOSE_FILE)" exec -T validator \
		sh -lc 'curl -fsS "http://127.0.0.1:$${READINESS_PORT:-8080}/ready_to_update"'

build:
	docker build --platform $(PLATFORM) -t $(IMAGE) "$(CURDIR)"

shell:
	docker run --rm -it $(ENV_FLAGS) --platform $(PLATFORM) $(WALLET_VOL) \
		-w /app/nova $(IMAGE) /bin/bash

run:
	@test -n "$(WALLET_NAME)" && test -n "$(WALLET_HOTKEY)" || \
		(echo "error: $(HELP_WALLET_REQ)" >&2 && exit 1)
	@test -f "$(ENV_FILE)" || (echo "error: missing $(ENV_FILE); copy example.env" >&2 && exit 1)
	docker run --rm -it $(ENV_FLAGS) --platform $(PLATFORM) $(WALLET_VOL) \
		-w /app/nova $(IMAGE) \
		--wallet.name "$(WALLET_NAME)" --wallet.hotkey "$(WALLET_HOTKEY)" $(EXTRA_ARGS)

# Test mode: no wallet/registration, no set_weights. Still hits SUBTENSOR_NETWORK
# (e.g. wss://entrypoint-finney.opentensor.ai:443) to read epoch_length / metagraph.
run-test:
	@test -f "$(ENV_FILE)" || (echo "error: missing $(ENV_FILE); copy example.env" >&2 && exit 1)
	docker run --rm -it $(ENV_FLAGS) --platform $(PLATFORM) \
		-w /app/nova $(IMAGE) \
		--test_mode $(EXTRA_ARGS)

# run-test + --local_input_file mounted read-only at /app/nova/local_input.txt.
# Validator processes one epoch from the file and exits.
run-local:
	@test -n "$(LOCAL_INPUT)" || (echo "error: set LOCAL_INPUT=/abs/path/to/file" >&2 && exit 1)
	@test -f "$(LOCAL_INPUT)" || (echo "error: $(LOCAL_INPUT) not found" >&2 && exit 1)
	@test -f "$(ENV_FILE)" || (echo "error: missing $(ENV_FILE); copy example.env" >&2 && exit 1)
	docker run --rm -it $(ENV_FLAGS) --platform $(PLATFORM) \
		-v "$(LOCAL_INPUT):/app/nova/local_input.txt:ro" \
		-w /app/nova $(IMAGE) \
		--test_mode --local_input_file /app/nova/local_input.txt $(EXTRA_ARGS)

run-bg:
	@test -n "$(WALLET_NAME)" && test -n "$(WALLET_HOTKEY)" || \
		(echo "error: $(HELP_WALLET_REQ)" >&2 && exit 1)
	@test -f "$(ENV_FILE)" || (echo "error: missing $(ENV_FILE)" >&2 && exit 1)
	docker rm -f "$(CONTAINER_NAME)" 2>/dev/null || true
	docker run -d $(ENV_FLAGS) --platform $(PLATFORM) $(WALLET_VOL) \
		--name "$(CONTAINER_NAME)" -w /app/nova $(IMAGE) \
		--wallet.name "$(WALLET_NAME)" --wallet.hotkey "$(WALLET_HOTKEY)" $(EXTRA_ARGS)

logs:
	docker logs -f "$(CONTAINER_NAME)"

stop:
	docker stop "$(CONTAINER_NAME)" 2>/dev/null || true

rm:
	docker rm -f "$(CONTAINER_NAME)" 2>/dev/null || true

clean:
	docker rmi "$(IMAGE)" 2>/dev/null || true

# Quick import smoke (does not validate chain / wallets / GPUs).
inspect:
	docker run --rm --platform $(PLATFORM) --entrypoint /app/nova/.venv/bin/python3 \
		-w /app/nova -e PYTHONPATH=/app/nova $(IMAGE) \
		-c "import neurons.validator.validator as v; print('ok', v.__file__)"
