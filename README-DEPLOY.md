# NOVA Validator — Docker Compose and Watchtower deployment

This guide covers deployment where the image comes from Docker Hub (`latest`, `main`, `sha-…`; CI gates in [docs/CICD.md](docs/CICD.md)).

## Prerequisites

1. Linux with Docker + [Compose v2](https://docs.docker.com/compose/) and, when using a GPU, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
2. A host directory holding `.bittensor/wallets` (Compose mounts it read-only at `/root/.bittensor/wallets:ro`).
3. An `.env` file at the repo root (start from `example.env`).

## Relevant variables

| Variable | Purpose |
|----------|---------|
| `WALLET_BIND` | Absolute host path to wallets in production. Dev fallback in Compose: `/tmp/nova-wallets` (create and copy wallets there if you need a smoke test). |
| `VALIDATOR_IMAGE` | Overrides the Docker image (e.g. `user/nova-validator:latest`). If unset: `nova-validator:local` from `docker compose build`. |
| `READINESS_PORT` | Defaults to `8080`. Must stay aligned with the image `HEALTHCHECK` and with Watchtower’s `/ready_to_update` probe. |
| `READINESS_HTTP_DISABLE` | `true`/`1` disables the readiness HTTP server — do **not** use this with Watchtower. |
| `AUTO_UPDATE=1` | Legacy; ignored by `validator.py` and logged as obsolete. |

## Quick start

```bash
cp example.env .env          # edit SUBTENSOR, keys, etc.
export WALLET_BIND="$HOME/.bittensor/wallets"
docker compose up -d --build
docker compose logs -f validator watchtower
```

Check readiness with Make (**stack must already be running**):

```bash
make compose-ready
```

Or manually inside the `validator` service container:

```bash
docker compose exec validator sh -lc 'curl -fsS http://127.0.0.1:${READINESS_PORT:-8080}/ready_to_update'
```

## Endpoints

- `GET /ready_to_update`: **200** when the validator is outside the critical epoch window (`process_epoch`, `set_weights`, payouts under `lifecycle.epoch_busy_scope`). **423** while that work runs — `scripts/pre-update.sh` exits **75** so Watchtower skips this rollout attempt.
- `GET /healthz`: **200** when the readiness HTTP server responds (used by the image `HEALTHCHECK`).

## Safe point and signals

The *safe point* is **after** `process_epoch`, `set_weights`, and any `dispatch_bounty_payouts` finish, still inside the same `async with lifecycle.epoch_busy_scope()` block.

On `SIGTERM`/`SIGINT`, *drain* mode turns on: the main loop only calls `exit` when not inside that critical section, within the **30m** `stop_grace_period` set in [docker-compose.yml](docker-compose.yml).

## Failure modes documented here

- **SIGKILL** / hard kill during scoring: work may be incomplete; pinning by digest reduces rollout regressions while debugging.
- **Crash during payouts**: errors are logged as in current code; the next cycle re-evaluates per existing validator logic.
- **Bad image pushed**: roll back by pulling an immutable SHA tag produced by the pipeline (`sha-<full>` from the metadata action).
- **Disk full during pull**: the old image may keep running until space is freed; prune intermediate images.
- **Flaky subtensor RPC**: `reconnect_subtensor` already handles failures in the loop; *drain* does **not** force exit mid-epoch — it waits until the current boundary work finishes.

## Rationale (tests + updater)

- **pytest + ruff** outside Docker and in-image *smokes*: see [docs/CICD.md](docs/CICD.md).
- **Watchtower + lifecycle hooks** instead of `auto_updater.py` in-process — avoids `git pull`/restart amid the GPU stack and matches Docker Hub artifact flows.

### Observability (Loki / Promtail)

Deferred as a future extension (out of current scope).

## Suggested local E2E smoke

```bash
make build && make inspect
docker compose config -q
make test       # quick YAML + readiness regression in unit tests (no daemon)
```

On Apple Silicon always use `PLATFORM=linux/amd64 make build` as in the main README.
