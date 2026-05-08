# NOVA Validator — Docker Compose and Watchtower deployment

This guide covers deployment where the image comes from Docker Hub (`latest`, `main`, `sha-…`; CI gates in [docs/CICD.md](docs/CICD.md)).

## Prerequisites

1. Linux with Docker + [Compose v2](https://docs.docker.com/compose/) and, when using a GPU, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
2. A host directory holding `.bittensor/wallets` (Compose mounts it read-only at `/root/.bittensor/wallets:ro`).
3. An `.env` file at the repo root (start from `example.env`).

## Relevant variables


| Variable                 | Purpose                                                                                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `WALLET_BIND`            | Absolute host path to wallets in production. Dev fallback in Compose: `/tmp/nova-wallets` (create and copy wallets there if you need a smoke test). |
| `VALIDATOR_IMAGE`        | Overrides the Docker image (e.g. `user/nova-validator:latest`). If unset: `nova-validator:local` from `docker compose build`.                       |
| `READINESS_PORT`         | Defaults to `8080`. Used when `AUTO_UPDATE` enables the readiness HTTP server; align with the image `HEALTHCHECK` and Watchtower’s `/ready_to_update` probe.                                  |
| `WATCHTOWER_NOTIFICATIONS` / `WATCHTOWER_NOTIFICATION_URL` | Optional Shoutrrr notifications for Watchtower. Set both to enable (e.g. `WATCHTOWER_NOTIFICATIONS=shoutrrr` and a valid `WATCHTOWER_NOTIFICATION_URL` per [Shoutrrr services](https://containrrr.dev/shoutrrr/v0.8/services/overview/)). Leave `WATCHTOWER_NOTIFICATIONS` unset or empty to run Watchtower without notifications (URL is ignored). If `WATCHTOWER_NOTIFICATIONS=shoutrrr` but the URL is empty or invalid, Watchtower may fail on startup. |
| `AUTO_UPDATE`            | Opt-in for safe rollouts: values `1` / `true` / `yes` (case-insensitive) start the readiness HTTP server (`/ready_to_update`, `/healthz`) so Watchtower lifecycle hooks work. Signal handlers (drain on SIGTERM/SIGINT) are always installed. |

## Watchtower image and polling

The stack uses **[`nickfedor/watchtower`](https://hub.docker.com/r/nickfedor/watchtower)** (maintained fork) instead of the discontinued `containrrr/watchtower` image. The service is configured with CLI flags and optional notification environment variables in [docker-compose.yml](docker-compose.yml):

- `--label-enable` — only labeled containers are monitored (validator has `com.centurylinklabs.watchtower.enable=true`).
- `--enable-lifecycle-hooks` — runs `scripts/pre-update.sh` before stopping the validator (gates on `/ready_to_update`).
- `--cleanup` — removes old images after update.
- `--interval 300` — checks for new images every **5 minutes** (same cadence as the previous `WATCHTOWER_POLL_INTERVAL=300` env).
- Notifications are opt-in via the env vars `WATCHTOWER_NOTIFICATIONS` and `WATCHTOWER_NOTIFICATION_URL`. Set both to enable Shoutrrr (for example `WATCHTOWER_NOTIFICATIONS=shoutrrr`, `WATCHTOWER_NOTIFICATION_URL=discord://token@id?title=Validator`); leave them unset to run Watchtower silently.

## Required runtime components

The recommended deployment is the full stack defined in [docker-compose.yml](docker-compose.yml):

- Set **`AUTO_UPDATE=1`** in `.env` so the `validator` starts the readiness HTTP server (`/ready_to_update`, `/healthz`). Watchtower’s `pre-update.sh` depends on that endpoint; without it, lifecycle hooks will fail and rollouts are unsafe.
- The `watchtower` service is part of the recommended deployment when you want automatic image updates from Docker Hub. It drives safe rollouts by calling `scripts/pre-update.sh`, which probes `/ready_to_update` and exits `75` while the validator is in its critical epoch window.

Operators who remove `watchtower` from their local compose must manage rollouts manually. The in-process `auto_updater.py` git-pull path is not used in this Compose flow.

## Update flow (Watchtower + readiness)

```mermaid
flowchart TD
  Watchtower[Watchtower_poll_cycle] --> Detect{New_image_available?}
  Detect -- "no" --> Sleep[Wait_next_poll]
  Detect -- "yes" --> PreUpdate[Run_pre-update_hook_in_validator_container]

  PreUpdate --> Curl[pre-update.sh_calls_ready_to_update]
  Curl --> Ready{HTTP_200?}
  Ready -- "no_(e.g._423_busy)" --> Skip[Exit_75_skip_rollout_attempt]
  Skip --> Sleep

  Ready -- "yes" --> Proceed[Exit_0_proceed_with_rollout]
  Proceed --> Stop[Docker_sends_SIGTERM_to_validator]
  Stop --> Drain[Validator_sets_drain_requested]

  Drain --> Busy{In_epoch_busy_scope?}
  Busy -- "yes" --> WaitSafe[Finish_scoring_weights_payouts]
  WaitSafe --> Exit[Exit_at_safe_point]
  Busy -- "no" --> Exit
  Exit --> Recreate[Watchtower_recreates_validator_with_new_image]
```

## Quick start

```bash
cp example.env .env          # edit SUBTENSOR, keys, set AUTO_UPDATE=1 for Watchtower + readiness, etc.
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

These endpoints exist only when **`AUTO_UPDATE` is enabled** (`1` / `true` / `yes`); otherwise the readiness HTTP server is not started.

- `GET /ready_to_update`: **200** when the validator is outside the critical epoch window (`process_epoch`, `set_weights`, payouts under `lifecycle.epoch_busy_scope`). **423** while that work runs — `scripts/pre-update.sh` exits **75** so Watchtower skips this rollout attempt.
- `GET /healthz`: **200** when the readiness HTTP server responds (used by the image `HEALTHCHECK` when configured).

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