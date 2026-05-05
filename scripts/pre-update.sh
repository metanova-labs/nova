#!/usr/bin/env bash
set -euo pipefail
# Called by Watchtower inside the validator container BEFORE it is stopped during an image rollout.
# Exit 0  → Watchtower proceeds with recreate.
# Exit 75 → EX_TEMPFAIL — skip this rollout attempt until the next poll (typically still scoring).
PORT="${READINESS_PORT:-8080}"
code=""
code="$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
  "http://127.0.0.1:${PORT}/ready_to_update" || true)"
code="${code:-000}"

if [[ "$code" == "200" ]]; then
  exit 0
fi
exit 75
