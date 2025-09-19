#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${QUANTCONNECT_USER_ID:-}" || -z "${QUANTCONNECT_API_TOKEN:-}" ]]; then
  echo "QUANTCONNECT credentials not set; skipping Lean dry run." >&2
  exit 0
fi

docker build -t vision-lean-ci -f lean/docker/Dockerfile .

docker run --rm \
  -e QUANTCONNECT_USER_ID \
  -e QUANTCONNECT_API_TOKEN \
  -v "$(pwd)/lean:/Lean" \
  vision-lean-ci \
  lean cloud backtest --project monthly_universe_alpha --config /Lean/config/monthly_universe_alpha.json --dry-run
