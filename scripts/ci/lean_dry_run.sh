#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${QUANTCONNECT_USER_ID:-}" || -z "${QUANTCONNECT_API_TOKEN:-}" ]]; then
  echo "QUANTCONNECT credentials not set; skipping Lean dry run." >&2
  exit 0
fi

# Install Lean CLI
echo "Installing Lean CLI..."
pip install --upgrade lean

# Login to QuantConnect
echo "Logging into QuantConnect..."
lean login --user-id "${QUANTCONNECT_USER_ID}" --api-token "${QUANTCONNECT_API_TOKEN}"

# Copy strategy files to the project directory
echo "Preparing VisionMonthlyAlpha project..."
mkdir -p ~/quantconnect-cli/VisionMonthlyAlpha
cp -f lean/algorithms/monthly_universe_alpha.py ~/quantconnect-cli/VisionMonthlyAlpha/main.py
cp -f lean/config/monthly_universe_alpha.json ~/quantconnect-cli/VisionMonthlyAlpha/config.json

# Push the project to cloud
echo "Pushing project to QuantConnect cloud..."
cd ~/quantconnect-cli
lean cloud push

# Run the cloud backtest
echo "Running cloud backtest..."
BACKTEST_OUTPUT=$(lean cloud backtest "VisionMonthlyAlpha" --name "ci-dry-run-$(date +%Y%m%d-%H%M%S)")

# Extract backtest ID from output
BACKTEST_ID=$(echo "$BACKTEST_OUTPUT" | grep "Backtest id:" | awk '{print $3}')
PROJECT_ID=$(echo "$BACKTEST_OUTPUT" | grep "project/" | sed 's/.*project\/\([0-9]*\).*/\1/')

echo "Backtest completed:"
echo "  Project ID: ${PROJECT_ID}"
echo "  Backtest ID: ${BACKTEST_ID}"
echo "  URL: https://www.quantconnect.com/project/${PROJECT_ID}/${BACKTEST_ID}"

# Check if backtest had errors
if echo "$BACKTEST_OUTPUT" | grep -q "An error occurred"; then
  echo "Warning: Backtest encountered runtime errors"
  exit 1
fi
