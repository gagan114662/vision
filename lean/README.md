# Lean / QuantConnect Integration

## Credentials
- **QuantConnect User ID**: 357130
- **Token**: `QUANTCONNECT_API_TOKEN` environment variable (store securely in Vault/Secrets Manager).
- Never commit the token; inject via runtime secrets. Example `.env` entry:
  ```bash
  export QUANTCONNECT_USER_ID=357130
  export QUANTCONNECT_API_TOKEN=... # stored securely
  ```

## Dockerized CLI Workflow
1. Build Lean CLI image using `lean/docker/Dockerfile`:
   ```bash
   docker build -t vision-lean -f lean/docker/Dockerfile .
   ```
2. Ensure credentials are available at runtime (either environment variables or Docker secrets).
3. Run the bundled monthly universe strategy:
 ```bash
 docker run --rm \
   -e QUANTCONNECT_USER_ID \
   -e QUANTCONNECT_API_TOKEN \
   -v $(pwd)/lean:/Lean \
   vision-lean \
   lean backtest monthly_universe_alpha --config /Lean/config/monthly_universe_alpha.json --data-folder /Lean/data
 ```
4. Capture output JSON and statistics to `lean/results/` with metadata (config hash, Git SHA).
5. Expose commands through MCP wrapper `strategy.eval.run_backtest` for agent access.

### CLI Helper
- Script: `scripts/tooling/run_monthly_backtest.py`
- Usage:
  ```bash
  export QUANTCONNECT_USER_ID=357130
  export QUANTCONNECT_API_TOKEN=... # secure storage
  python scripts/tooling/run_monthly_backtest.py
  ```
- Optional overrides: `--project`, `--algorithm-path`, `--config-path`, `--docker-image`, and `--param key=value`.

### Monthly Universe Alpha Strategy
- **File**: `lean/algorithms/monthly_universe_alpha.py`
- **Config**: `lean/config/monthly_universe_alpha.json`
- **Method**: Monthly coarse/fine universe selection (USA, market cap > $2B), scoring by 3-month momentum penalized by volatility, top 20 equal-weighted.
- **Expectations**: Backtest over 2010–present (~15 years) using QuantConnect price data; agents must log provenance IDs for data requests and Lean config hash for reproducibility.

## Validation Pipeline
- Stage 1: Historical backtest over 10+ years with monthly universe selection.
- Stage 2: Forward-walk / paper trading via Lean live mode; monitor slippage and execution quality.
- Stage 3: Human Alpha Council review before capital deployment.

## Next Steps
- Automate Lean backtest metadata ingestion into experiment tracker and provenance ledger.
- Extend algorithm library with additional strategies (pairs, factor rotation) following same provenance discipline.
- Wire `strategy.eval.run_backtest` MCP wrapper to consume `lean/config/monthly_universe_alpha.json` defaults when parameters omitted.
