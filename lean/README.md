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
1. Build Lean CLI image using `lean/docker/Dockerfile` (TBD).
2. Run backtests with
   ```bash
   docker run --rm \
     -e QUANTCONNECT_USER_ID \
     -e QUANTCONNECT_API_TOKEN \
     -v $(pwd)/lean:/Lean \
     quantconnect/lean:latest \
     lean backtest MyProject --data-folder /Lean/data
   ```
3. Capture output JSON and statistics to `lean/results/` with metadata (config hash, Git SHA).
4. Expose commands through MCP wrapper `strategy.eval.run_backtest` for agent access.

## Validation Pipeline
- Stage 1: Historical backtest over 10+ years with monthly universe selection.
- Stage 2: Forward-walk / paper trading via Lean live mode; monitor slippage and execution quality.
- Stage 3: Human Alpha Council review before capital deployment.

## Next Steps
- Author Dockerfile + compose for reproducible Lean environment.
- Define standard Lean project template in `lean/algorithms/`.
- Create script to register backtest metadata into experiment tracker and provenance ledger.
