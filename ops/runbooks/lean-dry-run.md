# Lean Dry-Run Workflow Guidance

## Purpose
Ensure QuantConnect Lean compilation/backtest issues are caught before strategy agents run large experiments. The GitHub Actions workflow `.github/workflows/lean-dryrun.yml` builds the Docker image and executes `lean cloud backtest --dry-run` weekly (and on demand).

## Secrets Configuration
1. In the GitHub repository settings (`Settings → Secrets and variables → Actions`), add:
   - `QUANTCONNECT_USER_ID` = `357130`
   - `QUANTCONNECT_API_TOKEN` = *(use secure storage; never commit)*
2. Optional: store the same secrets locally via a `.env` file when testing `scripts/ci/lean_dry_run.sh`.

## Manual Execution
```bash
export QUANTCONNECT_USER_ID=357130
export QUANTCONNECT_API_TOKEN=... # secure
scripts/ci/lean_dry_run.sh
```

## Monitoring
- GitHub Actions → workflow "Lean Dry Run" for success/failure history.
- Configure repository notifications to email/SMS Ops team on failures.
- Add metrics to Ops dashboards:
  - `lean_dryrun_duration` (job runtime)
  - `lean_dryrun_status` (0=fail,1=success)

## Incident Response
1. If dry-run fails, inspect action logs (likely compile error, missing dependency, or credential issue).
2. Create incident ticket with failure signature and assign to Ops/SRE + Strategy Lab.
3. Pause automated backtests referencing failing commit until resolved.
4. Document fix in `docs/stories/...` and run dry-run manually before re-enabling schedule.
