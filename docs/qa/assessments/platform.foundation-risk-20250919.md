# QA Risk Assessment — Platform Foundation
- **Epic**: platform
- **Story**: foundation
- **Date**: 2025-09-19

## Risk Drivers
1. **Data Integrity**: Multiple external vendors introduce schema drift and latency; failure could corrupt feature store.
2. **Strategy Overfitting**: Evolutionary loops may converge on historical artifacts without proper embargoing and statistical checks.
3. **Execution Latency**: Lean backtesting may not capture live venue latency; gap between model and execution performance.
4. **Compliance Exposure**: Insufficient mapping to Reg SCI/MiFID II may delay production or trigger regulatory penalties.
5. **Security Threats**: Agent misuse, prompt injection, or credential leakage could compromise trading decisions.
6. **Operational Complexity**: Large MCP/agent ecosystem increases change management and incident response burden.

## Mitigations
- Build redundant validation layers, anomaly detection, and daily QC reports; enforce provenance ledger reconciliation.
- Strict data splits with embargo, factor-neutral benchmarking, human review; integrate guardrail agents.
- Model latency budgets, incorporate market impact and slippage models; align Lean paper trading with execution analytics.
- Early compliance engagement, automated control mapping, audit trail tooling; maintain readiness reports.
- Zero-trust security controls, signed tool responses, red-team drills, Security Watch monitoring.
- Dedicated SRE team, documentation agents, incident playbooks, continuous semtools metadata audits.

## Risk Rating
- **Overall**: High
- **Probability**: Medium-High
- **Impact**: Critical (capital loss, compliance sanctions)
- **Action**: Mitigations mandatory before production deployment; track via QA gate updates.
