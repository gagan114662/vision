# Epic: Lean Integration & Evolutionary Loop
- Aligns with PRD Goals G1, G4
- Owner: Strategy Lab Agent

## Purpose
Operationalize QuantConnect Lean backtesting via MCP, track experiments, and launch AlphaEvolve-style strategy competitions tied to real validation gates.

## Stories
1. **Lean Docker Environment**: Create Dockerfile, compose, and secrets injection workflow.
2. **Lean MCP Wrapper**: Implement tool for backtests, result retrieval, and metadata logging.
3. **Experiment Tracker**: Integrate with provenance ledger, store metrics, statistical checks.
4. **Robust Validation Suite**: Build cross-validation, walk-forward, bootstrap, and benchmark neutrality checks preventing buy-and-hold replication or overfitting.
5. **Evolution Scheduler**: Build mutation/selection pipeline with guardrail agent filters, bias diagnostics, and human review hooks.
6. **Chart Intelligence Tools**: Generate visual overlays (price, signals, risk metrics) accessible via MCP so agents can "see" chart patterns during analysis.
