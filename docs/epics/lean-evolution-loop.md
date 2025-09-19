# Epic: Lean Integration & Evolutionary Loop
- Aligns with PRD Goals G1, G4
- Owner: Strategy Lab Agent

## Purpose
Operationalize QuantConnect Lean backtesting via MCP, track experiments, and launch AlphaEvolve-style strategy competitions tied to real validation gates.

## Stories
1. **Lean Docker Environment**: Create Dockerfile, compose, and secrets injection workflow.
2. **Lean MCP Wrapper**: Implement tool for backtests, result retrieval, and metadata logging.
3. **Experiment Tracker**: Integrate with provenance ledger, store metrics, statistical checks.
4. **Evolution Scheduler**: Build mutation/selection pipeline with guardrail agent filters and human review hooks.
