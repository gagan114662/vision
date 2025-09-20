# Story: Platform Foundation — Docs & Governance
- **Epic**: platform-foundation
- **Owner**: Scrum Master / Documentation Agent
- **Acceptance Criteria**:
  1. PRD, architecture, coding standards, tech stack, project structure, technical preferences committed.
  2. QA risk assessment completed and stored in `docs/qa/assessments/`.
  3. `.bmad-core` config references required always-load files.
  4. Plan for subsequent stories logged with blockers noted.
- **QA Hooks**: `@qa *risk platform.foundation`, `@qa *design platform.foundation` (pending), `@qa *gate platform.foundation` before completion.
- **Notes**: Future updates will shard architecture and PRD further as epics expand.

## Progress Notes (2025-09-19)
- QA design and gate documents created; status pending until validation scripts implemented.
- Additional epics (`data-provenance-mvp`, `mcp-agent-fabric`, `lean-evolution-loop`, `risk-compliance-execution`) drafted as part of sharding.
- Next actions: implement doc validation tests, update QA gate to PASS after CI integration.
- Doc integrity tests executed locally (`python3 -m unittest tests.qa.test_docs`); CI wiring outstanding.
- Added provenance server unit tests alongside doc checks and re-ran local suite.
- Added CI workflow and registry/schema coverage to QA tests; full suite passing locally.
- Added agent prompts and Meta-Supervisor workflow scaffolding for BMad-aligned orchestration.
- Extended MCP registry with market data and backtest schemas; QA tests updated accordingly.
- Hardened provenance MCP server with signed responses and extended unit tests verifying HMAC signatures.
- Extended MCP coverage with feature engineering, risk, and compliance servers plus schemas and tests; local suite expanded to 21 tests.
- Added default Lean monthly universe strategy and config; semtools metadata now references expanded toolset.
- Added CLI helper `scripts/tooling/run_monthly_backtest.py` to trigger default Lean strategy with secure env vars.
- Captured continuous innovation plan (Innovation Scout agent, research feed ingestion, Lean compile CI) across architecture and roadmap.
- Innovation Scout assets seeded (sources config, research feed server, workflow) plus Lean dry-run CI workflow and script.
- Updated PRD/architecture/roadmap to mandate anti-overfitting protocol, multi-regime validation, and chart-vision tooling.
- Implemented robustness MCP (`strategy.validation.run_robustness`) and chart visualization tool, plus semtools/registry integration with expanded QA coverage (28 tests).
- Added strategy validation workflow and innovation processing script linking research feed to backtests, robustness, and chart rendering.
- Added lean dry-run runbook and upgraded innovation processor to enforce credential checks and feed status updates.
- Added QuantConnect MCP integration blueprint, schemas, registry entries, and stubs pending upstream server vendoring.
- Documented vendor procedure for QuantConnect MCP server (`integrations/quantconnect_mcp/README.md`) so upstream repo can be cloned into workspace.
- Added tool enhancement backlog from `tooly.md` to upcoming to-do (LlamaIndex semtools, framework evaluation, registry/tooling patterns).
- Implemented semtools MCP stubs (parse/search) with CLI wrappers and tests for document intelligence workflows.
- Added `scripts/tooling/run_quantconnect_backtest.py` to sync local files with QuantConnect and automate compile/backtest polling via the MCP integration.
- Shipped Hidden Markov Model regime detection tool (`strategy.regime.detect_states`) with MCP schema, server wrapper, and unit tests (requires hmmlearn when enabled).
- Added Ornstein-Uhlenbeck mean reversion parameter estimator (`strategy.meanreversion.estimate_parameters`) with MCP schemas, server implementation, and tests.
- Enhanced semtools parse tool to optionally return structured markdown sections for downstream agents.
- Added Fourier cycle detection tool (`signal.fourier.detect_cycles`) with schemas, server implementation, and tests (returns dominant frequencies/amplitudes via FFT).
