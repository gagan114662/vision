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
