# Execution Roadmap & Validation Checkpoints

## Phase 0 — Planning & Governance (Week 0-2)
- Finalize stakeholder alignment, regulatory scope, capital constraints.
- Populate expert council roster; schedule Alpha Council cadence.
- Complete QA `@qa *design` and `@qa *gate` for platform foundation story.
- Deliver incident response runbook and documentation agent setup.

## Phase 1 — Data & Provenance MVP (Week 2-6)
- Integrate first exchange + fundamentals data providers; implement schema validation and QC scoring.
- Deploy immudb provenance ledger prototype with MCP access tool.
- Build data anomaly detection and vendor cross-check automation.
- Validation: provenance completeness > 95%, no unresolved QC alerts > 24h.

## Phase 2 — MCP & Agent Fabric (Week 6-10)
- Implement core MCP servers (market-data, provenance, Lean wrapper, QA).
- Configure semtools routing with embeddings, nightly audits, cost governance.
- Author LangGraph workflows for Strategy Lab, Data Edge, Risk Sentinel, QA.
- Introduce Innovation Scout agent monitoring external research feeds with provenance logging.
- Validation: tool routing accuracy > 90% on test corpus; signed responses enforced; research feed ingestion operating with daily updates.
- Incorporate tool augmentation backlog (see `tooly.md`): evaluate semtools for document intelligence, benchmark LangGraph/CrewAI/Semantic Kernel, and design registry-driven tool discovery patterns.

## Phase 3 — Lean Backtesting & Evolutionary Loop (Week 10-16)
- Containerize Lean environment; automate backtests via MCP.
- Develop experiment tracker integration with provenance ledger.
- Launch AlphaEvolve loop: mutation strategies, guardrail filtering, human review queue.
- Add Lean compilation dry-run pipeline (docker build + `lean cloud backtest --dry-run`) to reduce runtime failures.
- Implement robustness suite (cross-validation, walk-forward, bootstrap, benchmark neutrality) and visual analytics tooling.
- Validation: first strategy candidate passing statistical gates (Sharpe > 1.3, Calmar > 1.0) and QA approval.

## Phase 4 — Risk, Compliance & Execution Hardening (Week 16-22)
- Deploy risk MCP (VaR/CVaR, stress events) and compliance reporting automation.
- Implement smart order routing simulation using real historical execution data.
- Conduct red-team security drills, compliance walkthroughs.
- Innovation Scout delivers at least two curated external strategies/tools evaluated by Strategy Lab.
- Validation: compliance matrix coverage 100%; security findings resolved; mock SEV1 drill completed; robustness suite demonstrates stability across three distinct market regimes.

## Phase 5 — Paper Trading & Monitoring (Week 22-28)
- Initiate Lean paper trading with human oversight.
- Enhance observability dashboards, cost governance, documentation agents.
- Run Alpha Council reviews on live metrics; iterate on strategies failing forward tests.
- Monitor Lean compile/run SLO dashboards; auto-create incidents when thresholds breached.
- Validation: paper trading returns meet targets for 3 consecutive months; incident MTTR < 30 minutes.

## Phase 6 — Production Readiness (Post Week 28)
- Final regulatory sign-off, disaster recovery tests, capital allocation approval.
- Transition to limited live deployment with phased capital ramp.
- Maintain continuous AlphaEvolve cycles, compliance audits, and periodic human reviews.

## Key Checkpoints
- **Backtest Gate**: No Lean results accepted without provenance check, statistical validation, and QA gate pass.
- **Risk Gate**: Strategies exceeding limits automatically quarantined; human approval required for override.
- **Compliance Gate**: Report templates generated and reviewed before any live trade authorization.
- **Security Gate**: Prompt-injection tests and credential audits run monthly; failures block deployments.
- **Cost Gate**: Budget governance agent enforces quotas; overages require Alpha Council consent.
