# Tool Enhancement Backlog (tooly.md distilled)

## Objective
Accelerate tool capabilities by implementing the improvements catalogued in `tooly.md`. This backlog is structured so tasks can be tackled incrementally while keeping the MCP-first architecture.

## Categories & Key Actions

### 1. Document Intelligence (semtools)
- [x] Wrap `semtools parse` & `semtools search` as MCP tools (done).
- [ ] Add semantic chunking + Markdown-to-JSON parser for parsed docs.
- [ ] Cache parse/search outputs for large corpora.
- [ ] Add tooling to annotate results (highlight keywords, sentiment tagging).

### 2. MCP Framework Extensions
- [ ] Evaluate LangGraph vs CrewAI vs Microsoft Semantic Kernel for debate/role-based workflows (benchmark latency/token usage).
- [ ] Integrate multi-agent debate loop (Strategy Lab ↔ Risk Sentinel) using chosen framework.
- [ ] Add registry scoring metadata (latency, accuracy, freshness) to improve tool selection.

### 3. Performance & Cost
- [ ] Implement KV caching + token budget monitoring for expensive tool calls.
- [ ] Introduce speculative inference / batching for repeated tool usage patterns.
- [ ] Add observability dashboard for tool-level latency + cost metrics.

### 4. Math & Signal Tools
- [ ] Hidden Markov Model regime detector (HMM) tool namespace (`strategy.regime.*`).
- [ ] Ornstein–Uhlenbeck parameter estimator (`strategy.meanreversion.*`).
- [ ] Fourier/wavelet signal processing toolkit (`signal.*`).

### 5. Alternative Data
- [ ] Define schemas + MCP stubs for satellite/transaction/sentiment data sources.
- [ ] Build ingestion pipeline with provenance and quality checks.

### 6. Automation
- [x] QuantConnect backtest automation script (done).
- [ ] Schedule innovation pipeline runs (process innovation script in CI with project/backtest outputs).

## Execution Notes
- Align each new tool with Anthropic’s MCP design principles (clear schemas, actionable errors, verbosity controls).
- Ensure every addition includes unit tests + doc updates (use `tests/mcp/test_*.py` pattern).
- Update story log & roadmap as milestones are completed.
