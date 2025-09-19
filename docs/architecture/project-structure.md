# Project Structure
```
.
├── docs/
│   ├── architecture/
│   │   ├── coding-standards.md
│   │   ├── project-structure.md
│   │   └── tech-stack.md
│   ├── epics/
│   ├── stories/
│   ├── prd.md
│   ├── architecture.md
│   └── qa/
│       ├── assessments/
│       └── gates/
├── .bmad-core/
│   ├── core-config.yaml
│   └── data/
│       └── technical-preferences.md (TBD)
├── mcp/
│   ├── servers/ (custom MCP servers per namespace)
│   ├── schemas/ (JSON schemas for tool I/O)
│   └── wrappers/ (Lean, provenance ledger, QA, etc.)
├── agents/
│   ├── workflows/ (LangGraph/CrewAI definitions)
│   ├── prompts/ (role-specific prompts)
│   └── configs/ (Meta-Supervisor settings, semtools metadata)
├── data/
│   ├── raw/ (ingested real datasets; access-controlled)
│   ├── processed/
│   └── feature-store/
├── lean/
│   ├── algorithms/ (QuantConnect strategies)
│   ├── config/ (Lean CLI configs)
│   ├── docker/ (Dockerfiles, compose)
│   └── results/ (backtest outputs, metadata)
├── ops/
│   ├── runbooks/
│   └── monitoring/
├── scripts/
│   ├── tooling/ (CLI utilities)
│   └── maintenance/
└── README.md
```
