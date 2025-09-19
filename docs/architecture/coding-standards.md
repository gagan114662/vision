# Coding Standards
- Prefer Python 3.11+ for agent logic, tooling wrappers, and orchestration scripts; follow PEP 8 with black formatting and mypy type checking.
- Maintain pure ASCII files unless data demands otherwise; include concise comments only for complex logic.
- All MCP tool definitions must include JSON Schema validation, version numbers, verbosity controls, and actionable error messages.
- Lean algorithm code follows QuantConnect guidelines with explicit universe selection, risk controls (`SetRiskManagement`), and cost modeling.
- Secrets (API tokens, credentials) must never be hard-coded; load via environment variables managed by secrets vault.
- Enforce unit/integration tests via pytest with coverage > 85% for libraries; use Lean regression tests for strategy behavior.
- Commit messages follow Conventional Commits; no direct pushes to main without passing QA gate.
