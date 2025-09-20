# Epic: QuantConnect MCP Integration
- Aligns with PRD Goals G1, G4
- Owner: Strategy Lab / Ops SRE

## Purpose
Integrate QuantConnect's official MCP server so agents can control cloud projects (update files, trigger backtests, manage deployments) without relying on manual CLI invocations.

## Stories
1. **Dependency Assessment**: Review QuantConnect MCP server repo and identify required credentials, project structure, and runtime prerequisites.
2. **Sandbox Deployment**: Spin up the MCP server locally using provided Docker compose or Python entrypoints; validate communication with our agent orchestration layer.
3. **Tool Mapping**: Extend `mcp/registry.yaml` and semtools metadata with QuantConnect-specific actions (project sync, cloud backtest, live deploy) ensuring all use case schemas exist.
4. **Credential Management**: Store QC API keys securely (Vault/secrets manager); define RBAC tokens for agent access.
5. **Testing & Guardrails**: Implement integration tests to confirm project uploads, backtest success detection, and error handling. Add QA gates preventing unauthorized usage.
6. **Documentation & Ops**: Document setup, monitoring, and incident response; add runbooks for handling MCP downtime or QC API failures.
