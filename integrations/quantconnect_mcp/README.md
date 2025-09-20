# QuantConnect MCP Integration Plan

This directory captures integration steps for wiring the official [QuantConnect MCP Server](https://github.com/QuantConnect/mcp-server) into the platform once network access is available.

## Goals
- Allow agents to manage QuantConnect cloud projects (synchronize files, trigger backtests, manage deployments) via the MCP protocol.
- Reuse QuantConnect's maintained MCP implementation rather than homegrown CLI wrappers.
- Preserve strict credential handling and audit trails.

## Required Steps
1. **Vendor the Server**
   - Clone `QuantConnect/mcp-server` into `integrations/quantconnect_mcp/vendor/` (submodule or regular clone).
     ```bash
     cd integrations/quantconnect_mcp
     git clone https://github.com/QuantConnect/mcp-server vendor
     ```
   - Follow upstream instructions for installing dependencies (`pip install -r requirements.txt`).

2. **Configuration**
   - Provide QC API credentials via environment variables (`QC_MCP_USER_ID`, `QC_MCP_API_TOKEN`) stored in Vault/`.env.local`.
   - Define project ID/organization defaults in `agents/configs/quantconnect.yaml` (to be created) with RBAC scoping per agent role.

3. **Server Launch**
   - Start the MCP server (e.g., `python -m quantconnect_mcp.server --port 7001`).
   - Register the server endpoint in semtools routing and ensure TLS tunneling if exposed outside localhost.

4. **Tool Mapping**
   - Map MCP methods to our registry (see `mcp/schemas/tool.quantconnect.*`).
   - Primary tools: `quantconnect.project.sync`, `quantconnect.backtest.run`, `quantconnect.backtest.status`, `quantconnect.deployment.list`.

5. **Testing**
   - Implement integration tests that mock server responses for project sync/backtest flows.
   - Add QA gate verifying credentials are present before enabling quantconnect namespace.

6. **Operations**
   - Document monitoring (latency, error codes) and add incident response runbooks.
   - Configure semtools cost metadata so agents understand API budgets.

## Current Status
- Schemas, registry stubs, and agent metadata prepared.
- Awaiting vendored server + environment with network access to complete integration.
