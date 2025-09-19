# Technology Stack
## Core Languages & Frameworks
- Python 3.11+: agent orchestration, MCP servers, data engineering scripts.
- C#/Python Lean Algorithms: strategy implementations within QuantConnect.

## AI & Agent Tooling
- Anthropic MCP framework (custom registry + connectors).
- run-llama `semtools` for semantic tool routing and memory.
- LangGraph or CrewAI for deterministic multi-agent workflows.
- AlphaEvolve-inspired evolutionary scheduler for strategy competition.

## Data Infrastructure
- Ingestion: Apache Airflow or Prefect orchestrating ETL into object storage + warehouse (e.g., S3 + Snowflake/BigQuery) with immutability layers.
- Knowledge Graph: Neo4j or TigerGraph storing entity-event relationships sourced from real filings, transcripts, logistics feeds.
- Provenance Ledger: immudb or AWS QLDB providing append-only, verifiable records.
- Feature Store: Feast or custom store backed by parquet datasets with versioning.

## Backtesting & Execution
- QuantConnect Lean CLI in Docker (with secure credentials) for backtests/paper trading.
- Kubernetes or ECS to orchestrate Lean worker pods; colocation servers for low-latency execution.
- Smart order routing microservices implemented in Python/Go interfacing with broker/exchange APIs.

## Observability & Ops
- Logging: OpenTelemetry + ELK or Datadog.
- Metrics: Prometheus/Grafana or Datadog.
- Secrets: HashiCorp Vault or AWS Secrets Manager.
- CI/CD: GitHub Actions with security scans (Trivy, Bandit), linting (black, flake8, mypy), test suites.

## Security & Compliance
- Zero-trust network overlay (e.g., Tailscale, BeyondCorp).
- SIEM integrations (Splunk/Datadog) for Security Watch agent.
- Compliance tooling: automated report generators using Python, stored in compliance MCP.
