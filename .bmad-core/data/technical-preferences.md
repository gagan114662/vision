# Technical Preferences
- Favor Python 3.11+, black formatting, mypy strict mode.
- Prefer LangGraph for deterministic orchestration with CrewAI debate fallback.
- Use immudb for provenance ledger to guarantee tamper evidence.
- Default containerization via Docker; orchestration via Kubernetes.
- Observability stack: OpenTelemetry + Grafana + Loki.
- Security: enforce hardware-backed secrets (YubiHSM/AWS KMS) and signed MCP responses.
