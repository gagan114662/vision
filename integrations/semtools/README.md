# Semtools Integration Plan

Semtools (https://github.com/run-llama/semtools) provides ultra-fast document parsing and semantic search over local files. This integration will expose semtools capabilities through MCP so that agents can parse corpora and run fuzzy keyword searches without standing up heavy vector infrastructure.

## Goals
- Wrap `semtools parse` and `semtools search` commands behind MCP tools so agents can call them with structured inputs.
- Normalize output (captured stdout) into JSON responses that downstream tools/agents can process.
- Maintain safety: only allow working-directory relative paths and deny shell injection.
- Provide unit tests mocking subprocess execution so integration works even when semtools binary is absent.

## Implementation Steps
1. **CLI Wrapper** – Create a thin Python helper around `subprocess.run` to invoke semtools with explicit arguments and sanitized paths.
2. **MCP Server** – Expose two tools: `semtools.parse` (returns parsed markdown) and `semtools.search` (returns raw search output). Schemas live under `mcp/schemas/tool.semtools.*`.
3. **Semtools Metadata** – Register the tools in `agents/configs/semtools-metadata.yaml` so semtool routing knows about costs/latency.
4. **Tests** – Add unit tests mocking the subprocess call to validate argument construction and error handling.
5. **Docs** – Update story log / roadmap with semtools integration status.

## Prerequisites
- `semtools` binary available on PATH when running in production.
- LlamaParse API key configured via environment variable (`LLAMAPARSE_API_KEY`) if parsing PDFs is required.

## Future Enhancements
- Chunk the parse output into structured JSON (title, sections) instead of plain text.
- Cache parse/search results in local storage to avoid repeated CLI execution.
- Add MCP resource provider to stream large parse results via files rather than JSON.
