"""Utilities for loading and validating the MCP tool registry."""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import yaml


@dataclass(frozen=True)
class ToolDefinition:
    """Tool definition as captured in the registry YAML."""

    id: str
    namespace: str
    schema: str
    response_schema: Optional[str]
    description: str
    auth: str
    signing: str
    schema_path: Path
    response_schema_path: Optional[Path]


@dataclass(frozen=True)
class ToolImplementation:
    """Tool implementation discovered from decorated server functions."""

    name: str
    schema: Optional[str]
    schema_path: Optional[Path]
    module: str
    obj: Any


@dataclass(frozen=True)
class DiscoveryResult:
    """Discovery artefacts for implemented tools."""

    implementations: Dict[str, ToolImplementation]
    duplicate_modules: Dict[str, Set[str]]
    import_errors: List[str]


@dataclass(frozen=True)
class ValidationIssue:
    """Represents an inconsistency detected during validation."""

    tool_id: str
    kind: str
    detail: str


class MCPRegistry:
    """Load and validate the MCP registry against runtime implementations."""

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        self.registry_path = registry_path or Path(__file__).resolve().parent / "registry.yaml"
        self.registry_dir = self.registry_path.parent
        raw_data = yaml.safe_load(self.registry_path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, dict):
            raise ValueError("registry.yaml must define a mapping at the top level")
        self._data = raw_data

    def _resolve_path(self, path_str: Optional[str]) -> Optional[Path]:
        if not path_str:
            return None
        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate = (self.registry_dir / candidate).resolve()
        return candidate

    def tool_definitions(self) -> List[ToolDefinition]:
        namespaces = self._data.get("namespaces", {})
        definitions: List[ToolDefinition] = []
        for namespace, payload in namespaces.items():
            for tool in payload.get("tools", []) or []:
                tool_id = tool.get("id")
                schema = tool.get("schema")
                if not tool_id or not schema:
                    raise ValueError(f"Registry entry missing id/schema under namespace '{namespace}'")
                response_schema = tool.get("response_schema")
                definitions.append(
                    ToolDefinition(
                        id=tool_id,
                        namespace=namespace,
                        schema=schema,
                        response_schema=response_schema,
                        description=tool.get("description", ""),
                        auth=tool.get("auth", ""),
                        signing=tool.get("signing", ""),
                        schema_path=self._resolve_path(schema),
                        response_schema_path=self._resolve_path(response_schema),
                    )
                )
        return definitions

    def discover_implementations(self) -> DiscoveryResult:
        package = importlib.import_module("mcp.servers")
        implementations: Dict[str, ToolImplementation] = {}
        duplicate_modules: Dict[str, Set[str]] = {}
        import_errors: List[str] = []

        for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                module = importlib.import_module(module_info.name)
            except Exception as exc:  # pragma: no cover - exercised in integration tests
                import_errors.append(f"Failed to import {module_info.name}: {exc}")
                continue

            for attr in module.__dict__.values():
                name = getattr(attr, "_mcp_tool_name", None)
                if not name:
                    continue
                schema = getattr(attr, "_mcp_schema", None)
                impl = ToolImplementation(
                    name=name,
                    schema=schema,
                    schema_path=self._resolve_path(schema) if schema else None,
                    module=module_info.name,
                    obj=attr,
                )

                if name in implementations:
                    duplicate_modules.setdefault(name, {implementations[name].module}).add(module_info.name)
                else:
                    implementations[name] = impl

        return DiscoveryResult(implementations=implementations, duplicate_modules=duplicate_modules, import_errors=import_errors)

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        definitions = self.tool_definitions()
        discovery = self.discover_implementations()

        for error in discovery.import_errors:
            issues.append(ValidationIssue(tool_id="", kind="import_error", detail=error))

        for tool_id, modules in discovery.duplicate_modules.items():
            module_list = ", ".join(sorted(modules))
            issues.append(ValidationIssue(tool_id=tool_id, kind="duplicate_implementation", detail=module_list))

        seen_ids: Set[str] = set()
        for definition in definitions:
            if definition.id in seen_ids:
                issues.append(ValidationIssue(tool_id=definition.id, kind="duplicate_registry_entry", detail="duplicate id in registry"))
            else:
                seen_ids.add(definition.id)

            if not definition.schema_path or not definition.schema_path.exists():
                issues.append(ValidationIssue(tool_id=definition.id, kind="missing_schema_file", detail=definition.schema))

            if definition.response_schema and (not definition.response_schema_path or not definition.response_schema_path.exists()):
                issues.append(ValidationIssue(tool_id=definition.id, kind="missing_response_schema_file", detail=definition.response_schema))

            implementation = discovery.implementations.get(definition.id)
            if not implementation:
                issues.append(ValidationIssue(tool_id=definition.id, kind="missing_implementation", detail="Tool not implemented under mcp.servers"))
                continue

            if implementation.schema_path is None:
                issues.append(ValidationIssue(tool_id=definition.id, kind="implementation_missing_schema", detail=f"{implementation.module} lacks schema metadata"))
            else:
                if not implementation.schema_path.exists():
                    issues.append(ValidationIssue(tool_id=definition.id, kind="implementation_schema_missing", detail=str(implementation.schema_path)))
                else:
                    registry_path = definition.schema_path.resolve() if definition.schema_path else None
                    impl_path = implementation.schema_path.resolve()
                    if registry_path and registry_path != impl_path:
                        issues.append(ValidationIssue(tool_id=definition.id, kind="schema_mismatch", detail=f"registry={registry_path} implementation={impl_path}"))

        return issues


__all__ = [
    "MCPRegistry",
    "ToolDefinition",
    "ToolImplementation",
    "DiscoveryResult",
    "ValidationIssue",
]
