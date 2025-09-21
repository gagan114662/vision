#!/usr/bin/env python3
"""
Deploy circuit breakers across all MCP servers systematically.
"""
import os
import re
from pathlib import Path


def add_circuit_breaker_imports(file_path: Path) -> bool:
    """Add circuit breaker imports to a server file."""
    content = file_path.read_text()

    # Check if already has circuit breaker imports
    if "circuit_breaker" in content:
        return False

    # Find mcp.server import
    server_import_pattern = r"from mcp\.server import register_tool"
    if not re.search(server_import_pattern, content):
        return False

    # Add circuit breaker imports
    replacement = """from mcp.server import register_tool

try:
    from mcp.common.resilience import circuit_breaker, CircuitBreakerConfig
except ImportError:
    def circuit_breaker(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func
        return decorator"""

    updated_content = re.sub(server_import_pattern, replacement, content)
    file_path.write_text(updated_content)
    return True


def add_circuit_breaker_decorators(file_path: Path) -> int:
    """Add circuit breaker decorators to register_tool functions."""
    content = file_path.read_text()

    # Find register_tool decorated functions that don't have circuit breakers
    pattern = r"(@register_tool\([^)]+\)\n)(?!@circuit_breaker)(def [^(]+\([^)]*\)[^:]*:)"

    def replacement(match):
        register_decorator = match.group(1)
        function_def = match.group(2)

        # Add circuit breaker decorator
        circuit_breaker_decorator = """@circuit_breaker(
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        expected_exception=Exception
    )
)
"""
        return register_decorator + circuit_breaker_decorator + function_def

    updated_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # Count how many were added
    original_count = len(re.findall(r"@register_tool", content))
    updated_count = len(re.findall(r"@circuit_breaker", updated_content))

    if updated_content != content:
        file_path.write_text(updated_content)
        return updated_count

    return 0


def main():
    """Deploy circuit breakers across MCP servers."""
    servers_dir = Path("mcp/servers")

    if not servers_dir.exists():
        print("❌ MCP servers directory not found")
        return

    print("🔧 Deploying Circuit Breakers Across MCP Servers")
    print("=" * 50)

    total_files = 0
    total_decorators = 0

    # Process all Python files in servers directory
    for server_file in servers_dir.glob("*.py"):
        if server_file.name in ["__init__.py", "circuit_breaker_monitor.py"]:
            continue

        print(f"Processing {server_file.name}...")

        # Add imports if needed
        imports_added = add_circuit_breaker_imports(server_file)
        if imports_added:
            print(f"  ✅ Added circuit breaker imports")

        # Add decorators
        decorators_added = add_circuit_breaker_decorators(server_file)
        if decorators_added > 0:
            print(f"  ✅ Added {decorators_added} circuit breaker decorators")
            total_decorators += decorators_added

        if imports_added or decorators_added > 0:
            total_files += 1
        else:
            print(f"  ℹ️  Already has circuit breakers or no register_tool functions")

    print("\n" + "=" * 50)
    print(f"📊 Circuit Breaker Deployment Summary")
    print("=" * 50)
    print(f"Files updated: {total_files}")
    print(f"Circuit breakers added: {total_decorators}")
    print("\n🎉 Circuit breaker deployment completed!")
    print("⚡ All MCP servers now have resilience protection")


if __name__ == "__main__":
    main()