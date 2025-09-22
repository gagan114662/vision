
"""
MCP server functionality for tool registration and execution with schema validation.
"""

from typing import Dict, Any, Callable, Optional
import json
import asyncio
from pathlib import Path
import logging
from functools import wraps

logger = logging.getLogger(__name__)

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    logger.warning("jsonschema not available - schema validation disabled")

def load_json_schema(schema_path: str) -> Dict[str, Any]:
    """Load JSON schema from file"""
    if not schema_path:
        return {}

    schema_file = Path(schema_path)
    if not schema_file.exists():
        # Try relative to project root
        schema_file = Path(__file__).parent.parent / schema_path

    if not schema_file.exists():
        logger.warning(f"Schema not found: {schema_path}")
        return {}

    try:
        with open(schema_file) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load schema {schema_path}: {e}")
        return {}

def validate_with_schema(data: Any, schema: Dict[str, Any], schema_name: str = "data"):
    """Validate data against JSON schema"""
    if not JSONSCHEMA_AVAILABLE or not schema:
        return  # Skip validation if jsonschema not available

    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        from mcp.common.exceptions import SchemaValidationError
        error_path = '.'.join(str(p) for p in e.path) if e.path else 'root'
        raise SchemaValidationError(
            f"{schema_name} validation failed at {error_path}: {e.message}"
        )
    except Exception as e:
        logger.error(f"Schema validation error: {e}")

def register_tool(name=None, schema=None, response_schema=None):
    """Register a tool with runtime schema validation."""
    def decorator(func):
        # Load schemas at registration time
        input_schema = load_json_schema(schema) if schema else {}
        output_schema = load_json_schema(response_schema) if response_schema else {}

        # Store metadata
        func._mcp_tool_name = name
        func._mcp_schema = schema
        func._mcp_response_schema = response_schema

        @wraps(func)
        async def async_wrapper(params: Dict[str, Any]) -> Dict[str, Any]:
            # Validate input
            if input_schema:
                validate_with_schema(params, input_schema, f"Tool '{name}' input")

            # Call the actual function
            result = await func(params)

            # Validate output
            if output_schema:
                validate_with_schema(result, output_schema, f"Tool '{name}' output")

            return result

        @wraps(func)
        def sync_wrapper(params: Dict[str, Any]) -> Dict[str, Any]:
            # Validate input
            if input_schema:
                validate_with_schema(params, input_schema, f"Tool '{name}' input")

            # Call the actual function
            result = func(params)

            # Validate output
            if output_schema:
                validate_with_schema(result, output_schema, f"Tool '{name}' output")

            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            logger.info(f"Registered async tool: {name} with schema validation")
            return async_wrapper
        else:
            logger.info(f"Registered sync tool: {name} with schema validation")
            return sync_wrapper

    return decorator
