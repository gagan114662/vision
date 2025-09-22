
import jsonschema
from functools import wraps
from typing import Any, Dict, Callable
import json
from pathlib import Path

class SchemaValidationError(Exception):
    """Raised when schema validation fails"""
    pass

def load_json_schema(schema_path: str) -> Dict[str, Any]:
    """Load JSON schema from file"""
    schema_file = Path(schema_path)
    if not schema_file.exists():
        # Try relative to project root
        schema_file = Path(__file__).parent.parent / schema_path

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    with open(schema_file) as f:
        return json.load(f)

def register_tool_with_validation(name: str, schema: str, response_schema: Optional[str] = None):
    """Enhanced register_tool with runtime schema validation"""

    def decorator(func: Callable) -> Callable:
        # Load schemas at registration time
        input_schema = load_json_schema(schema)
        output_schema = load_json_schema(response_schema) if response_schema else None

        # Create validators
        input_validator = jsonschema.Draft7Validator(input_schema)
        output_validator = jsonschema.Draft7Validator(output_schema) if output_schema else None

        @wraps(func)
        async def async_wrapper(params: Dict[str, Any]) -> Dict[str, Any]:
            # Validate input
            errors = list(input_validator.iter_errors(params))
            if errors:
                error_messages = [f"{'.'.join(str(p) for p in err.path)}: {err.message}" for err in errors]
                raise SchemaValidationError(f"Input validation failed: {'; '.join(error_messages)}")

            # Call the actual function
            if asyncio.iscoroutinefunction(func):
                result = await func(params)
            else:
                result = func(params)

            # Validate output if schema provided
            if output_validator:
                errors = list(output_validator.iter_errors(result))
                if errors:
                    error_messages = [f"{'.'.join(str(p) for p in err.path)}: {err.message}" for err in errors]
                    raise SchemaValidationError(f"Output validation failed: {'; '.join(error_messages)}")

            return result

        @wraps(func)
        def sync_wrapper(params: Dict[str, Any]) -> Dict[str, Any]:
            # Validate input
            errors = list(input_validator.iter_errors(params))
            if errors:
                error_messages = [f"{'.'.join(str(p) for p in err.path)}: {err.message}" for err in errors]
                raise SchemaValidationError(f"Input validation failed: {'; '.join(error_messages)}")

            # Call the actual function
            result = func(params)

            # Validate output if schema provided
            if output_validator:
                errors = list(output_validator.iter_errors(result))
                if errors:
                    error_messages = [f"{'.'.join(str(p) for p in err.path)}: {err.message}" for err in errors]
                    raise SchemaValidationError(f"Output validation failed: {'; '.join(error_messages)}")

            return result

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
