"""
Common exceptions for MCP services
"""

class MCPError(Exception):
    """Base exception for MCP services"""
    pass

class DataUnavailableError(MCPError):
    """Raised when real data is not available and no fallback is configured"""
    pass

class SchemaValidationError(MCPError):
    """Raised when schema validation fails"""
    pass

class CircuitBreakerError(MCPError):
    """Raised when circuit breaker is open"""
    pass

class ConfigurationError(MCPError):
    """Raised when service configuration is invalid"""
    pass