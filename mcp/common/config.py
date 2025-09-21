"""
Configuration management for MCP servers.

Provides type-safe configuration loading with validation, environment variable
support, and fallbacks following the 12-factor app methodology.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_type_hints
import json
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_system"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    ssl_mode: str = "prefer"

    @classmethod
    def from_url(cls, url: str) -> 'DatabaseConfig':
        """Create config from database URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)

        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=parsed.path.lstrip('/') if parsed.path else "trading_system",
            user=parsed.username or "postgres",
            password=parsed.password or "",
        )


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    ssl: bool = False
    socket_timeout: float = 30.0
    socket_connect_timeout: float = 30.0
    max_connections: int = 50


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    structured: bool = True


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    bcrypt_rounds: int = 12
    rate_limit_per_minute: int = 60
    cors_origins: List[str] = field(default_factory=list)
    require_https: bool = True


@dataclass
class QuantConnectConfig:
    """QuantConnect API configuration."""
    user_id: str = ""
    api_token: str = ""
    base_url: str = "https://www.quantconnect.com/api/v2"
    timeout_seconds: float = 30.0
    max_retries: int = 3

    def __post_init__(self):
        if not self.user_id or not self.api_token:
            raise ConfigValidationError(
                "QuantConnect USER_ID and API_TOKEN must be configured"
            )


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    enable_tracing: bool = True
    metrics_port: int = 9090
    jaeger_endpoint: Optional[str] = None
    datadog_api_key: Optional[str] = None
    prometheus_endpoint: Optional[str] = None
    health_check_interval: int = 30


@dataclass
class MCPServerConfig:
    """Base MCP server configuration."""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    max_workers: int = 4
    request_timeout: float = 300.0

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    quantconnect: Optional[QuantConnectConfig] = None
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT


class ConfigLoader:
    """Configuration loader with environment variable support."""

    def __init__(self, prefix: str = "MCP"):
        self.prefix = prefix

    def _get_env_var(self, key: str, default: Any = None) -> Any:
        """Get environment variable with prefix."""
        full_key = f"{self.prefix}_{key.upper()}"
        value = os.getenv(full_key, default)

        # Convert string values to appropriate types
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                return value.lower() == 'true'
            elif value.isdigit():
                return int(value)
            elif value.replace('.', '').isdigit():
                return float(value)

        return value

    def _load_from_env(self, config_class: Type[T]) -> T:
        """Load configuration from environment variables."""
        hints = get_type_hints(config_class)
        kwargs = {}

        for field_name, field_type in hints.items():
            env_value = self._get_env_var(field_name)
            if env_value is not None:
                kwargs[field_name] = env_value

        return config_class(**kwargs)

    def _load_from_file(self, file_path: Path, config_class: Type[T]) -> T:
        """Load configuration from JSON file."""
        if not file_path.exists():
            logger.warning(f"Config file {file_path} not found, using defaults")
            return config_class()

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return config_class(**data)
        except (json.JSONDecodeError, TypeError) as e:
            raise ConfigValidationError(f"Invalid config file {file_path}: {e}")

    def load_config(
        self,
        config_class: Type[T] = MCPServerConfig,
        config_file: Optional[Union[str, Path]] = None
    ) -> T:
        """Load configuration with precedence: env vars > config file > defaults."""

        # Start with defaults
        config = config_class()

        # Override with config file if provided
        if config_file:
            file_path = Path(config_file)
            file_config = self._load_from_file(file_path, config_class)

            # Merge configurations (simple field override)
            for field_name in config.__dataclass_fields__:
                file_value = getattr(file_config, field_name, None)
                if file_value is not None:
                    setattr(config, field_name, file_value)

        # Override with environment variables (highest precedence)
        env_config = self._load_from_env(config_class)
        for field_name in config.__dataclass_fields__:
            env_value = getattr(env_config, field_name, None)
            if env_value is not None:
                setattr(config, field_name, env_value)

        # Special handling for nested configs
        if hasattr(config, 'database') and isinstance(config.database, DatabaseConfig):
            database_url = self._get_env_var("DATABASE_URL")
            if database_url:
                config.database = DatabaseConfig.from_url(database_url)

        # Special handling for security config
        if hasattr(config, 'security') and isinstance(config.security, SecurityConfig):
            secret_key = self._get_env_var("SECRET_KEY")
            if secret_key:
                config.security.secret_key = secret_key

        # Environment-specific overrides
        environment = Environment(self._get_env_var("ENVIRONMENT", "development"))
        config.environment = environment

        if environment == Environment.PRODUCTION:
            config.debug = False
            config.logging.level = "WARNING"
            config.security.require_https = True
        elif environment == Environment.DEVELOPMENT:
            config.debug = True
            config.logging.level = "DEBUG"
            config.security.require_https = False

        # Validate critical security config after all loading is complete
        if not config.security.secret_key:
            raise ConfigValidationError("SECRET_KEY must be set for security")

        return config

    def validate_config(self, config: MCPServerConfig) -> None:
        """Validate configuration and raise errors for invalid settings."""
        errors = []

        # Security validations
        if config.is_production:
            if not config.security.secret_key:
                errors.append("SECRET_KEY is required in production")

            if len(config.security.secret_key) < 32:
                errors.append("SECRET_KEY must be at least 32 characters")

            if not config.security.require_https:
                errors.append("HTTPS is required in production")

        # Database validations
        if not config.database.host:
            errors.append("Database host is required")

        if not (1 <= config.database.port <= 65535):
            errors.append("Database port must be between 1 and 65535")

        # QuantConnect validations
        if config.quantconnect:
            if not config.quantconnect.user_id.isdigit():
                errors.append("QuantConnect user_id must be numeric")

            if len(config.quantconnect.api_token) < 32:
                errors.append("QuantConnect api_token appears invalid")

        if errors:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")


# Global configuration instance
_config: Optional[MCPServerConfig] = None
_loader = ConfigLoader()


def get_config() -> MCPServerConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = _loader.load_config()
        _loader.validate_config(_config)
    return _config


def reload_config(config_file: Optional[Union[str, Path]] = None) -> MCPServerConfig:
    """Reload configuration from file and environment."""
    global _config
    _config = _loader.load_config(config_file=config_file)
    _loader.validate_config(_config)
    return _config


def override_config(**kwargs: Any) -> None:
    """Override specific configuration values (for testing)."""
    global _config
    if _config is None:
        _config = get_config()

    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)


# Configuration helpers for specific services
def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_config().database


def get_redis_config() -> RedisConfig:
    """Get Redis configuration."""
    return get_config().redis


def get_security_config() -> SecurityConfig:
    """Get security configuration."""
    return get_config().security


def get_quantconnect_config() -> QuantConnectConfig:
    """Get QuantConnect configuration."""
    config = get_config().quantconnect
    if config is None:
        raise ConfigValidationError("QuantConnect not configured")
    return config


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration."""
    return get_config().monitoring


__all__ = [
    "MCPServerConfig",
    "DatabaseConfig",
    "RedisConfig",
    "LoggingConfig",
    "SecurityConfig",
    "QuantConnectConfig",
    "MonitoringConfig",
    "Environment",
    "ConfigLoader",
    "ConfigValidationError",
    "get_config",
    "reload_config",
    "override_config",
    "get_database_config",
    "get_redis_config",
    "get_security_config",
    "get_quantconnect_config",
    "get_monitoring_config",
]