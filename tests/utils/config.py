"""Test configuration utilities."""
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from mcp.common.config import SecurityConfig, DatabaseConfig, QuantConnectConfig


def setup_test_environment() -> None:
    """Set up test environment with required configuration."""
    # Set a valid test secret key (32+ characters)
    os.environ["MCP_SECRET_KEY"] = "test_secret_key_32_characters_minimum_required"
    os.environ["MCP_ENVIRONMENT"] = "test"

    # Set other required environment variables for testing
    os.environ["MCP_DATABASE_URL"] = "sqlite:///test.db"
    os.environ["QUANTCONNECT_USER_ID"] = "test_user_123"
    os.environ["QUANTCONNECT_API_TOKEN"] = "test_token_123"


def create_test_config() -> Dict[str, Any]:
    """Create a valid test configuration."""
    return {
        "security": {
            "secret_key": "test_secret_key_32_characters_minimum_required",
            "jwt_algorithm": "HS256",
            "jwt_expiration_hours": 1,
            "bcrypt_rounds": 4,  # Lower for tests
            "rate_limit_per_minute": 1000,
            "cors_origins": ["http://localhost:3000"],
            "require_https": False
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "database": "test_trading_system",
            "user": "test_user",
            "password": "test_pass",
            "pool_size": 2,
            "max_overflow": 5,
            "pool_timeout": 5
        },
        "quantconnect": {
            "user_id": "test_user_123",
            "api_token": "test_token_123",
            "api_url": "https://api.quantconnect.com",
            "organization_id": "test_org",
            "rate_limit_delay": 0.1
        },
        "servers": {},
        "tools": {}
    }


def create_test_config_file() -> str:
    """Create a temporary test configuration file."""
    import json

    config_data = create_test_config()

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f, indent=2)
        return f.name


def cleanup_test_artifacts() -> None:
    """Clean up test artifacts like databases, keys, audit files."""
    artifacts = [
        "test.db",
        "provenance.db",
        "audit_trail.jsonl",
        "demo_audit_trail.jsonl"
    ]

    for artifact in artifacts:
        try:
            if os.path.exists(artifact):
                os.remove(artifact)
        except OSError:
            pass

    # Clean up keys directory
    keys_dir = Path("keys")
    if keys_dir.exists():
        import shutil
        shutil.rmtree(keys_dir, ignore_errors=True)


class TestConfigMixin:
    """Mixin class for test configuration setup."""

    @classmethod
    def setUpClass(cls):
        """Set up test configuration for the entire test class."""
        setup_test_environment()

    @classmethod
    def tearDownClass(cls):
        """Clean up test artifacts."""
        cleanup_test_artifacts()

    def setUp(self):
        """Set up individual test."""
        # Ensure environment is still configured
        setup_test_environment()