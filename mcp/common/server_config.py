"""
Server-specific configuration management for MCP servers.

This module provides configuration management specifically tailored for
individual MCP servers, including server-specific settings, tool parameters,
and runtime configuration.
"""
from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar
from enum import Enum

from .config import get_config, MCPServerConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServerType(Enum):
    """Types of MCP servers."""
    SIGNAL_PROCESSING = "signal_processing"
    RISK_MANAGEMENT = "risk_management"
    COMPLIANCE = "compliance"
    DATA_PROVIDER = "data_provider"
    EXECUTION = "execution"
    RESEARCH = "research"
    VISUALIZATION = "visualization"
    UTILITY = "utility"


@dataclass
class ToolConfig:
    """Configuration for individual MCP tools."""
    tool_name: str
    enabled: bool = True
    timeout_seconds: float = 30.0
    max_retries: int = 3
    rate_limit_per_minute: int = 60
    parameters: Dict[str, Any] = field(default_factory=dict)
    auth_required: bool = False
    logging_enabled: bool = True

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get tool parameter with fallback."""
        return self.parameters.get(key, default)

    def set_parameter(self, key: str, value: Any) -> None:
        """Set tool parameter."""
        self.parameters[key] = value


@dataclass
class ServerSpecificConfig:
    """Server-specific configuration."""
    server_name: str
    server_type: ServerType
    enabled: bool = True
    max_concurrent_requests: int = 10
    request_timeout: float = 300.0
    health_check_enabled: bool = True
    metrics_enabled: bool = True

    # Tool configurations
    tools: Dict[str, ToolConfig] = field(default_factory=dict)

    # Server-specific settings
    settings: Dict[str, Any] = field(default_factory=dict)

    def get_tool_config(self, tool_name: str) -> Optional[ToolConfig]:
        """Get configuration for a specific tool."""
        return self.tools.get(tool_name)

    def add_tool_config(self, tool_config: ToolConfig) -> None:
        """Add tool configuration."""
        self.tools[tool_config.tool_name] = tool_config

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get server setting with fallback."""
        return self.settings.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """Set server setting."""
        self.settings[key] = value


class ServerConfigManager:
    """Manages configuration for all MCP servers."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config/servers")

        # Try to get global config, use minimal config for demo if it fails
        try:
            self.global_config = get_config()
        except Exception as e:
            logger.warning(f"Failed to load global config: {e}, using minimal config for demo")
            # Set minimal config for demo purposes
            import os
            os.environ["MCP_SECRET_KEY"] = "demo_key_32_characters_minimum_req"
            os.environ["MCP_ENVIRONMENT"] = "development"
            # Clear the cached config and reload
            from .config import reload_config
            self.global_config = reload_config()

        self.server_configs: Dict[str, ServerSpecificConfig] = {}

        # Load all server configurations
        self._load_all_server_configs()

    def _load_all_server_configs(self) -> None:
        """Load configurations for all servers."""
        # Default configurations for known servers
        default_configs = self._get_default_server_configs()

        # Load from config files if they exist
        if self.config_dir.exists():
            for config_file in self.config_dir.glob("*.json"):
                try:
                    server_name = config_file.stem
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)

                    server_config = self._create_server_config_from_dict(server_name, config_data)
                    self.server_configs[server_name] = server_config
                    logger.info(f"Loaded config for server: {server_name}")

                except Exception as e:
                    logger.error(f"Failed to load config for {config_file}: {e}")

        # Apply defaults for servers without config files
        for server_name, default_config in default_configs.items():
            if server_name not in self.server_configs:
                self.server_configs[server_name] = default_config

        # Apply environment variable overrides
        self._apply_environment_overrides()

    def _get_default_server_configs(self) -> Dict[str, ServerSpecificConfig]:
        """Get default configurations for all known servers."""
        defaults = {}

        # Signal processing servers
        defaults["signal_fourier_server"] = ServerSpecificConfig(
            server_name="signal_fourier_server",
            server_type=ServerType.SIGNAL_PROCESSING,
            tools={
                "signal.fourier.detect_cycles": ToolConfig(
                    tool_name="signal.fourier.detect_cycles",
                    timeout_seconds=60.0,
                    parameters={
                        "window_size": 252,
                        "min_frequency": 0.1,
                        "max_frequency": 0.5,
                        "threshold": 0.1
                    }
                )
            },
            settings={
                "fft_implementation": "numpy",
                "enable_windowing": True,
                "default_sampling_rate": 1.0
            }
        )

        defaults["signal_wavelet_server"] = ServerSpecificConfig(
            server_name="signal_wavelet_server",
            server_type=ServerType.SIGNAL_PROCESSING,
            tools={
                "signal.wavelet.multiscale_decomposition": ToolConfig(
                    tool_name="signal.wavelet.multiscale_decomposition",
                    timeout_seconds=45.0,
                    parameters={
                        "wavelet_type": "haar",
                        "levels": 4,
                        "threshold": 0.01
                    }
                )
            },
            settings={
                "wavelet_library": "pywt",
                "enable_denoising": True
            }
        )

        # Risk management servers
        defaults["risk_server"] = ServerSpecificConfig(
            server_name="risk_server",
            server_type=ServerType.RISK_MANAGEMENT,
            tools={
                "risk.limits.evaluate_portfolio": ToolConfig(
                    tool_name="risk.limits.evaluate_portfolio",
                    timeout_seconds=30.0,
                    auth_required=True,
                    parameters={
                        "confidence_level": 0.95,
                        "lookback_days": 252,
                        "rebalance_threshold": 0.05
                    }
                )
            },
            settings={
                "var_method": "historical_simulation",
                "enable_stress_testing": True,
                "alert_thresholds": {
                    "var_breach": True,
                    "concentration_risk": True,
                    "leverage_limit": True
                }
            }
        )

        # Compliance servers
        defaults["compliance_server"] = ServerSpecificConfig(
            server_name="compliance_server",
            server_type=ServerType.COMPLIANCE,
            tools={
                "compliance.generate_summary": ToolConfig(
                    tool_name="compliance.generate_summary",
                    timeout_seconds=60.0,
                    auth_required=True,
                    parameters={
                        "regulations": ["MiFID_II", "SEC", "CFTC"],
                        "audit_level": "comprehensive"
                    }
                )
            },
            settings={
                "enable_real_time_monitoring": True,
                "regulatory_frameworks": ["US", "EU", "UK"],
                "alert_severity_threshold": "medium"
            }
        )

        # Research servers
        defaults["research_feed_server"] = ServerSpecificConfig(
            server_name="research_feed_server",
            server_type=ServerType.RESEARCH,
            tools={
                "research.feed.list_insights": ToolConfig(
                    tool_name="research.feed.list_insights",
                    timeout_seconds=30.0,
                    rate_limit_per_minute=120,
                    parameters={
                        "max_results": 100,
                        "relevance_threshold": 0.7,
                        "time_horizon_days": 30
                    }
                )
            },
            settings={
                "data_sources": ["internal", "external_apis"],
                "enable_sentiment_analysis": True,
                "cache_ttl_hours": 1
            }
        )

        # Execution servers
        defaults["quantconnect_stub"] = ServerSpecificConfig(
            server_name="quantconnect_stub",
            server_type=ServerType.EXECUTION,
            tools={
                "quantconnect.backtest.run": ToolConfig(
                    tool_name="quantconnect.backtest.run",
                    timeout_seconds=300.0,
                    auth_required=True,
                    parameters={
                        "default_start_date": "2020-01-01",
                        "default_end_date": "2023-12-31",
                        "default_cash": 100000
                    }
                ),
                "quantconnect.project.sync": ToolConfig(
                    tool_name="quantconnect.project.sync",
                    timeout_seconds=60.0,
                    auth_required=True
                )
            },
            settings={
                "api_base_url": "https://www.quantconnect.com/api/v2",
                "enable_live_trading": False,
                "default_algorithm_language": "Python"
            }
        )

        # Utility servers
        defaults["ally_shell_server"] = ServerSpecificConfig(
            server_name="ally_shell_server",
            server_type=ServerType.UTILITY,
            tools={
                "ops.shell.run_command": ToolConfig(
                    tool_name="ops.shell.run_command",
                    timeout_seconds=60.0,
                    auth_required=True,
                    parameters={
                        "allowed_commands": ["git", "lean", "python", "pip", "ls", "pwd", "find"],
                        "workspace_root": ".",
                        "enable_dry_run": True
                    }
                )
            },
            settings={
                "executor_type": "subprocess",
                "enable_provenance_logging": True,
                "security_level": "high"
            }
        )

        # Add more default configs for other servers
        for server_name in [
            "semtools_server", "feature_engineering_server", "mean_reversion_ou_server",
            "regime_hmm_server", "robustness_server", "signal_filter_server",
            "chart_server", "provenance_server", "autonomous_recovery_server"
        ]:
            if server_name not in defaults:
                defaults[server_name] = ServerSpecificConfig(
                    server_name=server_name,
                    server_type=self._infer_server_type(server_name)
                )

        return defaults

    def _infer_server_type(self, server_name: str) -> ServerType:
        """Infer server type from server name."""
        if "signal" in server_name or "fourier" in server_name or "wavelet" in server_name:
            return ServerType.SIGNAL_PROCESSING
        elif "risk" in server_name:
            return ServerType.RISK_MANAGEMENT
        elif "compliance" in server_name:
            return ServerType.COMPLIANCE
        elif "research" in server_name or "feed" in server_name:
            return ServerType.RESEARCH
        elif "chart" in server_name or "visualization" in server_name:
            return ServerType.VISUALIZATION
        elif "quantconnect" in server_name or "execution" in server_name:
            return ServerType.EXECUTION
        else:
            return ServerType.UTILITY

    def _create_server_config_from_dict(self, server_name: str, config_data: Dict[str, Any]) -> ServerSpecificConfig:
        """Create server config from dictionary data."""
        # Extract tools configuration
        tools = {}
        for tool_name, tool_data in config_data.get("tools", {}).items():
            tools[tool_name] = ToolConfig(
                tool_name=tool_name,
                **tool_data
            )

        return ServerSpecificConfig(
            server_name=server_name,
            server_type=ServerType(config_data.get("server_type", "utility")),
            enabled=config_data.get("enabled", True),
            max_concurrent_requests=config_data.get("max_concurrent_requests", 10),
            request_timeout=config_data.get("request_timeout", 300.0),
            health_check_enabled=config_data.get("health_check_enabled", True),
            metrics_enabled=config_data.get("metrics_enabled", True),
            tools=tools,
            settings=config_data.get("settings", {})
        )

    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to server configurations."""
        for server_name, server_config in self.server_configs.items():
            # Check for server-specific environment variables
            env_prefix = f"MCP_{server_name.upper()}"

            # Override enabled status
            enabled_key = f"{env_prefix}_ENABLED"
            if enabled_key in os.environ:
                server_config.enabled = os.environ[enabled_key].lower() == "true"

            # Override timeout
            timeout_key = f"{env_prefix}_TIMEOUT"
            if timeout_key in os.environ:
                try:
                    server_config.request_timeout = float(os.environ[timeout_key])
                except ValueError:
                    logger.warning(f"Invalid timeout value for {timeout_key}")

            # Override max concurrent requests
            concurrent_key = f"{env_prefix}_MAX_CONCURRENT"
            if concurrent_key in os.environ:
                try:
                    server_config.max_concurrent_requests = int(os.environ[concurrent_key])
                except ValueError:
                    logger.warning(f"Invalid max_concurrent value for {concurrent_key}")

    def get_server_config(self, server_name: str) -> Optional[ServerSpecificConfig]:
        """Get configuration for a specific server."""
        return self.server_configs.get(server_name)

    def get_tool_config(self, server_name: str, tool_name: str) -> Optional[ToolConfig]:
        """Get configuration for a specific tool."""
        server_config = self.get_server_config(server_name)
        if server_config:
            return server_config.get_tool_config(tool_name)
        return None

    def is_server_enabled(self, server_name: str) -> bool:
        """Check if a server is enabled."""
        server_config = self.get_server_config(server_name)
        return server_config.enabled if server_config else False

    def is_tool_enabled(self, server_name: str, tool_name: str) -> bool:
        """Check if a specific tool is enabled."""
        tool_config = self.get_tool_config(server_name, tool_name)
        return tool_config.enabled if tool_config else False

    def get_servers_by_type(self, server_type: ServerType) -> List[ServerSpecificConfig]:
        """Get all servers of a specific type."""
        return [
            config for config in self.server_configs.values()
            if config.server_type == server_type and config.enabled
        ]

    def save_server_config(self, server_name: str, config: ServerSpecificConfig) -> None:
        """Save server configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        config_file = self.config_dir / f"{server_name}.json"

        # Convert to dictionary for JSON serialization
        config_dict = {
            "server_type": config.server_type.value,
            "enabled": config.enabled,
            "max_concurrent_requests": config.max_concurrent_requests,
            "request_timeout": config.request_timeout,
            "health_check_enabled": config.health_check_enabled,
            "metrics_enabled": config.metrics_enabled,
            "tools": {
                tool_name: {
                    "enabled": tool.enabled,
                    "timeout_seconds": tool.timeout_seconds,
                    "max_retries": tool.max_retries,
                    "rate_limit_per_minute": tool.rate_limit_per_minute,
                    "parameters": tool.parameters,
                    "auth_required": tool.auth_required,
                    "logging_enabled": tool.logging_enabled
                }
                for tool_name, tool in config.tools.items()
            },
            "settings": config.settings
        }

        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Update in-memory configuration
        self.server_configs[server_name] = config
        logger.info(f"Saved configuration for server: {server_name}")

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of all server configurations."""
        return {
            "total_servers": len(self.server_configs),
            "enabled_servers": sum(1 for config in self.server_configs.values() if config.enabled),
            "servers_by_type": {
                server_type.value: len(self.get_servers_by_type(server_type))
                for server_type in ServerType
            },
            "servers": {
                name: {
                    "type": config.server_type.value,
                    "enabled": config.enabled,
                    "tools_count": len(config.tools),
                    "settings_count": len(config.settings)
                }
                for name, config in self.server_configs.items()
            }
        }


# Global server config manager instance
_server_config_manager: Optional[ServerConfigManager] = None


def get_server_config_manager() -> ServerConfigManager:
    """Get the global server configuration manager."""
    global _server_config_manager
    if _server_config_manager is None:
        _server_config_manager = ServerConfigManager()
    return _server_config_manager


def get_server_config(server_name: str) -> Optional[ServerSpecificConfig]:
    """Get configuration for a specific server."""
    return get_server_config_manager().get_server_config(server_name)


def get_tool_config(server_name: str, tool_name: str) -> Optional[ToolConfig]:
    """Get configuration for a specific tool."""
    return get_server_config_manager().get_tool_config(server_name, tool_name)


__all__ = [
    "ServerType",
    "ToolConfig",
    "ServerSpecificConfig",
    "ServerConfigManager",
    "get_server_config_manager",
    "get_server_config",
    "get_tool_config",
]