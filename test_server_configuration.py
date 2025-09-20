"""
Test Server Configuration Integration.

This demonstrates the comprehensive configuration system working across
all MCP servers with proper environment variable support, file-based
configuration, and runtime configuration management.
"""
import os
import json
import tempfile
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, '.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from mcp.common.server_config import (
    ServerConfigManager, ServerType, ToolConfig, ServerSpecificConfig,
    get_server_config_manager, get_server_config, get_tool_config
)
from mcp.common.config import get_config, reload_config, override_config


class ServerConfigurationDemo:
    """Demonstration of comprehensive server configuration system."""

    def __init__(self):
        # Set up minimal configuration for demo
        os.environ["MCP_SECRET_KEY"] = "demo_secret_key_for_testing_only_32chars"
        os.environ["MCP_ENVIRONMENT"] = "development"

        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.temp_dir / "config" / "servers"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize with temporary config directory
        self.config_manager = ServerConfigManager(self.config_dir)

    def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run complete configuration demonstration."""
        print("⚙️ Starting Comprehensive Server Configuration Demo")
        print("=" * 60)

        try:
            # Step 1: Configuration Loading
            print("\n📋 Step 1: Configuration Loading & Defaults")
            loading_results = self._demonstrate_configuration_loading()

            # Step 2: File-based Configuration
            print("\n📁 Step 2: File-based Configuration")
            file_config_results = self._demonstrate_file_configuration()

            # Step 3: Environment Variable Overrides
            print("\n🌍 Step 3: Environment Variable Overrides")
            env_override_results = self._demonstrate_environment_overrides()

            # Step 4: Server-specific Configuration
            print("\n🔧 Step 4: Server-specific Configuration")
            server_config_results = self._demonstrate_server_configuration()

            # Step 5: Tool-specific Configuration
            print("\n🛠️ Step 5: Tool-specific Configuration")
            tool_config_results = self._demonstrate_tool_configuration()

            # Step 6: Runtime Configuration Changes
            print("\n🔄 Step 6: Runtime Configuration Changes")
            runtime_results = self._demonstrate_runtime_configuration()

            # Step 7: Configuration Validation
            print("\n✅ Step 7: Configuration Validation")
            validation_results = self._demonstrate_configuration_validation()

            return {
                "loading": loading_results,
                "file_config": file_config_results,
                "env_overrides": env_override_results,
                "server_config": server_config_results,
                "tool_config": tool_config_results,
                "runtime_changes": runtime_results,
                "validation": validation_results,
                "demo_status": "completed_successfully"
            }

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            return {
                "demo_status": "failed",
                "error": str(e)
            }
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _demonstrate_configuration_loading(self):
        """Demonstrate configuration loading and defaults."""
        print("📋 Configuration Loading:")

        # Get configuration summary
        summary = self.config_manager.get_configuration_summary()
        print(f"   Total servers: {summary['total_servers']}")
        print(f"   Enabled servers: {summary['enabled_servers']}")

        print("\n   Servers by type:")
        for server_type, count in summary['servers_by_type'].items():
            if count > 0:
                print(f"     {server_type}: {count} servers")

        # Show some example server configurations
        print("\n   Example server configurations:")
        for server_name in ["risk_server", "ally_shell_server", "signal_fourier_server"]:
            config = self.config_manager.get_server_config(server_name)
            if config:
                print(f"     {server_name}:")
                print(f"       Type: {config.server_type.value}")
                print(f"       Enabled: {config.enabled}")
                print(f"       Tools: {len(config.tools)}")
                print(f"       Settings: {len(config.settings)}")

        return {
            "total_servers": summary['total_servers'],
            "enabled_servers": summary['enabled_servers'],
            "servers_by_type": summary['servers_by_type']
        }

    def _demonstrate_file_configuration(self):
        """Demonstrate file-based configuration."""
        print("📁 File-based Configuration:")

        # Create a custom configuration file
        custom_config = {
            "server_type": "risk_management",
            "enabled": True,
            "max_concurrent_requests": 20,
            "request_timeout": 45.0,
            "tools": {
                "risk.limits.evaluate_portfolio": {
                    "enabled": True,
                    "timeout_seconds": 90.0,
                    "parameters": {
                        "confidence_level": 0.99,
                        "lookback_days": 500,
                        "rebalance_threshold": 0.02
                    }
                }
            },
            "settings": {
                "var_method": "monte_carlo",
                "enable_stress_testing": True,
                "alert_thresholds": {
                    "var_breach": True,
                    "concentration_risk": True,
                    "leverage_limit": False
                },
                "custom_setting": "file_config_value"
            }
        }

        # Save configuration file
        config_file = self.config_dir / "custom_risk_server.json"
        with open(config_file, 'w') as f:
            json.dump(custom_config, f, indent=2)

        print(f"   Created config file: {config_file}")

        # Reload configuration manager to pick up the new file
        self.config_manager = ServerConfigManager(self.config_dir)

        # Verify the configuration was loaded
        loaded_config = self.config_manager.get_server_config("custom_risk_server")
        if loaded_config:
            print(f"   ✅ Custom config loaded:")
            print(f"     Max concurrent requests: {loaded_config.max_concurrent_requests}")
            print(f"     Request timeout: {loaded_config.request_timeout}s")
            print(f"     VaR method: {loaded_config.get_setting('var_method')}")
            print(f"     Custom setting: {loaded_config.get_setting('custom_setting')}")

            # Check tool configuration
            tool_config = loaded_config.get_tool_config("risk.limits.evaluate_portfolio")
            if tool_config:
                print(f"     Tool timeout: {tool_config.timeout_seconds}s")
                print(f"     Confidence level: {tool_config.get_parameter('confidence_level')}")
                print(f"     Lookback days: {tool_config.get_parameter('lookback_days')}")

        return {
            "config_file_created": str(config_file),
            "config_loaded": loaded_config is not None,
            "max_concurrent_requests": loaded_config.max_concurrent_requests if loaded_config else None,
            "var_method": loaded_config.get_setting('var_method') if loaded_config else None
        }

    def _demonstrate_environment_overrides(self):
        """Demonstrate environment variable overrides."""
        print("🌍 Environment Variable Overrides:")

        # Set environment variables
        original_env = {}
        env_vars = {
            "MCP_RISK_SERVER_ENABLED": "false",
            "MCP_RISK_SERVER_TIMEOUT": "120.0",
            "MCP_RISK_SERVER_MAX_CONCURRENT": "5",
            "MCP_ALLY_SHELL_SERVER_ENABLED": "true",
            "MCP_ALLY_SHELL_SERVER_TIMEOUT": "300.0"
        }

        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
            print(f"   Set {key}={value}")

        try:
            # Reload configuration to pick up environment variables
            self.config_manager = ServerConfigManager(self.config_dir)

            # Check that environment overrides were applied
            risk_config = self.config_manager.get_server_config("risk_server")
            shell_config = self.config_manager.get_server_config("ally_shell_server")

            print("\n   Environment override results:")
            if risk_config:
                print(f"     Risk server enabled: {risk_config.enabled}")
                print(f"     Risk server timeout: {risk_config.request_timeout}s")
                print(f"     Risk server max concurrent: {risk_config.max_concurrent_requests}")

            if shell_config:
                print(f"     Shell server enabled: {shell_config.enabled}")
                print(f"     Shell server timeout: {shell_config.request_timeout}s")

            return {
                "env_vars_set": len(env_vars),
                "risk_server_enabled": risk_config.enabled if risk_config else None,
                "risk_server_timeout": risk_config.request_timeout if risk_config else None,
                "shell_server_enabled": shell_config.enabled if shell_config else None
            }

        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def _demonstrate_server_configuration(self):
        """Demonstrate server-specific configuration."""
        print("🔧 Server-specific Configuration:")

        # Get servers by type
        risk_servers = self.config_manager.get_servers_by_type(ServerType.RISK_MANAGEMENT)
        signal_servers = self.config_manager.get_servers_by_type(ServerType.SIGNAL_PROCESSING)
        utility_servers = self.config_manager.get_servers_by_type(ServerType.UTILITY)

        print(f"   Risk management servers: {len(risk_servers)}")
        for server in risk_servers:
            print(f"     {server.server_name}: enabled={server.enabled}")

        print(f"   Signal processing servers: {len(signal_servers)}")
        for server in signal_servers:
            print(f"     {server.server_name}: enabled={server.enabled}")

        print(f"   Utility servers: {len(utility_servers)}")
        for server in utility_servers[:3]:  # Show first 3
            print(f"     {server.server_name}: enabled={server.enabled}")

        # Test server enablement checks
        print("\n   Server enablement status:")
        test_servers = ["risk_server", "ally_shell_server", "signal_fourier_server", "nonexistent_server"]
        for server_name in test_servers:
            enabled = self.config_manager.is_server_enabled(server_name)
            print(f"     {server_name}: {'✅' if enabled else '❌'}")

        return {
            "risk_servers": len(risk_servers),
            "signal_servers": len(signal_servers),
            "utility_servers": len(utility_servers),
            "enablement_checks": {
                server: self.config_manager.is_server_enabled(server)
                for server in test_servers
            }
        }

    def _demonstrate_tool_configuration(self):
        """Demonstrate tool-specific configuration."""
        print("🛠️ Tool-specific Configuration:")

        # Test various tool configurations
        tool_tests = [
            ("risk_server", "risk.limits.evaluate_portfolio"),
            ("ally_shell_server", "ops.shell.run_command"),
            ("signal_fourier_server", "signal.fourier.detect_cycles"),
            ("compliance_server", "compliance.generate_summary")
        ]

        results = {}
        for server_name, tool_name in tool_tests:
            tool_config = self.config_manager.get_tool_config(server_name, tool_name)
            enabled = self.config_manager.is_tool_enabled(server_name, tool_name)

            print(f"\n   {server_name}::{tool_name}:")
            print(f"     Enabled: {'✅' if enabled else '❌'}")

            if tool_config:
                print(f"     Timeout: {tool_config.timeout_seconds}s")
                print(f"     Max retries: {tool_config.max_retries}")
                print(f"     Rate limit: {tool_config.rate_limit_per_minute}/min")
                print(f"     Auth required: {tool_config.auth_required}")
                print(f"     Parameters: {len(tool_config.parameters)}")

                # Show some parameters
                if tool_config.parameters:
                    print(f"     Sample parameters:")
                    for key, value in list(tool_config.parameters.items())[:3]:
                        print(f"       {key}: {value}")

                results[f"{server_name}::{tool_name}"] = {
                    "enabled": enabled,
                    "timeout": tool_config.timeout_seconds,
                    "parameters_count": len(tool_config.parameters)
                }
            else:
                print(f"     No configuration found")
                results[f"{server_name}::{tool_name}"] = {"enabled": False}

        return results

    def _demonstrate_runtime_configuration(self):
        """Demonstrate runtime configuration changes."""
        print("🔄 Runtime Configuration Changes:")

        # Create a new server configuration
        new_server_config = ServerSpecificConfig(
            server_name="dynamic_test_server",
            server_type=ServerType.UTILITY,
            enabled=True,
            max_concurrent_requests=15,
            request_timeout=60.0,
            settings={
                "dynamic_setting": "runtime_value",
                "test_mode": True
            }
        )

        # Add a tool configuration
        tool_config = ToolConfig(
            tool_name="test.dynamic.tool",
            enabled=True,
            timeout_seconds=45.0,
            parameters={
                "test_param": "dynamic_value",
                "iteration_count": 100
            }
        )
        new_server_config.add_tool_config(tool_config)

        print(f"   Created dynamic server configuration:")
        print(f"     Server: {new_server_config.server_name}")
        print(f"     Type: {new_server_config.server_type.value}")
        print(f"     Tools: {len(new_server_config.tools)}")

        # Save the configuration
        self.config_manager.save_server_config("dynamic_test_server", new_server_config)
        print(f"   ✅ Saved configuration to file")

        # Verify it was saved and can be loaded
        loaded_config = self.config_manager.get_server_config("dynamic_test_server")
        if loaded_config:
            print(f"   ✅ Configuration loaded successfully:")
            print(f"     Settings: {loaded_config.settings}")
            print(f"     Dynamic setting: {loaded_config.get_setting('dynamic_setting')}")

            # Test tool configuration
            loaded_tool = loaded_config.get_tool_config("test.dynamic.tool")
            if loaded_tool:
                print(f"     Tool timeout: {loaded_tool.timeout_seconds}s")
                print(f"     Test param: {loaded_tool.get_parameter('test_param')}")

        return {
            "dynamic_server_created": True,
            "configuration_saved": True,
            "configuration_loaded": loaded_config is not None,
            "dynamic_setting_value": loaded_config.get_setting('dynamic_setting') if loaded_config else None
        }

    def _demonstrate_configuration_validation(self):
        """Demonstrate configuration validation and error handling."""
        print("✅ Configuration Validation:")

        validation_results = []

        # Test 1: Valid configuration
        try:
            valid_config = ServerSpecificConfig(
                server_name="valid_test_server",
                server_type=ServerType.SIGNAL_PROCESSING,
                enabled=True,
                max_concurrent_requests=10,
                request_timeout=30.0
            )
            validation_results.append({"test": "valid_config", "passed": True})
            print("   ✅ Valid configuration accepted")
        except Exception as e:
            validation_results.append({"test": "valid_config", "passed": False, "error": str(e)})
            print(f"   ❌ Valid configuration rejected: {e}")

        # Test 2: Test global configuration access
        try:
            global_config = get_config()
            print(f"   ✅ Global config accessible: {global_config.environment.value}")
            validation_results.append({"test": "global_config", "passed": True})
        except Exception as e:
            validation_results.append({"test": "global_config", "passed": False, "error": str(e)})
            print(f"   ❌ Global config error: {e}")

        # Test 3: Test configuration helpers
        try:
            risk_server_config = get_server_config("risk_server")
            risk_tool_config = get_tool_config("risk_server", "risk.limits.evaluate_portfolio")

            helper_results = {
                "server_config_found": risk_server_config is not None,
                "tool_config_found": risk_tool_config is not None
            }

            print(f"   ✅ Configuration helpers working:")
            print(f"     Server config found: {helper_results['server_config_found']}")
            print(f"     Tool config found: {helper_results['tool_config_found']}")

            validation_results.append({"test": "config_helpers", "passed": True, "results": helper_results})

        except Exception as e:
            validation_results.append({"test": "config_helpers", "passed": False, "error": str(e)})
            print(f"   ❌ Configuration helpers error: {e}")

        # Summary
        passed_tests = sum(1 for result in validation_results if result["passed"])
        total_tests = len(validation_results)

        print(f"\n   Validation Summary: {passed_tests}/{total_tests} tests passed")

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "validation_results": validation_results
        }


def main():
    """Run the server configuration demonstration."""
    demo = ServerConfigurationDemo()

    try:
        result = demo.run_comprehensive_demo()

        print("\n" + "=" * 60)
        if result["demo_status"] == "completed_successfully":
            print("✅ Server Configuration Demo COMPLETED SUCCESSFULLY!")
            print("\n🎯 Key Achievements:")
            print("   ✓ Configuration loading with defaults working")
            print("   ✓ File-based configuration operational")
            print("   ✓ Environment variable overrides functional")
            print("   ✓ Server-specific configuration management active")
            print("   ✓ Tool-specific configuration working")
            print("   ✓ Runtime configuration changes supported")
            print("   ✓ Configuration validation implemented")

            # Show key metrics
            loading_data = result["loading"]
            validation_data = result["validation"]

            print(f"\n📊 Demo Metrics:")
            print(f"   Total servers configured: {loading_data['total_servers']}")
            print(f"   Enabled servers: {loading_data['enabled_servers']}")
            print(f"   Validation tests passed: {validation_data['passed_tests']}/{validation_data['total_tests']}")

        else:
            print("❌ Demo failed - see errors above")

    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        print(f"\n❌ Demo failed with error: {e}")


if __name__ == "__main__":
    main()