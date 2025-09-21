"""
Autonomous Recovery MCP Server - Self-healing system following Anthropic's guidelines.

This server provides comprehensive tools for autonomous dependency resolution,
system repair, and intelligent recovery from any failure scenario.
"""
from __future__ import annotations

import os
import sys
import subprocess
import platform
import importlib
import tempfile
import shutil
import json
import time
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from mcp.server import register_tool

try:
    from mcp.common.resilience import circuit_breaker, CircuitBreakerConfig
except ImportError:
    def circuit_breaker(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func
        return decorator


@dataclass
class SystemDiagnostics:
    """Comprehensive system diagnostics."""
    python_version: str
    platform_info: str
    pip_version: str
    available_memory: int
    disk_space: int
    network_accessible: bool
    package_index_accessible: bool
    permissions_writable: bool
    virtual_env: Optional[str]


@dataclass
class DependencyAnalysis:
    """Analysis of dependency installation options."""
    package_name: str
    available_versions: List[str]
    installation_strategies: List[str]
    compatibility_issues: List[str]
    recommended_approach: str


class AutonomousRecoveryEngine:
    """Core autonomous recovery engine with comprehensive tooling."""

    def __init__(self):
        self.diagnostics_cache = {}
        self.recovery_history = []
        self.strategy_success_rates = {}

    def diagnose_system(self) -> SystemDiagnostics:
        """Comprehensive system diagnostics for intelligent recovery planning."""
        try:
            # Python diagnostics
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            platform_info = f"{platform.system()} {platform.release()} {platform.machine()}"

            # Pip diagnostics
            try:
                pip_result = subprocess.run([sys.executable, "-m", "pip", "--version"],
                                          capture_output=True, text=True, timeout=10)
                pip_version = pip_result.stdout.strip() if pip_result.returncode == 0 else "unavailable"
            except:
                pip_version = "unavailable"

            # Memory diagnostics (approximate)
            try:
                import psutil
                available_memory = psutil.virtual_memory().available // (1024 * 1024)  # MB
            except ImportError:
                available_memory = 0  # Unknown

            # Disk space diagnostics
            try:
                disk_space = shutil.disk_usage("/").free // (1024 * 1024 * 1024)  # GB
            except:
                disk_space = 0  # Unknown

            # Network diagnostics
            network_accessible = self._test_network_connectivity()
            package_index_accessible = self._test_package_index()

            # Permissions diagnostics
            permissions_writable = self._test_write_permissions()

            # Virtual environment detection
            virtual_env = os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_DEFAULT_ENV')

            return SystemDiagnostics(
                python_version=python_version,
                platform_info=platform_info,
                pip_version=pip_version,
                available_memory=available_memory,
                disk_space=disk_space,
                network_accessible=network_accessible,
                package_index_accessible=package_index_accessible,
                permissions_writable=permissions_writable,
                virtual_env=virtual_env
            )
        except Exception as e:
            # Return minimal diagnostics if comprehensive analysis fails
            return SystemDiagnostics(
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
                platform_info=platform.system(),
                pip_version="unknown",
                available_memory=0,
                disk_space=0,
                network_accessible=False,
                package_index_accessible=False,
                permissions_writable=False,
                virtual_env=None
            )

    def _test_network_connectivity(self) -> bool:
        """Test basic network connectivity."""
        try:
            import urllib.request
            urllib.request.urlopen('https://httpbin.org/get', timeout=5)
            return True
        except:
            return False

    def _test_package_index(self) -> bool:
        """Test PyPI accessibility."""
        try:
            import urllib.request
            urllib.request.urlopen('https://pypi.org/simple/', timeout=5)
            return True
        except:
            return False

    def _test_write_permissions(self) -> bool:
        """Test write permissions in site-packages."""
        try:
            import site
            site_packages = site.getsitepackages()[0]
            test_file = Path(site_packages) / ".autonomous_test"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except:
            return False

    def analyze_dependency(self, package_name: str) -> DependencyAnalysis:
        """Analyze dependency installation options and compatibility."""
        try:
            # Get available versions
            result = subprocess.run([
                sys.executable, "-m", "pip", "index", "versions", package_name
            ], capture_output=True, text=True, timeout=30)

            available_versions = []
            if result.returncode == 0:
                # Parse pip index output for versions
                for line in result.stdout.split('\n'):
                    if 'Available versions:' in line:
                        versions_part = line.split('Available versions:')[1].strip()
                        available_versions = [v.strip() for v in versions_part.split(',')]
                        break

            # Determine installation strategies based on system diagnostics
            diagnostics = self.diagnose_system()
            strategies = self._get_installation_strategies(package_name, diagnostics)

            # Check for compatibility issues
            compatibility_issues = self._check_compatibility_issues(package_name, diagnostics)

            # Recommend best approach
            recommended_approach = self._recommend_installation_approach(
                package_name, diagnostics, strategies, compatibility_issues
            )

            return DependencyAnalysis(
                package_name=package_name,
                available_versions=available_versions,
                installation_strategies=strategies,
                compatibility_issues=compatibility_issues,
                recommended_approach=recommended_approach
            )
        except Exception as e:
            return DependencyAnalysis(
                package_name=package_name,
                available_versions=[],
                installation_strategies=["standard_pip"],
                compatibility_issues=[f"Analysis failed: {e}"],
                recommended_approach="standard_pip"
            )

    def _get_installation_strategies(self, package_name: str, diagnostics: SystemDiagnostics) -> List[str]:
        """Get available installation strategies based on system state."""
        strategies = ["standard_pip"]

        if diagnostics.network_accessible:
            strategies.extend(["pip_no_cache", "pip_force_reinstall", "pip_upgrade"])

        if diagnostics.permissions_writable:
            strategies.append("pip_system_wide")
        else:
            strategies.append("pip_user_install")

        if diagnostics.virtual_env:
            strategies.append("pip_in_venv")

        # Platform-specific strategies
        if "Windows" in diagnostics.platform_info:
            strategies.append("windows_binary_wheel")
        elif "Linux" in diagnostics.platform_info:
            strategies.extend(["build_from_source", "apt_dependencies"])
        elif "Darwin" in diagnostics.platform_info:
            strategies.extend(["homebrew_dependencies", "conda_forge"])

        return strategies

    def _check_compatibility_issues(self, package_name: str, diagnostics: SystemDiagnostics) -> List[str]:
        """Check for known compatibility issues."""
        issues = []

        # Python version compatibility
        python_major, python_minor = map(int, diagnostics.python_version.split('.')[:2])
        if python_major < 3 or (python_major == 3 and python_minor < 8):
            issues.append(f"Python {diagnostics.python_version} may have compatibility issues")

        # Platform-specific issues
        if package_name in ["hmmlearn", "scipy", "numpy"] and "arm64" in diagnostics.platform_info:
            issues.append("ARM64 architecture may require specific wheel versions")

        # Memory constraints
        if diagnostics.available_memory < 1024 and package_name in ["torch", "tensorflow"]:
            issues.append("Insufficient memory for large ML packages")

        return issues

    def _recommend_installation_approach(self, package_name: str, diagnostics: SystemDiagnostics,
                                       strategies: List[str], issues: List[str]) -> str:
        """Recommend the best installation approach."""
        # Prioritize based on success rates and system state
        if not diagnostics.network_accessible:
            return "offline_installation_required"

        if not diagnostics.permissions_writable:
            return "pip_user_install"

        if diagnostics.virtual_env:
            return "pip_in_venv"

        # Check historical success rates
        best_strategy = "standard_pip"
        best_rate = 0.0

        for strategy in strategies:
            rate = self.strategy_success_rates.get(f"{package_name}:{strategy}", 0.5)
            if rate > best_rate:
                best_rate = rate
                best_strategy = strategy

        return best_strategy

    def execute_recovery_strategy(self, package_name: str, strategy: str) -> Dict[str, Any]:
        """Execute a specific recovery strategy with comprehensive error handling."""
        start_time = time.time()

        try:
            success = False
            error_message = ""
            details = {}

            if strategy == "standard_pip":
                success, error_message = self._execute_standard_pip(package_name)
            elif strategy == "pip_no_cache":
                success, error_message = self._execute_pip_no_cache(package_name)
            elif strategy == "pip_force_reinstall":
                success, error_message = self._execute_pip_force_reinstall(package_name)
            elif strategy == "pip_user_install":
                success, error_message = self._execute_pip_user_install(package_name)
            elif strategy == "build_from_source":
                success, error_message = self._execute_build_from_source(package_name)
            elif strategy == "conda_installation":
                success, error_message = self._execute_conda_installation(package_name)
            else:
                success, error_message = self._execute_custom_strategy(package_name, strategy)

            # Update success rates
            strategy_key = f"{package_name}:{strategy}"
            if strategy_key not in self.strategy_success_rates:
                self.strategy_success_rates[strategy_key] = 0.5

            # Exponential moving average
            alpha = 0.3
            old_rate = self.strategy_success_rates[strategy_key]
            new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * old_rate
            self.strategy_success_rates[strategy_key] = new_rate

            execution_time = time.time() - start_time

            result = {
                "package_name": package_name,
                "strategy": strategy,
                "success": success,
                "error_message": error_message,
                "execution_time": execution_time,
                "details": details
            }

            self.recovery_history.append(result)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = {
                "package_name": package_name,
                "strategy": strategy,
                "success": False,
                "error_message": f"Strategy execution failed: {e}",
                "execution_time": execution_time,
                "details": {}
            }
            self.recovery_history.append(result)
            return result

    def _execute_standard_pip(self, package_name: str) -> tuple[bool, str]:
        """Execute standard pip installation."""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "Installation timed out"
        except Exception as e:
            return False, str(e)

    def _execute_pip_no_cache(self, package_name: str) -> tuple[bool, str]:
        """Execute pip installation without cache."""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name, "--no-cache-dir"
            ], capture_output=True, text=True, timeout=300)

            return result.returncode == 0, result.stderr.strip() if result.returncode != 0 else ""
        except Exception as e:
            return False, str(e)

    def _execute_pip_force_reinstall(self, package_name: str) -> tuple[bool, str]:
        """Execute pip force reinstall."""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name, "--force-reinstall"
            ], capture_output=True, text=True, timeout=300)

            return result.returncode == 0, result.stderr.strip() if result.returncode != 0 else ""
        except Exception as e:
            return False, str(e)

    def _execute_pip_user_install(self, package_name: str) -> tuple[bool, str]:
        """Execute pip user installation."""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name, "--user"
            ], capture_output=True, text=True, timeout=300)

            return result.returncode == 0, result.stderr.strip() if result.returncode != 0 else ""
        except Exception as e:
            return False, str(e)

    def _execute_build_from_source(self, package_name: str) -> tuple[bool, str]:
        """Execute build from source installation."""
        try:
            # First try to install build dependencies
            subprocess.run([
                sys.executable, "-m", "pip", "install", "build", "setuptools", "wheel"
            ], capture_output=True, text=True, timeout=120)

            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name, "--no-binary", ":all:"
            ], capture_output=True, text=True, timeout=600)

            return result.returncode == 0, result.stderr.strip() if result.returncode != 0 else ""
        except Exception as e:
            return False, str(e)

    def _execute_conda_installation(self, package_name: str) -> tuple[bool, str]:
        """Execute conda installation if available."""
        try:
            # Check if conda is available
            conda_result = subprocess.run(["conda", "--version"],
                                        capture_output=True, text=True, timeout=10)
            if conda_result.returncode != 0:
                return False, "Conda not available"

            result = subprocess.run([
                "conda", "install", "-y", package_name
            ], capture_output=True, text=True, timeout=300)

            return result.returncode == 0, result.stderr.strip() if result.returncode != 0 else ""
        except Exception as e:
            return False, str(e)

    def _execute_custom_strategy(self, package_name: str, strategy: str) -> tuple[bool, str]:
        """Execute custom strategy based on package-specific knowledge."""
        # Package-specific strategies
        if package_name == "hmmlearn" and strategy == "scipy_first":
            try:
                # Install scipy dependencies first
                deps_result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "scipy", "scikit-learn", "numpy"
                ], capture_output=True, text=True, timeout=300)

                if deps_result.returncode == 0:
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", package_name
                    ], capture_output=True, text=True, timeout=300)
                    return result.returncode == 0, result.stderr.strip() if result.returncode != 0 else ""
                else:
                    return False, "Failed to install dependencies"
            except Exception as e:
                return False, str(e)

        return False, f"Unknown strategy: {strategy}"


# Initialize the recovery engine
recovery_engine = AutonomousRecoveryEngine()


@register_tool(
    name="autonomous.system.diagnose",
    schema="./schemas/tool.autonomous.system.diagnose.schema.json",
)
@circuit_breaker(
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=60.0,
        expected_exception=Exception
    )
)
def diagnose_system(params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform comprehensive system diagnostics for autonomous recovery planning."""
    include_cached = params.get("include_cached", True)

    if include_cached and "system_diagnostics" in recovery_engine.diagnostics_cache:
        cached_time = recovery_engine.diagnostics_cache.get("timestamp", 0)
        if time.time() - cached_time < 300:  # 5-minute cache
            diagnostics = recovery_engine.diagnostics_cache["system_diagnostics"]
            return {
                "diagnostics": diagnostics.__dict__,
                "cached": True,
                "timestamp": cached_time
            }

    diagnostics = recovery_engine.diagnose_system()

    # Cache the results
    recovery_engine.diagnostics_cache = {
        "system_diagnostics": diagnostics,
        "timestamp": time.time()
    }

    return {
        "diagnostics": diagnostics.__dict__,
        "cached": False,
        "timestamp": time.time()
    }


@register_tool(
    name="autonomous.dependency.analyze",
    schema="./schemas/tool.autonomous.dependency.analyze.schema.json",
)
@circuit_breaker(
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=60.0,
        expected_exception=Exception
    )
)
def analyze_dependency(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze dependency installation options and recommend recovery strategies."""
    package_name = params["package_name"]
    include_system_info = params.get("include_system_info", True)

    analysis = recovery_engine.analyze_dependency(package_name)

    result = {
        "analysis": analysis.__dict__,
        "timestamp": time.time()
    }

    if include_system_info:
        diagnostics = recovery_engine.diagnose_system()
        result["system_diagnostics"] = diagnostics.__dict__

    return result


@register_tool(
    name="autonomous.recovery.execute_strategy",
    schema="./schemas/tool.autonomous.recovery.execute_strategy.schema.json",
)
@circuit_breaker(
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=60.0,
        expected_exception=Exception
    )
)
def execute_recovery_strategy(params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a specific recovery strategy for dependency installation."""
    package_name = params["package_name"]
    strategy = params["strategy"]
    verify_installation = params.get("verify_installation", True)

    result = recovery_engine.execute_recovery_strategy(package_name, strategy)

    if verify_installation and result["success"]:
        # Verify the package can actually be imported
        try:
            __import__(package_name)
            result["import_verified"] = True
        except ImportError:
            result["import_verified"] = False
            result["verification_error"] = f"Package {package_name} installed but cannot be imported"

    return result


@register_tool(
    name="autonomous.recovery.auto_resolve",
    schema="./schemas/tool.autonomous.recovery.auto_resolve.schema.json",
)
@circuit_breaker(
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=60.0,
        expected_exception=Exception
    )
)
def auto_resolve_dependency(params: Dict[str, Any]) -> Dict[str, Any]:
    """Autonomously resolve dependency issues using intelligent strategy selection."""
    package_name = params["package_name"]
    max_attempts = params.get("max_attempts", 5)
    parallel_strategies = params.get("parallel_strategies", False)

    # Analyze the dependency first
    analysis = recovery_engine.analyze_dependency(package_name)

    strategies = analysis.installation_strategies
    if analysis.recommended_approach in strategies:
        # Put recommended approach first
        strategies = [analysis.recommended_approach] + [s for s in strategies if s != analysis.recommended_approach]

    attempts = []
    success = False

    if parallel_strategies and len(strategies) > 1:
        # Try multiple strategies in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(recovery_engine.execute_recovery_strategy, package_name, strategy): strategy
                for strategy in strategies[:3]
            }

            for future in concurrent.futures.as_completed(futures):
                strategy = futures[future]
                try:
                    result = future.result()
                    attempts.append(result)
                    if result["success"]:
                        success = True
                        break
                except Exception as e:
                    attempts.append({
                        "package_name": package_name,
                        "strategy": strategy,
                        "success": False,
                        "error_message": f"Parallel execution failed: {e}",
                        "execution_time": 0
                    })
    else:
        # Try strategies sequentially
        for i, strategy in enumerate(strategies[:max_attempts]):
            result = recovery_engine.execute_recovery_strategy(package_name, strategy)
            attempts.append(result)

            if result["success"]:
                success = True
                break

    # Final verification
    import_verified = False
    if success:
        try:
            __import__(package_name)
            import_verified = True
        except ImportError:
            import_verified = False

    return {
        "package_name": package_name,
        "success": success,
        "import_verified": import_verified,
        "attempts": attempts,
        "total_attempts": len(attempts),
        "analysis": analysis.__dict__,
        "timestamp": time.time()
    }


@register_tool(
    name="autonomous.recovery.get_history",
    schema="./schemas/tool.autonomous.recovery.get_history.schema.json",
)
@circuit_breaker(
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=60.0,
        expected_exception=Exception
    )
)
def get_recovery_history(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get the history of recovery attempts for learning and optimization."""
    limit = params.get("limit", 50)
    package_filter = params.get("package_filter")
    success_only = params.get("success_only", False)

    history = recovery_engine.recovery_history

    if package_filter:
        history = [h for h in history if h["package_name"] == package_filter]

    if success_only:
        history = [h for h in history if h["success"]]

    # Sort by timestamp (most recent first)
    history = sorted(history, key=lambda x: x.get("timestamp", 0), reverse=True)

    return {
        "history": history[:limit],
        "total_records": len(recovery_engine.recovery_history),
        "filtered_records": len(history),
        "strategy_success_rates": recovery_engine.strategy_success_rates
    }


__all__ = [
    "diagnose_system",
    "analyze_dependency",
    "execute_recovery_strategy",
    "auto_resolve_dependency",
    "get_recovery_history"
]