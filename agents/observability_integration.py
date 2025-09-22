"""
Integration of observability into agent workflows
"""

import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class AgentObservability:
    """Observability integration for trading agents"""

    def __init__(self):
        self.metrics_server = None
        self.tracing_server = None
        self.alerting_server = None
        self._initialize_servers()

    def _initialize_servers(self):
        """Initialize observability servers"""
        try:
            # Import observability servers
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))

            from mcp.servers.observability_server import ObservabilityServer
            from observability.advanced_tracing import get_tracer, initialize_tracing

            # Initialize tracing
            try:
                self.tracer = initialize_tracing("trading-agents")
                logger.info("Distributed tracing initialized")
            except Exception:
                logger.warning("Failed to initialize tracing")
                self.tracer = None

            # Initialize metrics server
            try:
                self.metrics_server = ObservabilityServer()
                logger.info("Metrics server initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize metrics server: {e}")

        except Exception as e:
            logger.warning(f"Failed to initialize observability: {e}")

    async def record_agent_execution(self, agent_name: str, operation: str, duration_ms: float, success: bool, **metadata):
        """Record agent execution metrics"""
        if not self.metrics_server:
            return

        try:
            await self.metrics_server.record_metric({
                "name": f"agent.{agent_name}.{operation}.duration_ms",
                "value": duration_ms,
                "metric_type": "timer",
                "labels": {
                    "agent": agent_name,
                    "operation": operation,
                    "success": str(success),
                    **metadata
                }
            })

            await self.metrics_server.record_metric({
                "name": f"agent.{agent_name}.{operation}.count",
                "value": 1,
                "metric_type": "counter",
                "labels": {
                    "agent": agent_name,
                    "operation": operation,
                    "success": str(success),
                    **metadata
                }
            })

        except Exception as e:
            logger.warning(f"Failed to record agent metrics: {e}")

    async def record_portfolio_metrics(self, portfolio_value: float, var_95: float, sharpe_ratio: float, **metrics):
        """Record portfolio performance metrics"""
        if not self.metrics_server:
            return

        try:
            portfolio_metrics = {
                "portfolio_value_usd": portfolio_value,
                "portfolio_var_95": var_95,
                "portfolio_sharpe_ratio": sharpe_ratio,
                **metrics
            }

            for metric_name, value in portfolio_metrics.items():
                await self.metrics_server.record_metric({
                    "name": metric_name,
                    "value": value,
                    "metric_type": "gauge",
                    "labels": {
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                })

        except Exception as e:
            logger.warning(f"Failed to record portfolio metrics: {e}")

    async def record_market_data_metrics(self, symbol: str, fetch_duration_ms: float, cache_hit: bool, data_points: int):
        """Record market data access metrics"""
        if not self.metrics_server:
            return

        try:
            await self.metrics_server.record_metric({
                "name": "market_data_fetch_duration_ms",
                "value": fetch_duration_ms,
                "metric_type": "timer",
                "labels": {
                    "symbol": symbol,
                    "cache_hit": str(cache_hit)
                }
            })

            await self.metrics_server.record_metric({
                "name": "market_data_points_fetched",
                "value": data_points,
                "metric_type": "gauge",
                "labels": {
                    "symbol": symbol
                }
            })

            # Update cache hit rate
            await self.metrics_server.record_metric({
                "name": "market_data_cache_hit_rate",
                "value": 1.0 if cache_hit else 0.0,
                "metric_type": "gauge",
                "labels": {
                    "symbol": symbol
                }
            })

        except Exception as e:
            logger.warning(f"Failed to record market data metrics: {e}")

    async def record_circuit_breaker_metrics(self, service_name: str, state: str, failure_count: int):
        """Record circuit breaker metrics"""
        if not self.metrics_server:
            return

        try:
            # Circuit breaker state (0=closed, 1=open, 2=half-open)
            state_value = {"closed": 0, "open": 1, "half_open": 2}.get(state, 0)

            await self.metrics_server.record_metric({
                "name": "circuit_breaker_state",
                "value": state_value,
                "metric_type": "gauge",
                "labels": {
                    "service": service_name,
                    "state": state
                }
            })

            await self.metrics_server.record_metric({
                "name": "circuit_breaker_failures_total",
                "value": failure_count,
                "metric_type": "counter",
                "labels": {
                    "service": service_name
                }
            })

        except Exception as e:
            logger.warning(f"Failed to record circuit breaker metrics: {e}")

    def trace_operation(self, operation_name: str, **tags):
        """Create a distributed trace for an operation"""
        if not self.tracer:
            return self._noop_context()

        return self.tracer.trace(operation_name, tags=tags)

    def _noop_context(self):
        """No-op context manager when tracing is unavailable"""
        class NoOpContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def set_tag(self, key, value):
                pass
            def log(self, message, **kwargs):
                pass

        return NoOpContext()

# Global observability instance
_observability = AgentObservability()

def get_observability() -> AgentObservability:
    """Get global observability instance"""
    return _observability

def observe_agent_operation(agent_name: str, operation: str):
    """Decorator to observe agent operations"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            obs = get_observability()
            start_time = time.time()

            with obs.trace_operation(f"{agent_name}.{operation}",
                                   agent=agent_name,
                                   operation=operation) as span:
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    span.set_tag("success", True)
                    return result

                except Exception as e:
                    success = False
                    span.set_tag("success", False)
                    span.set_tag("error", str(e))
                    raise

                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_tag("duration_ms", duration_ms)

                    # Record metrics
                    await obs.record_agent_execution(
                        agent_name, operation, duration_ms, success
                    )

        def sync_wrapper(*args, **kwargs):
            # For sync functions, use asyncio.run if needed
            obs = get_observability()
            start_time = time.time()

            with obs.trace_operation(f"{agent_name}.{operation}",
                                   agent=agent_name,
                                   operation=operation) as span:
                try:
                    result = func(*args, **kwargs)
                    success = True
                    span.set_tag("success", True)
                    return result

                except Exception as e:
                    success = False
                    span.set_tag("success", False)
                    span.set_tag("error", str(e))
                    raise

                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_tag("duration_ms", duration_ms)

                    # Record metrics (async)
                    try:
                        loop = asyncio.get_event_loop()
                        loop.create_task(obs.record_agent_execution(
                            agent_name, operation, duration_ms, success
                        ))
                    except:
                        pass  # Ignore if no event loop

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator