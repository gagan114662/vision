"""Performance monitoring and observability MCP server.

Provides comprehensive system monitoring, performance metrics collection,
and observability tools for the quantitative trading infrastructure.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

try:
    from mcp.server import register_tool
    from mcp.common.resilience import circuit_breaker, CircuitBreakerConfig
    from mcp.common.server_config import get_server_config, get_tool_config
except ImportError:  # pragma: no cover
    def register_tool(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func
        return decorator

    def circuit_breaker(*_args: Any, **_kwargs: Any):  # type: ignore
        def decorator(func: Any) -> Any:
            return func
        return decorator

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    level: AlertLevel
    title: str
    description: str
    timestamp: datetime
    source: str
    resolved: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    metric_name: str
    baseline_value: float
    threshold_warning: float
    threshold_critical: float
    created_at: datetime


class MetricsCollector:
    """Collects and stores performance metrics."""

    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self._metrics: Dict[str, Dict[MetricType, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=10000))
        )
        self._alerts: deque = deque(maxlen=1000)
        self._baselines: Dict[str, PerformanceBaseline] = {}
        self._lock = threading.RLock()

        # System metrics collection
        self._system_metrics_enabled = True
        self._collection_interval = 30  # seconds
        self._system_metrics_supported = psutil is not None
        self._system_metrics_enabled = self._system_metrics_supported

        # Performance tracking
        self._operation_timers: Dict[str, List[float]] = defaultdict(list)

        # Start background collection
        self._collection_task = None

    async def start_collection(self):
        """Start background metrics collection."""
        if not self._system_metrics_supported:
            logger.info("psutil not available; skipping system metrics collection")
            return

        if self._collection_task is None:
            self._collection_task = asyncio.create_task(self._collect_system_metrics())
            logger.info("Started metrics collection")

    async def stop_collection(self):
        """Stop background metrics collection."""
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None
            logger.info("Stopped metrics collection")

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric value."""
        with self._lock:
            point = MetricPoint(
                timestamp=datetime.now(timezone.utc),
                value=value,
                labels=labels or {}
            )

            self._metrics[name][metric_type].append(point)

            # Check for alerts
            self._check_metric_alerts(name, value, metric_type)

            # Clean old data
            self._cleanup_old_metrics()

    def _check_metric_alerts(self, name: str, value: float, metric_type: MetricType):
        """Check if metric value triggers alerts."""
        if name in self._baselines:
            baseline = self._baselines[name]

            if value >= baseline.threshold_critical:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    f"Critical: {name}",
                    f"Metric {name} exceeded critical threshold: {value:.2f} >= {baseline.threshold_critical:.2f}",
                    source="metrics_collector",
                    metadata={"metric_name": name, "value": value, "threshold": baseline.threshold_critical}
                )
            elif value >= baseline.threshold_warning:
                self._create_alert(
                    AlertLevel.WARNING,
                    f"Warning: {name}",
                    f"Metric {name} exceeded warning threshold: {value:.2f} >= {baseline.threshold_warning:.2f}",
                    source="metrics_collector",
                    metadata={"metric_name": name, "value": value, "threshold": baseline.threshold_warning}
                )

    def _create_alert(
        self,
        level: AlertLevel,
        title: str,
        description: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a new alert."""
        alert = Alert(
            alert_id=f"alert_{int(time.time() * 1000)}",
            level=level,
            title=title,
            description=description,
            timestamp=datetime.now(timezone.utc),
            source=source,
            metadata=metadata or {}
        )

        self._alerts.append(alert)
        logger.warning(f"Alert created: {level.value} - {title}")

    def _cleanup_old_metrics(self):
        """Clean up old metric data points."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)

        for metric_name in self._metrics:
            for metric_type in self._metrics[metric_name]:
                # Remove old points
                while (self._metrics[metric_name][metric_type] and
                       self._metrics[metric_name][metric_type][0].timestamp < cutoff_time):
                    self._metrics[metric_name][metric_type].popleft()

    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        if psutil is None:
            logger.debug("System metrics collection skipped due to missing psutil")
            return

        while self._system_metrics_enabled:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_metric("system.cpu.utilization", cpu_percent, MetricType.GAUGE)

                # Memory metrics
                memory = psutil.virtual_memory()
                self.record_metric("system.memory.utilization", memory.percent, MetricType.GAUGE)
                self.record_metric("system.memory.available_gb", memory.available / (1024**3), MetricType.GAUGE)

                # Disk metrics
                disk = psutil.disk_usage('/')
                self.record_metric("system.disk.utilization", disk.percent, MetricType.GAUGE)
                self.record_metric("system.disk.free_gb", disk.free / (1024**3), MetricType.GAUGE)

                # Network metrics
                network = psutil.net_io_counters()
                self.record_metric("system.network.bytes_sent", network.bytes_sent, MetricType.COUNTER)
                self.record_metric("system.network.bytes_recv", network.bytes_recv, MetricType.COUNTER)

                # Process metrics
                process = psutil.Process()
                self.record_metric("process.memory.rss_mb", process.memory_info().rss / (1024**2), MetricType.GAUGE)
                self.record_metric("process.cpu.percent", process.cpu_percent(), MetricType.GAUGE)

                await asyncio.sleep(self._collection_interval)

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(self._collection_interval)

    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get collected metrics."""
        with self._lock:
            if not start_time:
                start_time = datetime.now(timezone.utc) - timedelta(hours=1)
            if not end_time:
                end_time = datetime.now(timezone.utc)

            result = {}

            metrics_to_process = [metric_name] if metric_name else list(self._metrics.keys())

            for name in metrics_to_process:
                if name not in self._metrics:
                    continue

                result[name] = {}

                for metric_type, points in self._metrics[name].items():
                    # Filter by time range
                    filtered_points = [
                        p for p in points
                        if start_time <= p.timestamp <= end_time
                    ]

                    if filtered_points:
                        # Convert to serializable format
                        serializable_points = []
                        for point in filtered_points:
                            point_dict = asdict(point)
                            point_dict["timestamp"] = point.timestamp.isoformat()
                            serializable_points.append(point_dict)

                        result[name][metric_type.value] = {
                            "points": serializable_points,
                            "count": len(filtered_points),
                            "latest_value": filtered_points[-1].value if filtered_points else None,
                            "statistics": self._calculate_statistics([p.value for p in filtered_points])
                        }

            return result

    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical summary of values."""
        if not values:
            return {}

        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "percentile_95": statistics.quantiles(values, n=20)[18] if len(values) >= 2 else values[0],
            "percentile_99": statistics.quantiles(values, n=100)[98] if len(values) >= 2 else values[0]
        }

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        resolved: Optional[bool] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get system alerts."""
        with self._lock:
            filtered_alerts = []

            for alert in reversed(list(self._alerts)):  # Most recent first
                if level and alert.level != level:
                    continue
                if resolved is not None and alert.resolved != resolved:
                    continue

                alert_dict = asdict(alert)
                alert_dict["timestamp"] = alert.timestamp.isoformat()
                alert_dict["level"] = alert.level.value
                filtered_alerts.append(alert_dict)

                if len(filtered_alerts) >= limit:
                    break

            return filtered_alerts

    def set_baseline(
        self,
        metric_name: str,
        baseline_value: float,
        warning_threshold_pct: float = 20.0,
        critical_threshold_pct: float = 50.0
    ):
        """Set performance baseline for a metric."""
        baseline = PerformanceBaseline(
            metric_name=metric_name,
            baseline_value=baseline_value,
            threshold_warning=baseline_value * (1 + warning_threshold_pct / 100),
            threshold_critical=baseline_value * (1 + critical_threshold_pct / 100),
            created_at=datetime.now(timezone.utc)
        )

        self._baselines[metric_name] = baseline
        logger.info(f"Set baseline for {metric_name}: {baseline_value}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        with self._lock:
            now = datetime.now(timezone.utc)
            last_hour = now - timedelta(hours=1)

            # Get recent alerts
            recent_alerts = [
                alert for alert in self._alerts
                if alert.timestamp >= last_hour and not alert.resolved
            ]

            # Calculate health score
            critical_alerts = len([a for a in recent_alerts if a.level == AlertLevel.CRITICAL])
            warning_alerts = len([a for a in recent_alerts if a.level == AlertLevel.WARNING])

            health_score = 100
            health_score -= critical_alerts * 25  # Critical alerts heavily impact score
            health_score -= warning_alerts * 10   # Warning alerts moderately impact score
            health_score = max(0, health_score)

            # Determine health status
            if health_score >= 90:
                health_status = "excellent"
            elif health_score >= 75:
                health_status = "good"
            elif health_score >= 50:
                health_status = "fair"
            elif health_score >= 25:
                health_status = "poor"
            else:
                health_status = "critical"

            # Get latest system metrics
            system_metrics = {}
            for metric_name in ["system.cpu.utilization", "system.memory.utilization", "system.disk.utilization"]:
                if metric_name in self._metrics and MetricType.GAUGE in self._metrics[metric_name]:
                    gauge_data = self._metrics[metric_name][MetricType.GAUGE]
                    if gauge_data:
                        system_metrics[metric_name] = gauge_data[-1].value

            return {
                "health_score": health_score,
                "health_status": health_status,
                "active_alerts": {
                    "critical": critical_alerts,
                    "warning": warning_alerts,
                    "total": len(recent_alerts)
                },
                "system_metrics": system_metrics,
                "metrics_collected": len(self._metrics),
                "data_retention_hours": self.retention_hours,
                "timestamp": now.isoformat()
            }


# Global metrics collector
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


@register_tool(
    name="observability.metrics.record",
    schema="./schemas/tool.observability.metrics.record.schema.json"
)
@circuit_breaker(
    config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_seconds=30.0,
        expected_exception=Exception
    )
)
async def record_metric(params: Dict[str, Any]) -> Dict[str, Any]:
    """Record a performance metric."""
    try:
        name = params["name"]
        value = float(params["value"])
        metric_type_str = params.get("metric_type", "gauge")
        labels = params.get("labels", {})

        # Validate metric type
        try:
            metric_type = MetricType(metric_type_str)
        except ValueError:
            return {
                "error": f"Invalid metric type: {metric_type_str}",
                "valid_types": [t.value for t in MetricType]
            }

        collector = get_metrics_collector()
        collector.record_metric(name, value, metric_type, labels)

        return {
            "success": True,
            "metric_name": name,
            "value": value,
            "type": metric_type.value,
            "recorded_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Record metric error: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@register_tool(
    name="observability.metrics.get",
    schema="./schemas/tool.observability.metrics.get.schema.json"
)
@circuit_breaker(
    config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_seconds=30.0,
        expected_exception=Exception
    )
)
async def get_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get collected performance metrics."""
    try:
        metric_name = params.get("metric_name")
        hours_back = params.get("hours_back", 1)

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours_back)

        collector = get_metrics_collector()
        metrics = collector.get_metrics(metric_name, start_time, end_time)

        return {
            "success": True,
            "metrics": metrics,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours_back
            },
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Get metrics error: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@register_tool(
    name="observability.health.status",
    schema="./schemas/tool.observability.health.status.schema.json"
)
@circuit_breaker(
    config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_seconds=30.0,
        expected_exception=Exception
    )
)
async def get_health_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get overall system health status."""
    try:
        collector = get_metrics_collector()
        health_data = collector.get_system_health()

        return {
            "success": True,
            "health": health_data
        }

    except Exception as e:
        logger.error(f"Get health status error: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@register_tool(
    name="observability.alerts.get",
    schema="./schemas/tool.observability.alerts.get.schema.json"
)
@circuit_breaker(
    config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_seconds=30.0,
        expected_exception=Exception
    )
)
async def get_alerts(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get system alerts."""
    try:
        level_str = params.get("level")
        resolved = params.get("resolved")
        limit = params.get("limit", 100)

        level = None
        if level_str:
            try:
                level = AlertLevel(level_str)
            except ValueError:
                return {
                    "error": f"Invalid alert level: {level_str}",
                    "valid_levels": [l.value for l in AlertLevel]
                }

        collector = get_metrics_collector()
        alerts = collector.get_alerts(level, resolved, limit)

        return {
            "success": True,
            "alerts": alerts,
            "count": len(alerts),
            "filters": {
                "level": level_str,
                "resolved": resolved,
                "limit": limit
            },
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Get alerts error: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@register_tool(
    name="observability.baseline.set",
    schema="./schemas/tool.observability.baseline.set.schema.json"
)
@circuit_breaker(
    config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_seconds=30.0,
        expected_exception=Exception
    )
)
async def set_performance_baseline(params: Dict[str, Any]) -> Dict[str, Any]:
    """Set performance baseline for monitoring."""
    try:
        metric_name = params["metric_name"]
        baseline_value = float(params["baseline_value"])
        warning_threshold_pct = params.get("warning_threshold_pct", 20.0)
        critical_threshold_pct = params.get("critical_threshold_pct", 50.0)

        collector = get_metrics_collector()
        collector.set_baseline(
            metric_name,
            baseline_value,
            warning_threshold_pct,
            critical_threshold_pct
        )

        return {
            "success": True,
            "metric_name": metric_name,
            "baseline_value": baseline_value,
            "thresholds": {
                "warning_pct": warning_threshold_pct,
                "critical_pct": critical_threshold_pct
            },
            "set_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Set baseline error: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Initialize metrics collection on import
async def initialize_observability():
    """Initialize observability system."""
    collector = get_metrics_collector()
    await collector.start_collection()

    # Set default baselines
    collector.set_baseline("system.cpu.utilization", 50.0, 30.0, 60.0)  # 50% baseline, 65% warning, 80% critical
    collector.set_baseline("system.memory.utilization", 60.0, 25.0, 50.0)  # 60% baseline, 75% warning, 90% critical
    collector.set_baseline("system.disk.utilization", 70.0, 20.0, 35.0)   # 70% baseline, 84% warning, 95% critical

    logger.info("Observability system initialized")


__all__ = [
    "MetricsCollector",
    "MetricType",
    "AlertLevel",
    "get_metrics_collector",
    "record_metric",
    "get_metrics",
    "get_health_status",
    "get_alerts",
    "set_performance_baseline",
    "initialize_observability"
]
