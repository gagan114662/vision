"""
Production Monitoring and SLA Dashboard System.

Implements comprehensive monitoring, alerting, and dashboard generation
for production trading systems with real-time SLA tracking.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json
from collections import deque, defaultdict
import threading

logger = logging.getLogger(__name__)


class ServiceLevel(Enum):
    """SLA service levels."""
    CRITICAL = "critical"      # 99.99% uptime, <10ms latency
    HIGH = "high"             # 99.9% uptime, <100ms latency
    STANDARD = "standard"      # 99% uptime, <1s latency
    BEST_EFFORT = "best_effort" # No SLA guarantee


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5


class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class SLATarget:
    """Service Level Agreement targets."""
    service_name: str
    service_level: ServiceLevel
    uptime_target: float  # Percentage (e.g., 99.9)
    latency_p50_ms: float
    latency_p99_ms: float
    error_rate_threshold: float  # Percentage
    throughput_minimum: float  # Requests per second
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ServiceMetrics:
    """Real-time service metrics."""
    service_name: str
    timestamp: datetime
    uptime_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    latency_p50_ms: float
    latency_p99_ms: float
    throughput_rps: float
    cpu_percent: float
    memory_mb: float
    active_connections: int
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SLAViolation:
    """SLA violation record."""
    violation_id: str
    service_name: str
    violation_type: str
    severity: AlertSeverity
    timestamp: datetime
    duration_seconds: float
    metric_name: str
    expected_value: float
    actual_value: float
    impact_description: str
    remediation_action: Optional[str] = None


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    severity: AlertSeverity
    service_name: str
    alert_type: str
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    resolution_time: Optional[datetime] = None


class MetricsCollector:
    """Collect and aggregate service metrics."""

    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics_buffer: Dict[str, deque] = {}
        self.latency_samples: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def record_metric(
        self,
        service_name: str,
        metrics: ServiceMetrics
    ):
        """Record service metrics."""
        with self._lock:
            if service_name not in self.metrics_buffer:
                self.metrics_buffer[service_name] = deque(maxlen=self.retention_hours * 3600)

            self.metrics_buffer[service_name].append(metrics)

    def record_latency(self, service_name: str, latency_ms: float):
        """Record individual latency sample."""
        with self._lock:
            if service_name not in self.latency_samples:
                self.latency_samples[service_name] = []

            self.latency_samples[service_name].append(latency_ms)

            # Keep only recent samples (last hour)
            if len(self.latency_samples[service_name]) > 3600:
                self.latency_samples[service_name] = self.latency_samples[service_name][-3600:]

    def get_current_metrics(self, service_name: str) -> Optional[ServiceMetrics]:
        """Get most recent metrics for service."""
        with self._lock:
            if service_name in self.metrics_buffer and self.metrics_buffer[service_name]:
                return self.metrics_buffer[service_name][-1]
        return None

    def calculate_percentiles(self, service_name: str) -> Dict[str, float]:
        """Calculate latency percentiles."""
        with self._lock:
            if service_name not in self.latency_samples or not self.latency_samples[service_name]:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

            sorted_latencies = sorted(self.latency_samples[service_name])
            n = len(sorted_latencies)

            return {
                "p50": sorted_latencies[int(n * 0.50)],
                "p95": sorted_latencies[int(n * 0.95)],
                "p99": sorted_latencies[int(n * 0.99)]
            }

    def calculate_uptime(self, service_name: str, hours: int = 24) -> float:
        """Calculate uptime percentage over specified hours."""
        with self._lock:
            if service_name not in self.metrics_buffer:
                return 0.0

            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            recent_metrics = [
                m for m in self.metrics_buffer[service_name]
                if m.timestamp >= cutoff_time
            ]

            if not recent_metrics:
                return 0.0

            # Calculate uptime based on successful requests
            total_requests = sum(m.total_requests for m in recent_metrics)
            successful_requests = sum(m.successful_requests for m in recent_metrics)

            if total_requests == 0:
                return 100.0

            return (successful_requests / total_requests) * 100


class SLAMonitor:
    """Monitor SLA compliance and detect violations."""

    def __init__(self):
        self.sla_targets: Dict[str, SLATarget] = {}
        self.violations: List[SLAViolation] = []
        self.violation_counters: Dict[str, int] = defaultdict(int)
        self._violation_id_counter = 0

    def register_sla(self, sla_target: SLATarget):
        """Register SLA target for a service."""
        self.sla_targets[sla_target.service_name] = sla_target
        logger.info(f"SLA registered for {sla_target.service_name}: {sla_target.service_level.value}")

    def check_compliance(
        self,
        service_name: str,
        current_metrics: ServiceMetrics,
        uptime_percent: float
    ) -> List[SLAViolation]:
        """Check if current metrics comply with SLA."""
        if service_name not in self.sla_targets:
            return []

        sla = self.sla_targets[service_name]
        violations = []

        # Check uptime
        if uptime_percent < sla.uptime_target:
            violation = self._create_violation(
                service_name,
                "uptime",
                AlertSeverity.CRITICAL if sla.service_level == ServiceLevel.CRITICAL else AlertSeverity.HIGH,
                "uptime_percentage",
                sla.uptime_target,
                uptime_percent,
                f"Uptime {uptime_percent:.2f}% below target {sla.uptime_target}%"
            )
            violations.append(violation)

        # Check latency
        if current_metrics.latency_p50_ms > sla.latency_p50_ms:
            violation = self._create_violation(
                service_name,
                "latency_p50",
                AlertSeverity.HIGH,
                "p50_latency_ms",
                sla.latency_p50_ms,
                current_metrics.latency_p50_ms,
                f"P50 latency {current_metrics.latency_p50_ms:.2f}ms exceeds target {sla.latency_p50_ms}ms"
            )
            violations.append(violation)

        if current_metrics.latency_p99_ms > sla.latency_p99_ms:
            violation = self._create_violation(
                service_name,
                "latency_p99",
                AlertSeverity.MEDIUM,
                "p99_latency_ms",
                sla.latency_p99_ms,
                current_metrics.latency_p99_ms,
                f"P99 latency {current_metrics.latency_p99_ms:.2f}ms exceeds target {sla.latency_p99_ms}ms"
            )
            violations.append(violation)

        # Check error rate
        if current_metrics.total_requests > 0:
            error_rate = (current_metrics.failed_requests / current_metrics.total_requests) * 100
            if error_rate > sla.error_rate_threshold:
                violation = self._create_violation(
                    service_name,
                    "error_rate",
                    AlertSeverity.HIGH,
                    "error_rate_percentage",
                    sla.error_rate_threshold,
                    error_rate,
                    f"Error rate {error_rate:.2f}% exceeds threshold {sla.error_rate_threshold}%"
                )
                violations.append(violation)

        # Check throughput
        if current_metrics.throughput_rps < sla.throughput_minimum:
            violation = self._create_violation(
                service_name,
                "throughput",
                AlertSeverity.MEDIUM,
                "throughput_rps",
                sla.throughput_minimum,
                current_metrics.throughput_rps,
                f"Throughput {current_metrics.throughput_rps:.2f} RPS below minimum {sla.throughput_minimum} RPS"
            )
            violations.append(violation)

        # Store violations
        self.violations.extend(violations)

        return violations

    def _create_violation(
        self,
        service_name: str,
        violation_type: str,
        severity: AlertSeverity,
        metric_name: str,
        expected_value: float,
        actual_value: float,
        impact_description: str
    ) -> SLAViolation:
        """Create SLA violation record."""
        self._violation_id_counter += 1
        self.violation_counters[f"{service_name}_{violation_type}"] += 1

        return SLAViolation(
            violation_id=f"VIO-{self._violation_id_counter:06d}",
            service_name=service_name,
            violation_type=violation_type,
            severity=severity,
            timestamp=datetime.now(timezone.utc),
            duration_seconds=0,  # Will be updated when resolved
            metric_name=metric_name,
            expected_value=expected_value,
            actual_value=actual_value,
            impact_description=impact_description
        )

    def get_violation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent SLA violations."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_violations = [v for v in self.violations if v.timestamp >= cutoff_time]

        if not recent_violations:
            return {"total_violations": 0}

        # Group by service
        by_service = defaultdict(list)
        for violation in recent_violations:
            by_service[violation.service_name].append(violation)

        # Calculate statistics
        severity_counts = defaultdict(int)
        for violation in recent_violations:
            severity_counts[violation.severity.name] += 1

        return {
            "total_violations": len(recent_violations),
            "by_severity": dict(severity_counts),
            "by_service": {
                service: {
                    "count": len(violations),
                    "types": list(set(v.violation_type for v in violations))
                }
                for service, violations in by_service.items()
            },
            "most_common": self._get_most_common_violations(),
            "period_hours": hours
        }

    def _get_most_common_violations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most common violation types."""
        sorted_violations = sorted(
            self.violation_counters.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        return [
            {"type": violation_type, "count": count}
            for violation_type, count in sorted_violations
        ]


class AlertManager:
    """Manage system alerts and notifications."""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        self._alert_id_counter = 0
        self._lock = threading.Lock()

    def register_handler(self, severity: AlertSeverity, handler: Callable):
        """Register alert handler for specific severity."""
        self.alert_handlers[severity].append(handler)

    async def create_alert(
        self,
        severity: AlertSeverity,
        service_name: str,
        alert_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create and dispatch alert."""
        with self._lock:
            self._alert_id_counter += 1
            alert = Alert(
                alert_id=f"ALT-{self._alert_id_counter:06d}",
                severity=severity,
                service_name=service_name,
                alert_type=alert_type,
                message=message,
                timestamp=datetime.now(timezone.utc),
                details=details or {}
            )

            self.alerts.append(alert)

            # Trigger handlers
            for handler in self.alert_handlers[severity]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")

            logger.warning(f"Alert created: {alert.alert_id} - {message}")
            return alert

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.acknowledged:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                    return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all unacknowledged alerts."""
        with self._lock:
            return [a for a in self.alerts if not a.acknowledged]

    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]

        if not recent_alerts:
            return {"total_alerts": 0}

        # Calculate statistics
        severity_counts = defaultdict(int)
        service_counts = defaultdict(int)
        acknowledged_count = 0

        for alert in recent_alerts:
            severity_counts[alert.severity.name] += 1
            service_counts[alert.service_name] += 1
            if alert.acknowledged:
                acknowledged_count += 1

        # Calculate mean time to acknowledge (MTTA)
        acknowledged_alerts = [
            a for a in recent_alerts
            if a.acknowledged and a.resolution_time
        ]

        if acknowledged_alerts:
            mtta_seconds = statistics.mean([
                (a.resolution_time - a.timestamp).total_seconds()
                for a in acknowledged_alerts
                if a.resolution_time
            ])
        else:
            mtta_seconds = 0

        return {
            "total_alerts": len(recent_alerts),
            "active_alerts": len([a for a in recent_alerts if not a.acknowledged]),
            "acknowledged_alerts": acknowledged_count,
            "by_severity": dict(severity_counts),
            "by_service": dict(service_counts),
            "mean_time_to_acknowledge_seconds": mtta_seconds,
            "period_hours": hours
        }


class DashboardGenerator:
    """Generate real-time monitoring dashboards."""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        sla_monitor: SLAMonitor,
        alert_manager: AlertManager
    ):
        self.metrics_collector = metrics_collector
        self.sla_monitor = sla_monitor
        self.alert_manager = alert_manager

    def generate_service_dashboard(self, service_name: str) -> Dict[str, Any]:
        """Generate dashboard for specific service."""
        current_metrics = self.metrics_collector.get_current_metrics(service_name)
        if not current_metrics:
            return {"error": f"No metrics available for {service_name}"}

        uptime = self.metrics_collector.calculate_uptime(service_name)
        percentiles = self.metrics_collector.calculate_percentiles(service_name)

        # Get SLA target if exists
        sla_target = self.sla_monitor.sla_targets.get(service_name)

        dashboard = {
            "service_name": service_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health_status": self._determine_health_status(current_metrics, uptime),
            "current_metrics": {
                "uptime_percent": uptime,
                "throughput_rps": current_metrics.throughput_rps,
                "latency_p50_ms": percentiles["p50"],
                "latency_p95_ms": percentiles["p95"],
                "latency_p99_ms": percentiles["p99"],
                "error_rate": (
                    (current_metrics.failed_requests / max(1, current_metrics.total_requests)) * 100
                ),
                "cpu_percent": current_metrics.cpu_percent,
                "memory_mb": current_metrics.memory_mb,
                "active_connections": current_metrics.active_connections
            },
            "sla_compliance": {
                "has_sla": sla_target is not None,
                "service_level": sla_target.service_level.value if sla_target else None,
                "targets": {
                    "uptime": sla_target.uptime_target if sla_target else None,
                    "latency_p50_ms": sla_target.latency_p50_ms if sla_target else None,
                    "latency_p99_ms": sla_target.latency_p99_ms if sla_target else None,
                    "error_rate_threshold": sla_target.error_rate_threshold if sla_target else None
                } if sla_target else {}
            },
            "recent_violations": self._get_recent_violations(service_name),
            "active_alerts": self._get_service_alerts(service_name)
        }

        return dashboard

    def generate_system_dashboard(self) -> Dict[str, Any]:
        """Generate system-wide monitoring dashboard."""
        all_services = list(self.metrics_collector.metrics_buffer.keys())

        service_statuses = {}
        for service in all_services:
            metrics = self.metrics_collector.get_current_metrics(service)
            if metrics:
                uptime = self.metrics_collector.calculate_uptime(service)
                service_statuses[service] = {
                    "status": self._determine_health_status(metrics, uptime).value,
                    "uptime": uptime,
                    "throughput": metrics.throughput_rps
                }

        # Get violation and alert summaries
        violation_summary = self.sla_monitor.get_violation_summary()
        alert_stats = self.alert_manager.get_alert_statistics()

        dashboard = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_health": self._calculate_system_health(service_statuses),
            "total_services": len(all_services),
            "service_statuses": service_statuses,
            "sla_summary": {
                "total_violations_24h": violation_summary.get("total_violations", 0),
                "violations_by_severity": violation_summary.get("by_severity", {}),
                "most_common_violations": violation_summary.get("most_common", [])
            },
            "alert_summary": {
                "active_alerts": alert_stats.get("active_alerts", 0),
                "total_alerts_24h": alert_stats.get("total_alerts", 0),
                "alerts_by_severity": alert_stats.get("by_severity", {}),
                "mtta_seconds": alert_stats.get("mean_time_to_acknowledge_seconds", 0)
            },
            "top_issues": self._identify_top_issues(violation_summary, alert_stats)
        }

        return dashboard

    def _determine_health_status(self, metrics: ServiceMetrics, uptime: float) -> HealthStatus:
        """Determine service health status."""
        error_rate = (metrics.failed_requests / max(1, metrics.total_requests)) * 100

        if uptime < 95 or error_rate > 5:
            return HealthStatus.UNHEALTHY
        elif uptime < 99 or error_rate > 1:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _calculate_system_health(self, service_statuses: Dict[str, Dict]) -> str:
        """Calculate overall system health."""
        if not service_statuses:
            return HealthStatus.UNKNOWN.value

        unhealthy_count = sum(
            1 for s in service_statuses.values()
            if s["status"] == HealthStatus.UNHEALTHY.value
        )
        degraded_count = sum(
            1 for s in service_statuses.values()
            if s["status"] == HealthStatus.DEGRADED.value
        )

        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY.value
        elif degraded_count > len(service_statuses) * 0.3:
            return HealthStatus.DEGRADED.value
        else:
            return HealthStatus.HEALTHY.value

    def _get_recent_violations(self, service_name: str, limit: int = 5) -> List[Dict]:
        """Get recent violations for service."""
        service_violations = [
            v for v in self.sla_monitor.violations
            if v.service_name == service_name
        ][-limit:]

        return [
            {
                "violation_id": v.violation_id,
                "type": v.violation_type,
                "severity": v.severity.name,
                "timestamp": v.timestamp.isoformat(),
                "description": v.impact_description
            }
            for v in service_violations
        ]

    def _get_service_alerts(self, service_name: str) -> List[Dict]:
        """Get active alerts for service."""
        active_alerts = self.alert_manager.get_active_alerts()
        service_alerts = [a for a in active_alerts if a.service_name == service_name]

        return [
            {
                "alert_id": a.alert_id,
                "severity": a.severity.name,
                "type": a.alert_type,
                "message": a.message,
                "timestamp": a.timestamp.isoformat()
            }
            for a in service_alerts
        ]

    def _identify_top_issues(
        self,
        violation_summary: Dict,
        alert_stats: Dict
    ) -> List[Dict[str, Any]]:
        """Identify top system issues."""
        issues = []

        # Add critical violations
        if violation_summary.get("by_severity", {}).get("CRITICAL", 0) > 0:
            issues.append({
                "type": "critical_sla_violations",
                "count": violation_summary["by_severity"]["CRITICAL"],
                "priority": 1
            })

        # Add high alert count
        if alert_stats.get("by_severity", {}).get("CRITICAL", 0) > 0:
            issues.append({
                "type": "critical_alerts",
                "count": alert_stats["by_severity"]["CRITICAL"],
                "priority": 1
            })

        # Add services with violations
        for service, data in violation_summary.get("by_service", {}).items():
            if data["count"] > 5:  # Threshold for concern
                issues.append({
                    "type": "service_violations",
                    "service": service,
                    "count": data["count"],
                    "priority": 2
                })

        # Sort by priority and count
        issues.sort(key=lambda x: (x["priority"], -x.get("count", 0)))

        return issues[:10]  # Top 10 issues


class ProductionMonitor:
    """Main production monitoring system."""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.sla_monitor = SLAMonitor()
        self.alert_manager = AlertManager()
        self.dashboard_generator = DashboardGenerator(
            self.metrics_collector,
            self.sla_monitor,
            self.alert_manager
        )
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    async def start(self):
        """Start production monitoring."""
        self._running = True
        logger.info("Production monitoring started")

    async def stop(self):
        """Stop production monitoring."""
        self._running = False

        # Cancel monitoring tasks
        for task in self._monitoring_tasks.values():
            task.cancel()

        await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)
        logger.info("Production monitoring stopped")

    def register_service(
        self,
        service_name: str,
        sla_target: SLATarget
    ):
        """Register service for monitoring."""
        self.sla_monitor.register_sla(sla_target)
        logger.info(f"Service {service_name} registered for monitoring")

    async def record_request(
        self,
        service_name: str,
        latency_ms: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record individual request metrics."""
        self.metrics_collector.record_latency(service_name, latency_ms)

        # Update request counts (would be aggregated in production)
        current_metrics = self.metrics_collector.get_current_metrics(service_name)
        if current_metrics:
            if success:
                current_metrics.successful_requests += 1
            else:
                current_metrics.failed_requests += 1
            current_metrics.total_requests += 1

    async def check_health(self, service_name: str) -> Dict[str, Any]:
        """Perform health check for service."""
        current_metrics = self.metrics_collector.get_current_metrics(service_name)
        if not current_metrics:
            return {"status": HealthStatus.UNKNOWN.value, "error": "No metrics available"}

        uptime = self.metrics_collector.calculate_uptime(service_name)

        # Check for violations
        violations = self.sla_monitor.check_compliance(
            service_name,
            current_metrics,
            uptime
        )

        # Create alerts for critical violations
        for violation in violations:
            if violation.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                await self.alert_manager.create_alert(
                    severity=violation.severity,
                    service_name=service_name,
                    alert_type="sla_violation",
                    message=violation.impact_description,
                    details={"violation_id": violation.violation_id}
                )

        health_status = self.dashboard_generator._determine_health_status(
            current_metrics,
            uptime
        )

        return {
            "status": health_status.value,
            "uptime_percent": uptime,
            "violations": len(violations),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def get_service_dashboard(self, service_name: str) -> Dict[str, Any]:
        """Get service-specific dashboard."""
        return self.dashboard_generator.generate_service_dashboard(service_name)

    def get_system_dashboard(self) -> Dict[str, Any]:
        """Get system-wide dashboard."""
        return self.dashboard_generator.generate_system_dashboard()


__all__ = [
    "ProductionMonitor",
    "MetricsCollector",
    "SLAMonitor",
    "AlertManager",
    "DashboardGenerator",
    "SLATarget",
    "ServiceMetrics",
    "SLAViolation",
    "Alert",
    "ServiceLevel",
    "AlertSeverity",
    "HealthStatus"
]