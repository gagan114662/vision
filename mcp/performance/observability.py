"""
OpenTelemetry observability and performance monitoring system.

Provides distributed tracing, metrics collection, and comprehensive monitoring
for production-grade performance analysis and debugging.
"""
from __future__ import annotations

import asyncio
import logging
import time
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timezone
import json
import psutil
import gc

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics, baggage
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealthMetrics:
    """System health and resource utilization metrics."""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    network_io_bytes: int
    active_threads: int
    open_file_descriptors: int
    gc_collections: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MetricsCollector:
    """High-performance metrics collection and aggregation."""

    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.metrics_buffer: List[PerformanceMetric] = []
        self.system_metrics: List[SystemHealthMetrics] = []
        self._lock = threading.Lock()
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None

        # Metric aggregations
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}

    async def start_collection(self):
        """Start background metrics collection."""
        if self._running:
            return

        self._running = True
        self._collection_task = asyncio.create_task(self._collect_system_metrics())
        logger.info("Metrics collection started")

    async def stop_collection(self):
        """Stop metrics collection."""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collection stopped")

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "count",
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a custom metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            labels=labels or {}
        )

        with self._lock:
            self.metrics_buffer.append(metric)

            # Update aggregations
            if unit == "count":
                self.counters[name] = self.counters.get(name, 0) + value
            else:
                self.gauges[name] = value

            # Keep buffer size manageable
            if len(self.metrics_buffer) > 10000:
                self.metrics_buffer = self.metrics_buffer[-5000:]

    def record_histogram(self, name: str, value: float):
        """Record histogram metric for latency distributions."""
        with self._lock:
            if name not in self.histograms:
                self.histograms[name] = []
            self.histograms[name].append(value)

            # Keep histogram size reasonable
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]

    async def _collect_system_metrics(self):
        """Collect system health metrics periodically."""
        while self._running:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)

                # Memory metrics
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_mb = memory.used / (1024 * 1024)

                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_usage_percent = disk.percent

                # Network metrics
                network = psutil.net_io_counters()
                network_io_bytes = network.bytes_sent + network.bytes_recv

                # Process metrics
                process = psutil.Process()
                active_threads = process.num_threads()
                open_fds = process.num_fds() if hasattr(process, 'num_fds') else 0

                # Garbage collection
                gc_stats = gc.get_stats()
                gc_collections = sum(stat['collections'] for stat in gc_stats)

                health_metric = SystemHealthMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_mb=memory_mb,
                    disk_usage_percent=disk_usage_percent,
                    network_io_bytes=network_io_bytes,
                    active_threads=active_threads,
                    open_file_descriptors=open_fds,
                    gc_collections=gc_collections
                )

                with self._lock:
                    self.system_metrics.append(health_metric)
                    if len(self.system_metrics) > 1000:
                        self.system_metrics = self.system_metrics[-500:]

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"System metrics collection failed: {e}")
                await asyncio.sleep(self.collection_interval)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        with self._lock:
            # Calculate histogram percentiles
            histogram_stats = {}
            for name, values in self.histograms.items():
                if values:
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    histogram_stats[name] = {
                        "count": n,
                        "min": min(sorted_values),
                        "max": max(sorted_values),
                        "mean": sum(sorted_values) / n,
                        "p50": sorted_values[int(n * 0.5)],
                        "p95": sorted_values[int(n * 0.95)],
                        "p99": sorted_values[int(n * 0.99)]
                    }

            # Latest system metrics
            latest_system = self.system_metrics[-1] if self.system_metrics else None

            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": histogram_stats,
                "system_health": {
                    "cpu_percent": latest_system.cpu_percent if latest_system else 0,
                    "memory_percent": latest_system.memory_percent if latest_system else 0,
                    "memory_mb": latest_system.memory_mb if latest_system else 0,
                    "active_threads": latest_system.active_threads if latest_system else 0
                } if latest_system else {},
                "collection_stats": {
                    "total_metrics": len(self.metrics_buffer),
                    "system_metrics_count": len(self.system_metrics)
                }
            }


class DistributedTracer:
    """OpenTelemetry distributed tracing implementation."""

    def __init__(
        self,
        service_name: str = "mcp-trading-system",
        jaeger_endpoint: str = "http://localhost:14268/api/traces"
    ):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.tracer = None
        self.meter = None

        if OTEL_AVAILABLE:
            self._setup_tracing()
        else:
            logger.warning("OpenTelemetry not available, using mock tracer")

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing and metrics."""
        # Configure tracing
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()

        # Jaeger exporter
        jaeger_exporter = JaegerExporter(
            endpoint=self.jaeger_endpoint,
        )

        # Batch span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)

        self.tracer = trace.get_tracer(self.service_name)

        # Configure metrics
        metric_reader = PrometheusMetricReader()
        metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
        self.meter = metrics.get_meter(self.service_name)

        # Auto-instrumentation
        RequestsInstrumentor().instrument()
        AsyncioInstrumentor().instrument()

        logger.info(f"OpenTelemetry tracing configured for {self.service_name}")

    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Create a trace span for an operation."""
        if self.tracer:
            with self.tracer.start_as_current_span(operation_name) as span:
                # Add attributes to span
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
                yield span
        else:
            # Mock implementation when OpenTelemetry is not available
            yield None

    @asynccontextmanager
    async def async_trace_operation(self, operation_name: str, **attributes):
        """Create an async trace span for an operation."""
        if self.tracer:
            with self.tracer.start_as_current_span(operation_name) as span:
                # Add attributes to span
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
                yield span
        else:
            # Mock implementation
            yield None

    def record_exception(self, span, exception: Exception):
        """Record an exception in the current span."""
        if span and OTEL_AVAILABLE:
            span.record_exception(exception)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))


class PerformanceProfiler:
    """Advanced performance profiling and analysis."""

    def __init__(self):
        self.execution_times: Dict[str, List[float]] = {}
        self.function_calls: Dict[str, int] = {}
        self._lock = threading.Lock()

    def profile_function(self, func_name: Optional[str] = None):
        """Decorator for profiling function execution times."""
        def decorator(func: Callable):
            name = func_name or f"{func.__module__}.{func.__name__}"

            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    start_time = time.perf_counter()
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    finally:
                        execution_time = time.perf_counter() - start_time
                        self._record_execution(name, execution_time)
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    start_time = time.perf_counter()
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        execution_time = time.perf_counter() - start_time
                        self._record_execution(name, execution_time)
                return sync_wrapper

        return decorator

    def _record_execution(self, func_name: str, execution_time: float):
        """Record function execution time."""
        with self._lock:
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(execution_time)

            # Keep only recent executions
            if len(self.execution_times[func_name]) > 1000:
                self.execution_times[func_name] = self.execution_times[func_name][-500:]

            self.function_calls[func_name] = self.function_calls.get(func_name, 0) + 1

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self._lock:
            report = {}

            for func_name, times in self.execution_times.items():
                if times:
                    sorted_times = sorted(times)
                    n = len(sorted_times)

                    report[func_name] = {
                        "call_count": self.function_calls.get(func_name, 0),
                        "total_time": sum(sorted_times),
                        "average_time": sum(sorted_times) / n,
                        "min_time": min(sorted_times),
                        "max_time": max(sorted_times),
                        "p50_time": sorted_times[int(n * 0.5)],
                        "p95_time": sorted_times[int(n * 0.95)],
                        "p99_time": sorted_times[int(n * 0.99)]
                    }

            return report


class ObservabilityManager:
    """Centralized observability and monitoring manager."""

    def __init__(
        self,
        service_name: str = "mcp-trading-system",
        enable_tracing: bool = True,
        enable_metrics: bool = True,
        enable_profiling: bool = True
    ):
        self.service_name = service_name
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics
        self.enable_profiling = enable_profiling

        # Initialize components
        self.metrics_collector = MetricsCollector() if enable_metrics else None
        self.tracer = DistributedTracer(service_name) if enable_tracing else None
        self.profiler = PerformanceProfiler() if enable_profiling else None

        # Performance tracking
        self.start_time = time.time()
        self.request_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}

    async def start(self):
        """Start observability collection."""
        if self.metrics_collector:
            await self.metrics_collector.start_collection()
        logger.info("Observability manager started")

    async def stop(self):
        """Stop observability collection."""
        if self.metrics_collector:
            await self.metrics_collector.stop_collection()
        logger.info("Observability manager stopped")

    def record_request(self, endpoint: str, duration: float, success: bool = True):
        """Record API request metrics."""
        if self.metrics_collector:
            self.metrics_collector.record_metric(f"request_count_{endpoint}", 1, "count")
            self.metrics_collector.record_metric(f"request_duration_{endpoint}", duration, "seconds")
            self.metrics_collector.record_histogram(f"latency_{endpoint}", duration * 1000)  # ms

        self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1

        if not success:
            self.error_counts[endpoint] = self.error_counts.get(endpoint, 0) + 1

    def trace_operation(self, operation_name: str, **attributes):
        """Create trace span for operation."""
        if self.tracer:
            return self.tracer.trace_operation(operation_name, **attributes)
        else:
            return contextmanager(lambda: (yield None))()

    def async_trace_operation(self, operation_name: str, **attributes):
        """Create async trace span for operation."""
        if self.tracer:
            return self.tracer.async_trace_operation(operation_name, **attributes)
        else:
            return asynccontextmanager(lambda: (yield None))()

    def profile_function(self, func_name: Optional[str] = None):
        """Profile function performance."""
        if self.profiler:
            return self.profiler.profile_function(func_name)
        else:
            # No-op decorator
            def decorator(func):
                return func
            return decorator

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        uptime = time.time() - self.start_time

        health_data = {
            "status": "healthy",
            "uptime_seconds": uptime,
            "service_name": self.service_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "metrics_collector": self.metrics_collector is not None,
                "distributed_tracing": self.tracer is not None,
                "performance_profiler": self.profiler is not None
            }
        }

        # Add metrics summary
        if self.metrics_collector:
            health_data["metrics"] = self.metrics_collector.get_metrics_summary()

        # Add performance report
        if self.profiler:
            health_data["performance"] = self.profiler.get_performance_report()

        # Add request statistics
        health_data["requests"] = {
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values()),
            "error_rate": (
                sum(self.error_counts.values()) / max(1, sum(self.request_counts.values()))
            ),
            "endpoints": dict(self.request_counts)
        }

        return health_data


# Global observability instance
observability_manager = ObservabilityManager()


def get_observability_manager() -> ObservabilityManager:
    """Get the global observability manager instance."""
    return observability_manager


__all__ = [
    "ObservabilityManager",
    "MetricsCollector",
    "DistributedTracer",
    "PerformanceProfiler",
    "PerformanceMetric",
    "SystemHealthMetrics",
    "get_observability_manager"
]