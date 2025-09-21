import asyncio
import json
import time
import uuid
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, AsyncGenerator
import threading
from collections import defaultdict, deque
import logging
import traceback
from functools import wraps

logger = logging.getLogger(__name__)

class SpanKind(Enum):
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"

class SpanStatus(Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class SpanContext:
    """Represents the context of a distributed trace span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)

@dataclass
class Span:
    """Represents a single span in a distributed trace"""
    span_id: str
    trace_id: str
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    parent_span_id: Optional[str] = None
    span_kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.OK
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: Optional[float] = None

    def finish(self, status: SpanStatus = SpanStatus.OK):
        """Finish the span"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status

    def set_tag(self, key: str, value: Any):
        """Set a tag on the span"""
        self.tags[key] = value

    def log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span"""
        log_entry = {
            'timestamp': time.time(),
            'message': message,
            'level': level,
            **kwargs
        }
        self.logs.append(log_entry)

    def set_error(self, exception: Exception):
        """Mark span as error with exception details"""
        self.status = SpanStatus.ERROR
        self.set_tag('error', True)
        self.set_tag('error.kind', type(exception).__name__)
        self.set_tag('error.message', str(exception))
        self.log(f"Exception: {str(exception)}", level="error",
                stack_trace=traceback.format_exc())

@dataclass
class Trace:
    """Represents a complete distributed trace"""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    root_span: Optional[Span] = None
    total_duration_ms: Optional[float] = None
    service_map: Dict[str, List[str]] = field(default_factory=dict)

    def add_span(self, span: Span):
        """Add a span to the trace"""
        self.spans.append(span)

        # Set root span if this is the first span or has no parent
        if not self.root_span or not span.parent_span_id:
            self.root_span = span

    def finish(self):
        """Finish the trace"""
        self.end_time = time.time()
        if self.root_span and self.root_span.end_time:
            self.total_duration_ms = (self.end_time - self.start_time) * 1000

    def get_span_tree(self) -> Dict[str, Any]:
        """Get hierarchical representation of spans"""
        span_dict = {span.span_id: span for span in self.spans}
        root_spans = [span for span in self.spans if not span.parent_span_id]

        def build_tree(span: Span) -> Dict[str, Any]:
            children = [s for s in self.spans if s.parent_span_id == span.span_id]
            return {
                'span': asdict(span),
                'children': [build_tree(child) for child in children]
            }

        return {
            'trace_id': self.trace_id,
            'total_duration_ms': self.total_duration_ms,
            'root_spans': [build_tree(span) for span in root_spans]
        }

class TracingBackend:
    """Abstract base class for tracing backends"""

    def export_span(self, span: Span):
        """Export a completed span"""
        raise NotImplementedError

    def export_trace(self, trace: Trace):
        """Export a completed trace"""
        raise NotImplementedError

class InMemoryTracingBackend(TracingBackend):
    """In-memory tracing backend for development and testing"""

    def __init__(self, max_traces: int = 1000):
        self.max_traces = max_traces
        self.traces: Dict[str, Trace] = {}
        self.spans: Dict[str, Span] = {}
        self._lock = threading.RLock()

    def export_span(self, span: Span):
        """Store span in memory"""
        with self._lock:
            self.spans[span.span_id] = span

            # Add to trace
            if span.trace_id not in self.traces:
                self.traces[span.trace_id] = Trace(trace_id=span.trace_id)

            self.traces[span.trace_id].add_span(span)

            # Cleanup old traces if needed
            if len(self.traces) > self.max_traces:
                oldest_trace_id = min(self.traces.keys(),
                                    key=lambda tid: self.traces[tid].start_time)
                del self.traces[oldest_trace_id]

    def export_trace(self, trace: Trace):
        """Store complete trace"""
        with self._lock:
            self.traces[trace.trace_id] = trace

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID"""
        return self.traces.get(trace_id)

    def get_span(self, span_id: str) -> Optional[Span]:
        """Get span by ID"""
        return self.spans.get(span_id)

    def search_traces(self,
                     service: Optional[str] = None,
                     operation: Optional[str] = None,
                     min_duration_ms: Optional[float] = None,
                     max_duration_ms: Optional[float] = None,
                     status: Optional[SpanStatus] = None,
                     limit: int = 100) -> List[Trace]:
        """Search traces by criteria"""
        results = []

        for trace in self.traces.values():
            if len(results) >= limit:
                break

            # Filter by criteria
            if service and not any(service in span.tags.get('service.name', '') for span in trace.spans):
                continue

            if operation and not any(operation in span.operation_name for span in trace.spans):
                continue

            if min_duration_ms and (not trace.total_duration_ms or trace.total_duration_ms < min_duration_ms):
                continue

            if max_duration_ms and (not trace.total_duration_ms or trace.total_duration_ms > max_duration_ms):
                continue

            if status and not any(span.status == status for span in trace.spans):
                continue

            results.append(trace)

        return results

class JaegerTracingBackend(TracingBackend):
    """Jaeger tracing backend for production environments"""

    def __init__(self, jaeger_endpoint: str = "http://localhost:14268/api/traces"):
        self.jaeger_endpoint = jaeger_endpoint
        self._export_queue = asyncio.Queue()
        self._export_task = None

    async def start(self):
        """Start the export background task"""
        if not self._export_task:
            self._export_task = asyncio.create_task(self._export_worker())

    async def stop(self):
        """Stop the export background task"""
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass

    def export_span(self, span: Span):
        """Queue span for export to Jaeger"""
        try:
            self._export_queue.put_nowait(span)
        except asyncio.QueueFull:
            logger.warning("Jaeger export queue full, dropping span")

    def export_trace(self, trace: Trace):
        """Export complete trace to Jaeger"""
        for span in trace.spans:
            self.export_span(span)

    async def _export_worker(self):
        """Background worker to export spans to Jaeger"""
        import aiohttp

        batch_size = 100
        batch_timeout = 5.0  # seconds

        while True:
            try:
                spans_batch = []
                deadline = time.time() + batch_timeout

                # Collect batch of spans
                while len(spans_batch) < batch_size and time.time() < deadline:
                    try:
                        span = await asyncio.wait_for(
                            self._export_queue.get(),
                            timeout=deadline - time.time()
                        )
                        spans_batch.append(span)
                    except asyncio.TimeoutError:
                        break

                if spans_batch:
                    await self._send_spans_to_jaeger(spans_batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Jaeger export worker: {e}")
                await asyncio.sleep(1)

    async def _send_spans_to_jaeger(self, spans: List[Span]):
        """Send spans to Jaeger via HTTP"""
        import aiohttp

        # Convert spans to Jaeger format
        jaeger_data = self._convert_spans_to_jaeger_format(spans)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.jaeger_endpoint,
                    json=jaeger_data,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to send spans to Jaeger: {response.status}")
        except Exception as e:
            logger.error(f"Error sending spans to Jaeger: {e}")

    def _convert_spans_to_jaeger_format(self, spans: List[Span]) -> Dict[str, Any]:
        """Convert spans to Jaeger JSON format"""
        # Group spans by trace
        traces_data = defaultdict(list)

        for span in spans:
            jaeger_span = {
                'traceID': span.trace_id,
                'spanID': span.span_id,
                'parentSpanID': span.parent_span_id,
                'operationName': span.operation_name,
                'startTime': int(span.start_time * 1_000_000),  # microseconds
                'duration': int((span.duration_ms or 0) * 1000),  # microseconds
                'tags': [{'key': k, 'value': str(v)} for k, v in span.tags.items()],
                'logs': [
                    {
                        'timestamp': int(log['timestamp'] * 1_000_000),
                        'fields': [{'key': k, 'value': str(v)} for k, v in log.items() if k != 'timestamp']
                    }
                    for log in span.logs
                ]
            }
            traces_data[span.trace_id].append(jaeger_span)

        # Format for Jaeger
        return {
            'data': [
                {
                    'traceID': trace_id,
                    'spans': spans_list,
                    'processes': {
                        'p1': {
                            'serviceName': 'trading-system',
                            'tags': []
                        }
                    }
                }
                for trace_id, spans_list in traces_data.items()
            ]
        }

class DistributedTracer:
    """Main distributed tracing manager"""

    def __init__(self,
                 service_name: str,
                 backend: Optional[TracingBackend] = None,
                 sample_rate: float = 1.0):
        self.service_name = service_name
        self.backend = backend or InMemoryTracingBackend()
        self.sample_rate = sample_rate

        # Thread-local storage for current span context
        self._local = threading.local()

        # Global context for async operations
        self._async_context: Dict[asyncio.Task, SpanContext] = {}
        self._async_lock = threading.Lock()

        # Metrics
        self.metrics = {
            'traces_created': 0,
            'spans_created': 0,
            'spans_finished': 0,
            'errors': 0
        }

    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled"""
        import random
        return random.random() < self.sample_rate

    def _get_current_context(self) -> Optional[SpanContext]:
        """Get current span context"""
        # Try async context first
        try:
            current_task = asyncio.current_task()
            if current_task:
                with self._async_lock:
                    return self._async_context.get(current_task)
        except RuntimeError:
            pass

        # Fall back to thread-local
        return getattr(self._local, 'context', None)

    def _set_current_context(self, context: Optional[SpanContext]):
        """Set current span context"""
        # Try async context first
        try:
            current_task = asyncio.current_task()
            if current_task:
                with self._async_lock:
                    if context:
                        self._async_context[current_task] = context
                    else:
                        self._async_context.pop(current_task, None)
                return
        except RuntimeError:
            pass

        # Fall back to thread-local
        self._local.context = context

    def start_span(self,
                   operation_name: str,
                   parent_context: Optional[SpanContext] = None,
                   span_kind: SpanKind = SpanKind.INTERNAL,
                   tags: Optional[Dict[str, Any]] = None) -> Span:
        """Start a new span"""
        if not self._should_sample():
            # Return a no-op span for non-sampled traces
            return self._create_noop_span(operation_name)

        # Get parent context
        if not parent_context:
            parent_context = self._get_current_context()

        # Generate IDs
        span_id = str(uuid.uuid4())
        trace_id = parent_context.trace_id if parent_context else str(uuid.uuid4())
        parent_span_id = parent_context.span_id if parent_context else None

        # Create span
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            operation_name=operation_name,
            start_time=time.time(),
            parent_span_id=parent_span_id,
            span_kind=span_kind,
            tags=tags or {}
        )

        # Add service tags
        span.set_tag('service.name', self.service_name)
        span.set_tag('span.kind', span_kind.value)

        # Update metrics
        self.metrics['spans_created'] += 1
        if not parent_context:
            self.metrics['traces_created'] += 1

        return span

    def _create_noop_span(self, operation_name: str) -> Span:
        """Create a no-op span for non-sampled traces"""
        return Span(
            span_id="noop",
            trace_id="noop",
            operation_name=operation_name,
            start_time=time.time()
        )

    def finish_span(self, span: Span, status: SpanStatus = SpanStatus.OK):
        """Finish a span"""
        if span.span_id == "noop":
            return

        span.finish(status)

        # Export to backend
        try:
            self.backend.export_span(span)
            self.metrics['spans_finished'] += 1
        except Exception as e:
            logger.error(f"Error exporting span: {e}")
            self.metrics['errors'] += 1

    @contextmanager
    def trace(self,
              operation_name: str,
              span_kind: SpanKind = SpanKind.INTERNAL,
              tags: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations"""
        span = self.start_span(operation_name, span_kind=span_kind, tags=tags)

        # Set as current context
        old_context = self._get_current_context()
        new_context = SpanContext(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id
        )
        self._set_current_context(new_context)

        try:
            yield span
            self.finish_span(span, SpanStatus.OK)
        except Exception as e:
            span.set_error(e)
            self.finish_span(span, SpanStatus.ERROR)
            raise
        finally:
            self._set_current_context(old_context)

    @asynccontextmanager
    async def trace_async(self,
                         operation_name: str,
                         span_kind: SpanKind = SpanKind.INTERNAL,
                         tags: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Span, None]:
        """Async context manager for tracing operations"""
        span = self.start_span(operation_name, span_kind=span_kind, tags=tags)

        # Set as current context
        old_context = self._get_current_context()
        new_context = SpanContext(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id
        )
        self._set_current_context(new_context)

        try:
            yield span
            self.finish_span(span, SpanStatus.OK)
        except Exception as e:
            span.set_error(e)
            self.finish_span(span, SpanStatus.ERROR)
            raise
        finally:
            self._set_current_context(old_context)

# Global tracer instance
_global_tracer: Optional[DistributedTracer] = None

def initialize_tracing(service_name: str,
                      backend: Optional[TracingBackend] = None,
                      sample_rate: float = 1.0) -> DistributedTracer:
    """Initialize global tracer"""
    global _global_tracer
    _global_tracer = DistributedTracer(service_name, backend, sample_rate)
    return _global_tracer

def get_tracer() -> DistributedTracer:
    """Get global tracer instance"""
    if not _global_tracer:
        raise RuntimeError("Tracer not initialized. Call initialize_tracing() first.")
    return _global_tracer

# Decorators for automatic tracing
def trace_function(operation_name: Optional[str] = None,
                  span_kind: SpanKind = SpanKind.INTERNAL,
                  tags: Optional[Dict[str, Any]] = None):
    """Decorator to automatically trace function calls"""
    def decorator(func):
        op_name = operation_name or f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                async with tracer.trace_async(op_name, span_kind, tags) as span:
                    span.set_tag('function.name', func.__name__)
                    span.set_tag('function.module', func.__module__)
                    if args:
                        span.set_tag('function.args_count', len(args))
                    if kwargs:
                        span.set_tag('function.kwargs_count', len(kwargs))

                    result = await func(*args, **kwargs)
                    span.set_tag('function.completed', True)
                    return result
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.trace(op_name, span_kind, tags) as span:
                    span.set_tag('function.name', func.__name__)
                    span.set_tag('function.module', func.__module__)
                    if args:
                        span.set_tag('function.args_count', len(args))
                    if kwargs:
                        span.set_tag('function.kwargs_count', len(kwargs))

                    result = func(*args, **kwargs)
                    span.set_tag('function.completed', True)
                    return result
            return sync_wrapper

    return decorator

def trace_class_methods(cls):
    """Class decorator to trace all public methods"""
    for attr_name in dir(cls):
        if not attr_name.startswith('_'):
            attr = getattr(cls, attr_name)
            if callable(attr):
                traced_method = trace_function(
                    operation_name=f"{cls.__name__}.{attr_name}",
                    tags={'class.name': cls.__name__}
                )(attr)
                setattr(cls, attr_name, traced_method)
    return cls

# Integration helpers
class TracingMixin:
    """Mixin class to add tracing capabilities to any class"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracer = get_tracer()

    def _trace_operation(self, operation_name: str, **tags):
        """Create a trace context for an operation"""
        class_name = self.__class__.__name__
        full_operation = f"{class_name}.{operation_name}"
        all_tags = {'class.name': class_name, **tags}
        return self._tracer.trace(full_operation, tags=all_tags)

    async def _trace_operation_async(self, operation_name: str, **tags):
        """Create an async trace context for an operation"""
        class_name = self.__class__.__name__
        full_operation = f"{class_name}.{operation_name}"
        all_tags = {'class.name': class_name, **tags}
        return self._tracer.trace_async(full_operation, tags=all_tags)

# Example usage and testing
if __name__ == "__main__":
    # Initialize tracing
    tracer = initialize_tracing("trading-system")

    # Example traced function
    @trace_function("example.compute")
    def compute_something(x: int, y: int) -> int:
        time.sleep(0.1)  # Simulate work
        return x + y

    # Example traced async function
    @trace_function("example.async_compute")
    async def async_compute_something(x: int, y: int) -> int:
        await asyncio.sleep(0.1)  # Simulate async work
        return x * y

    # Example usage
    with tracer.trace("main_operation") as span:
        span.set_tag("user.id", "12345")
        span.log("Starting computation")

        result1 = compute_something(10, 20)
        span.log(f"Sync result: {result1}")

        # Nested span
        with tracer.trace("nested_operation") as nested_span:
            nested_span.set_tag("operation.type", "nested")
            time.sleep(0.05)

    # Print metrics
    print(f"Tracing metrics: {tracer.metrics}")

    # Print traces if using in-memory backend
    if isinstance(tracer.backend, InMemoryTracingBackend):
        for trace_id, trace in tracer.backend.traces.items():
            print(f"Trace {trace_id}: {len(trace.spans)} spans")
            for span in trace.spans:
                print(f"  Span: {span.operation_name} ({span.duration_ms:.2f}ms)")