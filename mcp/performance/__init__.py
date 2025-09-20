"""
Performance optimization modules for MCP servers.

Provides continuous batching, caching, streaming, parallel execution,
and comprehensive observability for dramatic throughput improvements
and latency reductions.
"""

from .continuous_batching import (
    ContinuousBatchingEngine,
    ToolRequest,
    ToolBatch,
    BatchingMetrics,
    BatchStatus,
    SchedulingStrategy
)

from .streaming import (
    RealTimeStreamingPipeline,
    StreamingDataManager,
    StreamEvent,
    StreamEventType,
    ProcessingPriority,
    QuestDBIntegration,
    MarketDataProcessor
)

from .parallel_execution import (
    ParallelExecutionEngine,
    OptimizedAsyncProcessor,
    WorkTask,
    ExecutionStrategy,
    WorkloadType,
    ExecutionMetrics,
    AdaptiveWorkloadScheduler
)

from .observability import (
    ObservabilityManager,
    MetricsCollector,
    DistributedTracer,
    PerformanceProfiler,
    PerformanceMetric,
    SystemHealthMetrics,
    get_observability_manager
)

__all__ = [
    # Continuous Batching
    "ContinuousBatchingEngine",
    "ToolRequest",
    "ToolBatch",
    "BatchingMetrics",
    "BatchStatus",
    "SchedulingStrategy",

    # Real-time Streaming
    "RealTimeStreamingPipeline",
    "StreamingDataManager",
    "StreamEvent",
    "StreamEventType",
    "ProcessingPriority",
    "QuestDBIntegration",
    "MarketDataProcessor",

    # Parallel Execution
    "ParallelExecutionEngine",
    "OptimizedAsyncProcessor",
    "WorkTask",
    "ExecutionStrategy",
    "WorkloadType",
    "ExecutionMetrics",
    "AdaptiveWorkloadScheduler",

    # Observability
    "ObservabilityManager",
    "MetricsCollector",
    "DistributedTracer",
    "PerformanceProfiler",
    "PerformanceMetric",
    "SystemHealthMetrics",
    "get_observability_manager"
]