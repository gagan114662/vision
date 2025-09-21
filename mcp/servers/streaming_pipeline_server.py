"""Real-time streaming pipeline MCP server.

Provides real-time data streaming capabilities for market data, signals,
and trading events with WebSocket support and event-driven processing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Callable, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque

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


class StreamType(Enum):
    """Types of data streams."""
    MARKET_DATA = "market_data"
    TRADING_SIGNALS = "trading_signals"
    PORTFOLIO_UPDATES = "portfolio_updates"
    RISK_ALERTS = "risk_alerts"
    COMPLIANCE_EVENTS = "compliance_events"
    SYSTEM_METRICS = "system_metrics"


class StreamQuality(Enum):
    """Stream quality levels."""
    REAL_TIME = "real_time"      # < 100ms latency
    NEAR_REAL_TIME = "near_real_time"  # < 1s latency
    DELAYED = "delayed"          # > 1s latency


@dataclass
class StreamEvent:
    """Individual stream event."""
    event_id: str
    stream_type: StreamType
    timestamp: datetime
    data: Dict[str, Any]
    source: str
    quality: StreamQuality
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StreamSubscription:
    """Stream subscription configuration."""
    subscription_id: str
    stream_type: StreamType
    filters: Dict[str, Any]
    callback_url: Optional[str] = None
    max_events_per_second: int = 100
    buffer_size: int = 1000
    active: bool = True
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class StreamBuffer:
    """Thread-safe circular buffer for stream events."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self._total_events = 0
        self._dropped_events = 0

    def add_event(self, event: StreamEvent) -> bool:
        """Add event to buffer. Returns True if added, False if dropped."""
        with self.lock:
            if len(self.buffer) >= self.max_size:
                self._dropped_events += 1
                return False

            self.buffer.append(event)
            self._total_events += 1
            return True

    def get_events(self, max_count: int = None) -> List[StreamEvent]:
        """Get events from buffer (FIFO)."""
        with self.lock:
            if max_count is None:
                events = list(self.buffer)
                self.buffer.clear()
            else:
                events = []
                for _ in range(min(max_count, len(self.buffer))):
                    if self.buffer:
                        events.append(self.buffer.popleft())

            return events

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.lock:
            return {
                "current_size": len(self.buffer),
                "max_size": self.max_size,
                "total_events": self._total_events,
                "dropped_events": self._dropped_events,
                "utilization": len(self.buffer) / self.max_size
            }


class StreamingPipeline:
    """Real-time streaming pipeline manager."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.pipeline_id = str(uuid.uuid4())

        # Stream management
        self._subscriptions: Dict[str, StreamSubscription] = {}
        self._stream_buffers: Dict[StreamType, StreamBuffer] = {}
        self._event_handlers: Dict[StreamType, List[Callable]] = defaultdict(list)

        # Performance tracking
        self._metrics = {
            "events_processed": 0,
            "events_dropped": 0,
            "subscriptions_active": 0,
            "avg_latency_ms": 0.0,
            "throughput_events_per_sec": 0.0
        }

        # Rate limiting
        self._rate_limiters: Dict[str, deque] = defaultdict(lambda: deque())

        # Initialize stream buffers
        for stream_type in StreamType:
            buffer_size = self.config.get(f"{stream_type.value}_buffer_size", 1000)
            self._stream_buffers[stream_type] = StreamBuffer(buffer_size)

        logger.info(f"Streaming pipeline initialized: {self.pipeline_id}")

    async def create_subscription(
        self,
        stream_type: StreamType,
        filters: Dict[str, Any],
        **kwargs
    ) -> str:
        """Create a new stream subscription."""
        subscription_id = str(uuid.uuid4())

        subscription = StreamSubscription(
            subscription_id=subscription_id,
            stream_type=stream_type,
            filters=filters,
            **kwargs
        )

        self._subscriptions[subscription_id] = subscription
        self._metrics["subscriptions_active"] = len(
            [s for s in self._subscriptions.values() if s.active]
        )

        logger.info(f"Created subscription {subscription_id} for {stream_type.value}")
        return subscription_id

    async def publish_event(
        self,
        stream_type: StreamType,
        data: Dict[str, Any],
        source: str,
        quality: StreamQuality = StreamQuality.REAL_TIME
    ) -> bool:
        """Publish an event to the streaming pipeline."""
        event = StreamEvent(
            event_id=str(uuid.uuid4()),
            stream_type=stream_type,
            timestamp=datetime.now(timezone.utc),
            data=data,
            source=source,
            quality=quality
        )

        # Add to stream buffer
        buffer = self._stream_buffers[stream_type]
        added = buffer.add_event(event)

        if added:
            self._metrics["events_processed"] += 1

            # Call event handlers
            for handler in self._event_handlers[stream_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")

            # Check subscriptions and deliver events
            await self._deliver_to_subscriptions(event)

        else:
            self._metrics["events_dropped"] += 1
            logger.warning(f"Event dropped for {stream_type.value} - buffer full")

        return added

    async def _deliver_to_subscriptions(self, event: StreamEvent) -> None:
        """Deliver event to matching subscriptions."""
        matching_subs = [
            sub for sub in self._subscriptions.values()
            if sub.stream_type == event.stream_type and sub.active
        ]

        for subscription in matching_subs:
            # Check rate limiting
            if not self._check_rate_limit(subscription):
                continue

            # Apply filters
            if not self._match_filters(event, subscription.filters):
                continue

            # Deliver event (in production would use WebSocket/HTTP callback)
            logger.debug(f"Delivering event {event.event_id} to {subscription.subscription_id}")

    def _check_rate_limit(self, subscription: StreamSubscription) -> bool:
        """Check if subscription is within rate limits."""
        now = time.time()
        sub_id = subscription.subscription_id
        rate_limiter = self._rate_limiters[sub_id]

        # Remove old timestamps (older than 1 second)
        while rate_limiter and now - rate_limiter[0] > 1.0:
            rate_limiter.popleft()

        # Check if under limit
        if len(rate_limiter) >= subscription.max_events_per_second:
            return False

        # Record this event
        rate_limiter.append(now)
        return True

    def _match_filters(self, event: StreamEvent, filters: Dict[str, Any]) -> bool:
        """Check if event matches subscription filters."""
        for filter_key, filter_value in filters.items():
            if filter_key == "symbols":
                # Symbol-based filtering
                event_symbol = event.data.get("symbol")
                if event_symbol not in filter_value:
                    return False

            elif filter_key == "min_confidence":
                # Confidence-based filtering for signals
                event_confidence = event.data.get("confidence", 0)
                if event_confidence < filter_value:
                    return False

            elif filter_key == "alert_level":
                # Alert level filtering
                event_level = event.data.get("level", "info")
                allowed_levels = filter_value if isinstance(filter_value, list) else [filter_value]
                if event_level not in allowed_levels:
                    return False

        return True

    async def get_stream_events(
        self,
        stream_type: StreamType,
        max_count: int = 100
    ) -> List[StreamEvent]:
        """Get events from a stream buffer."""
        buffer = self._stream_buffers[stream_type]
        return buffer.get_events(max_count)

    def get_subscription(self, subscription_id: str) -> Optional[StreamSubscription]:
        """Get subscription by ID."""
        return self._subscriptions.get(subscription_id)

    async def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a stream subscription."""
        if subscription_id in self._subscriptions:
            self._subscriptions[subscription_id].active = False
            self._metrics["subscriptions_active"] = len(
                [s for s in self._subscriptions.values() if s.active]
            )
            logger.info(f"Cancelled subscription {subscription_id}")
            return True
        return False

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        # Calculate throughput
        total_events = sum(
            buffer.get_stats()["total_events"]
            for buffer in self._stream_buffers.values()
        )

        # Get buffer statistics
        buffer_stats = {}
        for stream_type, buffer in self._stream_buffers.items():
            buffer_stats[stream_type.value] = buffer.get_stats()

        return {
            "pipeline_id": self.pipeline_id,
            "metrics": self._metrics,
            "buffer_stats": buffer_stats,
            "active_subscriptions": len([s for s in self._subscriptions.values() if s.active]),
            "total_subscriptions": len(self._subscriptions)
        }

    def add_event_handler(
        self,
        stream_type: StreamType,
        handler: Callable[[StreamEvent], None]
    ) -> None:
        """Add an event handler for a stream type."""
        self._event_handlers[stream_type].append(handler)
        logger.info(f"Added event handler for {stream_type.value}")


# Global pipeline instance
_streaming_pipeline = None


def get_pipeline() -> StreamingPipeline:
    """Get or create global streaming pipeline instance."""
    global _streaming_pipeline
    if _streaming_pipeline is None:
        _streaming_pipeline = StreamingPipeline()
    return _streaming_pipeline


@register_tool(
    name="streaming.subscription.create",
    schema="./schemas/tool.streaming.subscription.create.schema.json"
)
@circuit_breaker(
    config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_seconds=30.0,
        expected_exception=Exception
    )
)
async def create_stream_subscription(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new stream subscription."""
    try:
        stream_type_str = params["stream_type"]
        filters = params.get("filters", {})
        max_events_per_second = params.get("max_events_per_second", 100)
        buffer_size = params.get("buffer_size", 1000)

        # Validate stream type
        try:
            stream_type = StreamType(stream_type_str)
        except ValueError:
            return {
                "error": f"Invalid stream type: {stream_type_str}",
                "valid_types": [t.value for t in StreamType]
            }

        # Create subscription
        pipeline = get_pipeline()
        subscription_id = await pipeline.create_subscription(
            stream_type=stream_type,
            filters=filters,
            max_events_per_second=max_events_per_second,
            buffer_size=buffer_size
        )

        return {
            "success": True,
            "subscription_id": subscription_id,
            "stream_type": stream_type.value,
            "filters": filters,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Create subscription error: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@register_tool(
    name="streaming.events.publish",
    schema="./schemas/tool.streaming.events.publish.schema.json"
)
@circuit_breaker(
    config=CircuitBreakerConfig(
        failure_threshold=10,
        recovery_timeout_seconds=15.0,
        expected_exception=Exception
    )
)
async def publish_stream_event(params: Dict[str, Any]) -> Dict[str, Any]:
    """Publish an event to the streaming pipeline."""
    try:
        stream_type_str = params["stream_type"]
        data = params["data"]
        source = params.get("source", "unknown")
        quality_str = params.get("quality", "real_time")

        # Validate stream type
        try:
            stream_type = StreamType(stream_type_str)
        except ValueError:
            return {
                "error": f"Invalid stream type: {stream_type_str}",
                "valid_types": [t.value for t in StreamType]
            }

        # Validate quality
        try:
            quality = StreamQuality(quality_str)
        except ValueError:
            return {
                "error": f"Invalid quality: {quality_str}",
                "valid_qualities": [q.value for q in StreamQuality]
            }

        # Publish event
        pipeline = get_pipeline()
        success = await pipeline.publish_event(
            stream_type=stream_type,
            data=data,
            source=source,
            quality=quality
        )

        return {
            "success": success,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "stream_type": stream_type.value,
            "quality": quality.value
        }

    except Exception as e:
        logger.error(f"Publish event error: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@register_tool(
    name="streaming.events.get",
    schema="./schemas/tool.streaming.events.get.schema.json"
)
@circuit_breaker(
    config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_seconds=30.0,
        expected_exception=Exception
    )
)
async def get_stream_events(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get events from a stream."""
    try:
        stream_type_str = params["stream_type"]
        max_count = params.get("max_count", 100)

        # Validate stream type
        try:
            stream_type = StreamType(stream_type_str)
        except ValueError:
            return {
                "error": f"Invalid stream type: {stream_type_str}",
                "valid_types": [t.value for t in StreamType]
            }

        # Get events
        pipeline = get_pipeline()
        events = await pipeline.get_stream_events(stream_type, max_count)

        # Convert events to dict format
        events_data = []
        for event in events:
            event_dict = asdict(event)
            event_dict["timestamp"] = event.timestamp.isoformat()
            event_dict["stream_type"] = event.stream_type.value
            event_dict["quality"] = event.quality.value
            events_data.append(event_dict)

        return {
            "success": True,
            "stream_type": stream_type.value,
            "events": events_data,
            "count": len(events_data),
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Get events error: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@register_tool(
    name="streaming.pipeline.metrics",
    schema="./schemas/tool.streaming.pipeline.metrics.schema.json"
)
@circuit_breaker(
    config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_seconds=30.0,
        expected_exception=Exception
    )
)
async def get_pipeline_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get streaming pipeline metrics and statistics."""
    try:
        pipeline = get_pipeline()
        metrics = pipeline.get_pipeline_metrics()

        return {
            "success": True,
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Get metrics error: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@register_tool(
    name="streaming.subscription.cancel",
    schema="./schemas/tool.streaming.subscription.cancel.schema.json"
)
@circuit_breaker(
    config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_seconds=30.0,
        expected_exception=Exception
    )
)
async def cancel_stream_subscription(params: Dict[str, Any]) -> Dict[str, Any]:
    """Cancel a stream subscription."""
    try:
        subscription_id = params["subscription_id"]

        pipeline = get_pipeline()
        success = await pipeline.cancel_subscription(subscription_id)

        return {
            "success": success,
            "subscription_id": subscription_id,
            "cancelled_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Cancel subscription error: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Integration with market data pipeline
async def setup_market_data_streaming():
    """Setup streaming integration with market data server."""
    pipeline = get_pipeline()

    # Add handler for market data events
    async def market_data_handler(event: StreamEvent):
        """Handle market data events."""
        if event.stream_type == StreamType.MARKET_DATA:
            symbol = event.data.get("symbol")
            price = event.data.get("price")
            logger.debug(f"Market data: {symbol} @ ${price}")

    pipeline.add_event_handler(StreamType.MARKET_DATA, market_data_handler)


# Integration with trading agents
async def setup_trading_signals_streaming():
    """Setup streaming integration with trading agents."""
    pipeline = get_pipeline()

    # Add handler for trading signals
    async def signal_handler(event: StreamEvent):
        """Handle trading signal events."""
        if event.stream_type == StreamType.TRADING_SIGNALS:
            symbol = event.data.get("symbol")
            signal = event.data.get("signal")
            confidence = event.data.get("confidence")
            logger.info(f"Trading signal: {symbol} {signal} (confidence: {confidence:.2f})")

    pipeline.add_event_handler(StreamType.TRADING_SIGNALS, signal_handler)


__all__ = [
    "StreamingPipeline",
    "StreamType",
    "StreamQuality",
    "StreamEvent",
    "StreamSubscription",
    "get_pipeline",
    "create_stream_subscription",
    "publish_stream_event",
    "get_stream_events",
    "get_pipeline_metrics",
    "cancel_stream_subscription"
]