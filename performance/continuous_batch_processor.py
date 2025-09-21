import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
import threading
from collections import deque
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

class BatchPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class BatchItem(Generic[T]):
    data: T
    priority: BatchPriority
    timestamp: float
    timeout: Optional[float] = None
    callback: Optional[Callable[[Any], None]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class BatchConfig:
    max_batch_size: int = 32
    max_wait_time: float = 0.1  # 100ms
    min_batch_size: int = 1
    priority_boost_factor: float = 2.0
    timeout_seconds: float = 30.0

class ContinuousBatchProcessor(Generic[T, R]):
    """
    High-performance continuous batching processor that dynamically groups
    requests for efficient parallel processing with priority handling.
    """

    def __init__(
        self,
        batch_function: Callable[[List[T]], List[R]],
        config: Optional[BatchConfig] = None,
        max_workers: int = 4
    ):
        self.batch_function = batch_function
        self.config = config or BatchConfig()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self._queue: deque = deque()
        self._queue_lock = threading.RLock()
        self._processing = False
        self._processor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Performance metrics
        self._metrics = {
            'total_processed': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'avg_processing_time': 0.0,
            'queue_size_history': deque(maxlen=1000)
        }
        self._metrics_lock = threading.Lock()

        self.start()

    def start(self):
        """Start the continuous batch processor"""
        if not self._processing:
            self._processing = True
            self._shutdown_event.clear()
            self._processor_thread = threading.Thread(
                target=self._process_continuously,
                daemon=True
            )
            self._processor_thread.start()
            logger.info("Continuous batch processor started")

    def stop(self):
        """Stop the processor and wait for completion"""
        self._processing = False
        self._shutdown_event.set()
        if self._processor_thread:
            self._processor_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)
        logger.info("Continuous batch processor stopped")

    def submit(
        self,
        data: T,
        priority: BatchPriority = BatchPriority.MEDIUM,
        timeout: Optional[float] = None,
        callback: Optional[Callable[[R], None]] = None
    ) -> asyncio.Future:
        """Submit an item for batch processing"""
        future = asyncio.Future()
        item = BatchItem(
            data=data,
            priority=priority,
            timestamp=time.time(),
            timeout=timeout or self.config.timeout_seconds,
            callback=callback
        )

        with self._queue_lock:
            # Insert based on priority (higher priority items go first)
            inserted = False
            for i, existing_item in enumerate(self._queue):
                if item.priority.value > existing_item.priority.value:
                    self._queue.insert(i, (item, future))
                    inserted = True
                    break

            if not inserted:
                self._queue.append((item, future))

            # Update metrics
            with self._metrics_lock:
                self._metrics['queue_size_history'].append(len(self._queue))

        return future

    def _process_continuously(self):
        """Main processing loop that runs continuously"""
        while self._processing and not self._shutdown_event.is_set():
            try:
                batch = self._collect_batch()
                if batch:
                    self._process_batch(batch)
                else:
                    # No items to process, short sleep
                    time.sleep(0.001)  # 1ms
            except Exception as e:
                logger.error(f"Error in continuous batch processing: {e}")
                time.sleep(0.01)  # 10ms on error

    def _collect_batch(self) -> Optional[List[tuple]]:
        """Collect a batch of items based on configuration"""
        with self._queue_lock:
            if not self._queue:
                return None

            batch = []
            current_time = time.time()
            oldest_item_time = None

            # Collect items for batch
            while (
                self._queue and
                len(batch) < self.config.max_batch_size
            ):
                item, future = self._queue[0]

                # Check if item has timed out
                if current_time - item.timestamp > item.timeout:
                    self._queue.popleft()
                    if not future.done():
                        future.set_exception(TimeoutError("Batch item timed out"))
                    continue

                # Track oldest item time
                if oldest_item_time is None:
                    oldest_item_time = item.timestamp

                # Check if we should wait for more items
                wait_time = current_time - oldest_item_time
                if (
                    len(batch) >= self.config.min_batch_size and
                    wait_time < self._get_dynamic_wait_time(item.priority)
                ):
                    break

                # Add item to batch
                self._queue.popleft()
                batch.append((item, future))

            return batch if batch else None

    def _get_dynamic_wait_time(self, priority: BatchPriority) -> float:
        """Calculate dynamic wait time based on priority and queue state"""
        base_wait = self.config.max_wait_time

        # Reduce wait time for higher priority items
        priority_multiplier = {
            BatchPriority.CRITICAL: 0.1,
            BatchPriority.HIGH: 0.3,
            BatchPriority.MEDIUM: 1.0,
            BatchPriority.LOW: 2.0
        }

        # Adjust based on queue size (less wait time when queue is large)
        with self._queue_lock:
            queue_size = len(self._queue)

        queue_factor = max(0.1, 1.0 - (queue_size / (self.config.max_batch_size * 2)))

        return base_wait * priority_multiplier.get(priority, 1.0) * queue_factor

    def _process_batch(self, batch: List[tuple]):
        """Process a collected batch of items"""
        if not batch:
            return

        start_time = time.time()
        items = [item for item, _ in batch]
        futures = [future for _, future in batch]

        try:
            # Extract data for batch processing
            batch_data = [item.data for item in items]

            # Process batch
            results = self.batch_function(batch_data)

            # Distribute results to futures
            for i, (item, future, result) in enumerate(zip(items, futures, results)):
                if not future.done():
                    future.set_result(result)

                # Execute callback if provided
                if item.callback:
                    try:
                        item.callback(result)
                    except Exception as e:
                        logger.warning(f"Callback execution failed: {e}")

            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(len(batch), processing_time)

            logger.debug(f"Processed batch of {len(batch)} items in {processing_time:.3f}s")

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set exception for all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    def _update_metrics(self, batch_size: int, processing_time: float):
        """Update performance metrics"""
        with self._metrics_lock:
            self._metrics['total_processed'] += batch_size
            self._metrics['total_batches'] += 1

            # Update rolling averages
            total_batches = self._metrics['total_batches']
            self._metrics['avg_batch_size'] = (
                (self._metrics['avg_batch_size'] * (total_batches - 1) + batch_size) / total_batches
            )
            self._metrics['avg_processing_time'] = (
                (self._metrics['avg_processing_time'] * (total_batches - 1) + processing_time) / total_batches
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self._metrics_lock:
            current_queue_size = len(self._queue)
            return {
                **self._metrics.copy(),
                'current_queue_size': current_queue_size,
                'processing': self._processing
            }

    def get_queue_status(self) -> Dict[str, Any]:
        """Get detailed queue status"""
        with self._queue_lock:
            priority_counts = {p.name: 0 for p in BatchPriority}
            oldest_timestamp = None

            for item, _ in self._queue:
                priority_counts[item.priority.name] += 1
                if oldest_timestamp is None or item.timestamp < oldest_timestamp:
                    oldest_timestamp = item.timestamp

            return {
                'total_items': len(self._queue),
                'priority_breakdown': priority_counts,
                'oldest_item_age': time.time() - oldest_timestamp if oldest_timestamp else 0,
                'processing': self._processing
            }

class AdaptiveBatchProcessor(ContinuousBatchProcessor[T, R]):
    """
    Enhanced batch processor that adapts parameters based on performance metrics
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adaptation_interval = 10.0  # seconds
        self._last_adaptation = time.time()
        self._performance_history = deque(maxlen=100)

    def _process_batch(self, batch: List[tuple]):
        """Enhanced batch processing with performance tracking"""
        start_time = time.time()
        super()._process_batch(batch)
        processing_time = time.time() - start_time

        # Track performance for adaptation
        self._performance_history.append({
            'batch_size': len(batch),
            'processing_time': processing_time,
            'throughput': len(batch) / processing_time if processing_time > 0 else 0
        })

        # Adapt parameters if needed
        if time.time() - self._last_adaptation > self._adaptation_interval:
            self._adapt_parameters()
            self._last_adaptation = time.time()

    def _adapt_parameters(self):
        """Adapt batch processing parameters based on performance history"""
        if len(self._performance_history) < 10:
            return

        recent_performance = list(self._performance_history)[-10:]
        avg_throughput = sum(p['throughput'] for p in recent_performance) / len(recent_performance)
        avg_batch_size = sum(p['batch_size'] for p in recent_performance) / len(recent_performance)

        # Adapt batch size based on throughput
        if avg_throughput > 100:  # High throughput - increase batch size
            self.config.max_batch_size = min(128, int(self.config.max_batch_size * 1.1))
        elif avg_throughput < 20:  # Low throughput - decrease batch size
            self.config.max_batch_size = max(8, int(self.config.max_batch_size * 0.9))

        # Adapt wait time based on queue pressure
        with self._queue_lock:
            queue_size = len(self._queue)

        if queue_size > self.config.max_batch_size * 2:
            self.config.max_wait_time = max(0.01, self.config.max_wait_time * 0.9)
        elif queue_size < self.config.max_batch_size:
            self.config.max_wait_time = min(0.5, self.config.max_wait_time * 1.1)

        logger.debug(f"Adapted parameters: batch_size={self.config.max_batch_size}, wait_time={self.config.max_wait_time:.3f}")