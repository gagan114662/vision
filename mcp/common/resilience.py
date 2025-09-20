"""
Resilience and reliability patterns for MCP servers.

Implements circuit breakers, exponential backoff, timeout management,
and comprehensive error handling following software engineering best practices.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, Type, TypeVar, Union,
    runtime_checkable, Awaitable, Generator, AsyncGenerator
)
import threading
from datetime import datetime, timedelta

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')
CallableT = TypeVar('CallableT', bound=Callable[..., Any])


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Circuit is open, calls fail fast
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 60.0
    expected_exception: Type[Exception] = Exception
    success_threshold: int = 1  # Successful calls needed to close from half-open


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    def __init__(self, name: str, last_failure: Optional[str] = None):
        self.name = name
        self.last_failure = last_failure
        super().__init__(f"Circuit breaker '{name}' is OPEN. Last failure: {last_failure}")


@runtime_checkable
class RetryableError(Protocol):
    """Protocol for errors that should trigger retries."""
    should_retry: bool


class CircuitBreaker:
    """Circuit breaker implementation with comprehensive monitoring."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = threading.RLock()
        self._last_exception: Optional[str] = None

    def _log_state_change(self, old_state: CircuitState, new_state: CircuitState):
        """Log state transitions for observability."""
        logger.warning(
            f"Circuit breaker '{self.name}' state change: {old_state.value} -> {new_state.value}"
        )

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from OPEN to HALF_OPEN."""
        if self.stats.last_failure_time is None:
            return False
        return (time.time() - self.stats.last_failure_time) >= self.config.recovery_timeout_seconds

    def _record_success(self):
        """Record successful call."""
        with self._lock:
            old_state = self.stats.state
            self.stats.success_count += 1
            self.stats.total_successes += 1

            if self.stats.state == CircuitState.HALF_OPEN:
                if self.stats.success_count >= self.config.success_threshold:
                    self.stats.state = CircuitState.CLOSED
                    self.stats.failure_count = 0
                    self.stats.success_count = 0
                    self._log_state_change(old_state, self.stats.state)

    def _record_failure(self, exception: Exception):
        """Record failed call."""
        with self._lock:
            old_state = self.stats.state
            self.stats.failure_count += 1
            self.stats.total_failures += 1
            self.stats.last_failure_time = time.time()
            self._last_exception = str(exception)

            if self.stats.state == CircuitState.CLOSED:
                if self.stats.failure_count >= self.config.failure_threshold:
                    self.stats.state = CircuitState.OPEN
                    self._log_state_change(old_state, self.stats.state)
            elif self.stats.state == CircuitState.HALF_OPEN:
                self.stats.state = CircuitState.OPEN
                self._log_state_change(old_state, self.stats.state)

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self.stats.total_calls += 1

            if self.stats.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.stats.state = CircuitState.HALF_OPEN
                    self.stats.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' attempting recovery (HALF_OPEN)")
                else:
                    raise CircuitBreakerOpenError(self.name, self._last_exception)

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.expected_exception as e:
            self._record_failure(e)
            raise

    async def call_async(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute async function with circuit breaker protection."""
        with self._lock:
            self.stats.total_calls += 1

            if self.stats.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.stats.state = CircuitState.HALF_OPEN
                    self.stats.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' attempting recovery (HALF_OPEN)")
                else:
                    raise CircuitBreakerOpenError(self.name, self._last_exception)

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.expected_exception as e:
            self._record_failure(e)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self._lock:
            failure_rate = (
                self.stats.total_failures / max(1, self.stats.total_calls)
            ) * 100

            return {
                "name": self.name,
                "state": self.stats.state.value,
                "failure_count": self.stats.failure_count,
                "success_count": self.stats.success_count,
                "total_calls": self.stats.total_calls,
                "total_failures": self.stats.total_failures,
                "total_successes": self.stats.total_successes,
                "failure_rate_percent": round(failure_rate, 2),
                "last_failure_time": self.stats.last_failure_time,
                "last_exception": self._last_exception,
            }


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retriable_exceptions: tuple[Type[Exception], ...] = (Exception,)


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Retry exhausted after {attempts} attempts. Last error: {last_exception}")


def calculate_backoff_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate exponential backoff delay with jitter."""
    delay = min(
        config.initial_delay * (config.exponential_base ** attempt),
        config.max_delay
    )

    if config.jitter:
        import random
        delay *= (0.5 + random.random() * 0.5)  # Jitter between 50%-100%

    return delay


def retry(config: Optional[RetryConfig] = None):
    """Decorator for adding retry logic with exponential backoff."""
    retry_config = config or RetryConfig()

    def decorator(func: CallableT) -> CallableT:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception = None

                for attempt in range(retry_config.max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except retry_config.retriable_exceptions as e:
                        last_exception = e
                        if attempt == retry_config.max_attempts - 1:
                            break

                        delay = calculate_backoff_delay(attempt, retry_config)
                        logger.warning(
                            f"Attempt {attempt + 1}/{retry_config.max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)

                raise RetryExhaustedError(retry_config.max_attempts, last_exception)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception = None

                for attempt in range(retry_config.max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except retry_config.retriable_exceptions as e:
                        last_exception = e
                        if attempt == retry_config.max_attempts - 1:
                            break

                        delay = calculate_backoff_delay(attempt, retry_config)
                        logger.warning(
                            f"Attempt {attempt + 1}/{retry_config.max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)

                raise RetryExhaustedError(retry_config.max_attempts, last_exception)
            return sync_wrapper

    return decorator


@dataclass
class TimeoutConfig:
    """Configuration for timeout handling."""
    default_timeout: float = 30.0
    max_timeout: float = 300.0


class TimeoutError(Exception):
    """Raised when operation times out."""
    pass


@contextmanager
def timeout(seconds: float) -> Generator[None, None, None]:
    """Context manager for timeout handling in sync code."""
    import signal

    def timeout_handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@asynccontextmanager
async def async_timeout(seconds: float) -> AsyncGenerator[None, None]:
    """Context manager for timeout handling in async code."""
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {seconds} seconds")


class FallbackChain:
    """Implements fallback chain pattern for graceful degradation."""

    def __init__(self, name: str):
        self.name = name
        self.fallbacks: List[Callable[..., Any]] = []

    def add_fallback(self, func: Callable[..., Any]) -> 'FallbackChain':
        """Add a fallback function to the chain."""
        self.fallbacks.append(func)
        return self

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute fallback chain until one succeeds."""
        last_exception = None

        for i, fallback in enumerate(self.fallbacks):
            try:
                logger.info(f"Executing fallback {i + 1}/{len(self.fallbacks)} for '{self.name}'")
                return fallback(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Fallback {i + 1} failed for '{self.name}': {e}")

        raise RuntimeError(
            f"All fallbacks exhausted for '{self.name}'. Last error: {last_exception}"
        )

    async def execute_async(self, *args: Any, **kwargs: Any) -> Any:
        """Execute async fallback chain until one succeeds."""
        last_exception = None

        for i, fallback in enumerate(self.fallbacks):
            try:
                logger.info(f"Executing async fallback {i + 1}/{len(self.fallbacks)} for '{self.name}'")
                if asyncio.iscoroutinefunction(fallback):
                    return await fallback(*args, **kwargs)
                else:
                    return fallback(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Async fallback {i + 1} failed for '{self.name}': {e}")

        raise RuntimeError(
            f"All async fallbacks exhausted for '{self.name}'. Last error: {last_exception}"
        )


# Global circuit breaker registry for monitoring
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_or_create_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def get_all_circuit_breaker_stats() -> List[Dict[str, Any]]:
    """Get statistics for all circuit breakers."""
    return [cb.get_stats() for cb in _circuit_breakers.values()]


def reset_circuit_breaker(name: str) -> bool:
    """Manually reset a circuit breaker to CLOSED state."""
    if name in _circuit_breakers:
        cb = _circuit_breakers[name]
        with cb._lock:
            old_state = cb.stats.state
            cb.stats.state = CircuitState.CLOSED
            cb.stats.failure_count = 0
            cb.stats.success_count = 0
            cb._log_state_change(old_state, cb.stats.state)
        return True
    return False


def circuit_breaker(
    name: Optional[str] = None,
    config: Optional[CircuitBreakerConfig] = None
):
    """Decorator for adding circuit breaker protection."""
    def decorator(func: CallableT) -> CallableT:
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        cb = get_or_create_circuit_breaker(breaker_name, config)

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await cb.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return cb.call(func, *args, **kwargs)
            return sync_wrapper

    return decorator


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    "CircuitBreakerOpenError",
    "CircuitState",
    "RetryConfig",
    "RetryExhaustedError",
    "TimeoutConfig",
    "TimeoutError",
    "FallbackChain",
    "retry",
    "circuit_breaker",
    "timeout",
    "async_timeout",
    "get_or_create_circuit_breaker",
    "get_all_circuit_breaker_stats",
    "reset_circuit_breaker",
]