import asyncio
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union, Callable, TypeVar, Generic
import threading
from collections import OrderedDict
import logging
import redis
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CacheLevel(Enum):
    MEMORY = 1
    REDIS = 2
    DISK = 3

@dataclass
class CacheEntry(Generic[T]):
    value: T
    timestamp: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def update_access(self):
        self.access_count += 1

class CacheStrategy(Enum):
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns

@dataclass
class CacheConfig:
    memory_size_mb: int = 100
    redis_url: Optional[str] = None
    disk_cache_dir: str = "./cache"
    default_ttl: float = 3600.0  # 1 hour
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    compression_enabled: bool = True
    metrics_enabled: bool = True

class CacheBackend(ABC, Generic[T]):
    """Abstract base class for cache backends"""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry[T]]:
        pass

    @abstractmethod
    async def put(self, key: str, entry: CacheEntry[T]) -> bool:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    async def clear(self) -> bool:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass

class MemoryCache(CacheBackend[T]):
    """High-performance in-memory cache with configurable eviction strategies"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.max_size_bytes = config.memory_size_mb * 1024 * 1024
        self.current_size_bytes = 0

        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()

        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    async def get(self, key: str) -> Optional[CacheEntry[T]]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self.current_size_bytes -= entry.size_bytes
                self._misses += 1
                return None

            # Update access patterns
            entry.update_access()

            # Move to end for LRU
            if self.config.strategy == CacheStrategy.LRU:
                self._cache.move_to_end(key)

            self._hits += 1
            return entry

    async def put(self, key: str, entry: CacheEntry[T]) -> bool:
        entry.size_bytes = self._estimate_size(entry.value)

        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self.current_size_bytes -= old_entry.size_bytes
                del self._cache[key]

            # Evict entries if necessary
            while (
                self.current_size_bytes + entry.size_bytes > self.max_size_bytes and
                self._cache
            ):
                self._evict_one()

            # Add new entry
            self._cache[key] = entry
            self.current_size_bytes += entry.size_bytes
            return True

    async def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self.current_size_bytes -= entry.size_bytes
                del self._cache[key]
                return True
            return False

    async def clear(self) -> bool:
        with self._lock:
            self._cache.clear()
            self.current_size_bytes = 0
            return True

    def _evict_one(self):
        """Evict one entry based on the configured strategy"""
        if not self._cache:
            return

        if self.config.strategy == CacheStrategy.LRU:
            # Remove least recently used (first item)
            key, entry = self._cache.popitem(last=False)
        elif self.config.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
            entry = self._cache.pop(key)
        elif self.config.strategy == CacheStrategy.TTL:
            # Remove oldest entry
            key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
            entry = self._cache.pop(key)
        else:  # ADAPTIVE
            # Use hybrid approach
            current_time = time.time()
            scores = {}
            for k, e in self._cache.items():
                age = current_time - e.timestamp
                score = age / (e.access_count + 1)  # Higher score = more evictable
                scores[k] = score

            key = max(scores.keys(), key=lambda k: scores[k])
            entry = self._cache.pop(key)

        self.current_size_bytes -= entry.size_bytes
        self._evictions += 1

    def _estimate_size(self, value: Any) -> int:
        """Estimate the memory size of a value"""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value))

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            return {
                "level": "memory",
                "entries": len(self._cache),
                "size_bytes": self.current_size_bytes,
                "size_mb": self.current_size_bytes / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions
            }

class RedisCache(CacheBackend[T]):
    """Redis-based cache for distributed caching"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client = None
        if config.redis_url:
            try:
                self.redis_client = redis.from_url(config.redis_url, decode_responses=False)
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None

        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[CacheEntry[T]]:
        if not self.redis_client:
            return None

        try:
            data = self.redis_client.get(f"cache:{key}")
            if data is None:
                self._misses += 1
                return None

            entry = pickle.loads(data)
            if entry.is_expired():
                await self.delete(key)
                self._misses += 1
                return None

            entry.update_access()
            self._hits += 1
            return entry

        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._misses += 1
            return None

    async def put(self, key: str, entry: CacheEntry[T]) -> bool:
        if not self.redis_client:
            return False

        try:
            data = pickle.dumps(entry)
            redis_key = f"cache:{key}"

            if entry.ttl:
                self.redis_client.setex(redis_key, int(entry.ttl), data)
            else:
                self.redis_client.set(redis_key, data)

            return True

        except Exception as e:
            logger.error(f"Redis put error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        if not self.redis_client:
            return False

        try:
            result = self.redis_client.delete(f"cache:{key}")
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def clear(self) -> bool:
        if not self.redis_client:
            return False

        try:
            pattern = "cache:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        redis_info = {}
        if self.redis_client:
            try:
                info = self.redis_client.info()
                redis_info = {
                    "memory_usage": info.get("used_memory", 0),
                    "connected": True
                }
            except:
                redis_info = {"connected": False}

        return {
            "level": "redis",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            **redis_info
        }

class DiskCache(CacheBackend[T]):
    """SQLite-based disk cache for persistent storage"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.disk_cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.db_path = self.cache_dir / "cache.db"
        self._init_db()

        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    ttl REAL,
                    access_count INTEGER,
                    size_bytes INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)")

    async def get(self, key: str) -> Optional[CacheEntry[T]]:
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT value, timestamp, ttl, access_count, size_bytes FROM cache_entries WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()

                    if row is None:
                        self._misses += 1
                        return None

                    value, timestamp, ttl, access_count, size_bytes = row
                    entry = CacheEntry(
                        value=pickle.loads(value),
                        timestamp=timestamp,
                        ttl=ttl,
                        access_count=access_count,
                        size_bytes=size_bytes
                    )

                    if entry.is_expired():
                        await self.delete(key)
                        self._misses += 1
                        return None

                    # Update access count
                    entry.update_access()
                    conn.execute(
                        "UPDATE cache_entries SET access_count = ? WHERE key = ?",
                        (entry.access_count, key)
                    )

                    self._hits += 1
                    return entry

            except Exception as e:
                logger.error(f"Disk cache get error: {e}")
                self._misses += 1
                return None

    async def put(self, key: str, entry: CacheEntry[T]) -> bool:
        with self._lock:
            try:
                value_bytes = pickle.dumps(entry.value)
                entry.size_bytes = len(value_bytes)

                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries
                        (key, value, timestamp, ttl, access_count, size_bytes)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        key, value_bytes, entry.timestamp, entry.ttl,
                        entry.access_count, entry.size_bytes
                    ))

                return True

            except Exception as e:
                logger.error(f"Disk cache put error: {e}")
                return False

    async def delete(self, key: str) -> bool:
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    return cursor.rowcount > 0

            except Exception as e:
                logger.error(f"Disk cache delete error: {e}")
                return False

    async def clear(self) -> bool:
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM cache_entries")
                return True

            except Exception as e:
                logger.error(f"Disk cache clear error: {e}")
                return False

    def get_stats(self) -> Dict[str, Any]:
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*), SUM(size_bytes) FROM cache_entries")
                count, total_size = cursor.fetchone()
                total_size = total_size or 0

            return {
                "level": "disk",
                "entries": count,
                "size_bytes": total_size,
                "size_mb": total_size / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate
            }

        except Exception as e:
            logger.error(f"Disk cache stats error: {e}")
            return {
                "level": "disk",
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "error": str(e)
            }

class MultiTierCache(Generic[T]):
    """
    Multi-tier cache system with automatic promotion/demotion between cache levels
    """

    def __init__(self, config: CacheConfig):
        self.config = config

        # Initialize cache backends
        self.memory_cache = MemoryCache[T](config)
        self.redis_cache = RedisCache[T](config) if config.redis_url else None
        self.disk_cache = DiskCache[T](config)

        # Cache levels in order of preference
        self.cache_levels = [
            (CacheLevel.MEMORY, self.memory_cache),
            (CacheLevel.REDIS, self.redis_cache),
            (CacheLevel.DISK, self.disk_cache)
        ]

        # Filter out None backends
        self.cache_levels = [(level, cache) for level, cache in self.cache_levels if cache is not None]

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache, checking each tier in order"""
        cache_key = self._hash_key(key)

        for i, (level, cache) in enumerate(self.cache_levels):
            entry = await cache.get(cache_key)
            if entry is not None:
                # Promote to higher tiers
                await self._promote_entry(key, entry, i)
                return entry.value

        return None

    async def put(self, key: str, value: T, ttl: Optional[float] = None) -> bool:
        """Put value into cache at all appropriate levels"""
        cache_key = self._hash_key(key)
        ttl = ttl or self.config.default_ttl

        entry = CacheEntry(
            value=value,
            timestamp=time.time(),
            ttl=ttl
        )

        # Store in all cache levels
        results = []
        for level, cache in self.cache_levels:
            result = await cache.put(cache_key, entry)
            results.append(result)

        return any(results)

    async def delete(self, key: str) -> bool:
        """Delete from all cache levels"""
        cache_key = self._hash_key(key)

        results = []
        for level, cache in self.cache_levels:
            result = await cache.delete(cache_key)
            results.append(result)

        return any(results)

    async def clear(self) -> bool:
        """Clear all cache levels"""
        results = []
        for level, cache in self.cache_levels:
            result = await cache.clear()
            results.append(result)

        return all(results)

    async def _promote_entry(self, key: str, entry: CacheEntry[T], found_at_level: int):
        """Promote entry to higher cache levels"""
        cache_key = self._hash_key(key)

        # Promote to all higher levels
        for i in range(found_at_level):
            level, cache = self.cache_levels[i]
            await cache.put(cache_key, entry)

    def _hash_key(self, key: str) -> str:
        """Generate consistent hash for cache key"""
        return hashlib.sha256(key.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            "config": {
                "memory_size_mb": self.config.memory_size_mb,
                "default_ttl": self.config.default_ttl,
                "strategy": self.config.strategy.value
            },
            "levels": {}
        }

        for level, cache in self.cache_levels:
            stats["levels"][level.name.lower()] = cache.get_stats()

        return stats

def cache_with_multi_tier(
    cache: MultiTierCache,
    ttl: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None
):
    """Decorator for caching function results with multi-tier cache"""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def async_wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"

            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await cache.put(cache_key, result, ttl)
            return result

        def sync_wrapper(*args, **kwargs) -> T:
            return asyncio.run(async_wrapper(*args, **kwargs))

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator