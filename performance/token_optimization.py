import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
from collections import OrderedDict, defaultdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TokenizedSequence:
    """Represents a tokenized sequence with metadata"""
    tokens: List[int]
    text: str
    token_count: int
    encoding_time: float
    hash_key: str = field(init=False)

    def __post_init__(self):
        self.hash_key = hashlib.sha256(self.text.encode()).hexdigest()

@dataclass
class KVCacheEntry:
    """Key-Value cache entry for model states"""
    key_states: Any  # Attention key states
    value_states: Any  # Attention value states
    sequence_length: int
    layer_index: int
    timestamp: float
    access_count: int = 0
    memory_size: int = 0

    def update_access(self):
        self.access_count += 1

@dataclass
class TokenCacheStats:
    """Statistics for token cache performance"""
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens_saved: int = 0
    total_encoding_time_saved: float = 0.0
    memory_usage_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

class TokenCache:
    """High-performance token cache with compression and deduplication"""

    def __init__(self, max_size_mb: int = 500):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0

        self._token_cache: OrderedDict[str, TokenizedSequence] = OrderedDict()
        self._subsequence_cache: Dict[str, List[TokenizedSequence]] = defaultdict(list)
        self._lock = threading.RLock()

        self.stats = TokenCacheStats()

    def tokenize_with_cache(self, text: str, tokenizer_func) -> TokenizedSequence:
        """Tokenize text with caching and subsequence matching"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        with self._lock:
            # Check exact match cache
            if text_hash in self._token_cache:
                cached = self._token_cache[text_hash]
                self._token_cache.move_to_end(text_hash)  # LRU update
                self.stats.cache_hits += 1
                self.stats.total_tokens_saved += cached.token_count
                self.stats.total_encoding_time_saved += cached.encoding_time
                return cached

            # Check for subsequence matches
            best_match = self._find_best_subsequence_match(text)
            if best_match:
                # Use subsequence matching to reduce tokenization work
                remaining_text = text[len(best_match.text):]
                if remaining_text:
                    start_time = time.time()
                    remaining_tokens = tokenizer_func(remaining_text)
                    encoding_time = time.time() - start_time

                    # Combine tokens
                    combined_tokens = best_match.tokens + remaining_tokens
                    result = TokenizedSequence(
                        tokens=combined_tokens,
                        text=text,
                        token_count=len(combined_tokens),
                        encoding_time=encoding_time
                    )
                else:
                    result = best_match

                self.stats.cache_hits += 1
                self.stats.total_tokens_saved += best_match.token_count
                self.stats.total_encoding_time_saved += best_match.encoding_time
            else:
                # Full tokenization required
                start_time = time.time()
                tokens = tokenizer_func(text)
                encoding_time = time.time() - start_time

                result = TokenizedSequence(
                    tokens=tokens,
                    text=text,
                    token_count=len(tokens),
                    encoding_time=encoding_time
                )
                self.stats.cache_misses += 1

            # Cache the result
            self._cache_tokenized_sequence(result)
            return result

    def _find_best_subsequence_match(self, text: str) -> Optional[TokenizedSequence]:
        """Find the longest matching subsequence in cache"""
        best_match = None
        best_length = 0

        # Check for prefix matches
        for cached_hash, cached_seq in self._token_cache.items():
            if text.startswith(cached_seq.text) and len(cached_seq.text) > best_length:
                best_match = cached_seq
                best_length = len(cached_seq.text)

        return best_match

    def _cache_tokenized_sequence(self, sequence: TokenizedSequence):
        """Cache a tokenized sequence with size management"""
        sequence_size = self._estimate_size(sequence)

        # Remove old entries if needed
        while (
            self.current_size_bytes + sequence_size > self.max_size_bytes and
            self._token_cache
        ):
            self._evict_lru_entry()

        # Add new entry
        self._token_cache[sequence.hash_key] = sequence
        self.current_size_bytes += sequence_size
        self.stats.memory_usage_bytes = self.current_size_bytes

        # Also add to subsequence cache for prefix matching
        text_prefix = sequence.text[:min(100, len(sequence.text))]  # First 100 chars
        prefix_hash = hashlib.sha256(text_prefix.encode()).hexdigest()
        self._subsequence_cache[prefix_hash].append(sequence)

    def _evict_lru_entry(self):
        """Evict least recently used entry"""
        if self._token_cache:
            hash_key, sequence = self._token_cache.popitem(last=False)
            self.current_size_bytes -= self._estimate_size(sequence)

    def _estimate_size(self, sequence: TokenizedSequence) -> int:
        """Estimate memory size of a tokenized sequence"""
        return (
            len(sequence.tokens) * 4 +  # 4 bytes per token (int32)
            len(sequence.text.encode('utf-8')) +  # Text size in bytes
            64  # Overhead
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return {
            "cache_size": len(self._token_cache),
            "memory_usage_mb": self.current_size_bytes / (1024 * 1024),
            "hit_rate": self.stats.hit_rate,
            "tokens_saved": self.stats.total_tokens_saved,
            "encoding_time_saved_ms": self.stats.total_encoding_time_saved * 1000,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses
        }

class KVCache:
    """Key-Value cache for transformer model states"""

    def __init__(self, max_size_mb: int = 1000, max_sequence_length: int = 8192):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_sequence_length = max_sequence_length
        self.current_size_bytes = 0

        # Cache organized by sequence hash and layer
        self._kv_cache: Dict[str, Dict[int, KVCacheEntry]] = defaultdict(dict)
        self._access_order: OrderedDict[Tuple[str, int], float] = OrderedDict()
        self._lock = threading.RLock()

        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_saved_bytes': 0
        }

    def get_kv_states(
        self,
        sequence_hash: str,
        layer_index: int,
        sequence_length: int
    ) -> Optional[Tuple[Any, Any]]:
        """Get cached key-value states for a sequence and layer"""
        with self._lock:
            layer_cache = self._kv_cache.get(sequence_hash, {})
            entry = layer_cache.get(layer_index)

            if entry is None:
                self.stats['misses'] += 1
                return None

            # Check if cached sequence length is compatible
            if entry.sequence_length < sequence_length:
                self.stats['misses'] += 1
                return None

            # Update access tracking
            entry.update_access()
            access_key = (sequence_hash, layer_index)
            self._access_order[access_key] = time.time()
            self._access_order.move_to_end(access_key)

            self.stats['hits'] += 1
            self.stats['memory_saved_bytes'] += entry.memory_size

            # Return states (may need to truncate if longer than needed)
            if entry.sequence_length > sequence_length:
                # Truncate states to required length
                key_states = self._truncate_states(entry.key_states, sequence_length)
                value_states = self._truncate_states(entry.value_states, sequence_length)
                return key_states, value_states
            else:
                return entry.key_states, entry.value_states

    def cache_kv_states(
        self,
        sequence_hash: str,
        layer_index: int,
        key_states: Any,
        value_states: Any,
        sequence_length: int
    ):
        """Cache key-value states for a sequence and layer"""
        if sequence_length > self.max_sequence_length:
            return  # Don't cache very long sequences

        entry = KVCacheEntry(
            key_states=key_states,
            value_states=value_states,
            sequence_length=sequence_length,
            layer_index=layer_index,
            timestamp=time.time(),
            memory_size=self._estimate_kv_size(key_states, value_states)
        )

        with self._lock:
            # Evict entries if necessary
            while (
                self.current_size_bytes + entry.memory_size > self.max_size_bytes and
                self._access_order
            ):
                self._evict_lru_kv_entry()

            # Cache the entry
            self._kv_cache[sequence_hash][layer_index] = entry
            self.current_size_bytes += entry.memory_size

            # Track access order
            access_key = (sequence_hash, layer_index)
            self._access_order[access_key] = time.time()

    def _evict_lru_kv_entry(self):
        """Evict least recently used KV cache entry"""
        if not self._access_order:
            return

        # Get least recently used entry
        (sequence_hash, layer_index), _ = self._access_order.popitem(last=False)

        # Remove from cache
        if sequence_hash in self._kv_cache:
            layer_cache = self._kv_cache[sequence_hash]
            if layer_index in layer_cache:
                entry = layer_cache.pop(layer_index)
                self.current_size_bytes -= entry.memory_size
                self.stats['evictions'] += 1

                # Clean up empty sequence caches
                if not layer_cache:
                    del self._kv_cache[sequence_hash]

    def _estimate_kv_size(self, key_states: Any, value_states: Any) -> int:
        """Estimate memory size of key-value states"""
        try:
            key_size = len(pickle.dumps(key_states))
            value_size = len(pickle.dumps(value_states))
            return key_size + value_size
        except:
            # Fallback estimation
            return 1024 * 1024  # 1MB default

    def _truncate_states(self, states: Any, sequence_length: int) -> Any:
        """Truncate states to specified sequence length"""
        # This is framework-specific - implement based on your tensor library
        # For now, return as-is (implement truncation logic for your use case)
        return states

    def clear_sequence(self, sequence_hash: str):
        """Clear all cached states for a sequence"""
        with self._lock:
            if sequence_hash in self._kv_cache:
                layer_cache = self._kv_cache[sequence_hash]
                for layer_index, entry in layer_cache.items():
                    self.current_size_bytes -= entry.memory_size
                    # Remove from access order
                    access_key = (sequence_hash, layer_index)
                    self._access_order.pop(access_key, None)

                del self._kv_cache[sequence_hash]

    def get_stats(self) -> Dict[str, Any]:
        """Get KV cache performance statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0

        return {
            "cache_entries": sum(len(layer_cache) for layer_cache in self._kv_cache.values()),
            "sequences_cached": len(self._kv_cache),
            "memory_usage_mb": self.current_size_bytes / (1024 * 1024),
            "hit_rate": hit_rate,
            "memory_saved_mb": self.stats['memory_saved_bytes'] / (1024 * 1024),
            **self.stats
        }

class UnifiedTokenOptimizer:
    """
    Unified token optimization system combining token caching,
    KV caching, and intelligent batching strategies
    """

    def __init__(
        self,
        token_cache_size_mb: int = 500,
        kv_cache_size_mb: int = 1000,
        enable_compression: bool = True
    ):
        self.token_cache = TokenCache(max_size_mb=token_cache_size_mb)
        self.kv_cache = KVCache(max_size_mb=kv_cache_size_mb)
        self.enable_compression = enable_compression

        # Sequence-level optimization tracking
        self._sequence_patterns: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def optimize_tokenization(self, texts: List[str], tokenizer_func) -> List[TokenizedSequence]:
        """Optimize tokenization for a batch of texts"""
        results = []

        # Group similar texts for batch processing
        text_groups = self._group_similar_texts(texts)

        for group in text_groups:
            # Process each group efficiently
            for text in group:
                tokenized = self.token_cache.tokenize_with_cache(text, tokenizer_func)
                results.append(tokenized)

        return results

    def optimize_model_forward(
        self,
        sequence_hash: str,
        layer_index: int,
        sequence_length: int,
        compute_kv_func: Callable[[], Tuple[Any, Any]]
    ) -> Tuple[Any, Any]:
        """Optimize model forward pass with KV caching"""

        # Try to get cached KV states
        cached_states = self.kv_cache.get_kv_states(
            sequence_hash, layer_index, sequence_length
        )

        if cached_states is not None:
            return cached_states

        # Compute new KV states
        key_states, value_states = compute_kv_func()

        # Cache the computed states
        self.kv_cache.cache_kv_states(
            sequence_hash, layer_index, key_states, value_states, sequence_length
        )

        return key_states, value_states

    def _group_similar_texts(self, texts: List[str]) -> List[List[str]]:
        """Group similar texts for efficient batch processing"""
        # Simple grouping by length and prefix similarity
        length_groups = defaultdict(list)

        for text in texts:
            length_bucket = (len(text) // 100) * 100  # Group by ~100 char buckets
            length_groups[length_bucket].append(text)

        # Further group by prefix similarity within length groups
        final_groups = []
        for length_group in length_groups.values():
            prefix_groups = defaultdict(list)
            for text in length_group:
                prefix = text[:min(50, len(text))]  # First 50 chars as prefix
                prefix_groups[prefix].append(text)

            final_groups.extend(prefix_groups.values())

        return final_groups

    def analyze_sequence_patterns(self, sequence_hash: str, tokens: List[int]):
        """Analyze and store sequence patterns for optimization"""
        with self._lock:
            if sequence_hash not in self._sequence_patterns:
                self._sequence_patterns[sequence_hash] = {
                    'token_frequency': defaultdict(int),
                    'bigram_frequency': defaultdict(int),
                    'avg_length': 0,
                    'access_count': 0
                }

            pattern = self._sequence_patterns[sequence_hash]
            pattern['access_count'] += 1

            # Update token frequency
            for token in tokens:
                pattern['token_frequency'][token] += 1

            # Update bigram frequency
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                pattern['bigram_frequency'][bigram] += 1

            # Update average length
            pattern['avg_length'] = (
                (pattern['avg_length'] * (pattern['access_count'] - 1) + len(tokens)) /
                pattern['access_count']
            )

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        return {
            "token_cache": self.token_cache.get_stats(),
            "kv_cache": self.kv_cache.get_stats(),
            "sequence_patterns": len(self._sequence_patterns),
            "total_memory_saved_mb": (
                self.token_cache.stats.total_tokens_saved * 4 +  # 4 bytes per token
                self.kv_cache.stats['memory_saved_bytes']
            ) / (1024 * 1024)
        }

    def clear_caches(self):
        """Clear all caches"""
        self.token_cache._token_cache.clear()
        self.token_cache._subsequence_cache.clear()
        self.token_cache.current_size_bytes = 0

        self.kv_cache._kv_cache.clear()
        self.kv_cache._access_order.clear()
        self.kv_cache.current_size_bytes = 0

        with self._lock:
            self._sequence_patterns.clear()

# Utility functions for integration
def create_sequence_hash(text: str, model_name: str = "") -> str:
    """Create a hash for sequence identification"""
    content = f"{model_name}:{text}"
    return hashlib.sha256(content.encode()).hexdigest()

def estimate_token_savings(optimizer: UnifiedTokenOptimizer) -> Dict[str, float]:
    """Estimate computational and memory savings from optimization"""
    stats = optimizer.get_optimization_stats()

    token_stats = stats["token_cache"]
    kv_stats = stats["kv_cache"]

    # Estimate computational savings
    encoding_time_saved = token_stats["encoding_time_saved_ms"] / 1000.0  # Convert to seconds
    compute_saved_percentage = (
        token_stats["tokens_saved"] /
        (token_stats["tokens_saved"] + token_stats["cache_misses"] * 100)  # Assume 100 tokens per miss
    ) if token_stats["cache_misses"] > 0 else 0

    # Estimate memory savings
    memory_saved_mb = stats["total_memory_saved_mb"]

    return {
        "encoding_time_saved_seconds": encoding_time_saved,
        "compute_saved_percentage": compute_saved_percentage * 100,
        "memory_saved_mb": memory_saved_mb,
        "token_cache_hit_rate": token_stats["hit_rate"] * 100,
        "kv_cache_hit_rate": kv_stats["hit_rate"] * 100
    }