"""
Integration of performance optimization modules into agent workflows
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
import logging
from concurrent.futures import Future

logger = logging.getLogger(__name__)

class AgentPerformanceOptimizer:
    """Integrates performance optimization into agent operations"""

    def __init__(self):
        self.batch_processor = None
        self.cache_system = None
        self.token_optimizer = None
        self._initialize_performance_modules()

    def _initialize_performance_modules(self):
        """Initialize performance optimization modules"""
        try:
            from performance.continuous_batch_processor import ContinuousBatchProcessor, BatchConfig, BatchPriority
            from performance.multi_tier_cache import MultiTierCache, CacheConfig
            from performance.token_optimization import UnifiedTokenOptimizer

            # Initialize continuous batching for agent operations
            self.batch_processor = ContinuousBatchProcessor(
                batch_function=self._process_agent_batch,
                config=BatchConfig(
                    max_batch_size=16,
                    max_wait_time=0.05,  # 50ms for low latency
                    min_batch_size=1
                ),
                max_workers=4
            )

            # Initialize multi-tier cache for market data and analysis results
            cache_config = CacheConfig(
                memory_size_mb=200,
                redis_url=None,  # Will use in-memory only if Redis not available
                default_ttl=1800.0  # 30 minutes
            )
            self.cache_system = MultiTierCache(cache_config)

            # Initialize token optimization for LLM calls
            self.token_optimizer = UnifiedTokenOptimizer(
                token_cache_size_mb=100,
                kv_cache_size_mb=200
            )

            logger.info("Performance optimization modules initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize performance modules: {e}")

    async def _process_agent_batch(self, batch_items: List[Any]) -> List[Any]:
        """Process a batch of agent operations efficiently"""
        results = []

        # Group similar operations for batch processing
        operation_groups = {}
        for item in batch_items:
            op_type = item.get('operation_type', 'unknown')
            if op_type not in operation_groups:
                operation_groups[op_type] = []
            operation_groups[op_type].append(item)

        # Process each group
        for op_type, items in operation_groups.items():
            if op_type == 'market_data_fetch':
                batch_results = await self._batch_market_data_fetch(items)
            elif op_type == 'analysis':
                batch_results = await self._batch_analysis(items)
            elif op_type == 'llm_call':
                batch_results = await self._batch_llm_calls(items)
            else:
                # Process individually for unknown types
                batch_results = []
                for item in items:
                    result = await self._process_single_item(item)
                    batch_results.append(result)

            results.extend(batch_results)

        return results

    async def _batch_market_data_fetch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch process market data fetches"""
        # Extract unique symbols
        symbols = list(set(item['symbol'] for item in items))

        try:
            # Fetch all symbols at once
            from mcp.servers.market_data_server import get_real_time_market_data

            batch_result = await get_real_time_market_data({
                'symbols': symbols
            })

            # Distribute results back to individual requests
            results = []
            symbol_data = {data['symbol']: data for data in batch_result.get('data', [])}

            for item in items:
                symbol = item['symbol']
                if symbol in symbol_data:
                    results.append({
                        'success': True,
                        'data': symbol_data[symbol],
                        'symbol': symbol,
                        'cached': False
                    })
                else:
                    results.append({
                        'success': False,
                        'error': f'No data for {symbol}',
                        'symbol': symbol
                    })

            return results

        except Exception as e:
            logger.error(f"Batch market data fetch failed: {e}")
            # Return error for all items
            return [{'success': False, 'error': str(e), 'symbol': item['symbol']} for item in items]

    async def _batch_analysis(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch process analysis operations"""
        results = []

        # Group by analysis type
        analysis_groups = {}
        for item in items:
            analysis_type = item.get('analysis_type', 'general')
            if analysis_type not in analysis_groups:
                analysis_groups[analysis_type] = []
            analysis_groups[analysis_type].append(item)

        # Process each analysis type in batch
        for analysis_type, group_items in analysis_groups.items():
            if analysis_type == 'technical':
                group_results = await self._batch_technical_analysis(group_items)
            elif analysis_type == 'fundamental':
                group_results = await self._batch_fundamental_analysis(group_items)
            else:
                # Individual processing for unknown types
                group_results = []
                for item in group_items:
                    result = await self._process_single_analysis(item)
                    group_results.append(result)

            results.extend(group_results)

        return results

    async def _batch_technical_analysis(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch technical analysis using mathematical toolkits"""
        try:
            # Collect all data for batch processing
            all_symbols = [item['symbol'] for item in items]
            all_data = {}

            # Fetch historical data for all symbols
            for symbol in set(all_symbols):
                cache_key = f"historical_{symbol}_1d"
                cached_data = await self.cache_system.get(cache_key)

                if cached_data:
                    all_data[symbol] = cached_data
                else:
                    # Fetch and cache
                    from agents.core.orchestrator import TradingOrchestrator
                    orchestrator = TradingOrchestrator()
                    try:
                        historical_data = await orchestrator.get_historical_data(symbol)
                        all_data[symbol] = historical_data
                        await self.cache_system.put(cache_key, historical_data, ttl=1800)
                    except Exception as e:
                        logger.warning(f"Failed to get historical data for {symbol}: {e}")
                        all_data[symbol] = None

            # Process technical analysis for each item
            results = []
            for item in items:
                symbol = item['symbol']
                data = all_data.get(symbol)

                if data:
                    # Apply mathematical analysis
                    analysis_result = await self._apply_mathematical_analysis(symbol, data)
                    results.append({
                        'success': True,
                        'symbol': symbol,
                        'analysis': analysis_result
                    })
                else:
                    results.append({
                        'success': False,
                        'symbol': symbol,
                        'error': 'No historical data available'
                    })

            return results

        except Exception as e:
            logger.error(f"Batch technical analysis failed: {e}")
            return [{'success': False, 'error': str(e)} for _ in items]

    async def _apply_mathematical_analysis(self, symbol: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply mathematical toolkits to market data"""
        try:
            # Extract price data
            prices = [point['close'] for point in data]
            returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]

            # Use mathematical integration
            from agents.integration.mathematical_integration import MathematicalIntegration
            from mcp.servers.regime_hmm_server import HMMRegimeDetector
            from mcp.servers.mean_reversion_ou_server import OUMeanReversionAnalyzer
            from mcp.servers.signal_processing_server import SignalProcessor

            math_integration = MathematicalIntegration(
                hmm_server=HMMRegimeDetector(),
                ou_server=OUMeanReversionAnalyzer(),
                signal_server=SignalProcessor()
            )

            # Perform complete mathematical analysis
            analysis = await math_integration.analyze_market_regime({
                'prices': prices,
                'returns': returns,
                'raw_signals': returns  # Use returns as raw signals
            })

            return analysis

        except Exception as e:
            logger.error(f"Mathematical analysis failed for {symbol}: {e}")
            return {
                'regime': {'current': 'unknown', 'confidence': 0},
                'mean_reversion': {'z_score': 0},
                'signals': {'confidence': 0},
                'recommendation': 'HOLD'
            }

    async def _batch_fundamental_analysis(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch fundamental analysis"""
        # Simplified implementation - would integrate with fundamental data sources
        results = []
        for item in items:
            results.append({
                'success': True,
                'symbol': item['symbol'],
                'analysis': {
                    'pe_ratio': 15.0,  # Would fetch real data
                    'recommendation': 'HOLD'
                }
            })
        return results

    async def _batch_llm_calls(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch LLM calls with token optimization"""
        if not self.token_optimizer:
            # Process individually without optimization
            results = []
            for item in items:
                result = await self._process_single_llm_call(item)
                results.append(result)
            return results

        # Group similar prompts for token optimization
        results = []
        for item in items:
            prompt = item.get('prompt', '')
            cached_result = await self._get_cached_llm_result(prompt)

            if cached_result:
                results.append({
                    'success': True,
                    'response': cached_result,
                    'cached': True
                })
            else:
                # Process and cache
                result = await self._process_single_llm_call(item)
                if result.get('success'):
                    await self._cache_llm_result(prompt, result['response'])
                results.append(result)

        return results

    async def _process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item when batch processing isn't available"""
        try:
            operation_type = item.get('operation_type')
            if operation_type == 'market_data_fetch':
                return await self._single_market_data_fetch(item)
            elif operation_type == 'analysis':
                return await self._process_single_analysis(item)
            elif operation_type == 'llm_call':
                return await self._process_single_llm_call(item)
            else:
                return {'success': False, 'error': f'Unknown operation type: {operation_type}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _single_market_data_fetch(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Single market data fetch"""
        symbol = item['symbol']
        cache_key = f"market_data_{symbol}"

        # Check cache first
        cached_data = await self.cache_system.get(cache_key)
        if cached_data:
            return {'success': True, 'data': cached_data, 'cached': True}

        # Fetch from source
        try:
            from mcp.servers.market_data_server import get_real_time_market_data
            result = await get_real_time_market_data({'symbols': [symbol]})

            if result.get('data'):
                data = result['data'][0]
                await self.cache_system.put(cache_key, data, ttl=60)  # 1 minute cache
                return {'success': True, 'data': data, 'cached': False}
            else:
                return {'success': False, 'error': 'No data returned'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _process_single_analysis(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process single analysis operation"""
        return {'success': True, 'analysis': 'individual_processing'}

    async def _process_single_llm_call(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process single LLM call"""
        return {'success': True, 'response': 'llm_response_placeholder'}

    async def _get_cached_llm_result(self, prompt: str) -> Optional[str]:
        """Get cached LLM result"""
        if self.cache_system:
            cache_key = f"llm_{hash(prompt)}"
            return await self.cache_system.get(cache_key)
        return None

    async def _cache_llm_result(self, prompt: str, response: str):
        """Cache LLM result"""
        if self.cache_system:
            cache_key = f"llm_{hash(prompt)}"
            await self.cache_system.put(cache_key, response, ttl=3600)  # 1 hour

    # Public interface methods

    async def optimize_market_data_request(self, symbol: str, priority: str = "medium") -> Dict[str, Any]:
        """Optimized market data request using batching and caching"""
        if not self.batch_processor:
            return await self._single_market_data_fetch({'symbol': symbol})

        # Convert priority string to enum
        priority_map = {
            "low": BatchPriority.LOW,
            "medium": BatchPriority.MEDIUM,
            "high": BatchPriority.HIGH,
            "critical": BatchPriority.CRITICAL
        }

        batch_priority = priority_map.get(priority, BatchPriority.MEDIUM)

        # Submit to batch processor
        future = await self.batch_processor.submit(
            {
                'operation_type': 'market_data_fetch',
                'symbol': symbol
            },
            priority=batch_priority
        )

        return await future

    async def optimize_analysis_request(self, symbol: str, analysis_type: str = "technical") -> Dict[str, Any]:
        """Optimized analysis request"""
        if not self.batch_processor:
            return await self._process_single_analysis({
                'symbol': symbol,
                'analysis_type': analysis_type
            })

        future = await self.batch_processor.submit(
            {
                'operation_type': 'analysis',
                'symbol': symbol,
                'analysis_type': analysis_type
            },
            priority=BatchPriority.HIGH
        )

        return await future

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance optimization statistics"""
        stats = {
            'batch_processor': None,
            'cache_system': None,
            'token_optimizer': None
        }

        if self.batch_processor:
            stats['batch_processor'] = self.batch_processor.get_metrics()

        if self.cache_system:
            stats['cache_system'] = self.cache_system.get_stats()

        if self.token_optimizer:
            stats['token_optimizer'] = self.token_optimizer.get_optimization_stats()

        return stats

# Global performance optimizer instance
_performance_optimizer = None

def get_performance_optimizer() -> AgentPerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = AgentPerformanceOptimizer()
    return _performance_optimizer

# Decorator for performance-optimized operations
def optimize_performance(operation_type: str = "general"):
    """Decorator to add performance optimization to operations"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()

            # Extract relevant parameters for optimization
            if operation_type == "market_data" and len(args) > 0:
                symbol = args[0] if isinstance(args[0], str) else kwargs.get('symbol')
                if symbol:
                    return await optimizer.optimize_market_data_request(symbol)

            elif operation_type == "analysis" and len(args) > 0:
                symbol = args[0] if isinstance(args[0], str) else kwargs.get('symbol')
                analysis_type = kwargs.get('analysis_type', 'technical')
                if symbol:
                    return await optimizer.optimize_analysis_request(symbol, analysis_type)

            # Fallback to original function
            return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            # For sync functions, run in event loop if available
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            except:
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator