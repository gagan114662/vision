# Critical Gaps Action Plan

## Executive Summary
While circuit breakers and registry validation have been addressed, major gaps remain in real data adoption, runtime validation, observability integration, and mathematical toolkit utilization. This action plan addresses each gap systematically.

## 🔴 CRITICAL GAPS

### 1. Synthetic Fallback Elimination
**Files Affected:**
- `mcp/servers/market_data_server.py:544` - Remove price synthesis fallback
- `agents/core/orchestrator.py:193` - Remove deterministic mock fallback
- `test_quantconnect_real_backtest.py:214` - Remove stats fabrication

**Action Items:**
```python
# Instead of synthesizing, fail with clear error:
if not real_data_available:
    raise DataUnavailableError(f"No real data available for {symbol}")
    # DO NOT: return self._synthesize_fallback_data()
```

### 2. Runtime Schema Validation
**Files Affected:**
- `mcp/server.py` - Add schema enforcement to register_tool
- All MCP server implementations

**Implementation:**
```python
def register_tool(name: str, schema: str):
    def decorator(func):
        # Load and validate schema at registration
        schema_validator = load_json_schema(schema)

        @wraps(func)
        def wrapper(params: Dict[str, Any]):
            # Validate input against schema
            if not schema_validator.validate(params):
                raise SchemaValidationError(schema_validator.errors)

            result = func(params)

            # Validate output if response schema exists
            if response_schema:
                validate_response(result, response_schema)

            return result
        return wrapper
    return decorator
```

### 3. Observability Integration
**Current State:**
- `mcp/servers/observability_server.py:112` - Just caches psutil snapshots
- No dashboard integration
- No agent instrumentation

**Required Actions:**
1. Create Grafana dashboard configurations
2. Wire tracing into agent workflows
3. Add Prometheus metrics export
4. Implement distributed tracing context propagation

### 4. Mathematical Toolkit Integration
**Disconnected Components:**
- `mcp/servers/regime_hmm_server.py:23` - Falls back to centroid splitter
- `mcp/servers/mean_reversion_ou_server.py:60` - No consumers
- Wavelet/Fourier filters unused in `agents/workflows/complete_trading_workflow.py:147`

**Integration Plan:**
```python
# In complete_trading_workflow.py
async def analyze_market_regime(self, data):
    # Use HMM for regime detection
    regime = await self.hmm_server.detect_regime(data)

    # Apply OU mean reversion analysis
    mean_reversion_signals = await self.ou_server.analyze(data, regime)

    # Apply signal processing
    filtered_signals = await self.apply_wavelet_filter(mean_reversion_signals)

    return self.combine_signals(regime, mean_reversion_signals, filtered_signals)
```

### 5. Compliance & Provenance Hardening
**Issues:**
- `mcp/servers/compliance_server.py:26` - Simple rules aggregator
- `mcp/servers/provenance_server.py:24` - Local env vars, no vault
- No regulatory reporting automation

**Hardening Steps:**
1. Integrate with HashiCorp Vault for secrets
2. Implement immutable audit logging
3. Add FINRA/SEC reporting templates
4. Create compliance workflow automation

## 📋 IMPLEMENTATION PRIORITY

### Phase 1: Data Reality (Week 1)
- [ ] Remove ALL synthetic fallbacks
- [ ] Add proper error handling for missing data
- [ ] Implement retry logic with exponential backoff
- [ ] Add data quality validation

### Phase 2: Runtime Validation (Week 1-2)
- [ ] Implement schema validation decorator
- [ ] Add response validation
- [ ] Create validation metrics
- [ ] Add schema versioning

### Phase 3: Test Coverage (Week 2)
- [ ] Add integration tests for streaming
- [ ] Add portfolio management tests
- [ ] Add compliance workflow tests
- [ ] Achieve 80% code coverage

### Phase 4: Performance Integration (Week 3)
- [ ] Wire continuous batching into agents
- [ ] Implement cache warming strategies
- [ ] Add performance benchmarks
- [ ] Create load testing suite

### Phase 5: Observability (Week 3-4)
- [ ] Deploy Grafana dashboards
- [ ] Implement Prometheus exporters
- [ ] Add distributed tracing
- [ ] Create alerting rules

### Phase 6: Mathematical Integration (Week 4)
- [ ] Wire HMM into trading workflow
- [ ] Integrate OU mean reversion
- [ ] Apply signal filters in pipeline
- [ ] Add factor model integration

## 🎯 SUCCESS CRITERIA

1. **Zero Synthetic Fallbacks**: No code path returns synthetic data
2. **100% Schema Validation**: All MCP tools validate input/output
3. **80% Test Coverage**: Critical paths have integration tests
4. **Full Observability**: Every agent action is traced and monitored
5. **Mathematical Integration**: All toolkits actively used in workflows

## 🚨 RISK MITIGATION

1. **Data Unavailability**: Implement circuit breakers with proper backpressure
2. **Performance Degradation**: Use async/await properly, implement caching
3. **Compliance Violations**: Automated checks before any trade execution
4. **System Failures**: Implement proper error boundaries and recovery

## 📊 METRICS TO TRACK

- Synthetic fallback invocations (should be 0)
- Schema validation failures per hour
- Test coverage percentage
- P95 latency for all operations
- Mathematical toolkit utilization rate
- Compliance check pass rate

## 🔄 CONTINUOUS IMPROVEMENT

1. Weekly review of synthetic fallback attempts
2. Daily monitoring of schema validation errors
3. Continuous performance profiling
4. Regular compliance audit trails
5. Mathematical model backtesting

## IMMEDIATE NEXT STEPS

1. **TODAY**: Remove all synthetic fallbacks from market_data_server.py
2. **TOMORROW**: Implement runtime schema validation
3. **THIS WEEK**: Add integration test coverage
4. **NEXT WEEK**: Wire observability and mathematical toolkits

This action plan directly addresses the gaps identified in the review and provides a clear path to production readiness.