# ADR-002: Stage Graph Architecture

**Status**: Proposed  
**Date**: 2025-12-08  
**Authors**: Transformation Portal Architect  
**Related PRs**: PR-3 (Stage Graph Refactor)

---

## Context

Current pipeline implementations use monolithic `process()` functions that execute all transformations sequentially without:

1. **Cacheability**: Reprocessing identical inputs wastes computation
2. **Measurability**: Cannot identify performance bottlenecks at granular level
3. **Flexibility**: Cannot skip/reorder stages based on context
4. **Observability**: No stage-level metrics or tracing
5. **Testing**: Difficult to test individual transformations in isolation

**Example: Lux Depth V2 Pipeline**

Current monolithic approach:
```python
def process(self, image: np.ndarray) -> np.ndarray:
    # All operations in one function
    depth = estimate_depth(image)
    mask = segment_materials(image)
    enhanced = apply_tone_mapping(depth, mask)
    upscaled = upscale(enhanced)
    return upscaled
```

**Problems**:
- If upscaling fails, must recompute depth + segmentation
- Cannot cache intermediate results (depth maps, masks)
- Cannot measure which stage is slow
- Cannot intelligently skip stages (e.g., no upscaling needed)

---

## Decision

We will refactor pipelines to use a **Stage Graph Architecture** where:

1. **Stage**: Atomic, deterministic transformation `(input, config) -> output`
2. **Graph**: Directed Acyclic Graph (DAG) of stages with dependencies
3. **Caching**: Outputs cached by `(input_hash, config_hash, stage_version)`
4. **Policy Engine**: Intelligently selects graph parameters from context
5. **Observability**: Every stage reports timing, cache hit rate, errors

### Architecture

```
┌─────────────┐
│   Input     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Context Extract │  (resolution, HDR, material hints)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Policy Engine   │  (select graph params, enable/disable stages)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  Stage Graph    │────▶│ Cache Store  │
│                 │     └──────────────┘
│  ┌───────────┐  │
│  │ Depth Est │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Seg. Mask │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Tone Map  │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Upscale   │  │
│  └─────┬─────┘  │
└────────┼────────┘
         │
         ▼
    ┌─────────┐
    │ Output  │
    └─────────┘
```

### API Contracts

**Stage Base Class**:
```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class Stage(ABC):
    """Deterministic pipeline stage."""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
    
    @abstractmethod
    def execute(self, input_data: Any, config: Dict[str, Any]) -> Any:
        """Execute stage transformation."""
        pass
    
    def compute_cache_key(self, input_data: Any, config: Dict[str, Any]) -> str:
        """Compute deterministic cache key."""
        # Hash: input + config + stage_version
        pass
    
    def run(self, input_data: Any, config: Dict[str, Any], 
            cache: 'ArtifactStore') -> 'StageResult':
        """Run with caching and timing."""
        # Check cache
        # Execute if miss
        # Store result
        # Return metrics
        pass
```

**Pipeline Graph**:
```python
class PipelineGraph:
    """DAG of processing stages."""
    
    def __init__(self, stages: List[Stage]):
        self.stages = stages
    
    def execute(self, input_data: Any, config: Dict[str, Any]) -> Any:
        """Execute graph."""
        data = input_data
        for stage in self.stages:
            result = stage.run(data, config, cache)
            data = result.output
        return data
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get stage-level metrics."""
        return {
            stage.name: {
                "duration_ms": result.duration_ms,
                "cache_hit": result.cache_hit
            }
            for stage, result in self.results.items()
        }
```

**Policy Engine**:
```python
class PolicyEngine:
    """Context-aware parameter selection."""
    
    def apply(self, context: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rules based on context."""
        config = base_config.copy()
        
        # Example rules
        if context["is_uhd"]:
            config["enable_tiling"] = True
        
        if context["is_hdr"]:
            config["tone_map_operator"] = "aces"
        
        return config
```

---

## Consequences

### Positive

1. **10-20x Speedup on Re-processing**: Cached stages avoid redundant computation
2. **Performance Visibility**: Identify bottlenecks at stage level
3. **Intelligent Routing**: Skip unnecessary stages based on context
4. **Better Testing**: Test stages in isolation
5. **Robustness**: Failures isolated to single stage, can retry
6. **Observability**: Detailed metrics, tracing, profiling
7. **Flexibility**: Easy to add/remove/reorder stages

### Negative

1. **Complexity**: More code than monolithic approach
2. **Overhead**: Caching adds <5% overhead
3. **Learning Curve**: Developers must understand stage concept
4. **Debugging**: Distributed logic harder to trace (mitigated with observability)

### Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Cache invalidation bugs | High | Medium | Deterministic cache keys, versioning |
| Performance regression | Medium | Low | Benchmark before/after |
| Over-engineering | Low | Medium | Start simple, add complexity only when needed |
| Cache storage growth | Medium | Medium | LRU eviction, configurable size limits |

---

## Implementation Plan

### Phase 1: Infrastructure (Week 3)

1. Implement `core/pipeline/stage.py` (base class, caching, timing)
2. Implement `core/pipeline/graph.py` (DAG execution)
3. Implement `core/pipeline/policy.py` (context extraction, rules)
4. Write comprehensive tests (90%+ coverage)

**Success Criteria**:
- ✅ Stage infrastructure tests pass
- ✅ Caching correctness validated
- ✅ <5% overhead from abstraction

### Phase 2: Migrate Lux Depth V2 (Week 3-4)

1. Define stages: `DepthEstimationStage`, `MaterialSegmentationStage`, `ToneMappingStage`, `UpscalingStage`
2. Build graph: `build_lux_depth_graph(config)`
3. Integrate policy engine
4. Verify 66/66 tests pass
5. Benchmark performance

**Success Criteria**:
- ✅ All tests pass
- ✅ 10-20x speedup with warm cache
- ✅ Stage metrics available
- ✅ Zero feature regressions

### Phase 3: Documentation (Week 4)

1. Write stage development guide
2. Update pipeline architecture docs
3. Create examples for new pipelines
4. Document policy engine rules

---

## Examples

### Example 1: Lux Depth V2 Stage Graph

```python
# lux_depth_v2/stages.py
from transformation_portal.core.pipeline.stage import Stage

class DepthEstimationStage(Stage):
    def __init__(self, model):
        super().__init__(name="depth_estimation", version="1.0.0")
        self.model = model
    
    def execute(self, input_data, config):
        with torch.inference_mode():
            depth = self.model(input_data)
        return depth

# lux_depth_v2/pipeline.py
def build_graph(config):
    stages = [
        DepthEstimationStage(load_depth_model()),
        MaterialSegmentationStage(load_segmenter()),
        ToneMappingStage(),
        UpscalingStage()
    ]
    return PipelineGraph(stages)
```

### Example 2: Context-Aware Processing

```python
# Extract context
context = {
    "resolution": (7680, 4320),
    "is_uhd": True,
    "is_hdr": False
}

# Policy engine adjusts config
config = policy_engine.apply(context, base_config)
# Result: {"enable_tiling": True, "tile_size": 512}

# Execute graph
result = graph.execute(image, config)
```

### Example 3: Cache Hit Rate Reporting

```python
metrics = graph.get_metrics()
# {
#   "depth_estimation": {"duration_ms": 45, "cache_hit": False},
#   "material_segmentation": {"duration_ms": 30, "cache_hit": True},
#   "tone_mapping": {"duration_ms": 12, "cache_hit": False},
#   "upscaling": {"duration_ms": 150, "cache_hit": False}
# }

cache_hit_rate = sum(m["cache_hit"] for m in metrics.values()) / len(metrics)
# 0.25 (25% cache hit rate)
```

---

## Backward Compatibility

### Strategy

**Facade Pattern**:
```python
# Old API remains functional
class LuxDepthPipeline:
    def process(self, image: np.ndarray) -> np.ndarray:
        """Legacy API delegates to stage graph."""
        graph = build_graph(self.config)
        return graph.execute(image, self.config)
```

**Migration Path**:
1. Week 1: Stage graph available but not required
2. Week 2: Lux Depth V2 migrated, old API still works
3. Week 3+: New pipelines use stage graph exclusively
4. After 2 releases: Deprecate monolithic approach

---

## Success Metrics

### Performance Metrics

- ✅ **Cache Speedup**: 10-20x on warm cache
- ✅ **Overhead**: <5% when cache cold
- ✅ **Cache Hit Rate**: 60%+ in iterative workflows

### Quality Metrics

- ✅ **Test Coverage**: 90%+ on stage infrastructure
- ✅ **Zero Regressions**: All existing tests pass
- ✅ **Determinism**: Identical inputs produce identical outputs

### Observability Metrics

- ✅ **Stage Timing**: Sub-millisecond granularity
- ✅ **Cache Metrics**: Hit/miss rate per stage
- ✅ **Bottleneck Identification**: Slowest stage reported

---

## Alternatives Considered

### Alternative 1: Keep Monolithic Pipeline

**Pros**: Simplicity  
**Cons**: No caching, poor observability  
**Verdict**: ❌ Doesn't meet efficiency goals

### Alternative 2: External Workflow Engine (Airflow, Prefect)

**Pros**: Battle-tested, rich features  
**Cons**: Massive overhead, requires infrastructure  
**Verdict**: ❌ Overkill for single-machine processing

### Alternative 3: Lightweight Stage Graph (Selected)

**Pros**: Right-sized for use case, embedded, no external dependencies  
**Cons**: Must implement caching ourselves  
**Verdict**: ✅ **Best fit**

---

## Related Decisions

- **ADR-001**: Platform Core (provides caching infrastructure)
- **ADR-004**: Performance Optimization (uses stage metrics)
- **ADR-006**: Checkpoint/Resume (uses stage boundaries)

---

## References

- [ARCHITECTURE_HARDENING_PLAN.md](../ARCHITECTURE_HARDENING_PLAN.md)
- [Airflow Concepts](https://airflow.apache.org/docs/apache-airflow/stable/concepts/index.html)
- [Luigi Pipelines](https://luigi.readthedocs.io/en/stable/)
- [Functional Core, Imperative Shell](https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell)

---

**Decision**: ✅ **APPROVED**  
**Implementation**: PR-3 (Stage Graph Refactor)  
**Timeline**: Week 3-4  
**Next Review**: 2025-12-22
