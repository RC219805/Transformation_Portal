# ADR-001: Platform Core Extraction

**Status**: Proposed  
**Date**: 2025-12-08  
**Authors**: Transformation Portal Architect  
**Related PRs**: PR-2 (Platform Core)

---

## Context

Transformation Portal has evolved organically from a collection of scripts into a suite of pipelines (Lux Depth V2, video grading, TIFF processing, material response). This evolution has led to:

1. **Duplicated Infrastructure**: Each pipeline reimplements config loading, device detection, logging, and caching
2. **Inconsistent Patterns**: 5+ different CLI patterns, 3 logging approaches, varying config formats
3. **Coupling**: Pipelines import from each other, creating circular dependencies
4. **Maintenance Burden**: Bug fixes require changes across multiple files
5. **Testing Complexity**: Infrastructure code tested redundantly in each pipeline

**Example Duplication**:
- Device detection: Implemented in `lux_depth_v2/pipeline.py`, `luxury_video_master_grader.py`, `depth_tools.py`
- Config loading: YAML parsers in 4+ files with different error handling
- Caching: 3 different cache implementations with incompatible keys
- Logging: Mix of `print()`, `logging`, and structured logs

**Current State**:
- Lux Depth V2 has cleanest architecture but still duplicates device/config code
- Legacy pipelines have tightly coupled, hard-to-test implementations
- No single source of truth for best practices

---

## Decision

We will extract common infrastructure into a **Platform Core** module that provides:

1. **Unified Configuration**: Pydantic schemas, validators, preset registry
2. **Device Management**: CPU/CUDA/MPS detection, allocation, profiling
3. **Artifact Storage**: Content-addressed caching with manifests
4. **Security Primitives**: Path validation, safe image loading
5. **Observability**: Structured logging, metrics, tracing hooks

### Architecture

```
transformation_portal/
├── core/
│   ├── config/          # Single source of truth for configuration
│   │   ├── schema.py    # Pydantic models
│   │   ├── validator.py # Validation rules
│   │   ├── loader.py    # YAML/JSON safe loading
│   │   └── presets.py   # Preset registry
│   ├── device/          # Device management
│   │   ├── manager.py   # CPU/CUDA/MPS detection
│   │   ├── profiler.py  # Performance profiling
│   │   └── fallback.py  # Graceful degradation
│   ├── artifacts/       # Caching infrastructure
│   │   ├── store.py     # Content-addressed storage
│   │   ├── manifest.py  # Cache manifests
│   │   └── hashing.py   # SHA256 utilities
│   ├── security/        # Security layer
│   │   ├── paths.py     # Path validation
│   │   ├── images.py    # Safe image loading
│   │   └── validation.py # Input sanitization
│   └── observability/   # Logging & monitoring
│       ├── logging.py   # Structured logs
│       ├── metrics.py   # Prometheus-compatible
│       └── tracing.py   # OpenTelemetry hooks
```

### API Contracts

**Config Schema**:
```python
from transformation_portal.core.config import ProcessingConfig, DeviceConfig

config = ProcessingConfig(
    input_path="input.jpg",
    output_dir="output/",
    preset="interior_luxury",
    device=DeviceConfig(device="auto", enable_amp=True)
)
```

**Device Manager**:
```python
from transformation_portal.core.device import DeviceManager

manager = DeviceManager(preferred="auto")
tensor = manager.allocate_tensor((3, 512, 512))
metrics = manager.profile()
```

**Artifact Store**:
```python
from transformation_portal.core.artifacts import ArtifactStore

store = ArtifactStore(cache_dir=".cache")
cached = store.get(input_path, config_dict)
if cached is None:
    result = expensive_computation(input_path)
    store.put(input_path, config_dict, result)
```

**Security**:
```python
from transformation_portal.core.security import PathValidator

validator = PathValidator(allowed_base="/data/uploads")
safe_path = validator.validate(user_input)
```

---

## Consequences

### Positive

1. **Reduced Duplication**: 50% reduction in infrastructure code
2. **Consistency**: All pipelines use same patterns
3. **Testability**: Core infrastructure tested once, thoroughly
4. **Maintainability**: Bug fixes in one place
5. **Documentation**: Single reference for best practices
6. **Onboarding**: New contributors learn patterns once
7. **Security**: Centralized validation and sanitization

### Negative

1. **Refactoring Effort**: 1-2 weeks to extract and test
2. **Migration Complexity**: Existing pipelines need updates
3. **Breaking Changes**: (Mitigated with backward compatibility layer)
4. **Learning Curve**: Developers must learn new APIs

### Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking existing code | High | Medium | Maintain backward compatibility, deprecation warnings |
| Performance regression | Medium | Low | Benchmark before/after, optimize hot paths |
| Test failures | Medium | Low | Comprehensive test coverage (90%+) on core |
| Adoption resistance | Low | Medium | Clear migration guides, examples, gradual rollout |

---

## Implementation Plan

### Phase 1: Extract Core (Week 1)

1. Create `transformation_portal/core/` module structure
2. Implement each submodule with API contracts
3. Write comprehensive tests (target: 90%+ coverage)
4. Add to CI without requiring usage

**Success Criteria**:
- ✅ All core tests pass
- ✅ 90%+ test coverage
- ✅ Zero external dependencies (self-contained)
- ✅ Documentation complete

### Phase 2: Migrate Lux Depth V2 (Week 1-2)

1. Refactor `lux_depth_v2/config.py` to use `core.config`
2. Replace device detection with `core.device.DeviceManager`
3. Integrate `core.artifacts.ArtifactStore` for depth caching
4. Verify 66/66 tests still pass

**Success Criteria**:
- ✅ All Lux Depth V2 tests pass
- ✅ Performance neutral or improved
- ✅ No feature regressions

### Phase 3: Migrate Legacy Pipelines (Week 2)

1. Update `luxury_video_master_grader.py`
2. Update `luxury_tiff_batch_processor.py`
3. Add deprecation warnings to old patterns
4. Update examples and documentation

**Success Criteria**:
- ✅ All pipeline tests pass
- ✅ Deprecation warnings active
- ✅ Migration guides complete

### Phase 4: Deprecate Old Patterns (Week 3+)

1. Remove deprecated code after 2 release cycles
2. Enforce core usage in CI (linting rules)
3. Update all documentation

---

## Backward Compatibility

### Strategy

**Gradual Migration**:
- Old APIs remain functional with deprecation warnings
- New code uses core modules exclusively
- 2-release deprecation cycle before removal

**Compatibility Layer**:
```python
# lux_depth_v2/config.py (legacy)
import warnings
from transformation_portal.core.config import ProcessingConfig as _CoreConfig

class PipelineConfig(_CoreConfig):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "lux_depth_v2.config.PipelineConfig is deprecated. "
            "Use transformation_portal.core.config.ProcessingConfig instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
```

### Migration Guide

**Before (Legacy)**:
```python
# Duplicated device detection
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps"):
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

**After (Platform Core)**:
```python
# Unified device management
from transformation_portal.core.device import DeviceManager

manager = DeviceManager(preferred="auto")
device = manager.device
```

---

## Success Metrics

### Technical Metrics

- ✅ **Code Reduction**: 50% reduction in duplicated infrastructure
- ✅ **Test Coverage**: 90%+ on core modules
- ✅ **Performance**: <5% overhead from abstraction
- ✅ **Reliability**: Zero regressions in pipeline functionality

### Operational Metrics

- ✅ **Adoption**: 100% of new code uses core modules
- ✅ **Migration**: All production pipelines migrated within 3 weeks
- ✅ **Documentation**: Complete migration guides, API reference
- ✅ **Developer Satisfaction**: Positive feedback on consistency

---

## Alternatives Considered

### Alternative 1: Keep Status Quo

**Pros**: No refactoring effort required  
**Cons**: Continued duplication, inconsistency, maintenance burden  
**Verdict**: ❌ Technical debt compounds over time

### Alternative 2: Refactor Individual Pipelines

**Pros**: Incremental improvement  
**Cons**: Duplication remains, no shared benefits  
**Verdict**: ❌ Doesn't solve root cause

### Alternative 3: Microservices Architecture

**Pros**: Strong module boundaries  
**Cons**: Massive overhead, premature complexity  
**Verdict**: ❌ Overkill for current scale

### Alternative 4: Platform Core (Selected)

**Pros**: Shared infrastructure, consistency, maintainability  
**Cons**: Upfront refactoring effort  
**Verdict**: ✅ **Best balance of benefits and costs**

---

## Related Decisions

- **ADR-002**: Stage Graph Architecture (builds on Platform Core)
- **ADR-003**: Security Hardening (uses `core.security`)
- **ADR-004**: Performance Optimization (uses `core.device.profiler`)
- **ADR-005**: Validation Integration (uses `core.validation`)

---

## References

- [ARCHITECTURE_HARDENING_PLAN.md](../ARCHITECTURE_HARDENING_PLAN.md)
- [Lux Depth V2 Architecture](../../lux_depth_v2/README.md)
- [Python Dependency Injection Patterns](https://python-dependency-injector.ets-labs.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

**Decision**: ✅ **APPROVED**  
**Implementation**: PR-2 (Platform Core Extraction)  
**Timeline**: Week 1-2  
**Next Review**: 2025-12-22
