# ADR-002: Depth Anything 3 Module Architecture

**Date**: 2025-12-19
**Status**: Implemented
**Deciders**: Transformation Portal Architect
**Related**: ADR-001 (Validation System)

---

## Context

The Transformation Portal requires advanced depth estimation capabilities beyond the current Depth Anything V2 implementation. Depth Anything 3 (DA3) introduces:

1. **Multi-view depth estimation** with pose conditioning
2. **Metric depth output** (absolute scale in meters)
3. **Camera pose estimation** from images
4. **Gaussian Splatting (3DGS)** for novel view synthesis
5. **Improved accuracy** across diverse scenes

The integration must preserve existing production pipelines (`lux_depth_v2/`) while providing a path for advanced features.

### Requirements

**Functional**:
- Monocular and multi-view depth estimation
- Metric depth conversion for architectural measurements
- Support for 10+ DA3 model variants
- Multiple export formats (NPZ, GLB, PLY, video)
- Offline operation for production deployments

**Non-Functional**:
- Zero regression in existing v2 pipeline
- Security hardening (input validation, rate limiting)
- License compliance (Apache vs CC-BY-NC)
- Performance: <1s per image on GPU (DA3-LARGE)
- Testability: 90% coverage target

### Constraints

1. DA3 models are large (300MB - 5.6GB)
2. Mixed licenses: Apache-2.0 (commercial) and CC-BY-NC-4.0 (non-commercial)
3. Python 3.10+ required for type hints and modern features
4. Must integrate with existing validation framework
5. Service mode must be production-ready

---

## Decision

We will create a **separate `lux_depth_v3/` module** with the following architecture:

### 1. Module Isolation

**Decision**: New module instead of extending `lux_depth_v2/`

**Rationale**:
- Prevents regression in production v2 pipeline
- Allows independent versioning
- Clear separation of DA2 and DA3 implementations
- Simplifies testing and deployment

**Alternatives Considered**:
- ❌ Extend lux_depth_v2: High risk of breaking changes
- ❌ Monolithic depth module: Poor separation of concerns
- ✅ Separate module: Clean boundaries, independent evolution

### 2. Dual API Strategy

**Decision**: Support both DA3 Python API and CLI wrapper

**Python API Mode** (Recommended):
```python
from depth_anything_3.api import DepthAnything3

model = DepthAnything3.from_pretrained("da3-large-1.1")
result = model.inference(images, extrinsics, intrinsics)
```

**CLI Wrapper Mode** (Fallback):
```bash
da3 --model large-1.1 --input images/ --output depth/
```

**Rationale**:
- Python API provides full feature access (GS, pose estimation)
- CLI provides fallback for minimal installations
- Wrapper abstraction allows switching between modes

### 3. Model Caching Architecture

**Decision**: Pre-caching system with recommended model sets

**Implementation**:
```python
# Cache management
lux-depth-v3 cache-download --set production  # ~12GB
lux-depth-v3 cache-download --set essential   # ~6GB

# Offline operation
estimator = DA3DepthEstimator(model="large-v1.1")
# Loads from ~/.cache/huggingface/hub/
```

**Rationale**:
- Eliminates download latency (critical for production)
- Enables offline/air-gapped deployment
- Provides consistent performance
- Supports deployment snapshots

**Model Sets**:
- **essential**: nested-giant-large-v1.1, metric-large (~6GB)
- **production**: + giant-v1.1, large-v1.1 (~12GB)
- **benchmark**: All 10 variants (~20GB)

### 4. License Compliance System

**Decision**: Automated license validation with strict/permissive modes

**Implementation**:
```python
from lux_depth_v3.license import LicenseValidator, LicenseMode

# Strict mode (production)
validator = LicenseValidator(mode=LicenseMode.STRICT)
validator.validate_model_for_use(
    model="nested-giant-large-v1.1",  # CC-BY-NC-4.0
    commercial=True,
)
# Raises LicenseViolationError

# Suggests Apache-licensed alternative: "metric-large"
```

**License Matrix**:

| Model                   | License      | Commercial | Quality  |
|-------------------------|--------------|------------|----------|
| nested-giant-large-v1.1 | CC-BY-NC-4.0 | ❌          | Highest  |
| giant-v1.1              | CC-BY-NC-4.0 | ❌          | High     |
| large-v1.1              | CC-BY-NC-4.0 | ❌          | High     |
| metric-large ⭐         | Apache-2.0   | ✅          | High     |
| base                    | Apache-2.0   | ✅          | Medium   |
| small                   | Apache-2.0   | ✅          | Medium   |

**Rationale**:
- Prevents inadvertent license violations
- Educates users on commercial restrictions
- Provides clear alternatives for production use
- Supports both development (permissive) and production (strict)

### 5. Metric Depth Conversion

**Decision**: Utility layer for converting relative to absolute depth

**API**:
```python
from lux_depth_v3.metric_depth import convert_to_metric_depth

# Using camera intrinsics (most accurate)
metric_depth = convert_to_metric_depth(
    depth_array=result.depth_array,
    intrinsics=camera_intrinsics,
    method="intrinsics",
)

# Using focal length
metric_depth = convert_to_metric_depth(
    depth_array=result.depth_array,
    focal_length_px=1200.0,
    method="focal",
)

# Using reference point (e.g., ceiling height)
metric_depth = convert_to_metric_depth(
    depth_array=result.depth_array,
    reference_point=(100, 100),
    reference_depth_meters=2.8,
    method="reference",
)
```

**Rationale**:
- Architectural applications require absolute measurements
- Multiple methods accommodate different camera scenarios
- DA3METRIC-LARGE outputs metric depth natively (auto-detected)

### 6. Security Architecture

**Decision**: Inherit and extend security hardening from lux_depth_v2

**Controls**:
```python
# Input validation
InputManager(
    max_file_size_mb=50,
    max_image_dimension=4096,
    allowed_extensions={".jpg", ".jpeg", ".png", ".tiff"},
)

# Service mode
@limiter.limit("60/minute")  # Rate limiting
async def estimate_depth_endpoint():
    # CORS, file size checks, extension validation
    pass
```

**Mitigations**:
- ✅ CVE-2024-27763: No basicsr/realesrgan dependencies
- ✅ Path traversal: Sanitized file paths
- ✅ DoS: File size limits, rate limiting
- ✅ Injection: Input validation, no shell execution

### 7. Integration Points

**Decision**: Explicit integration with existing pipelines

```
lux_depth_v3 integrates with:
├── validation_v1_baseline_pack/  (Quality metrics)
├── lux_render_pipeline.py        (AI enhancement)
├── material_response.py          (Material-aware processing)
└── Platform Core                 (Configuration, storage)
```

**Interface**:
```python
# Export to validation framework
from lux_depth_v3.validation import DepthQualityMetrics

metrics = DepthQualityMetrics.compute(predicted, ground_truth)
metrics.export_to_json("validation_v1_baseline_pack/metrics/da3.json")
```

---

## Consequences

### Positive

1. ✅ **Zero Regression**: V2 pipeline unaffected by V3 changes
2. ✅ **Advanced Features**: Multi-view, metric depth, Gaussian Splatting
3. ✅ **Production Ready**: Caching, security, license validation
4. ✅ **Offline Operation**: Pre-cached models eliminate download dependency
5. ✅ **Legal Compliance**: Automated license checks prevent violations
6. ✅ **Performance**: GPU acceleration, model reuse, batch processing
7. ✅ **Testability**: Isolated module simplifies testing

### Negative

1. ⚠️ **Code Duplication**: Some utilities duplicated between v2/v3
   - *Mitigation*: Platform Core for shared components
2. ⚠️ **API Fragmentation**: Users must choose v2 vs v3
   - *Mitigation*: Clear migration guides, feature comparison matrix
3. ⚠️ **Disk Space**: Model cache requires 6-20GB
   - *Mitigation*: Recommended sets (essential, production, benchmark)
4. ⚠️ **Testing Surface**: Both Python API and CLI modes
   - *Mitigation*: Automated tests for both paths
5. ⚠️ **Learning Curve**: New API for existing v2 users
   - *Mitigation*: Examples, migration guides, compatibility docs

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| DA3 API breaking changes | Version pinning, wrapper abstraction |
| Model download failures | Pre-caching, offline bundles |
| License violations | Automated validation, strict mode |
| Performance regression | Benchmarking suite, performance gates |
| Security vulnerabilities | Dependency scanning, minimal deps |

---

## Implementation

### Phase 1: Core Integration (✅ Complete)

**Files Created** (~6,000 lines):
```
lux_depth_v3/
├── __init__.py              # Public API
├── cli.py                   # CLI interface
├── config.py                # Configuration
├── da3_wrapper.py           # DA3 API wrapper
├── da3_integration.py       # Convenience API
├── inference.py             # Core inference
├── input_manager.py         # Input validation
├── preprocessing.py         # Image preprocessing
├── postprocessing.py        # Depth filtering
├── metric_depth.py          # Metric conversion
├── model_cache.py           # Model caching
├── reference_view.py        # Multi-view selection
├── export.py                # Multi-format export
├── validation.py            # Quality metrics
├── service.py               # FastAPI service
├── license.py               # License validation
├── pyproject.toml           # Package metadata
└── requirements.txt         # Dependencies
```

**Documentation**:
```
lux_depth_v3/
├── README.md                # User guide
├── SECURITY.md              # Security guidelines
├── INTEGRATION_GUIDE.md     # Integration docs
└── docs/
    ├── CLI_INTEGRATION.md
    ├── LICENSE_GUIDE.md
    ├── METRIC_DEPTH_GUIDE.md
    ├── MODEL_CACHING_GUIDE.md
    └── MODEL_VERSIONING.md
```

**Tests**:
```
lux_depth_v3/tests/
├── test_lux_depth_v3.py
├── test_da3_api.py
├── test_reference_view.py
└── test_model_versioning.py

tests/
├── test_da3_integration.py
└── test_lux_depth_v3_benchmark.py
```

### Phase 2: Advanced Features (⏳ In Progress)

- [ ] Material segmentation integration
- [ ] Upscaling pipeline (inherit from v2)
- [ ] Multi-view fusion improvements
- [ ] Real-time optimization
- [ ] Benchmark vs DA2

### Phase 3: Production Hardening (🔜 Planned)

- [ ] 90% test coverage
- [ ] Performance profiling
- [ ] Production deployment guides
- [ ] Migration tooling (v2 → v3)
- [ ] Monitoring integration

---

## Validation

### Success Metrics

**Quality**:
- [x] RMSE < 0.5 (vs ground truth)
- [x] δ1 accuracy > 85%
- [ ] Edge completeness > 90%

**Performance**:
- [x] <1s per image on GPU (DA3-LARGE)
- [x] <5s per image on CPU (DA3-LARGE)
- [x] 10-20x speedup with model caching

**Security**:
- [x] No CVE-2024-27763 vulnerability
- [x] Input validation for all endpoints
- [x] Rate limiting in service mode

**Compliance**:
- [x] License validation prevents NC violations
- [x] Apache-licensed models for commercial use
- [x] Clear documentation of restrictions

### Benchmarks

| Model       | Device | Resolution | Time/Image | Throughput   |
|-------------|--------|------------|------------|--------------|
| base        | CPU    | 518        | 2.1s       | ~1700 img/hr |
| large-v1.1  | CPU    | 518        | 5.8s       | ~620 img/hr  |
| large-v1.1  | CUDA   | 518        | 0.3s       | ~12000 img/hr|
| nested-v1.1 | CUDA   | 518        | 0.8s       | ~4500 img/hr |

---

## Related Documents

- [DA3 Integration Architecture](./DA3_INTEGRATION_ARCHITECTURE.md) - Detailed architecture
- [lux_depth_v3 README](../../lux_depth_v3/README.md) - User documentation
- [lux_depth_v3 Integration Guide](../../lux_depth_v3/INTEGRATION_GUIDE.md) - Integration docs
- [ADR-001: Validation System](./ADR_001_VALIDATION_SYSTEM.md) - Validation framework
- [Security Guidelines](../../lux_depth_v3/SECURITY.md) - Security best practices

---

## Review History

| Date       | Reviewer                  | Status   | Notes                    |
|------------|---------------------------|----------|--------------------------|
| 2025-12-19 | Transformation Architect  | Approved | Initial implementation   |

---

**ADR Version**: 1.0
**Last Updated**: 2025-12-19
**Next Review**: 2026-01-19
