# V2 Enhancement Quick Reference

**Status:** Approved
**See Full Guidance:** `V2_ENHANCEMENT_ARCHITECTURAL_GUIDANCE.md`

---

## Quick Answers

### 1. What is V2 Enhancement?

**Depth-aware perceptual finishing for luxury real estate marketing output.**

- Tone mapping (depth-guided foreground/background separation)
- Clarity enhancement (multi-scale unsharp masking)
- Material-aware processing (wood warmth, metal highlights, etc.)
- Atmospheric effects (ambient occlusion, haze, light wrap)

**NOT:** Diffusion rendering, ML upscaling, content generation

---

### 2. Where Does Implementation Live?

**Primary:**
- `src/transformation_portal/lux_depth_v3/v2_enhance.py` (NEW)
- `src/transformation_portal/lux_depth_v3/v2_presets.py` (NEW)

**Reuses:**
- `src/transformation_portal/stage_graph/stages/enhancement.py` (core logic)
- `src/transformation_portal/lux_depth_v3/materials_v3_response.py` (material utilities)

**Entry Point:**
- `scripts/enhance_image.py` (update from passthrough to call `v2_enhance.enhance_image()`)

---

### 3. What Are Default Presets?

| Preset | Use Case | Strength | Clarity | Materials | Atmosphere |
|--------|----------|----------|---------|-----------|------------|
| `default` | Balanced | 0.7 | 0.5 | 0.6 | ✅ |
| `luxury_estate` | Premium marketing | 0.8 | 0.6 | 0.7 | ✅ |
| `architectural` | Technical viz | 0.6 | 0.7 | 0.5 | ❌ |
| `none` | Skip V2 | - | - | - | - |

---

### 4. ML Models or Image Processing?

**Decision: Image Processing Only**

**Allowed:**
- ✅ numpy, scipy, Pillow (core tier)
- ✅ scikit-image (optional, core tier)

**Banned for V2:**
- ❌ torch, diffusers, transformers
- ❌ Real-ESRGAN (explicitly banned)
- ❌ Any ML inference

**Upscaling:**
- Use Pillow's `LANCZOS` resampling
- Fallback: scipy bicubic
- NO ML upscaling (future ADR required)

**Rationale:**
- Commercial safety (BSD/MIT licenses only)
- Performance (<2s/image target)
- Installation footprint (~500MB vs ~10GB)
- Maintainability (stable APIs)

---

### 5. Integration Pattern

**Flow:**
```
scripts/enhance_image.py (CLI)
    ↓
v2_enhance.py (orchestration)
    ↓
EnhancementStage (core logic)
    ↓
materials_v3_response.py (material utilities)
```

**Reuse, Don't Reimplement:**
- `EnhancementStage` already has tone mapping, clarity, material processing
- `materials_v3_response.py` has material-aware utilities
- V2 is the *orchestration layer*, not a new implementation

---

## Implementation Checklist

### Phase 1: Core Implementation
- [ ] Create `src/transformation_portal/lux_depth_v3/v2_enhance.py`
- [ ] Create `src/transformation_portal/lux_depth_v3/v2_presets.py`
- [ ] Update `scripts/enhance_image.py` (passthrough → actual enhancement)
- [ ] Add unit tests (`tests/unit/lux_depth_v3/test_v2_enhance.py`)
- [ ] Add integration tests (`tests/integration/test_v2_orchestrator_integration.py`)

### Phase 2: Documentation
- [ ] Update README.md (V2 Enhancement section)
- [ ] Update CHANGELOG.md
- [ ] Add docstrings + type hints
- [ ] Create `docs/architecture/v2_enhancement_design.md`

### Phase 3: Validation
- [ ] Performance test (assert <2s/image, 400+ images/hour)
- [ ] Dependency audit (no ML imports)
- [ ] Backward compatibility (all existing tests pass)
- [ ] CI gates (lint, type check, tests)

---

## Performance Targets

| Metric | Target | Enforcement |
|--------|--------|-------------|
| Per-image time | <2s typical, <5s max | Performance test |
| Throughput | 400-600 images/hour | Benchmark test |
| Memory | <2GB peak | Profiling |
| Dependencies | Core tier only | Import scanner |

**Breakdown (1024x768):**
- I/O: ~50ms
- Depth load: ~30ms
- Tone mapping: ~100ms
- Clarity: ~200ms
- Materials: ~300ms
- Atmosphere: ~150ms
- **Total: ~830ms** ✅

---

## CLI Examples

```bash
# Default preset
lux-depth-v3 enhance input.jpg --output-dir ./output

# Luxury estate preset
lux-depth-v3 enhance input.jpg \
    --enable-v2 on \
    --v2-preset luxury_estate \
    --output-dir ./output

# Skip V2 (PBR only)
lux-depth-v3 enhance input.jpg \
    --enable-v2 off \
    --output-dir ./output

# Architectural preset
lux-depth-v3 enhance input.jpg \
    --v2-preset architectural \
    --output-dir ./output
```

---

## Python API Example

```python
from pathlib import Path
from transformation_portal.lux_depth_v3.v2_enhance import (
    enhance_image,
    V2EnhancementConfig,
)

config = V2EnhancementConfig.from_preset("luxury_estate")

report = enhance_image(
    input_path=Path("input.jpg"),
    output_path=Path("output/enhanced.jpg"),
    depth_map_path=Path("depth/input_depth.tiff"),
    config=config,
)

print(f"Status: {report['status']}")
```

---

## Testing Strategy

### Unit Tests
```python
# tests/unit/lux_depth_v3/test_v2_enhance.py

def test_enhance_image_default_preset():
    """Verify default preset produces valid output."""

def test_enhance_image_with_depth_map():
    """Verify depth-aware tone mapping."""

def test_enhance_image_without_depth_map():
    """Verify graceful degradation without depth."""

def test_preset_loading():
    """Verify all presets load correctly."""

def test_performance_budget():
    """Verify <2s/image budget."""
```

### Integration Tests
```python
# tests/integration/test_v2_orchestrator_integration.py

def test_orchestrator_v2_with_depth_maps():
    """End-to-end: V3 depth → V2 enhancement."""

def test_orchestrator_v2_preset_selection():
    """Verify preset propagation."""
```

---

## Key Constraints

### Backward Compatibility
- ✅ All existing V2Runner tests must pass
- ✅ CLI flags unchanged (`--enable-v2`, `--v2-preset`)
- ✅ Report JSON format preserved
- ✅ Orchestrator fail-fast validation preserved

### Security
- ✅ No ML dependencies in v2_enhance.py
- ✅ Path traversal prevention (input validation)
- ✅ Dependency audit in CI

### Maintainability
- ✅ Reuse existing components (no duplication)
- ✅ Clear separation of concerns
- ✅ Comprehensive documentation
- ✅ Type hints + docstrings

---

## Approval Status

**Architect Decision:** ✅ **APPROVED**

**Rationale:**
- Clear scope (image processing only)
- Commercially safe (BSD/MIT dependencies)
- Performance compliant (<2s/image)
- Backward compatible
- Well-defined integration pattern

**Priority:** High
**Complexity:** Medium
**Risk:** Low
**Estimated Effort:** 14-21 hours (2-3 days)

---

## References

- **Full Guidance:** `docs/architecture/decisions/V2_ENHANCEMENT_ARCHITECTURAL_GUIDANCE.md`
- **ADR-022:** `docs/architecture/ADR-022-v2-enhancement-optional.md`
- **EnhancementStage:** `src/transformation_portal/stage_graph/stages/enhancement.py`
- **Materials V3:** `src/transformation_portal/lux_depth_v3/materials_v3_response.py`
- **lux_render_pipeline:** `src/transformation_portal/pipelines/lux_render_pipeline.py`

---

**Last Updated:** 2025-02-07
**Authority:** Transformation Portal Architect
