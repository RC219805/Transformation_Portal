# V2 Enhancement Architectural Guidance

**Status:** Authoritative Guidance
**Date:** 2025-02-07
**Authority:** Transformation Portal Architect
**Context:** Implementation planning for V2 depth-aware enhancement functionality
**Related:** ADR-022, EnhancementStage, lux_render_pipeline.py, Materials V3

---

## Executive Summary

**V2 Enhancement is depth-aware image finishing for luxury real estate marketing**, distinct from diffusion-based rendering. It consumes depth maps from the V3 stage and applies perceptual improvements (tone mapping, clarity, material-aware processing) without ML model dependencies.

**Key Decision:** V2 should be **image processing only** (no ML models), reusing existing `EnhancementStage` + material response utilities, with a new `v2_enhance.py` module for orchestrator integration.

---

## 1. Scope and Capabilities of V2 Enhancement

### What V2 Enhancement IS

**Core Function:** Depth-aware perceptual finishing for marketing-ready output

**Primary Capabilities:**
1. **Depth-Aware Tone Mapping**
   - Foreground subject enhancement (boost highlights/clarity on primary subject)
   - Background atmospheric handling (subtle compression for depth perception)
   - Preserves spatial hierarchy established by V3 depth maps

2. **Clarity Enhancement**
   - Multi-scale unsharp masking for detail revelation
   - Edge-preserving sharpening (prevents halo artifacts)
   - Material-aware strength modulation

3. **Material-Specific Processing** (leveraging Materials V3 taxonomy)
   - Wood: warmth boost + grain enhancement
   - Metal: highlight enhancement + contrast lift
   - Glass: subtle highlight boost + transparency preservation
   - Textiles: micro-contrast for fabric texture
   - Leather: sheen enhancement
   - Follows existing material response patterns from `lux_render_pipeline.py` lines 300-520

4. **Atmospheric Effects**
   - Ambient occlusion (grounding for furniture/floor contact)
   - Depth-based haze/atmosphere for exterior scenes
   - Light wrap simulation (window reflections, fireplace glow)

5. **Color Grading** (optional preset-dependent)
   - Highlight warmth (luxury aesthetic)
   - Shadow toning
   - Saturation refinement

### What V2 Enhancement IS NOT

**Explicitly Out of Scope:**
- ❌ Diffusion-based generation (SDXL, ControlNet, Flux)
- ❌ Upscaling via ML models (Real-ESRGAN, ESRGAN, SD-Upscale)
- ❌ Inpainting or content generation
- ❌ Material segmentation (consumed from Materials V3, not produced)
- ❌ Depth estimation (consumed from V3, not produced)

**Rationale:**
- **Dependency governance:** Avoid ML model dependencies to maintain fast, deterministic processing
- **Commercial safety:** Image processing operations have clear licensing (BSD/MIT), ML models introduce complexity (see ADR-0015)
- **Performance:** Target 400-600 images/hour requires lightweight operations (<2s/image typical)
- **Maintainability:** Image processing is stable; diffusion pipelines require ongoing maintenance

---

## 2. Implementation Location and Structure

### Recommended Module Structure

```
src/transformation_portal/lux_depth_v3/
├── v2_enhance.py              # NEW: V2 enhancement implementation
├── v2_runner.py               # EXISTING: Subprocess wrapper (calls scripts/enhance_image.py)
└── orchestrator.py            # EXISTING: Consumes V2Runner

src/transformation_portal/stage_graph/stages/
└── enhancement.py             # EXISTING: Reusable enhancement logic

src/transformation_portal/lux_depth_v3/
├── materials_v3_response.py   # EXISTING: Material-aware processing utilities
└── materials_v3_taxonomy.py   # EXISTING: Material classification
```

### Primary Implementation: `src/transformation_portal/lux_depth_v3/v2_enhance.py`

**Design:**
```python
"""V2 depth-aware enhancement implementation.

Consumes depth maps from V3 stage and applies perceptual finishing
for luxury real estate marketing output.

Dependencies:
    - numpy, scipy (core)
    - PIL (image I/O)
    - EnhancementStage (reuse existing enhancement logic)
    - materials_v3_response (material-aware utilities)

NO ML DEPENDENCIES (torch, diffusers, transformers, etc.)
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from transformation_portal.stage_graph.stages.enhancement import EnhancementStage
from transformation_portal.lux_depth_v3.materials_v3_response import (
    generate_response_plan,
)
from transformation_portal.lux_depth_v3.materials_v3_taxonomy import (
    MaterialClass,
)


class V2EnhancementConfig:
    """Configuration for V2 enhancement."""

    def __init__(
        self,
        preset: str = "default",
        enhancement_strength: float = 0.7,
        clarity_strength: float = 0.5,
        material_strength: float = 0.6,
        depth_aware_tone_mapping: bool = True,
        atmospheric_effects: bool = True,
    ):
        self.preset = preset
        self.enhancement_strength = enhancement_strength
        self.clarity_strength = clarity_strength
        self.material_strength = material_strength
        self.depth_aware_tone_mapping = depth_aware_tone_mapping
        self.atmospheric_effects = atmospheric_effects

    @classmethod
    def from_preset(cls, preset: str) -> "V2EnhancementConfig":
        """Load configuration from preset name."""
        # Define preset-specific configurations
        if preset == "luxury_estate":
            return cls(
                preset=preset,
                enhancement_strength=0.8,
                clarity_strength=0.6,
                material_strength=0.7,
                depth_aware_tone_mapping=True,
                atmospheric_effects=True,
            )
        elif preset == "architectural":
            return cls(
                preset=preset,
                enhancement_strength=0.6,
                clarity_strength=0.7,
                material_strength=0.5,
                depth_aware_tone_mapping=True,
                atmospheric_effects=False,
            )
        else:  # default
            return cls(preset=preset)


def enhance_image(
    input_path: Path,
    output_path: Path,
    depth_map_path: Optional[Path] = None,
    material_masks: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[V2EnhancementConfig] = None,
) -> Dict[str, Any]:
    """
    Apply V2 depth-aware enhancement to input image.

    Args:
        input_path: Path to input image
        output_path: Path to output enhanced image
        depth_map_path: Optional path to depth map from V3 stage
        material_masks: Optional material segmentation masks
        config: Enhancement configuration

    Returns:
        Dict containing processing metadata
    """
    if config is None:
        config = V2EnhancementConfig()

    # Load image
    image = np.array(Image.open(input_path))

    # Load depth map if provided
    depth_map = None
    if depth_map_path and depth_map_path.exists():
        depth_map = np.array(Image.open(depth_map_path))
        # Normalize depth to [0, 1] if needed
        if depth_map.max() > 1.0:
            depth_map = depth_map / depth_map.max()

    # Apply enhancement using existing EnhancementStage
    enhancer = EnhancementStage(
        enhancement_strength=config.enhancement_strength,
        clarity_strength=config.clarity_strength,
        material_strength=config.material_strength,
    )

    # Create minimal context for stage execution
    from transformation_portal.stage_graph.stage import StageContext

    context = StageContext(device="cpu")
    context.set_artifact("image", image)
    if depth_map is not None:
        context.set_artifact("depth_map", depth_map)
    if material_masks:
        context.set_artifact("material_masks", material_masks)

    # Execute enhancement
    result = enhancer.compute(context)

    if result.status != "completed":
        raise RuntimeError(f"Enhancement failed: {result.error}")

    enhanced_image = result.artifacts["enhanced_image"]

    # Save output
    Image.fromarray(enhanced_image).save(output_path)

    return {
        "status": "success",
        "input": str(input_path),
        "output": str(output_path),
        "depth_map": str(depth_map_path) if depth_map_path else None,
        "preset": config.preset,
        "metadata": result.metadata,
    }
```

### Secondary: Update `scripts/enhance_image.py`

Replace passthrough with call to `v2_enhance.enhance_image()`:

```python
from transformation_portal.lux_depth_v3.v2_enhance import (
    enhance_image,
    V2EnhancementConfig,
)

def main() -> int:
    args = parse_arguments()
    configure_logging(args.verbose, args.quiet, args.log_file)

    # Build config from CLI args
    config = V2EnhancementConfig.from_preset(args.preset)

    # Find depth map if depth_dir provided
    depth_map_path = None
    if args.depth_dir:
        depth_map_path = find_depth_map(
            args.depth_dir,
            args.input_path.stem
        )

    # Enhance image
    report = enhance_image(
        input_path=args.input_path,
        output_path=args.output_dir / args.input_path.name,
        depth_map_path=depth_map_path,
        config=config,
    )

    # Write report
    report_path = args.output_dir / f"{args.input_path.stem}_report.json"
    atomic_write_json(report_path, report)

    return 0
```

---

## 3. Dependency Strategy

### Core Principle: Image Processing Only

**Approved Dependencies (Already in Core):**
- `numpy` - numerical operations
- `scipy` - filters (gaussian, bilateral, sobel)
- `Pillow` - image I/O
- `scikit-image` - optional (resize, transforms)

**Banned Dependencies for V2:**
- ❌ `torch` (ML framework - already in `[ml]` extra, not core)
- ❌ `diffusers` (diffusion models)
- ❌ `transformers` (language/vision models)
- ❌ `realesrgan` (explicitly banned, unmaintained)
- ❌ `opencv-python` for advanced operations (available in core but prefer scipy/skimage for portability)

**Rationale:**
1. **Commercial Safety:** Image processing libraries (BSD/MIT) vs ML models (mixed licenses, see ADR-0015)
2. **Installation Footprint:** Core dependencies ~500MB vs ML stack ~10GB
3. **Performance:** Image ops <2s/image vs model inference 5-30s/image
4. **Maintainability:** Stable scipy API vs evolving diffusion ecosystem

### Upscaling Strategy

**Decision: NO ML upscaling in V2**

**Options Analysis:**

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Real-ESRGAN | High quality | Banned (unmaintained), license complexity | ❌ REJECT |
| ESRGAN | Quality | Requires torch, complex setup | ❌ REJECT |
| SD-Upscale | Best quality | Requires diffusers, slow | ❌ REJECT |
| Lanczos (Pillow) | Fast, stable | Lower quality | ✅ DEFAULT |
| Bicubic (scipy/skimage) | Fast, deterministic | Acceptable quality | ✅ FALLBACK |

**Implementation:**
```python
def upscale_image(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Upscale image using Lanczos resampling.

    If scale == 1.0, returns image unchanged.
    For scale > 1.0, uses PIL's LANCZOS (high-quality downsampling-safe).
    """
    if abs(scale - 1.0) < 0.01:
        return image

    from PIL import Image
    h, w = image.shape[:2]
    new_size = (int(w * scale), int(h * scale))

    pil_img = Image.fromarray(image)
    upscaled = pil_img.resize(new_size, Image.LANCZOS)
    return np.array(upscaled)
```

**Future Migration Path (Separate ADR Required):**
If ML upscaling becomes critical:
1. Gate behind `[ml]` extra installation tier
2. Require explicit `--enable-ml-upscaler` flag
3. Fallback to Lanczos if ML unavailable
4. Document commercial licensing implications

---

## 4. Preset Definitions

### Default Presets

**`default`** (Balanced):
```yaml
preset: default
enhancement_strength: 0.7
clarity_strength: 0.5
material_strength: 0.6
depth_aware_tone_mapping: true
atmospheric_effects: true
upscale_factor: 1.0  # No upscaling
```

**`luxury_estate`** (Premium Marketing):
```yaml
preset: luxury_estate
enhancement_strength: 0.8
clarity_strength: 0.6
material_strength: 0.7
depth_aware_tone_mapping: true
atmospheric_effects: true
upscale_factor: 1.0
color_grading:
  highlight_warmth: 0.15
  shadow_toning: 0.05
```

**`architectural`** (Technical Visualization):
```yaml
preset: architectural
enhancement_strength: 0.6
clarity_strength: 0.7
material_strength: 0.5
depth_aware_tone_mapping: true
atmospheric_effects: false  # Preserve technical accuracy
upscale_factor: 1.0
```

**`none`** (Skip V2):
```yaml
# Handled by orchestrator: enable_v2=False
```

### Preset Loading

**Implementation:**
```python
# src/transformation_portal/lux_depth_v3/v2_presets.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml

@dataclass
class V2Preset:
    """V2 enhancement preset definition."""
    name: str
    enhancement_strength: float = 0.7
    clarity_strength: float = 0.5
    material_strength: float = 0.6
    depth_aware_tone_mapping: bool = True
    atmospheric_effects: bool = True
    upscale_factor: float = 1.0

    @classmethod
    def from_name(cls, name: str) -> "V2Preset":
        """Load preset by name."""
        if name == "luxury_estate":
            return cls(
                name=name,
                enhancement_strength=0.8,
                clarity_strength=0.6,
                material_strength=0.7,
            )
        elif name == "architectural":
            return cls(
                name=name,
                enhancement_strength=0.6,
                clarity_strength=0.7,
                material_strength=0.5,
                atmospheric_effects=False,
            )
        else:  # default
            return cls(name=name)

    @classmethod
    def from_yaml(cls, path: Path) -> "V2Preset":
        """Load preset from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

---

## 5. Integration Approach with Existing Code

### Integration Pattern: Reuse + Delegate

**Architecture:**
```
scripts/enhance_image.py (CLI entry point)
    ↓ calls
src/transformation_portal/lux_depth_v3/v2_enhance.py (business logic)
    ↓ delegates to
src/transformation_portal/stage_graph/stages/enhancement.py (core enhancement)
    ↓ uses
src/transformation_portal/lux_depth_v3/materials_v3_response.py (material utilities)
```

**Rationale:**
1. **Reuse existing enhancement logic** - `EnhancementStage` already implements tone mapping, clarity, material processing
2. **Maintain separation** - `v2_enhance.py` is the orchestration layer, not a reimplementation
3. **Preserve testability** - `EnhancementStage` remains testable independently
4. **Enable future migration** - `V2Runner` can eventually call module directly (ADR-022 migration path)

### Integration with V3 Orchestrator

**Current Flow:**
```
Orchestrator.run()
  ↓
V3 Depth Inference (produces depth maps)
  ↓
PBR Map Generation (consumes depth maps)
  ↓
V2 Enhancement (via V2Runner subprocess)
  ↓
Final Output
```

**Depth Map Discovery:**
```python
# In v2_enhance.py
def find_depth_map(depth_dir: Path, image_stem: str) -> Optional[Path]:
    """
    Find depth map for image.

    Expected naming conventions:
    - {image_stem}_depth.tiff (primary)
    - {image_stem}_depth.png
    - {image_stem}.tiff (fallback)
    """
    candidates = [
        depth_dir / f"{image_stem}_depth.tiff",
        depth_dir / f"{image_stem}_depth.png",
        depth_dir / f"{image_stem}.tiff",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None
```

### Material Mask Integration (Optional)

**Current State:** Materials V3 generates material masks but doesn't expose them to V2 yet

**Future Enhancement (Separate Work):**
```python
# In orchestrator.py (future)
def _prepare_v2_context(self, image_key: str) -> Dict[str, Any]:
    """Prepare context for V2 enhancement."""
    context = {}

    # Find depth map
    depth_path = self.depth_dir / f"{image_key}_depth.tiff"
    if depth_path.exists():
        context["depth_map_path"] = depth_path

    # Find material masks (if Materials V3 enabled)
    if self.config.enable_materials_v3:
        materials_dir = self.output_dir / "materials"
        material_masks = load_material_masks(materials_dir, image_key)
        if material_masks:
            context["material_masks"] = material_masks

    return context
```

**Not Required for Phase 1:** V2 can function with depth-only; material integration is progressive enhancement

---

## 6. Performance Requirements

### Target Performance

| Metric | Target | Rationale |
|--------|--------|-----------|
| Throughput | 400-600 images/hour | Batch processing requirement |
| Per-image time | <2s typical, <5s max | Interactive + batch friendly |
| Memory | <2GB peak | Headroom for batch workers |
| Dependencies | Core tier only | Fast installation |

### Performance Budget

**Operation Breakdown (1024x768 image):**
- Image I/O: ~50ms (load + save)
- Depth map load: ~30ms (TIFF decode)
- Tone mapping: ~100ms (depth-aware)
- Clarity enhancement: ~200ms (multi-scale unsharp mask)
- Material processing: ~300ms (per-material operations)
- Atmospheric effects: ~150ms (optional)
- **Total: ~830ms** (well under 2s budget)

**Optimization Strategy:**
1. **Lazy loading:** Only load depth map if depth_aware_tone_mapping enabled
2. **Conditional operations:** Skip atmospheric effects if preset disabled
3. **Numpy operations:** Use vectorized operations (no Python loops)
4. **Minimal allocations:** Reuse buffers where safe

---

## 7. Backward Compatibility

### CLI Compatibility

**Current Interface (ADR-022):**
```bash
python scripts/enhance_image.py INPUT_PATH \
    --depth-dir DEPTH_DIR \
    --output-dir OUTPUT_DIR \
    --preset default \
    --device cpu \
    --upscaler default \
    --log-file LOG_FILE
```

**Must Preserve:**
- All existing flags
- Default behavior (passthrough → actual enhancement)
- Report JSON format
- Exit codes

**Migration Strategy:**
```python
# Phase 1: Passthrough with warning (CURRENT STATE)
logger.warning("Running in passthrough mode")

# Phase 2: Actual enhancement with opt-in (THIS ADR)
logger.info(f"Applying V2 enhancement with preset '{preset}'")

# Phase 3: Module entrypoint (Future ADR)
# Replace subprocess with direct module call
```

### Orchestrator Compatibility

**Existing Test Contracts:**
- `test_orchestrator_v2_validation.py` - Fail-fast if script missing
- `test_v2_runner.py` - Subprocess invocation, report discovery
- CLI tests - `--enable-v2 on/off`, `--v2-preset`

**Must Pass:**
- All existing V2Runner tests
- Orchestrator fail-fast validation
- CLI flag parsing

---

## 8. Testing Requirements

### Unit Tests

**Coverage Requirements:**
```python
# tests/unit/lux_depth_v3/test_v2_enhance.py

def test_enhance_image_default_preset():
    """Verify default preset produces valid output."""

def test_enhance_image_with_depth_map():
    """Verify depth-aware tone mapping works."""

def test_enhance_image_without_depth_map():
    """Verify graceful degradation without depth."""

def test_preset_loading():
    """Verify all presets load correctly."""

def test_performance_budget():
    """Verify enhancement completes within 2s budget."""
```

### Integration Tests

```python
# tests/integration/test_v2_orchestrator_integration.py

def test_orchestrator_v2_with_depth_maps():
    """End-to-end: V3 depth → V2 enhancement."""

def test_orchestrator_v2_preset_selection():
    """Verify preset propagation through orchestrator."""
```

### Performance Tests

```python
# tests/performance/test_v2_throughput.py

def test_batch_throughput():
    """Verify 400+ images/hour throughput."""
    assert images_per_hour >= 400
```

---

## 9. Documentation Requirements

### User-Facing Documentation

**README.md Section:**
```markdown
### V2 Enhancement

V2 enhancement applies depth-aware finishing to your images:
- Depth-guided tone mapping
- Clarity and detail enhancement
- Material-specific processing
- Atmospheric effects

**Usage:**
```bash
lux-depth-v3 enhance INPUT \
    --enable-v2 on \
    --v2-preset luxury_estate \
    --output-dir OUTPUT
```

**Presets:**
- `default`: Balanced enhancement
- `luxury_estate`: Premium marketing aesthetic
- `architectural`: Technical visualization
- `none`: Skip V2 enhancement
```

**Changelog Entry:**
```markdown
### [0.2.0] - YYYY-MM-DD

#### Added
- **V2 Enhancement Implementation**: Depth-aware image finishing
  - Tone mapping, clarity, material-aware processing
  - Presets: default, luxury_estate, architectural
  - No ML dependencies (image processing only)
  - Performance: <2s/image typical
```

### Developer Documentation

**`docs/architecture/v2_enhancement_design.md`:**
- Architecture overview
- Integration patterns
- Preset system
- Performance considerations
- Extension points

---

## 10. Migration and Rollout Plan

### Phase 1: Foundation (Current)
- ✅ ADR-022: Passthrough script
- ✅ CLI flags: `--enable-v2`, `--v2-preset`
- ✅ V2Runner subprocess wrapper
- ✅ Fail-fast validation

### Phase 2: Implementation (This ADR)
1. Create `src/transformation_portal/lux_depth_v3/v2_enhance.py`
2. Create `src/transformation_portal/lux_depth_v3/v2_presets.py`
3. Update `scripts/enhance_image.py` to call `v2_enhance.enhance_image()`
4. Add unit tests
5. Add integration tests
6. Update documentation

**Acceptance Criteria:**
- All existing tests pass
- New tests achieve >90% coverage
- Performance: <2s/image for 1024x768
- Documentation complete

### Phase 3: Future (Separate ADR)
- Replace subprocess with direct module call in V2Runner
- Remove `scripts/enhance_image.py` dependency
- Optional: Add ML upscaling (gated behind `[ml]` extra)
- Optional: Material mask integration

---

## 11. Open Questions and Future Work

### Deferred Decisions

**ML Upscaling Integration:**
- **Question:** Should we support optional Real-ESRGAN/ESRGAN for users with `[ml]` installed?
- **Decision:** Defer to separate ADR. Current guidance: No ML in V2.
- **Rationale:** Avoids dependency complexity, licensing review, performance tuning

**Material Mask Integration:**
- **Question:** Should V2 consume Materials V3 segmentation masks?
- **Decision:** Optional, not required for Phase 2
- **Implementation:** Progressive enhancement, add when Materials V3 stabilizes

**Color Grading Presets:**
- **Question:** Should presets include color grading parameters?
- **Decision:** Basic support in presets, full LUT support deferred
- **Rationale:** Keep presets simple, avoid overloading configuration

### Research Questions

**Tone Mapping Algorithms:**
- Current: Simple zone-based (foreground boost, background compress)
- Investigate: Depth-guided bilateral tone mapping
- Investigate: Edge-aware tone curves

**Atmospheric Rendering:**
- Current: Heuristic masks (luminance + saturation)
- Investigate: Depth-based atmospheric scattering simulation

---

## 12. Required Enforcement

### CI Gates

**Pre-commit:**
- ✅ `v2_enhance.py` must pass flake8, pylint, mypy
- ✅ No ML imports in v2_enhance.py (verify with import scanner)

**CI Workflows:**
```yaml
# .github/workflows/test-v2-enhancement.yml
- name: Test V2 Enhancement
  run: |
    pytest tests/unit/lux_depth_v3/test_v2_enhance.py
    pytest tests/integration/test_v2_orchestrator_integration.py

- name: Performance Benchmark
  run: |
    pytest tests/performance/test_v2_throughput.py
    # Assert: throughput >= 400 images/hour
```

**Dependency Check:**
```bash
# Verify no banned dependencies in v2_enhance.py
python scripts/security/verify_banned_dependencies.py \
    --check-imports src/transformation_portal/lux_depth_v3/v2_enhance.py
```

### Documentation Enforcement

- ✅ README.md updated with V2 section
- ✅ CHANGELOG.md entry for V2 implementation
- ✅ Docstrings for all public functions
- ✅ Type hints for all parameters

---

## 13. Approval and Next Steps

### Architect Approval

**Status:** ✅ **APPROVED**

**Rationale:**
1. Clear separation: Image processing only (no ML complexity)
2. Reuses existing components (EnhancementStage, material utilities)
3. Commercially safe (BSD/MIT dependencies only)
4. Performance compliant (<2s/image, 400+ images/hour)
5. Backward compatible (preserves existing CLI, tests)
6. Well-defined scope (depth-aware finishing, not generation)

### Implementation Priority

**Priority:** High
**Complexity:** Medium
**Risk:** Low

**Estimated Effort:**
- `v2_enhance.py` implementation: 4-6 hours
- `v2_presets.py` + presets: 2-3 hours
- `scripts/enhance_image.py` update: 1-2 hours
- Unit tests: 3-4 hours
- Integration tests: 2-3 hours
- Documentation: 2-3 hours
- **Total: 14-21 hours** (2-3 days)

### Next Actions

**For Implementation Team:**
1. Create `src/transformation_portal/lux_depth_v3/v2_enhance.py` following structure above
2. Create `src/transformation_portal/lux_depth_v3/v2_presets.py` with default presets
3. Update `scripts/enhance_image.py` to call `v2_enhance.enhance_image()`
4. Add comprehensive tests (unit + integration)
5. Update README.md and CHANGELOG.md

**For Architect Review:**
1. Code review after implementation (verify no ML creep)
2. Performance validation (benchmark throughput)
3. Dependency audit (confirm no new dependencies added)

---

## Appendix A: Example Usage

### CLI Usage

```bash
# Basic usage (default preset)
lux-depth-v3 enhance input.jpg \
    --output-dir ./output

# Luxury estate preset
lux-depth-v3 enhance input.jpg \
    --enable-v2 on \
    --v2-preset luxury_estate \
    --output-dir ./output

# Skip V2 (PBR only)
lux-depth-v3 enhance input.jpg \
    --enable-v2 off \
    --output-dir ./output
```

### Python API Usage

```python
from pathlib import Path
from transformation_portal.lux_depth_v3.v2_enhance import (
    enhance_image,
    V2EnhancementConfig,
)

# Enhance with custom config
config = V2EnhancementConfig(
    preset="luxury_estate",
    enhancement_strength=0.8,
    clarity_strength=0.6,
)

report = enhance_image(
    input_path=Path("input.jpg"),
    output_path=Path("output/enhanced.jpg"),
    depth_map_path=Path("depth/input_depth.tiff"),
    config=config,
)

print(f"Enhanced image saved to {report['output']}")
```

---

## Appendix B: Comparison Matrix

| Feature | V2 Enhancement (This ADR) | Diffusion Rendering (lux_render_pipeline) | EnhancementStage (stage_graph) |
|---------|---------------------------|-------------------------------------------|-------------------------------|
| **Purpose** | Depth-aware finishing | Content generation | Perceptual improvement |
| **ML Models** | None | SDXL, ControlNet | None |
| **Dependencies** | Core only | `[ml]` extra | Core only |
| **Speed** | <2s/image | 20-60s/image | <1s/image |
| **Depth Awareness** | Yes (V3 maps) | Optional (ControlNet) | Yes (tone mapping) |
| **Materials** | Yes (V3 taxonomy) | No | Yes (basic) |
| **Upscaling** | Lanczos (optional) | SD-Upscale, Real-ESRGAN | Bicubic |
| **License** | BSD/MIT | Mixed (see ADR-0015) | BSD/MIT |
| **Use Case** | Marketing finishing | Render refinement | General enhancement |

---

**End of Architectural Guidance**
