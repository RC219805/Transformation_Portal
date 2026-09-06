# APEX Workflow Design: Fully Integrated Architecture

**Version:** 3.0.0
**Status:** Production-Ready Architecture
**Date:** 2026-02-07
**Scope:** Complete APEX pipeline with Depth Pro, DA3, PBR, Materials V3, and all advanced features

---

## Executive Summary

The **APEX (Architectural Photo Enhancement eXecution)** workflow is a production-grade, multi-backend depth-aware rendering pipeline for luxury real estate and architectural visualization. It combines:

- **Multi-backend depth intelligence** (Depth Pro metric depth + DA3 relative depth)
- **PBR materials generation** (normal maps, roughness, ambient occlusion)
- **Materials V3 semantic understanding** (room-aware tone mapping strategies)
- **Deterministic quality firewall** (performance regression detection)
- **License governance** (multi-layer enforcement for research models)
- **Production-grade orchestration** (stage graph, atomic IO, comprehensive provenance)

**Key Innovation:** Dual-depth fusion architecture combining metric depth (Depth Pro) for accurate 3D reconstruction and relative depth (DA3) for artistic depth-aware tone mapping.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Depth Backend Strategy](#depth-backend-strategy)
3. [Pipeline Stages](#pipeline-stages)
4. [Configuration Schema](#configuration-schema)
5. [Quality Tiers](#quality-tiers)
6. [Feature Matrix](#feature-matrix)
7. [Orchestration Flow](#orchestration-flow)
8. [License Governance](#license-governance)
9. [Performance Envelope](#performance-envelope)
10. [Operational Runbook](#operational-runbook)

---

## Architecture Overview

### System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                      APEX Workflow Engine                        │
│  "Context-aware rendering for luxury real estate ArchViz"       │
└─────────────────────────────────────────────────────────────────┘
           │
           ├─── Input Layer (Discovery + Hygiene)
           │    ├─ Multi-format support (JPEG, PNG, TIFF, RAW)
           │    ├─ Artifact exclusion (depth maps, temp files)
           │    └─ Dimension validation (max resolution caps)
           │
           ├─── Intelligence Layer (Depth + Materials)
           │    ├─ Depth Pro Backend (metric depth, focal length)
           │    ├─ DA3 Backend (relative depth, artistic)
           │    ├─ Materials V3 (semantic room classification)
           │    └─ Dual-depth fusion (metric + relative)
           │
           ├─── Enhancement Layer (PBR + Tone Mapping)
           │    ├─ PBR Generation (normal, roughness, AO)
           │    ├─ Room-aware tone mapping strategies
           │    ├─ Depth-guided perceptual enhancement
           │    └─ Color science (ACEScc, FilmicPro)
           │
           ├─── Quality Layer (Firewall + Provenance)
           │    ├─ Performance ledger (p95 regression detection)
           │    ├─ Visual quality metrics (PSNR, SSIM, VIF)
           │    ├─ Comprehensive metadata
           │    └─ Contract test enforcement
           │
           └─── Output Layer (Export + Archival)
                ├─ Multi-format export (TIFF 16-bit, JPEG, PNG)
                ├─ Atomic writes (no partial failures)
                ├─ Depth cache (skip re-inference)
                └─ Manifest + provenance ledger
```

### Design Principles

1. **Backend Agnostic:** Depth backends selected via config, not hardcoded
2. **Composable:** Stages can be enabled/disabled independently
3. **Deterministic:** Same input + config = same output (modulo RNG seeds)
4. **Observable:** Comprehensive logging, metrics, and provenance
5. **Safe:** Multi-layer license enforcement, atomic IO, graceful degradation
6. **Fast:** Parallel processing, depth caching, optimized vectorized ops

---

## Depth Backend Strategy

### Unified Backend Protocol

All depth backends implement `DepthBackend` protocol:

```python
from transformation_portal.depth.backends import DepthBackend, DepthResult

class DepthBackend(Protocol):
    name: str                    # Unique backend ID
    license_type: LicenseType    # COMMERCIAL | RESEARCH_ONLY
    requires_checkpoint: bool    # True for Depth Pro

    def compute(
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> DepthResult:
        """Estimate depth from image."""
        ...

    def get_cache_key(image) -> str:
        """Generate deterministic cache key."""
        ...

    def ensure_available() -> None:
        """Validate dependencies/checkpoints."""
        ...
```

### Backend Comparison Matrix

| Backend      | Depth Type | License       | Checkpoint | Focal Length | Device Support    | Quality Tier | Use Case                  |
|--------------|------------|---------------|------------|--------------|-------------------|--------------|---------------------------|
| **DA3**      | Relative   | MIT           | Auto-DL    | ❌            | MPS/CUDA/CPU      | Default      | Artistic depth effects    |
| **Depth Pro**| Metric     | Apple AMLR    | 1.9 GB     | ✅            | MPS/CUDA/CPU      | Research     | 3D reconstruction, VR/AR  |

### Dual-Depth Fusion Architecture

APEX workflows can use **both backends simultaneously** for complementary strengths:

```yaml
apex_dual_depth_preset:
  depth:
    primary_backend: depth_pro    # For 3D accuracy
    fallback_backend: da3          # For artistic enhancement
    fusion_mode: weighted          # Combine both for final depth
    fusion_weights:
      metric: 0.6                  # Depth Pro contribution
      relative: 0.4                # DA3 contribution

  materials_v3:
    use_primary_depth: true        # Use Depth Pro for PBR generation

  tone_mapping:
    use_fallback_depth: true       # Use DA3 for depth-aware tone curves
```

**Rationale:**
- **Depth Pro:** Provides accurate metric depth (meters) and focal length for PBR normal map generation, 3D export, and VR/AR workflows
- **DA3:** Provides artistically optimized relative depth for depth-aware tone mapping, atmospheric effects, and perceptual enhancement
- **Fusion:** Weighted blend preserves metric accuracy while incorporating artistic refinement

Explicit `depth_backend="depth_pro"` runs serialize the resolved model identity as `apple/ml-depth-pro` across effective config, per-image manifest config fingerprints, depth cache fingerprints, and run-card fingerprints. DA3 runs keep the existing DA3 `model_variant` names.

---

## Pipeline Stages

### Stage Graph Architecture

APEX uses a directed acyclic graph (DAG) of stages coordinated by `StageOrchestrator`:

```
Input Discovery
     ↓
Input Validation (dimension caps, format checks)
     ↓
Depth Inference (DA3 or Depth Pro)
     ├─→ [Optional] Dual Backend (parallel execution)
     ↓
Materials V3 Semantic Analysis (room classification)
     ↓
PBR Generation (normal, roughness, AO)
     ↓
Depth-Aware Tone Mapping (room-specific strategies)
     ↓
Perceptual Enhancement (color science, sharpening)
     ↓
Quality Validation (PSNR, SSIM, VIF)
     ↓
Export + Provenance (atomic writes, metadata)
```

### Stage Details

#### 1. Input Discovery Stage

**Responsibility:** Discover valid images, exclude artifacts

```python
from transformation_portal.lux_depth_v3.input_discovery import (
    discover_input_images
)

inputs = discover_input_images(
    input_path="input_images/",
    exclude_patterns=["*_depth*", "*_normal*", "*_rough*", "*_ao*"],
    max_resolution=8192,  # Prevent OOM on massive files
)
```

**Outputs:**
- `discovered_images`: List of valid input paths
- `excluded_artifacts`: List of excluded files (for audit)
- `hygiene_warnings`: List of questionable patterns

---

#### 2. Depth Inference Stage (Multi-Backend)

**Responsibility:** Estimate depth using selected backend(s)

**Single Backend (Default):**

```python
from transformation_portal.depth.backends import DepthBackendRegistry
from transformation_portal.lux_depth_v3 import EnhanceConfig

config = EnhanceConfig(
    depth_backend="da3",          # or "depth_pro"
    depth_device="mps",           # MPS for Apple Silicon
    depth_cache_enabled=True,     # Skip re-inference
)

registry = DepthBackendRegistry()
backend = registry.get_backend(config.depth_backend, config)
result = backend.compute(image, device=config.depth_device)

# DepthResult attributes:
# - depth_map: np.ndarray (H, W) float32
# - depth_units: "relative" | "meters"
# - focal_length_px: Optional[float] (Depth Pro only)
# - backend_id: str
# - metadata: Dict[str, Any]
```

**Dual Backend (Advanced):**

```python
# Primary: Depth Pro for metric depth
primary_config = EnhanceConfig(
    depth_backend="depth_pro",
    non_commercial_ok=True,
    accept_apple_depth_pro_research_license=True,
    depth_device="mps",
)
primary_backend = registry.get_backend("depth_pro", primary_config)
metric_result = primary_backend.compute(image)

# Fallback: DA3 for artistic depth
fallback_config = EnhanceConfig(depth_backend="da3", depth_device="mps")
fallback_backend = registry.get_backend("da3", fallback_config)
relative_result = fallback_backend.compute(image)

# Fusion
from transformation_portal.depth.fusion import fuse_depth_maps
fused_depth = fuse_depth_maps(
    metric_result.depth_map,
    relative_result.depth_map,
    weights=(0.6, 0.4),  # 60% metric, 40% relative
)
```

**Outputs:**
- `depth_map`: Depth values (H, W)
- `depth_provenance`: Backend metadata, timings, license info
- `depth_cache_entry`: Cache key for future runs

---

#### 3. Materials V3 Semantic Analysis

**Responsibility:** Classify room type and infer material properties

```python
from transformation_portal.lux_depth_v3.materials_v3 import (
    MaterialsV3Analyzer
)

analyzer = MaterialsV3Analyzer()
materials_result = analyzer.analyze(
    image=image,
    depth_map=depth_map,
    focal_length_px=metric_result.focal_length_px,  # If available
)

# Materials V3 outputs:
# - room_type: "living_room" | "kitchen" | "bedroom" | ...
# - materials: List[Material] (wood, glass, metal, fabric)
# - lighting_context: "natural" | "artificial" | "mixed"
# - semantic_zones: np.ndarray (H, W) segmentation mask
```

**Room-Specific Strategies:**

| Room Type        | Tone Mapping Strategy         | Depth Emphasis | PBR Roughness Bias |
|------------------|-------------------------------|----------------|--------------------|
| Sunroom          | Preserve highlights, warmth   | Soft           | Low (glossy)       |
| Home Cinema      | Deepen blacks, boost contrast | Strong         | Medium             |
| Kitchen          | Clean whites, cool tones      | Moderate       | Low (reflective)   |
| Bedroom          | Soft shadows, warmth          | Soft           | High (fabric)      |
| Exterior Pool    | HDR sky, water reflections    | Strong         | Low (water)        |

---

#### 4. PBR Generation Stage

**Responsibility:** Generate physically-based rendering maps from depth

```python
from transformation_portal.lux_depth_v3.pbr_processor import (
    PBRProcessor
)

pbr_processor = PBRProcessor(
    depth_map=depth_map,
    materials_context=materials_result,  # Room-aware PBR
    use_metric_depth=True,               # Use Depth Pro if available
)

pbr_maps = pbr_processor.generate_all()

# PBR outputs:
# - normal_map: np.ndarray (H, W, 3) RGB normal vectors
# - roughness_map: np.ndarray (H, W) surface roughness [0-1]
# - ao_map: np.ndarray (H, W) ambient occlusion [0-1]
# - metallic_map: Optional[np.ndarray] (advanced preset only)
```

**Normal Map Generation (Depth-Aware):**

```python
# If metric depth available (Depth Pro):
normal_map = compute_normals_from_metric_depth(
    depth_meters=metric_result.depth_map,
    focal_length_px=metric_result.focal_length_px,
    method="central_difference",  # Higher accuracy than Sobel
)

# If relative depth only (DA3):
normal_map = compute_normals_from_relative_depth(
    depth_normalized=relative_result.depth_map,
    method="sobel",  # Artistic approximation
)
```

---

#### 5. Depth-Aware Tone Mapping Stage

**Responsibility:** Apply room-specific tone curves guided by depth

```python
from transformation_portal.rendering.tone_mapping import (
    DepthAwareToneMapper,
    RoomStrategy,
)

strategy = RoomStrategy.from_room_type(materials_result.room_type)

tone_mapper = DepthAwareToneMapper(
    depth_map=relative_result.depth_map,  # Use DA3 for artistic depth
    strategy=strategy,
    color_space="ACEScc",  # Industry-standard color science
)

enhanced_rgb = tone_mapper.apply(image)

# Depth-aware effects:
# - Foreground subjects: preserve detail, avoid over-saturation
# - Mid-ground: apply room-specific tone curve
# - Background: atmospheric depth, subtle desaturation
```

**Strategy Examples:**

```python
# Sunroom Strategy: Preserve highlights, add warmth
class SunroomStrategy(ToneMappingStrategy):
    def apply(self, rgb, depth):
        # Depth-aware highlight preservation
        foreground_mask = depth < 0.3
        rgb[foreground_mask] = preserve_highlights(
            rgb[foreground_mask],
            max_clip=0.95,  # Prevent clipping
        )

        # Warmth in mid-ground
        midground_mask = (depth >= 0.3) & (depth < 0.7)
        rgb[midground_mask] = add_warmth(
            rgb[midground_mask],
            temperature_shift=+200K,
        )

        return rgb

# Cinema Strategy: Deep blacks, contrast boost
class CinemaStrategy(ToneMappingStrategy):
    def apply(self, rgb, depth):
        # Aggressive black point for background
        background_mask = depth > 0.7
        rgb[background_mask] = deepen_blacks(
            rgb[background_mask],
            black_lift=-0.05,
        )

        # Contrast boost for foreground subjects
        foreground_mask = depth < 0.3
        rgb[foreground_mask] = boost_contrast(
            rgb[foreground_mask],
            gamma=0.85,
        )

        return rgb
```

---

#### 6. Perceptual Enhancement Stage

**Responsibility:** Apply color science and perceptual sharpening

```python
from transformation_portal.perceptual.enhancement import (
    PerceptualEnhancer
)

enhancer = PerceptualEnhancer(
    color_space="ACEScc",
    apply_sharpening=True,
    sharpening_strength=0.3,  # Subtle for luxury real estate
    preserve_skin_tones=True,
)

final_rgb = enhancer.apply(enhanced_rgb, depth_map=relative_result.depth_map)
```

---

#### 7. Quality Validation Stage

**Responsibility:** Ensure outputs meet quality thresholds

```python
from transformation_portal.metrics.quality import validate_output_quality

quality_report = validate_output_quality(
    original=image,
    enhanced=final_rgb,
    depth_map=depth_map,
    thresholds={
        "psnr_min": 30.0,      # Prevent over-processing
        "ssim_min": 0.85,       # Structural similarity
        "vif_min": 0.70,        # Visual information fidelity
    },
)

if not quality_report.passed:
    logger.warning(f"Quality check failed: {quality_report.failures}")
    # Fallback to original or less aggressive preset
```

---

#### 8. Export + Provenance Stage

**Responsibility:** Atomic writes with comprehensive metadata

```python
from transformation_portal.lux_depth_v3.io_atomic import (
    atomic_write_image,
    atomic_write_metadata,
)

# Export enhanced image (16-bit TIFF for archival)
atomic_write_image(
    image=final_rgb,
    path=output_dir / f"{stem}_enhanced.tif",
    format="TIFF",
    bit_depth=16,
    compression="lzw",
)

# Export PBR maps
atomic_write_image(pbr_maps.normal_map, output_dir / f"{stem}_normal.png")
atomic_write_image(pbr_maps.roughness_map, output_dir / f"{stem}_roughness.png")
atomic_write_image(pbr_maps.ao_map, output_dir / f"{stem}_ao.png")

# Export depth map
atomic_write_image(
    depth_map,
    output_dir / f"{stem}_depth.png",
    format="PNG",
    bit_depth=16,  # 65535 levels of precision
)

# Export comprehensive metadata
provenance = {
    "input_path": str(input_path),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "pipeline_version": "3.0.0-apex",
    "depth_backend": {
        "primary": metric_result.backend_id,
        "fallback": relative_result.backend_id,
        "fusion_weights": (0.6, 0.4),
    },
    "materials_v3": {
        "room_type": materials_result.room_type,
        "lighting_context": materials_result.lighting_context,
    },
    "pbr": {
        "normal_method": "central_difference",
        "roughness_bias": 0.3,
    },
    "quality_metrics": quality_report.metrics,
    "license_compliance": {
        "depth_pro_accepted": config.accept_apple_depth_pro_research_license,
        "non_commercial_ok": config.non_commercial_ok,
    },
}

atomic_write_metadata(provenance, output_dir / f"{stem}_provenance.json")
```

---

## Configuration Schema

### APEX Preset (Full-Featured)

```yaml
# config/presets/apex_dual_depth.yaml

preset_name: apex_dual_depth
quality_tier: apex
version: 3.0.0

# Depth Intelligence (Dual Backend)
depth:
  primary_backend: depth_pro
  fallback_backend: da3
  fusion_mode: weighted
  fusion_weights:
    metric: 0.6
    relative: 0.4

  # Depth Pro Configuration
  depth_pro:
    checkpoint_path: checkpoints/depth_pro.pt
    device: mps
    non_commercial_ok: true
    accept_apple_depth_pro_research_license: true

  # DA3 Configuration
  da3:
    model_variant: METRIC_LARGE
    device: mps
    use_fp16: true  # 2x faster on MPS/CUDA

  # Caching
  cache_enabled: true
  cache_dir: .depth_cache

# Materials V3 (Semantic Understanding)
materials_v3:
  enabled: true
  use_primary_depth: true  # Use Depth Pro for PBR
  room_classification:
    confidence_threshold: 0.7
    fallback_strategy: generic

# PBR Generation
pbr:
  enabled: true
  normal_map:
    method: central_difference  # Requires metric depth
    strength: 1.0
  roughness_map:
    use_materials_bias: true  # Room-aware roughness
    base_roughness: 0.5
  ao_map:
    radius: 0.5
    intensity: 0.8
  metallic_map:
    enabled: false  # Experimental

# Tone Mapping (Room-Aware)
tone_mapping:
  enabled: true
  use_fallback_depth: true  # Use DA3 for artistic depth
  room_strategies:
    sunroom:
      preserve_highlights: true
      warmth_shift: +200K
    cinema:
      deepen_blacks: true
      contrast_gamma: 0.85
    kitchen:
      clean_whites: true
      coolness_shift: -100K
  color_space: ACEScc

# Perceptual Enhancement
enhancement:
  enabled: true
  sharpening:
    strength: 0.3
    radius: 1.0
  color_science:
    apply_filmic_curve: true
    preserve_skin_tones: true

# Quality Firewall
quality:
  validation_enabled: true
  thresholds:
    psnr_min: 30.0
    ssim_min: 0.85
    vif_min: 0.70
  regression_detection:
    enabled: true
    ledger_path: tools/performance_ledger.json
    block_on_regression: true

# Output Configuration
output:
  formats:
    - tiff_16bit  # Archival
    - jpeg_95     # Web preview
  include_depth_maps: true
  include_pbr_maps: true
  include_provenance: true
  atomic_writes: true

# Orchestration
orchestration:
  parallel_workers: 15
  max_batch_size: 50
  timeout_per_image: 120  # seconds
  graceful_degradation: true
```

---

## Quality Tiers

### Tier Taxonomy

| Tier           | Depth Backend(s)  | PBR     | Materials V3 | Tone Mapping | Target Use Case                  |
|----------------|-------------------|---------|--------------|--------------|----------------------------------|
| **basic**      | DA3 (small)       | ❌       | ❌            | Generic      | Thumbnails, previews             |
| **standard**   | DA3 (base)        | ✅       | ❌            | Generic      | General real estate photography  |
| **premium**    | DA3 (large)       | ✅       | ✅            | Room-aware   | Luxury real estate marketing     |
| **apex**       | Dual (DA3 + DP)   | ✅       | ✅            | Room-aware   | ArchViz, VR/AR, portfolio work   |
| **research**   | Depth Pro only    | ✅       | ✅            | Experimental | 3D reconstruction, research      |

### Tier Migration Path

```
basic → standard → premium → apex
                           ↘ research (for 3D workflows)
```

### APEX Gate Policy

The `apex` tier enforces fail-closed quality gates. Two policy switches govern recovery behavior; both are observable in run-card output and the gate fingerprint:

- **Depth fallback auto-upgrade.** `EnhanceConfig.__post_init__` flips the default `depth_fallback="fail"` to `"v2-auto"` when `quality_tier == "apex"`. Flat-distribution scenes that fail both DA3 (`APEX_DEPTH_PLATEAU` — `upper_iqr ≤ 1e-4`) and DA2 (`APEX_DEPTH_SATURATION_LOW` — `> 2 %` low-saturation pixels) recover via the V2 stage with independent depth instead of failing the batch. The run card records the full attempt history (`DA3 → DA2 → v2-auto`).
  - **Operator escape hatch:** pass `depth_fallback="apex-strict"` to keep fail-closed depth on APEX. The validator (`security.validate_depth_fallback`) accepts the value, `__post_init__` canonicalizes it to `"fail"`, and the auto-upgrade is suppressed for that run.
  - The fingerprint payload (`build_apex_depth_gate_fingerprint_payload`) includes `depth_fallback` so cache replays under one policy do not serve outputs from the other.
- **Materials V3 soft-passthrough on confidence-only blocks.** `_enforce_apex_materials_pixel_ops_gate` narrows the original fail-closed condition: when masks exist and every implemented pixel op is blocked solely by `below_confidence_threshold`, the gate emits the output without applying pixel ops and surfaces a non-fatal `APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE` warning instead of `APEX_MATERIALS_PIXEL_OPS_EMPTY`.
  - Mixed blocker sets (`missing_material_confidence`, `unsupported_confidence_score_type`, `below_coverage_threshold`, `no_implementation`, …) still fail closed.
  - The warning is mirrored in two places: `materials_v3.pixel_ops.passthrough_status` (canonical, consumed by the orchestrator's per-image manifest) and `materials_v3.segmentation_metadata.pixel_ops_passthrough` (consumed by the run-card cache).
  - The materials fingerprint payload (`build_materials_fingerprint_payload`) carries `pixel_ops_strict_policy_version` to mark this regime; bumping it invalidates caches when blocker semantics change.

Run-card material telemetry keeps the counting bases separate:

- `result_summary[].segmentation_status.materials_summary.masks_generated` / `mask_count` describe Materials V3/SAM2 mask candidates.
- `pixel_ops_applied` / `pixel_ops_applied_count` describe Materials V3 material pixel operations actually executed.
- `blocked_count` describes confidence/fail-closed candidate operations rejected by the Materials V3 pixel-op policy.
- V2 reports separately expose `enhancement_metadata.material_masks_supplied`, `material_masks_supplied_count`, and `v2_material_adjustments_applied`.
- Deprecated V2 `enhancement_metadata.materials_applied` remains a boolean compatibility alias for `v2_material_adjustments_applied`; it no longer means mask keys supplied.

The advisory warning `APEX_MATERIALS_SEGMENTATION_DOMINATES_NO_PIXEL_OPS` appears only when SAM2 runtime is available, total image runtime is positive, SAM2 runtime share is at least 90%, masks exist, and Materials V3 applied zero pixel ops. It is runtime-cost telemetry, not a failure code.

For V2 TIFF output, ICC profiles are preserved when available, with PIL `icc_profile` preferred and TIFF tag `34675` used as fallback. EXIF remains intentionally stripped for deterministic TIFF output; reports mark the result as partial metadata preservation when ICC is carried forward and EXIF is not written.

### APEX Promotion Eligibility & Soft-Passthrough

Promotion via `build_apex_evidence_bundle` reads a per-candidate evidence JSON file. To carry the soft-passthrough decision through to promotion (so the four images we previously rescued at runtime no longer block promotion), derive the evidence directly from each per-image manifest:

```python
from pathlib import Path
import json
from transformation_portal.evals.apex_evidence_bundle import (
    derive_materials_v3_evidence_from_manifest,
)

# `result` is one successful item returned by EnhanceOrchestrator.enhance_batch.
# Its manifest field is the exact evidence-bound combined-manifest path.
manifest_path = Path(result["manifest"])
evidence = derive_materials_v3_evidence_from_manifest(manifest_path)
Path("output/apex_eval/materials_v3_evidence.json").write_text(json.dumps(evidence))
# Then: tools/run_apex_eval.py --candidate-evidence materials_v3:<asset_id>=<evidence_path>
```

The orchestrator's `MaterialsV3Metadata` is the single source of truth; the helper renders it into the shape `_materials_status` consumes. When `passthrough_status.code == "APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE"`, the bundle keeps `failure_code = None` and promotion proceeds.

---

## Feature Matrix

| Feature                          | basic | standard | premium | apex | research |
|----------------------------------|-------|----------|---------|------|----------|
| Depth Estimation                 | ✅     | ✅        | ✅       | ✅    | ✅        |
| Metric Depth (Depth Pro)         | ❌     | ❌        | ❌       | ✅    | ✅        |
| Relative Depth (DA3)             | ✅     | ✅        | ✅       | ✅    | ❌        |
| Dual-Depth Fusion                | ❌     | ❌        | ❌       | ✅    | ❌        |
| PBR Normal Maps                  | ❌     | ✅        | ✅       | ✅    | ✅        |
| PBR Roughness Maps               | ❌     | ✅        | ✅       | ✅    | ✅        |
| PBR AO Maps                      | ❌     | ✅        | ✅       | ✅    | ✅        |
| Materials V3 (Room Classification) | ❌   | ❌        | ✅       | ✅    | ✅        |
| Room-Aware Tone Mapping          | ❌     | ❌        | ✅       | ✅    | ❌        |
| Depth-Aware Enhancement          | ❌     | ✅        | ✅       | ✅    | ❌        |
| Color Science (ACEScc)           | ❌     | ✅        | ✅       | ✅    | ✅        |
| Depth Caching                    | ✅     | ✅        | ✅       | ✅    | ✅        |
| Quality Firewall                 | ❌     | ❌        | ✅       | ✅    | ✅        |
| Focal Length Estimation          | ❌     | ❌        | ❌       | ✅    | ✅        |
| 3D Export (OBJ, PLY)             | ❌     | ❌        | ❌       | ❌    | ✅        |

---

## Orchestration Flow

### CLI Entry Point

```bash
# APEX workflow with dual-depth fusion
lux-depth-v3 \
  --input input_images/750_picacho/ \
  --output output/apex_run_$(date +%Y%m%d_%H%M%S) \
  --preset apex_dual_depth \
  --depth-backend depth_pro \
  --fallback-depth-backend da3 \
  --depth-device mps \
  --non-commercial-ok \
  --accept-apple-depth-pro-research-license \
  --materials-v3 \
  --pbr \
  --parallel 15 \
  --cache-depth \
  --quality-firewall
```

### Python API Entry Point

```python
from transformation_portal.lux_depth_v3 import (
    EnhanceConfig,
    ApexOrchestrator,
    Preset,
)

config = EnhanceConfig(
    # Depth configuration
    depth_backend="depth_pro",
    fallback_depth_backend="da3",
    depth_device="mps",
    non_commercial_ok=True,
    accept_apple_depth_pro_research_license=True,

    # Materials V3
    materials_v3_enabled=True,

    # PBR
    pbr_enabled=True,
    pbr_normal_method="central_difference",

    # Quality
    quality_tier="apex",
    preset=Preset.LUXURY_ESTATE,
    quality_firewall_enabled=True,

    # Orchestration
    parallel_workers=15,
    depth_cache_enabled=True,
)

orchestrator = ApexOrchestrator(config)
results = orchestrator.run(
    input_path="input_images/750_picacho/",
    output_path="output/apex_run/",
)

print(f"Processed {results.success_count}/{results.total_count} images")
print(f"Total time: {results.elapsed_time:.1f}s")
print(f"Avg time per image: {results.avg_time_per_image:.2f}s")
```

---

## License Governance

### Multi-Layer Enforcement

**Layer 1: Config Validation**

```python
# In EnhanceConfig.__post_init__()
if self.depth_backend == "depth_pro":
    if not self.non_commercial_ok:
        raise ValueError(
            "Depth Pro requires non_commercial_ok=True"
        )
    if not self.accept_apple_depth_pro_research_license:
        raise ValueError(
            "Depth Pro requires "
            "accept_apple_depth_pro_research_license=True"
        )
```

**Layer 2: Factory Validation**

```python
# In DepthBackendRegistry.get_backend()
def _validate_license(backend_cls, config):
    if backend_cls.license_type == LicenseType.RESEARCH_ONLY:
        if not config.non_commercial_ok:
            raise LicenseRestrictionError(...)
        if backend_cls.name == "depth_pro":
            if not config.accept_apple_depth_pro_research_license:
                raise LicenseRestrictionError(...)
```

**Layer 3: Runtime Validation**

```python
# In DepthProBackend.compute()
def compute(self, image, device=None):
    self._validate_license_runtime()  # Defense-in-depth
    # ... inference ...
```

### License Matrix

| Backend    | License       | Commercial Use | Requires Flags                                      |
|------------|---------------|----------------|-----------------------------------------------------|
| DA3        | MIT           | ✅ Allowed      | None                                                |
| Depth Pro  | Apple AMLR    | ❌ Forbidden    | `non_commercial_ok=True`<br>`accept_apple_depth_pro_research_license=True` |

---

## Performance Envelope

### Quality Firewall Thresholds

```json
{
  "apex_tier": {
    "p95_latency_max_ms": 15000,
    "mean_latency_max_ms": 12000,
    "regression_tolerance": 0.10,
    "failure_rate_max": 0.0
  }
}
```

### Expected Performance (Apple Silicon M1/M2)

| Stage                     | Device | Avg Time (ms) | p95 Time (ms) | Notes                              |
|---------------------------|--------|---------------|---------------|------------------------------------|
| Input Discovery           | CPU    | 50            | 100           | Negligible                         |
| Depth Pro Inference       | MPS    | 800           | 1200          | 1.9 GB checkpoint                  |
| DA3 Inference             | MPS    | 600           | 900           | FP16 acceleration                  |
| Dual-Depth Fusion         | CPU    | 20            | 40            | Vectorized numpy ops               |
| Materials V3 Analysis     | CPU    | 100           | 200           | Lightweight semantic classification|
| PBR Generation            | CPU    | 80            | 150           | Numba-accelerated if available     |
| Depth-Aware Tone Mapping  | CPU    | 50            | 100           | Vectorized color ops               |
| Quality Validation        | CPU    | 200           | 300           | PSNR/SSIM computation              |
| Export + Provenance       | CPU    | 100           | 200           | Atomic writes                      |
| **Total (APEX Tier)**     | Mixed  | **2000**      | **3100**      | ~2-3 seconds per image             |

### Performance Optimization Tips

1. **Use MPS on Apple Silicon:** 3-5x faster than CPU for depth inference
2. **Enable FP16:** 2x memory reduction, 1.3-1.5x speedup (minimal quality loss)
3. **Enable Depth Caching:** Skip re-inference for unchanged inputs
4. **Adjust Parallel Workers:** `parallel_workers = CPU_COUNT * 2` for I/O-bound stages
5. **Use Quality Tier Appropriately:** Don't use `apex` for thumbnails

---

## Operational Runbook

### Common Workflows

#### 1. Standard APEX Workflow (Dual-Depth)

```bash
# Research use only (Depth Pro)
lux-depth-v3 \
  --input input_images/ \
  --output output/apex_$(date +%Y%m%d) \
  --preset apex_dual_depth \
  --depth-device mps \
  --non-commercial-ok \
  --accept-apple-depth-pro-research-license \
  --parallel 15
```

#### 2. Production Workflow (Commercial, DA3 Only)

```bash
# Commercial use allowed (no Depth Pro)
lux-depth-v3 \
  --input input_images/ \
  --output output/premium_$(date +%Y%m%d) \
  --preset premium \
  --depth-backend da3 \
  --depth-device mps \
  --materials-v3 \
  --pbr \
  --parallel 15
```

#### 3. Research Workflow (3D Reconstruction)

```bash
# Metric depth only (no artistic enhancement)
lux-depth-v3 \
  --input input_images/ \
  --output output/research_$(date +%Y%m%d) \
  --preset research \
  --depth-backend depth_pro \
  --depth-device mps \
  --non-commercial-ok \
  --accept-apple-depth-pro-research-license \
  --export-3d \
  --format obj
```

### Troubleshooting

#### Issue: Depth Pro License Error

```
LicenseRestrictionError: Depth Pro requires
accept_apple_depth_pro_research_license=True
```

**Solution:** Add license acceptance flags:

```python
config = EnhanceConfig(
    depth_backend="depth_pro",
    non_commercial_ok=True,
    accept_apple_depth_pro_research_license=True,
)
```

#### Issue: Depth Pro Checkpoint Not Found

```
FileNotFoundError: Depth Pro checkpoint not found: checkpoints/depth_pro.pt
```

**Solution:** Download checkpoint (1.9 GB):

```bash
mkdir -p checkpoints
curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \
  -o checkpoints/depth_pro.pt
```

#### Issue: Quality Firewall Regression

```
QualityFirewallError: p95 latency increased by 15% (threshold: 10%)
```

**Solution:** Investigate performance regression:

```bash
# Check performance ledger
cat tools/performance_ledger.json | jq '.recent_runs[-5:]'

# Identify slow stages
grep "STAGE_TIMING" logs/apex_run.log

# Adjust config or accept regression
lux-depth-v3 ... --quality-firewall-override
```

---

## Testing Strategy

### Contract Tests (Required)

```python
def test_apex_preset_schema_invariants():
    """Validate APEX preset schema stability."""
    preset = load_preset("apex_dual_depth")

    assert preset["quality_tier"] == "apex"
    assert preset["depth"]["primary_backend"] == "depth_pro"
    assert preset["depth"]["fallback_backend"] == "da3"
    assert 0.0 <= preset["depth"]["fusion_weights"]["metric"] <= 1.0

def test_depth_backend_license_enforcement():
    """Ensure Depth Pro requires license acceptance."""
    config = EnhanceConfig(
        depth_backend="depth_pro",
        non_commercial_ok=False,
    )

    with pytest.raises(LicenseRestrictionError):
        registry = DepthBackendRegistry()
        backend = registry.get_backend("depth_pro", config)
```

### Integration Tests (CI Markers)

```python
@pytest.mark.ml
@pytest.mark.slow
def test_apex_dual_depth_end_to_end(tmp_path):
    """E2E test of APEX dual-depth workflow."""
    config = EnhanceConfig(
        depth_backend="depth_pro",
        fallback_depth_backend="da3",
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
        quality_tier="apex",
    )

    orchestrator = ApexOrchestrator(config)
    results = orchestrator.run(
        input_path="tests/fixtures/sample_image.jpg",
        output_path=tmp_path,
    )

    assert results.success_count == 1
    assert (tmp_path / "sample_image_enhanced.tif").exists()
    assert (tmp_path / "sample_image_depth.png").exists()
    assert (tmp_path / "sample_image_normal.png").exists()
```

---

## Future Enhancements

### Phase 4: Advanced Features (Post-3.0.0)

1. **Temporal Depth Consistency** (for video workflows)
   - Frame-to-frame depth smoothing
   - Optical flow-guided depth propagation

2. **3D Export Pipeline**
   - OBJ, PLY, USD export with metric depth
   - Integration with Blender/Unreal Engine

3. **Real-Time Preview Mode**
   - WebSocket-based live preview
   - Progressive refinement (coarse → fine)

4. **Custom Room Strategy Editor**
   - YAML-based strategy definitions
   - Visual strategy previewer

5. **Performance Ledger Dashboards**
   - Web UI for performance trends
   - Automated regression alerts

---

## Conclusion

The **APEX workflow** represents a production-grade, fully integrated depth-aware rendering pipeline optimized for luxury real estate and architectural visualization. By combining:

- **Dual-backend depth intelligence** (metric + relative)
- **Semantic room understanding** (Materials V3)
- **Advanced PBR generation** (normal, roughness, AO)
- **Deterministic quality enforcement** (firewall + provenance)
- **Multi-layer license governance** (safe research model integration)

...APEX delivers **professional-grade outputs** with **reproducible quality**, **comprehensive auditability**, and **safe change management**.

**Status:** Production-ready for research use (Depth Pro) and commercial use (DA3 only).

---

**Document Version:** 3.0.0
**Last Updated:** 2026-02-07
**Author:** Transformation Portal Engineering Team
**Related Docs:** ADR-019, QUALITY_FIREWALL_QUICK_REF.md, TEST_REPORT_LUX_DEPTH_V3_APEX.md
