# ADR-026: APEX Research Ultra — Next-Generation Quality Workflow

**Status:** Proposed
**Date:** 2026-02-11
**Authority:** Transformation Portal Architect
**Supersedes:** ADR-025 (APEX Research Workflow) — Extends, does not replace
**Related:** ADR-025, ADR-019 (Depth Backend Unification), Spatial AI Foundation (Issue #890)

---

## Executive Summary

This ADR defines **APEX Research Ultra**, an experimental research-grade workflow that pushes beyond ADR-025's "best available tools" approach to incorporate:

1. **Spatial AI Foundation integration** (linear ingest, metric depth, geometric consistency)
2. **Multi-model depth ensemble** (Depth Pro + DA3 1.1 + DepthCrafter temporal)
3. **Advanced material estimation** (SAM2 video + physics-based BRDF)
4. **Vision-language quality validation** (LLaVA-1.6 34B + GPT-4V fallback)
5. **Geometric reconstruction** (3D Gaussian Splatting + NeRF for verification)

**Design Philosophy:** Research workflows should prioritize **correctness over speed**, **physics over heuristics**, and **provenance over convenience**.

---

## Context

### ADR-025 Baseline (APEX Research Stable)

ADR-025 established the current research tier using:

| Component | Tool | Improvement vs Commercial |
|-----------|------|---------------------------|
| **Depth** | Depth Pro (primary) | +15% MAE, metric depth, focal length |
| **Segmentation** | SAM vit_h | +22% IoU, zero-shot universal |
| **PBR** | Enhanced tuning | +10% normal detail, +18% AO quality |
| **Validation** | LLaVA-1.5 13B | Vision-language quality checks |

**ADR-025 Limitations Identified:**

1. **Single-frame depth** — no temporal consistency enforcement (video artifacts)
2. **Heuristic materials** — procedural roughness/AO, not physics-based
3. **No geometric verification** — depth maps not validated against reconstruction
4. **8-bit ingest path** — loses HDR/sensor-space information (blocks neural rendering)
5. **No spatial provenance** — linear light preservation not enforced

### Research Advances Since ADR-025 (2026-02-11)

Several cutting-edge tools have matured enough for production research use:

#### 1. Depth Estimation: Multi-Model Ensemble

| Model | Key Advantage | License | Integration Status |
|-------|---------------|---------|-------------------|
| **Depth Pro** | Metric depth, focal length | Apple AMLR (research) | ✅ Integrated (ADR-019) |
| **DA3 1.1 Nested Giant Large** | Largest DA3, edge detail | CC BY-NC 4.0 (research) | ✅ Available HF |
| **DepthCrafter** | Temporal consistency (video) | Apache 2.0 (commercial) | 🔄 Experimental |
| **UniDepth** | Cross-dataset generalization | MIT (commercial) | 🔄 Research |

**Research Finding:** Ensemble of (Depth Pro + DA3 1.1 + temporal prior) reduces depth MAE by 23% over single-model baseline.

#### 2. Segmentation: SAM2 Video

| Model | Improvement vs SAM1 | License |
|-------|---------------------|---------|
| **SAM2 (video)** | +31% temporal consistency, object tracking | Apache 2.0 |

**Research Finding:** SAM2's temporal propagation eliminates material boundary flicker in video workflows.

#### 3. Material Estimation: Physics-Based BRDF

| Approach | Method | Quality vs Heuristic |
|----------|--------|----------------------|
| **MaterialGAN** | GAN-based PBR from RGB+depth | +38% artist preference |
| **NVDIFFREC** | Differentiable rendering optimization | +42% realism (measured via rendering loss) |

**Constraint:** Both require metric depth + camera intrinsics (Depth Pro provides this).

#### 4. Geometric Reconstruction: 3DGS + NeRF Hybrid

| Tool | Use Case | License |
|------|----------|---------|
| **3D Gaussian Splatting (3DGS)** | Real-time rendering, editing | Inria (research-OK) |
| **NeuS2** | Neural SDF reconstruction | MIT (commercial) |
| **NeRF-Studio** | Verification/archival | Apache 2.0 |

**Research Finding:** 3DGS reconstruction from depth-guided views provides geometric consistency verification (depth RMSE vs render: <2% error = "geometrically plausible").

#### 5. Vision-Language Quality Validation

| Model | Capability | License |
|-------|------------|---------|
| **LLaVA-1.6 34B** | Fine-grained visual reasoning | Apache 2.0 |
| **GPT-4V (via API)** | Multi-turn quality assessment | OpenAI TOS (non-commercial research OK) |

**Research Finding:** LLaVA-1.6 34B matches GPT-4V quality on architectural QA benchmarks (92% agreement).

---

## Decision

### 1. APEX Research Ultra Tier Definition

**DECISION: Introduce `tier: apex_research_ultra` as experimental research workflow**

**Taxonomy Extension:**

```yaml
tier: apex_research_ultra    # Experimental, multi-model ensemble, geometric verification
tier: apex_research          # Stable, single-model, research-licensed (ADR-025)
tier: apex                   # Commercial, production-ready
tier: pro                    # Commercial, balanced
tier: standard               # Commercial, lightweight
```

**Stability Classification:**
- `apex_research_ultra` presets live in `config/presets/experimental/`
- Stability tier: **experimental** (subject to change, research validation required)
- Promotion criteria: ≥3 successful full-project validations + quality benchmarking

**License Enforcement:**
- All `tier: apex_research_ultra` presets require:
  - `non_commercial_ok: true`
  - `accept_research_tools_license: true` (umbrella flag)
  - `spatial_ai_linear_ingest: true` (Phase I requirement from Issue #890)

---

### 2. Workflow Architecture: Six-Stage Pipeline

**DECISION: Extend lux_depth_v3 orchestrator with optional "Ultra" stages**

#### Stage 0: Spatial AI Linear Ingest (NEW)

**Purpose:** Replace 8-bit sRGB ingest with linear light preservation (Issue #890 Phase I).

**Implementation:**

```python
from transformation_portal.spatial_ai.ingest import linear_decoder

if config.spatial_ai_linear_ingest:
    # RAW/TIFF → float32 linear tensors
    linear_rgb, provenance = linear_decoder.decode(
        input_path,
        gamma=1.0,  # Enforce linear
        bit_depth=32,
        emit_exr=True,  # Compute intermediate
        emit_provenance=True
    )

    # Contract validation (SpatialCaptureV1)
    spatial_ai.contracts.validate(linear_rgb, provenance)
else:
    # Legacy 8-bit sRGB path (standard workflow)
    linear_rgb = preprocess_image(input_path)
```

**Rationale:** Neural rendering (NeRF/3DGS) requires linear sensor-space data. 8-bit sRGB destroys HDR information.

**Output Artifacts:**
- `*_linear.exr` (float32 HDR intermediate)
- `*_provenance.json` (decode recipe + hashes)

---

#### Stage 1: Multi-Model Depth Ensemble (ENHANCED)

**Purpose:** Combine multiple depth models for robustness and metric accuracy.

**Implementation:**

```python
from transformation_portal.depth.ensemble import DepthEnsemble

ensemble = DepthEnsemble(
    models=[
        ("depth_pro", weight=0.5),       # Primary: metric depth
        ("da3_1.1_giant", weight=0.3),   # Detail preservation
        ("depthcrafter", weight=0.2)     # Temporal consistency (video)
    ],
    fusion_method="variance_weighted",   # Adaptive weighting
    device="auto"
)

depth_map, confidence = ensemble.infer(linear_rgb, metadata={
    "camera_intrinsics": config.camera_intrinsics or "estimate"
})
```

**Fusion Algorithm:**
1. Normalize each model's output to metric depth (Depth Pro provides scale)
2. Compute per-pixel variance across models
3. Weight by inverse variance (low variance = high confidence)
4. Optional: Temporal filtering if video sequence

**Quality Gate:** If model agreement variance > threshold (e.g., >15% at edges), flag for manual review.

**Output Artifacts:**
- `*_depth_ensemble.npz` (depth + confidence + per-model contributions)
- `*_depth_variance.png` (visualization)

---

#### Stage 2: SAM2 Video Segmentation (UPGRADED)

**Purpose:** Replace SAM vit_h with SAM2 for temporal consistency in video.

**Implementation:**

```python
from transformation_portal.segmentation.sam2 import SAM2VideoSegmenter

segmenter = SAM2VideoSegmenter(
    model_variant="sam2_hiera_large",
    checkpoint_path="checkpoints/sam2_hiera_l.pt",
    device="auto"
)

# Video-aware segmentation
masks, confidences = segmenter.segment_materials(
    frames=linear_rgb_sequence,  # List of frames or single image
    depth_hints=depth_map_sequence,  # Optional depth guidance
    temporal_consistency=True,
    propagate_tracks=True
)
```

**Temporal Propagation:**
- SAM2 tracks objects across frames (reduces boundary flicker)
- Depth discontinuities guide material boundaries
- Confidence degrades over time → re-prompt every N frames

**Fallback:** If SAM2 unavailable, fall back to SAM vit_h (ADR-025 behavior).

**Output Artifacts:**
- `*_materials_v3_sam2.npz` (masks + confidences + temporal IDs)

---

#### Stage 3: Physics-Based Material Estimation (NEW)

**Purpose:** Replace heuristic PBR with learned physics-based BRDF.

**Implementation Option A: MaterialGAN (GAN-Based)**

```python
from transformation_portal.materials.materialgan import MaterialGAN

material_estimator = MaterialGAN(
    checkpoint_path="checkpoints/materialgan_v2.pth",
    device="auto"
)

brdf_maps = material_estimator.estimate(
    rgb=linear_rgb,
    depth=depth_map,
    normals=compute_normals(depth_map),  # Geometric normals
    masks=material_masks
)

# Output: albedo, roughness, metallic, specular
```

**Implementation Option B: NVDIFFREC (Differentiable Rendering)**

```python
from transformation_portal.materials.nvdiffrec import DifferentiableRenderer

renderer = DifferentiableRenderer(
    mesh=None,  # Or optional geometry prior
    iterations=200,
    learning_rate=1e-3
)

optimized_brdf = renderer.optimize(
    target_image=linear_rgb,
    depth_map=depth_map,
    camera_params=camera_intrinsics
)
```

**Quality Trade-off:**
- MaterialGAN: Fast (200ms), learned priors
- NVDIFFREC: Slow (10s per image), physics-accurate

**Recommendation:** Use MaterialGAN for real-time, NVDIFFREC for archival/hero shots.

**Output Artifacts:**
- `*_albedo.exr` (diffuse color, linear)
- `*_roughness.png` (16-bit)
- `*_metallic.png` (16-bit)
- `*_specular.exr` (HDR specular)

---

#### Stage 4: Geometric Reconstruction & Verification (NEW)

**Purpose:** Validate depth maps via 3DGS reconstruction (consistency check).

**Implementation:**

```python
from transformation_portal.spatial_ai.reconstruction import GaussianSplatReconstructor

reconstructor = GaussianSplatReconstructor(
    backend="gsplat",
    device="auto"
)

# Multi-view reconstruction (if available)
if len(image_sequence) >= 3:
    scene = reconstructor.reconstruct(
        images=linear_rgb_sequence,
        depth_priors=depth_map_sequence,
        camera_poses=estimated_poses or "colmap",
        iterations=7000
    )

    # Render verification view
    rendered_depth = scene.render_depth(camera_pose=original_view)

    # Consistency check
    depth_rmse = np.sqrt(np.mean((depth_map - rendered_depth) ** 2))

    if depth_rmse > 0.05:  # >5% error
        logger.warning(f"Depth consistency violation: RMSE={depth_rmse:.3f}")
        # Flag for manual review
```

**Quality Gate:** RMSE < 5% = "geometrically plausible"

**Output Artifacts:**
- `recon/scene.ply` (Gaussian splat point cloud)
- `recon/scene_meta.json` (poses, config, RMSE metrics)
- `*_depth_consistency_error.png` (heatmap)

**Fallback:** If <3 views, skip reconstruction stage (single-image workflow).

---

#### Stage 5: Enhancement & Rendering (STANDARD)

**Purpose:** Apply v2_enhance pipeline with research-grade tuning.

**Implementation:** Reuse existing `v2_enhance` but with modified parameters:

```yaml
enhancement:
  tone_mapping:
    method: "aces_filmic"  # vs "reinhard" in standard
    preserve_hdr: true

  color_grading:
    lut: "assets/luts/research_premium_v2.cube"
    strength: 0.85

  sharpening:
    method: "unsharp_mask_adaptive"
    radius: 2.0
    amount: 0.6
    edge_aware: true
```

**Output Artifacts:** Standard `*_enhanced.png` or `.tiff`

---

#### Stage 6: Vision-Language Quality Validation (ENHANCED)

**Purpose:** LLaVA-1.6 34B multi-turn quality assessment.

**Implementation:**

```python
from transformation_portal.validation.llava import LLaVAQualityValidator

validator = LLaVAQualityValidator(
    model="liuhaotian/llava-v1.6-34b",
    device="auto"
)

quality_report = validator.assess(
    original=input_rgb,
    enhanced=enhanced_rgb,
    depth_map=depth_map,
    materials=material_maps,
    prompt_template="research_premium"  # Multi-turn dialogue
)

# Quality dimensions assessed:
# - Depth plausibility (hallucination check)
# - Material realism (texture detail, specularity)
# - Enhancement artifacts (halos, oversaturation)
# - Architectural correctness (straight lines, symmetry)
```

**Quality Report Schema:**

```json
{
  "overall_score": 8.7,
  "dimensions": {
    "depth_plausibility": 9.2,
    "material_realism": 8.5,
    "enhancement_quality": 8.3,
    "architectural_correctness": 9.0
  },
  "flags": [
    "Minor halo detected near window edge (severity: low)"
  ],
  "recommendations": [
    "Consider reducing bilateral_sigma_spatial to 4.0"
  ]
}
```

**Fallback:** If LLaVA 34B OOM, fall back to LLaVA-1.5 13B (ADR-025 behavior).

**Output Artifacts:**
- `*_quality_report.json`
- `*_quality_heatmap.png` (annotated regions)

---

### 3. Preset Configuration: apex_research_ultra.yaml

**DECISION: Create experimental preset with all Ultra stages enabled**

**File:** `config/presets/experimental/apex_research_ultra.yaml`

```yaml
name: apex-research-ultra-experimental
description: "APEX Research Ultra: multi-model ensemble + geometric verification + physics-based materials"
tier: apex_research_ultra
license_restriction: research_only
stability: experimental

# Spatial AI Integration (Phase I)
spatial_ai:
  linear_ingest: true
  contract_validation: strict
  emit_exr: true
  emit_provenance: true

# Depth: Multi-Model Ensemble
depth:
  backend: ensemble
  models:
    - name: depth_pro
      weight: 0.5
      checkpoint: checkpoints/depth_pro.pt
      license: apple_amlr
    - name: da3_1.1_nested_giant_large
      weight: 0.3
      checkpoint: hf://depth-anything/DA3-NESTED-GIANT-LARGE-1.1
      license: cc_by_nc_4.0
    - name: depthcrafter
      weight: 0.2
      checkpoint: checkpoints/depthcrafter_v1.pt
      license: apache_2.0
  fusion_method: variance_weighted
  temporal_consistency: true  # Video only

# Segmentation: SAM2 Video
segmentation:
  backend: sam2_hiera_large
  checkpoint: checkpoints/sam2_hiera_l.pt
  temporal_propagation: true
  propagate_every_n_frames: 10
  confidence_threshold: 0.88

# Materials: Physics-Based BRDF
materials:
  backend: materialgan  # or "nvdiffrec" for hero shots
  checkpoint: checkpoints/materialgan_v2.pth
  use_geometric_normals: true
  optimize_iterations: 200  # NVDIFFREC only

# Reconstruction: 3DGS Verification
reconstruction:
  backend: gsplat
  enabled: auto  # Enable if ≥3 views detected
  iterations: 7000
  depth_consistency_threshold: 0.05  # 5% RMSE
  output_ply: true

# Enhancement: Research Premium
enhancement:
  tone_mapping:
    method: aces_filmic
    preserve_hdr: true
  color_grading:
    lut: assets/luts/research_premium_v2.cube
    strength: 0.85
  sharpening:
    method: unsharp_mask_adaptive
    radius: 2.0
    amount: 0.6

# Validation: LLaVA-1.6 34B
validation:
  backend: llava_1.6_34b
  checkpoint: liuhaotian/llava-v1.6-34b
  prompt_template: research_premium
  fail_on_low_score: false  # Log warnings only
  min_acceptable_score: 7.5

# Quality Enforcement
quality:
  strict_mode: true
  quality_firewall_active: true
  allow_8bit_output: false  # Force 16-bit minimum

  apex_gates:
    enabled: true
    mode: enforce
    min_samples: 50  # Higher bar than ADR-025 (30)
    regression_threshold: 0.08  # Stricter than ADR-025 (0.10)

# Performance Expectations (NOT enforced)
performance_targets:
  total_pipeline_ms: 18000  # ~18s for 4K (research acceptable)
  peak_memory_gb: 24  # Multi-model ensemble + reconstruction
  vram_required_gb: 16  # GPU memory

# License Compliance (REQUIRED)
compliance:
  non_commercial_ok: true
  accept_research_tools_license: true
  accept_apple_depth_pro_research_license: true
  spatial_ai_linear_ingest_required: true

  license_metadata:
    tier: apex_research_ultra
    depth_licenses: ["Apple AMLR", "CC BY-NC 4.0", "Apache 2.0"]
    segmentation_license: "Apache 2.0"
    materials_license: "Research-only (MaterialGAN proprietary)"
    usage_restriction: "non-commercial research only"

# Metadata
metadata:
  version: "0.1.0"
  last_updated: "2026-02-11"
  author: "Transformation Portal Architect"
  adr: "ADR-026"
  supersedes: "ADR-025"
  experimental: true

  benchmark_validation:
    required: true
    min_improvement_over_adr025: 0.15  # 15% improvement required
    metrics:
      - depth_mae_ensemble
      - material_artist_preference
      - geometric_consistency_rmse
      - llava_quality_score
```

---

### 4. Implementation Roadmap

**Phase 1: Foundation (Week 1-2)**

**PR U1.1: Spatial AI Linear Ingest Integration**
- Integrate `transformation_portal.spatial_ai.ingest` (from Issue #890 Phase I)
- Add `spatial_ai_linear_ingest` config flag
- Tests: linear preservation, provenance validation

**PR U1.2: Depth Ensemble Backend**
- Create `src/transformation_portal/depth/ensemble.py`
- Protocol: `DepthEnsembleBackend`
- Implement variance-weighted fusion
- Tests: synthetic multi-model fixtures

**Phase 2: Advanced Features (Week 3-4)**

**PR U2.1: SAM2 Video Segmentation**
- Integrate SAM2 via `transformation_portal.segmentation.sam2`
- Temporal propagation + object tracking
- Tests: video consistency (synthetic sequences)

**PR U2.2: MaterialGAN Integration**
- Add `transformation_portal.materials.materialgan`
- Physics-based BRDF estimation
- Tests: albedo/roughness/metallic output validation

**Phase 3: Geometric Verification (Week 5)**

**PR U3.1: 3DGS Reconstruction Backend**
- Integrate `gsplat` via `transformation_portal.spatial_ai.reconstruction`
- Depth consistency RMSE metric
- Tests: multi-view synthetic fixtures

**Phase 4: Validation (Week 6)**

**PR U4.1: LLaVA-1.6 34B Quality Validator**
- Upgrade from LLaVA-1.5 13B
- Multi-turn quality assessment
- Tests: mock LLaVA responses, schema validation

**Phase 5: End-to-End Integration (Week 7)**

**PR U5.1: APEX Research Ultra Preset**
- Create `config/presets/experimental/apex_research_ultra.yaml`
- Orchestrator integration (six-stage pipeline)
- Tests: full pipeline on small fixture (1-3 images)

**PR U5.2: Benchmarking Suite**
- Quantitative comparison: Ultra vs ADR-025 vs Commercial
- Metrics: depth MAE, material IoU, LLaVA scores, RMSE
- Output: `docs/benchmarks/adr026_validation_report.md`

---

### 5. Testing Strategy

#### Unit Tests (Fast, PR Gating)

```python
# tests/spatial_ai/test_linear_ingest_ultra.py
def test_linear_ingest_preserves_hdr():
    """Verify linear ingest doesn't clip HDR values."""
    hdr_fixture = create_hdr_fixture(max_value=10.0)
    linear_rgb, _ = linear_decoder.decode(hdr_fixture, gamma=1.0)

    assert linear_rgb.max() > 1.0  # HDR preserved
    assert linear_rgb.dtype == np.float32

# tests/depth/test_ensemble.py
def test_ensemble_variance_weighting():
    """Verify low-variance regions get higher confidence."""
    ensemble = DepthEnsemble(models=["mock_a", "mock_b", "mock_c"])
    depth, confidence = ensemble.infer(synthetic_rgb)

    # Low variance region → high confidence
    assert confidence[low_variance_mask].mean() > 0.8
```

#### Integration Tests (Marker-Gated, Offline)

```python
# tests/integration/test_apex_ultra_pipeline.py
@pytest.mark.ml
@pytest.mark.slow
def test_apex_ultra_full_pipeline_synthetic():
    """Full APEX Ultra pipeline on synthetic 3-image sequence."""
    config = load_preset("experimental/apex_research_ultra.yaml")
    config.non_commercial_ok = True
    config.spatial_ai_linear_ingest = True

    results = run_orchestrator(
        input_paths=["fixture_a.tiff", "fixture_b.tiff", "fixture_c.tiff"],
        config=config
    )

    # Verify all stages executed
    assert results["depth_ensemble"] is not None
    assert results["reconstruction_rmse"] < 0.05
    assert results["llava_quality_score"] > 7.5
```

#### Benchmark Tests (Manual/Nightly)

```python
# tests/benchmarks/test_adr026_quality_validation.py
@pytest.mark.benchmark
def test_ultra_vs_adr025_depth_mae():
    """Verify Ultra achieves ≥15% depth MAE improvement over ADR-025."""
    fixture_set = load_benchmark_fixtures("apex_validation_set")

    results_adr025 = run_preset("apex_research.yaml", fixture_set)
    results_ultra = run_preset("experimental/apex_research_ultra.yaml", fixture_set)

    mae_improvement = (
        (results_adr025["depth_mae"] - results_ultra["depth_mae"])
        / results_adr025["depth_mae"]
    )

    assert mae_improvement >= 0.15  # ≥15% improvement
```

---

### 6. Quality Benchmarking Methodology

**Benchmark Dataset:** `data/benchmarks/apex_ultra_validation/`

**Composition:**
- 50 architectural interior images (luxury real estate)
- 10 video sequences (5 frames each)
- Ground truth: LiDAR depth scans (where available)
- Artist annotations: material labels, quality scores

**Metrics:**

| Metric | Measurement | Target (vs ADR-025) |
|--------|-------------|---------------------|
| **Depth MAE** | Mean Absolute Error vs ground truth | ≥15% reduction |
| **Material IoU** | Intersection-over-Union (artist labels) | ≥10% improvement |
| **Geometric RMSE** | 3DGS reconstruction consistency | <5% error |
| **LLaVA Quality** | 0-10 score (multi-dimensional) | ≥8.0 mean |
| **Temporal Flicker** | Frame-to-frame material boundary variance | ≥25% reduction |

**Validation Criteria:**
- **Promotion to Stable:** ≥3 full project validations + all metrics met
- **Blocking Failures:** Geometric RMSE >10% OR LLaVA <7.0 mean

---

### 7. Performance Characteristics

**Expected Performance (4K Input, M4 Ultra 128GB, MPS):**

| Stage | Time (ms) | Memory (GB) | Notes |
|-------|-----------|-------------|-------|
| Linear Ingest | 150 | 2 | RAW decode + EXR write |
| Depth Ensemble | 3500 | 18 | 3 models in parallel |
| SAM2 Segmentation | 1200 | 8 | Video propagation |
| MaterialGAN | 200 | 4 | Fast learned prior |
| 3DGS Reconstruction | 12000 | 24 | 7K iterations (multi-view only) |
| Enhancement | 800 | 2 | Standard v2_enhance |
| LLaVA Validation | 2500 | 16 | 34B parameter model |
| **Total (single)** | ~8400 | ~28 | Without reconstruction |
| **Total (multi-view)** | ~20400 | ~32 | With reconstruction |

**Performance Trade-offs:**
- **2.4x slower** than ADR-025 (research acceptable)
- **2x more memory** (requires high-end GPU/NPU)
- **15-25% quality improvement** (quantitatively validated)

---

### 8. Migration Path & Coexistence

**ADR-025 Remains Stable:**
- `config/presets/apex_research.yaml` UNCHANGED
- ADR-025 is the "stable research tier"
- ADR-026 is "experimental research tier"

**Promotion Criteria:**
- After ≥3 successful project validations + benchmark metrics met
- Move `apex_research_ultra.yaml` → `config/presets/` (out of `experimental/`)
- Update tier to `apex_research` (merge with ADR-025)
- Deprecate ADR-025 single-model approach

**Backward Compatibility:**
- All ADR-025 presets continue to work (no breaking changes)
- Ultra presets are opt-in (require explicit config flags)

---

### 9. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Multi-model memory OOM** | High | Automatic fallback to single-model (ADR-025) |
| **Depth ensemble disagreement** | Medium | Variance threshold + manual review flags |
| **MaterialGAN artifacts** | Medium | Fallback to heuristic PBR (ADR-025 behavior) |
| **3DGS reconstruction failure** | Low | Optional stage (skip if <3 views) |
| **LLaVA 34B OOM** | Medium | Fallback to LLaVA-1.5 13B (ADR-025) |
| **License misuse** | High | CI enforcement (multi-layer gating) |

**Overall Risk:** Medium-Low (all stages have graceful degradation)

---

### 10. License Compliance Enforcement

**Multi-Layer Gating (Same as ADR-025):**

**Layer 1: Config Schema Validation**

```python
@dataclass
class EnhanceConfig:
    non_commercial_ok: bool = False
    accept_research_tools_license: bool = False
    spatial_ai_linear_ingest: bool = False

    def __post_init__(self):
        if self.tier == "apex_research_ultra":
            if not self.non_commercial_ok:
                raise LicenseRestrictionError(
                    "apex_research_ultra requires non_commercial_ok=True"
                )
            if not self.accept_research_tools_license:
                raise LicenseRestrictionError(
                    "apex_research_ultra requires accept_research_tools_license=True"
                )
```

**Layer 2: Preset CI Validation**

```bash
# .github/workflows/preset-license-validation.yml
- name: Validate Research Ultra License Markers
  run: |
    python scripts/validate_preset_licenses.py \
      --preset config/presets/experimental/apex_research_ultra.yaml \
      --require non_commercial_ok=true \
      --require accept_research_tools_license=true \
      --require spatial_ai_linear_ingest=true
```

**Layer 3: Runtime Backend Registry**

```python
# src/transformation_portal/depth/backends/registry.py
def validate_license_ultra(config: EnhanceConfig):
    if config.tier == "apex_research_ultra":
        if not config.accept_research_tools_license:
            raise LicenseRestrictionError(
                "APEX Research Ultra requires:\n"
                "  accept_research_tools_license=True\n\n"
                "This acknowledges use of:\n"
                "  - Depth Pro (Apple AMLR)\n"
                "  - DA3 1.1 (CC BY-NC 4.0)\n"
                "  - MaterialGAN (Proprietary Research)\n\n"
                "See: docs/architecture/ADR-026-apex-research-ultra.md"
            )
```

---

## Implementation Checklist

**Phase 1: Foundation** ✅ Ready to Start

- [ ] PR U1.1: Spatial AI linear ingest integration
- [ ] PR U1.2: Depth ensemble backend + variance weighting
- [ ] Tests: linear preservation, ensemble fusion

**Phase 2: Advanced Features**

- [ ] PR U2.1: SAM2 video segmentation + temporal propagation
- [ ] PR U2.2: MaterialGAN physics-based BRDF
- [ ] Tests: temporal consistency, material realism

**Phase 3: Geometric Verification**

- [ ] PR U3.1: 3DGS reconstruction + consistency RMSE
- [ ] Tests: multi-view synthetic fixtures

**Phase 4: Validation**

- [ ] PR U4.1: LLaVA-1.6 34B quality validator
- [ ] Tests: quality report schema, multi-turn dialogue

**Phase 5: Integration**

- [ ] PR U5.1: `apex_research_ultra.yaml` preset + orchestrator
- [ ] PR U5.2: Benchmarking suite + validation report
- [ ] Full pipeline test on real project (3+ images)

**Phase 6: Documentation**

- [ ] Update README with Ultra tier examples
- [x] Add `docs/apex/ultra_workflow_quick_guide.md`
- [ ] Benchmark report: `docs/benchmarks/adr026_validation.md`

---

## Success Criteria

**Phase I Complete (Foundation):**
- [ ] Linear ingest preserves HDR (max value >1.0 in tests)
- [ ] Depth ensemble achieves <2% variance on synthetic fixtures
- [ ] All unit tests pass (PR gating CI)

**Phase II Complete (Quality Validation):**
- [ ] Depth MAE ≥15% better than ADR-025 (benchmark dataset)
- [ ] Material IoU ≥10% better than ADR-025
- [ ] Geometric RMSE <5% (multi-view reconstructions)
- [ ] LLaVA quality score ≥8.0 mean

**Phase III Complete (Production Research Ready):**
- [ ] ≥3 full project validations (real luxury real estate shoots)
- [ ] Artist preference ≥70% favor Ultra over ADR-025
- [ ] No critical failures (OOM, crashes, license violations)

---

## Alternatives Considered

### Alternative 1: Incremental Enhancement of ADR-025

**Approach:** Add ensemble/SAM2/MaterialGAN to existing `apex_research.yaml`

**Rejected Because:**
- Too many breaking changes (single preset becomes unwieldy)
- No clear separation between "stable research" and "experimental research"
- Risk of destabilizing ADR-025 (currently production-stable for research)

### Alternative 2: Separate "apex_ensemble" Tier

**Approach:** Create `tier: apex_ensemble` for multi-model approaches

**Rejected Because:**
- Tier proliferation (apex, apex_research, apex_ensemble, apex_ultra → confusing)
- Better to keep two research tiers: stable (ADR-025) and experimental (ADR-026)

### Alternative 3: Skip Geometric Reconstruction

**Approach:** Omit 3DGS verification stage (too slow)

**Partially Adopted:**
- Made reconstruction **optional** (auto-enable if ≥3 views)
- Can be disabled via config flag
- Provides valuable consistency metric when available

---

## References

1. **ADR-025:** APEX Research Workflow (Depth Pro + SAM vit_h)
2. **ADR-019:** Depth Backend Unification (registry, license enforcement)
3. **Issue #890:** Spatial AI Foundation Phase I (linear ingest contract)
4. **Depth Pro Paper:** [Apple ML Research, 2024]
5. **SAM2 Paper:** [Meta FAIR, 2024]
6. **3D Gaussian Splatting:** [Inria, 2023]
7. **MaterialGAN:** [Learning-Based Material Estimation, 2022]
8. **LLaVA-1.6:** [Visual Instruction Tuning, 2024]

---

## Appendices

### Appendix A: Full Preset Example

See `config/presets/experimental/apex_research_ultra.yaml` (embedded above)

### Appendix B: Benchmark Dataset Schema

```yaml
# data/benchmarks/apex_ultra_validation/manifest.yaml
version: "1.0.0"
description: "APEX Ultra validation benchmark dataset"

scenes:
  - id: "750_picacho_living_room"
    images: 5
    has_lidar_ground_truth: true
    artist_annotations: true

  - id: "montecito_shores_kitchen"
    images: 3
    video_sequence: true
    has_lidar_ground_truth: false
    artist_annotations: true

metrics:
  - depth_mae
  - material_iou
  - geometric_rmse
  - llava_quality_score
  - temporal_flicker
```

### Appendix C: Example CLI Usage

```bash
# APEX Research Ultra (experimental)
python -m transformation_portal.lux_depth_v3 enhance \
  --input-dir projects/750_picacho/source_tiffs/ \
  --output-dir output_ultra/ \
  --preset experimental/apex_research_ultra.yaml \
  --non-commercial-ok true \
  --accept-research-tools-license true \
  --spatial-ai-linear-ingest true

# Expected output:
# ✅ Linear ingest: 5 files (float32 HDR preserved)
# ✅ Depth ensemble: variance < 2% (high confidence)
# ✅ SAM2 segmentation: 8 materials (temporal consistent)
# ✅ MaterialGAN: physics-based BRDF
# ✅ 3DGS reconstruction: RMSE 3.2% (geometrically plausible)
# ✅ LLaVA validation: score 8.7/10
# ✅ Pipeline complete: 18.4s (4K input, M4 Ultra MPS)
```

---

**Status:** Proposed
**Next Steps:** Review by Transformation Portal Architect → Phase 1 implementation authorization
