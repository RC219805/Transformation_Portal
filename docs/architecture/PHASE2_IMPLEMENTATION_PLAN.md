# Phase 2 Implementation Plan: SAM2, MaterialGAN, 3D Gaussian Splatting

**Status:** Proposed
**Date:** 2026-02-11
**Authority:** Transformation Portal Architect
**Prerequisites:** Phase 1 (PR #906) ✅, Phase 1.1 (PR #907) ✅
**Related ADRs:** ADR-023 (Isolation), ADR-026 (APEX Research Ultra)

---

## Executive Summary

Phase 2 extends the Spatial AI Foundation with three advanced computer vision capabilities:

1. **SAM2 Integration** — Segment Anything Model 2 for material/object segmentation with temporal consistency
2. **MaterialGAN Integration** — Physics-based material property inference and PBR texture generation
3. **3D Gaussian Splatting** — Depth-guided 3D scene reconstruction for geometric verification

**Key Design Principle:** Respect Phase 1.1 constraints. Don't fight them.

### Strategic Context

- **Phase 1 delivered:** Linear ingest pipeline, depth ensemble backend (DA3 + Depth Pro)
- **Phase 1.1 hardened:** Contract integrity (gamma=1.0, strict_ingest, EXR fail-loud), governance gates (AST isolation, HF revision pinning)
- **Phase 2 extends:** Add segmentation, material inference, geometric reconstruction on top of validated spatial foundation

### Success Criteria

Phase 2 is complete when:

1. ✅ SAM2 video segmentation integrated with temporal consistency
2. ✅ MaterialGAN generates PBR textures (albedo, normal, roughness, metallic)
3. ✅ 3D Gaussian Splatting reconstructs scenes from depth-guided views
4. ✅ All components respect ADR-023 isolation (no lux_depth_v3 imports)
5. ✅ All components respect Phase 1.1 constraints (OpenEXR preflight, lane-based strictness)
6. ✅ CI enforcement passes (isolation checker, HF revision validator)
7. ✅ Integration tests validate end-to-end workflows
8. ✅ Documentation complete with experimental → stable promotion path

---

## 1. Architecture Design

### 1.1 Design Constraints (Non-Negotiable)

These constraints from Phase 1.1 are **mandatory** and **enforced by CI**:

#### Constraint A: I/O Contract Discipline
- **Gamma enforcement:** `gamma=1.0` always enforced, no override parameter
- **Strict ingest mode:** `strict_ingest` flag controls 8-bit rejection (lane-based policy)
- **EXR fail-loud:** `emit_exr=True` raises RuntimeError if OpenEXR unavailable

#### Constraint B: Pipeline Isolation (ADR-023)
- **Zero imports from `lux_depth_v3`:** Phase 2 code must not import rendering pipeline code
- **Shared utilities in `core/` or `spatial_ai/`:** Common helpers go in approved shared namespaces
- **AST enforcement:** `scripts/security/verify_pipeline_isolation.py` runs in CI

#### Constraint C: Reproducibility Enforcement
- **HF revision pinning:** All HuggingFace models must pin to commit SHAs (no `main` or placeholders)
- **Experimental presets exempt:** Placeholders allowed only in `config/presets/experimental/`
- **CI validation:** `scripts/validation/validate_hf_revisions.py` blocks non-compliant merges

### 1.2 Architectural Invariants

These are repository-level architectural rules that Phase 2 must uphold:

#### Invariant 1: Modularity and Coupling Control
- Pipelines may share **interfaces and contracts**, not **internal implementations**
- SAM2, MaterialGAN, 3DGS are **separate modules** with clear boundaries
- No circular dependencies between Phase 2 components

#### Invariant 2: Contracts Over Convenience
- Each component defines explicit input/output contracts:
  - Data types (numpy arrays, torch tensors, metadata dicts)
  - Coordinate systems (image space, world space, normalized device coordinates)
  - Colorspace/gamma expectations (linear, sRGB, display-referred)
  - File format requirements (EXR, PNG, TIFF, mesh formats)

#### Invariant 3: Determinism and Reproducibility
- Pinned model weights (HF commit SHAs)
- Deterministic preprocessing (fixed random seeds where applicable)
- Versioned configuration schemas

#### Invariant 4: Security by Default
- All file paths sanitized (prevent path traversal)
- No unsafe deserialization (pickle banned without justification)
- No shell=True in subprocess calls
- Input validation before expensive operations

---

## 2. Namespace Structure

Phase 2 components live under `spatial_ai/` with clear module boundaries.

### 2.1 Directory Layout

```
src/transformation_portal/
├── spatial_ai/                    # Phase 2 namespace (isolated from lux_depth_v3)
│   ├── __init__.py
│   ├── ingest/                    # Phase 1: Linear decoder (COMPLETE)
│   │   ├── __init__.py
│   │   └── linear_decoder.py
│   │
│   ├── segmentation/              # Phase 2.1: SAM2 (NEW)
│   │   ├── __init__.py
│   │   ├── sam2_backend.py        # SAM2 video model wrapper
│   │   ├── mask_processor.py      # Temporal consistency, refinement
│   │   ├── material_classifier.py # Material type inference (optional CLIP integration)
│   │   └── contracts.py           # SegmentationResult, MaskMetadata contracts
│   │
│   ├── materials/                 # Phase 2.2: MaterialGAN (NEW)
│   │   ├── __init__.py
│   │   ├── material_gan.py        # MaterialGAN model wrapper
│   │   ├── pbr_generator.py       # PBR texture generation (albedo, normal, roughness, metallic)
│   │   ├── brdf_estimator.py      # BRDF parameter estimation from RGB+depth
│   │   └── contracts.py           # MaterialResult, PBRTextures contracts
│   │
│   ├── reconstruction/            # Phase 2.3: 3D Gaussian Splatting (NEW)
│   │   ├── __init__.py
│   │   ├── gaussian_splat.py      # 3DGS scene reconstruction
│   │   ├── depth_guidance.py      # Depth-guided view synthesis
│   │   ├── mesh_export.py         # Export to mesh formats (PLY, OBJ)
│   │   └── contracts.py           # ReconstructionResult, SceneGeometry contracts
│   │
│   └── orchestration/             # Phase 2.4: Integration (NEW)
│       ├── __init__.py
│       ├── spatial_pipeline.py    # End-to-end spatial workflow
│       └── validation.py          # Cross-component validation
│
├── core/                          # Shared utilities (OK to import in spatial_ai)
│   ├── processing/                # Existing: Image processing utilities
│   ├── validation/                # Existing: Input validation
│   └── geometry/                  # NEW: Geometric transforms, camera models
│       ├── __init__.py
│       ├── camera.py              # Camera intrinsics, extrinsics
│       ├── transforms.py          # 3D transforms, homogeneous coords
│       └── depth_utils.py         # Depth normalization, metric conversion
│
└── lux_depth_v3/                  # Rendering pipeline (ISOLATED, no imports from here)
    └── ...
```

### 2.2 Import Policy

**Allowed Imports:**
```python
# ✅ ALLOWED: Phase 2 → Phase 1 (same pipeline)
from transformation_portal.spatial_ai.ingest import LinearDecoder

# ✅ ALLOWED: Phase 2 → core utilities
from transformation_portal.core.geometry.camera import CameraIntrinsics
from transformation_portal.core.geometry.depth_utils import normalize_depth

# ✅ ALLOWED: Phase 2 internal (within spatial_ai)
from transformation_portal.spatial_ai.segmentation import sam2_backend
```

**Forbidden Imports:**
```python
# ❌ FORBIDDEN: Phase 2 → lux_depth_v3 (ADR-023 violation)
from transformation_portal.lux_depth_v3.utils import normalize_depth  # CI FAILS

# ❌ FORBIDDEN: lux_depth_v3 → spatial_ai (ADR-023 violation)
from transformation_portal.spatial_ai.ingest import LinearDecoder  # CI FAILS
```

**Enforcement:**
- CI runs `scripts/security/verify_pipeline_isolation.py` on every commit
- AST-based analysis detects violations with line numbers
- Violations block merge

---

## 3. Component Breakdown

### 3.1 Phase 2.1: SAM2 Integration

**Objective:** Universal object/material segmentation with temporal consistency for video workflows.

#### 3.1.1 Capabilities

- **Automatic mask generation:** Segment entire image into coherent regions
- **Prompted segmentation:** User-specified points/bounding boxes
- **Video tracking:** Temporal propagation of masks across frames
- **Material classification:** Optional CLIP-based semantic labeling

#### 3.1.2 Model Selection

| Model | Capability | License | HF Repository |
|-------|------------|---------|---------------|
| **SAM2 Base** | Image segmentation | Apache 2.0 | `facebook/sam2-hiera-base-plus` |
| **SAM2 Large** | Higher quality, video | Apache 2.0 | `facebook/sam2-hiera-large` |

**Decision:** Use SAM2 Large for research tier (better quality), SAM2 Base for speed tier.

**License Compliance:** Apache 2.0 is commercial-friendly. No tier restrictions needed.

#### 3.1.3 Technical Design

**Input Contract:**
```python
@dataclass
class SegmentationInput:
    """Input for SAM2 segmentation."""
    image: np.ndarray              # (H, W, 3) float32 linear RGB [0, ∞)
    gamma: float = 1.0             # Must be 1.0 (enforced by caller)
    mode: Literal["auto", "points", "bbox", "video"]
    prompts: Optional[List[Dict]] = None  # Points/boxes for prompted mode
    prev_masks: Optional[np.ndarray] = None  # Previous frame masks (video mode)
    frame_idx: Optional[int] = None  # Frame index in video sequence
```

**Output Contract:**
```python
@dataclass
class SegmentationResult:
    """Output from SAM2 segmentation."""
    masks: np.ndarray              # (N, H, W) bool, N masks
    scores: np.ndarray             # (N,) float32, confidence scores [0, 1]
    metadata: List[MaskMetadata]   # Per-mask metadata
    temporal_ids: Optional[np.ndarray] = None  # (N,) int, tracking IDs for video

@dataclass
class MaskMetadata:
    """Per-mask metadata."""
    area: int                      # Pixel count
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    stability_score: float         # Mask stability [0, 1]
    material_label: Optional[str] = None  # CLIP classification (if enabled)
    material_confidence: Optional[float] = None
```

**Implementation Files:**

1. **`sam2_backend.py`** — SAM2 model loading, inference
   - HuggingFace model loading with revision pinning
   - GPU/CPU device selection
   - Batched inference for efficiency
   - Error handling (OOM, missing weights)

2. **`mask_processor.py`** — Post-processing, temporal consistency
   - Mask refinement (morphological operations)
   - Temporal tracking (assign consistent IDs across frames)
   - Overlap resolution (handle overlapping masks)
   - Quality filtering (score thresholds, area thresholds)

3. **`material_classifier.py`** — Optional CLIP integration
   - Semantic labeling of segments (wood, marble, glass, metal, fabric)
   - Confidence scoring
   - Fallback to unlabeled if CLIP unavailable

4. **`contracts.py`** — Data contracts (SegmentationInput, SegmentationResult, MaskMetadata)

#### 3.1.4 Integration Points

**With Phase 1 Ingest:**
```python
from transformation_portal.spatial_ai.ingest import LinearDecoder
from transformation_portal.spatial_ai.segmentation import SAM2Backend

decoder = LinearDecoder(gamma=1.0, bit_depth=32)
ingest_result = decoder.decode("scene.tiff", emit_exr=False)

segmenter = SAM2Backend(model_size="large", device="cuda")
seg_result = segmenter.segment(
    image=ingest_result.linear_rgb,
    gamma=ingest_result.gamma,  # Pass through (must be 1.0)
    mode="auto"
)
```

**With Phase 1 Depth Ensemble:**
```python
from transformation_portal.depth.backends.ensemble import DepthEnsemble

ensemble = DepthEnsemble(backends=["depth_pro", "da3_large"])
depth_result = ensemble.estimate_depth(ingest_result.linear_rgb)

# Pass depth to material inference (Phase 2.2)
```

#### 3.1.5 Testing Strategy

**Unit Tests:**
- Model loading (with mocked weights for CI)
- Mask post-processing (morphological ops)
- Temporal tracking logic (ID consistency)
- Contract validation (input/output shapes, dtypes)

**Integration Tests:**
- End-to-end segmentation on test image
- Video mode with 3-frame sequence
- CLIP classification (if available)
- Error handling (OOM simulation, missing deps)

**Fixtures:**
- `tests/fixtures/spatial_ai/sample_linear_rgb.npy` — (512, 512, 3) float32
- `tests/fixtures/spatial_ai/sample_video_frames/` — 3-frame sequence

**CI Configuration:**
- Install SAM2 from HuggingFace (pinned revision)
- Skip CLIP tests if not in CI (optional dependency)
- Mock GPU with CPU fallback

---

### 3.2 Phase 2.2: MaterialGAN Integration

**Objective:** Physics-based material property inference and PBR texture generation from RGB + depth.

#### 3.2.1 Capabilities

- **BRDF estimation:** Infer diffuse albedo, specular reflectance, roughness, metallic
- **PBR texture generation:** Generate albedo, normal, roughness, metallic, AO maps
- **Metric depth requirement:** Requires metric depth (Depth Pro provides this)
- **Camera intrinsics:** Uses focal length from Depth Pro for geometric reasoning

#### 3.2.2 Model Selection

**Research Assessment:**

| Model | Method | Quality | License | Availability |
|-------|--------|---------|---------|--------------|
| **MaterialGAN** | GAN-based PBR | High | Research (CC BY-NC 4.0) | GitHub |
| **NVDIFFREC** | Differentiable render | Very High | BSD-3-Clause | NVIDIA |
| **Text2Mat** | Diffusion-based | Medium | Research | HuggingFace |

**Decision:** Start with **NVDIFFREC** (BSD-3-Clause = commercial OK, better quality).

**Fallback:** Heuristic PBR (existing `lux_depth_v3/pbr.py` approach) if NVDIFFREC unavailable.

**Note:** NVDIFFREC requires differentiable rendering (nvdiffrast), GPU-only.

#### 3.2.3 Technical Design

**Input Contract:**
```python
@dataclass
class MaterialInput:
    """Input for material property inference."""
    rgb: np.ndarray                # (H, W, 3) float32 linear RGB [0, ∞)
    depth: np.ndarray              # (H, W) float32 metric depth (meters)
    normals: Optional[np.ndarray] = None  # (H, W, 3) float32 surface normals
    camera: Optional[CameraIntrinsics] = None  # Camera parameters
    gamma: float = 1.0             # Must be 1.0
    masks: Optional[np.ndarray] = None  # (N, H, W) bool, per-material masks from SAM2
```

**Output Contract:**
```python
@dataclass
class MaterialResult:
    """Output from material property inference."""
    pbr_textures: PBRTextures
    material_params: Dict[str, Any]  # BRDF parameters
    quality_metrics: Dict[str, float]  # Reconstruction loss, etc.

@dataclass
class PBRTextures:
    """PBR texture maps."""
    albedo: np.ndarray             # (H, W, 3) float32 linear RGB
    normal: np.ndarray             # (H, W, 3) float32 world-space normals
    roughness: np.ndarray          # (H, W) float32 [0, 1]
    metallic: np.ndarray           # (H, W) float32 [0, 1]
    ambient_occlusion: np.ndarray  # (H, W) float32 [0, 1]
```

**Implementation Files:**

1. **`material_gan.py`** — NVDIFFREC model wrapper
   - Model initialization (nvdiffrast, neural network)
   - Optimization loop (differentiable rendering)
   - GPU memory management
   - Fallback to heuristic if GPU unavailable

2. **`pbr_generator.py`** — PBR texture generation
   - Per-material optimization (if masks provided)
   - Texture resolution upsampling
   - Normal map refinement (align with depth gradients)
   - AO baking from depth + normals

3. **`brdf_estimator.py`** — BRDF parameter estimation
   - Physical plausibility constraints (energy conservation)
   - Material presets (wood, marble, metal, glass)
   - Multi-bounce indirect lighting (optional)

4. **`contracts.py`** — Data contracts (MaterialInput, MaterialResult, PBRTextures)

#### 3.2.4 Integration Points

**With SAM2 Segmentation:**
```python
from transformation_portal.spatial_ai.segmentation import SAM2Backend
from transformation_portal.spatial_ai.materials import MaterialEstimator

# Segment scene
seg_result = segmenter.segment(ingest_result.linear_rgb, mode="auto")

# Estimate materials per-segment
material_estimator = MaterialEstimator(backend="nvdiffrec", device="cuda")
material_result = material_estimator.estimate(
    rgb=ingest_result.linear_rgb,
    depth=depth_result.depth_map,
    normals=depth_result.normals,  # From depth gradient
    camera=depth_result.camera_intrinsics,
    masks=seg_result.masks  # Per-segment optimization
)
```

**With Depth Ensemble:**
```python
# Depth Pro provides metric depth + camera intrinsics
depth_result = ensemble.estimate_depth(ingest_result.linear_rgb)
assert depth_result.is_metric  # Required for MaterialGAN
assert depth_result.camera_intrinsics is not None
```

#### 3.2.5 Testing Strategy

**Unit Tests:**
- BRDF parameter validation (energy conservation)
- PBR texture generation (shape, dtype, range)
- Heuristic fallback (no GPU)
- Contract validation

**Integration Tests:**
- End-to-end material estimation on test scene
- Per-segment optimization with SAM2 masks
- Quality metrics validation (reconstruction loss)

**Fixtures:**
- `tests/fixtures/spatial_ai/sample_depth_metric.npy` — (512, 512) float32 metric depth
- `tests/fixtures/spatial_ai/sample_normals.npy` — (512, 512, 3) float32 normals
- `tests/fixtures/spatial_ai/camera_intrinsics.json` — Camera parameters

**CI Configuration:**
- Install nvdiffrast (GPU required, skip on CPU CI runners)
- Test heuristic fallback on CPU
- Validate PBR texture quality thresholds

---

### 3.3 Phase 2.3: 3D Gaussian Splatting

**Objective:** Depth-guided 3D scene reconstruction for geometric verification and novel view synthesis.

#### 3.3.1 Capabilities

- **Scene reconstruction:** 3D Gaussian primitives from RGB + depth
- **Novel view synthesis:** Render from arbitrary camera viewpoints
- **Geometric verification:** Compare rendered depth vs input depth (RMSE < 2% = plausible)
- **Mesh export:** Convert to PLY, OBJ for 3D editing tools

#### 3.3.2 Model Selection

| Tool | Method | License | Repository |
|------|--------|---------|------------|
| **3D Gaussian Splatting (3DGS)** | Real-time rendering | Inria (research OK) | `graphdeco-inria/gaussian-splatting` |
| **NeuS2** | Neural SDF | MIT | `19reborn/NeuS2` |
| **NeRF-Studio** | NeRF variants | Apache 2.0 | `nerfstudio-project/nerfstudio` |

**Decision:** Use **3DGS** for real-time rendering and editing. **NeuS2** for high-quality SDF reconstruction (optional).

**License Compliance:**
- 3DGS: Inria license restricts commercial use → `tier: apex_research` or higher only
- NeuS2: MIT → commercial OK
- NeRF-Studio: Apache 2.0 → commercial OK

#### 3.3.3 Technical Design

**Input Contract:**
```python
@dataclass
class ReconstructionInput:
    """Input for 3D reconstruction."""
    rgb_images: List[np.ndarray]   # [(H, W, 3)] float32 linear RGB
    depth_maps: List[np.ndarray]   # [(H, W)] float32 metric depth
    cameras: List[CameraIntrinsics]  # Per-view camera parameters
    masks: Optional[List[np.ndarray]] = None  # [(H, W)] bool, foreground masks
    gamma: float = 1.0             # Must be 1.0
```

**Output Contract:**
```python
@dataclass
class ReconstructionResult:
    """Output from 3D reconstruction."""
    gaussians: GaussianCloud       # 3D Gaussian primitives
    mesh: Optional[Mesh] = None    # Extracted mesh (if requested)
    quality_metrics: Dict[str, float]  # RMSE, PSNR, SSIM
    render_fn: Callable            # Function to render novel views

@dataclass
class GaussianCloud:
    """3D Gaussian primitive cloud."""
    positions: np.ndarray          # (N, 3) float32, 3D positions
    colors: np.ndarray             # (N, 3) float32, RGB colors
    scales: np.ndarray             # (N, 3) float32, Gaussian scales
    rotations: np.ndarray          # (N, 4) float32, quaternions
    opacities: np.ndarray          # (N,) float32, alpha values

@dataclass
class Mesh:
    """Triangle mesh."""
    vertices: np.ndarray           # (V, 3) float32
    faces: np.ndarray              # (F, 3) int32
    vertex_colors: Optional[np.ndarray] = None  # (V, 3) float32
```

**Implementation Files:**

1. **`gaussian_splat.py`** — 3DGS scene reconstruction
   - 3DGS optimization (CUDA kernels)
   - Depth-guided initialization
   - Adaptive density control
   - Novel view rendering

2. **`depth_guidance.py`** — Depth-guided view synthesis
   - Depth reprojection (enforce geometric consistency)
   - Multi-view consistency loss
   - Occlusion handling

3. **`mesh_export.py`** — Export to standard formats
   - Poisson surface reconstruction (Gaussians → mesh)
   - PLY export (Gaussian cloud, mesh)
   - OBJ export with MTL (mesh + textures)

4. **`contracts.py`** — Data contracts (ReconstructionInput, ReconstructionResult, GaussianCloud, Mesh)

#### 3.3.4 Integration Points

**With Depth Ensemble:**
```python
from transformation_portal.spatial_ai.reconstruction import GaussianReconstructor

# Multi-view capture (assume 5 viewpoints)
views = []
for image_path in image_paths:
    ingest_result = decoder.decode(image_path)
    depth_result = ensemble.estimate_depth(ingest_result.linear_rgb)
    views.append({
        "rgb": ingest_result.linear_rgb,
        "depth": depth_result.depth_map,
        "camera": depth_result.camera_intrinsics
    })

# Reconstruct scene
reconstructor = GaussianReconstructor(device="cuda")
recon_result = reconstructor.reconstruct(
    rgb_images=[v["rgb"] for v in views],
    depth_maps=[v["depth"] for v in views],
    cameras=[v["camera"] for v in views]
)

# Render novel view
novel_camera = CameraIntrinsics(...)
rendered_rgb = recon_result.render_fn(novel_camera)
```

**Geometric Verification:**
```python
# Compare rendered depth vs input depth
rendered_depth = recon_result.render_depth(cameras[0])
rmse = np.sqrt(np.mean((rendered_depth - depth_maps[0]) ** 2))
if rmse < 0.02 * np.mean(depth_maps[0]):  # <2% RMSE
    print("Geometric reconstruction is plausible")
```

#### 3.3.5 Testing Strategy

**Unit Tests:**
- Gaussian primitive initialization
- Depth reprojection math
- Mesh export (PLY, OBJ format validity)
- Contract validation

**Integration Tests:**
- End-to-end reconstruction with synthetic 3-view scene
- Novel view rendering
- Geometric verification (RMSE threshold)
- Mesh export and re-import

**Fixtures:**
- `tests/fixtures/spatial_ai/multiview/` — 3-view synthetic scene (RGB, depth, camera)
- `tests/fixtures/spatial_ai/reference_mesh.ply` — Ground truth mesh

**CI Configuration:**
- Install 3DGS (CUDA required, skip on CPU CI runners)
- Test with synthetic data (avoid large downloads)
- Validate mesh export formats

---

## 4. Integration Strategy

### 4.1 End-to-End Spatial Pipeline

**Objective:** Orchestrate all Phase 2 components into a cohesive workflow.

**File:** `src/transformation_portal/spatial_ai/orchestration/spatial_pipeline.py`

**Workflow Stages:**

1. **Ingest** — Linear decode (Phase 1)
2. **Depth** — Ensemble depth estimation (Phase 1)
3. **Segment** — SAM2 material segmentation (Phase 2.1)
4. **Materials** — PBR texture generation (Phase 2.2)
5. **Reconstruct** — 3DGS scene reconstruction (Phase 2.3)
6. **Validate** — Geometric consistency checks

**Example:**
```python
from transformation_portal.spatial_ai.orchestration import SpatialPipeline

pipeline = SpatialPipeline(
    tier="apex_research_ultra",
    device="cuda",
    strict_ingest=True,  # Research mode
    emit_exr=True
)

result = pipeline.process(
    input_path="luxury_villa.tiff",
    stages=["ingest", "depth", "segment", "materials", "reconstruct"],
    output_dir="output/villa_spatial/"
)

# Outputs:
# - output/villa_spatial/linear_rgb.exr
# - output/villa_spatial/depth_metric.exr
# - output/villa_spatial/depth_normals.exr
# - output/villa_spatial/segmentation_masks.npz
# - output/villa_spatial/pbr_albedo.exr
# - output/villa_spatial/pbr_normal.exr
# - output/villa_spatial/pbr_roughness.exr
# - output/villa_spatial/pbr_metallic.exr
# - output/villa_spatial/reconstruction.ply
# - output/villa_spatial/provenance.json
```

### 4.2 Preset Strategy

**Experimental Presets:**
- Start in `config/presets/experimental/`
- Allow HF revision placeholders
- Document verification TODO

**Promotion Path:**
1. Develop in experimental preset
2. Verify model weights (HF commit SHAs)
3. Run 3+ full project validations
4. Update preset with pinned revisions
5. Move to `config/presets/` (stable)
6. CI validates on every push

**Example Preset:**
```yaml
# config/presets/experimental/spatial_ai_full.yaml
tier: apex_research_ultra
pipeline: spatial_ai
version: "2.0.0-experimental"

license_requirements:
  non_commercial_ok: true
  accept_research_tools_license: true

stages:
  ingest:
    gamma: 1.0
    bit_depth: 32
    strict_ingest: true
    emit_exr: true

  depth:
    ensemble:
      backends:
        - name: depth_pro
          weight: 0.6
        - name: da3_large
          weight: 0.4
          # Verified 2026-02-15 from https://huggingface.co/depth-anything/DA3-NESTED-GIANT-LARGE-1.1/commits/main
          revision: "NEEDS_VERIFICATION_0000000000000000000000"  # OK in experimental
    output_format: metric

  segmentation:
    model: sam2_large
    # Verified 2026-02-15 from https://huggingface.co/facebook/sam2-hiera-large/commits/main
    revision: "NEEDS_VERIFICATION_0000000000000000000000"  # OK in experimental
    mode: auto
    temporal_consistency: true

  materials:
    backend: nvdiffrec
    per_segment: true
    optimization_steps: 100

  reconstruction:
    method: 3dgs
    depth_guidance_weight: 0.5
    export_mesh: true
```

---

## 5. Testing Strategy

### 5.1 Test Pyramid

```
                    ╱╲
                   ╱  ╲  E2E Integration (3 tests)
                  ╱    ╲  - Full pipeline on test scene
                 ╱──────╲  - Multi-view reconstruction
                ╱        ╲ - Cross-component validation
               ╱──────────╲
              ╱            ╲ Component Integration (12 tests)
             ╱              ╲ - SAM2 + LinearDecoder
            ╱────────────────╲ - MaterialGAN + Depth
           ╱                  ╲ - 3DGS + Depth
          ╱────────────────────╲
         ╱                      ╲ Unit Tests (40+ tests)
        ╱________________________╲ - Model loading, contracts, post-processing
```

### 5.2 Unit Test Coverage

**Per Component:**

| Component | Unit Tests | Coverage Target |
|-----------|------------|-----------------|
| SAM2 Backend | 10 tests | ≥90% |
| Material Estimator | 12 tests | ≥85% |
| 3DGS Reconstructor | 10 tests | ≥85% |
| Orchestration | 8 tests | ≥90% |

**Test Types:**
- Model loading (mocked weights)
- Contract validation (input/output shapes)
- Error handling (OOM, missing deps)
- Determinism (fixed seeds)

### 5.3 Integration Test Coverage

**SAM2 Integration:**
- LinearDecoder → SAM2 (gamma=1.0 validation)
- SAM2 → Material Classifier (CLIP)
- Video mode temporal tracking

**MaterialGAN Integration:**
- SAM2 masks → Per-segment material optimization
- Depth → BRDF estimation (metric depth required)
- PBR texture export (EXR with preflight check)

**3DGS Integration:**
- Multi-view depth → Scene reconstruction
- Depth consistency verification (RMSE < 2%)
- Mesh export (PLY, OBJ)

### 5.4 Isolation Compliance Testing

**CI Enforcement:**
```bash
# Run in CI on every commit
python scripts/security/verify_pipeline_isolation.py

# Expected: PASS
# Forbidden imports detected: 0
```

**Manual Verification:**
```bash
# Check spatial_ai has no lux_depth_v3 imports
grep -r "from transformation_portal.lux_depth_v3" src/transformation_portal/spatial_ai/
# Expected: no results

# Check lux_depth_v3 has no spatial_ai imports
grep -r "from transformation_portal.spatial_ai" src/transformation_portal/lux_depth_v3/
# Expected: no results
```

### 5.5 HF Revision Validation Testing

**CI Enforcement:**
```bash
# Run in CI before merge
python scripts/validation/validate_hf_revisions.py

# Expected: PASS
# Placeholders in stable presets: 0
# Placeholders in experimental presets: OK (documented)
```

**Promotion Workflow:**
```bash
# Before promoting experimental → stable:
# 1. Verify HF commit hash manually
# 2. Update preset with real SHA
# 3. Re-run validation
python scripts/validation/validate_hf_revisions.py
# Expected: PASS
```

---

## 6. Implementation Phases

### Phase 2.1: SAM2 Integration (Week 1)

**Duration:** 5 days
**Owner:** Specialist (implementation), Architect (approval)

**Tasks:**
- [ ] Create `spatial_ai/segmentation/` module structure
- [ ] Implement SAM2 backend (model loading, inference)
- [ ] Implement mask processor (temporal tracking, refinement)
- [ ] Add material classifier (CLIP integration)
- [ ] Write unit tests (10 tests, ≥90% coverage)
- [ ] Write integration tests (SAM2 + LinearDecoder)
- [ ] Create experimental preset (`experimental/sam2_video.yaml`)
- [ ] Update documentation (`docs/spatial_ai/SAM2_INTEGRATION.md`)

**Deliverables:**
- Working SAM2 segmentation (auto + video modes)
- 10 passing unit tests
- 2 passing integration tests
- Experimental preset with placeholder revisions
- Documentation with API examples

**Success Criteria:**
- SAM2 segments test image (512x512) in <3 seconds on GPU
- Temporal tracking assigns consistent IDs across 3-frame sequence
- No imports from `lux_depth_v3` (CI passes)
- All tests green

---

### Phase 2.2: MaterialGAN Integration (Week 2)

**Duration:** 5 days
**Owner:** Specialist (implementation), Architect (approval)

**Tasks:**
- [ ] Create `spatial_ai/materials/` module structure
- [ ] Implement NVDIFFREC backend (differentiable rendering)
- [ ] Implement PBR generator (albedo, normal, roughness, metallic, AO)
- [ ] Implement BRDF estimator (parameter inference)
- [ ] Add heuristic fallback (CPU, no GPU)
- [ ] Write unit tests (12 tests, ≥85% coverage)
- [ ] Write integration tests (MaterialGAN + SAM2 + Depth)
- [ ] Create experimental preset (`experimental/material_gan.yaml`)
- [ ] Update documentation (`docs/spatial_ai/MATERIAL_GAN.md`)

**Deliverables:**
- Working material estimation (PBR textures)
- 12 passing unit tests
- 2 passing integration tests
- Experimental preset with placeholder revisions
- Documentation with PBR export examples

**Success Criteria:**
- MaterialGAN generates PBR textures (1024x1024) in <10 seconds on GPU
- Per-segment optimization works with SAM2 masks
- Heuristic fallback works on CPU (quality degradation acceptable)
- No imports from `lux_depth_v3` (CI passes)
- All tests green

---

### Phase 2.3: 3D Gaussian Splatting (Week 3)

**Duration:** 5 days
**Owner:** Specialist (implementation), Architect (approval)

**Tasks:**
- [ ] Create `spatial_ai/reconstruction/` module structure
- [ ] Implement 3DGS backend (Gaussian optimization)
- [ ] Implement depth guidance (geometric consistency)
- [ ] Implement mesh export (PLY, OBJ)
- [ ] Write unit tests (10 tests, ≥85% coverage)
- [ ] Write integration tests (3DGS + Depth)
- [ ] Create experimental preset (`experimental/gaussian_splat.yaml`)
- [ ] Update documentation (`docs/spatial_ai/3DGS_INTEGRATION.md`)

**Deliverables:**
- Working 3D reconstruction (Gaussian splatting)
- 10 passing unit tests
- 2 passing integration tests
- Experimental preset with placeholder revisions
- Documentation with mesh export examples

**Success Criteria:**
- 3DGS reconstructs 3-view scene (512x512) in <30 seconds on GPU
- Geometric verification RMSE < 2% of mean depth
- Mesh export generates valid PLY/OBJ files
- No imports from `lux_depth_v3` (CI passes)
- All tests green

---

### Phase 2.4: Orchestration & Validation (Week 4)

**Duration:** 5 days
**Owner:** Specialist (implementation), Architect (approval)

**Tasks:**
- [ ] Create `spatial_ai/orchestration/` module
- [ ] Implement SpatialPipeline (end-to-end orchestrator)
- [ ] Implement cross-component validation
- [ ] Add OpenEXR preflight checks to all scripts
- [ ] Create lane-based strictness configs (dev vs research)
- [ ] Write E2E tests (3 tests, full pipeline)
- [ ] Create unified experimental preset (`experimental/spatial_ai_full.yaml`)
- [ ] Update CLI (`transformation_portal spatial-ai process ...`)
- [ ] Verify HF revisions for promotion
- [ ] Update documentation (`docs/spatial_ai/PIPELINE.md`)

**Deliverables:**
- End-to-end spatial pipeline (all stages)
- 8 orchestration unit tests
- 3 E2E integration tests
- Unified experimental preset
- CLI integration
- Complete documentation

**Success Criteria:**
- Full pipeline processes test scene end-to-end (all stages)
- Lane-based strictness works (dev fast, research strict)
- OpenEXR preflight prevents late failures
- No imports from `lux_depth_v3` (CI passes)
- All HF revisions verified (ready for promotion)
- All tests green (60+ total)

---

### Phase 2.5: Hardening & Documentation (Week 5)

**Duration:** 3 days
**Owner:** Architect (review), Specialist (fixes)

**Tasks:**
- [ ] Architect review of architecture compliance
- [ ] Security audit (input validation, path sanitization)
- [ ] Performance profiling (identify bottlenecks)
- [ ] Documentation audit (accuracy, completeness)
- [ ] Promote verified presets to stable (if HF revisions pinned)
- [ ] Create migration guide for users
- [ ] Update CHANGELOG.md
- [ ] Create PR for Phase 2 merge

**Deliverables:**
- Architect approval (architecture compliance)
- Security approval (no vulnerabilities)
- Performance report (benchmark results)
- Complete documentation (API, guides, examples)
- Promoted stable presets (if verified)
- Migration guide
- PR ready for review

**Success Criteria:**
- Architect approval received
- No security vulnerabilities (path traversal, unsafe deserialization)
- Performance regression <5% vs Phase 1
- Documentation complete and accurate
- CI green (all checks passing)
- Ready for merge to main

---

## 7. Risk Assessment & Mitigation

### Risk 1: Model Dependency Hell

**Risk:** Phase 2 adds 3+ new ML models with complex dependency trees (SAM2, NVDIFFREC, 3DGS). Dependency conflicts likely.

**Impact:** HIGH — Installation failures, version conflicts, CI breakage

**Mitigation:**
- **Tier-based dependency isolation:** Optional dependencies via extras (`pip install .[spatial-ai]`)
- **Version pinning:** Pin all transitive dependencies in `requirements/spatial-ai.txt`
- **Fallback modes:** Heuristic fallbacks when models unavailable (CPU, no GPU)
- **CI testing:** Test installation on clean environment (Docker)

**Owner:** Specialist (dependency management), Architect (approval)

---

### Risk 2: GPU Memory Exhaustion

**Risk:** Phase 2 models are large (SAM2 Large: 2.4GB, NVDIFFREC: 1.5GB, 3DGS: 800MB). Multi-model workflows exceed VRAM.

**Impact:** MEDIUM — OOM crashes, user frustration

**Mitigation:**
- **Model unloading:** Explicit model unload between stages (`del model; torch.cuda.empty_cache()`)
- **Batching limits:** Enforce batch size limits based on available VRAM
- **CPU fallback:** Automatic CPU fallback when OOM detected
- **Documentation:** Clear VRAM requirements in README (≥12GB recommended)

**Owner:** Specialist (implementation), Architect (review)

---

### Risk 3: HF Revision Verification Overhead

**Risk:** Manually verifying HuggingFace commit SHAs for 3+ models is tedious and error-prone.

**Impact:** LOW — Slows promotion, potential mistakes

**Mitigation:**
- **Automated verification script:** `scripts/validation/verify_hf_model.py` (fetch commits, validate)
- **Documentation:** Step-by-step guide in `docs/apex/HUGGINGFACE_MODEL_PINNING.md`
- **Experimental tier:** Allow development without verification, promote when ready

**Owner:** Specialist (tooling), Architect (policy)

---

### Risk 4: ADR-023 Isolation Violations

**Risk:** Developers accidentally import from `lux_depth_v3` during Phase 2 development.

**Impact:** MEDIUM — CI failures, governance friction, rework

**Mitigation:**
- **Pre-commit hook:** Run isolation checker on every commit (fast fail)
- **Clear documentation:** Import policy in `docs/architecture/PHASE2_IMPLEMENTATION_PLAN.md` (this doc)
- **IDE hints:** Add `.editorconfig` / IDE settings to warn on forbidden imports
- **Architect review:** Review all PRs for isolation compliance

**Owner:** Architect (enforcement), Specialist (compliance)

---

### Risk 5: 3DGS License Ambiguity

**Risk:** 3DGS Inria license restricts commercial use. Unclear if "research tier" is sufficient.

**Impact:** HIGH — Legal risk, potential license violation

**Mitigation:**
- **Clear tier restriction:** 3DGS only in `tier: apex_research` or `tier: apex_research_ultra`
- **License check enforcement:** CLI fails if 3DGS used in commercial tier
- **Documentation:** License restrictions prominently documented in README
- **Alternative:** Offer NeuS2 (MIT) as commercial-friendly alternative

**Owner:** Architect (policy), Legal (if available)

---

### Risk 6: Performance Regression

**Risk:** Phase 2 adds computational overhead (segmentation, material estimation, reconstruction). Users expect Phase 1 speed.

**Impact:** MEDIUM — User complaints, velocity loss

**Mitigation:**
- **Opt-in stages:** Orchestration pipeline allows selective stages (`stages=["ingest", "depth"]` only)
- **Caching:** Cache intermediate results (depth, segmentation) to avoid recomputation
- **Profiling:** Profile each stage, optimize hot paths
- **Documentation:** Document performance expectations per-stage

**Owner:** Specialist (optimization), Architect (approval)

---

## 8. Compatibility Checklist Integration

Phase 2 must satisfy the compatibility checklist from Phase 1.1. Here's how:

### ✅ Make Phase 2 Scripts OpenEXR-Aware

**Implementation:**
```python
# In spatial_ai/orchestration/spatial_pipeline.py
def check_exr_support():
    """Verify OpenEXR/Imath available before setting emit_exr=True."""
    try:
        import OpenEXR
        import Imath
        return True
    except ImportError:
        return False

# Usage in pipeline
if config.emit_exr and not check_exr_support():
    raise RuntimeError(
        "EXR output requested but OpenEXR/Imath not installed.\n"
        "Install with: pip install OpenEXR Imath\n"
        "Or set emit_exr=False in preset."
    )
```

**Benefit:** Prevents "crash after 20 seconds" failures. Fail-fast with clear message.

---

### ✅ Treat strict_ingest as Lane Policy

**Implementation:**
```yaml
# config/presets/experimental/spatial_ai_dev.yaml (development lane)
ingest:
  strict_ingest: false  # Allow 8-bit for quick iteration

# config/presets/experimental/spatial_ai_research.yaml (research lane)
ingest:
  strict_ingest: true   # Enforce 16-bit+ precision
```

**CLI:**
```bash
# Development mode (fast)
transformation_portal spatial-ai process --preset spatial_ai_dev input.jpg

# Research mode (strict)
transformation_portal spatial-ai process --preset spatial_ai_research input.tiff
```

**Benefit:** Preserves velocity in dev, maintains rigor in research.

---

### ✅ Put SAM2/MaterialGAN/3DGS Code in Correct Namespace

**Implementation:**
- All Phase 2 code under `src/transformation_portal/spatial_ai/`
- Shared utilities in `src/transformation_portal/core/geometry/`
- Zero imports from `lux_depth_v3` (enforced by CI)

**Enforcement:**
```bash
# CI runs on every commit
python scripts/security/verify_pipeline_isolation.py
# Expected: PASS
```

**Benefit:** No governance friction, CI validates compliance.

---

### ✅ Keep Unverified Model Refs in experimental/

**Implementation:**
- All Phase 2 presets start in `config/presets/experimental/`
- Placeholders allowed: `revision: "NEEDS_VERIFICATION_0000000000000000000000"`
- CI skips HF validation for experimental/ (policy: `scripts/validation/validate_hf_revisions.py`)

**Promotion Workflow:**
1. Develop with placeholder
2. Verify HF commit SHA manually
3. Update preset with real SHA
4. Move to `config/presets/` (stable)
5. CI validates on merge

**Benefit:** No CI churn during development. Clear promotion path.

---

### ✅ Update Phase 2 Docs to Reflect New Invariants

**Documentation Updates:**

1. **`docs/spatial_ai/PHASE2_OVERVIEW.md`** — Architecture, capabilities, constraints
2. **`docs/spatial_ai/SAM2_INTEGRATION.md`** — SAM2 API, examples
3. **`docs/spatial_ai/MATERIAL_GAN.md`** — MaterialGAN API, PBR export
4. **`docs/spatial_ai/3DGS_INTEGRATION.md`** — 3DGS API, mesh export
5. **`docs/spatial_ai/PIPELINE.md`** — Orchestration, lane policies
6. **`README.md`** — Update with Phase 2 capabilities

**Key Sections:**
- I/O requirements (gamma=1.0, strict_ingest modes)
- EXR output (fail-loud behavior, preflight checks)
- Model loading (HF revision pinning policy)
- Pipeline isolation (ADR-023 compliance)

**Benefit:** New contributors understand constraints from day one.

---

## 9. Estimated Scope & Effort

### 9.1 Effort Breakdown

| Phase | Duration | Owner | Complexity |
|-------|----------|-------|------------|
| **Phase 2.1: SAM2** | 5 days | Specialist | Medium |
| **Phase 2.2: MaterialGAN** | 5 days | Specialist | High |
| **Phase 2.3: 3DGS** | 5 days | Specialist | High |
| **Phase 2.4: Orchestration** | 5 days | Specialist | Medium |
| **Phase 2.5: Hardening** | 3 days | Architect + Specialist | Low |
| **Total** | **23 days** (~5 weeks) | | |

**Critical Path:**
- Phase 2.1 → Phase 2.2 → Phase 2.4 (MaterialGAN depends on SAM2)
- Phase 2.3 can be parallelized (3DGS independent of SAM2)

**Acceleration Opportunity:**
- If Specialist + Architect work in parallel, compress to 4 weeks

---

### 9.2 Lines of Code Estimate

| Component | Estimated LOC | Tests LOC | Docs (pages) |
|-----------|---------------|-----------|--------------|
| SAM2 Backend | 800 | 400 | 3 |
| Material Estimator | 1000 | 500 | 4 |
| 3DGS Reconstructor | 900 | 450 | 3 |
| Orchestration | 600 | 300 | 5 |
| Core Geometry Utils | 400 | 200 | 2 |
| **Total** | **3700** | **1850** | **17** |

**Note:** These are estimates. Actual may vary by ±30%.

---

### 9.3 Dependency Additions

**New Python Dependencies:**

| Package | Purpose | License | Tier |
|---------|---------|---------|------|
| `sam2` (from HF) | SAM2 model | Apache 2.0 | All |
| `nvdiffrast` | Differentiable rendering | BSD-3-Clause | Research+ |
| `pytorch3d` | 3D transforms | BSD-3-Clause | All |
| `trimesh` | Mesh processing | MIT | All |
| `open3d` | Mesh export | MIT | All |
| `3dgs` (Inria) | Gaussian splatting | Inria (research) | Research only |

**Installation:**
```bash
# Base dependencies (all tiers)
pip install transformation-portal

# Phase 2 dependencies (spatial AI)
pip install transformation-portal[spatial-ai]

# Research tier only (includes 3DGS)
pip install transformation-portal[spatial-ai-research]
```

---

## 10. Success Metrics

Phase 2 is successful when:

### Technical Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Test Coverage** | ≥85% | pytest --cov |
| **CI Green** | 100% | All checks passing |
| **Isolation Compliance** | 100% | AST checker passes |
| **HF Revision Compliance** | 100% (stable presets) | Validation script passes |
| **Performance Regression** | <5% vs Phase 1 | Benchmark suite |
| **Documentation Coverage** | 100% public APIs | Docstring audit |

### Functional Metrics

| Capability | Success Criteria |
|------------|------------------|
| **SAM2 Segmentation** | Segments 512x512 image in <3s on GPU |
| **MaterialGAN PBR** | Generates 1024x1024 PBR textures in <10s on GPU |
| **3DGS Reconstruction** | Reconstructs 3-view scene in <30s, RMSE <2% |
| **E2E Pipeline** | Processes full workflow in <60s (512x512, GPU) |
| **OpenEXR Preflight** | Fails gracefully with clear message if missing |
| **Lane-Based Strictness** | Dev mode fast (<5s), research mode strict (correct) |

### Governance Metrics

| Policy | Compliance Check |
|--------|------------------|
| **ADR-023 Isolation** | Zero forbidden imports (AST checker) |
| **HF Revision Pinning** | All stable presets have commit SHAs |
| **Experimental Presets** | Placeholders only in experimental/ |
| **Security Posture** | No path traversal, no unsafe deserialization |
| **Backward Compatibility** | Phase 1 tests still passing |

---

## 11. Next Steps

### Immediate Actions (This Week)

1. **Architect Approval:** Review and approve this plan (REQUIRED before implementation)
2. **Session File:** Save this plan to session state for continuity
3. **ADR Creation:** Create ADR-027 (Phase 2 Integration Architecture) if needed
4. **Dependency Audit:** Review proposed dependencies for license/security issues
5. **Risk Review:** Validate risk mitigation strategies

### Week 1 (Phase 2.1)

1. Specialist implements SAM2 integration
2. Create experimental preset (`experimental/sam2_video.yaml`)
3. Write tests (10 unit, 2 integration)
4. CI green, isolation compliant
5. Architect review (mid-week checkpoint)

### Week 2 (Phase 2.2)

1. Specialist implements MaterialGAN integration
2. Create experimental preset (`experimental/material_gan.yaml`)
3. Write tests (12 unit, 2 integration)
4. CI green, isolation compliant
5. Architect review (mid-week checkpoint)

### Week 3 (Phase 2.3)

1. Specialist implements 3DGS integration
2. Create experimental preset (`experimental/gaussian_splat.yaml`)
3. Write tests (10 unit, 2 integration)
4. CI green, isolation compliant
5. Architect review (mid-week checkpoint)

### Week 4 (Phase 2.4)

1. Specialist implements orchestration
2. Create unified preset (`experimental/spatial_ai_full.yaml`)
3. Write E2E tests (3 tests)
4. CLI integration
5. Architect review (end-of-week)

### Week 5 (Phase 2.5)

1. Architect security audit
2. Performance profiling
3. Documentation audit
4. Preset promotion (if HF revisions verified)
5. Create PR for Phase 2 merge

---

## 12. Open Questions

### Q1: NVDIFFREC vs MaterialGAN?

**Question:** Should we use NVDIFFREC (BSD-3-Clause, commercial OK) or MaterialGAN (CC BY-NC 4.0, research only)?

**Recommendation:** Start with NVDIFFREC (license flexibility). Add MaterialGAN as alternative later if quality gap significant.

**Owner:** Architect (license review), Specialist (quality assessment)

---

### Q2: 3DGS License Restrictions?

**Question:** Inria license restricts commercial use. Is `tier: apex_research` sufficient, or do we need explicit license acceptance?

**Recommendation:**
- Tier restriction: `tier: apex_research` or higher only
- CLI check: Fail if 3DGS used in commercial tier
- Documentation: Prominently document license restrictions
- Consider: NeuS2 (MIT) as commercial alternative

**Owner:** Architect (policy), Legal (if available)

---

### Q3: Multi-View Input for 3DGS?

**Question:** 3DGS requires multiple viewpoints. How do users provide multi-view inputs?

**Options:**
1. **Video mode:** Extract frames from video (assume camera motion)
2. **Image directory:** User provides N images from different viewpoints
3. **Synthetic views:** Generate views from single image + depth (less accurate)

**Recommendation:** Support all three, with video mode as primary (easiest for users).

**Owner:** Specialist (implementation)

---

### Q4: CLIP Integration for Material Classification?

**Question:** Should material classification (SAM2 mask labeling) use CLIP, or defer to Phase 3?

**Recommendation:** Add optional CLIP integration in Phase 2.1 (simple, high value). Make it opt-in dependency.

**Owner:** Specialist (implementation)

---

## 13. Conclusion

Phase 2 is architecturally **feasible, well-scoped, and aligned with repository constraints**.

### Key Success Factors

1. **Respect Phase 1.1 constraints** — Don't fight the gates, design for them
2. **Enforce isolation (ADR-023)** — AST checker prevents violations
3. **Lane-based policies** — Dev fast, research strict
4. **Experimental → stable promotion** — Clear path for model verification
5. **Comprehensive testing** — Unit + integration + E2E + compliance

### Risk Posture

- **Technical risks:** MEDIUM (dependency hell, GPU OOM) — Mitigated with fallbacks, documentation
- **Governance risks:** LOW (ADR-023, HF policy) — Enforced by CI
- **License risks:** MEDIUM (3DGS Inria license) — Mitigated with tier restrictions
- **Timeline risks:** LOW (well-scoped, 5 weeks) — Buffer available if needed

### Recommendation

**PROCEED with Phase 2 implementation** following this plan.

**Prerequisites:**
- ✅ Architect approval of this plan
- ✅ Dependency review complete
- ✅ Risk mitigation strategies validated

**Next Action:** Architect review and approval.

---

**Document Status:** Proposed
**Review Date:** 2026-02-11
**Approvers:** Transformation Portal Architect (required)
**Implementation Start:** Upon approval
**Target Completion:** 5 weeks from start (Phase 2.5 complete)

---

*This plan is a living document. Updates expected as implementation progresses and new constraints emerge.*
