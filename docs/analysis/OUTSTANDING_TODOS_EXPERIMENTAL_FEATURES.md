# Outstanding TODOs for Advanced Features - Transformation Portal

**Generated**: 2026-02-18
**Last Updated**: 2026-03-31
**Status**: Based on experimental preset analysis and codebase review

---

## 🎯 High Priority: Experimental Feature Integration

### 1. Multi-Model Depth Ensemble
**Status**: ✅ Implemented (PR #906 + follow-up)
**Required For**: apex_research_ultra preset
**Components Needed**:
- [x] Ensemble backend implementation
- [x] Variance-weighted fusion algorithm
- [x] Temporal consistency filter (video, opt-in) — EMA filter in DepthCrafterBackend
- [x] DepthCrafter integration (disabled in default ensemble config) — `src/transformation_portal/depth/backends/depthcrafter.py`
- [x] Inter-model variance threshold validation
- [x] `.values()` → `.items()` metric alignment bug fixed (PR #906)

**Remaining**:
- DepthCrafter model checkpoint not yet available (backend uses synthetic fallback)
- Enable `depthcrafter` by default once checkpoint + model inference path are production-ready
- Performance profiling needed (expected: 2.4x slower than single model)

**Files**:
- `src/transformation_portal/depth/backends/ensemble.py` (ensemble + fusion)
- `src/transformation_portal/depth/backends/depthcrafter.py` (temporal backend)
- `src/transformation_portal/depth/backends/registry.py` (registration)
- `config/presets/experimental/apex_research_ultra.yaml` (lines 61-100)

---

### 2. SAM2 Video Segmentation
**Status**: ✅ **BACKEND IMPLEMENTED** (Model sources verified 2026-03-17)
**Required For**: sam2_segmentation, apex_research_ultra presets
**Components Implemented**:
- [x] SAM2Backend implementation — `src/transformation_portal/spatial_ai/segmentation/sam2_backend.py`
- [x] Temporal propagation logic — video mode with `temporal_ids` tracking
- [x] SHA-256 checkpoint verification — built into backend
- [x] Package: `sam2` (requires pip install)

**Verified HuggingFace Model Sources**:
| Tier | Model | HuggingFace Repo |
|------|-------|------------------|
| CI/Smoke | Tiny | `MackinationsAi/segment-anything-model-v2-tiny-hf` |
| Dev Default | Small | `MackinationsAi/segment-anything-model-v2-small-hf` |
| Production | Base+ | `MackinationsAi/segment-anything-model-v2-base-plus-hf` |
| Quality Max | Large | `MackinationsAi/segment-anything-model-v2-large-hf` |
| ONNX Portable | ONNX | `vietanhdev/segment-anything-2-onnx-models` |

**Model Lock Status**: ✅ All 6 models pinned in `config/model_lock_manifest.yaml` (revisions verified 2026-03-27)

**Files**:
- `src/transformation_portal/spatial_ai/segmentation/sam2_backend.py` (backend)
- `config/presets/experimental/sam2_segmentation.yaml` (preset)
- `config/model_lock_manifest.yaml` (model registry)
- `docs/guides/SAM2_INTEGRATION_GUIDE.md` (documentation)

**Installation**:
```bash
# Install SAM2
pip install sam2
```

---

### 3. Physics-Based Material Estimation
**Status**: ⚠️ Contract + provenance emission implemented; neural backends not implemented
**Required For**: apex_research_ultra, material_pbr presets

**Done (contract-hardening)**:
- [x] `AvailabilityState` / `BackendDecision` resolution model
- [x] `backend_decision` threaded into `PBRGenerationMetadata`
- [x] Per-segment `diagnostics.json` and `provenance.json` sidecars under `output/materials/segment_*`
- [x] Deterministic payload hashing for emitted texture artifacts
- [x] Atomic JSON writes for materials diagnostics/provenance
- [x] Regression coverage for backend decision serialization and persistence
- [x] Experimental preset accepts unresolved `repo_id` / `revision` as `null`
- [x] Config-time validation plus `strict_backend` fail-fast behavior for the single-image materials pipeline
- [x] Explicit `preset_family: materials_pbr` marker added for top-level materials presets to reduce schema drift
- [x] Placeholder/governance scanning aligned between preset health and materials schema policy
- [x] Runtime artifact attestation and isolated-worker requirements kept as execution-phase constraints in emitted backend/runtime requirements
- [x] Materials preset loader enforces research-tier opt-in and blocks unattested non-heuristic backends unless explicitly allowed for dev/experimental review
- [x] Runtime Spatial AI preset/config loading now applies materials governance validation before backend selection
- [x] Manifest-backed materials attestation checks for approved `repo_id` / exact `revision` tuples and checkpoint hash tuples
- [x] Materials provenance sidecars record governance overrides (`allow_research_materials`, `allow_unattested_materials`) for auditability

**Current emitted artifact contract**:
- `albedo.npy` - RGB diffuse color
- `normal.npy` - Normal map
- `roughness.npy` - Surface roughness map
- `metallic.npy` - Metalness map (or specular proxy for dielectric)
- `ao.npy` - Ambient occlusion map
- `height.npy` - Optional displacement / height map
- `diagnostics.json` - Per-segment generation metadata and texture shape diagnostics
- `provenance.json` - Per-segment backend decision and artifact payload hashes

**Still pending (real execution)**:
- Real backend execution for NVDIFFREC, MaterialGAN, and PBRFusion
- Isolated-worker execution and full runtime artifact verification for executable backends once real integrations land
- Continued schema discipline around availability-state vocabulary and provenance consumers

#### MaterialGAN (Phase 2.2C - Optional Enrichment)
**Current behavior**:
- `material_gan` requests resolve to `heuristic` with `input_contract_mismatch`

**Still Pending**:
- [ ] MaterialGAN backend implementation
- [ ] Model checkpoint download
- [ ] Geometric normal fusion
- [ ] Package: `materialgan` (not installed)

**Position in Roadmap**: MaterialGAN should be treated as an **optional SVBRDF enrichment backend**,
not the primary material path. Best introduced after NVDIFFREC normalization exists.

**Notes**:
- `src/transformation_portal/spatial_ai/materials/material_backend.py` currently records the fallback in metadata/provenance instead of attempting execution
- Packaging and license verification still need explicit resolution before shipping any real integration

#### NVDIFFREC (Phase 2.2A - Primary Material Path)
**Current behavior**:
- `nvdiffrec` requests resolve to `heuristic` with `input_contract_mismatch`

**Still Pending**:
- [ ] NVDIFFREC backend implementation
- [ ] Optimization loop (200 iterations)
- [ ] Learning rate tuning
- [ ] Multi-view input contract and camera-pose ingestion path

**Recommended Integration Priority**: NVDIFFREC should be implemented **first** because:
1. Better alignment with geometry-first pipeline
2. Better fit with mesh extraction / SuGaR / Poisson downstreams
3. Easier to frame as deterministic optimization vs GAN synthesis

**Notes**:
- `src/transformation_portal/spatial_ai/materials/material_backend.py` now emits explicit fallback provenance rather than implying NVDIFFREC executed
- Real execution belongs behind a richer multiview capture bundle than the current single-image materials API exposes

#### PBRFusion (Phase 5B - Optional Runtime-Gated Backend)
**Current behavior**:
- `pbr_fusion` requests resolve to `heuristic`
- Availability is recorded as `runtime_missing` when `PBRFUSION_PATH` is absent
- Availability is recorded as `integration_missing` when a runtime path exists but integration is not implemented

**Still Pending**:
- [ ] Direct PyTorch or ComfyUI worker integration
- [ ] Runtime bootstrap / pinned model revision contract
- [ ] Optional GPU-backed integration test lane

**Files**:
- `config/presets/experimental/apex_research_ultra.yaml` (lines 122-138)
- `config/presets/experimental/material_pbr.yaml`
- `src/transformation_portal/spatial_ai/materials/material_backend.py`
- `src/transformation_portal/spatial_ai/orchestration/pipeline.py`
- `tests/spatial_ai/materials/test_material_backend.py`
- `tests/spatial_ai/orchestration/test_pipeline.py`

**Governance**: Both backends should be gated under:
- Explicit optional extras in pyproject.toml
- Isolated env/worker (no core import coupling)
- Separate CI leg (avoid contaminating deterministic baseline)

**Next steps**:
1. Keep `AvailabilityState` values as the canonical availability vocabulary across emitted provenance.
2. Start PBRFusion integration behind optional dependencies and pinned revision gates.
3. Introduce a dedicated multi-view materials entrypoint before attempting real NVDIFFREC or MaterialGAN execution.

---

### 4. 3D Gaussian Splatting Reconstruction
**Status**: ✅ Verification Framework Implemented | ⚠️ Canonical Source Unresolved
**Required For**: gaussian_splat_3d, apex_research_ultra presets

**Components Implemented**:
- [x] GaussianBackend implementation — `src/transformation_portal/spatial_ai/reconstruction/gaussian_backend.py`
- [x] Multi-view camera pose handling
- [x] Depth consistency verification — `GeometricValidator`
- [x] PLY/OBJ export — `MeshExporter`
- [x] Attestation schema — `config/model_lock_manifest.yaml` (artifact_attestation section)

**Verification Framework** (✅ Implemented):
```yaml
# config/model_lock_manifest.yaml - artifact_attestation section
gaussian_splatting:
  backend: inria_graphdeco
  source_type: direct_checkpoint  # git_release | direct_checkpoint | local_import
  source_url: "PENDING_CANONICAL_URL"
  source_commit_or_tag: "PENDING_VERIFICATION"
  artifacts:
    - filename: "gaussian_splatting_base.pt"
      sha256: "PENDING_VERIFICATION"
      filesize_bytes: null  # null = validation should fail
  verification:
    method: sha256+source_commit
    required: true
```

**Canonical Source Status** (⚠️ Unresolved):
- No official HuggingFace artifact from Inria/GraphDECO found
- This is a **source-artifact attestation problem**, not a normal HF model-lock problem
- **Attestation is incomplete until source artifact is identified and hashed**

**What Must Be Verified** (full revision tuple, not just checkpoint):
- Code revision (exact commit from graphdeco-inria/gaussian-splatting)
- Rasterizer backend revision
- Optimizer behavior
- Serialization format of splat params

**Configuration**:
- 7K-30K iterations
- Memory: 6GB VRAM target
- Quality gate: RMSE < 5%

**Files**:
- `config/presets/experimental/gaussian_splat_3d.yaml`
- `config/presets/experimental/apex_research_ultra.yaml` (lines 140-161)
- `config/model_lock_manifest.yaml` (artifact_attestation section)
- `src/transformation_portal/spatial_ai/reconstruction/gaussian_backend.py`

**Remaining Action Items**:
```bash
# Install gsplat
pip install gsplat

# Run verification (shows pending canonical source)
python scripts/validation/verify_3dgs_artifacts.py

# Verify Inria 3DGS repo (canonical source pending)
# https://github.com/graphdeco-inria/gaussian-splatting
```

---

### 5. LLaVA Vision-Language Quality Validation
**Status**: ✅ **FULLY IMPLEMENTED** (2026-03-28) + Model Revisions Pinned + APEX Integration
**Required For**: apex_research_ultra preset
**Phase**: 2.2B (Structured Visual QA)

**Verified HuggingFace Model Sources**:
| Tier | Model | Use Case | HuggingFace Repo |
|------|-------|----------|------------------|
| **Primary** | LLaVA v1.6 Mistral 7B | Quality validation (balanced) | `llava-hf/llava-v1.6-mistral-7b-hf` |
| **CI/Smoke** | LLaVA OneVision 0.5B | Pipeline contract testing | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` |
| **Legacy** | LLaVA 1.5 7B | Backward compatibility | `llava-hf/llava-1.5-7b-hf` |
| **Quality Max** | LLaVA 1.5 13B | Highest quality (existing) | `llava-hf/llava-1.5-13b-hf` |

**Components**:
- [x] LLaVA backend implementation — `src/transformation_portal/evals/vision_language/llava_backend.py`
- [x] Structured prompt templates — `src/transformation_portal/evals/vision_language/llava_prompts.py`
- [x] Schema-constrained scoring — `src/transformation_portal/evals/vision_language/llava_scoring.py`
- [x] Model revision pinning in manifest
- [x] Integration with APEX evaluation harness — `src/transformation_portal/evals/apex_llava_integration.py`

**APEX + LLaVA Integration** (2026-03-28):
- `ApexLlavaConfig` dataclass for configuration
- `create_apex_harness_with_llava()` factory function
- `create_production_harness()`, `create_ci_smoke_harness()`, `create_quality_max_harness()` convenience functions
- Quality dimensions: `segmentation`, `architectural`, `depth`, `material`
- `build_material_quality_prompt()` for PBR texture quality assessment
- Tests: `tests/evals/test_apex_llava_integration.py` (25 tests)

**IMPORTANT**: LLaVA should be a **quality-evaluation backend**, not a generative UX feature.
Use for structured observations on:
- Segmentation mask leakage
- Missing object regions
- Texture seams
- Geometric distortion
- Reflective-material failure
- Silhouette integrity

**Output Contract** (schema-constrained, not free-form chat):
```json
{
  "artifact_id": "string",
  "passes_basic_quality": true,
  "issues": [
    {"type": "mask_leakage", "severity": "high", "evidence": "..."}
  ],
  "summary_score": 0.0
}
```

**Best Practice**: Use **hybrid evaluation**:
1. Deterministic numeric metrics first
2. LLaVA structured critique second
3. Aggregate policy layer last

**Quality Dimensions**:
- Depth plausibility
- Material realism
- Enhancement quality
- Architectural correctness

**Min Score**: 7.5/10 (non-blocking warnings)

**Files**:
- `config/presets/experimental/apex_research_ultra.yaml` (lines 184-207)
- `config/model_lock_manifest.yaml` (LLaVA entries added)

---

## 📋 Medium Priority: Model Pinning & Verification

### 6. HuggingFace Model Revision Pinning
**Status**: ✅ **COMPLETED** (2026-03-14, updated 2026-03-27)
**Affected Presets**:
- apex_research.yaml
- apex_research_canary.yaml
- apex_research_ultra.yaml (DA3 1.1 fallback)

**Completed Items**:
- [x] Verified DA3 1.1 Nested Giant Large revision
- [x] Pinned to 40-char commit SHA: `b2359bdf726fb44ef62acca04d629dcf158053e7`
- [x] Model ID corrected from `DA3-NESTED-GIANT-LARGE-1.1` to `DA3NESTED-GIANT-LARGE-1.1`
- [x] Validation script passed
- [x] SAM2 revisions pinned (6 models) — 2026-03-27
- [x] LLaVA revisions pinned (4 models total, 3 newly pinned) — 2026-03-27

**Total Models Pinned**: DA3 + SAM2 (6 models) + LLaVA (4 models)

**Verification Command**:
```bash
python scripts/validation/validate_hf_revisions.py
# ✅ All HuggingFace model revisions are properly pinned
```

**Cross-Reference**: See [TODO_ACTION_PLAN.md](./TODO_ACTION_PLAN.md) "COMPLETED: HuggingFace Model Revision Pinning"

---

### 7. Checkpoint SHA256 Verification
**Status**: ⚠️ Placeholders Present
**Missing Checksums**:

| Model | Checkpoint | Current Value |
|-------|-----------|---------------|
| Depth Pro | `checkpoints/depth_pro.pt` | ✅ Valid |
| SAM vit_h | `checkpoints/sam_vit_h_4b8939.pth` | ⚠️ Placeholder |
| SAM2 Hiera L | `checkpoints/sam2_hiera_l.pt` | ⚠️ Placeholder |
| MaterialGAN v2 | `checkpoints/materialgan_v2.pth` | ⚠️ Placeholder |
| DepthCrafter | `checkpoints/depthcrafter_v1.pt` | ⚠️ Placeholder |

**Action**:
```bash
# Generate checksums for downloaded models
shasum -a 256 checkpoints/*.pt checkpoints/*.pth
# Update YAML configs with verified hashes
```

---

## 🔧 Low Priority: Infrastructure & Polish

### 8. Upscaling Backend (Real-ESRGAN)
**Status**: ❌ Unavailable (BasicSR CVE)
**Required For**: `--emit-upscaled16` deliverables

**Current State**:
- Real-ESRGAN PyPI package banned (unmaintained)
- Local backend at `src/transformation_portal/upscaling/backends/realesrgan.py`
- Currently unavailable due to BasicSR CVE

**Alternative**:
- Consider ESRGAN-based alternatives
- Evaluate HAT (Hybrid Attention Transformer)
- ONNX export for production deployment

**Files**:
- `src/transformation_portal/upscaling/backends/realesrgan.py:182`

---

### 9. Documentation Gaps
**Status**: ✅ **RESOLVED** (2026-03-17)

**Documentation Status**:
- [x] `docs/apex/ultra_workflow_quick_guide.md` — ✅ Exists (381 lines)
- [x] `docs/architecture/ADR-026-apex-research-ultra.md` — ✅ Exists (30KB comprehensive ADR)
- [x] SAM2 integration guide — ✅ `docs/guides/SAM2_INTEGRATION_GUIDE.md` (408 lines)
- [x] 3D Gaussian Splatting workflow — ✅ `docs/guides/GAUSSIAN_SPLATTING_GUIDE.md` (created 2026-03-17)
- [x] MaterialGAN/NVDIFFREC usage examples — ✅ `docs/guides/MATERIAL_PBR_GUIDE.md` (409 lines)

**Note**: The original reference to `ultra_workflow_guide.md` was incorrect; the actual file is
`ultra_workflow_quick_guide.md` which provides equivalent functionality.

---

### 10. Test Coverage for Experimental Features
**Status**: ✅ **RESOLVED** (2026-03-17)

**Implemented**:
- [x] Ensemble depth backend tests — `tests/depth/backends/test_ensemble.py` (15 tests)
- [x] DepthCrafter temporal backend tests — `tests/depth/backends/test_depthcrafter.py` (24 tests)
- [x] SAM2 temporal propagation tests — `tests/spatial_ai/segmentation/test_sam2_backend_integration.py` (2 temporal tests)
- [x] 3DGS convergence tests — `tests/spatial_ai/reconstruction/test_convergence_tracking.py` (skeleton with skipif markers)
- [x] MaterialGAN material classification tests — `tests/spatial_ai/materials/test_materialgan_integration.py` (skeleton with skipif markers)
- [x] LLaVA quality validation tests — `tests/validation/test_llava_quality_validation.py` (21 active tests, 10 with skipif markers for model loading)

**Pattern (updated to match actual implementation)**:
```python
@pytest.mark.ml
@pytest.mark.skipif(not HAS_SAM2, reason="SAM2 package not installed (optional dependency)")
def test_sam2_temporal_propagation():
    ...
```

**Note**: Tests for MaterialGAN, 3DGS convergence, and LLaVA are intentionally skipped until
the corresponding features are implemented. The test skeletons document expected behavior
and serve as integration test targets for Phase 2.2/5 completion.

---

## 📊 Priority Matrix (Updated 2026-03-28)

### Recommended Execution Sequence

**Phase A — Model-Lock Hardening (PREREQUISITE)**

Before backend implementation, pin exact HF revisions for all identified model sources:
- SAM2 models (MackinationsAi tiers + ONNX)
- LLaVA models (v1.6-mistral-7b, onevision, 1.5-7b)

Add manifest validation that fails on missing revision, floating refs, or missing file hashes.

**Phase B–E Implementation Order** (optimized for delivery risk):

| Order | Phase | Feature | Rationale |
|-------|-------|---------|-----------|
| ~~1st~~ | ~~**2.2B**~~ | ~~LLaVA Quality Validation~~ | ✅ COMPLETED (2026-03-28) |
| 2nd | **2.2D** | 3DGS Verification Plumbing | Infrastructure can land before canonical source is resolved |
| 3rd | **2.2A** | NVDIFFREC Backend | Higher strategic value, but more fragile operationally |
| 4th | **2.2C** | MaterialGAN Backend | Optional enrichment, only after normalized contract exists |

### Full Priority Matrix

| Feature | Complexity | Value | Dependencies | Priority | Status |
|---------|-----------|-------|--------------|----------|--------|
| **Model-Lock Hardening** | Low | Critical | Manual revision pinning | **CRITICAL** | ✅ COMPLETED (2026-03-27) |
| SAM2 Segmentation | Medium | High | pip install, sources identified | **HIGH** | ✅ Backend done, sources identified |
| LLaVA Validation | Medium | High | HF sources identified | **HIGH** | ✅ **FULLY IMPLEMENTED** (2026-03-28) |
| NVDIFFREC | High | High | Graphics deps, worker isolation | **HIGH** | ⏳ Phase C (next up) |
| 3DGS Verification | Medium | Medium | Framework ready, source blocked | **MEDIUM** | ✅ Verification script done, canonical source pending |
| MaterialGAN | High | Medium | After NVDIFFREC contract | **MEDIUM** | ⏳ Phase D (last) |
| Depth Ensemble | High | Medium | DepthCrafter | **MEDIUM** | ✅ EMA implemented |
| Real-ESRGAN Fix | Medium | Low | Alternative upscaler | **LOW** | ❌ Blocked (CVE) |

---

## 🚀 Quick Start: Enable SAM2 (Easiest Win)

SAM2 is the lowest-hanging fruit with highest impact:

```bash
# 1. Install package
pip install sam2  # Note: package is 'sam2', not 'segment-anything-2'

# 2. Pin HF revision (Phase A hardening)
# Visit: https://huggingface.co/MackinationsAi/segment-anything-model-v2-base-plus-hf
# Copy exact commit SHA and update config/model_lock_manifest.yaml

# 3. Update config preset (if using HF integration)
# Edit: config/presets/experimental/sam2_segmentation.yaml

# 4. Download checkpoint
python scripts/download_sam2_checkpoint.py

# 5. Test
python -m transformation_portal.spatial_ai segment \
  --preset experimental/sam2_segmentation \
  --input test_image.tiff \
  --output test_output/
```

---

## 📝 Notes

1. **Experimental presets are intentionally placeholders** - they document aspirational features
2. **Research-only restrictions apply** to most experimental features
3. **Performance targets are estimates** - need real-world validation
4. **Fallback chains are critical** - graceful degradation to stable backends

**TODO Statistics (Updated 2026-05-11):**
- **Source code TODOs (`src/`):** 0 (all cleaned up)
- **Scanner-visible Test TODO comments (`tests/`):** 0
- **Governed NotImplementedError items:** 24 (abstract methods, phase gates, platform limits)
- **Documentation TODO markers:** ~38 (experimental-feature-focused subset; see Note below)

> **Note on Documentation TODO counts:** This ~38 count represents markers within **experimental feature documentation** (this file and related guides). Scanner-enforced source/test/script/web markers are tracked by `TODO_ACTION_PLAN.md` and `todo_scanner_snapshot.json`; as of 2026-05-11 there are 0 ungoverned scanner-visible TODOs. Documentation annotations retained for context are not implementation action items by themselves.

**Experimental Feature Tracking:**
- **Blocked on external artifact source**: 1 (3DGS canonical Inria checkpoint)
- **Pending exact revision pinning**: 0 (all completed 2026-03-27)
- **Pending implementation**: 2 (NVDIFFREC, MaterialGAN backends)

---

## 📊 Status Update (2026-03-28)

**Current Status:**
- All source code TODOs in `src/` have been cleaned up
- 0 scanner-visible test TODO comments remain
- 24 NotImplementedError instances are intentional and governed (abstract methods, phase gates, platform limitations)
- P3 experimental features (ComfyUI, NVDIFFREC, MaterialGAN) remain deferred

**2026-03-28 Updates:**
- ✅ **LLaVA + APEX Integration COMPLETED**
  - `src/transformation_portal/evals/apex_llava_integration.py` - Factory functions and config
  - `ApexLlavaConfig` dataclass with quality-dimension-specific defaults
  - `create_apex_harness_with_llava()`, `create_production_harness()`, `create_ci_smoke_harness()` factories
  - `build_material_quality_prompt()` for PBR texture quality assessment
  - 25 new tests in `tests/evals/test_apex_llava_integration.py`
- ✅ Phase 2.2B (LLaVA Quality Validation) now FULLY IMPLEMENTED

**2026-03-27 Updates:**
- ✅ Phase A (Model-Lock Hardening) completed
- ✅ SAM2 revisions pinned (all 6 models verified)
- ✅ LLaVA revisions pinned (4 models total, 3 newly pinned)
- ✅ 3DGS verification script created (`scripts/validation/verify_3dgs_artifacts.py`)
- ✅ LLaVA backend implemented with tests enabled (21 active tests)

**Historical Progress (through 2026-03-17):**
- ✅ HF Revision Pinning for DA3 - Completed 2026-03-14
- ✅ Multi-Model Depth Ensemble - EMA temporal filter implemented
- ✅ Documentation Gaps - All guides now exist (2026-03-17)
- ✅ Test Coverage for Experimental Features - Test skeletons added (2026-03-17)
- ✅ SAM2 Model Source Identification - MackinationsAi HF repos + ONNX alternatives (2026-03-17)
- ✅ LLaVA Model Source Identification - llava-hf repos for quality validation (2026-03-17)
- ✅ 3DGS Verification Framework - Attestation schema defined (2026-03-17)

**IMPORTANT DISTINCTION:**
- **Model source identification ✅** = repository locators discovered
- **Model revision verification ✅** = exact commit SHAs pinned (2026-03-27)
- **APEX integration ✅** = harness factory functions implemented (2026-03-28)

**Model Sources Identified (All Pinned):**

| Feature | HuggingFace Repo | Source Status | Revision Status |
|---------|------------------|---------------|-----------------|
| SAM2 Tiny | `MackinationsAi/segment-anything-model-v2-tiny-hf` | ✅ Identified | ✅ Pinned (2026-03-27) |
| SAM2 Small | `MackinationsAi/segment-anything-model-v2-small-hf` | ✅ Identified | ✅ Pinned (2026-03-27) |
| SAM2 Base+ | `MackinationsAi/segment-anything-model-v2-base-plus-hf` | ✅ Identified | ✅ Pinned (2026-03-27) |
| SAM2 Large | `MackinationsAi/segment-anything-model-v2-large-hf` | ✅ Identified | ✅ Pinned (2026-03-27) |
| SAM2 ONNX | `vietanhdev/segment-anything-2-onnx-models` | ✅ Identified | ✅ Pinned (2026-03-27) |
| LLaVA v1.6 7B | `llava-hf/llava-v1.6-mistral-7b-hf` | ✅ Identified | ✅ Pinned (2026-03-27) |
| LLaVA 0.5B | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | ✅ Identified | ✅ Pinned (2026-03-27) |
| LLaVA 1.5 13B | `llava-hf/llava-1.5-13b-hf` | ✅ Pinned | ✅ Complete |

**3DGS Attestation Status:**
- ✅ Verification framework implemented (attestation schema in `model_lock_manifest.yaml`)
- ⚠️ Canonical source unresolved (no official Inria HF artifact found)
- ⚠️ Attestation incomplete until source artifact is identified and hashed

**Documentation Now Available:**
- `docs/guides/SAM2_INTEGRATION_GUIDE.md` - SAM2 backend integration + model sources
- `docs/guides/GAUSSIAN_SPLATTING_GUIDE.md` - 3DGS workflow guide
- `docs/guides/MATERIAL_PBR_GUIDE.md` - PBR material generation
- `docs/apex/ultra_workflow_quick_guide.md` - APEX Research Ultra quick start
- `docs/architecture/ADR-026-apex-research-ultra.md` - Full ADR

**Test Skeletons Added:**
- `tests/spatial_ai/reconstruction/test_convergence_tracking.py` - 3DGS convergence
- `tests/spatial_ai/materials/test_materialgan_integration.py` - MaterialGAN
- `tests/validation/test_llava_quality_validation.py` - LLaVA quality validation
- `tests/evals/test_apex_llava_integration.py` - APEX + LLaVA integration (25 tests, NEW)

**Recommended Execution Sequence:**
| Order | Phase | Feature | Status | Prerequisite |
|-------|-------|---------|--------|--------------|
| 0 | **A** | Model-Lock Hardening | ✅ Complete (2026-03-27) | None |
| 1 | **B** | LLaVA Backend | ✅ Complete (2026-03-27) | Phase A complete |
| 2 | **D** | 3DGS Verification Plumbing | ✅ Script done, source pending | Phase A complete |
| 3 | **C** | NVDIFFREC Backend | ⏳ Waiting | Phase B complete |
| 4 | **E** | MaterialGAN Backend | ⏳ Waiting | Phase C complete |

---

## 🔗 References

- ADR-025: APEX Research Workflow (stable) — `docs/architecture/ADR-025-apex-research-workflow.md`
- ADR-026: APEX Research Ultra (experimental) — `docs/architecture/ADR-026-apex-research-ultra.md`
- ADR-027: Spatial AI Contract Isolation — `docs/architecture/ADR-027-phase2-spatial-ai-extension.md`
- ADR-031: Test Dependency Isolation
- ADR-032: Dependency Pinning Strategy
- Model Lock Manifest — `config/model_lock_manifest.yaml`
