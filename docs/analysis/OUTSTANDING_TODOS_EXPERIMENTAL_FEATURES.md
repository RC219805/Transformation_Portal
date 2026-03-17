# Outstanding TODOs for Advanced Features - Transformation Portal

**Generated**: 2026-02-18
**Last Updated**: 2026-03-16
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
**Status**: ⚠️ Placeholder + Missing Dependencies
**Required For**: sam2_segmentation, apex_research_ultra presets
**Components Needed**:
- [ ] SAM2 Hiera Large integration
- [ ] Temporal propagation logic
- [ ] HuggingFace revision verification
- [ ] Package: `segment_anything` (not installed)

**Missing Checkpoints**:
- `checkpoints/sam2_hiera_l.pt` (PLACEHOLDER)
- Revision: `NEEDS_VERIFICATION_SAM2_LARGE_20260211`

**Files**:
- `config/presets/experimental/sam2_segmentation.yaml`
- `config/presets/experimental/apex_research_ultra.yaml` (lines 102-120)

**Action Items**:
```bash
# Step 1: Verify HF model revision
# Visit: https://huggingface.co/facebook/sam2-hiera-large/commits/main
# Replace placeholder with 40-char commit SHA

# Step 2: Install SAM2
pip install segment-anything-2

# Step 3: Download checkpoint
python scripts/download_sam2_checkpoint.py
```

---

### 3. Physics-Based Material Estimation
**Status**: ❌ Not Implemented
**Required For**: apex_research_ultra, material_pbr presets
**Components Needed**:

#### MaterialGAN
- [ ] MaterialGAN v2 integration
- [ ] Model checkpoint download
- [ ] Geometric normal fusion
- [ ] Package: `materialgan` (not installed)

**Missing**:
- `checkpoints/materialgan_v2.pth` (PLACEHOLDER)
- `src/transformation_portal/spatial_ai/materials/material_backend.py:238` - NotImplementedError

#### NVDIFFREC
- [ ] NVDIFFREC integration
- [ ] Optimization loop (200 iterations)
- [ ] Learning rate tuning
- [ ] Fallback to heuristic PBR

**Missing**:
- `src/transformation_portal/spatial_ai/materials/material_backend.py:202` - NotImplementedError

**Files**:
- `config/presets/experimental/apex_research_ultra.yaml` (lines 122-138)
- `config/presets/experimental/material_pbr.yaml`
- `src/transformation_portal/spatial_ai/materials/material_backend.py`

---

### 4. 3D Gaussian Splatting Reconstruction
**Status**: ⚠️ Placeholder + Missing Dependencies
**Required For**: gaussian_splat_3d, apex_research_ultra presets
**Components Needed**:
- [ ] gsplat backend integration
- [ ] Multi-view camera pose estimation
- [ ] Depth consistency verification
- [ ] PLY/OBJ export
- [ ] Package: `gsplat` (not installed)

**Missing Checkpoints**:
- HF repo: `graphdeco-inria/gaussian-splatting`
- Revision: `NEEDS_VERIFICATION_0000000000000000000000`

**Configuration**:
- 7K-30K iterations
- Memory: 6GB VRAM target
- Quality gate: RMSE < 5%

**Files**:
- `config/presets/experimental/gaussian_splat_3d.yaml`
- `config/presets/experimental/apex_research_ultra.yaml` (lines 140-161)

**Action Items**:
```bash
# Install gsplat
pip install gsplat

# Verify Inria 3DGS repo
# https://github.com/graphdeco-inria/gaussian-splatting
```

---

### 5. LLaVA Vision-Language Quality Validation
**Status**: ⚠️ Placeholder
**Required For**: apex_research_ultra preset
**Components Needed**:
- [ ] LLaVA 1.6 34B integration
- [ ] Multi-turn quality assessment
- [ ] Quality dimension scoring
- [ ] Fallback to LLaVA 1.5 13B

**Model**:
- HF: `liuhaotian/llava-v1.6-34b`
- License: Apache 2.0 (commercial OK)

**Quality Dimensions**:
- Depth plausibility
- Material realism
- Enhancement quality
- Architectural correctness

**Min Score**: 7.5/10 (non-blocking warnings)

**Files**:
- `config/presets/experimental/apex_research_ultra.yaml` (lines 184-207)

---

## 📋 Medium Priority: Model Pinning & Verification

### 6. HuggingFace Model Revision Pinning
**Status**: ✅ **COMPLETED** (2026-03-14)
**Affected Presets**:
- apex_research.yaml
- apex_research_canary.yaml
- apex_research_ultra.yaml (DA3 1.1 fallback)

**Completed Items**:
- [x] Verified DA3 1.1 Nested Giant Large revision
- [x] Pinned to 40-char commit SHA: `b2359bdf726fb44ef62acca04d629dcf158053e7`
- [x] Model ID corrected from `DA3-NESTED-GIANT-LARGE-1.1` to `DA3NESTED-GIANT-LARGE-1.1`
- [x] Validation script passed

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
- [x] LLaVA quality validation tests — `tests/validation/test_llava_quality_validation.py` (skeleton with skipif markers)

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

## 📊 Priority Matrix

| Feature | Complexity | Value | Dependencies | Priority | Status |
|---------|-----------|-------|--------------|----------|--------|
| SAM2 Segmentation | Medium | High | pip install, checkpoint | **HIGH** | Pending |
| HF Revision Pinning | Low | High | Manual verification | ~~**HIGH**~~ | ✅ Completed |
| MaterialGAN/NVDIFFREC | High | Medium | Research models | **MEDIUM** | Pending |
| 3D Gaussian Splatting | Very High | Medium | gsplat, multi-view | **MEDIUM** | Pending |
| Depth Ensemble | High | Medium | DepthCrafter | **MEDIUM** | Partial (EMA only) |
| LLaVA Validation | Medium | Low | 34B model (large) | **LOW** | Pending |
| Real-ESRGAN Fix | Medium | Low | Alternative upscaler | **LOW** | Blocked (CVE) |

---

## 🚀 Quick Start: Enable SAM2 (Easiest Win)

SAM2 is the lowest-hanging fruit with highest impact:

```bash
# 1. Install package
pip install segment-anything-2

# 2. Verify HuggingFace revision
# Visit: https://huggingface.co/facebook/sam2-hiera-large
# Copy latest stable commit SHA

# 3. Update config
# Edit: config/presets/experimental/sam2_segmentation.yaml
# Replace: NEEDS_VERIFICATION_SAM2_LARGE_20260211
# With: <40-char-commit-sha>

# 4. Download checkpoint (if needed)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/sam2-hiera-large')"

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

**Total TODOs in codebase**: 254
**High-priority items**: 7
**Blocked on external dependencies**: 4

---

## 📊 Status Update (2026-03-17)

**Completed since document creation:**
- ✅ HF Revision Pinning (HuggingFace Model Revision Pinning) - Completed 2026-03-14
- ✅ Multi-Model Depth Ensemble - EMA temporal filter implemented
- ✅ Documentation Gaps - All guides now exist (2026-03-17)
- ✅ Test Coverage for Experimental Features - Test skeletons added (2026-03-17)

**Documentation Now Available:**
- `docs/guides/SAM2_INTEGRATION_GUIDE.md` - SAM2 backend integration
- `docs/guides/GAUSSIAN_SPLATTING_GUIDE.md` - 3DGS workflow guide
- `docs/guides/MATERIAL_PBR_GUIDE.md` - PBR material generation
- `docs/apex/ultra_workflow_quick_guide.md` - APEX Research Ultra quick start
- `docs/architecture/ADR-026-apex-research-ultra.md` - Full ADR

**Test Skeletons Added:**
- `tests/spatial_ai/reconstruction/test_convergence_tracking.py` - 3DGS convergence
- `tests/spatial_ai/materials/test_materialgan_integration.py` - MaterialGAN
- `tests/validation/test_llava_quality_validation.py` - LLaVA quality validation

**Still In Progress (Blocked on External Dependencies):**
- SAM2 Segmentation - Awaiting checkpoint verification
- 3D Gaussian Splatting - Model revision verification pending
- MaterialGAN/NVDIFFREC - Model integration pending
- LLaVA Quality Validation - Integration pending

---

## 🔗 References

- ADR-025: APEX Research Workflow (stable) — `docs/architecture/ADR-025-apex-research-workflow.md`
- ADR-026: APEX Research Ultra (experimental) — `docs/architecture/ADR-026-apex-research-ultra.md`
- ADR-027: Spatial AI Contract Isolation — `docs/architecture/ADR-027-phase2-spatial-ai-extension.md`
- ADR-031: Test Dependency Isolation
- ADR-032: Dependency Pinning Strategy
