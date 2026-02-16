# Outstanding TODOs for Advanced Features - Transformation Portal

**Generated**: 2026-02-18
**Status**: Based on experimental preset analysis and codebase review

---

## 🎯 High Priority: Experimental Feature Integration

### 1. Multi-Model Depth Ensemble
**Status**: ⚠️ Placeholder
**Required For**: apex_research_ultra preset
**Components Needed**:
- [ ] Ensemble backend implementation
- [ ] Variance-weighted fusion algorithm
- [ ] Temporal consistency filter (video)
- [ ] DepthCrafter integration
- [ ] Inter-model variance threshold validation

**Blockers**:
- DepthCrafter checkpoint not available
- Ensemble fusion logic not implemented
- Performance profiling needed (expected: 2.4x slower than single model)

**Files**:
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
**Status**: ⚠️ Needs Verification
**Affected Presets**:
- apex_research.yaml
- apex_research_canary.yaml
- apex_research_ultra.yaml (DA3 1.1 fallback)

**TODO Items**:
- [ ] Verify DA3 1.1 Nested Giant Large revision
- [ ] Pin to 40-char commit SHA
- [ ] Update `expected_sha256` if needed
- [ ] Run validation script

**Current State**:
```yaml
# apex_research.yaml line 155
# TODO(Phase 1.1): Pin revision to commit hash for reproducibility
# Omitted until manual verification complete (will default to 'main' in loader)
```

**Action**:
```bash
# Manual verification required
# Visit: https://huggingface.co/depth-anything/DA3-NESTED-GIANT-LARGE-1.1/commits/main
# Copy commit SHA from verified release
# Update all 3 presets
python scripts/validation/validate_hf_revisions.py
```

**Files**:
- `config/presets/apex_research.yaml:155`
- `config/presets/apex_research_canary.yaml:26`
- `config/presets/experimental/apex_research_ultra.yaml:80`

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
**Status**: ⚠️ Incomplete

**Missing Docs**:
- [ ] `docs/apex/ultra_workflow_guide.md` (referenced but not created)
- [ ] `docs/architecture/ADR-026-apex-research-ultra.md` (referenced but not created)
- [ ] SAM2 integration guide
- [ ] 3D Gaussian Splatting workflow
- [ ] MaterialGAN/NVDIFFREC usage examples

---

### 10. Test Coverage for Experimental Features
**Status**: ❌ Not Implemented

**Needed**:
- [ ] Ensemble depth backend tests
- [ ] SAM2 temporal propagation tests
- [ ] 3DGS convergence tests
- [ ] MaterialGAN material classification tests
- [ ] LLaVA quality validation tests

**Pattern**:
```python
@pytest.mark.experimental
@pytest.mark.skipif(not HAS_SAM2, reason="SAM2 not installed")
def test_sam2_temporal_propagation():
    ...
```

---

## 📊 Priority Matrix

| Feature | Complexity | Value | Dependencies | Priority |
|---------|-----------|-------|--------------|----------|
| SAM2 Segmentation | Medium | High | pip install, checkpoint | **HIGH** |
| HF Revision Pinning | Low | High | Manual verification | **HIGH** |
| MaterialGAN/NVDIFFREC | High | Medium | Research models | **MEDIUM** |
| 3D Gaussian Splatting | Very High | Medium | gsplat, multi-view | **MEDIUM** |
| Depth Ensemble | High | Medium | DepthCrafter | **MEDIUM** |
| LLaVA Validation | Medium | Low | 34B model (large) | **LOW** |
| Real-ESRGAN Fix | Medium | Low | Alternative upscaler | **LOW** |

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

## 🔗 References

- ADR-025: APEX Research Workflow (stable)
- ADR-026: APEX Research Ultra (experimental, not yet created)
- ADR-027: Spatial AI Contract Isolation
- ADR-031: Test Dependency Isolation
- ADR-032: Dependency Pinning Strategy
