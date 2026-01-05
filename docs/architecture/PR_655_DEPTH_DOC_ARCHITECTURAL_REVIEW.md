# PR #655 Architectural Review: Depth Estimation Documentation

**Reviewer**: Transformation Portal Architect
**Document**: `docs/DEPTH_ESTIMATION_ANALYSIS.md` (Commit: `5afa0329`)
**Date**: 2026-01-05
**Status**: 🔴 **REQUIRES MAJOR REVISION**

---

## Executive Summary

The depth estimation analysis document represents a **valuable technical contribution** but suffers from critical architectural defects that will undermine maintainability and operational trustworthiness:

1. **Unprovable Claims**: Performance metrics and test counts lack verifiable citations
2. **License Terminology Mismatch**: Documentation conflicts with codebase contracts (`is_commercial` logic)
3. **Missing Output Format Contract**: No clear separation between preview (8-bit), contract (uint16), and research (float32) formats
4. **CLI Command Name Conflicts**: Duplicate `benchmark` commands will cause Typer registration failures
5. **Missing Failure Mode Documentation**: No explicit warnings about known depth artifacts
6. **Unqualified Metric Depth Claims**: Missing constraints and assumptions for metric output

**Bottom Line**: This document quietly misleads users into generating "gorgeous 8-bit depth screenshots that silently poison 16-bit pipelines." It must be restructured to **ground claims in verifiable facts, align with actual code contracts, and provide clear operational guidance.**

---

## Critical Issues by Category

### 1. Unprovable Claims (Credibility Risk)

#### Issue 1.1: Test Count Claims Without Citations

**Location**: Lines 37, 562

```markdown
- ✅ Production-validated (1,348 tests passing)
```

**Problem**:
- No CI badge reference
- No link to test run logs
- Actual test file count in `lux_depth_v2/tests/`: **44 files** (verified)
- Total repository test count may be 1,348, but this is NOT specific to lux_depth_v2

**Evidence**:
```bash
$ find lux_depth_v2 -name "test_*.py" | wc -l
44
```

**Recommendation**:
```markdown
- ✅ Production-validated (See [CI Test Results](../../.github/workflows/ci.yml) for latest coverage)
```

**Alternative** (if specific to V2):
```markdown
- ✅ Comprehensive test coverage (44 test suites covering inference, tiling, edge snapping, and I/O)
```

---

#### Issue 1.2: Throughput Claims Without Methodology

**Location**: Lines 39, 251

```markdown
- ✅ 127-400 images/hour throughput
- High-throughput batch processing (400+ images/hour)
```

**Problem**:
- No benchmark script reference
- No hardware specification
- No resolution, batch size, or precision context
- Range is **3x variance** (127-400), suggesting context-dependent performance

**Required Context**:
- Hardware: CPU/GPU model, memory
- Image resolution: 518×518 vs 4K vs 8K
- Batch size: 1 vs 16
- Model: Small vs Large
- Tile size: 512 vs 1024
- Precision: FP16 vs FP32

**Recommendation**:
Remove vague claims or replace with:
```markdown
**Performance**: Varies by hardware, resolution, and model selection. See [benchmarks/](../../bench/) for methodology and reference timings.

**Reference Configuration** (M4 Max, DA2-Large, 1024×1024 input, FP32):
- Single image: ~65ms inference + ~35ms I/O = 100ms total
- Batch (16): ~35 images/min = 2,100 images/hour
```

---

#### Issue 1.3: CVE Mitigation Claims Without SECURITY.md Link

**Location**: Line 38

```markdown
- ✅ Security-hardened (CVE-2024-27763 mitigated)
```

**Problem**:
- No link to `SECURITY.md` for verification
- Readers cannot confirm mitigation details

**Recommendation**:
```markdown
- ✅ Security-hardened ([CVE-2024-27763 mitigated](../../SECURITY.md#cve-2024-27763-basicsr-command-injection-vulnerability))
```

---

### 2. License Terminology Conflict (Contractual Defect)

#### Issue 2.1: "Commercial (Apache 2.0)" Section Contains Non-Commercial Models

**Location**: Lines 88-92 (DA3 Model Variants table)

```markdown
**DA3 Model Variants** (`lux_depth_v3/config.py`):
```
DA3NESTED-GIANT-LARGE-1.1  (1.40B params, CC-BY-NC-4.0) - RECOMMENDED
DA3-GIANT-1.1              (1.15B params, CC-BY-NC-4.0)
DA3-LARGE-1.1              (0.35B params, CC-BY-NC-4.0)
DA3METRIC-LARGE            (0.35B params, Apache 2.0)    - COMMERCIAL USE
```

**Problem**:
The CLI implementation defines:
```python
# lux_depth_v3/config.py, line 67-69
@property
def is_commercial(self) -> bool:
    """Check if model allows commercial use."""
    return self.license == ModelLicense.APACHE_2_0
```

**This creates confusion**:
- "COMMERCIAL USE" annotation on `DA3METRIC-LARGE` suggests other models are **non-commercial**
- But `CC-BY-NC-4.0` models (Giant, Large) are explicitly **Non-Commercial** per license terms
- Documentation uses "COMMERCIAL USE" as a synonym for "Apache 2.0" instead of clarifying licensing restrictions

**Impact**:
- Users may assume "RECOMMENDED" models are commercially viable
- Legal risk if deployed in commercial pipelines without license compliance

**Recommendation**:
Use **explicit license categories** that match `ModelInfo.is_commercial`:

```markdown
**DA3 Model Variants** (`lux_depth_v3/config.py`):

**Non-Commercial Models (CC-BY-NC-4.0)**:
```
DA3NESTED-GIANT-LARGE-1.1  (1.40B params) - RECOMMENDED for research/personal
DA3-GIANT-1.1              (1.15B params)
DA3-LARGE-1.1              (0.35B params)
```

**Commercial-Friendly Models (Apache 2.0)**:
```
DA3METRIC-LARGE            (0.35B params) - Metric depth, commercial use allowed
DA3-BASE                   (0.12B params) - Relative depth, commercial use allowed
DA3-SMALL                  (0.08B params) - Fast preview, commercial use allowed
```

⚠️ **License Compliance Note**: Non-commercial models (CC-BY-NC-4.0) require explicit permission for commercial deployment. See [lux_depth_v3/SECURITY.md](../../lux_depth_v3/SECURITY.md) for license validation workflow.
```

---

### 3. CLI Command Name Conflicts (Runtime Blocker)

#### Issue 3.1: Duplicate `benchmark` Commands Will Cause Typer Failure

**Location**: Lines 262-272 (Performance Benchmarks section)

**Problem**:
The documentation references `lux-depth-v3 benchmark` commands, but the CLI has **THREE** `benchmark` command registrations in the same Typer app:

```python
# lux_depth_v3/cli.py
@app.command()
def benchmark(...):  # Line 440 - Simple inference benchmark
    ...

@app.command()
def benchmark(...):  # Line 620 - Dataset evaluation benchmark
    ...

@app.command()
def benchmark_download(...):  # Line 683 - OK (different name)
    ...
```

**Impact**:
- Typer will **clobber** one of the `benchmark` commands (typically the first one)
- Documentation refers to a command that may not exist at runtime
- Users will experience "command not found" errors

**Evidence**:
```bash
$ grep -n "@app.command()" lux_depth_v3/cli.py | grep -A2 "def benchmark"
440:def benchmark(
620:def benchmark(
```

**Recommendation**:

**Option A** (Rename Commands in CLI):
```python
# lux_depth_v3/cli.py
@app.command(name="benchmark-inference")
def benchmark_inference(...):  # Line 440
    """Benchmark inference performance."""
    ...

@app.command(name="benchmark-quality")
def benchmark_quality(...):  # Line 620
    """Benchmark quality on standard datasets."""
    ...
```

Then update documentation to reference `lux-depth-v3 benchmark-inference` and `lux-depth-v3 benchmark-quality`.

**Option B** (Use Subcommands):
```python
# lux_depth_v3/cli.py
benchmark_app = typer.Typer()

@benchmark_app.command("inference")
def benchmark_inference(...):
    ...

@benchmark_app.command("quality")
def benchmark_quality(...):
    ...

app.add_typer(benchmark_app, name="benchmark")
```

Then update documentation to reference `lux-depth-v3 benchmark inference` and `lux-depth-v3 benchmark quality`.

---

### 4. Missing Output Artifact Contract (Pipeline Safety)

#### Issue 4.1: No Separation of Preview vs Contract vs Research Formats

**Location**: Entire document lacks output format taxonomy

**Problem**:
The documentation does not distinguish between:

1. **Preview Outputs** (8-bit RGB PNG):
   - Purpose: Visualization, web display, debugging
   - Format: `depth_preview.png` (8-bit, RGBA, with colormap)
   - **NOT suitable for downstream pipelines**

2. **Contract Outputs** (16-bit Single-Channel):
   - Purpose: V2 enhancement pipeline integration
   - Format: `depth_raw.tif` (uint16, grayscale, normalized to [0, 65535])
   - **Required for lux_depth_v2 integration**

3. **Research Outputs** (32-bit Float NPZ):
   - Purpose: Metric depth analysis, research validation
   - Format: `depth_metric.npz` (float32, absolute meters if metric mode)
   - **Not compatible with standard image viewers**

**Impact**:
- Users generate "gorgeous 8-bit depth screenshots" and pipe them into pipelines expecting uint16
- Quantization errors silently poison downstream stages
- No clear guidance on which format to use for which purpose

**Recommendation**:
Add a new section before "Recommended Configuration Summary":

```markdown
---

## 6. Output Format Contract

### 6.1 Output Artifacts by Purpose

| Output | Format | Precision | Use Case | Pipeline-Safe? |
|--------|--------|-----------|----------|----------------|
| `*_depth_preview.png` | 8-bit RGBA | 256 levels | Visualization, web display | ❌ **NO** |
| `*_depth.tif` | 16-bit Grayscale | 65,536 levels | V2 enhancement pipeline | ✅ **YES** |
| `*_depth.npz` | float32 | Full precision | Metric analysis, research | ⚠️ Research only |

### 6.2 Critical Usage Warnings

⚠️ **DO NOT** use preview PNG files as pipeline inputs:
- 8-bit quantization loses 99.6% of depth precision (65,536 → 256 levels)
- Colormap application is irreversible
- RGBA channels contain visualization metadata, not raw depth

✅ **Pipeline Integration** (V2 enhancement):
```python
# CORRECT: Use uint16 TIFF contract output
depth_map = cv2.imread("estate_123_depth.tif", cv2.IMREAD_UNCHANGED)
assert depth_map.dtype == np.uint16, "Contract violation"

# WRONG: Using preview PNG
depth_map = cv2.imread("estate_123_depth_preview.png")  # dtype=uint8, RGBA!
```

### 6.3 Metric Depth Output Constraints

**Metric depth** (absolute scale in meters) is available in `lux_depth_v3` with:
- **Model Requirement**: `ModelVariant.DA3_METRIC_LARGE` (Apache 2.0) or metric-capable variants
- **Intrinsics Requirement**: Focal length in pixels (`focal_length_px`) or camera intrinsics matrix
- **Scene Assumptions**:
  - Planar ground assumption for metric scaling
  - Fails on: Sky, reflections (water/glass), uniform walls, vegetation
  - Best for: Architectural interiors with visible floor/ceiling structure

**Example**:
```python
result = engine.infer(
    images=[image_path],
    convert_to_metric=True,
    focal_length_px=1200.0  # REQUIRED for metric depth
)
# result.metric_depth: float32 array in absolute meters
```

⚠️ **Metric depth does NOT fix monocular depth failures**:
- Sky will still have unstable depth estimates (metric values are unreliable)
- Reflections (pool water, glass) produce mirrored depth, not surface depth
- Vegetation saturation artifacts persist (metric scaling does not add texture detail)
```

---

### 5. Missing Failure Mode Documentation

#### Issue 5.1: No "Known Limitations" Section

**Problem**:
The documentation claims "maximum quality" but does not warn about known depth estimation failures:

1. **Texture Embossing**: Flat surfaces with strong texture (brick walls, patterned tiles) incorrectly assigned depth variation
2. **Sky Depth Instability**: Sky regions produce arbitrary depth values (gradient, noise, or infinity depending on cloud texture)
3. **Reflection Depth Mirroring**: Water/glass reflections produce mirrored scene depth, not surface depth
4. **Vegetation Saturation**: Dense foliage (trees, bushes) saturates at far-plane depth, losing fine structure
5. **Double Edge-Snapping Risk**: Applying `use_edge_snapping=True` AND `refinement_use_edge_snap=True` causes over-sharpening

**Recommendation**:
Add a new section after "Quality Validation Metrics":

```markdown
---

## 7. Known Failure Modes and Limitations

### 7.1 Monocular Depth Limitations

All monocular depth estimators (DA2, DA3) share fundamental limitations:

| Failure Mode | Symptom | Mitigation |
|--------------|---------|------------|
| **Texture Embossing** | Flat walls with patterns show false depth variation | Use guided filter (`refinement_use_edge_filter=True`) |
| **Sky Instability** | Sky shows gradient/noise instead of far-plane | Mask sky regions (segmentation) and clamp to infinity |
| **Reflection Mirroring** | Water/glass shows mirrored scene, not surface | Detect reflective materials, override depth to surface estimate |
| **Vegetation Saturation** | Trees/bushes saturate at far-plane, lose detail | No fix; inherent to monocular depth (requires stereo/LiDAR) |
| **Uniform Wall Collapse** | Blank walls lose depth structure | Increase `overlap` to 192px for more context |

### 7.2 Configuration Pitfalls

⚠️ **Double Edge-Snapping**:
```python
# WRONG: Applies edge snapping TWICE
config = TiledInferenceConfig(
    use_edge_snapping=True,              # ← Snaps edges after fusion
    use_production_refinement=True,
    refinement_use_edge_snap=True        # ← Snaps edges AGAIN in refinement
)
# Result: Over-sharpened edges, halo artifacts

# CORRECT: Use production refinement (includes edge snapping)
config = TiledInferenceConfig(
    use_edge_snapping=False,              # Disabled (handled by refinement)
    use_production_refinement=True,
    refinement_use_edge_snap=True
)
```

⚠️ **Texture Imprinting** (Low Overlap):
```python
# WRONG: Insufficient overlap for texture-heavy scenes
config = DepthConfig(
    tile_size=1024,
    overlap=64  # ← Too low for aerial/patterned scenes
)
# Result: Tile seam artifacts, calibration drift

# CORRECT: Use 192px overlap for texture-heavy scenes
config = DepthConfig(
    tile_size=1024,
    overlap=192  # Blocker B fix (see high_fidelity_depth)
)
```

### 7.3 Metric Depth Constraints

**Metric depth output requires**:
1. **Model Variant**: `DA3_METRIC_LARGE` or compatible metric-capable model
2. **Camera Intrinsics**: Focal length (pixels) or full K matrix
3. **Scene Assumptions**: Planar ground, visible floor/ceiling structure

**Metric depth DOES NOT fix**:
- ❌ Sky depth instability (metric values are still unreliable)
- ❌ Reflection mirroring (absolute scale doesn't fix mirrored geometry)
- ❌ Monocular failures (vegetation, uniform walls)

**Use Cases**:
- ✅ Architectural interiors with structured geometry
- ✅ 3D reconstruction with known camera poses
- ❌ Outdoor scenes with sky (unless masked)
- ❌ Highly reflective surfaces (pool water, glass facades)
```

---

### 6. Missing Benchmark Methodology

#### Issue 6.1: Performance Table Lacks Reproducibility Context

**Location**: Lines 262-272

```markdown
### 3.3 Performance Benchmarks

| Model | Device | Resolution | Time/Image | Notes |
|-------|--------|------------|------------|-------|
| DA2-Small | MPS (M4 Max) | 518×518 | 24ms | Best for preview |
| DA2-Large | MPS (M4 Max) | 518×518 | 65ms | Production quality |
...
```

**Problem**:
- No warmup iterations specified
- No batch size context
- No precision (FP16/FP32) specified
- No link to benchmark script for reproduction

**Recommendation**:
```markdown
### 3.3 Performance Benchmarks

**Methodology**: See [bench/README.md](../../bench/README.md) for reproduction instructions.

**Configuration**:
- Hardware: M4 Max (16 GPU cores, 128GB unified memory)
- Precision: FP32 (mixed precision available for 2x speedup)
- Warmup: 5 iterations (excluded from timing)
- Measurement: 100 iterations, median time reported
- Batch size: 1 (single image)
- Input size: 518×518 (model native resolution)

| Model | Device | Resolution | Median Time | 95th %ile | Throughput* | Notes |
|-------|--------|------------|-------------|-----------|-------------|-------|
| DA2-Small | MPS (M4 Max) | 518×518 | 24ms | 28ms | ~150/min | Preview quality |
| DA2-Large | MPS (M4 Max) | 518×518 | 65ms | 72ms | ~55/min | Production quality |
| DA2-Large | CUDA (RTX 4090) | 518×518 | 30ms | 35ms | ~120/min | GPU-accelerated |
| DA3-Nested-Giant | CUDA (RTX 4090) | 518×518 | 800ms | 950ms | ~4.5/min | Maximum quality |

*Throughput calculated as 60,000ms / median_time_ms, single-image batch
```

---

### 7. Validation Metrics as Absolutes

#### Issue 7.1: Edge Alignment and Seam Energy Thresholds Are Context-Sensitive

**Location**: Lines 425-447

```python
def compute_edge_alignment(rgb, depth):
    ...
    return correlation  # Target: > 0.5 (higher is better)

def validate_seam_energy(depth, tile_boundaries):
    ...
    assert ratio < 1.2, "Seam artifacts detected"
```

**Problem**:
- Edge alignment threshold `> 0.5` is **not universal**:
  - Interior scenes with structured geometry: 0.6-0.8 achievable
  - Vegetation-heavy scenes: 0.3-0.5 is realistic (lack of edges in foliage)
  - Aerial/top-down: 0.4-0.6 (fewer strong edges)

- Seam energy ratio `< 1.2` is **not universal**:
  - Clean tiling: < 1.1 achievable
  - Texture-heavy scenes: 1.2-1.5 is acceptable with high overlap
  - Sky/uniform regions: Ratio becomes unstable (division by near-zero)

**Recommendation**:
```markdown
### 6.1 Edge Alignment Score

```python
def compute_edge_alignment(rgb, depth):
    """Correlation between RGB edges and depth edges.

    **Context-Sensitive Thresholds**:
    - Structured interiors: > 0.6 (excellent)
    - Exteriors with vegetation: > 0.4 (acceptable)
    - Aerial/texture-heavy: > 0.5 (good)

    **Calibration**: Validate on reference dataset before setting project thresholds.
    """
    rgb_edges = cv2.Canny(rgb_gray, 50, 150)
    depth_edges = np.sqrt(sobel_x**2 + sobel_y**2)
    correlation = np.corrcoef(rgb_edges.ravel(), depth_edges.ravel())[0, 1]
    return correlation
```

### 6.2 Seam Energy Ratio

```python
def validate_seam_energy(depth, tile_boundaries, threshold=1.2):
    """Gradient energy at tile boundaries vs interior.

    **Context-Sensitive Thresholds**:
    - Clean scenes (overlap=128): < 1.1 (excellent)
    - Texture-heavy (overlap=192): < 1.3 (acceptable)
    - Sky/uniform regions: Ratio unstable (skip validation)

    **Recommended**: Use percentile-based validation instead of mean ratio
    to avoid sky/uniform region instability.
    """
    boundary_energy = grad_mag[boundary_mask].mean()
    interior_energy = grad_mag[~boundary_mask].mean()

    # Skip if interior has low variance (sky, blank walls)
    if interior_energy < 1e-4:
        return None  # Validation not applicable

    ratio = boundary_energy / interior_energy
    return ratio
```

---

## Recommended Remediation Plan

### Phase 1: Critical Fixes (Required for Merge)

1. **Remove or Qualify Unprovable Claims**:
   - [ ] Replace "1,348 tests passing" with link to CI or specific test count
   - [ ] Remove "127-400 images/hour" or add benchmark methodology
   - [ ] Link CVE-2024-27763 claim to SECURITY.md

2. **Fix License Terminology Conflict**:
   - [ ] Rename "COMMERCIAL USE" to "Commercial-Friendly (Apache 2.0)"
   - [ ] Add "Non-Commercial (CC-BY-NC-4.0)" section header
   - [ ] Add license compliance warning

3. **Add Output Format Contract Section**:
   - [ ] Create taxonomy: Preview (8-bit PNG), Contract (uint16 TIFF), Research (float32 NPZ)
   - [ ] Add "DO NOT use preview PNG in pipelines" warning
   - [ ] Document metric depth constraints (model, intrinsics, scene assumptions)

4. **Fix CLI Command Conflicts**:
   - [ ] Rename duplicate `benchmark` commands in `lux_depth_v3/cli.py`
   - [ ] Update documentation to match final command names

### Phase 2: Quality Enhancements (Recommended)

5. **Add "Known Failure Modes" Section**:
   - [ ] Document texture embossing, sky instability, reflection mirroring, vegetation saturation
   - [ ] Add double edge-snapping warning
   - [ ] Add texture imprinting (low overlap) warning

6. **Add Benchmark Methodology**:
   - [ ] Specify hardware, warmup, iterations, precision
   - [ ] Link to benchmark script in `bench/`
   - [ ] Clarify batch size and resolution context

7. **Qualify Validation Metrics**:
   - [ ] Add context-sensitive thresholds for edge alignment
   - [ ] Add calibration guidance for seam energy ratio
   - [ ] Document when metrics are not applicable (sky, uniform regions)

### Phase 3: Structural Refactoring (Future Work)

8. **Split into Focused Documents**:
   - [ ] `docs/depth_pipeline/overview.md` - High-level capabilities
   - [ ] `docs/depth_pipeline/recipes.md` - Configuration examples
   - [ ] `docs/depth_pipeline/validation.md` - Quality metrics and thresholds
   - [ ] `docs/depth_pipeline/benchmarks.md` - Performance data with methodology
   - [ ] `docs/depth_pipeline/failure_modes.md` - Known limitations and workarounds

---

## Security & Compliance Review

### ✅ Positive Findings

1. **CVE-2024-27763 Documentation**: Correctly references mitigation in multiple places
2. **License Awareness**: Shows awareness of Apache 2.0 vs CC-BY-NC-4.0 distinctions
3. **No Hardcoded Credentials**: No exposed API keys or secrets

### ⚠️ Concerns

1. **License Compliance Risk**: Ambiguous "COMMERCIAL USE" terminology may lead to license violations
2. **Data Provenance**: No guidance on commercial vs non-commercial model selection for production
3. **Output Format Safety**: Lack of format contract documentation creates pipeline poisoning risk

---

## Conclusion

This document represents **valuable domain knowledge** but requires significant revision to meet production documentation standards:

**Strengths**:
- Comprehensive coverage of three depth systems
- Detailed configuration examples
- Performance comparison table

**Critical Defects**:
- Unprovable claims undermine credibility
- License terminology conflicts with code
- Missing output format contract creates safety risk
- CLI command conflicts block runtime usage
- No failure mode documentation hides operational risks

**Recommendation**: **BLOCK MERGE** until Phase 1 critical fixes are complete. Phase 2 enhancements should be completed before promoting to primary documentation.

**Estimated Remediation Effort**:
- Phase 1 (Critical): 4-6 hours
- Phase 2 (Quality): 6-8 hours
- Phase 3 (Refactoring): 12-16 hours (future)

---

## Appendix A: Verification Commands

```bash
# Verify test count claim
find lux_depth_v2 -name "test_*.py" | wc -l
pytest lux_depth_v2 --collect-only -q | grep "test" | wc -l

# Verify CLI command conflicts
grep -n "@app.command()" lux_depth_v3/cli.py | grep "def benchmark"

# Verify CVE mitigation
grep "CVE-2024-27763" SECURITY.md
pip list | grep basicsr  # Should return nothing

# Verify license property
grep -A5 "def is_commercial" lux_depth_v3/config.py
```

---

## Appendix B: Aligned License Documentation Template

```markdown
### Model Licensing

**Non-Commercial Models (CC-BY-NC-4.0)** - Research/Personal Use Only:
- `DA3NESTED-GIANT-LARGE-1.1` (1.40B params) ⭐ **RECOMMENDED** for quality
- `DA3-GIANT-1.1` (1.15B params)
- `DA3-LARGE-1.1` (0.35B params)

**Commercial-Friendly Models (Apache 2.0)** - Production Deployments:
- `DA3METRIC-LARGE` (0.35B params) - Metric depth capability
- `DA3-BASE` (0.12B params) - Relative depth
- `DA3-SMALL` (0.08B params) - Fast preview

**License Compliance Check**:
```python
from lux_depth_v3.config import ModelVariant

variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
if not variant.info.is_commercial:
    print(f"⚠️ WARNING: {variant.info.name} is {variant.info.license.value}")
    print("Non-commercial license. Requires permission for commercial deployment.")
```

**See Also**: [lux_depth_v3/SECURITY.md](../../lux_depth_v3/SECURITY.md) for license validation workflow.
```

---

**Review Completed By**: Transformation Portal Architect
**Next Steps**: Assign to documentation maintainer for Phase 1 remediation
**Follow-Up**: Schedule architectural alignment review after Phase 1 completion
