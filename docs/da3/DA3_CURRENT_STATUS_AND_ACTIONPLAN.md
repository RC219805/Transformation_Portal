# DA3 Current Status & Action Plan
**Created**: 2025-12-31
**Purpose**: Production-minded assessment before 750 Picacho R&D evaluation

---

## Current Reality Check

### 1. Installation Status
```
✓ Package metadata: depth-anything-3==0.0.0 (editable install)
✗ Actual module: depth_anything_3 NOT IMPORTABLE
✗ Source location: /Users/rc/Transformation_Portal/depth_anything_3_official/ NOT FOUND
```

**Diagnosis**: Broken editable install. The package metadata exists but the actual source code is missing or incorrectly linked.

### 2. Wrapper Status
```
✓ lux_depth_v3/ wrapper exists with 3 modes (Python API, CLI, placeholder)
✗ Python API mode: Cannot import depth_anything_3.api
✗ CLI mode: Unknown if `da3` command exists in PATH
✓ Placeholder mode: Will run but produces garbage outputs
```

**Diagnosis**: Wrapper is architecturally sound but **cannot access real DA3 models**.

### 3. Production Pipeline Status
```
✅ lux-depth-v2 APEX-100: PRODUCTION READY (SegFormer MaterialsV2)
✅ 750 Picacho APEX processing: Already executed successfully
✅ Repository: Clean, up-to-date, CI green
```

**Diagnosis**: Production pipeline is untouched and operational. DA3 evaluation is **optional R&D only**.

---

## The Honest Question

**Do you actually need DA3 evaluation right now?**

### Arguments AGAINST spending time on DA3:
1. **Production is already solved**: lux-depth-v2 APEX-100 works
2. **DA3 installation is broken**: Will require full reinstall/debug
3. **Non-commercial license**: DA3 NESTED-GIANT can never ship to clients
4. **Wrapper needs 7 critical fixes**: Even after install, wrapper won't work correctly
5. **ROI is low**: This is comparison benchmarking, not production capability

### Arguments FOR DA3 evaluation:
1. **Technical curiosity**: Want to know if DA3 outperforms APEX on architectural scenes
2. **Future roadmap**: Evaluating next-gen depth models for 2026 pipeline
3. **Research documentation**: Comparing APEX vs DA3 for technical reports

---

## Decision Point: Two Paths Forward

### Path A: Skip DA3 Evaluation (RECOMMENDED)

**Timeline**: Immediate
**Effort**: Zero
**Risk**: Zero

**Rationale**:
- Production pipeline (APEX-100) is validated and operational
- DA3 requires significant installation + wrapper fixes
- Non-commercial license means DA3 can **never** replace APEX in production
- 750 Picacho processing is **already complete** with APEX

**Action**:
1. Document "DA3 evaluation deferred to future R&D sprint"
2. Archive `lux_depth_v3/` as "experimental, requires DA3 installation"
3. Focus on production deliverables (750 Picacho client outputs)

---

### Path B: Fix DA3 and Run Evaluation (R&D Investment)

**Timeline**: 2-4 hours
**Effort**: High
**Risk**: Medium (time investment, may still fail)

**Required Steps**:

#### Step 1: Install DA3 Correctly (30-60 min)
```bash
# Remove broken editable install
pip uninstall depth-anything-3 -y

# Clone official DA3 repo
cd /Users/rc/Transformation_Portal
git clone https://github.com/DepthAnything/Depth-Anything-V3.git depth_anything_3_official

# Install with extras
cd depth_anything_3_official
pip install -e ".[all]"

# Verify installation
python -c "from depth_anything_3.api import DepthAnything3; print('✓ DA3 installed')"
```

#### Step 2: Apply Critical Wrapper Fixes (60-90 min)
Apply all 7 fixes from `DA3_WRAPPER_CRITICAL_FIXES.md`:
1. Fix model loading: `from_pretrained()` pattern
2. Fix inference signature: positional `images` argument
3. Fix CLI commands: positional paths
4. Fix subprocess deadlock: use DEVNULL or async drain
5. Fix backend health check: remove `/status` endpoint assumption
6. Rename placeholder class: `DepthAnything3Placeholder`
7. Add license enforcement: hard error on commercial use of NC models

#### Step 3: Prepare 750 Picacho for DA3 (30 min)
```bash
# Create R&D-only directory
mkdir -p 750Picacho_DA3_RND_ONLY/{staging_16bit_png,da3_output,comparison_crops}

# License notice
cat > 750Picacho_DA3_RND_ONLY/README_LICENSE_NOTICE.txt << 'EOF'
NON-COMMERCIAL R&D ONLY — DO NOT SHIP TO CLIENTS

This directory contains Depth Anything 3 (DA3) evaluation outputs.
DA3 NESTED-GIANT-LARGE is licensed CC-BY-NC-4.0 (non-commercial).

Purpose: Internal comparison of DA3 vs lux-depth-v2 APEX-100.
Production pipeline: lux-depth-v2 APEX-100 (commercial-friendly).
EOF

# Convert TIFFs to 16-bit PNG (DA3 compatible)
for tiff in 750Picacho_Source_TIFFs/*.tif*; do
    base=$(basename "$tiff" | sed 's/\.[^.]*$//')
    magick "$tiff" -colorspace RGB -depth 16 \
        "750Picacho_DA3_RND_ONLY/staging_16bit_png/${base}.png"
done
```

#### Step 4: Run DA3 Inference (15 min)
```bash
# Option A: Use fixed Python API wrapper
python -c "
from lux_depth_v3.da3_wrapper import DepthAnything3Wrapper
from pathlib import Path

wrapper = DepthAnything3Wrapper(
    model_name='da3nested-giant-large',
    device='mps',  # Apple Silicon
    commercial_use=False,
    validate_license_strict=True
)

images = list(Path('750Picacho_DA3_RND_ONLY/staging_16bit_png').glob('*.png'))
for img_path in images:
    pred = wrapper.inference(
        images=[str(img_path)],
        export_dir='750Picacho_DA3_RND_ONLY/da3_output',
        export_format='mini_npz'
    )
    print(f'Processed: {img_path.name} -> depth shape {pred.depth.shape}')
"

# Option B: Use official DA3 CLI directly (if working)
da3 auto 750Picacho_DA3_RND_ONLY/staging_16bit_png \
    --export-dir 750Picacho_DA3_RND_ONLY/da3_output \
    --export-format mini_npz \
    --model-dir depth-anything/DA3NESTED-GIANT-LARGE
```

#### Step 5: Comparison Analysis (30 min)
Compare DA3 vs APEX-100 on architectural failure modes:
- Edge stability (rails, mullions, cabinetry)
- Plane smoothness (walls, water, countertops)
- Reflection handling (glass, mirrors, pool)
- Seam visibility (tiling artifacts)

Document findings in `750Picacho_DA3_RND_ONLY/evaluation_report.md`.

---

## Recommendation

**Choose Path A** (Skip DA3 evaluation) unless you have a **specific business need** for DA3 benchmarking.

### Why Path A?
1. **Production is solved**: APEX-100 works and is production-ready
2. **Time vs value**: 2-4 hours of R&D for non-shippable comparison
3. **Broken install**: Starting from broken state increases risk
4. **Non-commercial blocker**: DA3 can never replace APEX in production pipeline

### When to revisit DA3?
- **Q1 2026**: Next-gen depth model evaluation sprint
- **After production stabilization**: When APEX-100 is deployed and stable
- **If Apache-licensed DA3 models emerge**: `da3-base`, `da3metric-large` (already Apache-2.0)

---

## If You Choose Path B: Quality Gates

Before declaring "DA3 evaluation complete":

1. **Installation verification**:
   ```python
   from depth_anything_3.api import DepthAnything3
   model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
   ```

2. **Wrapper fixes validated**:
   - All 7 critical fixes applied
   - License enforcement tested (raises RuntimeError on commercial use)
   - Python API and CLI modes both functional

3. **Outputs validated**:
   - Depth maps have sensible range (0-1 or metric scale)
   - No NaN or Inf values
   - Visual quality check on at least 2 scenes (Kitchen + Pool)

4. **Comparison documented**:
   - Side-by-side depth maps (DA3 vs APEX-100)
   - Quantitative metrics (RMSE on known-good reference if available)
   - Architectural failure mode assessment

5. **Hard boundary maintained**:
   - All DA3 outputs in `750Picacho_DA3_RND_ONLY/`
   - `.gitignore` excludes R&D directory
   - No DA3 artifacts mixed into production pipeline

---

## Next Action

**Your decision**:
- [ ] **Path A**: Skip DA3 evaluation, focus on production deliverables
- [ ] **Path B**: Invest 2-4 hours to fix DA3 and run evaluation

**If Path A** → Document decision in repo:
```bash
cat > DA3_EVALUATION_DEFERRED.md << 'EOF'
# DA3 Evaluation Deferred

**Date**: 2025-12-31
**Decision**: Skip DA3 evaluation for 750 Picacho project

## Rationale
- Production pipeline (lux-depth-v2 APEX-100) is validated and operational
- DA3 installation is broken, requires full reinstall + wrapper fixes
- Non-commercial license (CC-BY-NC-4.0) prevents production use
- Time investment (2-4 hours) not justified for optional R&D

## Future Revisit
- Q1 2026: Next-gen depth model evaluation sprint
- Consider Apache-licensed DA3 variants (da3-base, da3metric-large)

## Production Status
✅ lux-depth-v2 APEX-100: Production-ready
✅ 750 Picacho: Successfully processed with APEX
EOF
```

**If Path B** → Start with Step 1 (DA3 installation) and verify each step before proceeding.

---

**End of Status & Action Plan**
