# DA3 Geometry Evaluation - Quick Start

**Purpose**: Test if DA3 provides better global geometry than DA2 APEX on 750 Picacho scenes
**Time**: 90 minutes
**License**: Non-commercial R&D only

---

## The Question

Does DA3 produce **materially better global geometry** than your current tiled DA2 APEX pipeline on:
- Pool water plane (smooth, no ripples)
- Glass/reflections (windows, mirrors)
- Railings (thin structures, consistency)
- Sky/aerial (large-scale coherence)
- Long interior planes (walls, no depth wobble)

---

## Quick Start (3 Commands)

### 1. Install DA3 (if not already)
```bash
cd /Users/rc/Transformation_Portal
pip uninstall -y depth-anything-3 || true
mkdir -p external
git clone https://github.com/ByteDance-Seed/depth-anything-3 external/depth-anything-3
pip install -e external/depth-anything-3

# Verify
da3 --help
python -c "from depth_anything_3.api import DepthAnything3; print('✓ DA3 ready')"
```

### 2. Run Evaluation Script
```bash
cd /Users/rc/Transformation_Portal
./scripts/da3/run_da3_geometry_evaluation.sh
```

This will:
- Convert 6 TIFFs to 8-bit PNG (DA3 input format)
- Run **DA3NESTED-GIANT-LARGE-1.1** on all 6 scenes (quality ceiling)
- Run **DA3METRIC-LARGE** on Pool, Aerial, GreatRoom (Apache-licensed)
- Run **DA3MONO-LARGE** on Pool, Aerial, GreatRoom (Apache-licensed)
- Generate depth + confidence visualizations
- Create evaluation report template

### 3. Review Outputs
```bash
# View 3D GLB files (Mac Preview or online viewer)
open 750Picacho_DA3_RnD/outputs/DA3NESTED-GIANT-LARGE-1.1/750Picacho_Pool_16bit/output.glb

# View depth + confidence visualizations
open 750Picacho_DA3_RnD/comparisons/

# Fill out evaluation report
open 750Picacho_DA3_RnD/EVALUATION_REPORT.md
```

---

## What to Look For

### Pool Scene (Critical)
- **Water plane**: Is it smoother than APEX? (APEX has ripple artifacts)
- **Confidence map**: Does low-conf align with water surface?
- **Railings**: Are thin structures consistent?

### Aerial Scene
- **Sky handling**: Better segmentation than APEX?
- **Scale coherence**: Does metric depth feel plausible?

### Kitchen/GreatRoom
- **Detail vs Global trade-off**: Do you lose cabinetry edges (APEX strength)?
- **Wall planes**: More stable than APEX?
- **Glass**: Better reflection handling?

---

## Decision Criteria

**Keep DA3 if**:
- Pool water is **materially smoother**
- Confidence maps enable **better failure handling**
- At least one **Apache-licensed model** (METRIC or MONO) competes with NESTED

**Defer DA3 if**:
- Only NESTED wins (non-commercial blocker)
- Detail loss on cabinetry/railings
- No confidence benefit

---

## Integration (If DA3 Wins)

**Proposed Hybrid Architecture**:
```
Input Image
    ↓
DA3 (global anchor + confidence) ← New
    ↓
Tiled DA2 APEX (detail) ← Keep
    ↓
Confidence-weighted fusion ← New
    ↓
SegFormer MaterialsV2 ← Keep
```

**Code Location**: `lux_depth_v2/da3_integration.py` (new module)

---

## Files Created

```
750Picacho_DA3_RnD/
├── README_LICENSE_NOTICE.txt          # Non-commercial banner
├── inputs_png/                        # 8-bit sRGB PNGs (6 files)
├── outputs/
│   ├── DA3NESTED-GIANT-LARGE-1.1/    # Quality ceiling (all 6 scenes)
│   │   ├── 750Picacho_Pool_16bit/
│   │   │   ├── output.npz             # depth + conf arrays
│   │   │   └── output.glb             # 3D visualization
│   │   └── ...
│   ├── DA3METRIC-LARGE/               # Apache-2.0 (focus scenes)
│   └── DA3MONO-LARGE/                 # Apache-2.0 (focus scenes)
├── comparisons/                       # Depth + confidence visualizations
└── EVALUATION_REPORT.md               # Fill this out
```

---

## Processing Settings

**Resolution**: `--process-res 1024` (upper_bound_resize)
**Why**: Preserves detail while capping max resolution. DA3 at lower res will lose edge detail vs APEX.

**Export**: `mini_npz-glb`
**Why**: NPZ for arrays (depth/conf), GLB for visual QA

**Backend**: Yes (keeps model loaded for 6 images)
**Why**: 10-20x speedup vs reloading model each time

---

## Timeline

- **Install DA3**: 15 min (one-time)
- **Run script**: 30 min (6 scenes × 3 models)
- **Visual QA**: 15 min (GLB files + visualizations)
- **Report**: 15 min (fill out template)
- **Decision**: 15 min (keep or defer)

**Total**: 90 min

---

## Troubleshooting

**DA3 CLI not found**:
```bash
cd depth_anything_3_official
pip install -e .
```

**Backend fails to start**:
- Check GPU/MPS availability
- Try without backend (slower but works)
- Check logs: `/tmp/da3_backend_*.log`

**Out of memory**:
- Reduce `--process-res` from 1024 to 768 or 512
- Use CPU instead of GPU: `--device cpu`

**Conversion fails (ImageMagick)**:
```bash
brew install imagemagick  # macOS
```

---

## Production Boundaries (Non-Negotiable)

1. **All DA3 outputs stay in** `750Picacho_DA3_RnD/`
2. **Never mix DA3 artifacts** into production deliverables
3. **APEX-100 pipeline remains unchanged** during evaluation
4. **License compliance**: NESTED outputs marked non-commercial
5. **`.gitignore` excludes** R&D directory (already configured)

---

## After Evaluation

**If DA3 wins**:
1. Document integration architecture
2. Prototype confidence-weighted fusion
3. Test on new scene (not 750 Picacho)
4. Measure performance impact

**If APEX wins**:
1. Archive DA3 outputs for reference
2. Document findings in EVALUATION_REPORT.md
3. Revisit in Q1 2026 R&D sprint

---

**Ready? Run**: `./run_da3_geometry_evaluation.sh`
