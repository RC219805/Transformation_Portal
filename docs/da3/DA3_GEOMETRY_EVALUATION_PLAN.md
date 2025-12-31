# DA3 Geometry Engine Evaluation Plan
**Created**: 2025-12-31
**Purpose**: Answer specific architectural geometry questions for 750 Picacho
**Timeline**: 90 minutes (focused R&D test)

---

## The Specific Question

**Does DA3 produce materially better global geometry than tiled DA2 APEX on:**
1. **Pool water plane** (smooth, no ripple artifacts)
2. **Glass/reflections** (windows, mirrors)
3. **Railings** (thin structures, global consistency)
4. **Sky/aerial** (large-scale coherence)
5. **Long interior planes** (walls, floors - no depth wobble)

**And if yes**: Which DA3 variant is worth keeping in R&D toolbox given licensing?

---

## Test Strategy

### Track A: Quality Ceiling (Non-Commercial)
- **Model**: `DA3NESTED-GIANT-LARGE-1.1` (CC-BY-NC-4.0)
- **Purpose**: Establish upper bound of what DA3 can achieve
- **Scenes**: All 6 (Kitchen, GreatRoom, Pool, Aerial, Primary Bath, Primary Bedroom)

### Track B: Deployable Candidates (Apache 2.0)
- **Model 1**: `DA3METRIC-LARGE` (metric scaling + global geometry)
- **Model 2**: `DA3MONO-LARGE` (best monocular relative depth)
- **Purpose**: Test if Apache-licensed models still outperform DA2 APEX
- **Scenes**: Focus on failure modes (Pool, Aerial, GreatRoom glass)

---

## Setup Steps (30 min)

### Step 1: Fix DA3 Installation
```bash
cd /Users/rc/Transformation_Portal

# Remove broken install
pip uninstall depth-anything-3 -y

# Clone official repo
git clone https://github.com/DepthAnything/Depth-Anything-V3.git depth_anything_3_official

# Install (with all extras for completeness)
cd depth_anything_3_official
pip install -e ".[all]"

# Verify CLI and Python API
da3 --help
python -c "from depth_anything_3.api import DepthAnything3; print('✓ DA3 ready')"
```

### Step 2: Prepare Input Images (8-bit PNG)
```bash
cd /Users/rc/Transformation_Portal

# Create R&D directory structure
mkdir -p 750Picacho_DA3_RnD/{inputs_png,outputs,comparisons}

# License notice
cat > 750Picacho_DA3_RnD/README_LICENSE_NOTICE.txt << 'EOF'
NON-COMMERCIAL R&D ONLY — DO NOT SHIP TO CLIENTS

DA3 NESTED-GIANT-LARGE: CC-BY-NC-4.0 (non-commercial)
DA3 METRIC/MONO: Apache-2.0 (commercial-friendly)

Purpose: Geometry quality comparison vs lux-depth-v2 APEX-100.
Production pipeline: lux-depth-v2 APEX-100 remains unchanged.
EOF

# Convert TIFFs to 8-bit sRGB PNG (DA3 expects standard RGB)
for tiff in 750Picacho_Source_TIFFs/*.tif*; do
    base=$(basename "$tiff" | sed 's/\.[^.]*$//')
    magick "$tiff" \
        -colorspace sRGB \
        -depth 8 \
        -auto-orient \
        -strip \
        "750Picacho_DA3_RnD/inputs_png/${base}.png"
    echo "Converted: ${base}.png"
done
```

### Step 3: Update .gitignore
```bash
# Add R&D outputs to gitignore
cat >> .gitignore << 'EOF'

# DA3 R&D outputs (non-commercial)
750Picacho_DA3_RnD/
depth_anything_3_official/
EOF

git add .gitignore
git commit -m "chore: exclude DA3 R&D outputs from version control"
```

---

## Execution (45 min)

### Option A: CLI with Backend (Recommended for 6+ images)
```bash
cd /Users/rc/Transformation_Portal

# Create output structure
OUT_BASE="750Picacho_DA3_RnD/outputs"
mkdir -p "$OUT_BASE"

# ============================================================
# Track A: DA3NESTED-GIANT-LARGE-1.1 (Quality Ceiling)
# ============================================================
MODEL="depth-anything/DA3NESTED-GIANT-LARGE-1.1"
OUT_DIR="$OUT_BASE/DA3NESTED-GIANT-LARGE-1.1"

echo "Starting DA3 backend for: $MODEL"
da3 backend --model-dir "$MODEL" --gallery-dir "$OUT_DIR" &
BACKEND_PID=$!
sleep 10  # Let backend initialize

# Process all 6 scenes
for img in 750Picacho_DA3_RnD/inputs_png/*.png; do
    base=$(basename "$img" .png)
    echo "Processing: $base with DA3NESTED-GIANT-LARGE-1.1"

    da3 auto "$img" \
        --export-dir "$OUT_DIR/$base" \
        --export-format mini_npz-glb \
        --use-backend \
        --process-res 1024 \
        --process-res-method upper_bound_resize

    echo "✓ Completed: $base"
done

# Stop backend
kill $BACKEND_PID

# ============================================================
# Track B: DA3METRIC-LARGE (Apache 2.0, Metric Scaling)
# ============================================================
MODEL="depth-anything/DA3METRIC-LARGE"
OUT_DIR="$OUT_BASE/DA3METRIC-LARGE"

# Focus on failure modes: Pool, Aerial, GreatRoom
FOCUS_SCENES=("750Picacho_Pool_16bit" "750Picacho_Aerial" "750Picacho_GreatRoom")

echo "Starting DA3 backend for: $MODEL"
da3 backend --model-dir "$MODEL" --gallery-dir "$OUT_DIR" &
BACKEND_PID=$!
sleep 10

for scene in "${FOCUS_SCENES[@]}"; do
    img="750Picacho_DA3_RnD/inputs_png/${scene}.png"
    if [ -f "$img" ]; then
        echo "Processing: $scene with DA3METRIC-LARGE"

        da3 auto "$img" \
            --export-dir "$OUT_DIR/$scene" \
            --export-format mini_npz-glb \
            --use-backend \
            --process-res 1024 \
            --process-res-method upper_bound_resize

        echo "✓ Completed: $scene"
    fi
done

kill $BACKEND_PID

# ============================================================
# Track B: DA3MONO-LARGE (Apache 2.0, Monocular Best)
# ============================================================
MODEL="depth-anything/DA3MONO-LARGE"
OUT_DIR="$OUT_BASE/DA3MONO-LARGE"

echo "Starting DA3 backend for: $MODEL"
da3 backend --model-dir "$MODEL" --gallery-dir "$OUT_DIR" &
BACKEND_PID=$!
sleep 10

for scene in "${FOCUS_SCENES[@]}"; do
    img="750Picacho_DA3_RnD/inputs_png/${scene}.png"
    if [ -f "$img" ]; then
        echo "Processing: $scene with DA3MONO-LARGE"

        da3 auto "$img" \
            --export-dir "$OUT_DIR/$scene" \
            --export-format mini_npz-glb \
            --use-backend \
            --process-res 1024 \
            --process-res-method upper_bound_resize

        echo "✓ Completed: $scene"
    fi
done

kill $BACKEND_PID

echo "All DA3 processing complete!"
```

### Option B: Python API (For Tight Integration Experiments)
```python
# da3_geometry_test.py
from depth_anything_3.api import DepthAnything3
from pathlib import Path
import numpy as np

# Load model
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")

# Process single image
input_dir = Path("750Picacho_DA3_RnD/inputs_png")
output_dir = Path("750Picacho_DA3_RnD/outputs/DA3NESTED-GIANT-LARGE-1.1")

for img_path in input_dir.glob("*.png"):
    scene_name = img_path.stem
    out_scene_dir = output_dir / scene_name
    out_scene_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    prediction = model.inference(
        [str(img_path)],
        export_dir=str(out_scene_dir),
        export_format="mini_npz-glb",
        process_res=1024,
        process_res_method="upper_bound_resize"
    )

    print(f"✓ {scene_name}: depth shape {prediction.depth.shape}")

    # Access depth and confidence for downstream processing
    depth = prediction.depth[0]  # (H, W)
    conf = prediction.conf[0] if prediction.conf is not None else None

    # Save as 16-bit TIFF for lux-depth comparison
    # (convert DA3 normalized depth to 16-bit range)
    depth_16bit = (depth * 65535).astype(np.uint16)
    # Save with tifffile or PIL
```

---

## Comparison Protocol (15 min)

### Critical Evaluation Points

For each scene, compare DA3 outputs vs existing APEX-100 depth maps:

#### 1. **Pool Scene** (Primary Failure Mode Test)
- [ ] Water plane smoothness (no ripple artifacts)
- [ ] Railing consistency (thin structure preservation)
- [ ] Coping edge stability (no false depth jumps)
- [ ] Sky/horizon handling

**Visual Check**:
- Load DA3 GLB in 3D viewer
- Compare depth map heatmaps side-by-side
- Look for seams, tiling artifacts

#### 2. **Aerial Scene** (Global Scale Coherence)
- [ ] Sky segmentation quality
- [ ] Terrain/roofline geometry
- [ ] Large-scale spatial consistency
- [ ] Metric depth plausibility (if using DA3METRIC)

#### 3. **Kitchen/GreatRoom** (Detail vs Global Trade-off)
- [ ] Cabinetry edge preservation (DA2 APEX strength)
- [ ] Long wall plane stability (DA3 potential strength)
- [ ] Glass/window handling (reflections)
- [ ] Overall "room depth feel"

#### 4. **Confidence Maps** (New Capability)
Extract confidence from NPZ files:
```python
import numpy as np

data = np.load("750Picacho_DA3_RnD/outputs/DA3NESTED-GIANT-LARGE-1.1/750Picacho_Pool_16bit/output.npz")
depth = data['depth']
conf = data['conf'] if 'conf' in data else None

if conf is not None:
    # Low confidence regions = water, glass, reflections
    # Use this to gate edge refinement
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.subplot(131); plt.imshow(depth, cmap='turbo'); plt.title('Depth')
    plt.subplot(132); plt.imshow(conf, cmap='viridis'); plt.title('Confidence')
    plt.subplot(133); plt.imshow(conf < 0.3, cmap='gray'); plt.title('Low Conf Mask')
    plt.savefig('750Picacho_DA3_RnD/comparisons/Pool_confidence_analysis.png')
```

---

## Decision Criteria

### DA3 Wins If:
1. **Pool water plane is materially smoother** than APEX (no ripples)
2. **Confidence maps enable better failure mode handling** (suppress over-refinement)
3. **Global geometry is more coherent** on Aerial/GreatRoom
4. **At least one Apache-licensed model** (METRIC or MONO) competes with NESTED

### DA3 Loses If:
1. **Detail loss on cabinetry/railings** vs tiled DA2 APEX
2. **Processing time >> APEX** with no quality gain
3. **Only NESTED wins** (non-commercial blocker)
4. **No confidence benefit** (low-conf regions don't align with known failure modes)

---

## Integration Pathways (If DA3 Wins)

### High-Impact Hybrid Architecture
```
Input Image
    ↓
DA3 (global anchor) → Global depth + confidence
    ↓
Tiled DA2 APEX (detail) → Local refined depth
    ↓
Confidence-weighted fusion:
    - High conf regions: blend DA3 global + DA2 detail
    - Low conf regions: prefer DA3, reduce edge snapping
    ↓
SegFormer MaterialsV2 → Final output
```

**Implementation**:
- DA3 as new "global pass" in lux-depth-v2 pipeline
- Confidence-gated edge refinement (suppress on water/glass)
- Preserve existing APEX tiling for detail

### Code Sketch
```python
# In lux_depth_v2/pipeline.py or new da3_integration.py

def create_da3_global_anchor(image_path, model="da3metric-large"):
    """Generate DA3 global depth + confidence."""
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(f"depth-anything/{model.upper()}")
    pred = model.inference([image_path], process_res=1024)

    return {
        'depth': pred.depth[0],
        'confidence': pred.conf[0] if pred.conf else None
    }

def fuse_da3_apex(da3_anchor, apex_tiled, conf_threshold=0.3):
    """Weighted fusion favoring DA3 global shape, APEX detail."""
    conf = da3_anchor['confidence']

    if conf is None:
        # No confidence, simple blend
        return 0.6 * da3_anchor['depth'] + 0.4 * apex_tiled

    # High conf: blend both
    # Low conf: trust DA3, reduce APEX sharpening
    blend_weight = np.clip(conf, 0.3, 1.0)
    return blend_weight * apex_tiled + (1 - blend_weight) * da3_anchor['depth']
```

---

## Outputs to Preserve

After evaluation, keep:
1. **NPZ files**: Depth + confidence for all scenes/models
2. **GLB files**: Visual QA and 3D inspection
3. **Comparison report**: `750Picacho_DA3_RnD/EVALUATION_REPORT.md`
4. **Side-by-side crops**: Pool water, railings, glass regions

Delete after decision:
- PNG input conversions (can regenerate)
- Intermediate processing files

---

## Timeline

- **Setup** (30 min): Install DA3, convert TIFFs, create structure
- **Processing** (30 min): Run 2-3 models on 6 scenes
- **Comparison** (15 min): Visual QA + confidence analysis
- **Documentation** (15 min): Write evaluation report with decision

**Total**: 90 minutes

---

## Quality Gates

Before declaring evaluation complete:

- [ ] All 6 scenes processed with DA3NESTED-GIANT-LARGE-1.1
- [ ] At least 3 scenes processed with DA3METRIC-LARGE or DA3MONO-LARGE
- [ ] NPZ files verified (contain depth + conf arrays, no NaN)
- [ ] GLB visual inspection completed (at least Pool + Aerial)
- [ ] Confidence maps analyzed (low-conf aligns with water/glass)
- [ ] Side-by-side comparison with APEX depth maps (Pool mandatory)
- [ ] Decision documented: Keep DA3 or defer to future R&D

---

## Next Action

**Ready to proceed?**

```bash
# Quick install verification (5 min)
cd /Users/rc/Transformation_Portal
pip uninstall depth-anything-3 -y
git clone https://github.com/DepthAnything/Depth-Anything-V3.git depth_anything_3_official
cd depth_anything_3_official
pip install -e .
da3 --help  # Should work
python -c "from depth_anything_3.api import DepthAnything3; print('✓')"
```

**Once verified, run the full evaluation script above.**

---

**End of Geometry Evaluation Plan**
