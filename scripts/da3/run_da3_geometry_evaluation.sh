#!/bin/bash
#
# DA3 Geometry Evaluation Runner for 750 Picacho
# Purpose: Test if DA3 produces better global geometry than tiled DA2 APEX
# Focus: Pool water, glass, railings, sky, long planes
#
# NON-COMMERCIAL R&D ONLY
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "DA3 Geometry Engine Evaluation for 750 Picacho"
echo "=================================================="
echo ""

# ============================================================
# Phase 0: Verify DA3 Installation
# ============================================================
echo -e "${YELLOW}Phase 0: Verifying DA3 installation...${NC}"

if ! command -v da3 &> /dev/null; then
    echo -e "${RED}✗ da3 CLI not found${NC}"
    echo ""
    echo "Install DA3:"
    echo "  pip uninstall -y depth-anything-3 || true"
    echo "  mkdir -p external"
    echo "  git clone https://github.com/ByteDance-Seed/depth-anything-3 external/depth-anything-3"
    echo "  pip install -e external/depth-anything-3"
    exit 1
fi

if ! python -c "from depth_anything_3.api import DepthAnything3" 2>/dev/null; then
    echo -e "${RED}✗ DA3 Python API not importable${NC}"
    echo "Reinstall with: pip install -e external/depth-anything-3"
    exit 1
fi

echo -e "${GREEN}✓ DA3 CLI and Python API available${NC}"

# ============================================================
# Phase 1: Setup Directory Structure
# ============================================================
echo ""
echo -e "${YELLOW}Phase 1: Creating R&D directory structure...${NC}"

RND_ROOT="750Picacho_DA3_RnD"
mkdir -p "$RND_ROOT"/{inputs_png,outputs,comparisons}

# License notice
cat > "$RND_ROOT/README_LICENSE_NOTICE.txt" << 'EOF'
NON-COMMERCIAL R&D ONLY — DO NOT SHIP TO CLIENTS

This directory contains Depth Anything 3 (DA3) evaluation outputs.

Model Licenses:
- DA3NESTED-GIANT-LARGE-1.1: CC-BY-NC-4.0 (non-commercial only)
- DA3METRIC-LARGE: Apache-2.0 (commercial-friendly)
- DA3MONO-LARGE: Apache-2.0 (commercial-friendly)

Purpose: Internal geometry quality comparison vs lux-depth-v2 APEX-100.
Production pipeline: lux-depth-v2 APEX-100 (unchanged).

Evaluation Focus:
- Pool water plane smoothness
- Glass/reflection handling
- Railing consistency
- Sky/aerial coherence
- Long interior plane stability
EOF

echo -e "${GREEN}✓ Created $RND_ROOT/${NC}"

# ============================================================
# Phase 2: Convert TIFFs to 8-bit PNG
# ============================================================
echo ""
echo -e "${YELLOW}Phase 2: Converting TIFFs to 8-bit sRGB PNG...${NC}"

SOURCE_DIR="750Picacho_Source_TIFFs"
INPUT_DIR="$RND_ROOT/inputs_png"

if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}✗ Source directory not found: $SOURCE_DIR${NC}"
    exit 1
fi

TIFF_COUNT=$(find "$SOURCE_DIR" -type f \( -name "*.tif" -o -name "*.tiff" \) | wc -l | xargs)
echo "Found $TIFF_COUNT TIFF files to convert"

for tiff in "$SOURCE_DIR"/*.tif*; do
    [ -f "$tiff" ] || continue
    base=$(basename "$tiff" | sed 's/\.[^.]*$//')
    output_png="$INPUT_DIR/${base}.png"

    if [ -f "$output_png" ]; then
        echo "  ⊙ Skip (exists): ${base}.png"
        continue
    fi

    echo "  Converting: ${base}.png"
    magick "$tiff" \
        -colorspace sRGB \
        -depth 8 \
        -auto-orient \
        -strip \
        "$output_png"
done

PNG_COUNT=$(find "$INPUT_DIR" -name "*.png" | wc -l | xargs)
echo -e "${GREEN}✓ Converted $PNG_COUNT images to 8-bit PNG${NC}"

# ============================================================
# Phase 3: Run DA3 Inference (Track A + Track B)
# ============================================================
echo ""
echo -e "${YELLOW}Phase 3: Running DA3 inference...${NC}"

OUT_BASE="$RND_ROOT/outputs"
mkdir -p "$OUT_BASE"

# Function to run DA3 for a model
run_da3_model() {
    local MODEL=$1
    local MODEL_NAME=$2
    local SCENES=("${@:3}")

    echo ""
    echo "============================================================"
    echo "Model: $MODEL_NAME"
    echo "============================================================"

    OUT_DIR="$OUT_BASE/$MODEL_NAME"
    mkdir -p "$OUT_DIR"

    # Start backend
    echo "Starting DA3 backend..."
    da3 backend --model-dir "$MODEL" --gallery-dir "$OUT_DIR" &> /tmp/da3_backend_${MODEL_NAME}.log &
    BACKEND_PID=$!

    # Wait for backend to initialize
    sleep 15

    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo -e "${RED}✗ Backend failed to start${NC}"
        cat /tmp/da3_backend_${MODEL_NAME}.log
        return 1
    fi

    echo "Backend running (PID: $BACKEND_PID)"

    # Process scenes
    local processed=0
    for img in "${SCENES[@]}"; do
        if [ ! -f "$img" ]; then
            echo "  ⊙ Skip (not found): $(basename $img)"
            continue
        fi

        base=$(basename "$img" .png)
        scene_out="$OUT_DIR/$base"

        if [ -d "$scene_out" ] && [ -f "$scene_out/output.npz" ]; then
            echo "  ⊙ Skip (exists): $base"
            ((processed++))
            continue
        fi

        echo "  Processing: $base"

        if da3 auto "$img" \
            --export-dir "$scene_out" \
            --export-format mini_npz-glb \
            --use-backend \
            --process-res 1024 \
            --process-res-method upper_bound_resize \
            &> /tmp/da3_${MODEL_NAME}_${base}.log; then
            echo -e "    ${GREEN}✓ Success${NC}"
            ((processed++))
        else
            echo -e "    ${RED}✗ Failed (see /tmp/da3_${MODEL_NAME}_${base}.log)${NC}"
        fi
    done

    # Stop backend
    echo "Stopping backend..."
    kill $BACKEND_PID 2>/dev/null || true
    wait $BACKEND_PID 2>/dev/null || true

    echo -e "${GREEN}✓ Processed $processed scenes with $MODEL_NAME${NC}"
}

# Get all input PNGs
ALL_SCENES=("$INPUT_DIR"/*.png)

# Track A: DA3NESTED-GIANT-LARGE-1.1 (Quality Ceiling, All Scenes)
run_da3_model \
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1" \
    "DA3NESTED-GIANT-LARGE-1.1" \
    "${ALL_SCENES[@]}"

# Track B: DA3METRIC-LARGE (Apache 2.0, Focus Scenes)
# Focus on failure modes: Pool, Aerial, GreatRoom
FOCUS_SCENES=()
for scene in "Pool" "Aerial" "GreatRoom"; do
    found=$(find "$INPUT_DIR" -name "*${scene}*.png" | head -1)
    [ -n "$found" ] && FOCUS_SCENES+=("$found")
done

if [ ${#FOCUS_SCENES[@]} -gt 0 ]; then
    run_da3_model \
        "depth-anything/DA3METRIC-LARGE" \
        "DA3METRIC-LARGE" \
        "${FOCUS_SCENES[@]}"
fi

# Track B: DA3MONO-LARGE (Apache 2.0, Focus Scenes)
if [ ${#FOCUS_SCENES[@]} -gt 0 ]; then
    run_da3_model \
        "depth-anything/DA3MONO-LARGE" \
        "DA3MONO-LARGE" \
        "${FOCUS_SCENES[@]}"
fi

# ============================================================
# Phase 4: Generate Comparison Visualizations
# ============================================================
echo ""
echo -e "${YELLOW}Phase 4: Generating comparison visualizations...${NC}"

python3 << 'PYTHON_SCRIPT'
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

rnd_root = Path("750Picacho_DA3_RnD")
outputs = rnd_root / "outputs"
comparisons = rnd_root / "comparisons"
comparisons.mkdir(exist_ok=True)

models = ["DA3NESTED-GIANT-LARGE-1.1", "DA3METRIC-LARGE", "DA3MONO-LARGE"]

for model_name in models:
    model_dir = outputs / model_name
    if not model_dir.exists():
        continue

    print(f"\nProcessing: {model_name}")

    for scene_dir in model_dir.iterdir():
        if not scene_dir.is_dir():
            continue

        scene_name = scene_dir.name
        npz_file = scene_dir / "output.npz"

        if not npz_file.exists():
            continue

        # Load depth and confidence
        data = np.load(npz_file)
        depth = data['depth']
        conf = data.get('conf', None)

        # Create visualization
        fig, axes = plt.subplots(1, 3 if conf is not None else 2, figsize=(15, 5))

        axes[0].imshow(depth, cmap='turbo')
        axes[0].set_title(f'{scene_name} - Depth')
        axes[0].axis('off')

        if conf is not None:
            axes[1].imshow(conf, cmap='viridis')
            axes[1].set_title(f'{scene_name} - Confidence')
            axes[1].axis('off')

            # Low confidence mask
            axes[2].imshow(conf < 0.3, cmap='gray')
            axes[2].set_title(f'{scene_name} - Low Conf Mask (<0.3)')
            axes[2].axis('off')
        else:
            # Just show depth histogram
            axes[1].hist(depth.flatten(), bins=100, alpha=0.7)
            axes[1].set_title('Depth Distribution')
            axes[1].set_xlabel('Depth Value')
            axes[1].set_ylabel('Frequency')

        plt.tight_layout()
        out_path = comparisons / f"{model_name}_{scene_name}_analysis.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ {scene_name}: depth {depth.shape}, conf {conf.shape if conf is not None else 'N/A'}")

print("\n✓ Visualizations saved to 750Picacho_DA3_RnD/comparisons/")
PYTHON_SCRIPT

echo -e "${GREEN}✓ Comparison visualizations complete${NC}"

# ============================================================
# Phase 5: Generate Evaluation Report Template
# ============================================================
echo ""
echo -e "${YELLOW}Phase 5: Generating evaluation report template...${NC}"

cat > "$RND_ROOT/EVALUATION_REPORT.md" << 'EOF'
# DA3 Geometry Evaluation Report - 750 Picacho

**Date**: $(date +%Y-%m-%d)
**Evaluator**: [Your Name]
**Purpose**: Determine if DA3 provides better global geometry than DA2 APEX

---

## Models Tested

- **DA3NESTED-GIANT-LARGE-1.1** (CC-BY-NC-4.0): Quality ceiling
- **DA3METRIC-LARGE** (Apache-2.0): Metric scaling candidate
- **DA3MONO-LARGE** (Apache-2.0): Monocular best candidate

---

## Scene-by-Scene Evaluation

### 1. Pool Scene (Primary Failure Mode)

**APEX-100 Known Issues**:
- [ ] Water plane ripple artifacts
- [ ] Railing inconsistency
- [ ] Edge over-sharpening

**DA3 Results**:
- **Water Plane Smoothness**: [Better/Worse/Same]
- **Railing Consistency**: [Better/Worse/Same]
- **Confidence Map Quality**: [Aligns with water/glass? Yes/No]
- **Winner**: [DA3 / APEX / Tie]

**Notes**:


---

### 2. Aerial Scene (Global Scale)

**APEX-100 Known Issues**:
- [ ] Sky segmentation
- [ ] Large-scale coherence

**DA3 Results**:
- **Sky Handling**: [Better/Worse/Same]
- **Terrain Geometry**: [Better/Worse/Same]
- **Metric Depth Plausibility**: [Good/Poor/N/A]
- **Winner**: [DA3 / APEX / Tie]

**Notes**:


---

### 3. Kitchen/GreatRoom (Detail vs Global)

**APEX-100 Known Strengths**:
- [ ] Cabinetry edge preservation
- [ ] Fine detail from tiling

**DA3 Results**:
- **Detail Preservation**: [Better/Worse/Same]
- **Wall Plane Stability**: [Better/Worse/Same]
- **Glass/Window Handling**: [Better/Worse/Same]
- **Winner**: [DA3 / APEX / Tie]

**Notes**:


---

## Confidence Map Analysis

**Low-Confidence Regions Align With**:
- [ ] Water surfaces
- [ ] Glass/reflections
- [ ] Sky
- [ ] Ambiguous geometry

**Potential Use Cases**:
- [ ] Confidence-gated edge refinement
- [ ] Adaptive post-processing
- [ ] Failure mode detection

---

## Quantitative Metrics (Optional)

| Scene | Model | RMSE vs APEX | Processing Time |
|-------|-------|--------------|-----------------|
| Pool | NESTED | | |
| Pool | METRIC | | |
| Aerial | NESTED | | |
| Kitchen | NESTED | | |

---

## Decision

### DA3 Wins If:
- [ ] Pool water materially smoother
- [ ] Confidence enables better failure handling
- [ ] At least one Apache model competes with NESTED

### DA3 Loses If:
- [ ] Detail loss on cabinetry/railings
- [ ] No confidence benefit
- [ ] Only NESTED wins (non-commercial blocker)

### Final Verdict:

**[ ] Keep DA3 in R&D toolbox** - Model(s): __________________

**[ ] Defer DA3 to future sprint** - Reason: __________________

---

## Integration Pathway (If DA3 Wins)

**Proposed Architecture**:
```
Input → DA3 Global Anchor (+ confidence)
      ↓
      Tiled DA2 APEX (detail)
      ↓
      Confidence-weighted fusion
      ↓
      SegFormer MaterialsV2
```

**Next Steps**:
1.
2.
3.

---

**End of Evaluation Report**
EOF

echo -e "${GREEN}✓ Report template created: $RND_ROOT/EVALUATION_REPORT.md${NC}"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=================================================="
echo "DA3 Geometry Evaluation Complete!"
echo "=================================================="
echo ""
echo "Outputs:"
echo "  - NPZ files: $OUT_BASE/<model>/<scene>/output.npz"
echo "  - GLB files: $OUT_BASE/<model>/<scene>/output.glb"
echo "  - Visualizations: $RND_ROOT/comparisons/"
echo "  - Report template: $RND_ROOT/EVALUATION_REPORT.md"
echo ""
echo "Next Steps:"
echo "  1. Review GLB files in 3D viewer"
echo "  2. Compare depth maps vs existing APEX outputs"
echo "  3. Fill out EVALUATION_REPORT.md"
echo "  4. Decide: Keep DA3 or defer to future R&D"
echo ""
echo -e "${GREEN}Done!${NC}"
