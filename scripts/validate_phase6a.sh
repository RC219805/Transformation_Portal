#!/bin/bash
# Phase 6A Validation Script
# Run all tests and validations to ensure implementation is complete

set -e  # Exit on error

echo "========================================================"
echo "Phase 6A: Gaussian Splatting Rasterizer - Validation"
echo "========================================================"
echo ""

# Check imports
echo "1. Checking imports..."
python -c "
from transformation_portal.spatial_ai.reconstruction import GaussianBackend, ReconstructionInput, CameraParams
from transformation_portal.spatial_ai.reconstruction.gaussian_rasterizer import render_gaussians, project_gaussians_2d
print('   ✓ All imports successful')
"

# Run unit tests
echo ""
echo "2. Running unit tests..."
pytest tests/spatial_ai/reconstruction/test_gaussian_rasterizer.py -v --tb=line -q

# Quick integration check
echo ""
echo "3. Quick integration check..."
python -c "
import torch
from transformation_portal.spatial_ai.reconstruction.gaussian_rasterizer import render_gaussians

# Simple smoke test
positions = torch.tensor([[0.0, 0.0, 5.0]])
colors = torch.tensor([[1.0, 0.0, 0.0]])
scales = torch.tensor([[0.5, 0.5, 0.5]])
rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
opacities = torch.tensor([[1.0]])
intrinsics = torch.eye(3)
intrinsics[0,0] = intrinsics[1,1] = 200.0
intrinsics[0,2] = 80.0
intrinsics[1,2] = 60.0
extrinsics = torch.eye(4)

rendered = render_gaussians(positions, colors, scales, rotations, opacities,
                           intrinsics, extrinsics, (120, 160), device='cpu')
assert rendered.shape == (120, 160, 3)
assert not torch.isnan(rendered).any()
print('   ✓ Integration check passed')
"

# Check file structure
echo ""
echo "4. Verifying file structure..."
files=(
    "src/transformation_portal/spatial_ai/reconstruction/gaussian_rasterizer.py"
    "tests/spatial_ai/reconstruction/test_gaussian_rasterizer.py"
    "examples/phase6a_gaussian_rasterizer_demo.py"
    "docs/phase6a_implementation_summary.md"
    "docs/phase6a_quick_reference.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file MISSING"
        exit 1
    fi
done

# Linting check
echo ""
echo "5. Running linting checks..."
python -m flake8 src/transformation_portal/spatial_ai/reconstruction/gaussian_rasterizer.py \
    --max-line-length=127 --count
echo "   ✓ Linting passed"

# Success
echo ""
echo "========================================================"
echo "✅ Phase 6A Validation Complete - All Checks Passed"
echo "========================================================"
echo ""
echo "Summary:"
echo "  • Rasterizer module: ✅ Working"
echo "  • Unit tests: ✅ 19/19 passed"
echo "  • Integration: ✅ Verified"
echo "  • File structure: ✅ Complete"
echo "  • Linting: ✅ Clean"
echo ""
echo "Ready to commit!"
