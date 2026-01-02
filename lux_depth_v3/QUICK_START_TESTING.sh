#!/bin/bash
# Quick Start Guide for Lux Depth V3 Integration Testing
# Generated: 2026-01-02

set -e

echo "=================================================================="
echo "Lux Depth V3 - Quick Start Guide"
echo "=================================================================="
echo ""

echo "This guide will help you complete integration testing in 3 steps:"
echo ""
echo "  1. Install dependencies (~5-15 min)"
echo "  2. Run integration tests (~2-5 min)"
echo "  3. Run E2E pipeline test (~1-3 min)"
echo ""
echo "Total estimated time: 20-30 minutes"
echo ""

# Check if we're in the right directory
if [ ! -f "INSTALL_DEPENDENCIES.sh" ]; then
    echo "❌ ERROR: INSTALL_DEPENDENCIES.sh not found"
    echo "   Please run this script from lux_depth_v3/ directory"
    exit 1
fi

echo "=================================================================="
echo "STEP 1: Install Dependencies"
echo "=================================================================="
echo ""
echo "This will install:"
echo "  - NumPy, Pillow, pytest (core dependencies)"
echo "  - PyTorch (with hardware detection)"
echo "  - Depth Anything V3"
echo "  - (Optional) lux_depth_v2 if available"
echo ""
echo "Download size: ~2-5 GB"
echo "Time: ~5-15 minutes (depends on internet speed)"
echo ""

read -p "Continue with installation? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    echo ""
    echo "To install manually, run:"
    echo "  ./INSTALL_DEPENDENCIES.sh"
    exit 0
fi

echo ""
echo "Starting installation..."
echo ""

chmod +x INSTALL_DEPENDENCIES.sh
./INSTALL_DEPENDENCIES.sh

echo ""
echo "=================================================================="
echo "STEP 2: Verify Installation"
echo "=================================================================="
echo ""

python3 -c "import torch; import depth_anything_3; import numpy; import PIL; import pytest; print('✓ All dependencies installed successfully')" || {
    echo "❌ Installation verification failed"
    echo "   Some dependencies may not have installed correctly"
    echo "   Review error messages above and retry installation"
    exit 1
}

echo ""
echo "=================================================================="
echo "STEP 3: Run Integration Tests"
echo "=================================================================="
echo ""
echo "Running pytest tests..."
echo ""

if command -v pytest &> /dev/null; then
    pytest tests/ -v || {
        echo "⚠ Some tests failed"
        echo "   Review test output above for details"
        echo "   See TESTING_STATUS_REPORT.md for troubleshooting"
    }
else
    echo "⚠ pytest not found, skipping integration tests"
    echo "   Install with: pip install pytest"
fi

echo ""
echo "=================================================================="
echo "STEP 4: Run End-to-End Pipeline Test"
echo "=================================================================="
echo ""

# Check for test images
if [ ! -d "test_images" ] || [ -z "$(ls -A test_images 2>/dev/null)" ]; then
    echo "Creating test images..."
    mkdir -p test_images
    cd test_images
    python3 generate_test_image.py || {
        echo "⚠ Test image generation failed"
        echo "   Add your own test images to test_images/ directory"
    }
    cd ..
fi

echo ""
echo "Running E2E pipeline test..."
echo ""

lux-depth-v3 enhance \
  --input-dir test_images/ \
  --output-dir test_output/ \
  --model metric-large \
  --verbose || {
    echo "⚠ E2E test failed"
    echo "   Review error messages above"
    echo "   See TESTING_STATUS_REPORT.md for troubleshooting"
}

echo ""
echo "=================================================================="
echo "STEP 5: Validate Outputs"
echo "=================================================================="
echo ""

if [ -d "test_output" ]; then
    echo "Checking outputs..."
    echo ""

    if [ -d "test_output/depth" ]; then
        DEPTH_COUNT=$(ls test_output/depth/ 2>/dev/null | wc -l)
        echo "✓ Depth maps: $DEPTH_COUNT files in test_output/depth/"
    else
        echo "✗ No depth outputs found"
    fi

    if [ -d "test_output/v2" ]; then
        V2_COUNT=$(ls test_output/v2/ 2>/dev/null | wc -l)
        echo "✓ V2 enhanced: $V2_COUNT files in test_output/v2/"
    else
        echo "⚠ No V2 outputs found (V2 integration may not be installed)"
    fi

    if [ -d "test_output/manifests" ]; then
        MANIFEST_COUNT=$(ls test_output/manifests/ 2>/dev/null | wc -l)
        echo "✓ Manifests: $MANIFEST_COUNT files in test_output/manifests/"
    else
        echo "✗ No manifests found"
    fi

    echo ""
    echo "To view outputs:"
    echo "  open test_output/depth/"
    echo "  open test_output/v2/"
    echo "  cat test_output/manifests/*.json | python3 -m json.tool | less"
else
    echo "✗ No test_output/ directory found"
    echo "   E2E test may have failed - review error messages above"
fi

echo ""
echo "=================================================================="
echo "Integration Testing Complete!"
echo "=================================================================="
echo ""
echo "Next steps:"
echo "  1. Review outputs in test_output/ directory"
echo "  2. Read TESTING_STATUS_REPORT.md for detailed results"
echo "  3. Read INTEGRATION_TESTING_COMPLETE.md for next milestones"
echo ""
echo "To run additional tests:"
echo "  pytest tests/ -v                    # All tests"
echo "  pytest tests/ -k 'metric' -v        # Metric depth tests only"
echo "  pytest tests/ -k 'license' -v       # License tests only"
echo ""
echo "To process your own images:"
echo "  lux-depth-v3 enhance --input-dir YOUR_DIR/ --output-dir OUTPUT_DIR/ --model metric-large --verbose"
echo ""
echo "For help:"
echo "  lux-depth-v3 --help"
echo "  lux-depth-v3 enhance --help"
echo ""
echo "=================================================================="
