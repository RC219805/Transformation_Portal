#!/bin/bash
# Auto-generated dependency installation script for lux_depth_v3
# Generated: 2026-01-02

set -e  # Exit on error

echo "=========================================="
echo "Lux Depth V3 - Dependency Installation"
echo "=========================================="
echo ""

# Detect hardware
echo "=== Hardware Detection ==="
CUDA_AVAILABLE=false
MPS_AVAILABLE=false

if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    CUDA_AVAILABLE=true
elif [[ "$(uname)" == "Darwin" ]]; then
    # Check for Apple Silicon
    if [[ "$(uname -m)" == "arm64" ]]; then
        echo "✓ Apple Silicon detected (MPS support available)"
        MPS_AVAILABLE=true
    else
        echo "  Intel Mac detected (CPU only)"
    fi
else
    echo "  No GPU detected (CPU only)"
fi

echo ""
echo "=== Installing Core Dependencies ==="

# Install NumPy first (required by many packages)
echo "Installing NumPy..."
pip install numpy

# Install Pillow for image I/O
echo "Installing Pillow..."
pip install Pillow

# Install pytest for testing
echo "Installing pytest..."
pip install pytest pytest-cov

echo ""
echo "=== Installing PyTorch ==="

if [ "$CUDA_AVAILABLE" = true ]; then
    # Detect CUDA version
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
    echo "CUDA Version: $CUDA_VERSION"

    if [[ "$CUDA_VERSION" =~ ^12\. ]]; then
        echo "Installing PyTorch with CUDA 12.x support..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" =~ ^11\. ]]; then
        echo "Installing PyTorch with CUDA 11.x support..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "⚠ CUDA version $CUDA_VERSION not recognized, installing CPU version"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
elif [ "$MPS_AVAILABLE" = true ]; then
    echo "Installing PyTorch with Apple Silicon (MPS) support..."
    pip install torch torchvision
else
    echo "Installing PyTorch (CPU-only)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "=== Installing Depth Anything V3 ==="
pip install depth-anything-v3

echo ""
echo "=== Installing Additional Dependencies ==="
pip install tqdm pyyaml

echo ""
echo "=== Optional: Installing Lux Depth V2 ==="
if [ -d "../lux_depth_v2" ]; then
    echo "Found lux_depth_v2 in parent directory, installing..."
    pip install -e ../lux_depth_v2/
    echo "✓ lux_depth_v2 installed"
else
    echo "⚠ lux_depth_v2 not found at ../lux_depth_v2/"
    echo "  Orchestrator will fail without V2 - install manually if needed"
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Installed packages:"
pip list | grep -E "torch|numpy|Pillow|pytest|depth-anything|lux-depth"
echo ""

echo "=== Next Steps ==="
echo "1. Verify installation: python3 -c 'import torch; import depth_anything_3; print(\"✓ All imports OK\")'"
echo "2. Run tests: pytest tests/ -v"
echo "3. Try CLI: lux-depth-v3 --help"
echo ""

if [ "$CUDA_AVAILABLE" = true ]; then
    echo "GPU acceleration enabled (CUDA)"
elif [ "$MPS_AVAILABLE" = true ]; then
    echo "GPU acceleration enabled (Apple MPS)"
else
    echo "⚠ CPU-only mode - processing will be slower"
fi

echo ""
echo "See INTEGRATION_TEST_GUIDE.md for testing instructions"
