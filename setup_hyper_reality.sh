#!/bin/bash

# Transformation_Portal Hyper-Reality Enhancement Setup
# Optimized for Apple Silicon M4 Max

echo "=================================================="
echo "TRANSFORMATION PORTAL - HYPER REALITY ENHANCEMENT"
echo "Version: 3.0.0 | Target Quality: 105/100"
echo "=================================================="
echo ""

# Check if we're in the Transformation_Portal directory
if [[ ! -d "src" ]] || [[ ! -f "requirements.txt" ]]; then
    echo "⚠️  Please run this script from the Transformation_Portal root directory"
    echo "   Current directory: $(pwd)"
    exit 1
fi

echo "✓ Running from Transformation_Portal directory"
echo ""

# Check for Python virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [[ -d "venv" ]]; then
        echo "📦 Activating virtual environment..."
        source venv/bin/activate
    else
        echo "⚠️  No virtual environment found. Creating one..."
        python3 -m venv venv
        source venv/bin/activate
    fi
fi

echo "✓ Virtual environment active: $VIRTUAL_ENV"
echo ""

# Install required dependencies if not present
echo "📦 Checking dependencies..."

REQUIRED_PACKAGES=(
    "torch"
    "torchvision"
    "kornia"
    "opencv-python"
    "Pillow"
    "scipy"
    "scikit-image"
    "tqdm"
    "numpy"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    package_name="${package%%-*}"
    if ! python -c "import ${package_name}" 2>/dev/null; then
        echo "  Installing $package..."
        pip install "$package" --quiet
    else
        echo "  ✓ $package"
    fi
done

# Special handling for PyTorch with Metal Performance Shaders
echo ""
echo "🔧 Configuring PyTorch for hardware acceleration..."
python -c "
import torch
if torch.backends.mps.is_available():
    print('  ✓ Metal Performance Shaders (MPS) available')
    print(f'  ✓ PyTorch version: {torch.__version__}')
    device = torch.device('mps')
    print(f'  ✓ Using device: {device}')

    # Test MPS functionality
    try:
        test_tensor = torch.randn(1, 3, 512, 512).to(device)
        print('  ✓ MPS acceleration test passed')
    except Exception as e:
        print(f'  ⚠️  MPS test failed: {e}')
elif torch.cuda.is_available():
    print('  ✓ CUDA available')
    print(f'  ✓ PyTorch version: {torch.__version__}')
    print(f'  ✓ Using device: cuda')
else:
    print('  ⚠️  MPS/CUDA not available - will use CPU (slower)')
    print(f'  ✓ PyTorch version: {torch.__version__}')
"

echo ""
echo "📁 Verifying enhancement module installation..."

# Verify module structure
if [[ -f "src/enhancements/hyper_reality_enhancement.py" ]] && [[ -f "src/enhancements/__init__.py" ]]; then
    echo "  ✓ Module files present"
else
    echo "  ⚠️  Module files missing"
    exit 1
fi

# Test import
python -c "
import sys
sys.path.insert(0, 'src')
try:
    from enhancements import HyperRealityProcessor, EnhancementConfig, QualityMode
    print('  ✓ Module imports successfully')
except Exception as e:
    print(f'  ❌ Import failed: {e}')
    sys.exit(1)
"

echo ""
echo "📝 Creating utility scripts..."

# Create command-line interface
cat > enhance_hyper_reality.py << 'EOF'
#!/usr/bin/env python3
"""
Command-line interface for Hyper-Reality Enhancement
Part of Transformation_Portal
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from enhancements import enhance_image, QualityMode

def main():
    parser = argparse.ArgumentParser(
        description="Enhance images to hyper-reality quality (105/100)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quality Modes:
  STANDARD    (70-85)   Traditional photographic quality
  PREMIUM     (85-95)   Marketing-grade enhancement
  HYPER       (95-105)  Hyper-reality transcendence
  QUANTUM     (105-120) Quantum-amplified reality
  THEORETICAL (120-150) Theoretical maximum

Examples:
  %(prog)s input.jpg                    # Default 105/100 quality
  %(prog)s input.jpg -q 120             # Quantum mode (120/100)
  %(prog)s input.jpg -o output.jpg -i   # Save intermediate stages
        """
    )

    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output path (auto-generated if not specified)")
    parser.add_argument("-q", "--quality", type=int, default=105,
                       help="Target quality score (default: 105)")
    parser.add_argument("-i", "--intermediate", action="store_true",
                       help="Save intermediate enhancement stages")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Validate input
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)

    # Determine quality mode
    if args.quality <= 85:
        mode = "STANDARD"
    elif args.quality <= 95:
        mode = "PREMIUM"
    elif args.quality <= 105:
        mode = "HYPER"
    elif args.quality <= 120:
        mode = "QUANTUM"
    else:
        mode = "THEORETICAL"

    print(f"\n🚀 Enhancing to {mode} quality ({args.quality}/100)...")

    try:
        results = enhance_image(
            image_path=args.input,
            output_path=args.output,
            target_quality=args.quality,
            save_intermediate=args.intermediate
        )

        print(f"\n✅ Success! Quality achieved: {results['quality_score']}/100")

    except Exception as e:
        print(f"\n❌ Enhancement failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x enhance_hyper_reality.py
echo "  ✓ Command-line interface created (enhance_hyper_reality.py)"

# Create test script
cat > test_hyper_reality.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for Hyper-Reality Enhancement
Tests the 105/100 quality achievement pipeline
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from enhancements import HyperRealityProcessor, EnhancementConfig, QualityMode

def test_enhancement():
    """Test the enhancement pipeline"""

    print("\n🧪 Testing Hyper-Reality Enhancement Pipeline...")

    # Configure for maximum quality
    config = EnhancementConfig(
        target_quality=105,
        mode=QualityMode.QUANTUM
    )

    # Create processor
    processor = HyperRealityProcessor(config)

    # Find test image
    test_images = [
        "input_images/test.jpg",
        "data/samples/test.jpg",
        "examples/sample.jpg",
        "assets/test.jpg"
    ]

    test_image = None
    for img in test_images:
        if os.path.exists(img):
            test_image = img
            break

    if test_image is None:
        print("  ⚠️  No test image found")
        print("  Please place a test image in one of these locations:")
        for img in test_images:
            print(f"    - {img}")
        return False

    print(f"  Using test image: {test_image}")

    # Process image
    output_dir = Path("outputs/hyper_reality_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{Path(test_image).stem}_enhanced_105.jpg"

    try:
        results = processor.process_image(
            image_path=test_image,
            output_path=str(output_path),
            save_intermediate=True
        )

        print(f"\n✓ Enhancement successful!")
        print(f"  Final Quality: {results['quality_score']}/100")
        print(f"  Processing Time: {results['processing_time']:.2f}s")
        print(f"  Output: {results['output_path']}")

        return True

    except Exception as e:
        print(f"\n❌ Enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhancement()
    sys.exit(0 if success else 1)
EOF

chmod +x test_hyper_reality.py
echo "  ✓ Test script created (test_hyper_reality.py)"

echo ""
echo "=================================================="
echo "✅ SETUP COMPLETE"
echo "=================================================="
echo ""
echo "Available commands:"
echo ""
echo "  1. Test the enhancement pipeline:"
echo "     python test_hyper_reality.py"
echo ""
echo "  2. Enhance a single image:"
echo "     python enhance_hyper_reality.py <image_path>"
echo ""
echo "  3. Enhance with custom quality:"
echo "     python enhance_hyper_reality.py <image_path> -q 120"
echo ""
echo "  4. Save intermediate stages:"
echo "     python enhance_hyper_reality.py <image_path> -i"
echo ""
echo "  5. Use in Python:"
echo "     from enhancements import enhance_image"
echo "     enhance_image('input.jpg', target_quality=105)"
echo ""
echo "=================================================="
echo "Ready to achieve 105/100 quality transcendence! 🚀"
echo "=================================================="
