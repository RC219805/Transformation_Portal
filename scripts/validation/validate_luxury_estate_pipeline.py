#!/usr/bin/env python3
"""
Validate Luxury Estate Master Pipeline
======================================

Quick validation script to ensure all components are properly installed
and the pipeline can be initialized.

Usage:
    python scripts/validation/validate_luxury_estate_pipeline.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINES_DIR = REPO_ROOT / "scripts" / "pipelines"
UTILITIES_DIR = REPO_ROOT / "scripts" / "utilities"
os.environ.setdefault("TP_LUXURY_ESTATE_PIPELINE_LOG", "/tmp/tp-luxury-estate-pipeline.log")
for import_root in (PIPELINES_DIR, UTILITIES_DIR):
    root_text = str(import_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

print("=" * 80)
print("LUXURY ESTATE MASTER PIPELINE - VALIDATION TEST")
print("=" * 80)
print()

# Test 1: Python version
print("[1/8] Checking Python version...")
if sys.version_info < (3, 10):
    print(f"  ✗ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    print(f"  ✗ Python 3.10+ required")
    sys.exit(1)
print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Test 2: Core dependencies
print("\n[2/8] Checking core dependencies...")
try:
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    print("  ✓ NumPy, OpenCV, PyTorch, Pillow")
except ImportError as e:
    print(f"  ✗ Missing core dependency: {e}")
    sys.exit(1)

# Test 3: TIFF support
print("\n[3/8] Checking TIFF support...")
try:
    import tifffile

    print("  ✓ tifffile (16/32-bit TIFF support)")
except ImportError:
    print("  ⚠ tifffile not available - limited TIFF support")
    print("    Install with: pip install tifffile")

# Test 4: Depth pipeline
print("\n[4/8] Checking depth pipeline...")
try:
    from transformation_portal.depth.models import DepthAnythingV2Model, ModelBackend, ModelVariant
    from transformation_portal.depth.processors import (
        AtmosphericEffects,
        DepthAwareDenoise,
        DepthGuidedFilters,
        ZoneToneMapping,
    )

    print("  ✓ Depth Anything V2 pipeline available")
except ImportError as e:
    print(f"  ⚠ Depth pipeline not fully available: {e}")
    print("    Pipeline will skip depth processing")

# Test 5: Material Response
print("\n[5/8] Checking Material Response...")
try:
    from transformation_portal.processors.material_response.core import LightingProfile, MaterialAestheticProfile

    print("  ✓ Material Response Technology available")
except ImportError:
    print("  ⚠ Material Response not available")
    print("    Pipeline will use simplified enhancement")

# Test 6: Tone mapping
print("\n[6/8] Checking tone mapping...")
try:
    from tonemapper_agx_filmic import apply_filmic_hable, linear_to_srgb

    print("  ✓ Filmic (Hable) tone mapper")
except ImportError:
    print("  ✗ Tone mapping not available")
    sys.exit(1)

try:
    from tonemapper_agx_filmic import apply_agx_ocio

    print("  ✓ AgX OCIO tone mapper")
except ImportError:
    print("  ⚠ AgX OCIO not available (requires PyOpenColorIO)")

# Test 7: AI enhancement
print("\n[7/8] Checking AI enhancement...")
try:
    from controlnet_aux import CannyDetector
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

    print("  ✓ ControlNet + Stable Diffusion available")
except ImportError:
    print("  ⚠ AI enhancement not available")
    print("    Install with: pip install diffusers controlnet-aux transformers")

# Test 8: Real-ESRGAN
print("\n[8/8] Checking Real-ESRGAN upscaler...")
try:
    from realesrgan import RealESRGANer
    from realesrgan.archs.rrdbnet_arch import RRDBNet

    # Check for weights
    weights_path = Path("weights/RealESRGAN_x4plus.pth")
    if weights_path.exists():
        print("  ✓ Real-ESRGAN available (weights found)")
    else:
        print("  ⚠ Real-ESRGAN installed but weights missing")
        print(f"    Download to: {weights_path}")
        print("    URL: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
except ImportError:
    print("  ⚠ Real-ESRGAN not available")
    print("    Install with: pip install realesrgan basicsr")

# Test 9: Device detection
print("\n[Device Detection]")
print("-" * 80)
if torch.backends.mps.is_available():
    print("  ✓ Apple Metal (MPS) - ACCELERATED")
elif torch.cuda.is_available():
    print("  ✓ NVIDIA CUDA - ACCELERATED")
else:
    print("  ⚠ CPU only - Processing will be slower")

# Test 10: Pipeline import
print("\n[Pipeline Test]")
print("-" * 80)
try:
    from luxury_estate_master_pipeline import LuxuryEstateMasterPipeline, get_750_picacho_preset, get_aerial_preset

    print("  ✓ Pipeline module imports successfully")

    # Try to initialize preset
    preset = get_750_picacho_preset()
    print(f"  ✓ Loaded preset: {preset.name}")
    print(f"    - Depth: {'enabled' if preset.depth.enabled else 'disabled'}")
    print(f"    - Material Response: {'enabled' if preset.material_response.enabled else 'disabled'}")
    print(f"    - Tone Mapping: {preset.tone_mapping.method}")
    print(f"    - AI Enhancement: {'enabled' if preset.ai_enhancement.enabled else 'disabled'}")
    print(f"    - Upscaling: {preset.upscaling.method} ({preset.upscaling.scale_factor}x)")

except Exception as e:
    print(f"  ✗ Pipeline initialization failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 11: Check source images
print("\n[Source Images]")
print("-" * 80)
source_dir = Path("input_images/750_Picacho_HDR_sRGB_alpha_32-bit_TIFFs")
if source_dir.exists():
    images = list(source_dir.glob("*.tif")) + list(source_dir.glob("*.tiff"))
    if images:
        print(f"  ✓ Found {len(images)} source images:")
        for img in images:
            size_mb = img.stat().st_size / (1024 * 1024)
            print(f"    • {img.name} ({size_mb:.1f} MB)")
    else:
        print(f"  ⚠ No TIFF files found in {source_dir}")
else:
    print(f"  ⚠ Source directory not found: {source_dir}")

# Summary
print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)

# Check critical requirements
critical_ok = True
warnings = []

if sys.version_info < (3, 10):
    critical_ok = False

try:
    import cv2
    import numpy
    import torch
    from PIL import Image
except ImportError:
    critical_ok = False

try:
    from tonemapper_agx_filmic import apply_filmic_hable
except ImportError:
    critical_ok = False

try:
    from luxury_estate_master_pipeline import LuxuryEstateMasterPipeline
except ImportError:
    critical_ok = False

# Optional but recommended
try:
    import tifffile
except ImportError:
    warnings.append("tifffile missing - limited 16/32-bit TIFF support")

try:
    from transformation_portal.depth.models import DepthAnythingV2Model
except ImportError:
    warnings.append("Depth pipeline unavailable - will skip depth processing")

try:
    from diffusers import StableDiffusionControlNetImg2ImgPipeline
except ImportError:
    warnings.append("AI enhancement unavailable - will skip AI stage")

try:
    from realesrgan import RealESRGANer
except ImportError:
    warnings.append("Real-ESRGAN unavailable - will use Lanczos upscaling")

if critical_ok:
    print("\n✅ CORE PIPELINE: READY")
    print("\nYou can process images with:")
    print("  python luxury_estate_master_pipeline.py input.tif")

    if warnings:
        print(f"\n⚠️  OPTIONAL FEATURES: {len(warnings)} missing")
        for warning in warnings:
            print(f"  • {warning}")
        print("\nPipeline will work but some features will be disabled.")
    else:
        print("\n🎉 ALL FEATURES: AVAILABLE")
        print("Full pipeline with all enhancements ready!")

    print("\nTo process all 750 Picacho images:")
    print("  ./process_750_picacho_elite_batch.sh")

    sys.exit(0)
else:
    print("\n❌ CRITICAL DEPENDENCIES MISSING")
    print("\nInstall required dependencies:")
    print("  pip install -e .")
    print('  pip install -e ".[ml,tiff]"')
    sys.exit(1)
