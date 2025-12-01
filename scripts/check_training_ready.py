#!/usr/bin/env python3
"""
Pre-flight Check for Model Training
Verifies that all requirements are met before starting training.
"""

import sys
import shutil
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_check(passed, message):
    """Print a check result."""
    status = "✅" if passed else "❌"
    print(f"{status} {message}")
    return passed


def check_python_version():
    """Check Python version."""
    print_header("Python Version")
    version = sys.version_info
    required = (3, 10)
    passed = version >= required

    current = f"{version.major}.{version.minor}.{version.micro}"
    required_str = f"{required[0]}.{required[1]}+"

    if passed:
        print_check(True, f"Python {current} (>= {required_str})")
    else:
        print_check(False, f"Python {current} - Need {required_str}")
        print("   Install Python 3.10+ from https://www.python.org/")

    return passed


def check_pytorch():
    """Check PyTorch installation and GPU availability."""
    print_header("PyTorch & GPU")

    try:
        import torch
        version = torch.__version__
        print_check(True, f"PyTorch {version} installed")

        # Check CUDA
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print_check(True, f"CUDA available - {device_count}x {device_name}")
            print("   Training will be FAST (~3-4 hours)")
            return True

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print_check(True, "MPS available (Apple Silicon)")
            print("   Training will be FAST (~2.5-3.5 hours)")
            return True

        # CPU only
        print_check(True, "CPU only (no GPU detected)")
        print("   ⚠️  Training will be SLOW (~12-18 hours)")
        print("   Consider using synthetic data (--quickstart) for faster results")
        return True

    except ImportError:
        print_check(False, "PyTorch not installed")
        print("   Install: pip install torch torchvision")
        return False


def check_dependencies():
    """Check required Python packages."""
    print_header("Python Dependencies")

    required = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('tqdm', 'tqdm'),
        ('skimage', 'scikit-image'),
    ]

    all_passed = True
    for module, name in required:
        try:
            __import__(module)
            print_check(True, f"{name} installed")
        except ImportError:
            print_check(False, f"{name} not installed")
            all_passed = False

    if not all_passed:
        print("\n   Install all: pip install -r requirements/ml.txt")

    return all_passed


def check_disk_space():
    """Check available disk space."""
    print_header("Disk Space")

    try:
        cwd = Path.cwd()
        stat = shutil.disk_usage(cwd)
        free_gb = stat.free / (1024**3)

        required_gb = 10
        passed = free_gb >= required_gb

        if passed:
            print_check(True, f"{free_gb:.1f} GB free (>= {required_gb} GB required)")
        else:
            print_check(False, f"{free_gb:.1f} GB free - Need at least {required_gb} GB")
            print("   Free up disk space for checkpoints and logs")

        return passed
    except (OSError, PermissionError) as e:
        print_check(False, f"Could not check disk space: {e}")
        return False


def check_memory():
    """Check available RAM."""
    print_header("Memory (RAM)")

    try:
        # Try psutil first
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            avail_gb = mem.available / (1024**3)

            required_gb = 8
            passed = avail_gb >= required_gb

            if passed:
                print_check(True, f"{avail_gb:.1f} GB available (>= {required_gb} GB required)")
                print(f"   Total: {total_gb:.1f} GB")

                if avail_gb >= 16:
                    print("   💪 Plenty of memory - can use batch size 4-8")
                elif avail_gb >= 12:
                    print("   👍 Good memory - can use batch size 4")
                else:
                    print("   ⚠️  Minimum memory - use batch size 2")
            else:
                print_check(False, f"{avail_gb:.1f} GB available - Need at least {required_gb} GB")
                print("   Close other applications to free up memory")

            return passed

        except ImportError:
            # Fallback: try /proc/meminfo on Linux
            if Path('/proc/meminfo').exists():
                with open('/proc/meminfo') as f:
                    lines = f.readlines()
                    mem_avail = int(lines[2].split()[1]) / (1024**2)  # GB

                    required_gb = 8
                    passed = mem_avail >= required_gb

                    print_check(passed, f"{mem_avail:.1f} GB available")
                    if not passed:
                        print(f"   Need at least {required_gb} GB available")
                    return passed
            else:
                print_check(True, "Could not detect memory (psutil not installed)")
                print("   Ensure you have at least 8 GB RAM available")
                return True

    except Exception as e:
        print_check(True, f"Could not check memory: {e}")
        print("   Ensure you have at least 8 GB RAM available")
        return True


def check_training_data():
    """Check if training data is available."""
    print_header("Training Data")

    # Check 750 Picacho data
    picacho_dirs = [
        Path("projects/750_picacho_lane/Final_Production_UltraQuality"),
    ]
    # Use glob to match variations like "24098.00_750 PICACHO LANE_images" with spaces
    extracted_context = Path("extracted_context")
    if extracted_context.exists():
        picacho_dirs.extend(extracted_context.glob("*PICACHO*LANE*"))

    picacho_available = False
    for path in picacho_dirs:
        if path.exists():
            picacho_available = True
            print_check(True, f"750 Picacho data found: {path}")

    if not picacho_available:
        print_check(False, "750 Picacho data not found")
        print("   No problem - can use synthetic data instead")
        print("   Run: ./scripts/quickstart_training.sh")

    # Check training scripts
    scripts = [
        Path("scripts/train_with_750picacho.sh"),
        Path("scripts/quickstart_training.sh"),
        Path("src/enhancements/train_hyper_reality.py"),
    ]

    all_scripts = True
    for script in scripts:
        if script.exists():
            print_check(True, f"Training script: {script.name}")
        else:
            print_check(False, f"Missing: {script}")
            all_scripts = False

    return all_scripts


def check_training_infrastructure():
    """Check if training infrastructure is ready."""
    print_header("Training Infrastructure")

    required_files = [
        "src/enhancements/train_hyper_reality.py",
        "src/enhancements/hyper_reality_enhancement.py",
        "scripts/train_with_750picacho.sh",
        "scripts/quickstart_training.sh",
    ]

    all_passed = True
    for filepath in required_files:
        path = Path(filepath)
        if path.exists():
            print_check(True, f"{path.name}")
        else:
            print_check(False, f"Missing: {filepath}")
            all_passed = False

    return all_passed


def print_summary(checks):
    """Print summary and recommendations."""
    print_header("Summary")

    passed = sum(checks.values())
    total = len(checks)

    print(f"Checks passed: {passed}/{total}\n")

    if passed == total:
        print("✅ ALL CHECKS PASSED - Ready for training!\n")
        print("🚀 Recommended command:")
        print("   ./scripts/train_with_750picacho.sh")
        print("\n   Or for faster results with synthetic data:")
        print("   ./scripts/quickstart_training.sh")
        return True
    else:
        print("❌ Some checks failed - Fix issues before training\n")
        print("Failed checks:")
        for name, passed in checks.items():
            if not passed:
                print(f"   - {name}")
        print("\nRefer to docs/migrated/HOW_TO_TRAIN.md for detailed setup instructions")
        return False


def main():
    """Run all pre-flight checks."""
    print("\n" + "=" * 70)
    print("  TRANSFORMATION PORTAL - TRAINING PRE-FLIGHT CHECK")
    print("=" * 70)

    checks = {
        'Python Version': check_python_version(),
        'PyTorch & GPU': check_pytorch(),
        'Dependencies': check_dependencies(),
        'Disk Space': check_disk_space(),
        'Memory': check_memory(),
        'Training Data': check_training_data(),
        'Infrastructure': check_training_infrastructure(),
    }

    all_passed = print_summary(checks)

    print("\n📚 Documentation:")
    print("   - Quick Guide: docs/migrated/HOW_TO_TRAIN.md")
    print("   - Quick Reference: docs/migrated/TRAINING_QUICK_REFERENCE.md")
    print("   - Detailed Guide: docs/migrated/TRAINING_EXECUTION_GUIDE.md")
    print("   - Script Help: ./scripts/train_with_750picacho.sh --help")
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
