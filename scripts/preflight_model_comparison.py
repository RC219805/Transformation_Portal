#!/usr/bin/env python3
"""
Pre-flight check for multi-model validation suite.

Validates:
- Dataset availability and structure
- Python dependencies
- GPU/VRAM availability
- HuggingFace model accessibility
- Disk space

Usage:
    python scripts/preflight_model_comparison.py [--mode quick|full]
"""

import argparse
import sys
from pathlib import Path
import subprocess
import json


def check_dataset(input_dir: Path, labels_path: Path) -> bool:
    """Check dataset exists and is properly structured."""
    print("\n▶ Checking dataset...")
    
    if not input_dir.exists():
        print(f"  ❌ Input directory not found: {input_dir}")
        return False
    
    if not labels_path.exists():
        print(f"  ❌ Labels file not found: {labels_path}")
        return False
    
    # Count images
    image_exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_count = sum(len(list(input_dir.glob(ext))) for ext in image_exts)
    
    # Check labels CSV
    try:
        import pandas as pd
        df = pd.read_csv(labels_path)
        
        if "filename" not in df.columns or "scene_type" not in df.columns:
            print(f"  ❌ Labels CSV missing required columns (filename, scene_type)")
            return False
        
        label_count = len(df)
        
        print(f"  ✅ Dataset OK: {image_count} images, {label_count} labels")
        
        # Warn if counts don't match
        if image_count != label_count:
            print(f"  ⚠️  Image count ({image_count}) != label count ({label_count})")
        
        # Show scene type distribution
        scene_counts = df["scene_type"].value_counts()
        print(f"  📊 Scene distribution:")
        for scene_type, count in scene_counts.items():
            print(f"     - {scene_type}: {count}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error reading labels: {e}")
        return False


def check_python_deps() -> bool:
    """Check required Python packages."""
    print("\n▶ Checking Python dependencies...")
    
    required = [
        "torch",
        "transformers",
        "scipy",
        "scikit-learn",
        "pandas",
        "numpy",
        "Pillow",
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"\n  Install missing packages:")
        print(f"    pip install {' '.join(missing)}")
        return False
    
    return True


def check_gpu() -> dict:
    """Check GPU/VRAM availability."""
    print("\n▶ Checking GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✅ CUDA available: {device_name}")
            print(f"  📊 VRAM: {vram_gb:.1f}GB")
            
            return {
                "available": True,
                "backend": "cuda",
                "vram_gb": vram_gb,
                "device_name": device_name,
            }
        
        elif torch.backends.mps.is_available():
            print(f"  ✅ MPS (Apple Silicon) available")
            print(f"  ⚠️  VRAM not directly queryable on MPS")
            
            return {
                "available": True,
                "backend": "mps",
                "vram_gb": None,
            }
        
        else:
            print(f"  ⚠️  No GPU detected, will use CPU (slow)")
            return {
                "available": False,
                "backend": "cpu",
            }
    
    except Exception as e:
        print(f"  ❌ Error checking GPU: {e}")
        return {"available": False, "error": str(e)}


def check_hf_models(model_ids: list) -> bool:
    """Check HuggingFace model accessibility."""
    print("\n▶ Checking HuggingFace models...")
    
    try:
        from transformers import AutoModel
        
        for model_id in model_ids:
            try:
                # Try to load config only (fast check)
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_id)
                print(f"  ✅ {model_id}")
            except Exception as e:
                print(f"  ❌ {model_id}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error checking models: {e}")
        return False


def check_disk_space(min_gb: float = 10.0) -> bool:
    """Check available disk space."""
    print("\n▶ Checking disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        used_pct = (used / total) * 100
        
        print(f"  📊 Disk: {free_gb:.1f}GB free / {total_gb:.1f}GB total ({used_pct:.1f}% used)")
        
        if free_gb < min_gb:
            print(f"  ⚠️  Low disk space (< {min_gb}GB)")
            return False
        
        print(f"  ✅ Sufficient space")
        return True
        
    except Exception as e:
        print(f"  ❌ Error checking disk: {e}")
        return False


def check_git_status() -> dict:
    """Check git status for reproducibility."""
    print("\n▶ Checking git status...")
    
    try:
        # Get SHA
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        
        # Check for uncommitted changes
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        
        clean = len(status) == 0
        
        print(f"  📌 Commit: {sha}")
        
        if clean:
            print(f"  ✅ Clean working tree")
        else:
            print(f"  ⚠️  Uncommitted changes present")
            print(f"     (results may not be reproducible)")
        
        return {
            "sha": sha,
            "clean": clean,
        }
        
    except Exception as e:
        print(f"  ⚠️  Git check failed: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight check for multi-model validation suite"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Validation mode (quick or full)",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip HuggingFace model accessibility check",
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Multi-Model Validation Pre-Flight Check")
    print("="*70)
    print(f"Mode: {args.mode}")
    
    # Set paths based on mode
    if args.mode == "quick":
        input_dir = Path("data/validation_quick")
        labels_path = Path("data/validation_quick/labels.csv")
        models_to_check = [
            "depth-anything/Depth-Anything-V2-Large-hf",
            "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        ]
    else:
        input_dir = Path("data/validation_full")
        labels_path = Path("data/validation_full/labels.csv")
        models_to_check = [
            "depth-anything/Depth-Anything-V2-Large-hf",
            "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
            "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        ]
    
    # Run checks
    checks_passed = []
    
    checks_passed.append(("Dataset", check_dataset(input_dir, labels_path)))
    checks_passed.append(("Python Dependencies", check_python_deps()))
    
    gpu_info = check_gpu()
    checks_passed.append(("GPU", gpu_info.get("available", False)))
    
    if not args.skip_model_check:
        checks_passed.append(("HF Models", check_hf_models(models_to_check)))
    
    checks_passed.append(("Disk Space", check_disk_space()))
    
    git_info = check_git_status()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = all(status for _, status in checks_passed)
    
    for check_name, status in checks_passed:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {check_name}")
    
    print("")
    
    if all_passed:
        print("🎉 All checks passed! Ready to run validation suite.")
        print("")
        print("Next steps:")
        print(f"  ./scripts/run_model_comparison_suite.sh {args.mode}")
        return 0
    else:
        print("❌ Some checks failed. Please resolve issues before running.")
        print("")
        print("See error messages above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
