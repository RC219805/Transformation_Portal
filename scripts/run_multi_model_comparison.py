#!/usr/bin/env python3
"""
Multi-Model Depth Comparison Framework

Runs comprehensive A/B testing across multiple Depth Anything V2 variants:
- Relative depth models (Large, Giant)
- Metric depth models (Indoor, Outdoor)
- Input size sweeps per model
- Statistical comparison and CI-friendly reporting

Usage:
    python scripts/run_multi_model_comparison.py \
        --input-dir data/validation_full \
        --labels data/validation_full/labels.csv \
        --models all \
        --sweep-sizes 518 768 896 1022 \
        --output-root outputs/model_comparison

Models supported:
    - depth-anything/Depth-Anything-V2-Large-hf (baseline relative)
    - depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf
    - depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf

Optional (requires high VRAM):
    - LiheYoung/depth-anything-large-hf (v1 for comparison)
"""

import argparse
import subprocess
import json
import os
import sys
import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Model registry with metadata
MODEL_REGISTRY = {
    "DA2_Large": {
        "hf_id": "depth-anything/Depth-Anything-V2-Large-hf",
        "type": "relative",
        "description": "High-quality relative depth (baseline)",
        "vram_gb": 12,
        "default_input_size": 518,
    },
    "DA2_Metric_Indoor": {
        "hf_id": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "type": "metric",
        "description": "Absolute depth in meters (indoor, 0-20m)",
        "vram_gb": 12,
        "default_input_size": 518,
    },
    "DA2_Metric_Outdoor": {
        "hf_id": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "type": "metric",
        "description": "Absolute depth in meters (outdoor, 0-80m)",
        "vram_gb": 12,
        "default_input_size": 518,
    },
    "DA2_Giant": {
        "hf_id": "depth-anything/Depth-Anything-V2-Giant-hf",
        "type": "relative",
        "description": "Maximum capacity relative depth (1.3B params)",
        "vram_gb": 24,
        "default_input_size": 518,
        "requires_high_vram": True,
    },
}


def timestamp_tag():
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def get_git_sha():
    """Get short git commit SHA for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except:
        return "unknown"


def run_script(cmd, cwd=None, capture=False, timeout=None):
    """Run shell command with optional capture."""
    print(f"\n▶ Running: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"⚠️  Command timed out after {timeout}s")
        return None
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return None


def validate_vram(model_key: str) -> bool:
    """Check if system has enough VRAM for model."""
    model_info = MODEL_REGISTRY[model_key]
    required_gb = model_info.get("vram_gb", 12)

    try:
        import torch

        if torch.cuda.is_available():
            available_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if available_gb < required_gb:
                print(f"⚠️  {model_key} requires {required_gb}GB VRAM, only {available_gb:.1f}GB available")
                return False
        elif torch.backends.mps.is_available():
            # MPS doesn't expose VRAM directly; assume M-series has enough
            pass
        else:
            print(f"⚠️  No GPU detected; {model_key} may run slowly on CPU")
    except ImportError:
        print("⚠️  PyTorch not available for VRAM check")

    return True


def run_depth_validation_for_model(
    model_key: str,
    input_dir: Path,
    labels_path: Path,
    output_dir: Path,
    sweep_sizes: List[int],
    structure_subset_dir: Path = None,
) -> Dict[str, Any]:
    """
    Run full validation pipeline for a single model across all input sizes.
    Returns summary metrics.
    """
    model_info = MODEL_REGISTRY[model_key]
    model_id = model_info["hf_id"]
    model_type = model_info["type"]

    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_key}")
    print(f"HF ID: {model_id}")
    print(f"Type: {model_type}")
    print(f"{'=' * 70}\n")

    results = {
        "model_key": model_key,
        "model_id": model_id,
        "model_type": model_type,
        "sweep_results": {},
        "overall_best": {},
    }

    # Run validation for each input size
    for input_size in sweep_sizes:
        size_tag = f"input_{input_size}"
        size_output_dir = output_dir / size_tag
        size_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n▶ Running {model_key} @ input_size={input_size}")

        # Call the main validation runner with model override
        val_cmd = (
            f"python3 scripts/production_depth_validation_fixed.py "
            f"--input-dir {input_dir} "
            f"--output-dir {size_output_dir} "
            f"--model-id {model_id} "
            f"--input-size {input_size} "
            f"--labels {labels_path}"
        )

        result = run_script(val_cmd, capture=True, timeout=3600)

        if result and result.returncode == 0:
            # Parse validation_report.json if it exists
            report_path = size_output_dir / "validation_report.json"
            if report_path.exists():
                with open(report_path) as f:
                    report_data = json.load(f)

                results["sweep_results"][size_tag] = {
                    "input_size": input_size,
                    "lenient_pass": report_data.get("lenient_pass", 0),
                    "strict_pass": report_data.get("strict_pass", 0),
                    "lenient_pass_rate": report_data.get("lenient_pass_rate", 0.0),
                    "strict_pass_rate": report_data.get("strict_pass_rate", 0.0),
                    "total_images": report_data.get("total_images", 0),
                    "output_dir": str(size_output_dir),
                    "success": True,
                }
            else:
                results["sweep_results"][size_tag] = {
                    "input_size": input_size,
                    "success": False,
                    "error": "validation_report.json not found",
                }
        else:
            results["sweep_results"][size_tag] = {
                "input_size": input_size,
                "success": False,
                "error": result.stderr if result else "timeout or crash",
            }

    # Determine best input size by lenient pass rate
    best_size = None
    best_rate = 0.0
    for size_tag, size_result in results["sweep_results"].items():
        if size_result.get("success") and size_result.get("lenient_pass_rate", 0) > best_rate:
            best_rate = size_result["lenient_pass_rate"]
            best_size = size_result["input_size"]

    results["overall_best"] = {
        "input_size": best_size,
        "lenient_pass_rate": best_rate,
    }

    return results


def generate_comparison_report(
    all_results: Dict[str, Any],
    output_dir: Path,
    labels_path: Path,
) -> None:
    """
    Generate cross-model comparison CSVs and JSON summary.
    """
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON REPORT")
    print("=" * 70 + "\n")

    # Build comparison table
    rows = []
    for model_key, model_results in all_results.items():
        for size_tag, size_result in model_results["sweep_results"].items():
            if not size_result.get("success"):
                continue

            rows.append(
                {
                    "model": model_key,
                    "model_type": model_results["model_type"],
                    "input_size": size_result["input_size"],
                    "lenient_pass": size_result.get("lenient_pass", 0),
                    "strict_pass": size_result.get("strict_pass", 0),
                    "lenient_rate": size_result.get("lenient_pass_rate", 0.0),
                    "strict_rate": size_result.get("strict_pass_rate", 0.0),
                    "total_images": size_result.get("total_images", 0),
                }
            )

    df = pd.DataFrame(rows)

    # Save overall comparison
    overall_csv = output_dir / "comparison_overall.csv"
    df.to_csv(overall_csv, index=False)
    print(f"✅ Saved: {overall_csv}")

    # Pivot by model
    pivot_csv = output_dir / "comparison_by_model.csv"
    pivot = df.pivot_table(
        index="model",
        columns="input_size",
        values=["lenient_rate", "strict_rate"],
        aggfunc="mean",
    )
    pivot.to_csv(pivot_csv)
    print(f"✅ Saved: {pivot_csv}")

    # Best models summary
    best_models = df.loc[df.groupby("model")["lenient_rate"].idxmax()]
    best_csv = output_dir / "best_per_model.csv"
    best_models.to_csv(best_csv, index=False)
    print(f"✅ Saved: {best_csv}")

    print("\n📊 Best Performing Configurations:")
    print(best_models[["model", "input_size", "lenient_rate", "strict_rate"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Multi-model depth validation and A/B testing framework")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with validation images",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="CSV with ground truth scene labels",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["DA2_Large", "DA2_Metric_Indoor"],
        help="Model keys to test (from MODEL_REGISTRY)",
    )
    parser.add_argument(
        "--sweep-sizes",
        nargs="+",
        type=int,
        default=[518, 768, 896, 1022],
        help="Input sizes for sweep",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/model_comparison"),
        help="Root output directory",
    )
    parser.add_argument(
        "--structure-subset-dir",
        type=Path,
        help="Optional: directory with structure-dominated subset",
    )
    parser.add_argument(
        "--skip-vram-check",
        action="store_true",
        help="Skip VRAM validation (use with caution)",
    )

    args = parser.parse_args()

    # Expand "all" models
    if "all" in args.models:
        args.models = [k for k in MODEL_REGISTRY.keys() if not MODEL_REGISTRY[k].get("requires_high_vram")]

    # Create versioned output directory
    ts = timestamp_tag()
    sha = get_git_sha()
    run_dir = args.output_root / f"run_{ts}_{sha}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📌 Multi-Model Validation Run")
    print(f"   Output: {run_dir}")
    print(f"   Models: {', '.join(args.models)}")
    print(f"   Sizes:  {args.sweep_sizes}")
    print(f"   Commit: {sha}\n")

    # Validate models
    valid_models = []
    for model_key in args.models:
        if model_key not in MODEL_REGISTRY:
            print(f"⚠️  Unknown model: {model_key}, skipping")
            continue

        if not args.skip_vram_check and not validate_vram(model_key):
            print(f"⚠️  Skipping {model_key} due to insufficient VRAM")
            continue

        valid_models.append(model_key)

    if not valid_models:
        print("❌ No valid models to test. Exiting.")
        sys.exit(1)

    # Run validation for each model
    all_results = {}
    for model_key in valid_models:
        model_output_dir = run_dir / f"model_{model_key}"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = run_depth_validation_for_model(
                model_key=model_key,
                input_dir=args.input_dir,
                labels_path=args.labels,
                output_dir=model_output_dir,
                sweep_sizes=args.sweep_sizes,
                structure_subset_dir=args.structure_subset_dir,
            )
            all_results[model_key] = results
        except Exception as e:
            print(f"❌ Failed to run {model_key}: {e}")
            all_results[model_key] = {
                "model_key": model_key,
                "error": str(e),
                "sweep_results": {},
            }

    # Generate comparison report
    generate_comparison_report(
        all_results=all_results,
        output_dir=run_dir,
        labels_path=args.labels,
    )

    # Save full JSON summary
    summary_json = run_dir / "model_comparison_summary.json"
    summary = {
        "timestamp": ts,
        "git_sha": sha,
        "input_dir": str(args.input_dir),
        "labels": str(args.labels),
        "models_tested": valid_models,
        "sweep_sizes": args.sweep_sizes,
        "results": all_results,
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Full summary saved: {summary_json}")
    print(f"\n🎉 Multi-model comparison complete!")
    print(f"   Results: {run_dir}")


if __name__ == "__main__":
    main()
