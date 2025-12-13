#!/usr/bin/env python3
"""
Stage 6 Golden Baseline A/B Test (Production-Ready)

Runs A/B comparison between:
- Baseline APEX (SegFormer-only)
- Canary APEX + EfficientSAM

Uses LuxPipelineV2 API directly for maximum fidelity.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add lux_depth_v2 to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.pipeline import LuxPipelineV2


BENCHMARK_SET = {
    "interior_kitchen_750": {
        "path": "assets/phase2_bench/interior_kitchen_750.tiff",
        "baseline_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY,
        "canary_preset": Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
        "scene_type": "interior",
    },
    "exterior_pool_750": {
        "path": "assets/phase2_bench/exterior_pool_750.tiff",
        "baseline_preset": Preset.EXTERIOR_POOL_APEX_QUALITY,
        "canary_preset": Preset.EXTERIOR_POOL_APEX_QUALITY_EFFICIENTSAM,
        "scene_type": "exterior",
    },
    # Add more benchmark images as needed
}


def run_ab_test(benchmark_name: str, benchmark_config: dict, output_root: Path) -> dict:
    """Run A/B test for a single benchmark image."""
    input_path = Path(benchmark_config["path"])
    if not input_path.exists():
        print(f"⚠️  Skipping {benchmark_name}: input not found at {input_path}")
        return {"status": "skipped", "reason": "input_missing"}

    baseline_preset = benchmark_config["baseline_preset"]
    canary_preset = benchmark_config["canary_preset"]

    results = {
        "benchmark": benchmark_name,
        "scene_type": benchmark_config["scene_type"],
        "input_path": str(input_path),
        "runs": {},
    }

    # Run baseline (SegFormer-only APEX)
    print(f"\n{'='*60}")
    print(f"Running BASELINE: {benchmark_name} → {baseline_preset.value}")
    print(f"{'='*60}")
    
    baseline_out = output_root / f"{benchmark_name}_A_baseline"
    baseline_out.mkdir(parents=True, exist_ok=True)

    cfg_baseline = PipelineConfig()
    cfg_baseline.preset = baseline_preset
    cfg_baseline.apply_preset()
    cfg_baseline.output_dir = str(baseline_out)
    cfg_baseline.write_outputs = True

    try:
        t0 = time.time()
        pipeline_baseline = LuxPipelineV2(cfg_baseline)
        report_baseline = pipeline_baseline.process_one(input_path)
        baseline_elapsed = time.time() - t0

        results["runs"]["baseline"] = {
            "preset": baseline_preset.value,
            "status": report_baseline.get("status", "unknown"),
            "elapsed_s": baseline_elapsed,
            "timing_s": report_baseline.get("timing_s", 0.0),
            "report_path": str(baseline_out / f"{input_path.stem}_report.json"),
            "segmentation_v3": report_baseline.get("segmentation_v3"),
        }
        print(f"✓ Baseline complete in {baseline_elapsed:.2f}s")
    except Exception as e:
        print(f"✗ Baseline failed: {e}")
        results["runs"]["baseline"] = {"status": "error", "error": str(e)}

    # Run canary (EfficientSAM fusion enabled)
    print(f"\n{'='*60}")
    print(f"Running CANARY: {benchmark_name} → {canary_preset.value}")
    print(f"{'='*60}")
    
    canary_out = output_root / f"{benchmark_name}_B_efficientsam"
    canary_out.mkdir(parents=True, exist_ok=True)

    cfg_canary = PipelineConfig()
    cfg_canary.preset = canary_preset
    cfg_canary.apply_preset()
    cfg_canary.output_dir = str(canary_out)
    cfg_canary.write_outputs = True

    try:
        t0 = time.time()
        pipeline_canary = LuxPipelineV2(cfg_canary)
        report_canary = pipeline_canary.process_one(input_path)
        canary_elapsed = time.time() - t0

        results["runs"]["canary"] = {
            "preset": canary_preset.value,
            "status": report_canary.get("status", "unknown"),
            "elapsed_s": canary_elapsed,
            "timing_s": report_canary.get("timing_s", 0.0),
            "report_path": str(canary_out / f"{input_path.stem}_report.json"),
            "segmentation_v3": report_canary.get("segmentation_v3"),
        }
        print(f"✓ Canary complete in {canary_elapsed:.2f}s")
        
        # Stage 6.5 validation: canary must have segmentation_v3
        if not report_canary.get("segmentation_v3"):
            print("⚠️  WARNING: Canary run missing segmentation_v3 stats!")
        else:
            v3_report = report_canary["segmentation_v3"]
            print(f"\n📊 EfficientSAM V3 Stats:")
            print(f"   Backend: {v3_report.get('backend_v3')}")
            print(f"   Fusion Mode: {v3_report.get('fusion_mode')}")
            print(f"   Model: {v3_report.get('model')}")
            per_class = v3_report.get("per_class", {})
            for cls, stats in per_class.items():
                print(f"   {cls}: IoU={stats.get('iou_base_vs_refined', 0):.3f}, Applied={stats.get('fusion_applied', 0)}")

    except Exception as e:
        print(f"✗ Canary failed: {e}")
        results["runs"]["canary"] = {"status": "error", "error": str(e)}

    return results


def main() -> int:
    output_root = Path("outputs/stage6_ab")
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "stage": "6_ab_golden_baseline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "benchmarks": {},
    }

    for bench_name, bench_cfg in BENCHMARK_SET.items():
        result = run_ab_test(bench_name, bench_cfg, output_root)
        summary["benchmarks"][bench_name] = result

    # Write summary
    summary_path = output_root / "stage6_ab_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*60}")
    print(f"✓ Stage 6 A/B complete. Summary written to:")
    print(f"  {summary_path}")
    print(f"{'='*60}\n")

    # Check for failures
    any_failures = False
    any_missing_v3 = False
    
    for bench_name, result in summary["benchmarks"].items():
        runs = result.get("runs", {})
        if runs.get("baseline", {}).get("status") == "error":
            print(f"✗ Baseline failed for {bench_name}")
            any_failures = True
        if runs.get("canary", {}).get("status") == "error":
            print(f"✗ Canary failed for {bench_name}")
            any_failures = True
        if runs.get("canary", {}).get("segmentation_v3") is None:
            print(f"⚠️  Missing segmentation_v3 for {bench_name}")
            any_missing_v3 = True

    if any_failures:
        return 1
    if any_missing_v3:
        print("\n⚠️  Some canary runs missing segmentation_v3 stats (check implementation)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
