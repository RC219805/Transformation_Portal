#!/usr/bin/env python3
"""
Stage 6 Golden Baseline A/B: SegFormer-only vs FUSED (EfficientSAM)

This script:
- Uses the EXACT same pipeline invocation as cli.py
- Runs Kitchen APEX baseline + canary
- Extracts fusion stats if available
- Generates structured comparison output
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent to path so we can import lux_depth_v2
sys.path.insert(0, str(Path(__file__).parent.parent))

from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.logging_utils import setup_logging
from lux_depth_v2.pipeline import LuxPipelineV2


def run_single(
    input_path: Path,
    output_dir: Path,
    preset: Preset,
    run_name: str,
) -> Dict[str, Any]:
    """Run pipeline once and return structured result."""
    
    logger = setup_logging("INFO")
    logger.info(f"=== {run_name} ===")
    logger.info(f"Preset: {preset.value}")
    
    # Create config using same pattern as cli.py
    cfg = PipelineConfig(
        input_dir=None,
        depth_dir=None,
        output_dir=output_dir,
        preset=preset,
        device="auto",
        precision="fp16",
        upscale=4,
        upscaler_backend="torch",
    )
    
    # Initialize pipeline
    pipe = LuxPipelineV2(cfg, logger=logger)
    
    # Run single image (same call as cli.py line 356)
    result = pipe.process_one(input_path)
    
    logger.info(f"Status: {result.get('status', 'unknown')}")
    
    return {
        "run_name": run_name,
        "preset": preset.value,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "status": result.get("status"),
        "pipeline_result": result,
    }


def main() -> int:
    """Run Kitchen A/B test."""
    
    # Inputs
    kitchen_path = Path("assets/phase2_bench/750Picacho_Kitchen_Ultimate.tif")
    
    if not kitchen_path.exists():
        print(f"ERROR: Kitchen benchmark image not found: {kitchen_path}")
        print("Expected: assets/phase2_bench/750Picacho_Kitchen_Ultimate.tif")
        return 1
    
    # Outputs
    stage6_root = Path("outputs/stage6_smoke")
    stage6_root.mkdir(parents=True, exist_ok=True)
    
    baseline_out = stage6_root / "kitchen_A_baseline"
    canary_out = stage6_root / "kitchen_B_efficientsam"
    
    print("\n" + "="*60)
    print("Stage 6 Smoke Test: Kitchen APEX Baseline vs EfficientSAM")
    print("="*60 + "\n")
    
    results = []
    
    # Run A: Baseline APEX (SegFormer-only)
    try:
        baseline_result = run_single(
            input_path=kitchen_path,
            output_dir=baseline_out,
            preset=Preset.INTERIOR_LUXURY_APEX_QUALITY,
            run_name="A_BASELINE_APEX",
        )
        results.append(baseline_result)
    except Exception as e:
        print(f"ERROR in baseline run: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run B: Canary APEX + EfficientSAM
    try:
        canary_result = run_single(
            input_path=kitchen_path,
            output_dir=canary_out,
            preset=Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM,
            run_name="B_CANARY_EFFICIENTSAM",
        )
        results.append(canary_result)
    except Exception as e:
        print(f"ERROR in canary run: {e}")
        import traceback
        traceback.print_exc()
        # Continue to write what we have
    
    # Write summary
    summary = {
        "test_name": "Stage 6 Smoke: Kitchen APEX A/B",
        "input_image": str(kitchen_path),
        "runs": results,
    }
    
    summary_path = stage6_root / "stage6_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    
    print("\n" + "="*60)
    print(f"✓ Summary written: {summary_path}")
    print("="*60 + "\n")
    
    # Quick status
    for r in results:
        print(f"{r['run_name']:30} | {r['status']}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
