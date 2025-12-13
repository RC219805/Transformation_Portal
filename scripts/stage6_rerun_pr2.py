#!/usr/bin/env python3
"""
Stage 6 A/B Rerun with PR-2 Validation

Compares current PR-2 results against previous Stage 6 baseline.
Focus areas:
- fusion_applied rate per class
- IoU distributions (especially Kitchen/Pool)
- ROI usage / skip reasons
- runtime deltas
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

# Previous Stage 6 results (from your last run)
PREVIOUS_RESULTS = {
    "interior_bedroom": {
        "canary": {
            "fusion_applied_classes": ["glass"],
            "glass_iou": 0.431,
            "runtime_sec": 61.0,
        }
    },
    "exterior_aerial": {
        "canary": {
            "fusion_applied_classes": ["foliage"],
            "foliage_iou": 0.383,
            "runtime_sec": 56.2,
        }
    },
    "interior_kitchen": {
        "canary": {
            "fusion_applied_classes": [],
            "glass_iou": 0.297,  # rejected by gate
            "runtime_sec": 62.3,
        }
    },
    "exterior_pool": {
        "canary": {
            "fusion_applied_classes": [],
            "foliage_iou": 0.230,  # rejected by gate
            "runtime_sec": 43.9,
        }
    },
    "interior_bathroom": {
        "canary": {
            "error": "OOM",
        }
    },
}


def main() -> int:
    repo_root = Path(__file__).parent.parent
    output_dir = repo_root / "outputs" / "stage6_pr2_rerun"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Stage 6 PR-2 Rerun ===")
    print(f"Output: {output_dir}")
    print()

    # Run the golden baseline script
    script = repo_root / "scripts" / "stage6_ab_golden_baseline_v2.py"
    
    print("Running A/B test suite...")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print("FAILED:")
        print(result.stdout)
        print(result.stderr)
        return 1
    
    print(result.stdout)
    
    # Parse results and compare
    summary_path = repo_root / "outputs" / "stage6_ab" / "stage6_ab_summary.json"
    if not summary_path.exists():
        print(f"ERROR: Summary not found at {summary_path}")
        return 1
    
    with open(summary_path) as f:
        new_results = json.load(f)
    
    # Generate comparison report
    comparison = compare_results(new_results)
    
    comparison_path = output_dir / "pr2_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✅ Comparison saved: {comparison_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PR-2 IMPACT SUMMARY")
    print("=" * 60)
    
    for scene, delta in comparison.items():
        if isinstance(delta, dict) and "improvements" in delta:
            print(f"\n{scene}:")
            for imp in delta["improvements"]:
                print(f"  ✅ {imp}")
            for reg in delta.get("regressions", []):
                print(f"  ⚠️  {reg}")
    
    return 0


def compare_results(new_results: Dict) -> Dict:
    """Compare new results against previous baseline."""
    comparison = {}
    
    # Iterate through scenes we have previous data for
    for scene_key in PREVIOUS_RESULTS:
        if scene_key not in new_results:
            continue
        
        prev = PREVIOUS_RESULTS[scene_key].get("canary", {})
        new = new_results[scene_key].get("canary", {})
        
        improvements = []
        regressions = []
        
        # Check for OOM resolution
        if "error" in prev and "error" not in new:
            improvements.append("OOM issue resolved")
        
        # Check fusion application rate
        prev_applied = set(prev.get("fusion_applied_classes", []))
        new_applied = set(new.get("segmentation_v3", {}).get("per_class", {}).keys())
        
        if len(new_applied) > len(prev_applied):
            improvements.append(
                f"Fusion now applies to {len(new_applied)} classes (was {len(prev_applied)})"
            )
        
        # Check IoU improvements for known problematic classes
        for cls in ["glass", "water", "foliage"]:
            prev_iou_key = f"{cls}_iou"
            if prev_iou_key in prev:
                prev_iou = prev[prev_iou_key]
                new_iou = (
                    new.get("segmentation_v3", {})
                    .get("per_class", {})
                    .get(cls, {})
                    .get("iou_base_vs_refined", 0.0)
                )
                
                if new_iou > prev_iou + 0.05:  # meaningful improvement
                    improvements.append(
                        f"{cls} IoU: {prev_iou:.3f} → {new_iou:.3f} (+{new_iou-prev_iou:.3f})"
                    )
                elif new_iou < prev_iou - 0.05:  # regression
                    regressions.append(
                        f"{cls} IoU: {prev_iou:.3f} → {new_iou:.3f} ({new_iou-prev_iou:.3f})"
                    )
        
        # Check runtime delta
        prev_runtime = prev.get("runtime_sec", 0)
        new_runtime = new.get("runtime_sec", 0)
        if new_runtime > 0 and prev_runtime > 0:
            delta_pct = ((new_runtime - prev_runtime) / prev_runtime) * 100
            if abs(delta_pct) > 10:
                key = "improvements" if delta_pct < 0 else "regressions"
                (improvements if delta_pct < 0 else regressions).append(
                    f"Runtime: {prev_runtime:.1f}s → {new_runtime:.1f}s ({delta_pct:+.1f}%)"
                )
        
        comparison[scene_key] = {
            "improvements": improvements,
            "regressions": regressions,
            "prev": prev,
            "new": new,
        }
    
    return comparison


if __name__ == "__main__":
    raise SystemExit(main())
