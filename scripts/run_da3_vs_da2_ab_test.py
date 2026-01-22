#!/usr/bin/env python3
"""
DA3 vs DA2 A/B Validation - Decision-Grade Comparison
======================================================

Controlled comparison of DA3-Large-1.1 vs frozen v1.0 baseline (DA2-Large-hf).

Decision Thresholds (Non-Negotiable):
- Structure scenes: ≥60% pass (from 25% baseline)
- Overall lenient: ≥95% (from 84.8% baseline)
- Texture regression: ≤2% (maintain 97%+)

If DA3 fails ANY threshold → REJECT immediately.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "lux_depth_v3"))
sys.path.insert(0, str(Path(__file__).parent / "validation/depth_quality"))

try:
    from lux_depth_v3.config import DA3Config, DA3APIConfig, ModelVariant, InferenceMode
    from lux_depth_v3.inference import DA3InferenceEngine
    from lux_depth_v3.input_manager import ImageInput

    DA3_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  DA3 import failed: {e}")
    DA3_AVAILABLE = False

try:
    from high_fidelity_depth.quality_metrics import validate_depth_quality, EdgeMetrics

    QUALITY_METRICS_AVAILABLE = True
except ImportError:
    QUALITY_METRICS_AVAILABLE = False
    EdgeMetrics = None


@dataclass
class ABTestResult:
    """A/B test result for single image."""

    image_name: str

    # DA3 results
    da3_lenient_pass: bool
    da3_strict_pass: bool
    da3_scene_type: str
    da3_edge_f1: float
    da3_inference_time: float

    # DA2 baseline (from frozen metrics)
    da2_lenient_pass: bool
    da2_strict_pass: bool
    da2_scene_type: str
    da2_edge_f1: float

    # Comparison
    lenient_improved: bool
    strict_improved: bool
    edge_f1_delta: float


def load_baseline_metrics(baseline_dir: Path) -> Dict[str, dict]:
    """Load DA2 baseline metrics from v1.0 frozen pack."""
    metrics_dir = baseline_dir / "46img_validation_results"
    baseline = {}

    for json_file in sorted(metrics_dir.glob("*_metrics.json")):
        with open(json_file) as f:
            metrics = json.load(f)

        image_name = json_file.stem.replace("_metrics", "")
        baseline[image_name] = metrics

    print(f"✅ Loaded {len(baseline)} baseline metrics from {metrics_dir}")
    return baseline


def run_da3_validation(image_paths: List[Path], output_dir: Path) -> Dict[str, dict]:
    """Run DA3 inference and quality validation."""

    if not DA3_AVAILABLE:
        raise RuntimeError("DA3 not available - cannot run validation")

    # Initialize DA3 engine with higher processing resolution for architectural detail
    # DA3 defaults to 504px, but DA2 uses 518px
    # Try 1022px (2x) for better edge preservation
    api_config = DA3APIConfig(
        process_res=1022,
        process_res_method="upper_bound_resize",
    )

    config = DA3Config(
        model_variant=ModelVariant.DA3_LARGE_V1_1,
        inference_mode=InferenceMode.MONOCULAR,
        api=api_config,
    )

    print(f"Initializing DA3 engine (model: {config.model_variant})...")
    engine = DA3InferenceEngine(config, commercial_use=False)

    results = {}

    for img_path in tqdm(image_paths, desc="DA3 Validation"):
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)

            # Run DA3 inference
            t0 = time.time()
            image_input = ImageInput(path=img_path)
            da3_result = engine.infer([image_input])
            inference_time = time.time() - t0

            # Extract depth map (shape: 1, H, W)
            depth = da3_result.depth[0]  # Remove batch dimension

            # CRITICAL FIX: DA3 outputs metric/inverse depth in narrow range (~1.06-1.10)
            # Convert to disparity (inverse depth) for proper depth characteristics
            # This matches how depth-to-3D works in CV pipelines
            depth_range = depth.max() - depth.min()
            if depth_range < 1.0 and depth.mean() > 0.5:
                # Narrow range near 1.0 suggests inverse depth or normalized metric
                # Convert to disparity to get proper foreground/background separation
                depth_disparity = 1.0 / (depth + 1e-6)
                # Normalize disparity to [0, 1]
                depth_norm = (depth_disparity - depth_disparity.min()) / (depth_disparity.max() - depth_disparity.min() + 1e-8)
            else:
                # Standard min-max normalization for relative depth
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

            # Run quality validation (same as DA2 baseline)
            if QUALITY_METRICS_AVAILABLE:
                metrics_obj = validate_depth_quality(image_np, depth_norm)
                # Convert EdgeMetrics object to dict
                if hasattr(metrics_obj, "__dict__"):
                    metrics = {k: v for k, v in metrics_obj.__dict__.items()}
                elif hasattr(metrics_obj, "_asdict"):
                    metrics = metrics_obj._asdict()
                else:
                    # Assume it's already a dict
                    metrics = metrics_obj

                # CRITICAL FIX: Compute quality gates (missing from EdgeMetrics)
                # These fields are required for A/B comparison but not included in validate_depth_quality()
                scene_type = metrics.get("scene_type", "unknown")
                edge_f1 = metrics.get("edge_f1", 0.0)
                chamfer = metrics.get("chamfer_distance", float("inf"))
                edge_width = metrics.get("edge_width", float("inf"))

                # Get scene metadata for texture-specific gates
                scene_metadata = metrics.get("scene_metadata", {})
                hf_energy = scene_metadata.get("hf_energy", 0.0)
                depth_range = scene_metadata.get("depth_range", 0.0)

                # Apply quality gates (same logic as DA2 baseline v1.0)
                if "texture" in scene_type.lower():
                    # Texture scenes: prioritize smooth high-frequency and depth variation
                    smooth_hf = hf_energy < 0.015
                    not_flat = depth_range > 0.05
                    reasonable_edges = edge_f1 >= 0.25

                    # Lenient: smooth HF AND not-flat OR reasonable edges
                    lenient_pass = (smooth_hf and not_flat) or reasonable_edges

                    # Strict: very smooth HF AND not-flat AND good edges
                    very_smooth_hf = hf_energy < 0.008
                    good_edges = edge_f1 >= 0.35
                    strict_pass = very_smooth_hf and not_flat and good_edges
                else:
                    # Structure scenes: prioritize edge quality and localization
                    # Lenient: edge F1 >= 0.35 AND chamfer < 50px
                    lenient_pass = edge_f1 >= 0.35 and chamfer < 50.0

                    # Strict: edge F1 >= 0.50 AND chamfer < 25px AND edge width < 5px
                    strict_pass = edge_f1 >= 0.50 and chamfer < 25.0 and edge_width < 5.0

                # Add quality gate results to metrics (ensure Python bool, not numpy bool_)
                metrics["lenient_pass"] = bool(lenient_pass)
                metrics["strict_pass"] = bool(strict_pass)
            else:
                # Minimal metrics if quality module unavailable
                metrics = {
                    "edge_f1": 0.0,
                    "lenient_pass": False,
                    "strict_pass": False,
                    "scene_type": "unknown",
                }

            # Add inference time
            if isinstance(metrics, dict):
                metrics["inference_time"] = inference_time
            else:
                # Create new dict if metrics is frozen
                metrics = dict(metrics)
                metrics["inference_time"] = inference_time

            # Convert numpy types to Python types for JSON serialization
            metrics_clean = {}
            for k, v in metrics.items():
                if isinstance(v, np.integer):
                    metrics_clean[k] = int(v)
                elif isinstance(v, np.floating):
                    metrics_clean[k] = float(v)
                elif isinstance(v, np.ndarray):
                    metrics_clean[k] = v.tolist()
                else:
                    metrics_clean[k] = v

            results[img_path.stem] = metrics_clean

        except Exception as e:
            print(f"❌ {img_path.name}: {e}")
            results[img_path.stem] = {
                "error": str(e),
                "lenient_pass": False,
                "strict_pass": False,
                "scene_type": "error",
                "edge_f1": 0.0,
                "inference_time": 0.0,
            }

    return results


def compare_results(da3_results: Dict[str, dict], da2_baseline: Dict[str, dict]) -> List[ABTestResult]:
    """Generate A/B comparison."""

    comparisons = []

    for image_name in sorted(da2_baseline.keys()):
        if image_name not in da3_results:
            print(f"⚠️  {image_name} not in DA3 results, skipping")
            continue

        da3 = da3_results[image_name]
        da2 = da2_baseline[image_name]

        comparison = ABTestResult(
            image_name=image_name,
            da3_lenient_pass=da3.get("lenient_pass", False),
            da3_strict_pass=da3.get("strict_pass", False),
            da3_scene_type=da3.get("scene_type", "unknown"),
            da3_edge_f1=da3.get("edge_f1", 0.0),
            da3_inference_time=da3.get("inference_time", 0.0),
            da2_lenient_pass=da2.get("lenient_pass", False),
            da2_strict_pass=da2.get("strict_pass", False),
            da2_scene_type=da2.get("scene_type", "unknown"),
            da2_edge_f1=da2.get("edge_f1", 0.0),
            lenient_improved=(da3.get("lenient_pass", False) and not da2.get("lenient_pass", False)),
            strict_improved=(da3.get("strict_pass", False) and not da2.get("strict_pass", False)),
            edge_f1_delta=da3.get("edge_f1", 0.0) - da2.get("edge_f1", 0.0),
        )

        comparisons.append(comparison)

    return comparisons


def generate_decision_report(comparisons: List[ABTestResult], output_path: Path):
    """Generate go/no-go decision report."""

    # Aggregate stats
    total = len(comparisons)

    da3_lenient = sum(1 for c in comparisons if c.da3_lenient_pass)
    da2_lenient = sum(1 for c in comparisons if c.da2_lenient_pass)

    # Scene-stratified stats
    texture_comps = [c for c in comparisons if "texture" in c.da2_scene_type.lower()]
    structure_comps = [c for c in comparisons if "structure" in c.da2_scene_type.lower()]

    texture_da3_pass = sum(1 for c in texture_comps if c.da3_lenient_pass)
    structure_da3_pass = sum(1 for c in structure_comps if c.da3_lenient_pass)

    texture_da2_pass = sum(1 for c in texture_comps if c.da2_lenient_pass)
    structure_da2_pass = sum(1 for c in structure_comps if c.da2_lenient_pass)

    # Decision logic
    da3_overall_pct = da3_lenient / total * 100
    da2_overall_pct = da2_lenient / total * 100

    structure_da3_pct = (structure_da3_pass / len(structure_comps) * 100) if structure_comps else 0
    structure_da2_pct = (structure_da2_pass / len(structure_comps) * 100) if structure_comps else 0

    texture_da3_pct = (texture_da3_pass / len(texture_comps) * 100) if texture_comps else 0
    texture_da2_pct = (texture_da2_pass / len(texture_comps) * 100) if texture_comps else 0

    texture_regression = texture_da2_pct - texture_da3_pct

    # Apply decision thresholds
    meets_overall = da3_overall_pct >= 95.0
    meets_structure = structure_da3_pct >= 60.0
    meets_texture = texture_regression <= 2.0

    if meets_overall and meets_structure and meets_texture:
        decision = "✅ ADOPT"
        rationale = "DA3 meets all decision thresholds"
    elif structure_da3_pct > structure_da2_pct and texture_regression <= 5.0:
        decision = "⚠️  DEFER"
        rationale = "DA3 shows promise but fails strict thresholds"
    else:
        decision = "❌ REJECT"
        rationale = "DA3 fails to justify upgrade complexity"

    # Generate report
    report = f"""# DA3 vs DA2 A/B Validation Report

**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**Decision**: {decision}
**Rationale**: {rationale}

---

## Summary Statistics

| Metric | DA3-Large-1.1 | DA2-Large-hf (Baseline) | Δ |
|--------|---------------|-------------------------|---|
| **Overall Lenient Pass** | {da3_lenient}/{total} ({da3_overall_pct:.1f}%) | \
{da2_lenient}/{total} ({da2_overall_pct:.1f}%) | {da3_overall_pct - da2_overall_pct:+.1f}% |
| **Texture Scenes** | {texture_da3_pass}/{len(texture_comps)} ({texture_da3_pct:.1f}%) | \
{texture_da2_pass}/{len(texture_comps)} ({texture_da2_pct:.1f}%) | {texture_da3_pct - texture_da2_pct:+.1f}% |
| **Structure Scenes** | {structure_da3_pass}/{len(structure_comps)} ({structure_da3_pct:.1f}%) | \
{structure_da2_pass}/{len(structure_comps)} ({structure_da2_pct:.1f}%) | {structure_da3_pct - structure_da2_pct:+.1f}% |

---

## Decision Thresholds

| Threshold | Target | DA3 Result | Status |
|-----------|--------|------------|--------|
| Overall Lenient | ≥95% | {da3_overall_pct:.1f}% | {"✅ PASS" if meets_overall else "❌ FAIL"} |
| Structure Scenes | ≥60% | {structure_da3_pct:.1f}% | {"✅ PASS" if meets_structure else "❌ FAIL"} |
| Texture Regression | ≤2% | {texture_regression:.1f}% | {"✅ PASS" if meets_texture else "❌ FAIL"} |

---

## Recommendation

{decision}

**Next Steps:**
"""

    if "ADOPT" in decision:
        report += """
1. Update production config to use DA3-Large-1.1
2. Run full validation on expanded dataset (100+ images)
3. Deploy to production pipeline
"""
    elif "DEFER" in decision:
        report += """
1. Analyze failure modes in structure scenes
2. Consider DA3-Giant or input-size sweep
3. Re-run A/B test after improvements
"""
    else:  # REJECT
        report += """
1. Continue with DA2-Large-hf baseline
2. Investigate input-size sweep (518 → 1022px)
3. Defer DA3 until v1.2 or architectural changes
"""

    report += f"""
---

## Detailed Results

Total comparisons: {total}

### Improvements
- Lenient upgrades: {sum(1 for c in comparisons if c.lenient_improved)}
- Strict upgrades: {sum(1 for c in comparisons if c.strict_improved)}
- Edge F1 improved: {sum(1 for c in comparisons if c.edge_f1_delta > 0)}

### Regressions
- Lenient downgrades: {sum(1 for c in comparisons if c.da2_lenient_pass and not c.da3_lenient_pass)}
- Edge F1 degraded: {sum(1 for c in comparisons if c.edge_f1_delta < -0.05)}

---

*A/B validation completed. Decision is final unless new data emerges.*
"""

    output_path.write_text(report)
    print(f"\n✅ Decision report: {output_path}")
    print(report)

    return decision


def main():
    parser = argparse.ArgumentParser(description="DA3 vs DA2 A/B Validation")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("validation_v1_baseline_pack"),
        help="Baseline metrics directory",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/validation_full"),
        help="Validation images directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/da3_ab_validation"),
        help="Output directory",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DA3 vs DA2 A/B Validation - Decision-Grade Comparison")
    print("=" * 70)
    print()

    # Load baseline
    print("Step 1: Loading DA2 baseline metrics...")
    da2_baseline = load_baseline_metrics(args.baseline_dir)

    # Get image paths (only those in baseline)
    baseline_names = set(da2_baseline.keys())
    all_images = list(args.image_dir.glob("*.jpg"))
    image_paths = [p for p in all_images if p.stem in baseline_names]

    print(f"✅ Found {len(image_paths)} images matching baseline")

    # Run DA3 validation
    print("\nStep 2: Running DA3 validation...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    da3_results = run_da3_validation(image_paths, args.output_dir)

    # Save DA3 results
    da3_results_path = args.output_dir / "da3_metrics.json"
    with open(da3_results_path, "w") as f:
        json.dump(da3_results, f, indent=2)
    print(f"✅ DA3 results saved: {da3_results_path}")

    # Compare
    print("\nStep 3: Generating A/B comparison...")
    comparisons = compare_results(da3_results, da2_baseline)

    # Save comparisons
    comparisons_path = args.output_dir / "ab_comparisons.json"
    with open(comparisons_path, "w") as f:
        json.dump([asdict(c) for c in comparisons], f, indent=2)
    print(f"✅ Comparisons saved: {comparisons_path}")

    # Generate decision report
    print("\nStep 4: Generating decision report...")
    decision_path = args.output_dir / "DA3_DECISION_REPORT.md"
    decision = generate_decision_report(comparisons, decision_path)

    print("\n" + "=" * 70)
    print(f"FINAL DECISION: {decision}")
    print("=" * 70)


if __name__ == "__main__":
    main()
