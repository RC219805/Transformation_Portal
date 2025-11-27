"""
750 Picacho Lane - Quick Visual Comparison Tool
Generates side-by-side comparisons and quality metrics

Usage:
    python compare_outputs.py --scene Aerial [--show]
"""

import argparse
from pathlib import Path
from typing import Tuple
import sys

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("⚠️  PIL and numpy required: pip install Pillow numpy")
    sys.exit(1)


SCENES = ["Aerial", "GreatRoom", "Kitchen", "Pool", "PrimaryBathroom", "PrimaryBedroom"]


def analyze_channels(img: Image.Image) -> dict:
    """Analyze RGB channel distribution"""
    arr = np.array(img)

    if len(arr.shape) != 3 or arr.shape[2] < 3:
        return {"error": "Not RGB image"}

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    return {
        "R": {"mean": float(np.mean(r)), "std": float(np.std(r)), "max": int(np.max(r))},
        "G": {"mean": float(np.mean(g)), "std": float(np.std(g)), "max": int(np.max(g))},
        "B": {"mean": float(np.mean(b)), "std": float(np.std(b)), "max": int(np.max(b))},
        "brightness": float(np.mean(arr)),
        "contrast": float(np.std(arr)),
        "is_neutral_gray": abs(np.mean(r) - np.mean(g)) < 1 and abs(np.mean(g) - np.mean(b)) < 1
    }


def assess_quality(metrics: dict, scene_type: str) -> Tuple[int, str]:
    """
    Assess quality score based on metrics

    Returns:
        (score, reason) tuple
    """
    score = 100
    issues = []

    # Check for neutral gray contamination
    if metrics.get("is_neutral_gray", False):
        score -= 25
        issues.append("Neutral gray contamination (CRITICAL)")

    # Check RGB channel separation
    r_mean = metrics["R"]["mean"]
    g_mean = metrics["G"]["mean"]
    b_mean = metrics["B"]["mean"]

    if max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean)) < 5:
        score -= 15
        issues.append("Poor channel separation")

    # Check dynamic range
    contrast = metrics.get("contrast", 0)
    if contrast < 50:
        score -= 10
        issues.append(f"Low contrast ({contrast:.1f})")

    # Scene-specific checks
    if scene_type == "pool" and b_mean <= max(r_mean, g_mean):
        score -= 15
        issues.append("Pool lacks blue water character")

    if scene_type in ["interior", "bathroom", "bedroom"] and r_mean <= g_mean:
        score -= 8
        issues.append("Lacks warm interior character")

    # Brightness checks
    brightness = metrics.get("brightness", 128)
    if scene_type == "aerial" and brightness < 90:
        score -= 10
        issues.append(f"Too dark for aerial ({brightness:.1f})")
    elif scene_type in ["interior", "kitchen"] and brightness < 120:
        score -= 8
        issues.append(f"Too dark for interior ({brightness:.1f})")

    reason = "; ".join(issues) if issues else "Good quality"
    return max(0, score), reason


def compare_scene(scene_name: str, base_dir: Path) -> dict:
    """Compare all versions of a scene"""

    scene_type = {
        "Aerial": "aerial",
        "Pool": "pool",
        "GreatRoom": "interior",
        "Kitchen": "kitchen",
        "PrimaryBathroom": "bathroom",
        "PrimaryBedroom": "bedroom"
    }.get(scene_name, "unknown")

    results = {}

    # Source
    source_path = base_dir / "JPEGs" / f"750Picacho_{scene_name}.jpg"
    if source_path.exists():
        img = Image.open(source_path)
        metrics = analyze_channels(img)
        score, reason = assess_quality(metrics, scene_type)
        results["Source"] = {
            "path": source_path,
            "metrics": metrics,
            "score": score,
            "reason": reason
        }

    # Final Production
    final_path = base_dir / "Final_Production" / f"750Picacho_{scene_name}_luxury.tif"
    if final_path.exists():
        img = Image.open(final_path)
        metrics = analyze_channels(img)
        score, reason = assess_quality(metrics, scene_type)
        results["Final Production"] = {
            "path": final_path,
            "metrics": metrics,
            "score": score,
            "reason": reason
        }

    # Ultimate Quality
    ultimate_path = base_dir / "Ultimate_Quality" / f"750Picacho_{scene_name}_ultimate.tif"
    if ultimate_path.exists():
        img = Image.open(ultimate_path)
        metrics = analyze_channels(img)
        score, reason = assess_quality(metrics, scene_type)
        results["Ultimate Quality"] = {
            "path": ultimate_path,
            "metrics": metrics,
            "score": score,
            "reason": reason
        }

    # Phase3 Refined
    refined_path = base_dir / "Phase3_Refined" / f"750Picacho_{scene_name}_refined.tif"
    if refined_path.exists():
        img = Image.open(refined_path)
        metrics = analyze_channels(img)
        score, reason = assess_quality(metrics, scene_type)
        results["Phase3 Refined"] = {
            "path": refined_path,
            "metrics": metrics,
            "score": score,
            "reason": reason
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare 750 Picacho outputs")
    parser.add_argument(
        "--scene",
        choices=SCENES,
        required=True,
        help="Scene to analyze"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/Users/rc/Desktop/Cache/750_LightFiction_Final_Views"),
        help="Base directory with outputs"
    )

    args = parser.parse_args()

    print("="*80)
    print(f"SCENE COMPARISON: {args.scene}")
    print("="*80)

    results = compare_scene(args.scene, args.base_dir)

    for version, data in results.items():
        print(f"\n{version}:")
        print(f"  Path: {data['path'].name}")

        metrics = data['metrics']
        if "error" not in metrics:
            print(f"  RGB Means: R={metrics['R']['mean']:.1f} | G={metrics['G']['mean']:.1f} | B={metrics['B']['mean']:.1f}")
            print(f"  Brightness: {metrics['brightness']:.1f}")
            print(f"  Contrast: {metrics['contrast']:.1f}")
            print(f"  Neutral Gray: {'⚠️  YES (ISSUE)' if metrics['is_neutral_gray'] else '✓ No'}")

        print(f"  Quality Score: {data['score']}/100")
        print(f"  Assessment: {data['reason']}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)

    # Find best version
    best_version = max(results.items(), key=lambda x: x[1]['score'])
    print(f"\nBest Version: {best_version[0]} ({best_version[1]['score']}/100)")

    # Identify issues
    issues_found = []
    for version, data in results.items():
        if data['score'] < 85:
            issues_found.append(f"{version}: {data['reason']}")

    if issues_found:
        print("\n⚠️  Issues Found:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("\n✓ All versions above 85/100 quality threshold")


if __name__ == "__main__":
    main()
