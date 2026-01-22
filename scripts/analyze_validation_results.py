#!/usr/bin/env python3
"""Analyze validation results and generate confusion matrix."""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


def load_metrics(output_dir: Path) -> List[Dict]:
    """Load all metrics JSON files."""
    metrics = []
    for json_file in sorted(output_dir.glob("*_metrics.json")):
        with open(json_file) as f:
            data = json.load(f)
            metrics.append(data)
    return metrics


def validate_completeness(metrics: List[Dict]) -> Tuple[bool, List[str]]:
    """Check all metrics are populated (no None/null values)."""
    required_keys = [
        "scene_type",
        "edge_f1",
        "lenient_pass",
        "strict_pass",
        "classification_factors",
    ]
    errors = []

    for m in metrics:
        image = m.get("image", "unknown")
        for key in required_keys:
            if key not in m or m[key] is None:
                errors.append(f"{image}: missing or null '{key}'")

    return len(errors) == 0, errors


def infer_expected_labels(image_name: str) -> str:
    """Infer expected scene type from filename patterns."""
    img_lower = image_name.lower()

    # Texture-dominated (water, glass, foliage, reflective)
    texture_patterns = ["pool", "ocean", "water", "glass", "aerial", "foliage", "trees"]
    # Structure-dominated (interiors, architecture)
    structure_patterns = [
        "kitchen",
        "bathroom",
        "bedroom",
        "living",
        "great",
        "interior",
        "entry",
        "dining",
        "office",
        "courtyard",
    ]

    for pattern in texture_patterns:
        if pattern in img_lower:
            return "texture_dominated"

    for pattern in structure_patterns:
        if pattern in img_lower:
            return "structure_dominated"

    # Default: if it has "room" or architectural terms, structure; else texture
    if any(x in img_lower for x in ["room", "lobby", "hall", "suite"]):
        return "structure_dominated"

    return "texture_dominated"  # Conservative default


def compute_confusion_matrix(metrics: List[Dict]) -> Dict:
    """Generate confusion matrix and classification accuracy."""
    confusion = defaultdict(lambda: defaultdict(int))
    correct = 0
    total = len(metrics)

    for m in metrics:
        predicted = m["scene_type"]
        expected = infer_expected_labels(m["image"])

        confusion[expected][predicted] += 1
        if predicted == expected:
            correct += 1

    accuracy = (correct / total * 100) if total > 0 else 0.0

    return {
        "accuracy_pct": accuracy,
        "correct": correct,
        "total": total,
        "confusion": dict(confusion),
    }


def compute_pass_rates(metrics: List[Dict]) -> Dict:
    """Compute pass rates overall and by scene type."""
    overall_lenient = sum(1 for m in metrics if m.get("lenient_pass"))
    overall_strict = sum(1 for m in metrics if m.get("strict_pass"))
    total = len(metrics)

    by_scene = defaultdict(lambda: {"total": 0, "lenient": 0, "strict": 0, "metrics": []})

    for m in metrics:
        scene = m["scene_type"]
        by_scene[scene]["total"] += 1
        if m.get("lenient_pass"):
            by_scene[scene]["lenient"] += 1
        if m.get("strict_pass"):
            by_scene[scene]["strict"] += 1
        by_scene[scene]["metrics"].append(m)

    return {
        "overall": {
            "lenient_pass": overall_lenient,
            "strict_pass": overall_strict,
            "total": total,
            "lenient_pct": (overall_lenient / total * 100) if total > 0 else 0,
            "strict_pct": (overall_strict / total * 100) if total > 0 else 0,
        },
        "by_scene": dict(by_scene),
    }


def print_report(metrics: List[Dict], confusion: Dict, pass_rates: Dict):
    """Print formatted analysis report."""
    print("=" * 80)
    print("VALIDATION RESULTS ANALYSIS")
    print("=" * 80)
    print()

    # Completeness check
    is_complete, errors = validate_completeness(metrics)
    if is_complete:
        print("✅ COMPLETENESS CHECK: PASSED")
        print(f"   All {len(metrics)} metrics files fully populated (no nulls)")
    else:
        print("❌ COMPLETENESS CHECK: FAILED")
        for err in errors:
            print(f"   {err}")
        print()
        sys.exit(1)

    print()

    # Classification accuracy
    print("CLASSIFICATION ACCURACY")
    print("-" * 80)
    print(f"Accuracy: {confusion['accuracy_pct']:.1f}% ({confusion['correct']}/{confusion['total']})")
    target = 90.0
    status = "✅ MEETS TARGET" if confusion["accuracy_pct"] >= target else "⚠️  BELOW TARGET"
    print(f"Target:   {target}% - {status}")
    print()

    # Confusion matrix
    print("CONFUSION MATRIX")
    print("-" * 80)
    print(f"{'Expected':<25} {'Predicted →':<15}")
    print(f"{'↓':<25} {'structure':<15} {'texture':<15}")
    print("-" * 80)

    for expected in sorted(confusion["confusion"].keys()):
        row = confusion["confusion"][expected]
        struct = row.get("structure_dominated", 0)
        tex = row.get("texture_dominated", 0)
        print(f"{expected:<25} {struct:<15} {tex:<15}")
    print()

    # Pass rates
    print("PASS RATES")
    print("-" * 80)
    overall = pass_rates["overall"]
    print(f"Overall:  Lenient={overall['lenient_pass']}/{overall['total']} ({overall['lenient_pct']:.1f}%)")
    print(f"          Strict={overall['strict_pass']}/{overall['total']} ({overall['strict_pct']:.1f}%)")
    print()

    print("By Scene Type:")
    for scene, stats in sorted(pass_rates["by_scene"].items()):
        lenient_pct = (stats["lenient"] / stats["total"] * 100) if stats["total"] > 0 else 0
        strict_pct = (stats["strict"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {scene}:")
        print(f"    Lenient: {stats['lenient']}/{stats['total']} ({lenient_pct:.1f}%)")
        print(f"    Strict:  {stats['strict']}/{stats['total']} ({strict_pct:.1f}%)")
    print()

    # Top failures
    print("TOP FAILURES (by edge_f1)")
    print("-" * 80)
    sorted_by_f1 = sorted(metrics, key=lambda x: x.get("edge_f1", 0))[:5]
    for m in sorted_by_f1:
        print(f"{m['image']:<40} scene={m['scene_type']:<20} edge_f1={m['edge_f1']:.3f}")
    print()

    # Classification decision distribution
    print("DECISION RULES USED")
    print("-" * 80)
    decisions = Counter(m["classification_factors"].get("decision_rule", "unknown") for m in metrics)
    for rule, count in decisions.most_common():
        print(f"  {rule:<40} {count:>3}")
    print()

    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if confusion["accuracy_pct"] >= 90:
        print("✅ Classifier is stable. Proceed to gate calibration.")
    elif confusion["accuracy_pct"] >= 80:
        print("⚠️  Classifier is marginal. Review misclassifications before gate tuning.")
    else:
        print("❌ Classifier needs improvement. DO NOT tune gates yet.")

    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_validation_results.py <output_dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist")
        sys.exit(1)

    metrics = load_metrics(output_dir)
    if not metrics:
        print(f"Error: No *_metrics.json files found in {output_dir}")
        sys.exit(1)

    confusion = compute_confusion_matrix(metrics)
    pass_rates = compute_pass_rates(metrics)

    print_report(metrics, confusion, pass_rates)

    # Save summary
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "confusion": confusion,
                "pass_rates": pass_rates,
                "total_images": len(metrics),
            },
            f,
            indent=2,
        )
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
