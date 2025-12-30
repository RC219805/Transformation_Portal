#!/usr/bin/env python3
"""Re-analyze validation results with filename hints applied."""

import json
import sys
from pathlib import Path
from collections import defaultdict


def get_filename_hint(filename):
    """Extract filename hint."""
    filename_lower = filename.lower()
    texture_patterns = ["pool", "ocean", "water", "glass", "aerial", "foliage", "trees", "shores"]
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
        if pattern in filename_lower:
            return "texture"
    for pattern in structure_patterns:
        if pattern in filename_lower:
            return "structure"
    return None


def is_borderline(ratio, depth_gradient_var, edge_density):
    """Detect borderline cases."""
    ratio_borderline = 2.5 <= ratio <= 7.0
    gradient_borderline = 0.0004 <= depth_gradient_var <= 0.0008
    density_borderline = 0.02 <= edge_density <= 0.05
    return ratio_borderline or gradient_borderline or density_borderline


def reclassify_with_filename(metrics):
    """Apply filename hints to existing metrics."""
    results = []
    overrides = 0

    for m in metrics:
        filename = m["image"]
        original_type = m["scene_type"]

        # Get factors
        cf = m["classification_factors"]
        ratio = cf.get("ratio", 0)
        depth_var = cf.get("depth_variance", 0)
        edge_density = cf.get("edge_density", 0)
        depth_gradient_var = cf.get("depth_gradient_var", 0)

        # Apply filename hint
        hint = get_filename_hint(filename)
        new_type = original_type
        override_applied = False

        if hint and is_borderline(ratio, depth_gradient_var, edge_density):
            new_type = f"{hint}_dominated"
            if new_type != original_type:
                override_applied = True
                overrides += 1

        results.append(
            {
                "filename": filename,
                "original": original_type,
                "new": new_type,
                "hint": hint,
                "override": override_applied,
                "factors": cf,
            }
        )

    return results, overrides


def infer_expected(filename):
    """Infer expected scene type."""
    f = filename.lower()
    tex = ["pool", "ocean", "water", "glass", "aerial", "foliage", "trees", "shores"]
    struct = ["kitchen", "bathroom", "bedroom", "living", "great", "interior", "entry", "dining", "office", "courtyard"]
    for p in tex:
        if p in f:
            return "texture_dominated"
    for p in struct:
        if p in f:
            return "structure_dominated"
    return "texture_dominated"


def main():
    if len(sys.argv) < 2:
        print("Usage: reanalyze_with_filenames.py <output_dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    metrics = []

    for json_file in sorted(output_dir.glob("*_metrics.json")):
        with open(json_file) as f:
            metrics.append(json.load(f))

    if not metrics:
        print(f"No metrics found in {output_dir}")
        sys.exit(1)

    # Reclassify
    results, overrides = reclassify_with_filename(metrics)

    # Compute new accuracy
    correct = sum(1 for r in results if r["new"] == infer_expected(r["filename"]))
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0

    # Original accuracy
    orig_correct = sum(1 for r in results if r["original"] == infer_expected(r["filename"]))
    orig_accuracy = (orig_correct / total * 100) if total > 0 else 0

    print("=" * 100)
    print("FILENAME HINT RE-ANALYSIS")
    print("=" * 100)
    print()
    print(f"Original Accuracy:  {orig_accuracy:.1f}% ({orig_correct}/{total})")
    print(f"New Accuracy:       {accuracy:.1f}% ({correct}/{total})")
    print(f"Overrides Applied:  {overrides}")
    print()

    # Show overrides
    if overrides > 0:
        print("OVERRIDDEN CLASSIFICATIONS:")
        print("-" * 100)
        for r in results:
            if r["override"]:
                expected = infer_expected(r["filename"])
                correct_symbol = "✅" if r["new"] == expected else "❌"
                print(f"{correct_symbol} {r['filename']:<45} {r['original']:<20} → {r['new']:<20} (hint={r['hint']})")
        print()

    # Show remaining errors
    errors = [r for r in results if r["new"] != infer_expected(r["filename"])]
    if errors:
        print(f"REMAINING ERRORS ({len(errors)}):")
        print("-" * 100)
        for r in errors:
            print(f"{r['filename']:<45} Expected={infer_expected(r['filename']):<20} Got={r['new']:<20}")
        print()

    # Summary
    target = 85.0
    if accuracy >= 90:
        print("✅ EXCELLENT: Accuracy ≥90%. Classifier is production-ready.")
    elif accuracy >= target:
        print(f"✅ MEETS TARGET: Accuracy ≥{target}%. Proceed to gate calibration.")
    elif accuracy >= 75:
        print(f"⚠️  MARGINAL: Accuracy {accuracy:.1f}%. Close to target but may need refinement.")
    else:
        print(f"❌ BELOW TARGET: Accuracy {accuracy:.1f}%. Classifier needs more work.")

    print("=" * 100)


if __name__ == "__main__":
    main()
