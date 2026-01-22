#!/usr/bin/env python3
"""
Analyze Validation V2 Results with Classification Report
=========================================================

Generates:
- Classification report (precision/recall/F1/support per class)
- Confusion matrix (correct convention: rows=true, cols=predicted)
- Pass rates stratified by scene type
- Feature separability visualization

Usage:
    python scripts/analyze_validation_v2.py --results-dir outputs/validation_v2_20251218_170022_8197588
"""

import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
)


def load_metrics(results_dir: Path) -> List[Dict]:
    """Load all metrics JSON files."""
    metrics_files = sorted(glob.glob(str(results_dir / "*_metrics.json")))

    if not metrics_files:
        raise ValueError(f"No metrics files found in {results_dir}")

    metrics_list = []
    for mf in metrics_files:
        with open(mf) as f:
            data = json.load(f)
        data["_filename"] = Path(mf).name
        metrics_list.append(data)

    return metrics_list


def infer_ground_truth(filename: str) -> str:
    """
    Infer ground truth scene type from filename patterns.

    This is a heuristic until we have explicit labels.csv.
    """
    filename_lower = filename.lower()

    # Texture patterns
    texture_patterns = [
        "pool",
        "ocean",
        "water",
        "glass",
        "aerial",
        "foliage",
        "trees",
        "shores",
        "beach",
        "sea",
    ]

    # Structure patterns
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
        "room",
        "hall",
        "lobby",
    ]

    # Check patterns
    for p in texture_patterns:
        if p in filename_lower:
            return "texture_dominated"

    for p in structure_patterns:
        if p in filename_lower:
            return "structure_dominated"

    # Default: unknown (exclude from classification metrics)
    return "unknown"


def compute_pass_flags(metrics: Dict) -> Tuple[bool, bool]:
    """
    Compute lenient and strict pass flags using EdgeMetrics.passed() logic.
    """
    edge_f1 = metrics.get("edge_f1", 0)
    edge_overlap = metrics.get("edge_overlap", 0)
    edge_count_ratio = metrics.get("edge_count_ratio", 0)
    halo_score = metrics.get("halo_score", 0)
    overshoot_penalty = metrics.get("overshoot_penalty", 0)

    # Lenient
    passed_lenient = edge_f1 >= 0.30 and edge_overlap >= 0.40 and edge_count_ratio <= 3.0 and overshoot_penalty <= 0.5

    # Strict
    passed_strict = (
        edge_f1 >= 0.45 and edge_overlap >= 0.50 and edge_count_ratio <= 2.0 and halo_score >= 0.7 and overshoot_penalty <= 0.3
    )

    return passed_lenient, passed_strict


def generate_classification_report(metrics_list: List[Dict]) -> str:
    """Generate sklearn classification report."""
    y_true = []
    y_pred = []
    filenames = []

    for m in metrics_list:
        # Get ground truth from filename
        gt = infer_ground_truth(m["image"])

        # Skip unknown ground truth
        if gt == "unknown":
            continue

        pred = m.get("scene_type", "unknown")

        if pred != "unknown":
            y_true.append(gt)
            y_pred.append(pred)
            filenames.append(m["image"])

    if not y_true:
        return "No labeled data for classification report"

    # Generate report
    report = classification_report(
        y_true,
        y_pred,
        target_names=["structure_dominated", "texture_dominated"],
        digits=3,
        zero_division=0,
    )

    # Add accuracy scores
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    header = (
        f"Classification Report (N={len(y_true)} samples)\n"
        f"{'=' * 80}\n"
        f"Accuracy: {acc:.3f} ({100 * acc:.1f}%)\n"
        f"Balanced Accuracy: {balanced_acc:.3f} ({100 * balanced_acc:.1f}%)\n"
        f"\n{report}\n"
    )

    return header


def generate_confusion_matrix(metrics_list: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """Generate confusion matrix with correct convention (rows=true, cols=pred)."""
    y_true = []
    y_pred = []

    for m in metrics_list:
        gt = infer_ground_truth(m["image"])

        if gt == "unknown":
            continue

        pred = m.get("scene_type", "unknown")

        if pred != "unknown":
            y_true.append(gt)
            y_pred.append(pred)

    if not y_true:
        return np.array([]), []

    # Confusion matrix: rows=true, cols=pred
    labels = ["structure_dominated", "texture_dominated"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return cm, labels


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], output_path: Path):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Scene Classification)",
    )

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ Confusion matrix saved: {output_path}")


def plot_feature_separability(metrics_list: List[Dict], output_path: Path):
    """Plot features vs ground truth to assess separability."""
    # Extract features
    structure_data = {
        "ratio": [],
        "depth_var": [],
        "edge_density": [],
        "depth_grad_var": [],
    }
    texture_data = {
        "ratio": [],
        "depth_var": [],
        "edge_density": [],
        "depth_grad_var": [],
    }

    for m in metrics_list:
        gt = infer_ground_truth(m["image"])

        if gt == "unknown":
            continue

        cf = m.get("classification_factors", {})

        data_dict = structure_data if gt == "structure_dominated" else texture_data
        data_dict["ratio"].append(cf.get("ratio", 0))
        data_dict["depth_var"].append(cf.get("depth_variance", 0))
        data_dict["edge_density"].append(cf.get("edge_density", 0))
        data_dict["depth_grad_var"].append(cf.get("depth_gradient_var", 0))

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    features = [
        ("ratio", "Edge Ratio (raw/structure)"),
        ("depth_var", "Depth Variance"),
        ("edge_density", "Edge Density"),
        ("depth_grad_var", "Depth Gradient Variance"),
    ]

    for idx, (key, label) in enumerate(features):
        ax = axes[idx // 2, idx % 2]

        # Scatter plot
        ax.scatter(
            structure_data[key],
            [0] * len(structure_data[key]),
            c="blue",
            alpha=0.6,
            s=100,
            label="Structure",
        )
        ax.scatter(
            texture_data[key],
            [1] * len(texture_data[key]),
            c="red",
            alpha=0.6,
            s=100,
            label="Texture",
        )

        ax.set_xlabel(label, fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Structure", "Texture"])
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Feature Separability by Scene Type", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150)
    print(f"✅ Feature separability plot saved: {output_path}")


def stratified_pass_rates(metrics_list: List[Dict]) -> str:
    """Compute pass rates stratified by scene type."""
    texture_count = 0
    structure_count = 0
    texture_lenient = 0
    texture_strict = 0
    structure_lenient = 0
    structure_strict = 0

    for m in metrics_list:
        scene_type = m.get("scene_type", "unknown")

        passed_lenient, passed_strict = compute_pass_flags(m)

        if scene_type == "texture_dominated":
            texture_count += 1
            if passed_lenient:
                texture_lenient += 1
            if passed_strict:
                texture_strict += 1
        elif scene_type == "structure_dominated":
            structure_count += 1
            if passed_lenient:
                structure_lenient += 1
            if passed_strict:
                structure_strict += 1

    report = (
        f"Pass Rates Stratified by Scene Type\n"
        f"{'=' * 80}\n\n"
        f"Texture-dominated scenes (N={texture_count}):\n"
        f"  Lenient pass: {texture_lenient}/{texture_count} "
        f"({100 * texture_lenient / max(texture_count, 1):.1f}%)\n"
        f"  Strict pass:  {texture_strict}/{texture_count} "
        f"({100 * texture_strict / max(texture_count, 1):.1f}%)\n\n"
        f"Structure-dominated scenes (N={structure_count}):\n"
        f"  Lenient pass: {structure_lenient}/{structure_count} "
        f"({100 * structure_lenient / max(structure_count, 1):.1f}%)\n"
        f"  Strict pass:  {structure_strict}/{structure_count} "
        f"({100 * structure_strict / max(structure_count, 1):.1f}%)\n\n"
        f"Overall:\n"
        f"  Lenient pass: {texture_lenient + structure_lenient}/{texture_count + structure_count} "
        f"({100 * (texture_lenient + structure_lenient) / max(texture_count + structure_count, 1):.1f}%)\n"
        f"  Strict pass:  {texture_strict + structure_strict}/{texture_count + structure_count} "
        f"({100 * (texture_strict + structure_strict) / max(texture_count + structure_count, 1):.1f}%)\n"
    )

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze validation V2 results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory with *_metrics.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as results-dir)",
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"❌ Error: Results directory not found: {args.results_dir}")
        return 1

    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    print(f"Loading metrics from {args.results_dir}...")
    metrics_list = load_metrics(args.results_dir)
    print(f"✅ Loaded {len(metrics_list)} metrics files\n")

    # Generate classification report
    print("Classification Report:")
    print("=" * 80)
    report = generate_classification_report(metrics_list)
    print(report)

    # Save to file
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✅ Classification report saved: {report_path}\n")

    # Generate confusion matrix
    cm, labels = generate_confusion_matrix(metrics_list)
    if cm.size > 0:
        print("\nConfusion Matrix (rows=true, cols=predicted):")
        print("=" * 80)
        print(f"                  {labels[0]:>20s}  {labels[1]:>20s}")
        for i, label in enumerate(labels):
            print(f"{label:>20s}  {cm[i, 0]:>20d}  {cm[i, 1]:>20d}")
        print()

        # Plot confusion matrix
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(cm, labels, cm_path)

    # Stratified pass rates
    print("\n" + stratified_pass_rates(metrics_list))

    # Save pass rates
    pass_rates_path = output_dir / "pass_rates_stratified.txt"
    with open(pass_rates_path, "w") as f:
        f.write(stratified_pass_rates(metrics_list))
    print(f"✅ Pass rates saved: {pass_rates_path}\n")

    # Feature separability plot
    sep_path = output_dir / "feature_separability.png"
    plot_feature_separability(metrics_list, sep_path)

    print("\n" + "=" * 80)
    print("✅ Analysis complete")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
