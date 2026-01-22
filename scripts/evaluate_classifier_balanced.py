#!/usr/bin/env python3
"""
Evaluate classifier with balanced accuracy and per-class metrics.
Usage:
  python scripts/evaluate_classifier_balanced.py \
      --metrics-dir outputs/validation_full_* \
      --labels data/validation_full/labels.csv
"""

import argparse
import json
import glob
import pandas as pd
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    confusion_matrix,
)


def load_metrics(metrics_dir):
    metrics = {}
    for path in glob.glob(f"{metrics_dir}/*_metrics.json"):
        with open(path) as f:
            d = json.load(f)
        image = d.get("image")
        pred = d.get("scene_type")
        if image and pred:
            metrics[image] = pred
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    # Load ground truth
    df = pd.read_csv(args.labels)
    gt = dict(zip(df.filename, df.scene_type))

    # Load predictions
    preds = load_metrics(args.metrics_dir)

    # Filter common images
    common = [img for img in preds if img in gt]
    y_true = [gt[img] for img in common]
    y_pred = [preds[img] for img in common]

    # Balanced accuracy
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    print("\n" + "=" * 60)
    print("BALANCED CLASSIFICATION EVALUATION")
    print("=" * 60)
    print(f"\nDataset: {len(common)} images")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print("\nPer-Class Metrics:")
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    # Save CSV
    pd.DataFrame(report).transpose().to_csv("classification_metrics.csv")
    print("\n✓ Saved classification_metrics.csv")


if __name__ == "__main__":
    main()
