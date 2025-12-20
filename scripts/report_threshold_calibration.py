#!/usr/bin/env python3
"""
Stratified Threshold Calibration Report
Usage:
  python scripts/report_threshold_calibration.py \
      --metrics-dir outputs/validation_full_50img_* \
      --labels data/validation_full/labels.csv
"""

import argparse
import json
import glob
import pandas as pd
from pathlib import Path

def load_jsons(metrics_dir):
    rows = []
    for path in glob.glob(f"{metrics_dir}/*_metrics.json"):
        with open(path) as f:
            data = json.load(f)
        
        # Flatten metrics for stratification
        row = {
            "image": data.get("image"),
            "scene_type": data.get("scene_type"),
            "edge_f1": data.get("edge_f1"),
            "chamfer_px": data.get("chamfer_px"),
            "lenient_pass": data.get("lenient_pass"),
            "strict_pass": data.get("strict_pass"),
        }
        
        # Add classification factors if present
        if "classification_factors" in data:
            for k, v in data["classification_factors"].items():
                row[f"clf_{k}"] = v
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    df_metrics = load_jsons(args.metrics_dir)
    df_labels = pd.read_csv(args.labels)

    print(f"\n{'='*60}")
    print("STRATIFIED THRESHOLD CALIBRATION REPORT")
    print(f"{'='*60}")
    print(f"Metrics loaded: {len(df_metrics)}")
    print(f"Labels loaded: {len(df_labels)}")

    # Join on filename to get expected scene type
    df = df_metrics.merge(
        df_labels, 
        left_on="image", 
        right_on="filename", 
        how="left",
        suffixes=("_pred", "_true")
    )
    
    # Rename columns for clarity
    if "scene_type" in df.columns and "scene_type_pred" not in df.columns:
        df = df.rename(columns={"scene_type": "scene_type_pred"})

    # Report by predicted scene type
    print(f"\n{'='*60}")
    print("METRICS BY PREDICTED SCENE TYPE")
    print(f"{'='*60}")
    
    pred_col = "scene_type_pred" if "scene_type_pred" in df.columns else "scene_type"
    overall = df.groupby(pred_col).agg({
        "edge_f1": ["count", "mean", "std", "min", "max"],
        "chamfer_px": ["mean", "std", "min", "max"],
        "lenient_pass": "mean",
        "strict_pass": "mean"
    }).round(3)
    
    print(overall)

    # Report by true scene type
    if "scene_type_true" in df.columns:
        print(f"\n{'='*60}")
        print("METRICS BY TRUE SCENE TYPE")
        print(f"{'='*60}")
        
        by_true = df.groupby("scene_type_true").agg({
            "edge_f1": ["count", "mean", "std", "min", "max"],
            "chamfer_px": ["mean", "std", "min", "max"],
            "lenient_pass": "mean",
            "strict_pass": "mean"
        }).round(3)
        
        print(by_true)
        by_true.to_csv("stratified_true_class.csv")
        print("\n✓ Saved stratified_true_class.csv")

    # Save overall report
    overall.to_csv("stratified_overall.csv")
    print("✓ Saved stratified_overall.csv")

    # Percentile analysis for threshold calibration
    print(f"\n{'='*60}")
    print("PERCENTILE ANALYSIS (for threshold calibration)")
    print(f"{'='*60}")
    
    pred_col = "scene_type_pred" if "scene_type_pred" in df.columns else "scene_type"
    for scene_type in df[pred_col].unique():
        if pd.isna(scene_type):
            continue
        subset = df[df[pred_col] == scene_type]
        print(f"\n{scene_type} (n={len(subset)}):")
        print(f"  edge_f1: {subset['edge_f1'].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()}")
        print(f"  chamfer: {subset['chamfer_px'].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()}")

if __name__ == "__main__":
    main()
