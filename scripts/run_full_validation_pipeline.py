#!/usr/bin/env python3
"""
Combined runner that sequences:

1) Classifier evaluation
2) DepthAnything V2 Large HF input-size sweep
3) Stratified threshold calibration

Generates timestamped artifact directory and CI-friendly JSON summary.

Usage:
    python scripts/run_full_validation_pipeline.py \
        --validation-dir outputs/validation_full_* \
        --labels data/validation_full/labels.csv \
        --structure-input-dir data/structure_subset \
        --sweep-sizes 518 768 896 1022

Example:
    python scripts/run_full_validation_pipeline.py \
      --validation-dir outputs/validation_v2_20251218_170022_8197588 \
      --labels data/validation_full/labels.csv \
      --structure-input-dir data/structure_subset \
      --sweep-sizes 518 768 896 1022
"""

import argparse
import subprocess
import json
import os
import datetime
from pathlib import Path


def run_script(cmd, cwd=None, capture=False):
    print(f"\n▶ Running: {cmd}")
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=capture, text=True)


def timestamp_tag():
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-dir", required=True, help="Path to an existing validation output directory with *_metrics.json files"
    )
    parser.add_argument("--labels", required=True, help="CSV file path for ground-truth labels")
    parser.add_argument(
        "--structure-input-dir", required=True, help="Directory containing structure-dominated images for sweep"
    )
    parser.add_argument(
        "--sweep-sizes", nargs="+", type=int, required=True, help="list of DepthAnything V2 input sizes for the sweep"
    )
    parser.add_argument(
        "--model-id", default="depth-anything/Depth-Anything-V2-Large-hf", help="HF model to use for the sweep"
    )
    parser.add_argument("--output-root", default="outputs/full_validation_pipeline", help="Base output directory")

    args = parser.parse_args()

    # Create versioned output path
    ts = timestamp_tag()
    out_base = Path(args.output_root) / f"run_{ts}"
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"📌 Writing artifacts to: {out_base}")

    summary = {
        "timestamp": ts,
        "validation_dir": str(args.validation_dir),
        "structure_sweep": {},
        "classifier_report": {},
        "stratified_report": {},
    }

    #### Step 1 — Balanced Classifier Evaluation
    classifier_json = out_base / "classifier_report.json"
    classifier_cmd = (
        f"python3 scripts/evaluate_classifier_balanced.py --metrics-dir {args.validation_dir} --labels {args.labels}"
    )

    result = run_script(classifier_cmd, capture=True)
    with open(classifier_json, "w") as f:
        f.write(result.stdout)
    summary["classifier_report_cmd"] = classifier_cmd
    summary["classifier_report_out"] = str(classifier_json)
    print(f"✅ Classifier evaluation complete | saved: {classifier_json}")

    #### Step 2 — Run Input Size Sweep
    sweep_results = {}
    for size in args.sweep_sizes:
        sweep_tag = f"input_{size}"
        sweep_dir = out_base / f"sweep_{sweep_tag}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        sweep_cmd = (
            f"python3 scripts/run_input_size_sweep.py "
            f"--input-dir {args.structure_input_dir} "
            f"--output-dir {sweep_dir} "
            f"--sizes {size} "
            f"--model-id {args.model_id}"
        )

        result = run_script(sweep_cmd, capture=True)
        sweep_results[sweep_tag] = {
            "cmd": sweep_cmd,
            "log": result.stdout.strip(),
            "output_dir": str(sweep_dir),
            "success": (result.returncode == 0),
        }
        print(f"✅ Sweep done for input_size={size} | {sweep_dir}")

    summary["structure_sweep"] = sweep_results

    #### Step 3 — Stratified Threshold Calibration
    stratified_json = out_base / "stratified_report.json"
    strat_cmd = f"python3 scripts/report_threshold_calibration.py --metrics-dir {args.validation_dir} --labels {args.labels}"

    result = run_script(strat_cmd, capture=True)
    with open(stratified_json, "w") as f:
        f.write(result.stdout)

    summary["stratified_report_cmd"] = strat_cmd
    summary["stratified_report_out"] = str(stratified_json)
    print(f"✅ Stratified calibration complete | saved: {stratified_json}")

    #### Final Output
    summary_json = out_base / "pipeline_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎉 Full pipeline complete!")
    print(f"📌 Summary saved to: {summary_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
