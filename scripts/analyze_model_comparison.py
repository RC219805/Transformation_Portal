#!/usr/bin/env python3
"""
Statistical Model Comparison Analysis

Performs rigorous statistical testing on multi-model validation results:
- Paired t-tests for depth quality metrics
- McNemar's test for pass/fail rates
- Confidence intervals for metric differences
- Stratified analysis by scene type

Usage:
    python scripts/analyze_model_comparison.py \
        --comparison-dir outputs/model_comparison/run_* \
        --baseline-model DA2_Large \
        --confidence-level 0.95
"""

import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report


def load_model_metrics(model_dir: Path) -> pd.DataFrame:
    """Load all _metrics.json files for a model run."""
    metrics_files = list(model_dir.glob("*/*_metrics.json"))

    rows = []
    for mf in metrics_files:
        with open(mf) as f:
            data = json.load(f)

        # Extract key fields
        row = {
            "image": data.get("image"),
            "scene_type": data.get("scene_type"),
            "edge_f1": data.get("edge_f1"),
            "chamfer_px": data.get("chamfer_px"),
            "lenient_pass": data.get("lenient_pass"),
            "strict_pass": data.get("strict_pass"),
            "seam_ratio": data.get("seam_ratio", 0.0),
            "hf_energy": data.get("classification_factors", {}).get("hf_energy"),
            "depth_range": data.get("classification_factors", {}).get("depth_range"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def paired_comparison(
    baseline: pd.DataFrame,
    treatment: pd.DataFrame,
    metric: str,
    confidence: float = 0.95,
) -> Dict:
    """
    Perform paired t-test on a specific metric.

    Returns:
        dict with test statistic, p-value, confidence interval, effect size
    """
    # Merge on image to ensure pairing
    merged = baseline.merge(
        treatment,
        on="image",
        suffixes=("_baseline", "_treatment"),
    )

    baseline_vals = merged[f"{metric}_baseline"].dropna()
    treatment_vals = merged[f"{metric}_treatment"].dropna()

    if len(baseline_vals) == 0 or len(treatment_vals) == 0:
        return {
            "metric": metric,
            "n": 0,
            "error": "Insufficient paired data",
        }

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_vals, treatment_vals)

    # Cohen's d (effect size)
    diff = treatment_vals - baseline_vals
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0.0

    # Confidence interval for mean difference
    alpha = 1 - confidence
    ci = stats.t.interval(
        confidence,
        len(diff) - 1,
        loc=diff.mean(),
        scale=diff.sem(),
    )

    return {
        "metric": metric,
        "n": len(diff),
        "baseline_mean": float(baseline_vals.mean()),
        "treatment_mean": float(treatment_vals.mean()),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "cohens_d": float(cohens_d),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
    }


def mcnemar_test(
    baseline: pd.DataFrame,
    treatment: pd.DataFrame,
    pass_col: str = "lenient_pass",
) -> Dict:
    """
    McNemar's test for paired binary outcomes (pass/fail).

    Returns:
        dict with test statistic, p-value, and contingency table
    """
    merged = baseline.merge(
        treatment,
        on="image",
        suffixes=("_baseline", "_treatment"),
    )

    baseline_pass = merged[f"{pass_col}_baseline"].fillna(False).astype(bool)
    treatment_pass = merged[f"{pass_col}_treatment"].fillna(False).astype(bool)

    # Build contingency table
    # [baseline_fail & treatment_fail, baseline_fail & treatment_pass]
    # [baseline_pass & treatment_fail, baseline_pass & treatment_pass]

    b = int((~baseline_pass & treatment_pass).sum())  # baseline fail, treatment pass
    c = int((baseline_pass & ~treatment_pass).sum())  # baseline pass, treatment fail

    # McNemar statistic (continuity correction)
    if b + c == 0:
        return {
            "pass_col": pass_col,
            "n": len(merged),
            "error": "No discordant pairs (b + c = 0)",
        }

    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, 1)

    return {
        "pass_col": pass_col,
        "n": len(merged),
        "baseline_pass_count": int(baseline_pass.sum()),
        "treatment_pass_count": int(treatment_pass.sum()),
        "both_fail": int((~baseline_pass & ~treatment_pass).sum()),
        "baseline_fail_treatment_pass": b,
        "baseline_pass_treatment_fail": c,
        "both_pass": int((baseline_pass & treatment_pass).sum()),
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def stratified_analysis(
    baseline: pd.DataFrame,
    treatment: pd.DataFrame,
    stratify_by: str = "scene_type",
) -> pd.DataFrame:
    """
    Compare models stratified by a categorical variable (e.g., scene_type).
    """
    strata = []

    for stratum_val in baseline[stratify_by].dropna().unique():
        baseline_stratum = baseline[baseline[stratify_by] == stratum_val]
        treatment_stratum = treatment[treatment[stratify_by] == stratum_val]

        merged = baseline_stratum.merge(
            treatment_stratum,
            on="image",
            suffixes=("_baseline", "_treatment"),
        )

        if len(merged) == 0:
            continue

        # Compute key metrics
        strata.append(
            {
                stratify_by: stratum_val,
                "n": len(merged),
                "baseline_lenient_rate": baseline_stratum["lenient_pass"].mean(),
                "treatment_lenient_rate": treatment_stratum["lenient_pass"].mean(),
                "lenient_rate_delta": (treatment_stratum["lenient_pass"].mean() - baseline_stratum["lenient_pass"].mean()),
                "baseline_edge_f1_mean": baseline_stratum["edge_f1"].mean(),
                "treatment_edge_f1_mean": treatment_stratum["edge_f1"].mean(),
                "edge_f1_delta": (treatment_stratum["edge_f1"].mean() - baseline_stratum["edge_f1"].mean()),
            }
        )

    return pd.DataFrame(strata)


def main():
    parser = argparse.ArgumentParser(description="Statistical comparison of multi-model validation results")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        required=True,
        help="Path to model_comparison run directory",
    )
    parser.add_argument(
        "--baseline-model",
        default="DA2_Large",
        help="Baseline model key for comparison",
    )
    parser.add_argument(
        "--treatment-models",
        nargs="+",
        help="Treatment model keys (default: all non-baseline)",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: comparison_dir/analysis)",
    )

    args = parser.parse_args()

    if not args.comparison_dir.exists():
        print(f"❌ Comparison directory not found: {args.comparison_dir}")
        return 1

    output_dir = args.output_dir or (args.comparison_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline model data
    baseline_dir = args.comparison_dir / f"model_{args.baseline_model}"
    if not baseline_dir.exists():
        print(f"❌ Baseline model directory not found: {baseline_dir}")
        return 1

    print(f"\n{'=' * 70}")
    print(f"STATISTICAL MODEL COMPARISON")
    print(f"{'=' * 70}\n")
    print(f"Baseline: {args.baseline_model}")
    print(f"Confidence: {args.confidence_level}")
    print(f"Output: {output_dir}\n")

    baseline_df = load_model_metrics(baseline_dir)
    print(f"Loaded baseline: {len(baseline_df)} images\n")

    # Discover treatment models if not specified
    if not args.treatment_models:
        model_dirs = [d for d in args.comparison_dir.glob("model_*") if d.is_dir()]
        args.treatment_models = [d.name.replace("model_", "") for d in model_dirs if d.name != f"model_{args.baseline_model}"]

    print(f"Treatment models: {', '.join(args.treatment_models)}\n")

    # Run comparisons
    all_comparisons = []

    for treatment_key in args.treatment_models:
        treatment_dir = args.comparison_dir / f"model_{treatment_key}"
        if not treatment_dir.exists():
            print(f"⚠️  Skipping {treatment_key}: directory not found")
            continue

        treatment_df = load_model_metrics(treatment_dir)
        print(f"\n▶ Comparing {args.baseline_model} vs {treatment_key}")
        print(f"  Treatment: {len(treatment_df)} images")

        # Paired comparisons on continuous metrics
        metrics_to_test = ["edge_f1", "chamfer_px", "seam_ratio"]

        comparison = {
            "baseline": args.baseline_model,
            "treatment": treatment_key,
            "continuous_metrics": {},
            "binary_metrics": {},
            "stratified": {},
        }

        for metric in metrics_to_test:
            result = paired_comparison(
                baseline_df,
                treatment_df,
                metric,
                confidence=args.confidence_level,
            )
            comparison["continuous_metrics"][metric] = result

            if "error" not in result:
                sig = "✓" if result["significant"] else " "
                print(
                    f"  [{sig}] {metric}: Δ={result['mean_diff']:.4f}, p={result['p_value']:.4f}, d={result['cohens_d']:.2f}"
                )

        # McNemar's test on pass/fail
        for pass_col in ["lenient_pass", "strict_pass"]:
            result = mcnemar_test(baseline_df, treatment_df, pass_col)
            comparison["binary_metrics"][pass_col] = result

            if "error" not in result:
                sig = "✓" if result["significant"] else " "
                delta = result["treatment_pass_count"] - result["baseline_pass_count"]
                print(f"  [{sig}] {pass_col}: Δ={delta}, p={result['p_value']:.4f}")

        # Stratified analysis
        strat_df = stratified_analysis(baseline_df, treatment_df, "scene_type")
        comparison["stratified"]["scene_type"] = strat_df.to_dict(orient="records")

        print(f"\n  Stratified by scene_type:")
        print(strat_df.to_string(index=False))

        all_comparisons.append(comparison)

    # Save results
    summary_json = output_dir / "statistical_comparison.json"
    with open(summary_json, "w") as f:
        json.dump(all_comparisons, f, indent=2)

    print(f"\n✅ Statistical analysis saved: {summary_json}")

    # Generate summary CSV
    summary_rows = []
    for comp in all_comparisons:
        for metric, result in comp["continuous_metrics"].items():
            if "error" in result:
                continue
            summary_rows.append(
                {
                    "baseline": comp["baseline"],
                    "treatment": comp["treatment"],
                    "metric": metric,
                    "mean_diff": result["mean_diff"],
                    "p_value": result["p_value"],
                    "significant": result["significant"],
                    "cohens_d": result["cohens_d"],
                    "ci_lower": result["ci_lower"],
                    "ci_upper": result["ci_upper"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "statistical_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Summary CSV saved: {summary_csv}\n")

    print("🎉 Statistical comparison complete!")


if __name__ == "__main__":
    main()
