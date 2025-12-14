#!/usr/bin/env python3
"""
PR-4D Material Histogram Aggregation

Aggregates materials_v3_response_plan data from multiple scenes to identify
the best next material for pixel ops implementation.

Outputs:
  - pr4d_histogram_aggregate.json (raw aggregated data)
  - pr4d_aggregated_stats.json (ranked material recommendations)
  - pr4d_material_recommendations.md (human-readable report)
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pr4d_data"


def load_reports() -> List[Dict[str, Any]]:
    """Load all *_report.json files from PR-4D data collection."""
    reports = []
    
    for report_path in OUTPUT_DIR.glob("**/*_report.json"):
        try:
            with open(report_path) as f:
                data = json.load(f)
                # Extract scene name from path
                scene_name = report_path.parent.name
                data["_scene_name"] = scene_name
                reports.append(data)
                print(f"✅ Loaded: {scene_name}")
        except Exception as e:
            print(f"⚠️  Error loading {report_path}: {e}", file=sys.stderr)
    
    return reports


def aggregate_histograms(reports: List[Dict]) -> Dict:
    """Aggregate reason histograms across all scenes."""
    pixel_ops_reasons = Counter()
    refinement_reasons = Counter()
    
    for report in reports:
        plan = report.get("materials_v3_response_plan", {})
        summary = plan.get("summary", {})
        
        # Aggregate pixel ops reasons
        for reason, count in summary.get("pixel_ops_reasons", {}).items():
            pixel_ops_reasons[reason] += count
        
        # Aggregate refinement reasons
        for reason, count in summary.get("refinement_reasons", {}).items():
            refinement_reasons[reason] += count
    
    return {
        "pixel_ops_reasons": dict(pixel_ops_reasons),
        "refinement_reasons": dict(refinement_reasons),
        "num_scenes": len(reports),
    }


def compute_material_stats(reports: List[Dict]) -> Dict[str, Dict]:
    """Compute per-material statistics across all scenes."""
    material_stats = defaultdict(lambda: {
        "appearances": 0,
        "total_coverage": 0.0,
        "total_coverage_px": 0,
        "total_mean_conf": 0.0,
        "total_edge_conf": 0.0,
        "should_refine_count": 0,
        "should_apply_count": 0,
        "refinement_reasons": Counter(),
        "pixel_ops_reasons": Counter(),
    })
    
    for report in reports:
        plan = report.get("materials_v3_response_plan", {})
        per_class = plan.get("per_class", {})
        
        for material, class_plan in per_class.items():
            if not class_plan.get("present", False):
                continue
            
            stats = material_stats[material]
            stats["appearances"] += 1
            stats["total_coverage"] += class_plan.get("coverage", 0.0)
            stats["total_coverage_px"] += class_plan.get("coverage_px", 0)
            stats["total_mean_conf"] += class_plan.get("mean_conf", 0.0)
            stats["total_edge_conf"] += class_plan.get("edge_conf", 0.0)
            
            # Refinement decision
            refinement = class_plan.get("refinement", {})
            if refinement.get("should_refine_edges", False):
                stats["should_refine_count"] += 1
            reason = refinement.get("reason")
            if reason:
                stats["refinement_reasons"][reason] += 1
            
            # Pixel ops decision
            pixel_ops = class_plan.get("pixel_ops", {})
            if pixel_ops.get("should_apply", False):
                stats["should_apply_count"] += 1
            reason = pixel_ops.get("reason")
            if reason:
                stats["pixel_ops_reasons"][reason] += 1
    
    # Compute averages
    material_summary = {}
    for material, stats in material_stats.items():
        n = stats["appearances"]
        if n == 0:
            continue
        
        material_summary[material] = {
            "appearances": n,
            "avg_coverage": stats["total_coverage"] / n,
            "avg_coverage_px": stats["total_coverage_px"] // n,
            "avg_mean_conf": stats["total_mean_conf"] / n,
            "avg_edge_conf": stats["total_edge_conf"] / n,
            "should_refine_rate": stats["should_refine_count"] / n,
            "should_apply_rate": stats["should_apply_count"] / n,
            "refinement_reasons": dict(stats["refinement_reasons"]),
            "pixel_ops_reasons": dict(stats["pixel_ops_reasons"]),
        }
    
    return material_summary


def rank_materials(material_stats: Dict[str, Dict]) -> List[tuple]:
    """Rank materials by implementation priority."""
    ranked = []
    
    for material, stats in material_stats.items():
        # Skip materials with insufficient data
        if stats["appearances"] < 2:
            continue
        
        # Compute priority score
        # Factors: frequency, coverage, confidence, signal (no_implementation count)
        frequency_score = min(stats["appearances"] / 6.0, 1.0)  # normalize to max 6 scenes
        coverage_score = min(stats["avg_coverage"] / 0.3, 1.0)  # 30% coverage = full score
        confidence_score = stats["avg_mean_conf"]
        
        # Bonus for "no_implementation" signal (needs pixel ops)
        no_impl_count = stats["pixel_ops_reasons"].get("no_implementation", 0)
        signal_score = min(no_impl_count / 3.0, 1.0)  # 3+ appearances = full signal
        
        # Weighted priority
        priority = (
            frequency_score * 0.3 +
            coverage_score * 0.25 +
            confidence_score * 0.15 +
            signal_score * 0.30
        )
        
        ranked.append((material, priority, stats))
    
    # Sort by priority (descending)
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def generate_markdown_report(
    histograms: Dict,
    material_stats: Dict[str, Dict],
    ranked: List[tuple],
    output_path: Path
) -> None:
    """Generate human-readable markdown report."""
    lines = []
    
    lines.append("# PR-4D Material Recommendations\n")
    lines.append(f"**Generated:** {Path.cwd()}\n")
    lines.append(f"**Scenes analyzed:** {histograms['num_scenes']}\n")
    lines.append("")
    
    lines.append("## Global Reason Histograms\n")
    lines.append("### Pixel Ops Reasons\n")
    for reason, count in sorted(histograms["pixel_ops_reasons"].items(), key=lambda x: -x[1]):
        lines.append(f"- `{reason}`: {count}")
    lines.append("")
    
    lines.append("### Refinement Reasons\n")
    for reason, count in sorted(histograms["refinement_reasons"].items(), key=lambda x: -x[1]):
        lines.append(f"- `{reason}`: {count}")
    lines.append("")
    
    lines.append("## Ranked Material Recommendations\n")
    lines.append("| Rank | Material | Priority | Appearances | Avg Coverage | Avg Conf | No-Impl Count |")
    lines.append("|------|----------|----------|-------------|--------------|----------|---------------|")
    
    for rank, (material, priority, stats) in enumerate(ranked, 1):
        no_impl = stats["pixel_ops_reasons"].get("no_implementation", 0)
        lines.append(
            f"| {rank} | **{material}** | {priority:.3f} | {stats['appearances']} | "
            f"{stats['avg_coverage']:.1%} | {stats['avg_mean_conf']:.3f} | {no_impl} |"
        )
    
    lines.append("")
    lines.append("## Material Statistics\n")
    
    for material, priority, stats in ranked:
        lines.append(f"### {material.capitalize()} (Priority: {priority:.3f})\n")
        lines.append(f"- **Appearances:** {stats['appearances']}")
        lines.append(f"- **Avg Coverage:** {stats['avg_coverage']:.1%} ({stats['avg_coverage_px']:,} px)")
        lines.append(f"- **Avg Mean Confidence:** {stats['avg_mean_conf']:.3f}")
        lines.append(f"- **Avg Edge Confidence:** {stats['avg_edge_conf']:.3f}")
        lines.append(f"- **Should Refine Rate:** {stats['should_refine_rate']:.1%}")
        lines.append(f"- **Should Apply Rate:** {stats['should_apply_rate']:.1%}")
        lines.append("")
        
        if stats["pixel_ops_reasons"]:
            lines.append("**Pixel Ops Reasons:**")
            for reason, count in sorted(stats["pixel_ops_reasons"].items(), key=lambda x: -x[1]):
                lines.append(f"  - `{reason}`: {count}")
            lines.append("")
        
        if stats["refinement_reasons"]:
            lines.append("**Refinement Reasons:**")
            for reason, count in sorted(stats["refinement_reasons"].items(), key=lambda x: -x[1]):
                lines.append(f"  - `{reason}`: {count}")
            lines.append("")
    
    lines.append("## Recommendation\n")
    if ranked:
        top_material, top_priority, top_stats = ranked[0]
        lines.append(f"**Top candidate:** `{top_material}` (priority {top_priority:.3f})\n")
        lines.append(f"- Appears in {top_stats['appearances']} scenes")
        lines.append(f"- {top_stats['avg_coverage']:.1%} average coverage")
        lines.append(f"- {top_stats['avg_mean_conf']:.3f} average confidence")
        
        no_impl = top_stats["pixel_ops_reasons"].get("no_implementation", 0)
        if no_impl > 0:
            lines.append(f"- **{no_impl} occurrences** need pixel ops implementation")
    else:
        lines.append("No materials meet minimum criteria (2+ appearances).")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"✅ Markdown report: {output_path}")


def main():
    """Main aggregation workflow."""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("PR-4D Material Histogram Aggregation")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    # Load reports
    reports = load_reports()
    if not reports:
        print("❌ No reports found in", OUTPUT_DIR, file=sys.stderr)
        sys.exit(1)
    
    print()
    
    # Aggregate histograms
    histograms = aggregate_histograms(reports)
    histogram_path = OUTPUT_DIR / "pr4d_histogram_aggregate.json"
    with open(histogram_path, "w") as f:
        json.dump(histograms, f, indent=2)
    print(f"✅ Histogram aggregate: {histogram_path}")
    
    # Compute material stats
    material_stats = compute_material_stats(reports)
    
    # Rank materials
    ranked = rank_materials(material_stats)
    
    # Save ranked stats
    stats_output = {
        "num_scenes": len(reports),
        "materials": {
            material: {**stats, "priority": priority}
            for material, priority, stats in ranked
        },
        "ranked_order": [material for material, _, _ in ranked],
    }
    stats_path = OUTPUT_DIR / "pr4d_aggregated_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_output, f, indent=2)
    print(f"✅ Aggregated stats: {stats_path}")
    
    # Generate markdown report
    md_path = OUTPUT_DIR / "pr4d_material_recommendations.md"
    generate_markdown_report(histograms, material_stats, ranked, md_path)
    
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✅ Aggregation complete")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    if ranked:
        print()
        print("Top 3 Recommendations:")
        for rank, (material, priority, stats) in enumerate(ranked[:3], 1):
            no_impl = stats["pixel_ops_reasons"].get("no_implementation", 0)
            print(f"  {rank}. {material:12s}  priority={priority:.3f}  "
                  f"appearances={stats['appearances']}  "
                  f"coverage={stats['avg_coverage']:.1%}  "
                  f"no_impl={no_impl}")


if __name__ == "__main__":
    main()
