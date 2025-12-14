#!/usr/bin/env python3
"""
PR-4D Histogram Aggregation and Material Analysis Script

Aggregates materials_v3 histogram data from multiple scenes and generates
actionable recommendations for wood pixel ops implementation.

Usage:
    python scripts/pr4d_aggregate_histograms.py
"""

import json
import logging
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MaterialStats:
    """Statistics for a single material across all scenes."""
    name: str
    frequency: int = 0  # Number of scenes containing this material
    total_coverage: float = 0.0  # Sum of coverage percentages
    total_boundary_pixels: int = 0
    total_edge_alignment: float = 0.0
    no_implementation_count: int = 0
    below_threshold_count: int = 0
    not_in_canary_count: int = 0
    scenes_present: List[str] = field(default_factory=list)
    
    @property
    def avg_coverage(self) -> float:
        """Average coverage across scenes where material appears."""
        return self.total_coverage / self.frequency if self.frequency > 0 else 0.0
    
    @property
    def avg_boundary_pixels(self) -> float:
        """Average boundary pixels across scenes."""
        return self.total_boundary_pixels / self.frequency if self.frequency > 0 else 0.0
    
    @property
    def avg_edge_alignment(self) -> float:
        """Average edge alignment across scenes."""
        return self.total_edge_alignment / self.frequency if self.frequency > 0 else 0.0
    
    def recommendation_score(self) -> float:
        """Calculate recommendation score (higher = better candidate)."""
        # Scoring criteria:
        # - High frequency (max 4 scenes = 40 points)
        # - High no_implementation count (max 4 = 30 points)
        # - Good boundary pixels (>= 500 = 20 points)
        # - Good edge alignment (>= 0.15 = 10 points)
        score = 0.0
        score += (self.frequency / 4.0) * 40  # Frequency contribution
        score += (self.no_implementation_count / 4.0) * 30  # Need contribution
        score += min(self.avg_boundary_pixels / 500, 1.0) * 20  # Edge quality
        score += min(self.avg_edge_alignment / 0.15, 1.0) * 10  # Alignment quality
        return score
    
    def recommendation_tier(self) -> str:
        """Get recommendation tier based on score."""
        score = self.recommendation_score()
        if score >= 80:
            return "⭐ Top Choice"
        elif score >= 60:
            return "Alternative"
        elif score >= 40:
            return "Lower Priority"
        else:
            return "Not Recommended"


@dataclass
class AggregatedData:
    """Aggregated histogram data across all scenes."""
    pixel_ops_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    refinement_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    materials: Dict[str, MaterialStats] = field(default_factory=dict)
    total_scenes: int = 0
    scene_names: List[str] = field(default_factory=list)
    collection_metadata: Dict = field(default_factory=dict)


def get_git_info() -> Dict[str, str]:
    """Get current git commit and branch information."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return {'commit': commit, 'branch': branch}
    except Exception:
        return {'commit': 'unknown', 'branch': 'unknown'}


def find_report_files(base_dir: Path) -> List[Path]:
    """Find all report.json files in the data collection directory."""
    report_files = list(base_dir.rglob('*_report.json'))
    logger.info(f"Found {len(report_files)} report files in {base_dir}")
    return report_files


def validate_schema_version(report_data: Dict) -> bool:
    """Validate that report contains schema v3.1 materials data."""
    try:
        summary = report_data.get('materials_v3_response_plan', {}).get('summary', {})
        return 'pixel_ops_reasons' in summary and 'refinement_reasons' in summary
    except Exception as e:
        logger.warning(f"Schema validation failed: {e}")
        return False


def aggregate_reports(report_files: List[Path]) -> AggregatedData:
    """Aggregate data from all report files."""
    aggregated = AggregatedData()
    
    for report_file in report_files:
        try:
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            # Validate schema
            if not validate_schema_version(report_data):
                logger.warning(f"Skipping {report_file.name}: Invalid schema or missing v3.1 data")
                continue
            
            scene_name = report_file.stem.replace('_report', '')
            aggregated.scene_names.append(scene_name)
            aggregated.total_scenes += 1
            
            # Extract summary data
            summary = report_data['materials_v3_response_plan']['summary']
            
            # Aggregate pixel ops reasons
            for reason, count in summary.get('pixel_ops_reasons', {}).items():
                aggregated.pixel_ops_reasons[reason] += count
            
            # Aggregate refinement reasons
            for reason, count in summary.get('refinement_reasons', {}).items():
                aggregated.refinement_reasons[reason] += count
            
            # Aggregate material statistics
            for material_name, material_data in summary.get('materials', {}).items():
                if material_name not in aggregated.materials:
                    aggregated.materials[material_name] = MaterialStats(name=material_name)
                
                stats = aggregated.materials[material_name]
                stats.frequency += 1
                stats.scenes_present.append(scene_name)
                
                # Accumulate values
                stats.total_coverage += material_data.get('coverage', 0.0)
                
                edge_signals = material_data.get('edge_signals', {})
                stats.total_boundary_pixels += edge_signals.get('boundary_pixels', 0)
                stats.total_edge_alignment += edge_signals.get('edge_alignment', 0.0)
                
                # Count reasons
                pixel_ops = material_data.get('pixel_ops', {})
                if pixel_ops.get('reason') == 'no_implementation':
                    stats.no_implementation_count += 1
                
                refinement = material_data.get('refinement', {})
                if refinement.get('reason') == 'below_coverage_threshold':
                    stats.below_threshold_count += 1
                elif refinement.get('reason') == 'not_in_canary_set':
                    stats.not_in_canary_count += 1
            
            logger.info(f"✓ Processed {scene_name}")
            
        except Exception as e:
            logger.error(f"Error processing {report_file}: {e}")
            continue
    
    # Add collection metadata
    git_info = get_git_info()
    aggregated.collection_metadata = {
        'timestamp': datetime.now().isoformat(),
        'git_commit': git_info['commit'],
        'git_branch': git_info['branch'],
        'total_scenes_analyzed': aggregated.total_scenes,
        'scene_names': aggregated.scene_names
    }
    
    return aggregated


def save_json_report(data: AggregatedData, output_path: Path):
    """Save aggregated data as JSON."""
    output_data = {
        'metadata': data.collection_metadata,
        'histograms': {
            'pixel_ops_reasons': dict(data.pixel_ops_reasons),
            'refinement_reasons': dict(data.refinement_reasons)
        },
        'materials': {
            name: {
                'frequency': stats.frequency,
                'scenes_present': stats.scenes_present,
                'avg_coverage': round(stats.avg_coverage, 2),
                'avg_boundary_pixels': round(stats.avg_boundary_pixels, 1),
                'avg_edge_alignment': round(stats.avg_edge_alignment, 3),
                'counts': {
                    'no_implementation': stats.no_implementation_count,
                    'below_threshold': stats.below_threshold_count,
                    'not_in_canary': stats.not_in_canary_count
                },
                'recommendation_score': round(stats.recommendation_score(), 2),
                'recommendation_tier': stats.recommendation_tier()
            }
            for name, stats in data.materials.items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"✓ Saved JSON report to {output_path}")


def generate_markdown_report(data: AggregatedData, output_path: Path):
    """Generate human-readable markdown analysis."""
    
    # Sort materials by recommendation score
    ranked_materials = sorted(
        data.materials.values(),
        key=lambda m: m.recommendation_score(),
        reverse=True
    )
    
    # Find top recommendation
    top_material = ranked_materials[0] if ranked_materials else None
    
    with open(output_path, 'w') as f:
        # Header
        f.write("# PR-4D Material Selection Analysis\n\n")
        f.write(f"**Generated**: {data.collection_metadata['timestamp']}\n")
        f.write(f"**Git Commit**: {data.collection_metadata['git_commit']}\n")
        f.write(f"**Scenes Analyzed**: {data.total_scenes}\n\n")
        
        # Aggregated histograms
        f.write("## Aggregated Histograms\n\n")
        f.write(f"**Total Scenes**: {data.total_scenes}\n\n")
        
        f.write("### Pixel Ops Reasons\n\n")
        for reason, count in sorted(data.pixel_ops_reasons.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- `{reason}`: {count}\n")
        f.write("\n")
        
        f.write("### Refinement Reasons\n\n")
        for reason, count in sorted(data.refinement_reasons.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- `{reason}`: {count}\n")
        f.write("\n")
        
        # Material statistics table
        f.write("## Material Statistics\n\n")
        f.write("| Material | Frequency | Avg Coverage | Boundary Px | Edge Align | no_impl | Recommendation |\n")
        f.write("|----------|-----------|--------------|-------------|------------|---------|----------------|\n")
        
        for stats in ranked_materials:
            freq_str = f"{stats.frequency}/{data.total_scenes}"
            coverage_str = f"{stats.avg_coverage:.1f}%"
            boundary_str = f"{stats.avg_boundary_pixels:.0f}"
            edge_str = f"{stats.avg_edge_alignment:.2f}"
            f.write(f"| {stats.name} | {freq_str} | {coverage_str} | {boundary_str} | {edge_str} | "
                   f"{stats.no_implementation_count} | {stats.recommendation_tier()} |\n")
        
        f.write("\n")
        
        # Detailed recommendation
        if top_material and top_material.name != 'glass':
            f.write(f"## Recommendation: {top_material.name.title()}\n\n")
            f.write("**Rationale**:\n\n")
            
            # Frequency check
            if top_material.frequency == data.total_scenes:
                f.write(f"- ✅ Appears in all {data.total_scenes} scenes (100% frequency)\n")
            else:
                f.write(f"- ✅ Appears in {top_material.frequency}/{data.total_scenes} scenes "
                       f"({100*top_material.frequency/data.total_scenes:.0f}% frequency)\n")
            
            # Coverage check
            if top_material.avg_coverage >= 5.0:
                f.write(f"- ✅ High coverage ({top_material.avg_coverage:.1f}% avg) - good sample size\n")
            else:
                f.write(f"- ⚠️ Moderate coverage ({top_material.avg_coverage:.1f}% avg)\n")
            
            # Edge signals check
            if top_material.avg_boundary_pixels >= 250 and top_material.avg_edge_alignment >= 0.10:
                f.write(f"- ✅ Strong edge signals (boundary_pixels: {top_material.avg_boundary_pixels:.0f}, "
                       f"edge_alignment: {top_material.avg_edge_alignment:.2f})\n")
            else:
                f.write(f"- ⚠️ Moderate edge signals (boundary_pixels: {top_material.avg_boundary_pixels:.0f}, "
                       f"edge_alignment: {top_material.avg_edge_alignment:.2f})\n")
            
            # Implementation need
            if top_material.no_implementation_count > 0:
                f.write(f"- ✅ `no_implementation` in {top_material.no_implementation_count} scene(s) (clear need)\n")
            
            # Scenes present
            f.write(f"- ✅ Present in scenes: {', '.join(top_material.scenes_present)}\n")
            
            # Halo risk assessment
            if 'Pool' not in ' '.join(top_material.scenes_present):
                f.write("- ✅ Low halo risk (stable boundaries in interior scenes)\n")
            else:
                f.write("- ⚠️ Halo risk consideration (present in Pool scene - check edge quality)\n")
            
            f.write("\n**Next Steps**:\n\n")
            material_snake = top_material.name.lower().replace(' ', '_')
            f.write(f"1. Implement {top_material.name} pixel ops in `lux_depth_v2/materials_v3_pixel_ops.py`\n")
            f.write(f"2. Add {top_material.name} eligibility to `lux_depth_v2/materials_v3_response.py`\n")
            f.write(f"3. Create canary preset: `interior_luxury_apex_quality_materials_v3_{material_snake}`\n")
            f.write(f"4. Create validation preset: `interior_luxury_apex_quality_materials_v3_{material_snake}_validate`\n")
            f.write(f"5. Run two-pass validation (normal + forced-apply)\n")
            f.write(f"6. Review boundary quality in all scenes (especially if Pool present)\n")
            f.write(f"7. Monitor for halo artifacts during validation\n")
            
        elif top_material and top_material.name == 'glass':
            f.write("## Note: Glass Already Implemented\n\n")
            f.write("Glass ranks highest but is already implemented in PR-4C.\n")
            f.write(f"Next candidate: **{ranked_materials[1].name.title()}**\n\n")
            
            # Show next candidate details
            next_material = ranked_materials[1]
            f.write("**Next Candidate Rationale**:\n\n")
            f.write(f"- Frequency: {next_material.frequency}/{data.total_scenes}\n")
            f.write(f"- Coverage: {next_material.avg_coverage:.1f}%\n")
            f.write(f"- Edge Signals: {next_material.avg_boundary_pixels:.0f} boundary px, "
                   f"{next_material.avg_edge_alignment:.2f} alignment\n")
            f.write(f"- no_implementation: {next_material.no_implementation_count}\n")
        
        f.write("\n")
        
        # Scene-by-scene breakdown
        f.write("## Scene-by-Scene Breakdown\n\n")
        for scene_name in data.scene_names:
            f.write(f"### {scene_name}\n\n")
            scene_materials = [m for m in data.materials.values() if scene_name in m.scenes_present]
            if scene_materials:
                f.write("**Materials Present**:\n\n")
                for mat in sorted(scene_materials, key=lambda m: m.avg_coverage, reverse=True):
                    idx = mat.scenes_present.index(scene_name)
                    # Note: We don't have per-scene coverage stored, so we use average
                    f.write(f"- {mat.name}: ~{mat.avg_coverage:.1f}% coverage\n")
            f.write("\n")
    
    logger.info(f"✓ Saved markdown report to {output_path}")


def main():
    """Main execution function."""
    logger.info("PR-4D Histogram Aggregation Started")
    logger.info("=" * 50)
    
    # Configuration
    base_dir = Path("outputs/pr4d_wood_data")
    json_output = Path("outputs/pr4d_histogram_aggregate.json")
    md_output = Path("outputs/pr4d_material_recommendations.md")
    
    # Validate base directory exists
    if not base_dir.exists():
        logger.error(f"Base directory not found: {base_dir}")
        logger.error("Run pr4d_collect_wood_histograms.sh first to collect data")
        return 1
    
    # Find and aggregate reports
    logger.info(f"Searching for reports in {base_dir}")
    report_files = find_report_files(base_dir)
    
    if not report_files:
        logger.error("No report files found!")
        return 1
    
    logger.info(f"Processing {len(report_files)} report files...")
    aggregated = aggregate_reports(report_files)
    
    if aggregated.total_scenes == 0:
        logger.error("No valid reports processed!")
        return 1
    
    logger.info(f"Successfully aggregated {aggregated.total_scenes} scenes")
    logger.info(f"Total materials found: {len(aggregated.materials)}")
    
    # Generate outputs
    logger.info("Generating outputs...")
    save_json_report(aggregated, json_output)
    generate_markdown_report(aggregated, md_output)
    
    # Summary
    logger.info("=" * 50)
    logger.info("Aggregation Complete!")
    logger.info(f"Scenes Analyzed: {aggregated.total_scenes}")
    logger.info(f"Materials Found: {len(aggregated.materials)}")
    logger.info(f"JSON Report: {json_output}")
    logger.info(f"Markdown Report: {md_output}")
    
    # Show top recommendation
    if aggregated.materials:
        top_material = max(aggregated.materials.values(), key=lambda m: m.recommendation_score())
        logger.info(f"\nTop Recommendation: {top_material.name.upper()}")
        logger.info(f"  Score: {top_material.recommendation_score():.1f}/100")
        logger.info(f"  Frequency: {top_material.frequency}/{aggregated.total_scenes}")
        logger.info(f"  Avg Coverage: {top_material.avg_coverage:.1f}%")
    
    return 0


if __name__ == '__main__':
    exit(main())
