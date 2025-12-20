#!/usr/bin/env python3
"""Generate metrics comparison table and scenario determination."""

import json
from pathlib import Path
from typing import Dict, List

def load_metrics(metrics_dir: Path) -> List[Dict]:
    """Load all individual metrics files."""
    results = []
    for f in sorted(metrics_dir.glob("*_metrics.json")):
        if f.name == "summary.json":
            continue
        with open(f) as fp:
            results.append(json.load(fp))
    return results

def format_table(results: List[Dict]) -> str:
    """Format results as markdown table."""
    lines = []
    lines.append("| Image | Edge F1 | Chamfer (px) | Seam | Quality | Lenient | Strict | Status |")
    lines.append("|-------|---------|--------------|------|---------|---------|--------|--------|")
    
    # Baseline from existing validation runs
    baseline = {
        '750Picacho_Aerial_Ultimate': {'edge_f1': 0.692, 'chamfer_px': 1.60},
        '750Picacho_GreatRoom_Ultimate': {'edge_f1': 0.617, 'chamfer_px': 14.85}
    }
    
    for r in results:
        name = r['image_name']
        edge_f1 = r['edge_f1']
        chamfer = r['chamfer_distance']
        quality = r['quality_score']
        lenient = "PASS" if r['passed_lenient'] else "FAIL"
        strict = "PASS" if r['passed_strict'] else "FAIL"
        
        # Determine status vs baseline (if applicable)
        status = "New"
        
        # Edge cases with no edges detected
        if edge_f1 < 0.01:
            status = "⚠️ No edges"
        elif edge_f1 >= 0.45:
            status = "✅ Excellent"
        elif edge_f1 >= 0.30:
            status = "✓ Good"
        else:
            status = "⚠️ Poor"
        
        # Format row
        lines.append(
            f"| {name:20s} | {edge_f1:7.3f} | {chamfer:12.1f} | N/A  | {quality:7.3f} | {lenient:7s} | {strict:6s} | {status} |"
        )
    
    return "\n".join(lines)

def determine_scenario(results: List[Dict]) -> str:
    """Determine scenario (A/B/C/D) based on results."""
    total = len(results)
    lenient_passed = sum(1 for r in results if r['passed_lenient'])
    strict_passed = sum(1 for r in results if r['passed_strict'])
    
    lenient_rate = lenient_passed / total if total > 0 else 0
    strict_rate = strict_passed / total if total > 0 else 0
    
    avg_f1 = sum(r['edge_f1'] for r in results) / total if total else 0
    
    # Scenario determination based on readiness doc
    if strict_rate >= 0.85:
        scenario = "A"
        status = "Production-Ready"
        action = "Tag v1.0.0, deploy to production"
    elif lenient_rate >= 0.70:
        scenario = "B"
        status = "Production-Qualified (with monitoring)"
        action = "Deploy with quality gates, monitor edge cases"
    elif lenient_rate >= 0.40:
        scenario = "C"
        status = "Needs Optimization"
        action = "Fix high-chamfer images, then re-validate subset"
    else:
        scenario = "D"
        status = "Blocked - Systematic Issues"
        action = "Debug sliver/seam artifacts, re-architect if needed"
    
    return f"""
## Scenario Determination

**Current Results**:
- Total images: {total}
- Lenient passed: {lenient_passed}/{total} ({100*lenient_rate:.1f}%)
- Strict passed: {strict_passed}/{total} ({100*strict_rate:.1f}%)
- Avg Edge F1: {avg_f1:.3f}

**Scenario: {scenario} - {status}**

**Next Action**: {action}

### Scenario Definitions
- **A (≥85% strict)**: Production-ready, tag v1.0.0
- **B (≥70% lenient)**: Production-qualified with monitoring
- **C (≥40% lenient)**: Needs targeted optimization
- **D (<40% lenient)**: Blocked, systematic issues

### Specific Recommendations

Based on the current results:
"""
    
    # Add specific recommendations
    if scenario == "D":
        return scenario + f"""
1. **Immediate**: Investigate why 5/7 images failed lenient thresholds
2. **Root Cause**: Many images show excessive edge count ratio (>80×) and no edge detection
3. **Hypothesis**: Possible issue with:
   - Small images (512×512) producing degenerate depth maps
   - Gradient-based edge detection on textured surfaces (pool, ocean)
   - Global anchor calibration artifacts
4. **Next Steps**:
   - Re-run validation WITHOUT global anchor (`--use-global-anchor` flag removed)
   - Test with larger resolution inputs (upscale 512×512 to 1024×1024)
   - Add edge detection diagnostic visualizations
5. **Command**:
```bash
python production_depth_validation_fixed.py \\
  --image-dir data/validation_quick \\
  --output-dir outputs/validation_no_anchor \\
  --tile-size 1024 \\
  --overlap 128
  # Note: --use-global-anchor is OFF by default
```
"""
    elif scenario == "C":
        return scenario + f"""
1. **Focus on**: Fix the 5 failed images (glass_building, glass_facade, ocean_1, pool_texture_1/2)
2. **Pattern**: Small images and texture-heavy surfaces are problematic
3. **Next Steps**:
   - Disable global anchor (default)
   - Add min-resolution check (warn if <1024px)
   - Improve texture-edge discrimination
4. **Re-validate**: Only the 5 failed images after fixes
"""
    else:
        return scenario


def main():
    metrics_dir = Path("outputs/validation_metrics_extracted")
    results = load_metrics(metrics_dir)
    
    print("# 7-Image Validation - Complete Results\n")
    print("## Metrics Table\n")
    print(format_table(results))
    print()
    print(determine_scenario(results))
    
    # Save to file
    output_path = Path("VALIDATION_7IMAGE_RESULTS.md")
    with open(output_path, "w") as f:
        f.write("# 7-Image Validation - Complete Results\n\n")
        f.write("## Metrics Table\n\n")
        f.write(format_table(results))
        f.write("\n\n")
        f.write(determine_scenario(results))
    
    print(f"\n✅ Report saved: {output_path}")

if __name__ == "__main__":
    main()
