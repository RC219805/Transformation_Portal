# Quick Start: Production Validation

## TL;DR

```bash
# Run production validation (takes ~3-4 hours for 6 images at 2048px)
cd /Users/rc/Transformation_Portal
python production_validation_750_picacho.py \
    --preset production \
    --max-dimension 2048

# Results will be in:
# - outputs/production_validation/production_validation_report.json
# - outputs/production_validation/<image_name>/ (visual outputs for priority scenes)
```

## What It Does

Validates the high-fidelity depth pipeline on the full 750_Picacho dataset with:

- ✅ Edge F1 + precision/recall breakdown
- ✅ Halo/overshoot detection
- ✅ Detail benefit metric
- ✅ Seam validation
- ✅ Yellow flag detection
- ✅ Materials V3 readiness assessment

## Presets

```bash
# Fast preview (fastest, lower quality)
python production_validation_750_picacho.py --preset preview --max-dimension 1024

# Production (balanced - RECOMMENDED)
python production_validation_750_picacho.py --preset production --max-dimension 2048

# Hero (highest quality, slowest)
python production_validation_750_picacho.py --preset hero --max-dimension 2048
```

## Options

```bash
--input-dir PATH          # Input directory (default: 750_Picacho/Source_TIFFs_Base)
--output-dir PATH         # Output directory (default: outputs/production_validation)
--preset {preview,production,hero}  # Processing preset
--max-dimension INT       # Max dimension for resize (default: 4096, use 2048 for memory safety)
--save-all-visuals        # Save visual outputs for ALL images (not just priority scenes)
```

## Output Structure

```
outputs/production_validation/
├── production_validation_report.json  # Comprehensive JSON report
├── 750Picacho_Kitchen_16bit/          # Per-image outputs (priority scenes)
│   ├── depth_baseline.tiff
│   ├── depth_tiled.tiff
│   ├── depth_final.tiff
│   └── edge_overlay.png
├── 750Picacho_Pool_16bit/
│   └── ...
└── ...
```

## Report Contents

### JSON Report Schema

```json
{
  "dataset": "...",
  "preset": { "name": "Production", "tile_size": 1024, ... },
  "timestamp": "2025-12-17 20:00:00",
  "summary": {
    "total_images": 6,
    "processed": 6,
    "lenient_pass": 5,
    "strict_pass": 3,
    "lenient_pass_rate": 0.833,
    "strict_pass_rate": 0.500
  },
  "aggregate_metrics": {
    "edge_f1": { "mean": 0.625, "std": 0.082, "min": 0.511, "max": 0.732, ... },
    "edge_precision": { ... },
    "edge_recall": { ... },
    "chamfer_distance": { ... },
    "halo_score": { ... },
    "detail_benefit": { ... }
  },
  "per_image_metrics": {
    "750Picacho_Kitchen_16bit.tiff": { "edge_f1": 0.625, ... },
    ...
  },
  "priority_scenes": { ... },
  "yellow_flags": [
    {
      "flag": "PRECISION_LOW",
      "description": "...",
      "impact": "...",
      "recommendation": "..."
    }
  ],
  "materials_v3_ready": true,
  "go_no_go_decision": {
    "recommendation": "GO" | "INVESTIGATE" | "NO-GO",
    "criteria": { ... }
  }
}
```

## Interpreting Results

### Quality Gates (Materials V3 Readiness)

**Lenient Mode (80% pass required):**
- Edge F1 ≥ 0.30
- Edge Precision ≥ 0.25
- Edge Recall ≥ 0.25
- Edge Count Ratio ≤ 3.0
- Seam Passed: True
- Overshoot Penalty ≤ 0.50

**Strict Mode (ideal target):**
- Edge F1 ≥ 0.45
- Edge Precision ≥ 0.40
- Edge Recall ≥ 0.40
- Edge Count Ratio ≤ 2.0
- Seam Passed: True
- Halo Score ≥ 0.70
- Overshoot Penalty ≤ 0.30
- Detail Benefit ≥ 1.0

### Yellow Flags

| Flag | Meaning | Action |
|------|---------|--------|
| `PRECISION_LOW` | Too many false positive edges | Reduce edge snap strength, review global anchor |
| `DETAIL_BENEFIT_LOW` | Tiling adds noise, not detail | Disable global anchor on low-detail scenes |
| `OVERSHOOT_HIGH` | Visible ringing/halo artifacts | Reduce edge snap strength |
| `EDGE_EXPLOSION` | Too many artifact edges | Check global sharpening settings |

### Go/No-Go Recommendation

- **GO:** Pass rate ≥ 80%, no critical yellow flags → Ready for Materials V3
- **INVESTIGATE:** Pass rate ≥ 80%, but yellow flags present → Fix flags first
- **NO-GO:** Pass rate < 80% → Pipeline needs debugging

## Priority Scenes

The script automatically saves visual outputs for these scenes:
- `750Picacho_Kitchen_16bit.tiff` (glass/metal complexity)
- `750Picacho_GreatRoom_Ultimate.tif` (large planar structures)
- `750Picacho_Aerial_Ultimate.tif` (different scale/context)
- `750Picacho_Pool_16bit.tiff` (baseline reference)

## Memory Considerations

### Current Limitation

- **4K+ native processing:** Causes OOM (even on 64GB M4 Max)
- **Recommended:** Use `--max-dimension 2048` for reliability
- **Future:** Streaming architecture for native 4K+ (1-2 weeks)

### Expected Runtime

| Resolution | Time per Image | Total (6 images) |
|------------|----------------|------------------|
| 1024px | ~30s | ~3 minutes |
| 2048px | ~5 minutes | ~30 minutes |
| 4096px | OOM | N/A (blocked) |

## Troubleshooting

### "Invalid buffer size" error
→ Reduce `--max-dimension` (try 1024 or 2048)

### "No images successfully processed"
→ Check input directory path, ensure images exist

### Script hangs/killed
→ Out of memory - reduce `--max-dimension`

### Missing visual outputs
→ Only priority scenes get visuals by default (use `--save-all-visuals` for all)

## Example: Full Validation Run

```bash
# Complete production validation
python production_validation_750_picacho.py \
    --input-dir /Users/rc/Transformation_Portal/input_images/750_Picacho/Source_TIFFs_Base \
    --output-dir outputs/production_validation_$(date +%Y%m%d_%H%M%S) \
    --preset production \
    --max-dimension 2048

# Check results
cat outputs/production_validation_*/production_validation_report.json | jq '.go_no_go_decision'

# View priority scene overlays
open outputs/production_validation_*/750Picacho_Kitchen_16bit/edge_overlay.png
open outputs/production_validation_*/750Picacho_Pool_16bit/edge_overlay.png
```

## Next Steps After Validation

1. **Review JSON report** - Check pass rates and yellow flags
2. **Inspect visual gallery** - Verify edge alignment in priority scenes
3. **Investigate yellow flags** - Follow recommendations if any
4. **Materials V3 handoff** - If GO decision, proceed with integration
5. **Optimize memory** - If needed for native 4K+ support

## Support

See comprehensive documentation in:
- `PRODUCTION_VALIDATION_COMPLETE.md` - Full report with findings
- `PRODUCTION_VALIDATION_EXECUTION_SUMMARY.md` - Execution summary
- Code: `production_validation_750_picacho.py` - Well-documented script

---

**Quick Check:**
```bash
# Verify script is ready
python production_validation_750_picacho.py --help

# Run on single priority image (fastest test)
# (Note: Script processes full dataset, but you can Ctrl+C after first image for quick test)
```
