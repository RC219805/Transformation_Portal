# Phase 1 Execution - Ready to Launch

**Date**: 2025-12-22
**Branch**: `exploration/parameter-sweep-documentation`
**Baseline Commit**: `0779a57`
**Status**: ✅ **ALL SYSTEMS GO**

---

## ✅ Pre-Flight Verification Complete

### Environment Status
- ✅ Branch: `exploration/parameter-sweep-documentation`
- ✅ Dependencies: All installed (numpy, cv2, tifffile, torch, tqdm)
- ✅ GPU: MPS (Apple Silicon) available with fallback enabled
- ✅ Input TIFFs: 6 files verified in `projects/750_picacho_lane/Final_Production_UltraQuality/`
- ✅ Sweep runner: `exploration/sweep_runner.py` operational
- ✅ Test script: `lux_depth_v2/test_750_picacho.py` validated

### Baseline Generation Complete
- ✅ **32 output files** generated (6 images × ~5 formats each)
- ✅ **7.9GB total** output size
- ✅ All reports show `status: "ok"`
- ✅ Processing time: 15-27 seconds per image
- ✅ Baseline parameters saved: `sweep_runs/baseline/params.json`
- ✅ Notes template created: `sweep_runs/baseline/notes.md`

### Baseline Configuration (Commit 0779a57)
```json
{
  "depth_percentile_clip_low": 0.5,
  "depth_percentile_clip_high": 99.5,
  "depth_gamma": 1.0,
  "edge_filter_radius": 5,
  "edge_filter_sigma_color": 75,
  "edge_filter_sigma_space": 75,
  "banding_suppression": 0.005,
  "materials_confidence_curve": "linear",
  "materials_edge_weight": 1.0,
  "materials_low_confidence_suppress": null,
  "local_contrast_gain": 2.0,
  "saturation_protection": 1.0
}
```

**Important Note**: The baseline correctly shows:
- `materials_v3_enabled: false` - This is EXPECTED (baseline doesn't use Materials V3)
- `depth: null` - This is EXPECTED (baseline doesn't use depth-aware processing)

The Phase 1 sweeps will test whether **adding** these features improves quality.

---

## 🚀 Execution Options

### Option 1: Full Phase 1 with Live Monitoring (RECOMMENDED)
**Duration**: ~2-4 hours
**Sweeps**: All 9 single-parameter sweeps (Depth × 4, Materials V3 × 3, Color/Tone × 2)
**Output**: 27 run directories + baseline

```bash
bash exploration/phase1_live_monitor.sh --all
```

**Features**:
- ✅ Real-time progress updates every 30 seconds
- ✅ Live metrics (elapsed time, sweeps completed, output count)
- ✅ Full execution log saved automatically
- ✅ Final summary with processing statistics

---

### Option 2: Standard Full Phase 1 (No Live Monitoring)
```bash
bash exploration/execute_phase1.sh --all
```

---

### Option 3: Category-Specific Execution

**Depth Parameters Only** (~30-45 minutes):
```bash
bash exploration/phase1_live_monitor.sh --depth
# Or without monitoring:
bash exploration/execute_phase1.sh --depth
```

**Materials V3 Only** (~20-30 minutes):
```bash
bash exploration/phase1_live_monitor.sh --materials
```

**Color/Tone Only** (~15-20 minutes):
```bash
bash exploration/phase1_live_monitor.sh --color
```

---

## 📊 Expected Output Structure

After Phase 1 completion:

```
sweep_runs/
├── baseline/                          # ✅ Already generated
│   ├── outputs/                       # 32 files, 7.9GB
│   ├── params.json
│   └── notes.md
│
├── depth_gamma_delta0/                # Baseline gamma value
├── depth_gamma_delta1/                # Delta-1 (gamma=1.1)
├── depth_gamma_delta2/                # Delta-2 (gamma=0.9)
├── depth_percentile_clip_low_delta0/
├── depth_percentile_clip_low_delta1/
├── depth_percentile_clip_low_delta2/
├── depth_edge_filter_sigma_color_delta0/
├── depth_edge_filter_sigma_color_delta1/
├── depth_edge_filter_sigma_color_delta2/
├── depth_banding_suppression_delta0/
├── depth_banding_suppression_delta1/
├── depth_banding_suppression_delta2/
├── materials_v3_confidence_curve_delta0/
├── materials_v3_confidence_curve_delta1/
├── materials_v3_confidence_curve_delta2/
├── materials_v3_edge_alignment_weight_delta0/
├── materials_v3_edge_alignment_weight_delta1/
├── materials_v3_edge_alignment_weight_delta2/
├── materials_v3_low_confidence_threshold_delta0/
├── materials_v3_low_confidence_threshold_delta1/
├── materials_v3_low_confidence_threshold_delta2/
├── color_tone_local_contrast_gain_delta0/
├── color_tone_local_contrast_gain_delta1/
├── color_tone_local_contrast_gain_delta2/
├── color_tone_saturation_protection_delta0/
├── color_tone_saturation_protection_delta1/
└── color_tone_saturation_protection_delta2/

exploration/sweep_archive/2025-12-22/
└── [Complete archive of all sweep results + PHASE1_SUMMARY.txt]
```

**Total**: 28 directories (1 baseline + 27 sweep runs)

---

## 📝 Post-Execution Review Process

### 1. Automated Archive
Results will be automatically archived to:
```
exploration/sweep_archive/2025-12-22/
```

### 2. Review Quality
For each sweep directory, review:
```bash
# View notes and qualitative assessment
cat sweep_runs/depth_gamma_delta1/notes.md

# Check metrics
cat sweep_runs/depth_gamma_delta1/metrics.json

# Compare outputs side-by-side with baseline
ls sweep_runs/baseline/outputs/
ls sweep_runs/depth_gamma_delta1/outputs/
```

### 3. Decision Gate
For each parameter sweep, decide:
- ✅ **SHIP**: Measurable improvement, zero artifacts → Candidate for Phase 2
- 📦 **ARCHIVE**: No improvement or artifacts present → Keep for reference
- 🔄 **REFINE**: Promising but needs adjustment → Re-sweep with different deltas

### 4. Phase 2 Consideration
If 2-3 parameters show clear improvement:
- Proceed to **Phase 2**: Combined parameter sweeps
- Test interactions between winning parameters
- Maximum 15 combinations (not full combinatorial)

---

## 🔍 Quality Validation Criteria

Each sweep must demonstrate:

1. **Visual Improvement** (Qualitative)
   - Depth edges sharper without halos
   - Materials V3 masks align with actual surfaces
   - Color/tone natural (not "AI-enhanced" look)

2. **Zero New Artifacts** (Critical)
   - No banding in skies/gradients
   - No color shifts or saturation clipping
   - No edge halos or ringing
   - 16-bit precision maintained

3. **Reversible** (Safety)
   - Can return to baseline by rolling back parameters
   - Changes are config-only (no code modifications)

4. **Explainable** (Documentation)
   - Can justify improvement in <1 paragraph
   - Clear cause-and-effect relationship

---

## 📋 Troubleshooting

### If a Sweep Fails
```bash
# Check the specific parameter's output
ls -la sweep_runs/depth_gamma_delta1/

# View error logs
cat sweep_runs/phase1_execution_*.log | grep -A5 "error\|failed"

# Re-run just that parameter
python exploration/sweep_runner.py --mode single --category depth --parameter gamma
```

### Check Disk Space
```bash
# Phase 1 will generate ~200-250GB total
df -h .
```

### Monitor Progress Manually
```bash
# Watch output directories being created
watch -n 10 "ls -lh sweep_runs/ | tail -20"

# Check process status
ps aux | grep sweep_runner
```

---

## ✅ Ready to Execute

**Baseline verified and approved.**
**All systems operational.**
**Phase 1 sweep infrastructure ready.**

**Recommended command**:
```bash
bash exploration/phase1_live_monitor.sh --all
```

This will execute all 9 single-parameter sweeps with live progress monitoring, completing in ~2-4 hours.

---

**Questions before starting?**
- Estimated disk space available: ~500GB recommended (250GB minimum)
- Estimated completion time: 2-4 hours for full Phase 1
- Can be interrupted and resumed (sweeps are independent)
- All work stays on exploration branch (main branch locked at 0779a57)
