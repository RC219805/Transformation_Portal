# Water Detection Validation Dataset (v0)

**Version**: v0  
**Created**: 2024-12-14  
**Updated**: 2025-12-15 (Baseline v1 added)

## Overview

Synthetic water detection test dataset for pool and ocean scenes with negative controls (blue walls, glass buildings).

## Dataset Structure

```
data/water_v0/
├── images/
│   ├── pool/          # 6 pool scenes
│   │   ├── pool_0001.jpg ... pool_0009.jpg
│   │   └── neg_blue_wall_0001.jpg (negative control)
│   └── ocean/         # 6 ocean scenes
│       ├── ocean_0001.jpg ... ocean_0009.jpg
│       └── neg_glass_building_0001.jpg (negative control)
├── ground_truth.json  # Labels, should_detect flags, difficulty, tags
├── baseline_ci_v0.json  # Single threshold baseline (83.3% pool recall)
└── baseline_ci_v1.json  # Two-stage gating baseline (100% pool recall)
```

## Ground Truth Schema

```json
{
  "version": "unknown",
  "root": "data/water_v0/images",
  "images": {
    "pool/pool_0001.jpg": {
      "label": "pool",
      "should_detect": true,
      "difficulty": "easy",
      "tags": ["synthetic"]
    },
    "pool/neg_blue_wall_0001.jpg": {
      "label": "pool",
      "should_detect": false,
      "difficulty": "hard",
      "tags": ["synthetic", "negative_control", "flat_blue_surface"]
    }
  }
}
```

### Fields

- **label**: `pool` | `ocean` (scene type, NOT detection label)
- **should_detect**: `true` (water) | `false` (hard negative)
- **difficulty**: `easy` | `medium` | `hard`
- **tags**: `["synthetic", "negative_control", "flat_blue_surface", "glass_grid", ...]`

## Baselines

### v0 (Single Threshold)

**Date**: 2024-12-14  
**Config**: `water_candidate_confidence_threshold: 0.4` (single gate)

**Results**:
- Pool recall: 83.3% (5/6) - missed pool_0008
- Ocean recall: 100% (6/6)
- False trigger rate: 0% (0/2)

**File**: `baseline_ci_v0.json`

### v1 (Two-Stage Gating)

**Date**: 2025-12-15  
**Config**:
- `water_candidate_threshold: 0.25` (Stage A - candidate detection)
- `water_candidate_confidence_threshold: 0.4` (Stage B - injection decision)
- `water_saturation_boost_enabled: True` (+0.15 for low-sat pools)

**Results**:
- Pool recall: **100% (6/6)** ← recovered pool_0008
- Ocean recall: 100% (6/6)
- False trigger rate: 0% (0/2)

**File**: `baseline_ci_v1.json`

**Key Difference**: v1 recovered pool_0008 via saturation boost (0.255 + 0.15 = 0.405 > 0.4)

## Running Validation

```bash
python scripts/prw_water_validation.py \
  --ground-truth data/water_v0/ground_truth.json \
  --output report.json \
  --seed 42
```

**Output**: JSON report with per-image results and summary statistics.

## Test Cases

### Positive Controls (should_detect=true)

**Pools** (6):
- pool_0001, pool_0003, pool_0005, pool_0007, pool_0009: Clear pools (easy/medium)
- pool_0008: Low-saturation pool (hard) - **recovered in v1**

**Oceans** (6):
- ocean_0001, ocean_0003, ocean_0004, ocean_0005, ocean_0007, ocean_0009: Ocean scenes (easy/medium)

### Negative Controls (should_detect=false)

**Hard Negatives** (2):
- `neg_blue_wall_0001.jpg`: Flat blue painted wall (tests flat surface suppressor)
- `neg_glass_building_0001.jpg`: Blue-tinted glass building (tests glass suppressor)

**Expected**: Both should trigger suppressors and be correctly rejected.

## Key Metrics

### Recall
- **Pool recall**: Fraction of pools detected (target: 100%)
- **Ocean recall**: Fraction of oceans detected (target: 100%)

### Precision
- **False trigger rate**: Fraction of negatives incorrectly detected (target: 0%)

### Coverage
- **coverage_all**: Raw coverage (all images, including misses)
- **coverage_detected**: Coverage only when detected (cleaner metric)

### Confidence (v1 only)
- **confidence_raw**: Pre-suppressor confidence
- **confidence_after_suppressors**: Post-suppressor, pre-boost
- **confidence_final**: Post-boost, pre-injection gate

## Validation History

| Date | Version | Pool Recall | Ocean Recall | FT Rate | Notes |
|------|---------|-------------|--------------|---------|-------|
| 2024-12-14 | v0 | 83.3% (5/6) | 100% (6/6) | 0% | Missed pool_0008 |
| 2025-12-15 | v1 | **100% (6/6)** | 100% (6/6) | 0% | Two-stage gating |

## References

- **Implementation**: `TWO_STAGE_GATING_IMPLEMENTATION_SUMMARY.md`
- **Comparison**: `BASELINE_V0_VS_V1_COMPARISON.md`
- **Harness**: `scripts/prw_water_validation.py`
- **Detector**: `lux_depth_v2/water_candidate.py`
- **Integration**: `lux_depth_v2/materials_v3.py`
