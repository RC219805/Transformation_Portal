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
├── ground_truth.json          # Labels, should_detect flags, difficulty, tags
├── baseline_ci_audit_v0.json  # Immutable historical baseline (100% pool recall)
└── baseline_ci_current_v1.json # Current enforced baseline (83.3% pool recall)
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

### Baseline Governance Scheme

**Audit Baselines** (Immutable):
- Historical reference baselines for regression analysis
- Never modified after creation
- Used for tracking long-term performance trends

**Current Baselines** (Mutable with Validation):
- Actively enforced baseline for CI regression checks
- Can be regenerated with validated threshold changes
- Requires holdout validation before modification

---

### baseline_ci_audit_v0.json (AUDIT - IMMUTABLE)

**Date**: 2024-12-14  
**Purpose**: Historical baseline from initial two-stage gating implementation  
**Status**: ✅ Immutable (historical audit baseline)

**Configuration**:
- Two-stage gating with saturation boost
- `water_candidate_threshold: 0.25` (Stage A)
- `water_candidate_confidence_threshold: 0.4` (Stage B)
- `water_saturation_boost_enabled: True` (+0.15 boost)
- Glass suppressor: alignment=0.15, grid=0.25, penalty=0.6

**Results**:
- **Pool recall**: 100% (6/6) - including pool_0008
- **Ocean recall**: 100% (6/6)
- **False trigger rate**: 0% (0/2)

**Why 100% Pool Recall**: Saturation boost recovered pool_0008 (0.255 + 0.15 = 0.405 > 0.4)

---

### baseline_ci_current_v1.json (CURRENT - ENFORCED BY CI)

**Date**: 2025-12-15 (Governance cleanup)  
**Purpose**: Current enforced baseline with safe, conservative thresholds  
**Status**: ⚠️ Mutable (requires holdout validation for updates)

**Configuration**:
- Two-stage gating with saturation boost
- `water_candidate_threshold: 0.25` (Stage A)
- `water_candidate_confidence_threshold: 0.4` (Stage B)
- `water_saturation_boost_enabled: True` (+0.15 boost)
- Glass suppressor: alignment=**0.15**, grid=**0.25**, penalty=**0.6** (safe, conservative)

**Results**:
- **Pool recall**: 83.3% (5/6) - **pool_0008 missed** (known limitation)
- **Ocean recall**: 100% (6/6)
- **False trigger rate**: 0% (0/2)

**Known Limitation**: pool_0008 (low-saturation pool with subtle tile grid)
- Requires lower glass suppressor alignment threshold (0.11) to detect tiles
- Current threshold (0.15) is conservative to prevent architectural glass false positives
- **Mitigation**: Experimental multi-scale logic exists but not validated with holdout set
- **Decision**: Accept 83.3% pool recall until proper validation framework exists

**CI Integration**: `.github/workflows/ci-consolidated.yml` references this baseline

---

### Baseline Update Policy

**To update baseline_ci_current_v1.json**:

1. **Create holdout validation set** (PR-W1.2):
   - 10-20 real architectural glass negatives
   - Additional low-saturation pools (real-world diversity)
   - Ensure no overlap with test set

2. **Validate threshold changes**:
   - ROC analysis on holdout set (precision/recall tradeoff)
   - Document validation results with metrics
   - Ensure no regression on negative controls

3. **Regenerate baseline**:
   ```bash
   python scripts/prw_water_validation.py \
     --ground-truth data/water_v0/ground_truth.json \
     --subset-file data/water_v0/ci_subset.txt \
     --output data/water_v0/baseline_ci_current_v1.json \
     --seed 42
   ```

4. **Document changes**:
   - Update this README with new baseline metrics
   - Create ADR in `docs/architecture/` if significant
   - Commit with clear rationale and validation results

**DO NOT** tune thresholds on test set without holdout validation (overfitting risk).

---

### v0 (Single Threshold) - DEPRECATED

**Status**: ❌ Deprecated (replaced by two-stage gating in PR-W4)

**Status**: ❌ Deprecated (replaced by two-stage gating in PR-W4)

**Historical Context**:  
Original single-threshold gating system. Replaced by two-stage gating (candidate detection + injection decision) for better observability and control.

## Running Validation

```bash
# Full dataset validation
python scripts/prw_water_validation.py \
  --ground-truth data/water_v0/ground_truth.json \
  --output report.json \
  --seed 42

# CI subset validation (14 images)
python scripts/prw_water_validation.py \
  --ground-truth data/water_v0/ground_truth.json \
  --subset-file data/water_v0/ci_subset.txt \
  --output ci_report.json \
  --seed 42
```

**Output**: JSON report with per-image results and summary statistics.

## Test Cases

### Positive Controls (should_detect=true)

**Pools** (6):
- pool_0001, pool_0003, pool_0005, pool_0007, pool_0009: Clear pools (easy/medium)
- **pool_0008**: Low-saturation pool with tile grid (hard) - **missed by current baseline** (known limitation)

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
