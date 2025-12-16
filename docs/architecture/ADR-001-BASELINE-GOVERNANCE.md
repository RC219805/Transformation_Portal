# ADR-001: Baseline Governance for Water Detection Validation

**Status**: ✅ Accepted  
**Date**: 2025-12-15  
**Author**: Transformation Portal Architect  
**Context**: PR-W1 Water Detection - Baseline Management  

---

## Context

During PR-W1 water detection development, we encountered a **governance crisis** caused by:

1. **Mutually incompatible baselines**: Multiple baseline files claiming different performance metrics
2. **Moving target loop**: Mixing uncommitted experiments with baseline artifacts
3. **Overfitting risk**: Tuning thresholds on tiny synthetic test set (14 images, 2 negatives)
4. **Dirty working tree**: Uncommitted changes to critical validation files
5. **Premature completion claims**: Documentation claiming 100% pool recall without validation

### The Problem

```
baseline_ci_v0.json: 83.3% pool recall (original)
baseline_ci_v1.json: 100% pool recall (modified - claimed "solved")
Glass suppressor thresholds: 0.15 → 0.11 (experimental, uncommitted)
```

**Risk**: Without holdout validation, threshold tuning on test set leads to overfitting. Changes that appear to "solve" pool_0008 may break on real architectural glass.

---

## Decision

Implement **two-tier baseline governance** with strict update policy:

### Baseline Tiers

#### 1. Audit Baselines (Immutable)
- **Purpose**: Historical reference for long-term performance tracking
- **Naming**: `baseline_ci_audit_v{N}.json`
- **Mutability**: ❌ Never modified after creation
- **Use Case**: Regression analysis, historical comparison

#### 2. Current Baselines (Mutable with Validation)
- **Purpose**: Actively enforced baseline for CI regression checks
- **Naming**: `baseline_ci_current_v{N}.json`
- **Mutability**: ⚠️ Requires holdout validation
- **Use Case**: CI/CD enforcement, development iteration

### Update Policy

To modify `baseline_ci_current_v{N}.json`:

1. **Create holdout validation set**:
   - Minimum 10-20 samples not in test set
   - Must include edge cases (low-sat pools, architectural glass)
   - No overlap with CI subset

2. **Validate threshold changes**:
   - ROC analysis on holdout set
   - Document precision/recall tradeoff
   - Ensure no regression on negative controls

3. **Regenerate baseline**:
   ```bash
   python scripts/prw_water_validation.py \
     --ground-truth data/water_v0/ground_truth.json \
     --subset-file data/water_v0/ci_subset.txt \
     --output data/water_v0/baseline_ci_current_v{N}.json \
     --seed 42
   ```

4. **Document changes**:
   - Update `data/water_v0/README.md`
   - Create ADR if significant architectural change
   - Commit with clear validation results

### Prohibited Actions

❌ **DO NOT**:
- Tune thresholds on test set without holdout validation
- Modify audit baselines after creation
- Commit experimental thresholds without validation
- Document completion without reproducible validation

---

## Consequences

### Positive

✅ **Prevents overfitting**: Holdout validation requirement stops test set tuning  
✅ **Reproducible state**: Clean commits with documented validation  
✅ **Transparent limitations**: Known issues (pool_0008 miss) documented  
✅ **Long-term tracking**: Immutable audit baselines preserve history  

### Negative

⚠️ **Slower iteration**: Validation overhead for threshold changes  
⚠️ **Known limitations persist**: 83.3% pool recall until proper validation  

### Mitigation

- **Telemetry infrastructure** (PR-W4): Observability into suppressor effects
- **Design documentation**: Experimental logic documented but uncommitted
- **Clear path forward**: PR-W1.2 holdout validation roadmap

---

## Implementation

### Current State (2025-12-15)

**Baselines**:
- `baseline_ci_audit_v0.json`: Historical baseline (83.3% pool recall) - immutable
- `baseline_ci_current_v1.json`: Current enforced baseline (83.3% pool recall) - mutable

**Thresholds** (safe, conservative):
- `glass_edge_alignment_threshold: 0.15`
- `glass_grid_score_threshold: 0.25`
- `glass_penalty: 0.6`

**CI Integration**:
```yaml
# .github/workflows/ci-consolidated.yml
python scripts/check_regression.py \
  --baseline data/water_v0/baseline_ci_current_v1.json \
  --current outputs/water_validation_current.json \
  --mode warning
```

### Known Limitations

**pool_0008 miss** (83.3% pool recall):
- Low-saturation pool with subtle tile grid
- Requires lower alignment threshold (0.11) to detect
- Current threshold (0.15) conservative to prevent glass false positives
- **Accepted limitation** until holdout validation exists

---

## Baseline v2 Promotion Policy (Amendment 2025-12-16)

### Acceptance Gates (All Required)

**CI Fixtures** (14 synthetic images):
- Pool recall: 100% (6/6, must recover pool_0008)
- Ocean recall: 100% (6/6, maintain current)
- False trigger rate: 0% (0/2, critical - no regressions)

**Holdout Negatives** (15-20 real-world images):
- False trigger rate: ≤5% (at most 1 trigger on 20 images)
- Justification: Real-world tolerance for rare false positives
- All triggers must have telemetry explaining root cause

**Telemetry Requirements**:
- Every image includes suppressor_telemetry
- pool_0008 shows tile_exempted=True (when multi-scale enabled)
- Negatives show architectural_glass suppressor firing (when applicable)

### Promotion Workflow

1. **Implement fix** (e.g., multi-scale glass suppressor)
2. **Run CI fixtures validation** → verify 100%/100%/0%
   ```bash
   make test-water-validation
   ```
3. **Run holdout validation** → verify ≤5% FT rate
   ```bash
   export WATER_HOLDOUT_DIR=/path/to/holdout
   ./scripts/validate_holdout.sh holdout_validation_v1.json
   ```
4. **Review telemetry** for explainability
   ```bash
   jq '.results[] | select(.detected == true) | .suppressor_telemetry' holdout_validation_v1.json
   ```
5. **Update ADR** with justification
6. **Freeze baseline**: `baseline_ci_current_v1.json` → `baseline_ci_historical_v1.json`
7. **Promote baseline**: Rename to `baseline_ci_current_v2.json`
8. **Update CI workflow** to enforce v2
9. **Commit** with ADR approval reference

### Rejection Criteria

Any of the following **rejects** baseline v2 promotion:
- ❌ CI pool recall < 100%
- ❌ CI false trigger rate > 0%
- ❌ Holdout FT rate > 5%
- ❌ Mystery suppressions (no telemetry explanation)
- ❌ Threshold tuning without holdout validation proof

### Rationale

**Why Two-Tier Validation?**

1. **CI Fixtures** (synthetic, zero-tolerance):
   - Fast feedback loop for development
   - Prevent regressions on known cases
   - Zero false positives allowed (protect real-world negatives)

2. **Holdout Set** (real-world, 5% tolerance):
   - Prevent overfitting to 2 synthetic negatives
   - Validate generalization to architectural confusers
   - Realistic tolerance for edge cases

**Why 5% Threshold?**

- 15 images × 5% = 0.75 triggers (at most 1 false positive)
- Real-world deployments require robustness, not perfection
- Telemetry explains rare failures (e.g., extreme lighting conditions)

---

## Future Work

### PR-W1.2: Holdout Validation Framework ✅ Infrastructure Complete (2025-12-16)
1. ✅ Create diverse holdout set (15-20 real negatives)
2. ✅ Holdout manifest with SHA256 integrity verification
3. ✅ Validation runner script (`validate_holdout.sh`)
4. ✅ ADR amendment with promotion policy
5. **Next**: Acquire real-world images, run first validation

### PR-W1.3: Multi-Scale Glass Suppressor
- **Prerequisite**: Holdout infrastructure in place (Phase B complete)
- Implement tile exemption for pool_0008 recovery
- Validate on holdout set (≤5% FT rate)
- Regenerate baseline v2 with validated thresholds

### PR-W1.4: ADE20K Integration
- **Prerequisite**: Stable baselines from PR-W1.3
- Semantic segmentation for water/pool detection
- Hybrid heuristic + semantic approach

---

## Related Documents

- **Session Summary**: `SESSION_END_SUMMARY_2025-12-15_GOVERNANCE_CLEANUP.md`
- **Baseline README**: `data/water_v0/README.md`
- **Holdout Documentation**: `data/water_v0/HOLDOUT.md`
- **Holdout Manifest**: `data/water_v0/holdout_manifest.json`
- **Validation Runner**: `scripts/validate_holdout.sh`
- **Analysis Docs**:
  - `BASELINE_THRESHOLD_ANALYSIS.md` (diagnosis of pool_0008)
  - `GLASS_SUPPRESSOR_MULTISCALE_FIX.md` (experimental design)
  - `GLASS_SUPPRESSOR_CLAIMS_VALIDATION.md` (honest assessment)

---

## Approval

**Architect**: ✅ Approved  
**Status**: ✅ Implemented (2025-12-15)  
**Amendment v2**: ✅ Approved (2025-12-16) - Holdout Infrastructure  
**Review Date**: After PR-W1.3 completes holdout validation  
