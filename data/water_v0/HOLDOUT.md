# Water Detection Holdout Set

**Version**: v1  
**Created**: 2025-12-16  
**Purpose**: Prevent overfitting to synthetic CI fixtures

## Overview

The holdout set consists of 15-20 real-world negative images (architectural glass, painted walls, reflective surfaces) that are **NOT** part of the CI fixtures. This prevents threshold tuning to just 2 synthetic negatives.

### The Overfitting Problem

**CI Fixtures** (14 synthetic images):
- 6 pools (positives)
- 6 ocean scenes (positives)
- **2 negatives only** (architectural glass, blue wall)

**Risk**: Tuning thresholds on 2 negatives → overfitting to specific synthetic characteristics. Real architectural glass may break suppressor logic.

**Solution**: Holdout set with 15-20 diverse real-world confusers validates threshold changes.

## Storage

**Images are NOT committed to git** (large binaries, private dataset).

**Access**: Images stored locally, referenced by `holdout_manifest.json` with SHA256 hashes for integrity verification.

## Usage

### 1. Set Holdout Directory

```bash
export WATER_HOLDOUT_DIR=/path/to/private/holdout/images
```

### 2. Run Validation

```bash
./scripts/validate_holdout.sh holdout_validation.json
```

### 3. Review Results

```bash
# Summary metrics
jq '.summary' holdout_validation.json

# Check false triggers (should be ≤5%)
jq '.results[] | select(.detected == true)' holdout_validation.json

# Review suppressor telemetry for triggers
jq '.results[] | select(.detected == true) | .suppressor_telemetry' holdout_validation.json
```

## Acceptance Gates

### Baseline v2+ Promotion Requirements

**All Required**:
- ✅ **CI Fixtures**: 100% pool recall, 100% ocean recall, 0% false trigger rate
- ✅ **Holdout Set**: ≤5% false trigger rate (at most 1 trigger on 15-20 images)
- ✅ **Telemetry**: All triggers must have explainable suppressor telemetry

**Justification**: 
- CI fixtures provide synthetic baseline (zero regressions allowed)
- Holdout provides real-world validation (5% tolerance for edge cases)
- Telemetry ensures explainability (no mystery suppressions)

### Rejection Criteria

Any of the following **rejects** baseline v2+ promotion:
- ❌ CI pool recall < 100%
- ❌ CI false trigger rate > 0%
- ❌ Holdout FT rate > 5%
- ❌ Mystery suppressions (no telemetry explanation)
- ❌ Threshold tuning without holdout validation proof

## Image Categories

### 1. Architectural Glass (3-5 images)
**Difficulty**: Hard  
**Examples**: Modern façades, reflective windows, curtain walls  
**Confuser**: Blue tint, edge alignment, specular highlights

### 2. Blue Painted Walls (2-3 images)
**Difficulty**: Medium  
**Examples**: Flat surfaces, exterior/interior  
**Confuser**: Saturation, hue similarity

### 3. Reflective Stone/Concrete (2-3 images)
**Difficulty**: Medium-Hard  
**Examples**: Polished surfaces, wet pavement  
**Confuser**: Specular highlights, transient water-like appearance

### 4. Skylight Reflections (1-2 images)
**Difficulty**: Hard  
**Examples**: Glass roof, interior reflections  
**Confuser**: Blue hue, brightness, diffused light

### 5. Pool Tiles (2-3 images)
**Difficulty**: Hard  
**Examples**: Close-up grid patterns, no water visible  
**Confuser**: Grid alignment, ceramic blue color

### 6. Ocean Horizon Glare (1-2 images)
**Difficulty**: Medium  
**Examples**: Sun reflection, bright highlights  
**Confuser**: Brightness saturation, overexposure artifacts

## Governance

### Version Control
- Holdout set version controlled via manifest (not images)
- SHA256 hashes verify image integrity
- Manifest updates require ADR justification

### Adding Images
1. Document justification (new confuser category, failure case)
2. Generate SHA256 hash
3. Update `holdout_manifest.json`
4. Update this README
5. Commit with ADR reference

### Baseline Promotion
- Holdout validation required before any baseline v2+ promotion
- Results must be committed alongside baseline update
- ADR-001 documents acceptance criteria

### Image Acquisition
- Real-world photography (luxury real estate, architectural)
- Public domain / Creative Commons (attributed in manifest)
- Synthetic generation (documented in manifest)

## Workflow Integration

### Pre-Promotion Checklist

Before promoting baseline v2+:

```bash
# 1. Run CI fixtures validation
make test-water-validation

# 2. Run holdout validation
export WATER_HOLDOUT_DIR=/path/to/holdout
./scripts/validate_holdout.sh holdout_validation_v1.json

# 3. Review results
jq '.summary | {
  ci_pool_recall,
  ci_false_trigger_rate,
  holdout_false_trigger_rate
}' holdout_validation_v1.json

# 4. Verify acceptance gates
# CI: 100% / 100% / 0%
# Holdout: ≤5% FT rate

# 5. Review telemetry for any triggers
jq '.results[] | select(.detected == true)' holdout_validation_v1.json
```

### Threshold Tuning Protocol

When modifying water detection thresholds:

1. **Local Testing**: Experiment on development branch
2. **CI Validation**: Ensure no regressions on synthetic fixtures
3. **Holdout Validation**: Run against real-world negatives
4. **Telemetry Review**: Explain all false triggers
5. **ADR Documentation**: Update ADR-001 with justification
6. **Baseline Promotion**: Freeze historical baseline, promote new version

## Related Documents

- **ADR-001**: `docs/architecture/ADR-001-BASELINE-GOVERNANCE.md`
- **Baseline README**: `data/water_v0/README.md`
- **Validation Script**: `scripts/prw_water_validation.py`
- **Holdout Runner**: `scripts/validate_holdout.sh`

---

**Status**: Infrastructure Complete (2025-12-16)  
**Next**: Acquire real-world images, generate SHA256 hashes, run first validation
