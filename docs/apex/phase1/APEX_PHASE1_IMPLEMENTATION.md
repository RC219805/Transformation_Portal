# APEX Real Pipeline Integration - Phase 1 Complete

## Summary

Successfully transitioned APEX from synthetic/dry-run mode to executing real V1+V2 workflows with actual performance measurement.

## Changes Implemented

### 1. Pipeline Integration (`scripts/apex_matrix_runner.py`)

**Core Implementation:**
- Replaced `NotImplementedError` with real pipeline invocation using `EnhanceOrchestrator`
- Added V1/V2 workflow differentiation:
  - V1: Depth-only workflow (Stage A)
  - V2: Full depth + enhancement workflow (Stage A + B)
- Integrated `timing_context` for GPU-synchronized timing measurements
- Added per-image timeout protection (300s default)
- Implemented error handling with continue-on-error support

**Performance Capsule Generation:**
- Real timing data from `timing_context` (total, load_decode, etc.)
- Actual image metadata (dimensions, pixel counts, hashes)
- Device info from orchestrator config
- Workflow version tracking (v1/v2)
- Zone information for multi-zone analysis
- `is_synthetic=False` flag for real data vs `is_synthetic=True` for dry-run

**CLI Enhancements:**
- `--input-dir`: Required for real execution (points to test images)
- `--sample-size`: Limit number of images processed (default: all)
- `--dry-run`: Preserved for testing/validation (uses synthetic data)
- Validation: Ensures `--input-dir` provided when not in dry-run mode

### 2. Synthetic Data Labeling (`scripts/apex_pr_comment.py`)

**Conditional Warning Banner:**
- Added `is_synthetic` parameter to `generate_pr_comment()`
- Synthetic banner only shown when `--synthetic` flag passed or data is marked synthetic
- Real data reports use clean header: "# 🎯 APEX Performance Report"
- Maintained backward compatibility with existing signature

**CLI Addition:**
- `--synthetic` flag to explicitly mark report as dry-run mode

### 3. CI Workflow Updates (`.github/workflows/apex_performance.yml`)

**Real Execution Configuration:**
- Removed `--dry-run` flag
- Added `--input-dir ./input_images/750_picacho/source_jpegs`
- Set `--sample-size 3` for fast CI feedback (<5 min target)
- Maintained shadow mode for initial rollout
- Updated comments to reflect real data collection

**Workflow Notes:**
- Uses existing 750 Picacho test images (6 available, 3 processed per workflow)
- Gate remains in shadow mode (informational only, non-blocking)
- Both V1 and V2 workflows run in parallel matrix

### 4. Contract Test Updates (`tests/test_apex_contract_verification.py`)

**Test Evolution:**
- Replaced `NotImplementedError` assertion with real execution checks
- Updated `test_dry_run_flag_documented()` to verify CLI flags
- Added `test_real_execution_requires_input_dir()` to ensure validation
- All 17 contract tests passing (1 skipped as expected)

## Technical Details

### V1 vs V2 Workflow Distinction

The implementation correctly handles workflow version routing:

```python
# V1: depth_backend enabled, no V2 preset
config = EnhanceConfig(
    model_variant=ModelVariant.METRIC_LARGE,
    depth_device=device,
    v2_preset=None,  # V1 workflow
    ...
)

# V2: depth_backend + V2 enhancement
config = EnhanceConfig(
    model_variant=ModelVariant.METRIC_LARGE,
    depth_device=device,
    v2_preset="default",  # V2 workflow
    ...
)
```

### Timing Instrumentation

Uses `timing_context` with device synchronization:

```python
with timing_context("total", timings, device=device):
    with timing_context("load_decode", timings, device=device):
        # Load image
    # Process image
# timings = {"total": 12.34, "load_decode": 0.56, ...}
```

### Error Handling

- Timeout protection using `signal.SIGALRM` (Unix only)
- Per-image exception isolation (one failure doesn't abort batch)
- Graceful degradation with informative logging
- Capsules only created for successful runs

## Acceptance Criteria Status

✅ **Matrix runner executes real V1+V2 workflows without --dry-run flag**
- Implementation complete in `run_apex_for_config()`
- V1/V2 differentiation via `v2_preset` configuration

✅ **Real performance capsules generated with accurate timing data**
- `timing_context` integration with device synchronization
- `is_synthetic=False` for real data
- Full metadata capture (dimensions, hashes, device info)

✅ **CI workflow configured for real data collection in shadow mode**
- `--input-dir` pointing to test images
- `--sample-size 3` for fast feedback
- Shadow mode maintained for safe rollout

✅ **No synthetic labels in normal operation**
- Conditional banner based on `is_synthetic` flag
- Clean report header for real data
- Backward compatibility preserved

## Testing

### Smoke Tests Run
```bash
# Dry run mode (synthetic)
python scripts/apex_matrix_runner.py \
  --run-id test123 --commit-sha abc123 \
  --zones local --output-dir ./test_output \
  --dry-run

# Real execution (requires images)
python scripts/apex_matrix_runner.py \
  --run-id test123 --commit-sha abc123 \
  --zones local --input-dir ./input_images/750_picacho/source_jpegs \
  --sample-size 3 --output-dir ./test_output
```

### Contract Tests
```bash
pytest tests/test_apex_contract_verification.py -v
# Result: 17 passed, 1 skipped
```

## Deployment Notes

### Shadow Mode Rollout
- Gate is in **shadow mode** (informational only, non-blocking)
- Allows baseline establishment without blocking PRs
- Monitor for 2-3 weeks before switching to enforce mode

### CI Resource Usage
- 3 images × 2 workflows = 6 processing runs per PR
- Expected runtime: <5 minutes total
- Artifact retention: 3 days for results, 90 days for ledger

### Next Steps
1. Monitor shadow mode results for baseline stability
2. Adjust performance thresholds if needed
3. Switch gate to enforce mode once baseline is stable
4. Consider increasing `--sample-size` for more robust measurements

## Architecture Alignment

This implementation follows repository principles:

- **Golden Path Preservation**: No changes to default outputs or behavior
- **Minimal Surface Area**: Surgical changes only to required files
- **Contract Stability**: Maintained `PerformanceCapsule` schema v2.0.0
- **Testing Rigor**: Updated contract tests to match new reality
- **Documentation Fidelity**: Comments match actual behavior
- **CI Green**: All tests passing, no regressions

## Files Changed

1. `scripts/apex_matrix_runner.py` - Real pipeline integration
2. `scripts/apex_pr_comment.py` - Conditional synthetic warning
3. `.github/workflows/apex_performance.yml` - Real execution config
4. `tests/test_apex_contract_verification.py` - Updated contract tests
5. `APEX_PHASE1_IMPLEMENTATION.md` - This documentation

---

**Status**: ✅ **SUCCEEDED**
**Date**: 2026-02-08
**Reviewed By**: Transformation Portal Architect
