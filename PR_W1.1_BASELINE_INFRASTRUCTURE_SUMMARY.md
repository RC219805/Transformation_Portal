# PR-W1.1: Water Detection Baseline Infrastructure

**Status:** ✅ Complete  
**Branch:** `feature/water-baseline-infrastructure`  
**PR Number:** TBD (#559)

## Summary

Established baseline infrastructure for water detection regression testing. CI can now validate water detection without committing images to git by using deterministic synthetic fixtures.

## Changes Made

### 1. Ground Truth Schema (`data/water_v0/ground_truth.schema.json`)
- JSON Schema v7 validation for water detection ground truth
- Required fields: `root`, `images` with `label` (pool/ocean) and `should_detect` (bool)
- Optional fields: `difficulty`, `tags`, `bbox`, `expected_mask_coverage`
- Validates image path format and value constraints

### 2. Ground Truth Validator (`scripts/validate_ground_truth.py`)
- Validates ground truth JSON against schema
- Checks image paths are relative and consistent
- Returns exit code 0/1 for CI use
- Can skip file existence checks with `--skip-file-check`

### 3. Synthetic Fixture Generator (`scripts/gen_water_ci_fixture.py`)
- Generates deterministic synthetic images (seed-based)
- Creates 14 images matching `ci_subset.txt`:
  - 6 pool scenes (standard, bright, dark, low saturation)
  - 6 ocean scenes (standard, waves, calm, green)
  - 2 hard negatives (blue wall, reflective glass)
- Output is pixel-perfect deterministic (verified with SHA256)
- Generates corresponding `ground_truth.json` with metadata
- Default: 512x512 images (CI-friendly size)

**Determinism verified:**
```bash
$ sha256sum data/water_v0/images/pool/pool_0001.jpg
a1c437aaa1913ac82e32b8be5741b6400740da7facf4ac9f619d658c4f2bbfa9

$ sha256sum /tmp/test/images/pool/pool_0001.jpg  # Regenerated with --seed 42
a1c437aaa1913ac82e32b8be5741b6400740da7facf4ac9f619d658c4f2bbfa9
```

### 4. Baseline Report (`data/water_v0/baseline_ci_v0.json`)
- Pinned baseline validation report (v0 stub detector)
- Generated from synthetic fixtures with `--seed 42`
- Tracks: recall, coverage, edge alignment, false trigger rate, stability
- Current baseline reflects stub detector (0% recall, as expected)
- Will be updated when real detector is implemented

**Baseline structure:**
```json
{
  "summary": {
    "pool_recall": 0.0,
    "ocean_recall": 0.0,
    "false_trigger_rate": 0.0,
    ...
  },
  "results": [...]
}
```

### 5. Harness Path Resolution Fix (`scripts/prw_water_validation.py`)
- Fixed path resolution to be relative to ground truth file location
- Added `gt_base_dir` parameter to `validate_dataset()`
- Now resolves `root` field correctly

### 6. CI Regression Job (`.github/workflows/ci-consolidated.yml`)
- New job: `water-regression` (Stage 3.5)
- Runs after `test-core` success
- Steps:
  1. Generate synthetic CI fixtures (deterministic)
  2. Run validation harness → `current_report.json`
  3. Check regression vs baseline (warn mode)
  4. Upload current report as artifact (30-day retention)
- `continue-on-error: true` - Warn-only mode for now
- Will switch to fail mode after detector implementation and threshold calibration

### 7. Pre-commit Safety Hook (`scripts/pre-commit-water-safety.sh`)
- Prevents accidentally committing images to git
- Checks for staged files in `data/water_v0/images/`
- Warns on large files (>1MB)
- Install: `cp scripts/pre-commit-water-safety.sh .git/hooks/pre-commit`

### 8. .gitignore Update
- Added `data/water_v0/images/` to ignore list
- Ensures images are never committed

### 9. Documentation Update (`data/water_v0/README.md`)
- Complete guide on what's tracked in git vs generated
- Instructions for running validation locally
- CI workflow explanation
- Baseline calibration process
- Safety checks for preventing image commits

## Verification

### Ground Truth Validation
```bash
$ python scripts/validate_ground_truth.py data/water_v0/ground_truth.json
✅ Validation passed (14 images)
```

### Fixture Generation (Deterministic)
```bash
$ python scripts/gen_water_ci_fixture.py --seed 42 --output data/water_v0/images/
✅ Generated 14 images in data/water_v0/images
✅ Ground truth saved: data/water_v0/ground_truth.json

📊 Summary:
  • Pool scenes: 6
  • Ocean scenes: 6
  • Hard negatives: 2
  • Total: 14
```

### Baseline Generation
```bash
$ python scripts/prw_water_validation.py \
    --ground-truth data/water_v0/ground_truth.json \
    --subset-file data/water_v0/ci_subset.txt \
    --output outputs/validation_report.json \
    --seed 42 \
    --no-scipy-warning

✅ Validation report written to outputs/validation_report.json

📊 Summary:
  Total images: 14 (12 water, 2 hard negatives)
  Pool recall: 0.0% (0/6 detected)
  Ocean recall: 0.0% (0/6 detected)
  False trigger rate: 0.0% (0/2)
```

### Regression Check
```bash
$ python scripts/check_regression.py \
    --baseline data/water_v0/baseline_ci_v0.json \
    --current outputs/validation_report.json \
    --mode warning

✅ No regression detected
```

### Safety Check
```bash
$ git diff --cached --name-only | grep -E '^data/water_v0/images/' \
    && echo "ERROR: images staged" || echo "OK: no images staged"
✅ OK: no images staged
```

## Files Changed

**Created:**
- `data/water_v0/ground_truth.schema.json` - JSON Schema validation
- `data/water_v0/ground_truth.json` - Ground truth annotations (14 images)
- `data/water_v0/baseline_ci_v0.json` - Pinned baseline report
- `scripts/validate_ground_truth.py` - Schema validator
- `scripts/gen_water_ci_fixture.py` - Synthetic fixture generator
- `scripts/pre-commit-water-safety.sh` - Pre-commit safety hook

**Modified:**
- `data/water_v0/README.md` - Updated documentation
- `scripts/prw_water_validation.py` - Fixed path resolution
- `.github/workflows/ci-consolidated.yml` - Added water regression job
- `.gitignore` - Added `data/water_v0/images/`

**Not committed (generated):**
- `data/water_v0/images/` - Synthetic images (14 total)

## Acceptance Criteria

- ✅ Ground truth schema validated
- ✅ CI can run harness without committing images
- ✅ Baseline report pinned and reproducible
- ✅ Warn-only CI regression job exists
- ✅ No images accidentally committed (verified with safety check)
- ✅ Deterministic fixture generation verified (SHA256 match)
- ✅ Complete workflow tested locally

## Next Steps

1. Create PR #559 from `feature/water-baseline-infrastructure` → `main`
2. Verify CI passes on PR
3. Merge PR #559
4. Begin PR-W1.2: Implement real water detector
5. After detector implementation:
   - Collect diverse labeled images for threshold calibration
   - Re-generate baseline with real detector
   - Calibrate regression thresholds
   - Switch CI to fail mode

## Testing Instructions for Reviewers

```bash
# Clone branch
git checkout feature/water-baseline-infrastructure

# Generate synthetic fixtures
python scripts/gen_water_ci_fixture.py --seed 42 --output data/water_v0/images/

# Validate schema
python scripts/validate_ground_truth.py data/water_v0/ground_truth.json

# Run validation harness
python scripts/prw_water_validation.py \
  --ground-truth data/water_v0/ground_truth.json \
  --subset-file data/water_v0/ci_subset.txt \
  --output /tmp/test_report.json \
  --seed 42 \
  --no-scipy-warning

# Check regression
python scripts/check_regression.py \
  --baseline data/water_v0/baseline_ci_v0.json \
  --current /tmp/test_report.json \
  --mode warning

# Verify determinism (should match baseline exactly)
diff data/water_v0/baseline_ci_v0.json /tmp/test_report.json

# Safety check
git diff --cached --name-only | grep -E '^data/water_v0/images/' \
  && echo "ERROR" || echo "OK"
```

## Notes

- Baseline reflects stub detector (0% recall) - expected behavior
- CI job is warn-only to avoid blocking PRs during detector development
- Synthetic fixtures are deterministic but simplified (not photorealistic)
- Real labeled images should be collected for threshold calibration
- Pre-commit hook is optional but recommended for contributors

---

**Ready for PR #559**
