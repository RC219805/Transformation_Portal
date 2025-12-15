# Water Detection Dataset v0

This dataset contains labeled images for validating water detection in pool and ocean scenes.

## Structure

```
water_v0/
├── images/               # ⚠️  NOT TRACKED IN GIT (generated synthetically)
│   ├── pool/             # Pool scenes
│   └── ocean/            # Ocean scenes
├── ground_truth.json     # ✅ TRACKED: Ground truth labels (v0 schema)
├── ground_truth.schema.json  # ✅ TRACKED: JSON Schema validation
├── baseline_ci_v0.json   # ✅ TRACKED: Pinned baseline report for regression checks
├── ci_subset.txt         # ✅ TRACKED: 14 images for fast CI validation
├── LABELING_GUIDE.md     # ✅ TRACKED: Labeling instructions
└── README.md             # ✅ TRACKED: This file
```

## What's Tracked in Git

**Metadata and baselines only:**
- ✅ `ground_truth.json` - Ground truth annotations
- ✅ `ground_truth.schema.json` - JSON Schema for validation
- ✅ `baseline_ci_v0.json` - Pinned baseline validation report
- ✅ `ci_subset.txt` - List of images for CI testing
- ✅ Documentation files

**NOT tracked in git:**
- ❌ `images/` directory - Images are generated synthetically or stored privately

## How CI Obtains Images

CI uses **synthetic image generation** to avoid committing large binary files:

```bash
# CI runs this before validation:
python scripts/gen_water_ci_fixture.py --seed 42 --output data/water_v0/images/
```

This generates 14 deterministic synthetic images matching `ci_subset.txt`:
- 6 pool scenes (various lighting conditions)
- 6 ocean scenes (waves, calm, different colors)
- 2 hard negatives (blue wall, reflective glass)

**Deterministic:** Same seed always produces identical images (pixel-perfect).

## Schema Version

**v0** - Two-label schema (pool, ocean) with negative controls

See `ground_truth.schema.json` for complete JSON Schema validation.

## Key Fields

- `label`: `pool` or `ocean` (folder organization)
- `should_detect`: `true` for water, `false` for hard negatives
- `difficulty`: `easy` | `medium` | `hard`
- `tags`: Track failure modes (e.g., `low-light`, `reflection`, `waves`)

## Hard Negatives

Hard negatives (`should_detect: false`) are critical for measuring false trigger rate:

**Pool hard negatives**:
- Blue painted walls
- Blue sky through windows
- Blue fabric/umbrellas

**Ocean hard negatives**:
- Reflective glass buildings
- Blue painted surfaces
- Sky reflections

## Running Validation Locally

### Step 1: Generate Synthetic Images

```bash
python scripts/gen_water_ci_fixture.py \
    --seed 42 \
    --output data/water_v0/images/
```

### Step 2: Validate Ground Truth Schema

```bash
python scripts/validate_ground_truth.py data/water_v0/ground_truth.json
```

### Step 3: Run Validation Harness

Full dataset:
```bash
python scripts/prw_water_validation.py \
    --ground-truth data/water_v0/ground_truth.json \
    --output water_validation_report.json \
    --seed 42
```

CI subset only (faster):
```bash
python scripts/prw_water_validation.py \
    --ground-truth data/water_v0/ground_truth.json \
    --subset-file data/water_v0/ci_subset.txt \
    --output ci_report.json \
    --seed 42
```

### Step 4: Check for Regression

```bash
python scripts/check_regression.py \
    --baseline data/water_v0/baseline_ci_v0.json \
    --current ci_report.json \
    --mode fail
```

## CI Regression Job

The CI workflow includes a **warn-only** water regression check:

```yaml
water-regression:
  name: Water Detection Regression (Warn)
  runs-on: ubuntu-24.04
  continue-on-error: true  # Warn-only for now
  steps:
    - Generate synthetic fixtures (deterministic)
    - Run validation harness
    - Check for regression vs baseline
    - Upload current report as artifact
```

**Exit codes:**
- `0` - No regression
- `2` - Regression detected (warning mode, continues)
- `1` - Regression detected (error mode, would fail)

Currently set to **warning mode** to avoid blocking PRs during initial calibration.

## Baseline Calibration

The baseline report (`baseline_ci_v0.json`) is pinned and tracks:
- **Recall:** % of water images correctly detected (pool/ocean separate)
- **Coverage:** Mean/median % of image covered by water mask
- **Edge alignment:** Quality of mask boundary vs image gradients
- **False trigger rate:** % of hard negatives incorrectly flagged as water
- **Stability:** Consistency across noise perturbations

**Thresholds are targets, not calibrated against real data yet.**

Complete PR-W1 (real detector) to establish meaningful baseline metrics.

## Threshold Calibration Process

**Separate from baseline establishment:**

1. Collect diverse labeled images (20+ pool, 20+ ocean, 4+ hard negatives)
2. Run validation to generate current report
3. Analyze metrics distribution (histograms, percentiles)
4. Set thresholds at acceptable failure rates:
   - Recall: >= 90% (10% miss rate)
   - Edge alignment: >= 0.7 (subjective quality)
   - False trigger rate: <= 5% (95% specificity)
5. Update `check_regression.py` with calibrated thresholds
6. Re-pin baseline if needed

## Safety: Preventing Accidental Image Commits

**Pre-commit hook:**

```bash
# Install hook to prevent staging images
cp scripts/pre-commit-water-safety.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**Manual check:**

```bash
git diff --cached --name-only | grep -E '^data/water_v0/images/' \
  && echo "ERROR: images staged" || echo "OK: no images staged"
```

## Status

**Dataset:** Synthetic fixtures available (deterministic generation)

**Validation harness:** Complete and deterministic (PR-W4 merged)

**Detector:** WaterCandidateDetector heuristic (merged in PR #558)

**Baseline:** Pinned at v0 (uncalibrated heuristic baseline)

**CI Regression:** Active in warn-only mode

## Next Steps

1. ✅ Establish baseline infrastructure (this PR - PR-W1.1)
2. ✅ Implement water detector (PR #558 merged)
3. Calibrate thresholds with expanded dataset (PR-W1.2 pending)
4. Collect diverse labeled images for validation
5. Tune thresholds against measured failure modes
6. Switch CI regression to fail mode when stable
7. Monitor for regressions in production
