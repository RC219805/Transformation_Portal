# Water Detection: Exact Current State & Meaningful Next Steps

**Date:** 2025-12-14  
**Assessment Type:** Engineering Reality Check  
**Purpose:** Zero-bullshit status + executable path forward

---

## What Actually Exists Right Now

### ✅ PR-W4: Validation Infrastructure (COMPLETE)

**Files That Work:**
- `scripts/prw_water_validation.py` - Full harness, linting clean, 16/16 tests passing
- `tests/test_prw_water_validation.py` + `tests/test_prw_water_validation_deterministic.py`
- `scripts/check_regression.py` - CI-ready regression checker
- `docs/WATER_GROUND_TRUTH_SCHEMA_FINAL.md` - Finalized schema
- `.gitignore` - Correct patterns for water datasets

**What It Can Do:**
- Run on a labeled dataset and produce JSON reports
- Compute edge alignment, stability, false trigger rate (when detector provides mask)
- Deterministic results with `--seed` flag
- Support CI subsets via `--subset-file`
- Detect regressions across runs

**What It Cannot Do:**
- Provide real water masks (detector is stub)
- Calibrate thresholds (no dataset exists)
- Measure quality in production (awaiting data)

### ⏳ PR-W1: Water Detector (STUB ONLY)

**Current State:**
- File exists: `lux_depth_v2/water_candidate.py`
- Returns: `{"present": False, "mask": None, "coverage": 0.0, "confidence": 0.0}`
- Purpose: Test scaffold only

**What's Missing:**
- Actual heuristic implementation (HSV, components, texture)
- Scene-aware tuning (pool vs ocean)
- Mask generation logic
- Confidence scoring

**Impact:**
- Harness metrics default to 0.0 / 0
- No real validation possible yet

### ❌ Dataset v0 (DOES NOT EXIST)

**Current State:**
- Schema documented
- Directory scaffold exists (`data/water_v0/`)
- Zero images committed
- No ground_truth.json

**What's Needed:**
- 20+ pool images (labeled)
- 20+ ocean images (labeled)
- 20+ hard negatives (blue walls, sky, reflections)
- `ground_truth.json` mapping paths to labels
- CI subset selection (12 images minimum)

**Impact:**
- Cannot calibrate thresholds
- Cannot measure real-world performance
- Cannot establish baseline for regression tracking

---

## What "Production-Ready" Actually Means

### Current Claim (Often Seen):
> "PR-W4 complete and production-ready"

### Engineering Reality:
- **Harness:** Production-quality code ✅
- **Detector:** Stub only ❌
- **Data:** Nonexistent ❌
- **Thresholds:** Uncalibrated ❌
- **Regression Baseline:** Impossible without data ❌

### Accurate Statement:
> "Validation infrastructure is complete and integration-ready. Detector remains a stub pending PR-W1. Thresholds uncalibrated pending dataset v0."

---

## Meaningful Progress: Two Viable Paths

### Path A: Data-First (Recommended)

**Why:** You can't validate what you can't measure.

**72-Hour Plan:**

#### Hour 0-12: Build Dataset v0
- Collect 60 images minimum (20 pool / 20 ocean / 20 hard negatives)
- Create `data/water_v0/ground_truth.json`
- Write `data/water_v0/LABELING_GUIDE.md`
- Select 12-image CI subset (`ci_subset.txt`)

**Hard Negatives Must Include:**
- Blue walls / paint
- Sky through windows
- TV screens / monitors
- Glossy marble reflections
- Pool covers / tarps
- Tiled surfaces with strong texture

#### Hour 12-24: Run Baseline + Identify Failures
- Run harness on dataset v0 with stub detector
- Generate `baseline_stub.json`
- Create `docs/WATER_V0_FAILURES.md` listing top failure buckets
- Categorize: false triggers, missed detections, low confidence

#### Hour 24-48: Implement PR-W1 Detector (Failure-Driven)
- Focus on largest failure buckets only
- Priority order:
  1. HSV/chroma gating (reduce obvious non-water blues)
  2. Connected component filtering (min area + shape sanity)
  3. Texture sanity check (Laplacian variance)
- Return `implementation: "heuristic_v1_hsv_components"`

#### Hour 48-60: Calibrate + Lock Thresholds
- Re-run harness with real detector
- Derive thresholds from dataset v0 quantiles (not aspirational targets)
- Update `baseline_v1.json`
- Document threshold derivation in `docs/WATER_V0_CALIBRATION.md`

#### Hour 60-72: CI Integration + Second Iteration
- Add GitHub Actions job (warning mode)
- Run on CI subset (12 images)
- Upload JSON artifact
- Second detector iteration based on remaining failures

**Deliverables:**
- Dataset v0 (committed: ground_truth.json, ci_subset.txt, LABELING_GUIDE.md)
- Detector v1 (real mask generation, confidence scoring)
- Calibrated thresholds (data-driven, not guessed)
- CI regression job (warning mode, artifact upload)

**Quality Gate:**
- Detector v1 must improve ≥2 of {precision, recall, edge alignment} without regressing others by >10%

---

### Path B: Ship Conservative Stub (If Deadline Forces It)

**Why:** Only if you absolutely must ship something today.

**What to Do:**
1. Improve stub detector minimally (6 hours):
   - HSV/YCbCr constraints (blue hue range + saturation floor)
   - Component filtering (min area 0.01% of image, aspect ratio 0.2-5.0)
   - Cheap texture check (Laplacian std < threshold)
2. Deploy behind `water_detection_enabled=false` (opt-in only)
3. Log every detection for offline analysis
4. Commit to building dataset v0 in parallel

**Critical Non-Negotiables:**
- Do NOT claim "production-ready detection"
- Do NOT enable by default
- DO commit to Path A immediately after shipping

**Risk:**
- You're shipping unvalidated code
- You'll have no idea if it works until you build the dataset anyway
- Technical debt accumulates fast

---

## Concrete "Next 72 Hours" Recommendation

### If You Have 72 Hours: Path A (Data-First Hybrid)

**Day 1 (Hours 0-24):**
- **Hour 0-6:** Collect 20 pool images
- **Hour 6-12:** Collect 20 ocean + 20 hard negatives
- **Hour 12-18:** Write ground_truth.json, LABELING_GUIDE.md, ci_subset.txt
- **Hour 18-24:** Run baseline harness; create WATER_V0_FAILURES.md

**Day 2 (Hours 24-48):**
- **Hour 24-36:** Implement PR-W1 detector (fix top 3 failure buckets)
- **Hour 36-42:** Re-run harness; validate improvements
- **Hour 42-48:** Lock thresholds from dataset v0 stats

**Day 3 (Hours 48-72):**
- **Hour 48-60:** Add CI job (warning mode); test on subset
- **Hour 60-66:** Second detector iteration (remaining failures)
- **Hour 66-72:** Documentation + PR hygiene

**Acceptance Gates:**
- Dataset v0 has ≥60 images, balanced classes
- Detector v1 produces non-zero edge alignment on ≥70% of true water images
- CI job runs successfully and uploads artifact
- No "production-ready" claims without calibrated thresholds

---

## What NOT to Do

### ❌ Do NOT:
1. **Claim "all PRs complete"** - Only W4 is complete; W1 is stub; W2/W3 don't exist
2. **Ship uncalibrated thresholds as production** - They're targets, not validated gates
3. **Enable water detection by default** - Stub detector has unknown FP rate
4. **Commit full-res images to repo** - Use Git LFS or external storage + commit thumbnails
5. **Invent performance numbers** - "Reduces FP 40% → 10%" is fiction without dataset
6. **Over-precision in plans** - "Hour 3-5: achieve 90% recall" is not realistic
7. **Skip hard negatives** - You need blue walls, sky, screens to prevent silent FP disasters

---

## Engineering Checklist for "Actually Done"

### Validation Infrastructure (PR-W4):
- [x] Harness script exists and runs
- [x] Tests passing (16/16)
- [x] Linting clean
- [x] Deterministic with --seed
- [x] Schema documented
- [x] Regression checker works
- [x] CI subset support

### Water Detector (PR-W1):
- [ ] Returns non-empty mask
- [ ] Confidence scoring implemented
- [ ] Scene-aware tuning (pool vs ocean)
- [ ] Implementation version tracked
- [ ] Edge alignment > 0 on real images

### Dataset v0:
- [ ] ≥60 labeled images collected
- [ ] ground_truth.json committed
- [ ] LABELING_GUIDE.md written
- [ ] CI subset selected (12 images)
- [ ] Hard negatives included

### Calibration:
- [ ] Baseline report generated
- [ ] Thresholds derived from data
- [ ] Calibration methodology documented
- [ ] Regression baseline committed

### CI Integration:
- [ ] GitHub Actions job added
- [ ] Runs on CI subset
- [ ] Uploads JSON artifact
- [ ] Emits warnings on regression

**Current Score:** 7/26 (27%)  
**Blocker:** No dataset, no detector implementation

---

## Bottom Line

**What you have:**
- Excellent validation infrastructure (PR-W4)
- Clean code, good tests, solid design

**What you don't have:**
- A detector that produces masks
- Data to validate against
- Calibrated thresholds
- Regression baseline

**What "meaningful progress" looks like:**
- Path A: 72 hours to dataset + detector v1 + calibration
- Path B: 6 hours to minimally less-broken stub + commit to Path A

**What to avoid:**
- Claiming completion without data
- Shipping uncalibrated detection as "production"
- Inventing performance metrics before measurement

---

## Recommended Next Action (Right Now)

1. **Merge PR-W4** with accurate description:
   - "Validation infrastructure complete; detector stub; awaiting dataset v0"
2. **Start collecting images** (Path A, Hour 0):
   - Pool: 20 images
   - Ocean: 20 images
   - Hard negatives: 20 images (blue walls, sky, reflections, screens)
3. **Create ground_truth.json** (Path A, Hour 12)
4. **Run baseline harness** (Path A, Hour 18)
5. **Implement detector v1** driven by failures (Path A, Hour 24-36)

**If you paste your current PR-W4 git diff, I'll verify it matches the "accurate wording" standard before merge.**
