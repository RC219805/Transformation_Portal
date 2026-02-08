# APEX Real Pipeline Integration

**Status:** ✅ **Implemented** (Phase 2 - Hybrid CI)

Real pipeline execution is implemented in `scripts/apex_matrix_runner.py` with explicit dependency checks, device auto-detection, and fail-fast error handling.

---

## CI Strategy: Hybrid Lanes

### PR Lane (Default): Synthetic + Dry-Run

**Purpose:** Fast, deterministic plumbing validation without ML dependencies.

- Runs on every PR
- `--dry-run --synthetic`
- No torch/transformers required
- Validates: schema, contracts, aggregation, reporting
- Runtime: ~30s per job

**What it tests:**
- ✅ Observation/capsule serialization
- ✅ Ledger schema + migrations
- ✅ Aggregation logic (buckets, stats)
- ✅ Gate evaluation (shadow mode)
- ✅ PR comment generation

**What it doesn't test:**
- ❌ Real inference timings
- ❌ Model availability
- ❌ Device-specific behavior

### Real Execution Lane: Manual/Nightly

**Purpose:** Validate real pipeline performance with actual models.

- Triggered manually via `workflow_dispatch` or nightly
- Requires ML dependencies (~5GB: torch, transformers, models)
- Uses committed test fixtures (3 tiny images)
- Runtime: ~2-5 min per job (depending on device)

**How to trigger:**

1. **Locally:**
   ```bash
   pip install -e ".[ml]"  # Install torch + transformers

   python scripts/apex_matrix_runner.py \
     --run-id test-$(git rev-parse --short HEAD) \
     --commit-sha $(git rev-parse HEAD) \
     --workflow-versions v1 v2 \
     --zones local \
     --input-dir ./tests/fixtures/apex_images \
     --sample-size 3 \
     --output-dir ./apex_results \
     --ledger-db ./apex_performance.db
   ```

2. **Via workflow_dispatch** (future):
   - GitHub Actions → APEX Performance Matrix → Run workflow
   - Select branch
   - Check "Enable real execution" input
   - CI will install ML deps + run real pipeline

---

## Implementation Details

### Dependency Checks (Fail-Fast)

Real execution validates ML dependencies before any processing:

```python
ml_available, missing = check_ml_dependencies(require_torch=True)
if not ml_available:
    raise RuntimeError(
        f"Real execution requires: {', '.join(missing)}\n"
        "Install with: pip install torch transformers\n"
        "Or use --dry-run for synthetic testing."
    )
```

**Exit immediately if:**
- `torch` not available
- `transformers` not available
- Input directory missing/empty

### Device Auto-Detection

If `--device` not specified:

```python
if torch.backends.mps.is_available():  # Apple Silicon
    device = "mps"
elif torch.cuda.is_available():         # NVIDIA GPU
    device = "cuda"
else:
    device = "cpu"
```

**Override with:**
- `--device cpu` (CI default)
- `--device mps` (local Mac)
- `--device cuda` (GPU runners)

### Test Fixtures

**Location:** `tests/fixtures/apex_images/`

| File | Size | Purpose |
|------|------|---------|
| `apex_test_aerial.jpg` | ~11KB | Sky/exterior scene |
| `apex_test_interior.jpg` | ~11KB | Interior scene |
| `apex_test_pool.jpg` | ~11KB | Water/pool scene |

**Design:**
- Tiny (128×128) for fast checkout
- Committed (not LFS) for determinism
- Not benchmarks - only plumbing validation

### Synthetic Flag Semantics

| Mode | `--dry-run` | `--synthetic` | `is_synthetic` in capsule |
|------|-------------|---------------|---------------------------|
| **PR CI** | ✅ | ✅ (auto) | `true` |
| **Real local** | ❌ | ❌ | `false` |
| **Real CI** | ❌ | ❌ | `false` |

`--synthetic` is auto-enabled when `--dry-run` is used.

---

## Local Testing Workflow

### Quick Smoke Test (Synthetic)
```bash
python scripts/apex_matrix_runner.py \
  --run-id test-dry \
  --commit-sha test \
  --workflow-versions v1 \
  --zones local \
  --output-dir /tmp/apex_test \
  --dry-run
```

### Real Pipeline (Requires ML Deps)
```bash
# 1. Install ML dependencies
pip install torch transformers

# 2. Run with fixtures
python scripts/apex_matrix_runner.py \
  --run-id test-real \
  --commit-sha $(git rev-parse HEAD) \
  --workflow-versions v1 v2 \
  --zones local \
  --input-dir ./tests/fixtures/apex_images \
  --sample-size 3 \
  --output-dir ./apex_results \
  --ledger-db ./apex_performance.db \
  --device cpu

# 3. Generate PR comment
python scripts/apex_pr_comment.py \
  --ledger-db ./apex_performance.db \
  --output apex_comment.md

# 4. Verify comment
cat apex_comment.md
```

**Expected output:**
- 3 capsules per workflow (v1 + v2)
- Real timings (not fixed 10.0s)
- `is_synthetic=false` in capsules
- PR comment: **not** labeled `[SYNTHETIC DATA]`

---

## Troubleshooting

### Error: "Real execution requires ML dependencies"
**Cause:** Running without `--dry-run` but torch/transformers not installed.

**Fix:**
```bash
pip install torch transformers
# Or use --dry-run if you only want plumbing validation
```

### Error: "--input-dir required for real execution"
**Cause:** Forgot to specify input directory.

**Fix:**
```bash
--input-dir ./tests/fixtures/apex_images
```

### Slow CI / Model downloads
**Cause:** First run downloads models from HuggingFace.

**Fix (future):**
- Add workflow cache for `~/.cache/huggingface`
- Or use self-hosted runner with pre-cached models

### MPS not available on Linux
**Cause:** Ubuntu runners don't support Apple's Metal Performance Shaders.

**Fix:**
```bash
--device cpu  # Explicit override (already default in CI)
```

---

## Design Rationale

### Why hybrid CI?

**PR gating must be fast and deterministic:**
- 30s dry-run beats 5min real run
- No flakiness from model downloads
- No variability from shared CPU contention

**Real runs validate end-to-end correctness:**
- Smoke test: "does the pipeline actually run?"
- Offline mode enforcement: transformers/HF caching works
- Device selection logic: cpu/cuda/mps routing correct

### Why tiny fixtures?

Large test images bloat the repo and slow checkout. The fixtures are **not benchmarks** - they validate plumbing, not performance thresholds.

Performance baselines come from nightly runs with representative datasets (not committed).

### Why not real runs in PRs (yet)?

**Blocked on:**
1. **Model caching strategy** - 5GB download per run is wasteful
2. **Runner stability** - shared ubuntu-latest has noisy CPU timings
3. **Threshold tuning** - need 2+ weeks shadow data before enforcement

Once those are solved, we can optionally enable real runs in PRs (but synthetic mode will stay for fast iteration).

---

## See Also

- [`tests/fixtures/apex_images/README.md`](../../../tests/fixtures/apex_images/README.md) - Fixture specs
- [`docs/apex/phase1/`](../phase1/) - Scaffold + contracts design
- [Issue #868](https://github.com/RC219805/Transformation_Portal/issues/868) - Real pipeline integration tracking
