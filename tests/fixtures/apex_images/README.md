# APEX Test Fixtures

**Purpose:** Deterministic CI fixtures for APEX performance gate testing.

## Images

| Filename | Size | Purpose |
|----------|------|---------|
| `apex_test_aerial.jpg` | ~11KB | Sky/exterior scene |
| `apex_test_interior.jpg` | ~11KB | Interior/architectural scene |
| `apex_test_pool.jpg` | ~11KB | Water/pool scene |

## Design Constraints

- **Tiny (128×128):** ~11KB each for fast CI checkout/processing
- **Committed (not LFS):** Ensures CI always has deterministic inputs
- **Not benchmarks:** These validate schema/plumbing, not performance thresholds

## Usage in CI

The PR gating workflow runs in **dry-run + synthetic mode** and does not process these files.

Real pipeline integration uses these fixtures via `workflow_dispatch` or local testing:

```bash
python scripts/apex_matrix_runner.py \
  --run-id test-001 \
  --commit-sha $(git rev-parse HEAD) \
  --workflow-versions v1 v2 \
  --zones local \
  --input-dir ./tests/fixtures/apex_images \
  --sample-size 3 \
  --output-dir ./apex_results \
  --device cpu
```

## Adding New Fixtures

Keep total fixture set under 50KB to prevent repo bloat. If larger images are needed for smoke tests, use dynamic generation or external storage.
