# APEX Real Pipeline Integration

**Status:** Planned (tracked in issue #XXX)
**Priority:** High
**Complexity:** Medium

## Overview

APEX currently runs in **dry-run/scaffolding mode** with synthetic data. This issue tracks the implementation of real pipeline execution for both V1 and V2 workflows.

## Current State

✅ **Completed:**
- End-to-end APEX architecture (contracts, aggregation, comparison, gating)
- SQLite ledger with schema v2.0.0
- GitHub Actions CI integration
- PR comment generation with verdict + heatmap
- Performance bucketing with concept-based specificity
- Zone-aware metrics collection

⚠️  **Synthetic (dry-run only):**
- `scripts/apex_matrix_runner.py` generates mock `PerformanceCapsule` data
- All CI runs use `--dry-run` flag
- Gate verdicts are based on synthetic measurements

## Real Integration Requirements

### 1. Connect to Orchestrator

**File:** `scripts/apex_matrix_runner.py` (function `run_apex_for_config`)

Replace `NotImplementedError` with:

```python
from transformation_portal.orchestrator import UnifiedOrchestrator
from transformation_portal.config import load_config
from transformation_portal.metrics import timing_context

# Load config for workflow version
config_path = Path(f"config/{config_version}_stable.yaml")
config = load_config(config_path)

# Initialize orchestrator
orchestrator = UnifiedOrchestrator(config)

# Process images with timing capture
capsules = []
for image_path in input_images:
    with timing_context(device=config.device) as timer:
        result = orchestrator.run(image_path)

    # Create PerformanceCapsule from result + timer
    capsule = PerformanceCapsule(
        image_id=image_path.name,
        image_path=str(image_path),
        input_hash=hash_image(image_path),
        original_shape=result.original_shape,
        enforced_shape=result.enforced_shape,
        pixel_count=result.pixel_count,
        dimension_adjustment=result.dimension_adjustment,
        backend_id=config.backend_id,
        device=config.device,
        timings=timer.timings,  # From timing_context
        workflow_version=run_spec.workflow_version,
        zone=zone,
        scene_type=detect_scene_type(image_path),
    )
    capsules.append(capsule)

return Observation(run_spec=run_spec, zone=zone, capsules=capsules)
```

### 2. Add Input Dataset Configuration

**File:** `scripts/apex_matrix_runner.py` (CLI args)

```python
parser.add_argument(
    "--input-dir",
    type=Path,
    default=Path("input_images/source_tiffs"),
    help="Directory containing input images"
)
parser.add_argument(
    "--sample-size",
    type=int,
    default=6,
    help="Number of images to process per run (for CI speed)"
)
```

### 3. Update CI Workflow

**File:** `.github/workflows/apex_performance.yml`

**Change:**
```yaml
# BEFORE (dry-run)
--dry-run

# AFTER (real execution)
--input-dir ./input_images/ci_test_set \
--sample-size 3
```

**Add test dataset preparation:**
```yaml
- name: Download Test Images
  run: |
    mkdir -p input_images/ci_test_set
    # Option 1: Store small test images in repo under assets/
    cp assets/test_images/*.jpg input_images/ci_test_set/

    # Option 2: Download from release artifacts
    gh release download ci-test-images --pattern '*.jpg' --dir input_images/ci_test_set
```

### 4. Add Minimum Sample Size Gating

**File:** `src/transformation_portal/metrics/gate.py`

```python
def evaluate_gate(
    judgement: Judgement,
    mode: GateMode = GateMode.ENFORCE,
    min_samples_per_bucket: int = 5,  # NEW
) -> GateDecision:
    """Evaluate gate and return decision.

    Args:
        judgement: Computed judgement with bucket stats
        mode: Gate enforcement mode
        min_samples_per_bucket: Minimum samples required for reliable p95

    Returns:
        GateDecision with pass/warn/fail verdict
    """
    # Check for insufficient data
    for bucket_name, stats in judgement.bucket_stats.items():
        if stats.count < min_samples_per_bucket:
            logger.warning(
                f"Bucket {bucket_name} has only {stats.count} samples "
                f"(min: {min_samples_per_bucket}). Verdict: insufficient-data"
            )
            if mode == GateMode.ENFORCE:
                # Treat insufficient data as a warning, not a blocker
                # (Alternative: fail hard if you want strict coverage)
                pass

    # ... (rest of existing logic)
```

### 5. Shadow Mode Rollout Plan

**Phase 1: Local validation** (1-2 days)
- Run real pipeline locally on 5-10 images
- Verify capsules have realistic timings
- Confirm ledger aggregation is correct

**Phase 2: Shadow mode in CI** (1 week)
```yaml
# In workflow:
--input-dir ./input_images/ci_test_set \
--sample-size 3 \
--gate-mode shadow  # Reports but doesn't block
```
- Monitor for crashes
- Validate that p95 values are sane
- Compare V1 vs V2 regression reports

**Phase 3: Enforcement mode** (after 1 week clean shadow)
```yaml
--gate-mode enforce  # Actually blocks merges
```

## Test Plan

### Unit Tests
```bash
# New test file: tests/test_apex_real_integration.py

def test_real_pipeline_produces_valid_capsules(tmp_path):
    """Verify real orchestrator emits valid PerformanceCapsules."""
    ...

def test_timing_context_captures_phases(tmp_path):
    """Verify timing breakdown matches expected stages."""
    ...
```

### Integration Test
```bash
# Run actual APEX with real images (non-CI, manual validation)
python scripts/apex_matrix_runner.py \
  --run-id local-$(date +%s) \
  --commit-sha $(git rev-parse HEAD) \
  --workflow-versions v1 v2 \
  --zones local \
  --input-dir ./input_images/source_tiffs \
  --sample-size 6 \
  --output-dir ./apex_results_real
```

Expected:
- 6 images × 2 workflows = 12 capsules
- Timings are non-trivial (> 1s, realistic distribution)
- Ledger contains aggregated stats
- PR comment generator works on real data

## Success Criteria

- [ ] Real pipeline execution works without `--dry-run`
- [ ] Timing measurements are accurate (GPU sync verified)
- [ ] Aggregated p95 values match manual measurements
- [ ] Gate blocks PRs when V2 regresses > 15%
- [ ] PR comments show real performance deltas
- [ ] CI runs complete in < 5 minutes with 3-image test set

## Dependencies

- Issue #XXX: Unified orchestrator V2 implementation
- Issue #XXX: Scene type auto-detection
- PR #864: APEX scaffolding (prerequisite)

## Timeline

- Week 1: Local integration + validation
- Week 2: Shadow mode in CI
- Week 3: Enforcement mode (if shadow is clean)

---

**Tracked in:** GitHub Issue #XXX
**Owner:** TBD
**Milestone:** APEX Production Readiness
