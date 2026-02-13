# ADR-023 Implementation Guide

**Target:** Transformation Portal Specialist
**Authority:** Transformation Portal Architect
**ADR:** ADR-023 (Post-PR #841 Hardening Strategy)
**Date:** 2026-02-05

---

## Overview

This document provides implementation guidance for Phase 2 (Performance Ledger) and Phase 3 (Backend Selection Truth) as approved in ADR-023.

**Status:**
- ✅ Phase 1: Input Hygiene (COMPLETED - commit 4761e2e5)
- 🎯 Phase 2: Performance Ledger (READY TO IMPLEMENT)
- 🎯 Phase 3: Backend Selection Truth (READY TO IMPLEMENT)

---

## Phase 2: Performance Ledger Implementation

### Files to Create/Modify

#### 1. Complete `tools/performance_ledger.py`

**Current Status:** Skeleton created with dataclasses and stubs
**TODO:** Implement actual manifest parsing logic

**Key Functions to Implement:**

```python
def parse_manifests(manifests_dir: Path) -> List[Dict[str, Any]]:
    """IMPLEMENTED: Load all manifest JSONs from directory."""
    # ✅ Already implemented

def extract_timings(manifests: List[Dict[str, Any]]) -> Tuple[List[float], int, int]:
    """TODO: Extract timing from actual manifest schema.

    Current manifest structure (from src/transformation_portal/lux_depth_v3/manifest.py):
    - CombinedManifest.timing: TimingMetadata
      - total_sec: float
      - depth_sec: float
      - v2_sec: float (optional)
    - CombinedManifest.status: str ("success", "error", etc.)

    Implementation:
    1. Iterate over manifests
    2. Extract manifest.get("timing", {}).get("total_sec")
    3. Extract manifest.get("status") for success/failure count
    4. Return (timings, success_count, failure_count)
    """
    # TODO: Replace placeholder with actual extraction

def compute_statistics(timings: List[float]) -> Statistics:
    """✅ Already implemented with NumPy."""
    # No changes needed

def capture_environment() -> EnvironmentMetadata:
    """✅ Already implemented with torch detection."""
    # No changes needed

def detect_regressions(...) -> List[Regression]:
    """✅ Already implemented with threshold checks."""
    # No changes needed

def format_markdown(...) -> str:
    """✅ Already implemented with table formatting."""
    # No changes needed

def main() -> int:
    """TODO: Wire up the full workflow.

    Implementation:
    1. Parse args (already done)
    2. If --manifests-dir and --output (baseline mode):
       a. Parse manifests
       b. Extract timings
       c. Compute statistics
       d. Capture environment
       e. Create Baseline dataclass
       f. Write to --output as JSON
    3. If --baseline and --compare (comparison mode):
       a. Load baseline JSON
       b. Parse comparison manifests
       c. Extract timings
       d. Compute current statistics
       e. Detect regressions
       f. Format markdown report
       g. Write to --output as markdown
       h. If --emit-json, write current stats as JSON
    4. Return 0 on success, 1 on regression (for CI usage)
    """
    # TODO: Implement full workflow
```

**Testing Requirements:**

```bash
# Unit tests to create
tests/test_performance_ledger.py

test_parse_manifests_valid_directory()
test_parse_manifests_empty_directory()
test_parse_manifests_invalid_json()

test_extract_timings_from_manifests()
test_extract_timings_handles_missing_fields()

test_compute_statistics_correctness()
test_compute_statistics_empty_list_raises()

test_detect_regressions_p95_threshold()
test_detect_regressions_mean_threshold()
test_detect_regressions_failure_rate()
test_detect_regressions_no_regressions()

test_format_markdown_with_regressions()
test_format_markdown_without_regressions()

test_baseline_serialization_roundtrip()
```

**Manual Testing:**

```bash
# 1. Create test manifests directory with known values
mkdir -p test_manifests
cat > test_manifests/manifest_1.json <<'EOF'
{
  "timing": {"total_sec": 10.5},
  "status": "success"
}
EOF

cat > test_manifests/manifest_2.json <<'EOF'
{
  "timing": {"total_sec": 12.3},
  "status": "success"
}
EOF

# 2. Capture baseline
python tools/performance_ledger.py \
  --manifests-dir test_manifests \
  --output test_baseline.json

# 3. Verify baseline JSON structure
cat test_baseline.json | python -m json.tool

# 4. Create comparison manifests (slower)
mkdir -p test_manifests_slow
cat > test_manifests_slow/manifest_1.json <<'EOF'
{
  "timing": {"total_sec": 15.0},
  "status": "success"
}
EOF

# 5. Run comparison
python tools/performance_ledger.py \
  --baseline test_baseline.json \
  --compare test_manifests_slow \
  --output test_report.md

# 6. Verify report shows regression
cat test_report.md

# Cleanup
rm -rf test_manifests test_manifests_slow test_baseline.json test_report.md
```

---

## Phase 3: Backend Selection Truth Implementation

### Files to Modify

#### 1. `src/transformation_portal/lux_depth_v3/manifest.py`

**Add BackendSelectionMetadata dataclass:**

```python
@dataclass
class BackendSelectionMetadata:
    """Backend selection audit trail (added in v2.0.1).

    Tracks requested vs resolved backend for transparency.
    """
    requested_backend: Optional[str]  # User-specified or None (auto)
    resolved_backend: str             # Actual backend used
    resolution_status: str            # "success", "fallback", "error"
    resolution_reason: Optional[str]  # Why fallback occurred (if any)
    model_id: str                     # HuggingFace model ID or checkpoint path
    device: str                       # Resolved device (mps/cuda/cpu)
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BackendSelectionMetadata:
        """Deserialize from dictionary."""
        schema_version = data.get("schema_version", "1.0")
        if schema_version != "1.0":
            raise ValueError(f"Unsupported BackendSelectionMetadata schema: {schema_version}")

        return cls(
            requested_backend=data.get("requested_backend"),
            resolved_backend=data["resolved_backend"],
            resolution_status=data["resolution_status"],
            resolution_reason=data.get("resolution_reason"),
            model_id=data["model_id"],
            device=data["device"],
            schema_version=schema_version,
        )
```

**Modify CombinedManifest dataclass:**

```python
@dataclass
class CombinedManifest:
    # ... existing fields ...
    backend_selection: Optional[BackendSelectionMetadata] = None  # NEW

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with backend_selection."""
        data = asdict(self)
        # Existing serialization logic...
        # backend_selection will be included automatically by asdict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CombinedManifest:
        """Deserialize from dictionary with backend_selection."""
        # ... existing deserialization ...

        # Deserialize backend_selection if present
        backend_selection = None
        if "backend_selection" in data and data["backend_selection"] is not None:
            backend_selection = BackendSelectionMetadata.from_dict(data["backend_selection"])

        return cls(
            # ... existing fields ...
            backend_selection=backend_selection,
        )
```

#### 2. `src/transformation_portal/lux_depth_v3/orchestrator.py`

**Add helper method:**

```python
def _capture_backend_metadata(
    self,
    requested: Optional[str],
    engine: DA3InferenceEngine,
) -> BackendSelectionMetadata:
    """Capture backend selection decision for manifest.

    Args:
        requested: User-specified backend (from config.depth_backend)
        engine: Initialized DA3InferenceEngine instance

    Returns:
        BackendSelectionMetadata with selection audit trail
    """
    resolved = "depth_anything_v3"  # Current reality (DA3 is only backend)
    status = "success"
    reason = None

    # Check for mismatch (e.g., user requested depth_pro but got DA3)
    if requested and requested != resolved:
        status = "fallback"
        reason = f"Requested '{requested}' not available, using '{resolved}' (ADR-019 not yet implemented)"

    return BackendSelectionMetadata(
        requested_backend=requested,
        resolved_backend=resolved,
        resolution_status=status,
        resolution_reason=reason,
        model_id=self.config.model_variant.huggingface_id,
        device=str(engine.device),
    )
```

**Modify `enhance_batch()` method:**

```python
def enhance_batch(self, input_dir: Path, image_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Process a batch of images with backend selection truth.

    Changes in v2.0.1 (ADR-023):
    - Capture backend selection metadata
    - Log truth line on startup
    - Include backend_selection in manifest
    """
    batch_start = time.time()

    # ... existing input discovery code ...

    # NEW: Capture backend selection BEFORE processing
    backend_metadata = self._capture_backend_metadata(
        requested=self.config.depth_backend,
        engine=self.engine,
    )

    # NEW: Log truth line
    logger.info(
        "Backend selection: requested=%s resolved=%s status=%s device=%s model=%s",
        backend_metadata.requested_backend or "auto",
        backend_metadata.resolved_backend,
        backend_metadata.resolution_status,
        backend_metadata.device,
        backend_metadata.model_id,
    )

    # NEW: Log warning if fallback occurred
    if backend_metadata.resolution_status == "fallback":
        logger.warning("Backend fallback: %s", backend_metadata.resolution_reason)

    # ... existing batch processing code ...

    # MODIFY: Add backend_selection to manifest
    # Find where CombinedManifest is created and add:
    manifest = CombinedManifest(
        # ... existing fields ...
        backend_selection=backend_metadata,  # NEW
    )

    # ... rest of enhance_batch ...
```

**Testing Requirements:**

```bash
# Unit tests to create/modify
tests/test_backend_selection.py

test_backend_selection_metadata_schema()
test_backend_selection_serialization_roundtrip()

test_capture_backend_metadata_success_path()
  # requested=None (auto), resolved=depth_anything_v3, status=success

test_capture_backend_metadata_explicit_da3()
  # requested=depth_anything_v3, resolved=depth_anything_v3, status=success

test_capture_backend_metadata_fallback()
  # requested=depth_pro, resolved=depth_anything_v3, status=fallback

test_manifest_includes_backend_selection()
  # CombinedManifest.to_dict() includes backend_selection

test_manifest_backward_compatible()
  # CombinedManifest.from_dict() handles missing backend_selection

# Integration test
tests/test_orchestrator.py (modify existing)

test_enhance_batch_logs_backend_truth_line()
  # Verify INFO log contains backend selection details

test_enhance_batch_warns_on_fallback()
  # Verify WARNING log when requested != resolved

test_enhance_batch_manifest_includes_backend()
  # Verify manifest JSON contains backend_selection
```

---

## Implementation Checklist

### Phase 2: Performance Ledger

- [ ] Complete `extract_timings()` with actual manifest schema
- [ ] Complete `main()` workflow (baseline mode + comparison mode)
- [ ] Create unit tests in `tests/test_performance_ledger.py`
- [ ] Run manual testing with test manifests
- [ ] Verify baseline JSON schema matches documentation
- [ ] Verify markdown report format matches examples
- [ ] Test edge cases (empty manifests, missing fields, corrupted JSON)

### Phase 3: Backend Selection Truth

- [ ] Add `BackendSelectionMetadata` to `manifest.py`
- [ ] Update `CombinedManifest` serialization/deserialization
- [ ] Add `_capture_backend_metadata()` to `orchestrator.py`
- [ ] Modify `enhance_batch()` to capture + log backend selection
- [ ] Include `backend_selection` in manifest creation
- [ ] Create unit tests for `BackendSelectionMetadata`
- [ ] Create unit tests for backend capture logic
- [ ] Add integration test for truth-line logging
- [ ] Verify manifest JSON output includes backend_selection
- [ ] Test backward compatibility (old manifests without backend_selection)

### Integration Testing

- [ ] Run full batch with `lux-depth-v3` and verify:
  - Truth-line log appears on startup
  - Manifest contains `backend_selection` field
  - Performance ledger can parse new manifests
- [ ] Test with `--depth-backend depth_pro` (should show fallback)
- [ ] Verify no breaking changes to existing workflows

### Documentation

- [ ] Update `CHANGELOG.md` with v2.0.1 changes
- [ ] Update `README.md` if user-facing behavior changes
- [ ] Ensure `docs/performance/README.md` is accurate
- [ ] Ensure `ADR-023` reflects final implementation

---

## Rollback Plan

If regressions occur:

1. **Phase 2 (Performance Ledger):**
   - Tool is standalone, no orchestrator changes
   - Rollback: Delete `tools/performance_ledger.py` and documentation
   - Impact: Zero (tool was opt-in)

2. **Phase 3 (Backend Selection Truth):**
   - Changes are additive (new manifest field, logging)
   - Rollback: Revert orchestrator.py and manifest.py changes
   - Impact: Minimal (no behavior changes, only metadata/logging)

**Rollback Command:**
```bash
git revert <commit-sha>
git push origin main
```

---

## Success Criteria

**Phase 2 Complete When:**
- ✅ Performance ledger parses manifests correctly
- ✅ Statistics computation validated against known dataset
- ✅ Regression detection works with defined thresholds
- ✅ Baseline can be captured and compared
- ✅ Unit tests pass
- ✅ Manual testing successful

**Phase 3 Complete When:**
- ✅ Manifest includes `backend_selection` metadata
- ✅ Truth-line logging emitted on startup
- ✅ Fallback warning logged when mismatch occurs
- ✅ Unit tests pass
- ✅ Integration test validates logging + manifest
- ✅ Backward compatibility verified

**Both Phases Complete When:**
- ✅ All tests passing
- ✅ CI green
- ✅ Documentation updated
- ✅ No breaking changes to existing workflows
- ✅ Architect review approved

---

## Notes for Specialist

### Manifest Schema Reference

Current `CombinedManifest` structure (from `src/transformation_portal/lux_depth_v3/manifest.py`):

```python
@dataclass
class CombinedManifest:
    input: InputMetadata
    depth: DepthMetadata
    v2: Optional[V2Metadata]
    timing: TimingMetadata
    repro: ReproMetadata
    config_fingerprint: ConfigFingerprint
    # ADD: backend_selection: Optional[BackendSelectionMetadata] = None
```

Where `TimingMetadata` contains:
```python
@dataclass
class TimingMetadata:
    total_sec: float
    depth_sec: float
    v2_sec: Optional[float]
```

### Current Orchestrator Structure

`enhance_batch()` currently:
1. Discovers images with `discover_images()`
2. Initializes `DA3InferenceEngine`
3. Processes images in batch
4. Creates `CombinedManifest` per image
5. Writes manifest to disk

**Insertion Point for Phase 3:**
- Capture backend metadata AFTER engine initialization
- Log truth line BEFORE batch processing loop
- Include backend_selection in manifest creation

### No Breaking Changes Requirement

**Critical:** All changes must be additive:
- New manifest field is OPTIONAL
- Logging is INFO/WARNING level (not ERROR)
- No new exceptions raised
- Existing manifests remain valid
- No CLI flag changes

---

**Implementation Priority:** Implement Phase 2 and Phase 3 in parallel (single PR) as approved in ADR-023.

**Questions?** Escalate to Architect before deviating from this plan.
