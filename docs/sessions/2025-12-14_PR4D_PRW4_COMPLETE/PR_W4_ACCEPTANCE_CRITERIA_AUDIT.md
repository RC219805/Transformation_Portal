# PR-W4 Acceptance Criteria - Line-by-Line Audit

## Source: `docs/PR_WATER_MASK_STRUCTURE.md` Section PR-W4

### Acceptance Criteria from Specification

```
- ✅ Validation harness runs on pool/ocean/non-water scenes
- ✅ Edge alignment metric (primary) computed for all detections
- ✅ Stability metric tracks consistency across perturbations
- ✅ False-positive rate computed for non-water scenes
- ✅ JSON report with summary statistics
- ✅ Performance metrics (processing time) included
```

---

## Actual Implementation Status

### ✅ PASS: "Validation harness runs on pool/ocean/non-water scenes"

**Evidence**:
- `scripts/prw_water_validation.py` exists and is executable
- CLI accepts `--input-dir`, `--ground-truth`, `--output`
- `validate_dataset()` loops over images and calls `validate_single()`
- Ground truth JSON maps image paths to scene_type (pool|ocean|non_water)
- Harness runs end-to-end without errors

**Verdict**: ✅ **PASS** - Fully implemented and tested

---

### ❌ FAIL: "Edge alignment metric (primary) computed for all detections"

**Evidence**:

**Code** (`scripts/prw_water_validation.py`, lines 121-122):
```python
# Compute edge alignment (water mask not available in dict, skip for now)
edge_score = 0.0  # TODO: Get mask from internal water detector
```

**Code** (`scripts/prw_water_validation.py`, line 131):
```python
# Count boundary pixels (mask not available)
boundary_px = 0
```

**Result Data** (all validation results):
```json
{
  "edge_alignment_score": 0.0,
  "boundary_px": 0
}
```

**Method Implementation** (`scripts/prw_water_validation.py`, lines 147-174):
```python
def _compute_edge_alignment(
    self, rgb01: np.ndarray, mask: Optional[np.ndarray]
) -> float:
    """
    Primary metric: edge alignment vs image gradients.
    
    High score = mask boundaries align with image edges.
    """
    if mask is None:
        return 0.0  # <-- Always returns 0.0 because mask is always None
```

**Root Cause**:
- MaterialsV3Engine.process() returns dict without mask
- Line 118: `mask=None,  # Mask not included in JSON dict`
- No mechanism to access mask for boundary analysis

**Spec Says** (line 1099):
> "Edge alignment vs gradients (primary metric)"

**Verdict**: ❌ **FAIL** - Method implemented, but always returns 0.0 because mask unavailable

**Gap**: Primary metric cannot be computed in current implementation

---

### ⚠️ PARTIAL PASS: "Stability metric tracks consistency across perturbations"

**Evidence**:

**Code** (`scripts/prw_water_validation.py`, lines 124-125):
```python
# Compute stability
stability = self._compute_stability(rgb01, depth)
```

**Method Implementation** (`scripts/prw_water_validation.py`, lines 176-218):
```python
def _compute_stability(self, rgb01: np.ndarray, depth: np.ndarray) -> float:
    """
    Stability across minor perturbations (resize/compress jitter).
    
    High score = consistent detection under perturbations.
    """
    # Baseline detection
    baseline_report = self.engine.process(...)
    baseline_coverage = baseline_report.get('materials_v3', {}).get('water_candidate', {}).get('coverage', 0.0)
    
    # Perturbation 1: slight resize (95%)
    resized_report = self.engine.process(...)
    
    # Perturbation 2: JPEG compression simulation (noise)
    noisy_report = self.engine.process(...)
    
    # Compute coverage variance
    coverages = [baseline_coverage, resized_coverage, noisy_coverage]
    std = np.std(coverages)
    
    # Low variance = high stability
    stability = 1.0 - min(std * 5, 1.0)
    return float(stability)
```

**What Works**:
- Runs detection 3 times (baseline, resize, noise)
- Computes coverage variance
- Returns stability score (0-1)
- Actually returns real values (not 0.0)

**What's Limited**:
- Stability of **coverage numbers only** (scalar)
- Cannot measure stability of **mask boundaries** (no mask access)
- Less meaningful than spec intended (spec shows mask boundary analysis in comments)

**Spec Intent** (from context):
Stability should ideally measure whether mask boundaries remain consistent, not just coverage percentage.

**Verdict**: ⚠️ **PARTIAL PASS** - Functional but limited scope

**Gap**: Works for coverage stability, missing boundary stability

---

### ✅ PASS: "False-positive rate computed for non-water scenes"

**Evidence**:

**Code** (`scripts/prw_water_validation.py`, lines 127-128):
```python
# Check false positive
is_fp = (expected_scene == "non_water" and water.present)
```

**Report Generation** (`scripts/prw_water_validation.py`, lines 279-281):
```python
# False positives
"false_positive_count": sum(r.is_false_positive for r in results),
"false_positive_rate": sum(r.is_false_positive for r in results) / max(len(non_water_results), 1),
```

**Validation Logic**:
1. Ground truth labels image as "non_water"
2. Detector reports water.present = True
3. Harness marks as false positive
4. Summary calculates FP count and rate

**Verdict**: ✅ **PASS** - Fully implemented and correct

---

### ✅ PASS: "JSON report with summary statistics"

**Evidence**:

**Code** (`scripts/prw_water_validation.py`, lines 253-293):
```python
def generate_report(
    self, results: List[ValidationResult], output_path: Path
):
    """Generate JSON validation report."""
    # Summary statistics
    pool_results = [r for r in results if r.scene_type == "pool"]
    ocean_results = [r for r in results if r.scene_type == "ocean"]
    non_water_results = [r for r in results if r.scene_type == "non_water"]
    
    summary = {
        "total_images": len(results),
        "pool_scenes": len(pool_results),
        "ocean_scenes": len(ocean_results),
        "non_water_scenes": len(non_water_results),
        
        # Coverage stats
        "pool_avg_coverage": ...,
        "ocean_avg_coverage": ...,
        
        # Edge alignment (primary metric)
        "pool_avg_edge_alignment": ...,
        "ocean_avg_edge_alignment": ...,
        
        # Stability
        "overall_avg_stability": ...,
        
        # False positives
        "false_positive_count": ...,
        "false_positive_rate": ...,
        
        # Performance
        "avg_processing_time_ms": ...,
    }
    
    report = {
        "summary": summary,
        "results": [vars(r) for r in results]
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
```

**Output**:
- Valid JSON format
- Summary statistics by scene type
- Individual results array
- Human-readable (indent=2)
- Console summary printed after generation

**Tests** (`tests/test_prw_water_validation.py`):
- `test_report_generation()` validates JSON structure
- `test_summary_statistics()` validates calculations

**Verdict**: ✅ **PASS** - Fully implemented and tested

---

### ✅ PASS: "Performance metrics (processing time) included"

**Evidence**:

**Code** (`scripts/prw_water_validation.py`, lines 102-104):
```python
# Process
start = time.perf_counter()
result = self.engine.process(rgb01, segmentation_result, depth_map=depth)
elapsed_ms = (time.perf_counter() - start) * 1000
```

**ValidationResult** (`scripts/prw_water_validation.py`, line 57):
```python
@dataclass
class ValidationResult:
    # ...
    processing_time_ms: float
```

**Report Summary** (`scripts/prw_water_validation.py`, line 281):
```python
"avg_processing_time_ms": np.mean([r.processing_time_ms for r in results]),
```

**Verdict**: ✅ **PASS** - Fully implemented

---

## Summary Table

| Criterion | Status | Evidence | Gap |
|-----------|--------|----------|-----|
| Harness runs on pool/ocean/non-water | ✅ PASS | Script exists, CLI works, tests pass | None |
| Edge alignment metric (primary) | ❌ FAIL | Always returns 0.0, mask unavailable | Mask access mechanism |
| Stability metric | ⚠️ PARTIAL | Coverage variance works, boundary stability missing | Mask access for boundary analysis |
| False-positive rate | ✅ PASS | FP detection and rate calculation correct | None |
| JSON report with summary | ✅ PASS | Valid JSON, all stats, human-readable | None |
| Performance metrics | ✅ PASS | Processing time tracked and reported | None |

**Overall Score**: **4.5 / 6 criteria met (75%)**

---

## Critical Gaps

### Gap 1: Primary Metric Non-Functional

**Criterion**: "Edge alignment metric (primary) computed for all detections"

**Status**: Method exists but returns 0.0

**Blocker**: No access to water mask from MaterialsV3Engine

**Impact**: 
- Cannot validate boundary quality (the primary thing to validate)
- Report shows edge_alignment_score: 0.0 for all images
- Target threshold (≥0.6) cannot be tested

**Fix Required**: Implement mask export mechanism (2-3 hours)

### Gap 2: Stability Limited Scope

**Criterion**: "Stability metric tracks consistency across perturbations"

**Status**: Works for coverage, incomplete for boundaries

**Limitation**: Can only measure scalar stability, not spatial consistency

**Impact**: Less meaningful than intended, but still valuable

**Fix Required**: Would benefit from mask access, but functional without it

---

## Additional Testing Gaps

Beyond the 6 acceptance criteria, the spec also lists tests:

From `docs/PR_WATER_MASK_STRUCTURE.md` line 1128:
```
3. **Create tests in `tests/test_prw_water_validation.py`**:
   - test_validation_result_dataclass() - verify schema
   - test_edge_alignment_computation() - verify edge alignment metric
   - test_stability_computation() - verify stability metric
   - test_boundary_extraction() - verify boundary extraction
   - test_false_positive_detection() - verify FP logic
   - test_report_generation() - verify JSON report structure
```

**Actual Tests Created**: 12 tests in `tests/test_prw_water_validation.py`

| Required Test | Status | Notes |
|--------------|--------|-------|
| test_validation_result_dataclass | ✅ | Exists |
| test_edge_alignment_computation | ⚠️ | Exists but tests method in isolation (with synthetic mask), not end-to-end |
| test_stability_computation | ✅ | Exists |
| test_boundary_extraction | ✅ | Exists |
| test_false_positive_detection | ✅ | Exists |
| test_report_generation | ✅ | Exists |

**Test Gap**: Edge alignment tests work with synthetic masks but cannot test real end-to-end behavior because masks unavailable from engine.

---

## Recommended Actions

### To Meet All Acceptance Criteria

**Priority 1: Fix Primary Metric (Critical)**

Implement mask availability mechanism:

**Option A: Debug export flag** (2 hours)
```python
@dataclass
class MaterialsV3Config:
    water_validation_emit_mask: bool = False
    
# In _detect_water_candidate():
if self.config.water_validation_emit_mask and detector_result.mask is not None:
    import base64
    mask_bytes = (detector_result.mask * 255).astype(np.uint8).tobytes()
    water_candidate_dict['mask_base64'] = base64.b64encode(mask_bytes).decode('ascii')
    water_candidate_dict['mask_shape'] = list(detector_result.mask.shape)
```

**Option B: Direct detector API** (1 hour)
```python
# In harness, bypass engine and call detector directly for validation
from lux_depth_v2.water_candidate import WaterCandidateDetector
detector = WaterCandidateDetector()
result = detector.detect(rgb01)
mask = result['mask']
edge_score = self._compute_edge_alignment(rgb01, mask)
```

**Priority 2: Document Limitations**

Add to PR description:
- Primary metric blocked pending mask availability
- Stability limited to coverage (not boundaries)
- Thresholds are targets, not validated

**Priority 3: Update Tests**

Add end-to-end test once mask available:
```python
def test_edge_alignment_end_to_end():
    """Edge alignment computed from real engine output."""
    config = MaterialsV3Config(
        water_detection_enabled=True,
        water_validation_emit_mask=True
    )
    harness = WaterValidationHarness(config)
    result = harness.validate_single(pool_image_path, "pool")
    assert result.edge_alignment_score > 0.0, "Should compute real edge score"
```

---

## Honest Verdict

**Can PR-W4 be merged as-is?**

**Yes, with caveats**:
- Infrastructure is solid and tested
- Non-blocked metrics work correctly
- Foundation is valuable for future validation

**Should it be claimed as "complete"?**

**No**:
- Primary metric (edge alignment) is non-functional
- 4.5/6 criteria met ≠ "complete"
- Would need follow-up PR to implement mask access

**What's the right narrative?**

"PR-W4 implements validation harness infrastructure with coverage, FP, stability, and performance metrics. Edge alignment (primary quality metric) is implemented but blocked pending mask availability mechanism. Recommend merge with follow-up to unlock primary metric."

**For production deployment?**

**Not yet**:
- Need mask access implemented
- Need full PR-W1 detector (not stub)
- Need labeled dataset and threshold calibration

**Current value**:
- Enables systematic testing once unblocked
- Provides actionable metrics today (coverage, FP rate)
- Clear path to completion
