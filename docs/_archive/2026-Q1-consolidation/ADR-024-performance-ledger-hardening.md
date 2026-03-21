# ADR-024: Performance Ledger Design Hardening

**Status:** Accepted
**Date:** 2026-02-07
**Authors:** Transformation Portal Architect
**Supersedes:** N/A (initial ledger hardening)

---

## Context

The performance ledger system had several critical design flaws that would cause production issues:

1. **Tautology Test**: Test always passed (`assert bucket is not None or bucket is None`)
2. **Semantic Specificity Bug**: Range buckets falsely appeared more specific than single-concept buckets
3. **Fragile Name-Based Tests**: Tests checked bucket names rather than filter contracts
4. **Missing Boundary Tests**: Off-by-one errors not prevented
5. **Code Repetition**: No fixture factory for test capsules
6. **Inaccurate GPU Timing**: No synchronization before reading timers
7. **Missing Tail Latency Support**: No p90/p99 threshold infrastructure

These were not superficial issues—they represented fundamental gaps between "feels like it works" and "reliable performance guardrail."

---

## Decision

### 1. Always-Return-Bucket Contract (SRE Best Practice)

**Problem:** `get_bucket_for_capsule()` could return `None`, causing samples to be dropped.

**Solution:** Implement catch-all bucket with very lenient thresholds.

```python
PerformanceBucket(
    name="unknown",
    filters={},  # Matches everything
    p50_threshold_sec=60.0,
    p95_threshold_sec=120.0,
    description="Catch-all bucket for unclassified scenarios",
)
```

**Contract:**
- `get_bucket_for_capsule()` now returns `PerformanceBucket` (not `Optional[PerformanceBucket]`)
- Raises `ValueError` only if catch-all is missing (indicates config bug)
- Unknown scenarios become visible via bucket name rather than being silently dropped

**Rationale:** "Tail latency is where monsters live" (Google SRE Book). Must measure even when categorization is imperfect.

---

### 2. Concept-Based Specificity Scoring

**Problem:** `pixel_count_min + pixel_count_max` made range buckets look more specific than single-concept buckets.

**Solution:** Implement semantic scoring:

```python
def compute_specificity(filters: Dict[str, Any]) -> int:
    """Compute specificity score based on concepts, not key count.

    Scoring:
    - scene_type: +10 (primary discriminator)
    - device: +5 (hardware-specific)
    - backend_id: +5 (model-specific)
    - pixel_count range: +3 (counts once even if min+max)
    - dimension_adjustment: +1 (secondary detail)
    """
    score = 0
    if "scene_type" in filters:
        score += 10
    if "device" in filters:
        score += 5
    if "backend_id" in filters:
        score += 5
    # pixel_count range counts as ONE concept
    if "pixel_count_min" in filters or "pixel_count_max" in filters:
        score += 3
    if "dimension_adjustment" in filters:
        score += 1
    return score
```

**Benefits:**
- Scene-type buckets (score=10) correctly beat range buckets (score=3)
- Device-specific buckets (score=5) beat generic ranges
- Tie-breaking by name ensures determinism

---

### 3. Filter-Based Contract Tests

**Problem:** Tests checked bucket names (implementation detail) rather than filter coverage (contract).

**Before (Fragile):**
```python
assert any("aerial" in name for name in bucket_names)
```

**After (Robust):**
```python
assert any(b.filters.get("scene_type") == "aerial" for b in DEFAULT_BUCKETS)
assert any(b.filters == {} for b in DEFAULT_BUCKETS), "Missing catch-all bucket"
```

**Benefits:**
- Tests survive bucket renames
- Tests verify actual matching behavior
- Catch-all presence is now contract-enforced

---

### 4. Boundary Tests for Ranges

**Added:**
```python
def test_pixel_count_range_boundaries(make_capsule):
    bucket = PerformanceBucket(
        filters={"pixel_count_min": 20_000_000, "pixel_count_max": 50_000_000},
        ...
    )

    # Exactly at min (inclusive)
    assert bucket.matches(make_capsule(pixel_count=20_000_000))

    # Exactly at max (inclusive)
    assert bucket.matches(make_capsule(pixel_count=50_000_000))

    # Just below min (exclusive)
    assert not bucket.matches(make_capsule(pixel_count=19_999_999))

    # Just above max (exclusive)
    assert not bucket.matches(make_capsule(pixel_count=50_000_001))
```

**Benefits:**
- Prevents off-by-one errors
- Documents range semantics (inclusive bounds)
- Catches boundary condition bugs early

---

### 5. Pytest Fixture Factory

**Problem:** Repetitive capsule creation, poor error messages on failure.

**Solution:**
```python
@pytest.fixture
def make_capsule():
    def _make(**overrides):
        base = {
            "image_id": "test_image",
            "pixel_count": 48_000_000,
            "timings": {"total": 10.0},
            # ... sensible defaults
        }
        base.update(overrides)
        return PerformanceCapsule(**base)
    return _make
```

**Usage:**
```python
def test_something(make_capsule):
    capsule = make_capsule(scene_type="pool", pixel_count=20_000_000)
    assert ...
```

**Benefits:**
- DRY (Don't Repeat Yourself)
- Clear override intent
- Better failure messages (parameter names visible)

---

### 6. GPU/MPS Synchronization for Accurate Timing

**Problem:** GPU/MPS operations are asynchronous. Can't just time Python call.

**Solution:**
```python
class TimingContext:
    def __init__(self, phase_name: str, device: Optional[str] = None):
        self.device = device
        # ...

    def _sync_device(self):
        if self.device in {"mps", "cuda"}:
            try:
                import torch
                if self.device == "mps" and torch.backends.mps.is_available():
                    torch.mps.synchronize()
                elif self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass  # Fall back to CPU timing

    def __enter__(self):
        self._sync_device()  # Sync before start
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._sync_device()  # CRITICAL: Sync before stop
        self.elapsed = time.perf_counter() - self.start
```

**Benefits:**
- Accurate GPU timing (not just kernel launch time)
- Graceful fallback when torch unavailable
- Zero overhead when device="cpu"

---

### 7. Multi-Grade Threshold Support

**Added to PerformanceBucket:**
```python
@dataclass
class PerformanceBucket:
    p50_threshold_sec: float
    p90_threshold_sec: Optional[float] = None
    p95_threshold_sec: Optional[float] = None
    p99_threshold_sec: Optional[float] = None

    def check_threshold(self, percentile: int, value: float) -> bool:
        threshold = getattr(self, f"p{percentile}_threshold_sec", None)
        if threshold is None:
            return True  # No threshold = pass
        return value <= threshold
```

**Benefits:**
- Future-proof for tail latency analysis
- Flexible threshold enforcement
- Optional thresholds don't break existing buckets

---

## Consequences

### Positive

1. **No More Silent Sample Loss**: Catch-all bucket ensures all capsules are measured
2. **Correct Specificity Ordering**: Scene-type buckets correctly preferred over generic ranges
3. **Robust Contract Tests**: Tests survive implementation changes
4. **Boundary Correctness**: Off-by-one errors prevented
5. **DRY Tests**: Fixture factory eliminates repetition
6. **Accurate GPU Timing**: MPS/CUDA timings now reflect actual execution time
7. **Future-Proof Thresholds**: p90/p99 support ready for use

### Technical Debt Eliminated

- ✅ Tautology test replaced with real contract test
- ✅ Specificity is semantic, not key-count based
- ✅ Tests check filters, not names
- ✅ Boundary conditions fully tested
- ✅ Fixtures eliminate test repetition
- ✅ Timing is GPU-synchronized
- ✅ Multi-grade thresholds supported

### Breaking Changes

**Type signature change:**
```python
# Before
def get_bucket_for_capsule(...) -> Optional[PerformanceBucket]:

# After
def get_bucket_for_capsule(...) -> PerformanceBucket:
```

**Migration:**
- Code checking `if bucket is None` can be simplified
- `detect_regression()` in `ledger.py` simplified (always has bucket)
- No external API users known; change is internal to metrics module

---

## Alternatives Considered

### Option B: "Return Optional, Log Warning"

**Rejected:** Warnings get ignored in production. Silent failures are worse than loud failures.

### Option C: "Use Default Bucket When None"

**Rejected:** Which default? Arbitrary choice hides the real issue. Better to make unknown scenarios explicit.

---

## Implementation Status

- ✅ `compute_specificity()` function added
- ✅ `PerformanceBucket.specificity` property added
- ✅ `get_bucket_for_capsule()` updated to use specificity
- ✅ Catch-all bucket added to `DEFAULT_BUCKETS`
- ✅ Tautology test replaced with `test_always_returns_bucket_never_none`
- ✅ `test_specificity_score_prevents_range_cheating` added
- ✅ `test_default_buckets_cover_apex_scenarios` checks filters
- ✅ `test_pixel_count_range_boundaries` added
- ✅ `make_capsule` pytest fixture added
- ✅ Tests refactored to use `make_capsule`
- ✅ `TimingContext` updated with GPU sync
- ✅ Multi-grade threshold support added (p90, p99)
- ✅ Documentation updated

**Tests passing:** 43 passed, 1 skipped (CUDA not available)

---

## References

- Google SRE Book: "Monitoring Distributed Systems" (tail latency importance)
- Martin Fowler: "Test Fixtures" (DRY testing patterns)
- NVIDIA CUDA Best Practices: GPU synchronization for accurate timing
- Performance ledger implementation: `src/transformation_portal/metrics/`

---

## Success Criteria

After fixes:
- ✅ No tautology tests
- ✅ Specificity is semantic, not key-count based
- ✅ Tests check filters, not names
- ✅ Boundary conditions fully tested
- ✅ Fixtures eliminate repetition
- ✅ Timing is GPU-synchronized
- ✅ System is future-proof for multi-grade thresholds

**Status:** All criteria met. This is the difference between "feels like it works" and "reliable performance guardrail."
