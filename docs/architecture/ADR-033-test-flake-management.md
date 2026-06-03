# ADR-033: Test Flake Management

**Status:** Active
**Date:** 2026-02-16
**Authors:** RC219805 + GitHub Copilot CLI
**Related:** Issue #796 (CI Health & Stability), ADR-031 (Test Dependency Isolation)

---

## Context

Test flakiness (intermittent failures) erodes CI reliability and developer confidence. A test that passes/fails non-deterministically creates several problems:

1. **False negatives** - Real bugs masked by flaky test noise
2. **Wasted time** - Developers re-running CI or debugging phantom failures
3. **Degraded trust** - "Just re-run it" becomes the default response
4. **Hidden technical debt** - Underlying race conditions, timing dependencies, or environmental coupling go unaddressed

This repository previously had **100% CI failure rate** due to systemic issues. While ADR-031 addressed dependency isolation (a major flake source), we need ongoing monitoring to prevent regression and catch new sources of flakiness.

---

## Decision

We will implement **automated flake rate tracking** with three components:

### 1. Flake Ledger (`tests/flake_ledger.json`)

**Purpose:** Persistent storage of test execution history

**Schema:**
```json
{
  "version": "1.0.0",
  "last_updated": "2026-02-16T01:31:00Z",
  "config": {
    "flake_threshold": 0.01,        // 1% - monitored
    "quarantine_threshold": 0.03,    // 3% - auto-quarantine eligible
    "min_runs_for_analysis": 10,     // Minimum runs before calculating rate
    "auto_quarantine_enabled": false // Manual review required for now
  },
  "tests": {
    "tests/path/test_example.py::test_foo": {
      "test_id": "tests/path/test_example.py::test_foo",
      "total_runs": 50,
      "passes": 48,
      "failures": 2,
      "flake_count": 1,              // Times outcome switched
      "flake_rate": 0.02,             // 2%
      "last_run": "2026-02-16T01:31:00Z",
      "last_outcome": "passed",
      "last_failure": "2026-02-15T12:00:00Z",
      "status": "monitored",          // stable | monitored | quarantined
      "history": [...]                // Last 20 runs
    }
  }
}
```

**Flake Detection Logic:**
- A "flake" is when a test's outcome **differs from the previous run**
- Example: `passed → failed → passed` = 2 flakes
- Flake rate = `flake_count / total_runs`

**Status Classification:**
- **stable**: Flake rate < 1%
- **monitored**: 1% ≤ flake rate < 3%
- **quarantined**: Flake rate ≥ 3%

---

### 2. Tracking Script (`scripts/track_test_flakes.py`)

**Purpose:** Parse pytest JSON reports and update ledger

**Usage:**
```bash
# Run tests with JSON reporting
pytest --json-report --json-report-file=report.json

# Update ledger
python scripts/track_test_flakes.py report.json
```

**Features:**
- Parses `pytest-json-report` output
- Updates test execution counts
- Detects flake transitions
- Calculates flake rates
- Maintains history (last 20 runs per test)

---

### 3. CI Integration

**Workflow:** `.github/workflows/ci-quality-firewall.yml`

**Job:** `flake-analysis` (runs after `test-core` and `test-ml`)

**Steps:**
1. Restore prior `flake-ledger` artifact from latest successful `main` run (if available)
2. Collect JSON reports from all test jobs
3. Update flake ledger with `track_test_flakes.py`
4. Generate markdown flake report with `analyze_flakes.py`
5. Upload updated ledger + report as CI artifacts
6. Warn if repo-wide flake rate > 1% (non-blocking)

**Artifacts:**
- `flake-ledger` - Updated ledger (retained 30 days)
- `flake-report` - Human-readable markdown summary (retained 30 days)

---

## Quarantine Mechanism

### When to Quarantine

A test should be quarantined if:
- **Flake rate ≥ 3%** over 10+ runs, **OR**
- **Manual decision** by maintainer (persistent flakiness, blocking PRs)

### How to Quarantine

**Option A: `pytest-rerunfailures` (RECOMMENDED)**

```python
import pytest

@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_sometimes_flaky():
    # Test will be rerun up to 3 times if it fails
    pass
```

**Option B: Manual skip with tracking**

```python
@pytest.mark.skip(reason="Quarantined: 5% flake rate - Issue #XYZ")
def test_flaky_needs_fix():
    pass
```

Add `pytest-rerunfailures>=14.0` to `requirements/dev.in` (already done).

### Quarantine Exit Criteria

To remove quarantine status:
1. Root cause identified and fixed
2. Test passes 20 consecutive runs locally
3. Flake rate drops to <1% in CI (ledger verification)
4. Remove `@pytest.mark.flaky` decorator or `@pytest.mark.skip`

---

## Flake Rate Target

**Repository-wide goal:** **< 1% flake rate**

**Calculation:**
```
repo_flake_rate = sum(all_flake_counts) / sum(all_total_runs)
```

**Current baseline:** Not yet established (ledger empty)

**Enforcement:**
- ⚠️ **Warn** if flake rate > 1% (non-blocking)
- ❌ **Future:** Block if flake rate > 3% (after baseline established)

---

## Common Flake Sources & Fixes

### 1. Race Conditions / Timing

**Symptom:** Test fails intermittently with timeout or assertion errors

**Fix:**
- Use proper synchronization (locks, events, conditions)
- Avoid `time.sleep()` - use polling with timeout
- Increase timeouts for genuinely slow operations

**Example:**
```python
# BAD - brittle timing
time.sleep(0.5)
assert result is not None

# GOOD - wait with timeout
def wait_for(condition, timeout=5.0):
    start = time.time()
    while not condition():
        if time.time() - start > timeout:
            raise TimeoutError()
        time.sleep(0.01)

wait_for(lambda: result is not None)
```

### 2. Environmental Dependencies

**Symptom:** Test passes locally, fails in CI (or vice versa)

**Fix:**
- Mock external services (network, filesystem, time)
- Use fixtures to control environment
- Avoid assuming file paths, user directories, network state

**Example:**
```python
# BAD - assumes ~/data exists
data_path = Path.home() / "data" / "test.jpg"

# GOOD - use tmp_path fixture
def test_foo(tmp_path):
    data_path = tmp_path / "test.jpg"
```

### 3. Test Order Dependencies

**Symptom:** Test passes in isolation, fails when run with others

**Fix:**
- Ensure tests are independent (no shared state)
- Use fresh fixtures per test
- Avoid global state mutation

**Example:**
```python
# BAD - global state
CACHE = {}

def test_foo():
    CACHE["key"] = "value"

# GOOD - fixture isolation
@pytest.fixture
def cache():
    return {}

def test_foo(cache):
    cache["key"] = "value"
```

### 4. Non-deterministic Inputs

**Symptom:** Random test failures with different values

**Fix:**
- Seed random generators
- Use property-based testing (Hypothesis) with examples
- Avoid relying on dict/set iteration order (Python 3.7+ is ordered, but be explicit)

**Example:**
```python
# BAD - random without seed
import random
value = random.randint(1, 100)

# GOOD - seeded or deterministic
random.seed(42)
value = random.randint(1, 100)

# BETTER - Hypothesis with fixed examples
from hypothesis import given, example, strategies as st

@given(st.integers(min_value=1, max_value=100))
@example(42)  # Always test this specific case
def test_foo(value):
    pass
```

---

## Workflow

### For Developers

**When you see a flaky test:**
1. Check flake ledger: `.venv/bin/python scripts/analyze_flakes.py`
2. Reproduce locally: Run test 20+ times
3. Identify root cause (use debugger, add logging)
4. Fix or quarantine (with issue tracking)
5. Verify fix: Run 20+ times locally, monitor CI

**When writing new tests:**
- Avoid timing dependencies
- Use fixtures for isolation
- Mock external dependencies
- Test deterministically

### For CI

**On every PR:**
1. Tests run with `--json-report`
2. Flake ledger updated with results
3. Flake report generated
4. Warning if repo flake rate > 1%

**Monthly review:**
- Check ledger for newly monitored tests
- Triage quarantined tests (fix vs disable)
- Update thresholds if needed

---

## Rationale

### Why Track Flakes?

**Visibility:** You can't fix what you can't measure.

**Proactive:** Catch flakes early before they become chronic.

**Data-driven:** Prioritize fixes by flake rate (3% flake = higher priority than 1.5%).

### Why Not Just Re-run Failed Tests Automatically?

Re-running **masks the problem** instead of fixing it. While `pytest-rerunfailures` is useful for **quarantine**, it should not be the default. We want to:
1. **Fix root causes** (race conditions, timing issues)
2. **Track trends** (is flakiness increasing?)
3. **Maintain trust** (green CI means "works", not "eventually passed")

### Why Non-blocking (For Now)?

During the **baseline establishment phase** (first 30 days), we:
- Gather data without disrupting workflow
- Identify chronic flakes vs one-off environmental issues
- Set realistic thresholds based on actual data

After baseline:
- Transition to **blocking** for repo flake rate > 3%
- Require fixes or quarantine for new tests with flake rate > 1%

---

## Enforcement Checklist

- [x] Flake ledger schema defined (`tests/flake_ledger.json`)
- [x] Tracking script implemented (`scripts/track_test_flakes.py`)
- [x] Analysis script implemented (`scripts/analyze_flakes.py`)
- [x] pytest-json-report added to CI dependencies
- [x] pytest-rerunfailures added to dev dependencies
- [x] CI job added (`flake-analysis`)
- [x] Test jobs generate JSON reports
- [x] ADR published (this document)
- [x] CONTRIBUTING.md updated with flake guidance
- [ ] Baseline established (30 days of data)
- [ ] Thresholds tuned based on baseline
- [ ] Transition to blocking enforcement (after baseline)

---

## References

- **Issue #796:** CI Health & Stability
- **ADR-031:** Test Dependency Isolation Contract
- **PR #952:** Flake Rate Monitoring Implementation (this PR)
- **pytest-json-report:** https://pypi.org/project/pytest-json-report/
- **pytest-rerunfailures:** https://pypi.org/project/pytest-rerunfailures/
- **Hypothesis:** https://hypothesis.readthedocs.io/

---

## Appendix: Example Flake Report

```markdown
## 📊 Flake Analysis Report

**Total tests tracked:** 2,491

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Stable | 2,485 | 99.8% |
| 🟡 Monitored | 5 | 0.2% |
| 🔴 Quarantined | 1 | 0.0% |

**Repository-wide flake rate:** 0.85% (21 flakes / 2,491 runs)

**Thresholds:** Flake = 1.0%, Quarantine = 3.0%

### ⚠️ Top Flaky Tests

| Test | Status | Flake Rate | Runs |
|------|--------|------------|------|
| `test_segmentation_integration` | 🔴 | 3.50% | 7/200 |
| `test_material_classifier_gpu` | 🟡 | 1.80% | 9/500 |
| `test_depth_estimation_batch` | 🟡 | 1.20% | 6/500 |
```

---

**End of ADR-033**
