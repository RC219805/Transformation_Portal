# ML CI Optimization: Quick Reference Card

**Status:** APPROVED - Ready for Implementation
**Full Review:** [ML_CI_OPTIMIZATION_STRATEGIC_REVIEW.md](./ML_CI_OPTIMIZATION_STRATEGIC_REVIEW.md)

---

## TL;DR: Architect Decisions

✅ **APPROVED:** ML test conditional execution with path-based gating
✅ **APPROVED:** Dedicated `requirements/ml-ci.txt` with CPU-only torch
✅ **APPROVED:** CI tier structure (PR/Main/Nightly)
❌ **REJECTED:** Deleting `AGENT_TOOLSDIRECTORY` (known anti-pattern)
❌ **REJECTED:** "Fallback to not slow" retry logic (masks flakes)
⚠️ **REQUIRED:** Create ADR-034 (CI Test Execution Tiers) before implementation

---

## Critical Changes Required

### 1. STOP Deleting AGENT_TOOLSDIRECTORY ❌

**Current Code (BROKEN):**
```yaml
- name: Free disk space
  run: sudo rm -rf "$AGENT_TOOLSDIRECTORY"
```

**REPLACE WITH:**
```yaml
- name: Free disk space (selective)
  run: |
    sudo rm -rf /usr/share/dotnet /opt/ghc /usr/local/share/boost
    sudo docker image prune --all --force
    # DO NOT delete AGENT_TOOLSDIRECTORY - breaks Python
    # See: docs/incidents/2026-02-03_ci_python_toolcache_deletion.md
```

**Why:** This broke CI completely in incident 2026-02-03. Don't repeat.

---

### 2. ML Path Matcher (APPROVED)

```yaml
run_ml_check:
  outputs:
    run_ml: ${{ steps.check.outputs.run_ml }}
  steps:
    - name: Fail-safe check
      run: |
        # ALWAYS run on main/develop
        if [[ "${{ github.ref }}" == 'refs/heads/main' || "${{ github.ref }}" == 'refs/heads/develop' ]]; then
          echo "run_ml=true" >> $GITHUB_OUTPUT
          exit 0
        fi

        # Check ML paths
        changed=$(git diff --name-only HEAD~1)
        for path in src/transformation_portal/{depth,upscaling,diffusion,spatial_ai} \
                    requirements/ml*.{in,txt} pyproject.toml; do
          if echo "$changed" | grep -q "$path"; then
            echo "run_ml=true" >> $GITHUB_OUTPUT
            exit 0
          fi
        done

        echo "run_ml=false" >> $GITHUB_OUTPUT
```

**Fail-safe:** ALWAYS TRUE on main/develop (Quality Firewall)

---

### 3. Create requirements/ml-ci.in (REQUIRED)

```ini
# requirements/ml-ci.in
# CPU-optimized ML dependencies for CI (fast, deterministic)

# Strict pins for CI reproducibility
torch==2.10.0+cpu
torchvision==0.25.0+cpu

# Standard ML dependencies with range pins
-c constraints.txt  # Banned package enforcement
transformers>=4.57.0,<6
diffusers>=0.38.0,<1
sentence-transformers>=3.1.0,<6  # CVE-73169 minimum
controlnet-aux>=0.0.6,<1
huggingface-hub>=0.19.0,<2

# Skip heavy optional deps:
# - accelerate (not needed for CPU)
# - coremltools (Apple Silicon only)
# - rawpy (tested separately)
```

**Then compile:**
```bash
cd requirements && pip-compile ml-ci.in --output-file ml-ci.txt
```

---

### 4. Add --durations=20 to pytest (APPROVED)

**File:** `pyproject.toml`

```toml
[tool.pytest.ini_options]
addopts = "--strict-markers --tb=short -p no:warnings --durations=20"
```

**Benefit:** Surface slowest tests automatically for quarantine decisions

---

## CI Tier Structure (ADR-034)

| Tier | Trigger | Tests | Coverage | Timeout | Goal |
|------|---------|-------|----------|---------|------|
| **PR Default** | Every PR push | `not ml and not slow` | Yes | 20min | Fast feedback |
| **Main Push** | Push to main/develop | `not slow` (includes ML) | Yes | 45min | Quality Firewall |
| **Nightly** | Schedule | All (includes slow) | Yes | 120min | Comprehensive |
| **Benchmark** | Manual | `benchmark` only | No | 180min | Performance validation |

---

## Implementation Order (PHASED)

### Phase 1: Safety First (1 day) ✅
1. Remove `AGENT_TOOLSDIRECTORY` deletion
2. Audit for scikit-learn force-reinstall (remove if found)
3. Kill "fallback to not slow" retry logic

**Exit:** 3 consecutive green CI runs

---

### Phase 2: Infrastructure (2 days) ✅
4. Create `requirements/ml-ci.in` and compile
5. Update `pyproject.toml` (add `--durations=20`)
6. Test locally: `pytest -m "ml" --durations=20`

**Exit:** ML tests pass with new deps

---

### Phase 3: Conditional Execution (3 days) ⚠️
7. Add `run_ml_check` job (with fail-safe)
8. Update `test-ml` to depend on `run_ml_check`
9. Add `validate-ml-skip` audit job

**Exit:** Manual test of path-based gating

---

### Phase 4: Optimization (2 days) 🚀
10. Switch to `requirements/ml-ci.txt`
11. Disable coverage on PR tier (keep on main)
12. Quarantine slow tests (`@pytest.mark.slow`)

**Exit:** ML tests <15min on PR tier

---

### Phase 5: Documentation (1 day) 📝
13. Create ADR-034 (CI Test Execution Tiers)
14. Update CONTRIBUTING.md
15. Add performance monitoring

**Exit:** ADR merged, docs updated

---

## Quality Gates (PRESERVED)

✅ **Test Isolation (ADR-031):** Unchanged
✅ **Dependency Constraints (ADR-032):** Enhanced (validates ml-ci.in)
✅ **Coverage Threshold (25%):** Preserved (collected on main)
✅ **Security Scanning:** Unchanged
🆕 **ML Duration Budget:** 15min on PR, 25min on main

---

## Rollback Plan

### Triggers
- ML tests fail >10% (flake spike)
- Main CI blocked >2 hours
- False negative on ML skip (regression shipped)
- Performance degradation >20%

### Procedure
```bash
git revert <optimization-commit>
# OR emergency override:
# Set run_ml_check output to always TRUE
```

---

## Monitoring (30 days)

- **Week 1-2:** Daily audit of `run_ml=false` PRs (spot-check accuracy)
- **Week 3-4:** Weekly audit
- **Week 4:** Performance baseline comparison

**Metrics:**
- ML test duration (target: <15min PR, <25min main)
- Skip accuracy (target: <5% false negatives)
- Flake rate (target: <2%)

---

## Delegation

**Architect retains:**
- CI tier contracts (ADR-034)
- Fail-safe behavior
- Quality gate thresholds

**Specialist can implement:**
- Test quarantine (`@pytest.mark.slow`)
- Path matcher refinements
- Performance profiling

**Escalate if:**
- Regression shipped due to ML skip
- Targets not met after Phase 4
- False negative rate >5%

---

## Key Success Factors

1. ✅ Fix anti-patterns (toolcache deletion) FIRST
2. ✅ Implement fail-safe (always run on main) BEFORE optimization
3. ⚠️ Monitor skip decisions for 30 days
4. 📝 Document CI tiers in ADR-034

---

**Expected Outcome:**
- PR feedback: 30min → 15min (50% faster)
- Main quality gate: Unchanged
- Developer experience: Improved

**Timeline:** 9 working days with proper validation

---

**Approver:** Transformation Portal Architect
**Date:** 2026-02-18
**Next Step:** Phase 1 - Remove toolcache deletion
