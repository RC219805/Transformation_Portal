# PR Review Action Items: #793, #792, #790

**Date:** 2026-02-02
**Owner:** Transformation Portal Architect
**Priority:** P0 (Critical - Blocks Future Work)

---

## Immediate Actions (Today)

### 1. Close PR #790: Depth Pro Backend
**Status:** ❌ REJECT

**Actions:**
- [ ] Close PR with comment explaining rejection
- [ ] Link to full assessment: `PR_ARCHITECTURAL_ASSESSMENT_793_792_790.md`
- [ ] Link to ADR-018 for correct approach

**Comment Template:**
```markdown
## PR Rejected: Duplicates Merged PR #780

Thank you for this proposal. After architectural review, this PR is being rejected for the following reasons:

### Primary Issues
1. **Duplicates PR #780:** `DepthProStage` already implements Depth Pro integration (merged, 429 LOC, 22 tests)
2. **Violates ADR-018:** Skips validation phases and introduces unapproved architecture
3. **Over-Engineered:** 2,336 LOC for what should be simple wiring (~10 LOC)
4. **Failing Checks:** CI failures unresolved

### What Should Happen Instead

Per ADR-018 binding roadmap, PR2 (Wiring) should be minimal:

```python
# src/transformation_portal/stage_graph/graph.py
def select_depth_stage(config):
    backend = config.get("depth_backend", "depth_anything_v3")
    if backend == "depth_pro":
        return DepthProStage()  # Already exists from PR #780
    else:
        return DepthAnythingStage()
```

This simple wiring will be implemented separately, followed by 6-week validation (PR3).

### Next Steps
1. Close this PR
2. Implement correct minimal PR2 wiring
3. Complete ADR-018 validation phase
4. Reconsider unified architecture after production data (April 2026+)

**References:**
- Full Assessment: `docs/pr_archive/architecture/PR_ARCHITECTURAL_ASSESSMENT_793_792_790.md`
- ADR-018: `docs/architecture/ADR-018-depth-pro-integration.md`
- Governance: `docs/architecture/agent_governance.md`

**Architect Decision:** Final per governance precedence order.
```

---

### 2. Close PR #792: v3.0 Contracts
**Status:** ❌ REJECT

**Actions:**
- [ ] Close PR with comment explaining rejection
- [ ] Link to full assessment
- [ ] Set expectation for future reconsideration (April 2026)

**Comment Template:**
```markdown
## PR Rejected: Premature v3.0 Architecture

Thank you for this architectural proposal. After review, this PR is being rejected as premature.

### Primary Issues
1. **Too Early:** v3.0 introduced while v2.0.0 still in progress (target: Aug 2026)
2. **Skips Validation:** Depth Pro experimental period incomplete (6-week soak required)
3. **Violates ADR-018:** Phased approach mandates validation before architecture
4. **Breaking Changes:** Threatens v2.0.0 Golden Path stability promise
5. **Premature Abstraction:** Unifying 2 backends doesn't justify protocol overhead (wait for N≥5)

### Why Not Now?

**Insufficient Production Data:**
- Depth Pro is experimental (not validated in production)
- No benchmark data vs DA3
- Unknown if metric depth provides real value
- Unknown if performance acceptable

**Governance Violation:**
- ADR-018 is binding, prescribes phased approach
- No superseding ADR for v3.0 leap
- Silence is not approval (explicit Architect sign-off required)

**Semantic Mismatch:**
- DA3: relative depth (0-1 normalized)
- Depth Pro: metric depth (meters, absolute scale)
- Unified contract masks critical difference or forces compromise

### When to Reconsider

**April 2026+ (After Validation Complete)**

Required before v3.0 architecture:
- ✅ Complete ADR-018 PR2 (wiring) and PR3 (validation)
- ✅ 6-week experimental soak period
- ✅ Production benchmarks show Depth Pro value
- ✅ Write superseding ADR with migration plan
- ✅ Coordinate with v2.0.0 finalization (Aug 2026)

### Next Steps
1. Close this PR (defer, not reject forever)
2. Complete ADR-018 phased approach
3. Gather production data (6 weeks)
4. Design v3.0 informed by real-world needs (April 2026+)

**References:**
- Full Assessment: `docs/pr_archive/architecture/PR_ARCHITECTURAL_ASSESSMENT_793_792_790.md`
- ADR-018: `docs/architecture/ADR-018-depth-pro-integration.md`
- Executive Summary: `docs/architecture/PR_REVIEW_EXECUTIVE_SUMMARY.md`

**Architect Decision:** Defer until validation complete (April 2026).
```

---

### 3. Hold PR #793: Dependency Updates
**Status:** ⏸️ HOLD

**Actions:**
- [ ] Comment on PR requesting security review
- [ ] Request `safety-report.json` analysis
- [ ] Block merge until #792/#790 closed

**Comment Template:**
```markdown
## PR On Hold: Awaiting Security Review and Isolation

Thank you for the automated dependency updates. This PR is on hold pending:

### Required Reviews
1. **Security Analysis:** Review `safety-report.json` for CRITICAL/HIGH vulnerabilities
2. **Breaking Changes:** Identify major version bumps in core dependencies
3. **Compatibility Testing:** Validate on Python 3.10, 3.11, 3.12

### Isolation Requirement

**BLOCKER:** PRs #792 and #790 are being rejected due to architectural concerns.

**This PR must merge in isolation** to avoid integration chaos:
- Dependency updates + architectural changes = impossible to debug
- Must merge alone with 48-hour stability window
- No other PRs merged during soak period

### Timeline

**After #792/#790 Closed:**
1. Complete security review (check safety-report.json)
2. Identify breaking changes (numpy, torch, transformers, pillow, opencv)
3. Test full CI matrix (core + ML tests, all Python versions)
4. Merge alone (no other PRs)
5. 48-hour soak period (monitor for regressions)

**Estimated Merge:** +1 week after #792/#790 resolution.

### High-Risk Dependencies to Check

```bash
# Compare versions
diff <(git show main:requirements/all.txt) requirements/all.txt | grep -E "numpy|torch|transformers|pillow|opencv"
```

**Questions:**
- Any CRITICAL/HIGH CVEs?
- Any major version bumps?
- Any known incompatibilities?

**References:**
- Full Assessment: `docs/pr_archive/architecture/PR_ARCHITECTURAL_ASSESSMENT_793_792_790.md`

**Status:** Hold until #792/#790 resolved, then review and merge in isolation.
```

---

## Short-Term Actions (This Week)

### 4. Implement ADR-018 PR2 (Correct Wiring)
**Status:** 🔄 TODO

**Description:** Simple feature flag wiring per ADR-018.

**Steps:**
- [ ] Create new branch: `feat/depth-pro-wiring-pr2`
- [ ] Implement minimal wiring (~10 LOC)
- [ ] Add integration test (mocked, no checkpoint)
- [ ] Update ADR-018 (mark PR2 complete)
- [ ] Submit PR for review
- [ ] Merge after approval

**Implementation:**
```python
# File: src/transformation_portal/stage_graph/graph.py
# Location: In stage selection logic

def select_depth_stage(config: dict) -> Stage:
    """
    Select depth estimation stage based on configuration.

    Supports:
    - depth_anything_v3 (default, production)
    - depth_pro (experimental, requires opt-in)

    See ADR-018 for Depth Pro integration roadmap.
    """
    backend = config.get("depth_backend", "depth_anything_v3")

    if backend == "depth_pro":
        from .stages.depth_pro import DepthProStage
        return DepthProStage()
    elif backend == "depth_anything_v3":
        from .stages.depth import DepthAnythingStage
        return DepthAnythingStage()
    else:
        raise ValueError(
            f"Unknown depth backend: {backend}. "
            f"Supported: depth_anything_v3 (default), depth_pro (experimental)"
        )
```

**Test:**
```python
# File: tests/integration/test_depth_backend_selection.py

def test_depth_backend_default():
    """Default backend is depth_anything_v3."""
    config = {}
    stage = select_depth_stage(config)
    assert isinstance(stage, DepthAnythingStage)

def test_depth_backend_depth_pro():
    """Depth Pro backend selectable via config."""
    config = {"depth_backend": "depth_pro"}
    stage = select_depth_stage(config)
    assert isinstance(stage, DepthProStage)

def test_depth_backend_invalid():
    """Unknown backend raises ValueError."""
    config = {"depth_backend": "invalid"}
    with pytest.raises(ValueError, match="Unknown depth backend"):
        select_depth_stage(config)
```

**Effort:** 2-3 hours (implementation + tests + review).

---

### 5. Update ADR-018 Status
**Status:** 🔄 TODO

**Actions:**
- [ ] Mark PR1 complete (DepthProStage merged)
- [ ] Mark PR2 in progress (wiring implementation)
- [ ] Document PR #790 rejection rationale
- [ ] Clarify PR3 timeline (6-week validation)

**Diff:**
```diff
# docs/architecture/ADR-018-depth-pro-integration.md

 | Phase | PR | Scope | Status |
 |-------|-----|-------|--------|
-| **PR1: Stage** | #780 | Add `DepthProStage` class | ✅ Merged |
-| **PR2: Wiring** | TBD | Wire stage into preset loader | Planned |
-| **PR3: Validation** | TBD | Integration tests, benchmarks | Planned |
+| **PR1: Stage** | #780 | Add `DepthProStage` class | ✅ Merged (2026-02-02) |
+| **PR2: Wiring** | TBD | Wire stage into preset loader | 🔄 In Progress |
+| **PR3: Validation** | TBD | Integration tests, benchmarks | Planned (March 2026) |
+
+**Note:** PR #790 was rejected as over-engineered duplicate of PR1. PR2 implemented as minimal wiring instead.
```

---

### 6. Add Governance CI Gate
**Status:** 🔄 TODO

**Description:** Prevent architectural changes without ADR updates.

**Implementation:**
```yaml
# File: .github/workflows/build.yml
# Location: Add new job before 'test'

  adr-compliance:
    name: ADR Compliance Check
    runs-on: ubuntu-24.04
    timeout-minutes: 5

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Need full history for git diff

      - name: Check for architectural changes
        run: |
          # Files that indicate architectural changes
          ARCH_FILES=$(git diff --name-only origin/main... | grep -E \
            "stage_graph/.*\.py|contracts/|protocols/|lux_depth_v3/.*\.py" || true)

          if [ -n "$ARCH_FILES" ]; then
            echo "::notice::Architectural changes detected in:"
            echo "$ARCH_FILES"

            # Check for corresponding ADR updates
            ADR_FILES=$(git diff --name-only origin/main... | grep -E \
              "docs/architecture/ADR-.*\.md" || true)

            if [ -z "$ADR_FILES" ]; then
              echo "::error::Architectural changes require ADR update"
              echo "Changed files:"
              echo "$ARCH_FILES"
              echo ""
              echo "Please:"
              echo "1. Update existing ADR if modifying approved architecture"
              echo "2. Create new ADR if introducing new architecture"
              echo "3. Get Architect approval before merging"
              exit 1
            else
              echo "::notice::ADR updated:"
              echo "$ADR_FILES"
            fi
          fi
```

**Effort:** 1 hour.

---

## Medium-Term Actions (Next 6 Weeks)

### 7. ADR-018 PR3: Validation Phase
**Status:** ⏳ PLANNED

**Timeline:** March - April 2026

**Tasks:**
- [ ] Integration tests with real Depth Pro checkpoint (excluded from CI)
- [ ] Benchmark Depth Pro vs DA3 (inference time, quality)
- [ ] User testing (architectural visualization use cases)
- [ ] Production monitoring (6-week soak period)
- [ ] Bug tracking (experimental tier issues)

**Success Criteria:**
- No critical bugs during soak period
- Inference time ≤ 2x DA3 for equivalent resolution
- Visual quality acceptable for 10+ scenes
- User feedback positive

**Deliverables:**
- Validation report (benchmarks, bugs, feedback)
- Decision document (promote / keep experimental / deprecate)
- If promoting: ADR update with tier change

---

### 8. Merge PR #793 (After Resolution)
**Status:** ⏳ BLOCKED

**Blocked By:** PRs #792/#790 closure

**When Unblocked:**
- [ ] Review `safety-report.json` (CRITICAL/HIGH CVEs)
- [ ] Identify breaking changes (major version bumps)
- [ ] Test on Python 3.10, 3.11, 3.12
- [ ] Merge in isolation (no other PRs)
- [ ] 48-hour soak period
- [ ] Monitor for regressions

**Timeline:** +1 week after #792/#790 closed.

---

## Long-Term Actions (April 2026+)

### 9. v3.0 Architecture Decision
**Status:** ⏳ DEFERRED

**Prerequisites:**
- ✅ ADR-018 PR3 validation complete
- ✅ 6-week Depth Pro soak period
- ✅ Production data gathered
- ✅ v2.0.0 stability confirmed

**Decision Points:**

**A. Depth Pro Tier Promotion**
- Promote to canary (add to optional extras)
- Keep experimental (continue feature flag)
- Deprecate (remove if no value)

**B. Architecture Path**
- If promoted: consider v3.0 unified contracts
- If experimental: continue dual-track
- If deprecated: remove and simplify

**C. ADR Process**
- Write superseding ADR if promoting
- Document migration plan from v2.0.0
- Coordinate with v2.0.0 finalization (Aug 2026)

**Timeline:** April 2026 earliest.

---

## Success Metrics

### Immediate (This Week)
- [ ] PRs #790 and #792 closed with explanation
- [ ] PR #793 on hold with clear requirements
- [ ] ADR-018 PR2 implemented and merged
- [ ] Governance CI gate added

### Short-Term (1 Month)
- [ ] PR #793 merged in isolation
- [ ] 48-hour soak period clean (no regressions)
- [ ] Depth Pro wiring functional
- [ ] Integration tests passing

### Medium-Term (6 Weeks)
- [ ] ADR-018 PR3 validation complete
- [ ] Production data collected
- [ ] Tier decision made (promote/keep/deprecate)
- [ ] v2.0.0 Golden Path stable

### Long-Term (6 Months)
- [ ] v2.0.0 finalized (Aug 2026)
- [ ] Depth Pro tier stable (canary or deprecated)
- [ ] v3.0 architecture decision informed by data
- [ ] Governance process strengthened

---

## Communication Checklist

### Immediate
- [ ] Comment on PR #790 (reject with explanation)
- [ ] Comment on PR #792 (reject with explanation)
- [ ] Comment on PR #793 (hold with requirements)
- [ ] Update project board (if exists)

### Short-Term
- [ ] Notify maintainers of decisions
- [ ] Update documentation index
- [ ] Post ADR-018 PR2 implementation PR

### Medium-Term
- [ ] Monthly update on Depth Pro validation
- [ ] Report bugs/issues in experimental tier
- [ ] Community feedback collection (if applicable)

---

## Escalation Protocol

**If Disagreement with Decisions:**
1. Review `docs/architecture/agent_governance.md`
2. Architect has final authority on:
   - Security and dependency governance
   - CI/CD policy
   - Cross-module contracts
   - Public API stability
3. Escalation path: Repository owner (if governance violated)

**Silence is not approval.**
Explicit Architect sign-off required for architectural changes.

---

## Document History

- **2026-02-02:** Created after comprehensive PR review
- **Next Review:** After ADR-018 PR2 merged

**Owner:** Transformation Portal Architect
**Status:** ACTIVE
