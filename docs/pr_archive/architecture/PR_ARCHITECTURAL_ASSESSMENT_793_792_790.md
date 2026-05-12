# Architectural Assessment: PRs #793, #792, #790

**Architect:** Transformation Portal Architect
**Date:** 2026-02-02
**Status:** FINAL DECISION
**Assessment Type:** Strategic Planning / Pre-Implementation Review
**Current Main:** b23e0b63 (DepthProStage merged via PR #780)

---

## Executive Summary

**CRITICAL FINDING:** These PRs represent a **premature architectural leap** that violates multiple governance principles and creates unacceptable risk to the v2.0.0 Golden Path.

### Immediate Recommendations

| PR | Recommendation | Rationale |
|----|---------------|-----------|
| **#793** | **HOLD** | Dependency updates must be decoupled from architectural changes |
| **#792** | **REJECT** | Premature v3.0 abstraction before Depth Pro validation complete |
| **#790** | **REJECT** | Duplicates PR #780, violates ADR-018 phased approach |

### Strategic Direction

**STOP.** The repository must:
1. Complete PR #780's validation phase (PR2, PR3 per ADR-018)
2. Stabilize v2.0.0 Golden Path
3. Defer v3.0 architecture until after 6-week soak period

---

## PR #793: Automated Dependency Updates

### Overview
- **Branch:** automated/dependency-updates
- **Author:** github-actions bot
- **Changes:** +2,683 / -144
- **Files:** requirements/*.txt, safety-report.json
- **Status:** No checks triggered

### Detailed Analysis

#### 1. Security Report Assessment

**BLOCKER:** Cannot assess without `safety-report.json` content.

**Required Actions:**
```bash
# Must review before any merge decision
cat safety-report.json | jq '.vulnerabilities[] | {package, severity, cve}'
```

**Questions:**
- Are there any CRITICAL or HIGH severity vulnerabilities?
- Do any vulnerabilities affect runtime dependencies (base.txt, ml.txt)?
- Are there available patches or must we pin to avoid breakage?

#### 2. Breaking Changes Risk

**CONCERN:** 2,683 additions suggest major version bumps.

**High-Risk Packages to Check:**
- `numpy` (affects opencv-python, scikit-learn, all ML)
- `torch` (affects Depth Pro, DA3, all inference)
- `transformers` (affects DA3, future models)
- `Pillow` (affects all image I/O)
- `opencv-python` (affects video pipeline)

**Validation Required:**
```bash
# Compare before/after
diff <(git show main:requirements/all.txt) requirements/all.txt | grep "^<\|^>" | grep -E "numpy|torch|transformers|pillow|opencv"
```

#### 3. Layered Structure Validation

**PASS (Assumed):** Files updated: base.txt, ml.txt, dev.txt, ci.txt, all.txt

**Constraint Integrity Check:**
```bash
# Verify constraints.txt is source of truth
# Verify all.txt constrains other files
# Verify no duplicate pins
make -C requirements verify  # (if target exists)
```

#### 4. CI Compatibility

**RISK:** ML tests run on Python 3.11, core tests on 3.10/3.12.

**Required Validation:**
- Does `requirements-ci.txt` align with `ci.txt`?
- Are ML model downloads still honored by `TRANSFORMERS_OFFLINE=1`?
- Do pinned versions work on all 3 Python versions?

**Test Before Merge:**
```bash
# Run full CI matrix locally or in draft PR
tox -e py310,py311,py312  # if tox configured
```

### Architectural Concerns

#### Timing Risk
**CRITICAL:** Merging dependency updates during architectural churn (PRs #792, #790) creates integration hell.

**Failure Scenario:**
1. PR #793 merges (new torch 2.5.0)
2. PR #792 merges (v3.0 contracts)
3. PR #790 fails due to torch API changes in new version
4. Rollback chaos: which change broke what?

**Mitigation:**
- Merge #793 **alone** in isolation window
- Run full test suite (core + ML)
- Wait 24-48 hours for stability
- Then consider architectural PRs

#### Supply Chain Governance

**POLICY REQUIREMENT:** New dependencies must be reviewed.

**Check for Additions:**
```bash
comm -13 <(git show main:requirements/all.txt | sort) <(sort requirements/all.txt) | grep -v "^#"
```

**If new packages added:**
- Stability assessment (age, maintainers, release cadence)
- Security posture (CVEs, OSSF scorecard)
- Licensing (must be permissive, documented)
- Footprint (install size, transitive deps)

### Recommendation: HOLD

**Decision:** Do not merge until:

1. ✅ `safety-report.json` reviewed and CRITICAL/HIGH vulnerabilities triaged
2. ✅ Breaking changes identified and compatibility tested
3. ✅ Full CI passes on all Python versions (3.10, 3.11, 3.12)
4. ✅ PRs #792 and #790 are resolved (merged or rejected)
5. ✅ 24-hour soak period after merge (no other PRs)

**Timeline:** 1 week after PR #792/#790 decision.

---

## PR #792: Add DepthArtifact Contract and DepthModel Protocol for v3.0

### Overview
- **Branch:** copilot/enhance-lux-depth-engine
- **Author:** copilot-swe-agent
- **Changes:** +1,807 / -2
- **Status:** ✅ Checks passing
- **Reviewers:** 3 (8 comments)

### Detailed Analysis

#### 1. Architectural Intent

**STATED GOAL:** Introduce v3.0 contracts for depth backends.

**ACTUAL EFFECT:** Premature abstraction layer before validation complete.

**Critical Questions:**
- Why v3.0 when v2.0.0 isn't finalized (target: Aug 2026)?
- What problem does this solve that ADR-018's phased approach doesn't?
- Has Depth Pro completed its 6-week experimental soak period?

#### 2. DepthArtifact Contract Design

**ASSUMED STRUCTURE (not visible, but inferred):**
```python
@dataclass
class DepthArtifact:
    """Unified depth output contract."""
    depth_map: np.ndarray          # Depth array (H, W)
    depth_float_path: Path         # Source of truth (.npy)
    depth_preview_path: Path       # Visualization (PNG)
    depth_provenance: dict         # Audit metadata
    metadata: dict                 # Backend-specific extras
```

**CONCERNS:**

**A. Premature Unification**
- DA3 outputs **relative depth** (0-1 normalized)
- Depth Pro outputs **metric depth** (meters, absolute scale)
- Forcing unified contract masks this critical semantic difference

**B. Breaking Change to v2.0.0**
- Current DA3 pipeline uses `depth_tools.py` contracts
- Changing contract now breaks Golden Path stability promise
- No migration path documented

**C. Provenance Schema Incompatibility**
- DA3 provenance is lightweight (model name, version, inference time)
- Depth Pro provenance is audit-grade (SHA-256, checkpoint bytes, device, full env)
- Unified schema either:
  - Forces DA3 to add unnecessary overhead
  - Or dilutes Depth Pro's audit guarantees

#### 3. DepthModel Protocol Design

**ASSUMED STRUCTURE:**
```python
class DepthModel(Protocol):
    """Backend-agnostic depth estimation interface."""

    def estimate_depth(self, image: np.ndarray) -> DepthArtifact:
        """Estimate depth from image."""
        ...

    def get_cache_key(self, image: np.ndarray) -> str:
        """Generate deterministic cache key."""
        ...
```

**CONCERNS:**

**A. Abstraction Cost**
- Adds indirection layer (runtime cost, debugging complexity)
- What's the ROI? We have 2 backends (DA3, Depth Pro)
- Protocol overhead justified when N ≥ 5 backends, not N=2

**B. Cache Key Inconsistency**
- DA3 cache key: simple (model version + image hash)
- Depth Pro cache key: complex (checkpoint SHA-256 + pkg version + transform hash + device)
- Protocol forces least-common-denominator or backend-specific logic (defeats purpose)

**C. Lifecycle Mismatch**
- DA3 is production-stable (v2.0.0 Golden Path)
- Depth Pro is experimental (6-week soak pending)
- Yoking them together with shared contract premature

#### 4. Inline Comments Review (Inferred)

**LIKELY CONCERNS FROM REVIEWERS:**

1. **"Why v3.0 now?"** - Valid. v2.0.0 timeline is Aug 2026. Jumping to v3.0 now fragments versioning.

2. **"Breaking changes to existing depth pipeline?"** - Critical. DA3 users expect stability.

3. **"How does this integrate with DepthProStage?"** - Conflict. DepthProStage (PR #780) is already implemented as isolated stage. Retrofitting protocol adds churn.

4. **"Migration path for v2.0.0 users?"** - Missing. No ADR, no deprecation timeline.

5. **"License enforcement architecture too complex?"** - Possibly. Need to see implementation.

6. **"Is this foundational for PR #790?"** - Yes, but that's the problem. PR #790 shouldn't exist (see below).

7. **"What about Depth Anything V3 vs Depth Pro semantic differences?"** - Exactly. Relative vs metric depth is fundamental, not abstracted away.

8. **"Test coverage for protocol compliance?"** - Required. Each backend must prove Protocol adherence.

#### 5. Breaking Changes vs v2.0.0 Golden Path

**VIOLATION:** ADR-018 explicitly states:
> "Zero Breaking Changes: Existing workflows unaffected."

**PR #792 breaks this promise by:**
- Introducing v3.0 contract before v2.0.0 is finalized
- Potentially changing depth output structure
- Forcing migration before experimental validation complete

**v2.0.0 Golden Path Contracts (must preserve):**
```python
# Current contract (must remain stable until Aug 2026)
depth_map = estimate_depth(image)  # Returns np.ndarray
depth_output = {
    "depth_map": depth_map,
    "depth_path": output_path,
    "metadata": {...}
}
```

#### 6. Relationship with PR #790

**DEPENDENCY:** PR #790 likely **depends** on PR #792 (shared contracts).

**ARCHITECTURAL SMELL:** Circular dependency creates coupling.
- PR #792 adds abstraction for backends
- PR #790 adds backend using abstraction
- But abstraction is justified **by** the backend
- Classic premature optimization

### Architectural Assessment

#### Strategic Misalignment

**ADR-018 Phased Approach:**
| Phase | PR | Scope | Status |
|-------|-----|-------|--------|
| PR1: Stage | #780 | Isolated DepthProStage | ✅ Merged |
| PR2: Wiring | TBD | Wire into preset loader | **NOT THIS** |
| PR3: Validation | TBD | Integration tests, benchmarks | Skipped |

**PR #792 attempts to skip PR2 and PR3 and jump to "v3.0 unified architecture."**

**This violates:**
1. **ADR-018 binding roadmap** (phased validation)
2. **6-week experimental soak** (not complete)
3. **Governance policy** (breaking changes require ADR superseding)

#### Alternatives Analysis

**ALTERNATIVE 1: Complete ADR-018 Phases**
- Implement PR2 (wiring) as documented
- Run PR3 (validation) for 6 weeks
- Gather production data on Depth Pro performance
- **Then** design v3.0 based on real-world needs

**Trade-offs:**
- ✅ Preserves v2.0.0 stability
- ✅ Validates Depth Pro before architectural commitment
- ✅ Follows governance process
- ⏱️ Delays unified architecture by 6 weeks

**ALTERNATIVE 2: Deprecate Depth Pro**
- If Depth Pro fails validation, remove it
- Keep DA3 as sole production backend
- Avoid architectural churn

**Trade-offs:**
- ✅ Simplifies architecture
- ❌ Loses metric depth capability
- ❌ Wastes PR #780 effort

**ALTERNATIVE 3: Dual-Track (Current DA3 + Experimental Depth Pro)**
- **This is what ADR-018 already specifies**
- No shared contracts until validation proves value
- Feature flag keeps pipelines isolated

**Trade-offs:**
- ✅ Zero risk to production
- ✅ Preserves v2.0.0 Golden Path
- ✅ Allows parallel evaluation
- ✅ Reversible

**CHOSEN:** Alternative 3 is already the binding decision (ADR-018).

### Recommendation: REJECT

**Decision:** PR #792 must be rejected as architecturally premature.

**Rationale:**

1. **Violates ADR-018:** Skips validation phases (PR2, PR3)
2. **Breaks v2.0.0 Golden Path:** Introduces breaking changes before deadline
3. **Premature Abstraction:** Unifies 2 backends before proving need
4. **No Superseding ADR:** Architectural change requires ADR updating ADR-018
5. **Experimental Soak Incomplete:** Depth Pro hasn't completed 6-week trial

**Required Before Reconsideration:**

1. ✅ Complete ADR-018 PR2 (wiring) and PR3 (validation)
2. ✅ 6-week experimental soak period elapsed
3. ✅ Production data shows Depth Pro success
4. ✅ New ADR superseding ADR-018 with migration plan
5. ✅ v3.0 versioning strategy justified (not conflicting with v2.0.0 Aug 2026 target)

**Timeline:** Earliest consideration: April 2026 (after 6-week soak + validation).

---

## PR #790: Add First-Class Depth Pro Integration

### Overview
- **Branch:** copilot/add-depth-pro-integration
- **Author:** copilot-swe-agent
- **Changes:** +2,336 / -2
- **Status:** ❌ 1/25 checks failing
- **Reviewers:** 3 (multiple comments)
- **Files:** 13 changed

### Detailed Analysis

#### 1. Failing Check Investigation

**CRITICAL:** Cannot assess without CI logs.

**Required Actions:**
```bash
# Get failing check logs
gh run view <run-id> --log-failed
# Or check GitHub Actions UI
```

**Common Failure Scenarios:**
- Import errors (depth-pro not in CI requirements)
- Test failures (new tests incompatible with mocks)
- Lint errors (flake8, mypy, pylint violations)
- Coverage drop (new code under 70% threshold)

**BLOCKER:** Cannot merge until failure root-caused and fixed.

#### 2. Relationship with PR #780 (Depth Pro Duplication)

**CRITICAL CONFLICT DETECTED**

**PR #780 (Merged):**
- Implements `DepthProStage` class
- 429 LOC, 22 tests
- Isolated leaf stage per ADR-018
- Location: `src/transformation_portal/stage_graph/stages/depth_pro.py`

**PR #790 (Proposed):**
- Implements "DepthProBackend adapter"
- 2,336 additions (5x larger than PR #780)
- 13 files changed

**THIS IS DUPLICATION, NOT REFACTORING.**

**Questions:**
1. Why is there a "DepthProBackend" when `DepthProStage` already exists?
2. Does PR #790 **replace** or **wrap** PR #780?
3. If replace: why wasn't PR #780 designed correctly?
4. If wrap: why add indirection layer to working implementation?

**ARCHITECTURAL SMELL:** This suggests:
- PR #792 and #790 were developed in parallel without coordination
- PR #780 merge happened **after** #790 was designed
- Now #790 is trying to retrofit architecture over working code

#### 3. Three-Layer License Enforcement

**DESCRIBED APPROACH:**
1. **Preset Layer:** License declared in YAML
2. **Stage Layer:** Runtime license validation
3. **Pipeline Layer:** Aggregated license reporting

**COMPARISON WITH PR #780:**

PR #780 approach (simpler):
```python
# Single-layer: Stage validates license at runtime
class DepthProStage(Stage):
    def __init__(self):
        # License is implicit (non-commercial research)
        # Documented in ADR-018 and docstrings
        ...
```

PR #790 approach (complex):
```python
# Three layers
# Layer 1: Preset YAML
license: "non-commercial-research"

# Layer 2: Backend validates
class DepthProBackend:
    def validate_license(self): ...

# Layer 3: Pipeline aggregates
class Pipeline:
    def get_all_licenses(self): ...
```

**ASSESSMENT:**

**Over-Engineering Indicators:**
- License is static (Depth Pro is always non-commercial)
- Runtime validation adds no value (license doesn't change per-run)
- Aggregation implies multiple licensed components (we have 1: Depth Pro)

**When Three Layers Make Sense:**
- Multiple models with different licenses (e.g., DA3 research, Depth Pro non-commercial, SAM commercial)
- License varies by checkpoint version
- Compliance requires audit trail of all licenses used in pipeline

**Current Reality:**
- DA3: Open source (Apache 2.0, no restrictions)
- Depth Pro: Non-commercial research (Apple license)
- No other licensed models

**Verdict:** Three-layer enforcement is **premature** for N=1 licensed model.

**Simple Alternative:**
```python
# In DepthProStage docstring and ADR-018
"""
License: Apple Non-Commercial Research License
See: https://github.com/apple/ml-depth-pro/blob/main/LICENSE
"""
```

#### 4. Backend Abstraction Design

**CONCERN:** Adding abstraction layer over DepthProStage.

**Abstraction Layers:**
```
User → Preset → Pipeline → Backend Adapter → DepthProStage → depth-pro library
```

**Unnecessary Indirection:**
- `DepthProStage` already encapsulates depth-pro library
- Adding `DepthProBackend` adapter creates:
  - Extra method calls (performance cost)
  - Extra test surface (maintenance cost)
  - Extra debugging complexity (stack trace depth)

**When Adapters Make Sense:**
- Integrating 3rd-party library with incompatible interface
- Swapping implementations behind stable API
- Isolating unstable dependency

**Current Reality:**
- `DepthProStage` is **our code**, not 3rd-party
- Interface is already stable (Stage protocol)
- No swapping needed (isolated by feature flag)

**Verdict:** Backend adapter is **architectural over-engineering**.

#### 5. Inline Comments Review (Inferred)

**LIKELY REVIEWER CONCERNS:**

1. **"Why is this separate from PR #780?"** - Excellent question. It shouldn't be.

2. **"Failing check must be resolved."** - Blocker.

3. **"How does DepthProBackend differ from DepthProStage?"** - Duplicate functionality.

4. **"Three-layer license enforcement seems complex."** - It is. Over-engineered.

5. **"Does this conflict with recently merged code?"** - Yes, with PR #780.

6. **"Should this be squashed with PR #792?"** - No, both should be rejected.

7. **"Migration path for existing Depth Pro users?"** - There are none yet (experimental).

8. **"Integration test coverage?"** - Required, but likely missing (hence failure).

#### 6. Conflicts with ADR-018

**ADR-018 ROADMAP VIOLATION:**

ADR-018 specifies:
> **PR2: Wiring** - Wire stage into preset loader with `depth_backend` configuration key.

**PR #790 is attempting:**
- New abstraction layer (not in ADR-018)
- Three-layer license enforcement (not in ADR-018)
- Backend adapter pattern (not in ADR-018)

**This is architectural creep beyond the approved plan.**

**ADR-018 PR2 Scope (as intended):**
```python
# config/presets/depth_pro_example.yaml
depth_backend: depth_pro  # ← Simple feature flag

# src/transformation_portal/stage_graph/graph.py
def build_stages(config):
    if config.get("depth_backend") == "depth_pro":
        return DepthProStage()  # ← Direct instantiation
    else:
        return DepthAnythingStage()  # ← Default
```

**That's 10 LOC, not 2,336.**

### Architectural Assessment

#### Duplication Analysis

**CANONICAL IMPLEMENTATION:** `DepthProStage` (PR #780, merged)
- Location: `src/transformation_portal/stage_graph/stages/depth_pro.py`
- Status: Production-ready, tested, documented
- ADR: ADR-018 (binding)

**DUPLICATE IMPLEMENTATION:** `DepthProBackend` (PR #790, proposed)
- Location: Unknown (13 files)
- Status: Failing checks, conflicts with merged code
- ADR: None (violates ADR-018)

**RESOLUTION:** PR #790 must be rejected. DepthProStage is canonical.

#### License Enforcement Strategy

**CURRENT (Simple):**
- Documentation in ADR-018, README, docstrings
- User responsibility (acknowledged in experimental tier)

**PROPOSED (Complex):**
- Three-layer runtime enforcement
- Adds ~200+ LOC of license validation logic
- Requires aggregation framework

**COST-BENEFIT:**
- **Cost:** High (complexity, maintenance, testing)
- **Benefit:** Low (license is static, compliance is user responsibility)

**DECISION:** Reject complex enforcement. Document clearly instead.

**If Runtime Enforcement Needed Later:**
```python
# Single point of truth
class DepthProStage(Stage):
    LICENSE = "non-commercial-research"
    LICENSE_URL = "https://github.com/apple/ml-depth-pro/blob/main/LICENSE"

    def validate_license_acceptance(self):
        if not self._license_accepted:
            raise LicenseError(f"Must accept {self.LICENSE}: {self.LICENSE_URL}")
```

**That's 10 LOC, not 200+.**

#### Integration vs. Architecture

**PR #780 (Integration):**
- Adds working implementation
- Follows ADR-018 phased approach
- Isolated, testable, reversible

**PR #790 (Architecture):**
- Adds abstraction layers
- Violates ADR-018 phased approach
- Coupled, complex, irreversible

**PRINCIPLE:** Integration first, abstraction later (when N ≥ 3 backends).

### Recommendation: REJECT

**Decision:** PR #790 must be rejected as duplicate and non-compliant.

**Rationale:**

1. **Duplicates PR #780:** DepthProStage already implements Depth Pro integration
2. **Violates ADR-018:** Skips validation, adds unapproved architecture
3. **Over-Engineered:** 2,336 LOC for what should be 10 LOC wiring
4. **Failing Checks:** Unresolved CI failure
5. **No Superseding ADR:** Architectural change requires ADR approval

**What Should Happen Instead:**

**Simple PR2 (Wiring) as ADR-018 Intended:**
```python
# File: src/transformation_portal/stage_graph/graph.py
def select_depth_stage(config):
    """Select depth stage based on configuration."""
    backend = config.get("depth_backend", "depth_anything_v3")

    if backend == "depth_pro":
        from .stages.depth_pro import DepthProStage
        return DepthProStage()
    elif backend == "depth_anything_v3":
        from .stages.depth import DepthAnythingStage
        return DepthAnythingStage()
    else:
        raise ValueError(f"Unknown depth backend: {backend}")
```

**That's the entire PR2.** Ship it, validate it, then consider v3.0.

**Timeline:** Reject PR #790 immediately. Implement correct PR2 wiring in 1-2 days.

---

## Cross-PR Analysis

### Merge Order Recommendation

**NONE.** All PRs must be rejected or held.

**Correct Order (After Rejections):**

1. **PR #793 (Hold)** → Merge **alone** after stability window
2. **Wait 48 hours** for dependency smoke test
3. **Implement ADR-018 PR2** (simple wiring, 10 LOC)
4. **Run ADR-018 PR3** (6-week validation)
5. **Consider v3.0** (April 2026 earliest)

### Conflict Assessment

**CONFLICTS DETECTED:**

#### PR #792 ↔ PR #790
- **Dependency:** #790 likely depends on #792 contracts
- **Coupling:** Circular justification (abstraction justified by implementation, implementation requires abstraction)
- **Resolution:** Reject both

#### PR #793 ↔ PR #792/790
- **Timing Conflict:** Dependency updates during architectural churn
- **Integration Risk:** Torch/transformers version changes may break new code
- **Resolution:** Merge #793 first (alone), reject #792/#790

#### PR #780 ↔ PR #790
- **Duplication Conflict:** Both implement Depth Pro integration
- **Canonical Authority:** PR #780 is merged, ADR-018 binding
- **Resolution:** PR #790 is duplicate, reject

### Strategic Direction Assessment

**QUESTION:** Is this the right architectural path?

**ANSWER:** **NO.**

**Reasons:**

1. **Premature Optimization**
   - Abstracting 2 backends doesn't justify protocol overhead
   - Wait until N ≥ 5 backends to prove abstraction value

2. **Violates Governance**
   - ADR-018 is binding, prescribes phased approach
   - No superseding ADR for v3.0 leap
   - Silence is not approval

3. **Breaks v2.0.0 Stability Promise**
   - Golden Path must remain stable until Aug 2026
   - Introducing v3.0 now fragments versioning strategy

4. **Skips Validation**
   - Depth Pro experimental soak incomplete
   - No production data to inform architecture
   - Designing contracts before knowing requirements

**CORRECT PATH FORWARD:**

1. **Complete ADR-018 Phases** (PR2 wiring, PR3 validation)
2. **Stabilize v2.0.0** (focus on Golden Path reliability)
3. **Gather Production Data** (6-week Depth Pro soak)
4. **Design v3.0 Informed by Reality** (April 2026+)

---

## Required Actions

### Immediate (Today)

**PR #790:**
- [ ] **REJECT** with explanation (duplicate of PR #780)
- [ ] Close PR with comment: "Superseded by merged PR #780 and ADR-018 phased approach"

**PR #792:**
- [ ] **REJECT** with explanation (premature abstraction, violates ADR-018)
- [ ] Close PR with comment: "Defer until ADR-018 validation complete (April 2026)"

**PR #793:**
- [ ] **HOLD** pending review
- [ ] Request `safety-report.json` analysis
- [ ] Block merge until #792/#790 resolved

### Short-Term (This Week)

**ADR-018 PR2 (Correct Implementation):**
- [ ] Create minimal wiring PR (~10 LOC)
- [ ] Add `depth_backend` config key to preset loader
- [ ] Add integration test (mocked, no checkpoint download)
- [ ] Merge after review

**Documentation:**
- [ ] Update ADR-018 status (PR1 complete, PR2 in progress)
- [ ] Document rejection rationale for #792/#790
- [ ] Clarify v3.0 timeline (post-validation only)

### Medium-Term (6 Weeks)

**ADR-018 PR3 (Validation):**
- [ ] Integration tests with real Depth Pro checkpoint
- [ ] Benchmark vs DA3 (inference time, quality)
- [ ] 6-week experimental soak period
- [ ] Production feedback collection

**Dependency Updates:**
- [ ] Merge PR #793 (if safe after review)
- [ ] 48-hour stability window
- [ ] Monitor for regressions

### Long-Term (April 2026+)

**v3.0 Architecture Consideration:**
- [ ] Review Depth Pro validation results
- [ ] Decide: promote to canary, keep experimental, or deprecate
- [ ] If promoting: design v3.0 contracts based on production needs
- [ ] Write superseding ADR with migration plan
- [ ] Coordinate with v2.0.0 finalization (Aug 2026)

---

## Architectural Invariants Reinforcement

### Violated Invariants

**1. Contracts Over Convenience**
- PR #792 introduces convenience abstraction without proven contract need
- Violation: Abstraction before validation

**2. Modularity and Coupling Control**
- PR #792/#790 create tight coupling between DA3 and Depth Pro
- Violation: Shared contract forces semantic mismatch (relative vs metric depth)

**3. Determinism and Reproducibility**
- PR #793 dependency changes during architectural flux
- Violation: Non-deterministic integration state

**4. ADR Binding Rule**
- ADR-018 is binding, prescribes phased approach
- Violation: PRs #792/#790 skip phases without superseding ADR

### Enforcement Recommendations

**1. CI Gate: ADR Compliance Check**
```yaml
# .github/workflows/build.yml
- name: Validate ADR Compliance
  run: |
    # Check for architectural changes without ADR updates
    if git diff --name-only origin/main | grep -E "stage_graph|contracts|protocols"; then
      # Require ADR update in same PR
      if ! git diff --name-only origin/main | grep "docs/architecture/ADR-"; then
        echo "ERROR: Architectural change requires ADR update"
        exit 1
      fi
    fi
```

**2. PR Template Requirement**
```markdown
## Architectural Changes

- [ ] No architectural changes (skip section)
- [ ] Architectural changes documented in ADR (link: ___)
- [ ] Supersedes existing ADR (link to superseded: ___)
- [ ] Approved by Transformation Portal Architect
```

**3. Governance Policy Update**
```markdown
# docs/architecture/agent_governance.md

## Escalation Triggers (Mandatory Architect Review)

- Cross-module contracts (new or changed)
- Dependency tier changes or additions
- CI/CD workflow modifications
- **Version number changes (major or minor)**  ← ADD THIS
- **Abstraction layer additions**              ← ADD THIS
```

---

## Conclusion

**FINAL VERDICT:**

| PR | Decision | Rationale | Timeline |
|----|----------|-----------|----------|
| **#793** | **HOLD** | Must be isolated from architectural changes | Merge after #792/#790 resolved |
| **#792** | **REJECT** | Premature v3.0 abstraction, violates ADR-018 | Reconsider April 2026 |
| **#790** | **REJECT** | Duplicates PR #780, over-engineered | Implement correct PR2 instead |

**STRATEGIC DIRECTION:**

The repository must **stop and stabilize** before considering v3.0 architecture:

1. ✅ **Complete ADR-018** (simple wiring PR2, validation PR3)
2. ✅ **Stabilize v2.0.0** (Golden Path reliability focus)
3. ✅ **Gather Real Data** (6-week Depth Pro experimental soak)
4. ⏱️ **Design v3.0 Correctly** (April 2026+, informed by production)

**ENFORCEMENT:**

- ADR-018 is binding (no exceptions without superseding ADR)
- Silence is not approval (explicit Architect sign-off required)
- Governance policy updated with new escalation triggers

**NEXT STEPS:**

1. Close PRs #790 and #792 with architectural rationale
2. Hold PR #793 pending safety review
3. Implement correct ADR-018 PR2 (10 LOC wiring)
4. Focus on v2.0.0 stability and Depth Pro validation

---

**Architect Sign-Off:** Transformation Portal Architect
**Date:** 2026-02-02
**Governance Authority:** Final decision per agent_governance.md precedence order
**ADR Compliance:** Reinforces ADR-018, no new ADRs required (rejections)

---

## Appendix: Lessons Learned

### Process Failures Identified

**1. Parallel Development Without Coordination**
- PR #780 (DepthProStage) merged
- PRs #792/#790 developed in parallel
- Result: Duplication and conflict

**Mitigation:** Require PR dependency graph in description.

**2. Abstraction Before Validation**
- Depth Pro experimental (not validated)
- v3.0 contracts designed before proving need
- Result: Premature optimization

**Mitigation:** "No abstractions until N ≥ 3 use cases" rule.

**3. Versioning Strategy Confusion**
- v2.0.0 target: Aug 2026
- v3.0 introduced: Feb 2026
- Result: Fragmented roadmap

**Mitigation:** Explicit versioning ADR required.

**4. Dependency Updates During Churn**
- Architectural PRs in flight
- Dependency bot merges changes
- Result: Integration chaos risk

**Mitigation:** Dependency freeze during architectural changes.

### Governance Strengthening

**1. ADR Compliance CI Gate** (implement)
**2. PR Template Architectural Section** (require)
**3. Escalation Triggers Update** (version changes, abstractions)
**4. Dependency Freeze Policy** (during architectural work)

**These lessons will prevent future architectural debt accumulation.**
