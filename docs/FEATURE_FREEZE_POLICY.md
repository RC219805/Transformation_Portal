# Feature Freeze Policy - Narrative Consolidation Period

**Period**: December 20, 2025 - January 10, 2026  
**Duration**: 3 weeks  
**Objective**: Consolidate narrative structure while preserving technical capability

---

## Purpose

The Transformation Portal is entering a **narrative consolidation period** to improve documentation clarity and user onboarding. This freeze prevents feature sprawl during reorganization.

**This is NOT**:
- A code freeze (development continues)
- A deprecation period (no features removed)
- A quality gate (codebase quality is excellent)

**This IS**:
- A pause on new feature surfaces
- A focus on validation and performance
- A documentation reorganization window

---

## What Is Blocked 🚫

### New Feature Surfaces
- ❌ New pipelines (no lux_depth_v4, no unified_v2)
- ❌ New modules (no new src/ or root-level packages)
- ❌ New presets (interior_luxury, exterior_showcase are sufficient)
- ❌ New stages in async pipeline
- ❌ New metrics in quality control system
- ❌ New LUT categories (film_emulation, location_aesthetic, material_response exist)

### Experimental Features
- ❌ New AI model integrations (Depth Anything 3 already deferred)
- ❌ New rendering techniques
- ❌ New training infrastructure components

### Documentation Sprawl
- ❌ New top-level documentation sections in README
- ❌ New "Latest Update" or "NEW" banners
- ❌ New docs subdirectories (48 already exists)

---

## What Is Allowed ✅

### Security & Stability
- ✅ Security patches (CVE fixes, input validation)
- ✅ Bug fixes (functional regressions, edge cases)
- ✅ CI/CD improvements (test reliability, pipeline speed)
- ✅ Dependency updates (security, compatibility)

### Validation & Performance
- ✅ Validation tightening (test coverage, quality metrics)
- ✅ Performance profiling (benchmarks, bottleneck analysis)
- ✅ Edge refinement validation (current work stream)
- ✅ Quality baseline establishment

### Ergonomics & UX
- ✅ CLI improvements (help text, error messages)
- ✅ Logging enhancements (clarity, structured output)
- ✅ Documentation clarification (fix broken links, improve examples)
- ✅ Example workflow updates

### Infrastructure
- ✅ Docker image optimization
- ✅ Observability improvements (Prometheus metrics)
- ✅ Deployment configuration refinement

---

## Approval Process

### For Blocked Items
If you believe a blocked item is critical:

1. **Create GitHub Issue** with label `feature-freeze-exception`
2. **Justify Urgency**:
   - Why can't this wait 3 weeks?
   - What breaks if we don't add this now?
   - Is this truly a new surface or a refinement?
3. **Request Architect Review** (@transformation-portal-architect)
4. **Wait for Approval** before proceeding

### For Allowed Items
- No special approval needed
- Follow normal PR review process
- Mention in PR description: "Feature freeze: allowed (bug fix / validation / ergonomics)"

---

## Freeze Exceptions (Pre-Approved)

### Edge Refinement Promotion
- **Status**: ✅ Approved
- **Scope**: Add `--edge-refinement` CLI flag to lux_depth_v2
- **Justification**: Already implemented (Dec 20), just needs validation and config promotion
- **Constraint**: No new refinement techniques during freeze

### Golden Path Documentation
- **Status**: ✅ Approved
- **Scope**: README restructure, training section collapse
- **Justification**: Core freeze objective
- **Constraint**: No content deletion, only reorganization

---

## Rationale

### Why Now?
1. **Recent work signals production readiness**: PR #574 security hardening, Phase 2 deployment
2. **Documentation has reached navigation threshold**: 888 .md files, 48 subdirectories
3. **Multiple overlapping features**: Async, unified, context-aware, training at equal weight
4. **Commercial readiness preparation**: Clear onboarding path needed

### Why 3 Weeks?
- Week 1: Golden Path declaration, README restructure
- Week 2: Edge refinement validation, training docs collapse
- Week 3: Validation, performance baseline, communication
- Buffer for holidays/unexpected delays

### Why Feature Surface (Not Code Freeze)?
- **Allows**: Refinement of existing capabilities
- **Blocks**: Addition of new decision branches
- **Preserves**: Development velocity on validation/performance
- **Prevents**: Further narrative complexity during consolidation

---

## Success Criteria

The freeze can be lifted when:

1. ✅ Golden Path is clearly defined in README (lines 1-100)
2. ✅ Training docs are gated (not prominently displayed)
3. ✅ Edge refinement validation is complete
4. ✅ Feature freeze policy is documented in CONTRIBUTING.md
5. ✅ Success metrics are achieved:
   - Time to "How do I process images?" <2 min
   - Decision branches presented: 1 + "Advanced"
   - Lines to scroll before lux-depth-v2: <10

**Target Completion**: January 10, 2026  
**Review Date**: January 8, 2026 (assess if extension needed)

---

## Communication

### GitHub Issue Template
A new issue template (`.github/ISSUE_TEMPLATE/feature_freeze_check.md`) will guide contributors.

### Contributor Guide
`CONTRIBUTING.md` updated with freeze policy section.

### README Badge (Optional)
Consider adding temporary badge:
```markdown
![Status](https://img.shields.io/badge/status-narrative_consolidation-yellow)
```

### GitHub Discussion
Pinned discussion: "Golden Path Consolidation - Dec 20-Jan 10"

---

## FAQ

**Q: Can I fix a bug in lux_depth_v2?**  
A: ✅ Yes - bug fixes are allowed.

**Q: Can I add a new preset for "modern_minimalist"?**  
A: ❌ No - new presets are blocked. Use existing presets with custom parameters.

**Q: Can I improve error messages in the CLI?**  
A: ✅ Yes - ergonomics improvements are allowed.

**Q: Can I integrate Depth Anything 4 when it releases?**  
A: ❌ No - new model integrations are blocked. File issue for post-freeze.

**Q: Can I add unit tests for edge cases?**  
A: ✅ Yes - validation tightening is encouraged.

**Q: Can I optimize GPU memory usage in async pipeline?**  
A: ✅ Yes - performance improvements are allowed.

**Q: Can I add training data augmentation techniques?**  
A: ⚠️ Maybe - request exception if critical for research. Otherwise wait.

**Q: Can I update documentation to clarify confusion?**  
A: ✅ Yes - documentation clarification is allowed (not new sections).

---

## Enforcement

### Pull Request Checklist
All PRs during freeze period must answer:
- [ ] Does this add a new feature surface? (If yes → requires exception)
- [ ] Is this a bug fix / validation / ergonomics improvement? (If yes → proceed)
- [ ] Does this modify README structure? (If yes → aligns with Golden Path consolidation?)

### CI/CD
No automated enforcement (relies on code review).

### Architect Review
Transformation Portal Architect has final authority on exception requests.

---

## Post-Freeze

After January 10, 2026:
- Feature surface freeze **lifts**
- New pipelines/modules **allowed** (with architectural review)
- Documentation sprawl **still discouraged** (lessons learned applied)
- Golden Path **remains canonical** (advanced features clearly labeled)

---

**Document Control**:
- Version: 1.0
- Effective: December 20, 2025
- Expiration: January 10, 2026
- Owner: Transformation Portal Architect
- Review Date: January 8, 2026

---

**Status**: ✅ **ACTIVE**
