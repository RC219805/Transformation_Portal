# Archived Scripts - Transformation Portal

**Archive Date:** 2026-02-10
**Archived By:** Transformation Portal Architect
**Reason:** Obsolescence / Supersession

---

## Purpose of This Archive

This directory contains scripts that were part of the Transformation Portal development history but are **no longer actively used or maintained** in production. They are preserved for:

1. **Historical Reference:** Understanding design evolution and earlier approaches
2. **Educational Value:** Examples of experimental patterns and proof-of-concept work
3. **Git History Preservation:** Maintaining clean `git mv` history for future archaeology
4. **Potential Future Reuse:** Components may inform future feature development

**⚠️ Important:** Code in this archive is **not maintained**, **not tested** in CI, and **may have unpatched security issues**. Do not use in production.

---

## Archived Files

### 1. `context_aware_rendering.py`

**Archive Date:** 2026-02-10
**Original Location:** `scripts/context_aware_rendering.py`
**Status:** Superseded
**Replacement:** Spatial AI orchestration (Phase 2, ADR-027)

#### What It Did
Context-aware rendering pipeline that integrated architectural context (plans, elevations, specs) into the rendering workflow:
- Room-specific material enhancement
- Dimension-aware composition
- Style-consistent color grading
- Context-driven depth processing

#### Why Archived
- **Superseded By:** Spatial AI pipeline orchestration provides superior context integration
- **Architecture Shift:** Moved from monolithic context pipeline to modular orchestration
- **Maintenance Burden:** Complex dependencies on `architectural_context_extractor` (also experimental)
- **No Production Usage:** Not integrated into any production workflows

#### Historical Context
Part of early exploration of context-aware processing. Valuable as proof-of-concept demonstrating:
- Architectural metadata extraction and use
- Room-type-specific rendering strategies
- Integration patterns for external architectural documents

**Key Commit:** 7558ff8b (APEX scaffolding)
**Last Modified:** 2026-02-07

---

### 2. `premium_context_pipeline.py`

**Archive Date:** 2026-02-10
**Original Location:** `scripts/premium_context_pipeline.py`
**Status:** Superseded (depends on archived `context_aware_rendering.py`)
**Replacement:** Production orchestrator (`src/transformation_portal/orchestrator.py`)

#### What It Did
Flagship pipeline combining:
1. Context extraction from architectural documents
2. Strategy derivation for room-specific rendering
3. Depth-aware processing (Depth Anything V2)
4. Material Response enhancement
5. Context-driven color grading
6. Quality metrics and validation

#### Why Archived
- **Dependency on Archived Code:** Imports `context_aware_rendering.py` (now archived)
- **Superseded By:** Production orchestrator with Spatial AI integration
- **No Active Usage:** Not imported by any production code
- **Complexity vs. Value:** Overlapping functionality with simpler, more maintainable patterns

#### Historical Context
Represented an ambitious attempt at "ultimate context-aware rendering". Demonstrated feasibility of:
- Multi-stage orchestration
- Context-driven parameter selection
- Quality tier differentiation (standard/premium/ultimate)

Valuable as reference for future premium/tiered workflow designs.

**Referenced In:**
- `docs/historical/CONTEXT_SYSTEM_COMPLETE.md`
- `docs/historical/STATUS.md`
- `docs/status/NEXT_STEPS.md`

**Last Modified:** 2026-02-07

---

### 3. `pipelines/lux_render_pipeline_plus_v3.py`

**Archive Date:** 2026-02-10
**Original Location:** `scripts/pipelines/lux_render_pipeline_plus_v3.py`
**Status:** Abandoned / Incomplete
**Replacement:** `src/transformation_portal/lux_depth_v3/pbr_processor.py`

#### What It Did
PBR (Physically Based Rendering) pipeline with:
- ACEScg → Rec.709 color space handling
- Full PBR map support: albedo, normal, roughness, metallic, AO, displacement
- Environment map sampling
- Multi-light simulation
- Quality presets (draft/preview/final)

#### Why Archived
- **Incomplete Implementation:** Contains NotImplementedError stub for Parallax Occlusion Mapping (POM)
  ```python
  # Line 208: raise NotImplementedError("POM displacement logic stub")
  ```
- **Superseded By:** Production PBR processor with cleaner architecture
- **Zero External Dependencies Claim:** Not accurate (imports numpy, PIL)
- **No Active Usage:** Not imported by any production or test code
- **Abandoned:** No meaningful commits in 5+ months

#### Historical Context
Early exploration of PBR rendering with NumPy/Pillow only (avoiding heavy ML dependencies). Demonstrates:
- PBR math implementation (BRDF, Fresnel, GGX)
- Color space transformation (sRGB ↔ linear)
- Multi-map compositing patterns

Valuable reference for understanding PBR fundamentals, but production code uses more robust libraries.

**TODO Reference:** `docs/analysis/TODO_INVENTORY.md:208` (POM stub)
**Last Modified:** 2026-02-07

---

### 4. `productivity/ci_monitor.py`

**Archive Date:** 2026-06-01
**Original Location:** `productivity/scripts/ci_monitor.py`
**Status:** Retired placeholder tooling
**Replacement:** Current CI status is documented in
`docs/ci/WORKFLOW_MATRIX.md` and `.github/workflows/README.md`

#### What It Did
Generated a local text dashboard from hard-coded example metrics for workflow
status, cache hit rate, test counts, and build time.

#### Why Archived
- **Placeholder Data:** The script did not query GitHub Actions or any live CI
  source.
- **Misleading Root Bundle:** It lived under the retired root `productivity/`
  bundle, which is now historical evidence rather than current tooling.
- **No Active Usage:** No maintained Make target, workflow, or validation lane
  depends on it.

#### Historical Context
Part of a November 2025 productivity-suite documentation bundle. The Markdown
evidence for that bundle is preserved under
`docs/historical/productivity-suite-2025/`.

---

## Migration Path

If you need functionality from archived code:

1. **Review Git History:**
   ```bash
   git log --follow -- archive/scripts/[filename]
   git show <commit>:scripts/[filename]
   ```

2. **Extract Specific Functions:**
   - Identify self-contained logic
   - Copy with attribution comment
   - Add tests for extracted code
   - Update to current code standards

3. **Consult Documentation:**
   - Check `docs/historical/CONTEXT_SYSTEM_COMPLETE.md` for context system evolution
   - Check `docs/architecture/` for ADRs related to superseding decisions
   - Check `CHANGELOG.md` for version history

4. **Ask the Architect:**
   - If unsure about rationale for archival
   - If considering reintroduction of archived patterns
   - If designing similar features

---

## Related Archived Modules

Other archived components (different locations):

| Component | Location | Status | Replacement |
|-----------|----------|--------|-------------|
| `depth_canonical/` | `src/transformation_portal/depth_canonical/` | Deleted | ADR-019 backend registry |
| `architectural_context_extractor.py` | (unknown) | External dependency | None (experimental) |
| PR #98 action items | `docs/pr_reports/` | Archived | Completed or obsolete |

---

## Archival Policy

Per `docs/architecture/agent_governance.md`:

- **Architect Authority:** Decision to archive requires Architect approval
- **Git History Preservation:** Use `git mv` (not delete) to preserve history
- **Documentation Required:** This README documents rationale and context
- **No CI Enforcement:** Archived code exempt from linting, testing, security scans
- **Reintroduction Process:** Requires new ADR if archived code patterns return

---

## Maintenance

**This archive is NOT maintained.** Updates only occur for:
- Correcting historical inaccuracies in this README
- Adding newly archived scripts
- Removing files if completely obsoleted and no historical value

**Do NOT:**
- Update code in archived files
- Fix bugs or security issues
- Add new features
- Run linters or formatters

If code is valuable enough to maintain, it should be un-archived and moved back to active development.

---

## Contact

Questions about archived code? Consult:
- **Transformation Portal Architect** (governance authority)
- **Git History** (`git log --follow -- archive/scripts/[file]`)
- **ADR Documents** (`docs/architecture/decisions/`)

---

**Document Version:** 1.0
**Last Updated:** 2026-02-10
**Next Review:** Not scheduled (archive is static)
