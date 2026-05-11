# ADR-045: Monolith Decomposition Residuals — Governance Pattern

**Status:** Proposed
**Date:** 2026-05-04
**Decision Makers:** Architect (review) + Specialist (implementation)
**Replaces:** None
**Supersedes:** None
**Related:** [ADR-043 Orchestrator Decomposition Strategy](ADR-043-orchestrator-decomposition.md), [Development Roadmap Q2 2026 §4](DEVELOPMENT_ROADMAP_2026_Q2.md), [Monolith Decomposition Targets](MONOLITH_DECOMPOSITION_TARGETS.md)

---

## Context

ADR-043 successfully decomposed `EnhanceOrchestrator` (was 6,108 LOC) into five focused seams (`config_resolver`, `pipeline_coordinator`, `execution_engine`, `artifact_manager`, `validators/`). It explicitly scoped that work to a single class and a single module. It did **not** establish a reusable governance pattern for the rest of the repo.

The 2026-05-04 audit ([TODO Inventory refresh](../analysis/TODO_INVENTORY.md), snapshot at [`todo_scanner_snapshot.json`](../analysis/todo_scanner_snapshot.json)) triggered this ADR. The 2026-05-11 scanner refresh kept the same conclusion for decomposition planning and refreshed the tracked snapshot to 24 governed `NotImplementedError` items, 0 ungoverned TODOs, and 1,570 files scanned. The audit confirms:

- `src/` source-code TODOs about decomposition: **0** (CI-enforced via `enforcement.yml`).
- Five known monoliths still exceed informal "acceptable" sizes: `app.py` (10,039 LOC), `lux_depth_v3/orchestrator.py` (7,257), `lux_depth_v3/segmentation_backend.py` (2,519), `pipelines/rendering_4k_pipeline.py` (2,380), `spatial_ai/orchestration/pipeline.py` (2,121).
- The Q2 2026 roadmap (§4 "Orchestrator Residual Slimming & Boundary Enforcement") commits to ratcheting `orchestrator.py` down by ~200 LOC/quarter but does not bind the other four files.

Without a shared governance pattern, future extractions will drift in shape, tests, and re-export contracts — undoing the consistency ADR-043 achieved.

### Current Problems

1. **No standing pattern:** Each future decomposition would re-derive its own seam discipline, risking inconsistency.
2. **No persistent target list:** Without a tracked list of candidate seams, decomposition becomes ad-hoc and reactive.
3. **No bright line for boundary drift:** Once a module is decomposed, helper logic can creep back without a stated invariant to point to.
4. **No backward-compat contract:** ADR-043 chose re-export shims for compatibility, but that choice has not been generalized.

### Metrics (informational only — this ADR does not bind specific values)

| Module | LOC | TODO/FIXME markers | Has destination seams? |
|---|---:|---:|---|
| `app.py` | 10,039 | 0 | No (greenfield) |
| `lux_depth_v3/orchestrator.py` | 7,257 | 0 | Yes (ADR-043 modules) |
| `lux_depth_v3/segmentation_backend.py` | 2,519 | 0 | No |
| `pipelines/rendering_4k_pipeline.py` | 2,380 | 0 | No |
| `spatial_ai/orchestration/pipeline.py` | 2,121 | 0 (1 governed phase gate) | No |

---

## Decision

Adopt a single, repo-wide governance pattern for monolith decomposition, recorded here, and tracked operationally via [`MONOLITH_DECOMPOSITION_TARGETS.md`](MONOLITH_DECOMPOSITION_TARGETS.md).

### Pattern (binding when invoked)

A decomposition PR claiming this ADR **must**:

1. **Pick a named seam from the targets doc.** No ad-hoc seams; if a new seam is needed, the targets doc is amended in a separate PR first.
2. **Extract by responsibility, not by line count.** Lines are an indicator, not the goal. The seam must encapsulate one cohesive responsibility and expose a minimal data-class or callable contract.
3. **Add the new module + its unit tests first.** The new module lands green before any caller switches.
4. **Maintain backward compatibility via re-exports** from the original file (e.g., `from .new_module import Symbol  # noqa: F401`). No call-site rewrites in unrelated modules during the same PR.
5. **Keep imports acyclic.** Use `TYPE_CHECKING` guards if a circular type reference appears; never solve a cycle by re-merging modules.
6. **Preserve lazy-import discipline.** Heavy ML imports must remain lazy in the new module — the CLAUDE.md "lazy imports are mandatory" rule still applies.
7. **Run governance gates clean:** `python scripts/validation/scan_todo_inventory.py --check-governance`, `make check-json-serialization check-yaml-governance check-python-headers`, plus the per-target verification commands listed in the targets doc.
8. **Cite this ADR in the PR description and update the corresponding row in the targets-doc status table.**

### Out of scope (binding)

- This ADR does **not** schedule any specific extraction. Schedule lives in `MONOLITH_DECOMPOSITION_TARGETS.md` and the quarterly roadmap.
- This ADR does **not** introduce new test markers, new linters, or new CI jobs. It reuses the existing scanner and contract tests.
- This ADR does **not** override module-specific governance (`app.py` hardening, run-card contract, ingest contract, evidence/attestation layering). When a seam touches one of those, the relevant ADR is the binding authority and this one is supplementary.

---

## Alternatives Considered

### Alternative 1 — Do nothing; rely on per-PR judgment
Rejected. ADR-043's success came from a written pattern. Without one, future decompositions re-derive seam discipline and quickly diverge in shape.

### Alternative 2 — One ADR per monolith
Rejected. Five ADRs to repeat the same pattern is governance overhead. ADR-043 already exists for the orchestrator-specific decisions; subsequent work needs the *pattern*, not five copies of it.

### Alternative 3 — Bind specific LOC ratchets per file
Rejected. The Q2 roadmap already binds `orchestrator.py` to ~200 LOC/quarter. Binding the other monoliths the same way would force extractions where the seam isn't ready, encouraging line-count-driven (rather than responsibility-driven) splits.

### Alternative 4 — Pattern ADR + persistent targets doc (selected)
Selected. Captures the cross-cutting pattern once, lets per-target judgment live in a maintainable list, and keeps schedule in the roadmap.

---

## Consequences

### Positive
- Future decomposition PRs have a checklist instead of a debate.
- The targets doc surfaces decomposition intent to reviewers without adding source-code TODOs.
- Consistent backward-compat strategy (re-export shims) preserves call-sites.
- `app.py`, `segmentation_backend.py`, `rendering_4k_pipeline.py`, and `spatial_ai/orchestration/pipeline.py` get a path to decomposition without each requiring its own ADR.

### Negative
- Adds a small bookkeeping cost: the targets doc must be kept current as seams land.
- Risk that the targets doc becomes stale if not refreshed in the monthly inventory cadence.

### Risks
- **Boundary drift back into source modules.** Mitigation: continue the existing CLAUDE.md "do not re-monolithize" invariant; require PR descriptions to cite ADR-045.
- **Premature extraction.** Mitigation: Pattern step 1 forbids ad-hoc seams; new seams must land in the targets doc in a separate PR with reviewer sign-off.
- **Test coverage erosion.** Mitigation: Pattern step 3 mandates new tests on the new module before any caller switches.

---

## Implementation Plan

This ADR is governance, not implementation. Activation follows three phases:

### Phase 1 — Approve and link (this PR)
- Land this ADR (status: Proposed) plus `MONOLITH_DECOMPOSITION_TARGETS.md` and the inventory refresh.
- Cross-link from `TODO_INVENTORY_QUICK_REF.md` and `DEVELOPMENT_ROADMAP_2026_Q2.md` §4.

### Phase 2 — First exercise (next PR, separate ticket)
- Select a single seam from `MONOLITH_DECOMPOSITION_TARGETS.md` (recommended starter: Target 1, Seam 1A — config-fingerprint helpers).
- Extract under this ADR's pattern. Update the targets-doc status table.

### Phase 3 — Ratchet
- Each subsequent quarter, the architect reviews the targets doc, retires shipped seams, and re-ranks remaining ones in light of new pressure.
- Status of this ADR moves from `Proposed` → `Active` after the first successful exercise of the pattern lands on `main`.

---

## Success Criteria

- [ ] First decomposition PR cites ADR-045 and follows the pattern checklist (target: 2026-Q2).
- [ ] Targets doc status table is updated with each shipped seam.
- [ ] No future PR introduces a new seam without first amending the targets doc.
- [ ] `python scripts/validation/scan_todo_inventory.py --check-governance` continues to exit 0 across all decomposition PRs.
- [ ] `orchestrator.py` continues to track the Q2 roadmap's ~200 LOC/quarter ratchet (or documents why the ratchet is infeasible for that quarter).

---

## Enforcement

- **Code review:** PRs claiming "decomposition" or "extract" in the title must cite ADR-045 in the description; reviewers verify the eight pattern requirements.
- **Governance scanner:** existing `scan_todo_inventory.py --check-governance` continues to block ungoverned TODOs; this ADR adds no new scanner rules.
- **Boundary invariant:** CLAUDE.md "Lux Depth V3 is a decomposed orchestrator (do not re-monolithize)" generalizes — extracted seams across the repo must not be re-merged absent a superseding ADR.

---

## Review and Maintenance

- **Owner:** Transformation Portal Architect
- **Cadence:** Reviewed alongside the monthly TODO inventory refresh; updated when a new monolith is identified or an existing target is retired.
- **Promotion criteria:** Move status `Proposed → Active` after the first successful pattern exercise (Phase 2). Move `Active → Superseded` only via a follow-on ADR.

---

## References

- [ADR-043 Orchestrator Decomposition Strategy](ADR-043-orchestrator-decomposition.md) — predecessor; concrete instance the pattern generalizes.
- [Monolith Decomposition Targets](MONOLITH_DECOMPOSITION_TARGETS.md) — operational target list.
- [Development Roadmap Q2 2026 §4](DEVELOPMENT_ROADMAP_2026_Q2.md) — residual-slimming pressure.
- [TODO Inventory 2026-05-11 Refresh](../analysis/TODO_INVENTORY.md) — current scanner baseline and the earlier 2026-05-04 audit that triggered this ADR.
- [`docs/analysis/todo_scanner_snapshot.json`](../analysis/todo_scanner_snapshot.json) — reproducible scanner baseline.
- [CLAUDE.md "do not re-monolithize"](../../CLAUDE.md) — invariant generalized by this ADR.
