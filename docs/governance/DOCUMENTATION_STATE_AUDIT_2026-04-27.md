# Documentation State Audit - 2026-04-27

**Scope:** Repo-wide documentation classification after merged PRs #1552 through
PR #1562.

**Baseline:** `main` at PR #1562.

## Summary

Current-facing documentation is limited to the main README, documentation map,
setup/portal/frontdoor/CLI guides, API contract docs, CI workflow matrix,
governance docs, and active APEX/archive-gate docs. Older project reports and
session outputs are retained for historical context but must not be presented as
live operator guidance.

## Recent Merge Topics Captured

| PRs | Current-state documentation impact |
| --- | --- |
| #1561, #1562 | Typed API v1 envelope foundation plus typed OpenAPI response models for health/readiness routes. Existing `/healthz`, `/ready`, and `/v1/readiness` wire contracts remain stable. |
| #1559 | Docker image now has a `HEALTHCHECK`; Compose reads the root `.env` template with `required: false`. |
| #1553, #1558, #1560 | CI hardening and the current 30-workflow inventory are captured in `docs/ci/WORKFLOW_MATRIX.md`. |
| #1555, #1557 | Archive gate fixity preflight and the 2026-04-27 Gates A/B/C audit are canonical evidence for archive readiness. |
| #1552, #1554, #1556 | APEX model-family characterization, structured failure-code surfacing, confidence-only Materials V3 passthrough, V2 fallback, and SAM2 tile-merge regression coverage are current APEX state. |

## Agent And Copilot Instruction Surface

Live agent/Copilot guidance is current-facing and must stay aligned with this
audit:

- `.github/copilot-instructions.md`
- `.github/agents/README.md`
- `.github/agents/QUICK_START_v2.md`
- `.github/agents/transformation-portal-architect.md`
- `.github/agents/portal-app-steward.md`
- `.github/agents/transformation-portal-specialist.md`
- `docs/architecture/agent_governance.md`
- `docs/guides/CUSTOM_AGENT_GUIDE.md`
- `docs/reference/AGENT_QUICK_REFERENCE.md`
- `tests/test_custom_agent_config.py`

Archived RAG and milestone notes under `.github/agents/_archive/` and
`.github/agents/rag_system/_archive/` are historical evidence, not live
instructions.

## Top-Level Documentation Classification

| Directory | Classification | Current guidance |
| --- | --- | --- |
| `docs/api/` | Maintained | API and machine-mode contract documentation. |
| `docs/apex/` | Maintained | APEX governance, policy, and contract docs. |
| `docs/archive/` | Mixed | Archive machine schemas/docs are current; older phase manifests are historical. |
| `docs/architecture/` | Maintained / ADR history | ADRs and active architecture docs remain canonical unless marked superseded. |
| `docs/ci/`, `docs/ci_cd/` | Maintained | `docs/ci/WORKFLOW_MATRIX.md` is the current workflow inventory. |
| `docs/cli/` | Maintained | Current CLI references and Lux Depth V3 guide. |
| `docs/compliance/`, `docs/contracts/`, `docs/deployment/`, `docs/governance/`, `docs/performance/`, `docs/schemas/`, `docs/validation/` | Maintained | Active governance, contract, schema, validation, and deployment surfaces. |
| `docs/guides/` | Mixed | Current setup, portal, frontdoor, troubleshooting, and Lux Depth guides are maintained; older project-specific guides are historical unless linked from the documentation map. |
| `docs/750_picacho/`, `docs/projects/`, `docs/quality_analysis/`, `docs/visual_review/` | Historical project records | 2025 project-specific analysis; do not use as current pipeline guidance. |
| `docs/depth_model/`, `docs/depth_pipeline/`, `docs/pipeline/`, `docs/pipeline_docs/` | Historical / superseded guidance | Current depth and pipeline guidance lives in the main README, Lux Depth V3 CLI guide, ADR-019, ADR-0015, and APEX docs. |
| `docs/deliverables/`, `docs/project-status/`, `docs/reports/`, `docs/status/`, `docs/summaries/`, `docs/session_summaries/`, `docs/sessions/`, `docs/historical/`, `docs/pr_archive/`, `docs/pr_reports/`, `docs/pr_summaries/`, `docs/verification/` | Historical records | Retained as point-in-time reports, checklists, PR/session evidence, and delivery snapshots. |
| `docs/_archive/` | Archive-only | Retired or consolidated material. |
| `docs/brand/`, `docs/decisions/`, `docs/deprecation/`, `docs/development/`, `docs/examples/`, `docs/fixes/`, `docs/implementation/`, `docs/implementation_notes/`, `docs/incidents/`, `docs/investigations/`, `docs/materials/`, `docs/migration/`, `docs/operations/`, `docs/optimization/`, `docs/processing/`, `docs/quick_references/`, `docs/reference/`, `docs/spatial_ai/`, `docs/testing/`, `docs/version_history/`, `docs/workflow/`, `docs/workflows/` | Mixed | Use only when linked from the documentation map or when the file itself states a current owner/status. Otherwise treat as supporting or historical context. |

## Stale Reference Disposition

- `START_HERE` references are no longer current entry points. Use
  `README.md`, `docs/README.md`, and `docs/governance/DOCUMENTATION_MAP.md`.
- `PIPELINE_V1.1.0` material is historical project context, not current
  operator guidance.
- 2025 project status reports are retained as dated evidence; current repo state
  is defined by `main`, the main README, and the documentation map.
- Depth-model upgrade reports under `docs/depth_model/` are superseded for live
  operations by current Lux Depth V3 docs and ADRs.

## Validation Commands

```bash
git diff --check
make check-docs
make check-stale-docs
python scripts/governance/check_docs_structure.py --all
```
