# Transformation Portal Documentation

This directory contains both maintained operator documentation and historical
records. Use this page for current navigation; point-in-time reports remain in
place for audit context but are not live runbooks unless they are linked below
as canonical documents.

**Current baseline:** repo-wide refresh audit dated April 29, 2026, building on
`main` through PR #1562.

## Start Here

| Need | Current Document |
| --- | --- |
| Repository overview and setup | [Main README](../README.md) |
| Full documentation map | [Documentation Map](governance/DOCUMENTATION_MAP.md) |
| Documentation refresh audit | [2026-04-29 Documentation Refresh Audit](governance/DOCUMENTATION_REFRESH_AUDIT_2026-04-29.md) |
| Documentation state audit | [2026-04-27 Documentation State Audit](governance/DOCUMENTATION_STATE_AUDIT_2026-04-27.md) |
| Local setup and environment checks | [Setup Guide](guides/SETUP_GUIDE.md) |
| Portal and orchestrator contracts | [Portal + Orchestrator Quickstart](guides/PORTAL_ORCHESTRATOR_QUICKSTART.md) |
| Managed front door | [Portal Secure Front Door Quickstart](guides/PORTAL_SECURE_FRONTDOOR_QUICKSTART.md) |
| Lux Depth V3 CLI | [Lux Depth V3 CLI Guide](cli/LUX_DEPTH_V3_CLI_GUIDE.md) |
| FastVLM advisory captioning runtime | [FastVLM Runtime](runtimes/fastvlm.md) |
| CI workflow inventory | [Workflow Matrix](ci/WORKFLOW_MATRIX.md) |

## Current Maintained Surfaces

| Area | Canonical Docs | Notes |
| --- | --- | --- |
| Portal / API | [Portal Quickstart](guides/PORTAL_ORCHESTRATOR_QUICKSTART.md), [API docs](api/) | `/healthz`, `/ready`, and `/v1/readiness` are current health/readiness surfaces. PR #1562 added typed OpenAPI response models while preserving the existing wire contracts. |
| Secure front door | [Front Door Quickstart](guides/PORTAL_SECURE_FRONTDOOR_QUICKSTART.md) | Node 22.x is the enforced local/runtime contract for `web/secure-landing`. |
| Docker / environment | [Main README](../README.md), [`.env.example`](../.env.example) | Docker Compose reads the root `.env` template with `required: false`; set `TP_API_KEY` for non-throwaway runs. |
| CI / validation | [Workflow Matrix](ci/WORKFLOW_MATRIX.md), [CI/CD Workflows](ci_cd/CI_CD_WORKFLOWS.md) | The current GitHub Actions inventory contains 30 workflows after the Phase 1.4 refresh. |
| Agent / Copilot guidance | [Custom Agent Guide](guides/CUSTOM_AGENT_GUIDE.md), [Agent Quick Reference](reference/AGENT_QUICK_REFERENCE.md), [Copilot Instructions](../.github/copilot-instructions.md), [CLAUDE.md](../CLAUDE.md) | Live agent behavior is governed by `.github/agents/`, Copilot instructions, `CLAUDE.md`, and `docs/architecture/agent_governance.md`. |
| Skill progression | [Skill Progress Tracks](guides/SKILL_PROGRESS_TRACKS.md) | Maps recurring PR review themes to evidence-linked drills, acceptance tests, and review checklists. |
| Archive gates | [Archive Machine Contract](api/ARCHIVE_MACHINE_MODE_CONTRACT.md), [2026-04-27 Archive Gates Audit](governance/audit/archive-gates-2026-04-27.md) | Gates A/B/C are documented with the April 27 readiness audit and normalized JSON evidence. |
| APEX / Materials | [APEX Governance Status](apex/GOVERNANCE_STATUS.md), [APEX Workflow Design](architecture/APEX_WORKFLOW_DESIGN.md), [APEX Model Family Characterization](validation/APEX_MODEL_FAMILY_CHARACTERIZATION_PROTOCOL.md) | Recent merges added offline model-family characterization, failure-code surfacing, confidence-only pixel-op passthrough, V2 fallback, and SAM2 tile-merge regression coverage. |
| Advisory captioning | [FastVLM Runtime](runtimes/fastvlm.md) | Optional subprocess-isolated sidecars only; captions are advisory and never satisfy APEX or Materials V3 gates. |
| Dependency policy | [ADR-032](architecture/ADR-032-dependency-pinning-strategy.md), [Retired ML Lock Lanes](governance/RETIRED_ML_LOCK_LANES_2026-04-30.md), [AGENTS.md](../AGENTS.md) | Layered lockfiles and the Apple Silicon target-owned ML lane are the current dependency governance model. |

## Historical And Archive Material

The repo intentionally retains older project reports, PR notes, quality studies,
and session artifacts. Treat these directories as historical unless a current
map explicitly links a document as canonical:

- `docs/750_picacho/`, `docs/projects/`, and `docs/quality_analysis/` contain
  2025 project-specific analysis and delivery records.
- `docs/depth_model/`, `docs/pipeline/`, and `docs/pipeline_docs/` contain older
  depth-model and luxury-pipeline evaluation material. Current depth behavior is
  described in the main README, CLI guide, and ADR-019/ADR-0015.
- `docs/reports/`, `docs/status/`, `docs/session_summaries/`,
  `docs/sessions/`, `docs/historical/`, and `docs/pr_archive/` are
  point-in-time records.
- `docs/_archive/` contains intentionally retired or consolidated material.

Do not use historical documents as operator runbooks without first checking the
current documentation map.

## Documentation Governance

- `docs/README.md` is the only maintained file allowed directly under `docs/`.
- New documents must live in an approved top-level directory from
  [Documentation Retention Policy](governance/DOCUMENTATION_POLICY.md).
- Current navigation belongs in [Documentation Map](governance/DOCUMENTATION_MAP.md);
  duplicate or superseded material should be archived, labeled historical, or
  removed from current indexes.
- Repo-wide classification evidence lives in
  [2026-04-29 Documentation Refresh Audit](governance/DOCUMENTATION_REFRESH_AUDIT_2026-04-29.md)
  and its inventory CSV.
- Validate documentation structure with:

```bash
make check-docs
make check-stale-docs
make check-doc-heading-links
python3 scripts/governance/check_docs_structure.py --all
```

**Last Updated:** 2026-04-29
