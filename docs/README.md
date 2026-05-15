# Transformation Portal Documentation

This directory contains both maintained operator documentation and historical
records. Use this page for current navigation; point-in-time reports remain in
place for audit context but are not live runbooks unless they are linked below
as canonical documents.

**Current baseline:** repo-wide refresh audit dated May 11, 2026, building on
`main` through PR #1721, with the May 12 architecture triage overlay for
`docs/architecture` file dispositions and the May 12 CLI reference alignment
for `docs/cli`.

## Start Here

| Need | Current Document |
| --- | --- |
| Repository overview and setup | [Main README](../README.md) |
| Full documentation map | [Documentation Map](governance/DOCUMENTATION_MAP.md) |
| Documentation refresh audit | [2026-05-11 Documentation Refresh Audit](governance/DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md) |
| Architecture triage inventory | [2026-05-12 Architecture Inventory](governance/audit/architecture-inventory-2026-05-12.csv) |
| CLI triage inventory | [2026-05-12 CLI Inventory](governance/audit/cli-inventory-2026-05-12.csv) |
| Prior documentation state audit | [2026-04-27 Documentation State Audit](governance/DOCUMENTATION_STATE_AUDIT_2026-04-27.md) |
| Local setup and environment checks | [Setup Guide](guides/SETUP_GUIDE.md) |
| Portal and orchestrator contracts | [Portal + Orchestrator Quickstart](guides/PORTAL_ORCHESTRATOR_QUICKSTART.md) |
| Managed front door | [Portal Secure Front Door Quickstart](guides/PORTAL_SECURE_FRONTDOOR_QUICKSTART.md) |
| Managed paid-pilot staging | [Managed Paid-Pilot Staging Runbook](deployment/managed_paid_pilot_staging_runbook.md) |
| Lux Depth V3 CLI | [Lux Depth V3 CLI Guide](cli/LUX_DEPTH_V3_CLI_GUIDE.md) |
| CLI entrypoint reference | [CLI Reference](cli/CLI_REFERENCE.md) |
| FastVLM advisory captioning runtime | [FastVLM Runtime](runtimes/fastvlm.md) |
| CI workflow inventory | [Workflow Matrix](ci/WORKFLOW_MATRIX.md) |

## Current Maintained Surfaces

| Area | Canonical Docs | Notes |
| --- | --- | --- |
| Portal / API | [Portal Quickstart](guides/PORTAL_ORCHESTRATOR_QUICKSTART.md), [API docs](api/) | `/healthz`, `/ready`, `/v1/readiness`, and job lifecycle routes are current governed surfaces. PR #1562 and later follow-ups added typed OpenAPI response models while preserving existing wire contracts. |
| Secure front door | [Front Door Quickstart](guides/PORTAL_SECURE_FRONTDOOR_QUICKSTART.md) | Node 22.x is the enforced local/runtime contract for `web/secure-landing`. |
| Managed paid-pilot staging | [Managed Paid-Pilot Staging Runbook](deployment/managed_paid_pilot_staging_runbook.md), [Paid-Pilot Managed-Services Smoke Gate](deployment/paid_pilot_services.md) | Phase 5.A is locally validated at `07a3e8e847dee4a6e1ccf46d6dcd80b612fe3753`; managed-provider validation remains pending until the same gate passes against provider Postgres, Redis, and S3-compatible storage. |
| Docker / environment | [Main README](../README.md), [`.env.example`](../.env.example) | Docker Compose reads the root `.env` template with `required: false`; set `TP_API_KEY` for non-throwaway runs. |
| CI / validation | [Workflow Matrix](ci/WORKFLOW_MATRIX.md), [CI/CD Workflows](ci_cd/CI_CD_WORKFLOWS.md) | The current GitHub Actions inventory contains 30 workflows after the Phase 1.4 refresh. |
| Agent / Copilot guidance | [Custom Agent Guide](guides/CUSTOM_AGENT_GUIDE.md), [Agent Quick Reference](reference/AGENT_QUICK_REFERENCE.md), [Copilot Instructions](../.github/copilot-instructions.md), [CLAUDE.md](../CLAUDE.md) | Live agent behavior is governed by `.github/agents/`, Copilot instructions, `CLAUDE.md`, and `docs/architecture/agent_governance.md`. |
| Skill progression | [Skill Progress Tracks](guides/SKILL_PROGRESS_TRACKS.md) | Maps recurring PR review themes to evidence-linked drills, acceptance tests, and review checklists. |
| TODO governance | [TODO Inventory](analysis/TODO_INVENTORY.md), [TODO Action Plan](analysis/TODO_ACTION_PLAN.md), [TODO Quick Reference](architecture/TODO_INVENTORY_QUICK_REF.md), [TODO Priority Schema](governance/todo_priority_schema.yaml) | Current scanner-governed baseline: 24 governed `NotImplementedError` items, 0 ungoverned TODOs, snapshot refreshed May 11, 2026. |
| Archive gates | [Archive Machine Contract](api/ARCHIVE_MACHINE_MODE_CONTRACT.md), [2026-04-27 Archive Gates Audit](governance/audit/archive-gates-2026-04-27.md) | Gates A/B/C are documented with the April 27 readiness audit and normalized JSON evidence. |
| APEX / Materials | [APEX Governance Status](apex/GOVERNANCE_STATUS.md), [APEX Workflow Design](architecture/APEX_WORKFLOW_DESIGN.md), [APEX Model Family Characterization](validation/APEX_MODEL_FAMILY_CHARACTERIZATION_PROTOCOL.md) | Recent merges added offline model-family characterization, failure-code surfacing, confidence-only pixel-op passthrough, V2 fallback, and SAM2 tile-merge regression coverage. |
| CLI references | [CLI Index](cli/README.md), [CLI Reference](cli/CLI_REFERENCE.md), [Lux Depth V3 CLI Guide](cli/LUX_DEPTH_V3_CLI_GUIDE.md), [PBR CLI Testing Guide](cli/PBR_CLI_TESTING_GUIDE.md) | Maintained CLI docs use repo-governed `.venv` and Make targets; old PBR coverage/checklist and CLI v1.3 notes are historical evidence. |
| Advisory captioning | [FastVLM Runtime](runtimes/fastvlm.md) | Optional subprocess-isolated sidecars only; captions are advisory and never satisfy APEX or Materials V3 gates. |
| Portal UX/UI planning | [Portal UX/UI Status Snapshot](architecture/DNA_UX_UI_STRATEGY_REBASELINE_2026-04-08.md) | Current planning context through #1721; status snapshot only, not a next-PR selector. |
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
- `docs/historical/architecture/` and `docs/pr_archive/architecture/` contain
  files moved out of `docs/architecture` by the May 12 architecture triage.
- `docs/historical/cli/` contains point-in-time CLI implementation and coverage
  records moved out of current `docs/cli` navigation by the May 12 CLI alignment.
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
  [2026-05-11 Documentation Refresh Audit](governance/DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md)
  and its inventory CSV.
- Architecture-specific disposition evidence lives in
  [2026-05-12 Architecture Inventory](governance/audit/architecture-inventory-2026-05-12.csv).
- CLI-specific disposition evidence lives in
  [2026-05-12 CLI Inventory](governance/audit/cli-inventory-2026-05-12.csv).
- Validate documentation structure with:

```bash
make check-docs
make check-stale-docs
make check-doc-heading-links
python3 scripts/governance/check_docs_structure.py --all
```

**Last Updated:** 2026-05-15
