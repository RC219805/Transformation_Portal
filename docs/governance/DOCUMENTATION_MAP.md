# Documentation Map

**Purpose:** Current source of truth for finding maintained Transformation
Portal documentation.

**Last Updated:** 2026-05-12
**Maintainer:** Repository Architect
**Current baseline:** repo-wide refresh audit dated May 11, 2026, building on
`main` through PR #1721, with the May 12 architecture triage overlay for
`docs/architecture` file dispositions and the May 12 CLI reference alignment
for `docs/cli`.

Historical reports remain available for audit context, but they are not current
operator guidance unless they are linked here as canonical documents.

## Start Here

| Topic | Canonical Document | Purpose |
| --- | --- | --- |
| Repository overview | [README.md](../../README.md) | Project overview, install path, current operational surfaces |
| Documentation index | [docs/README.md](../README.md) | Current docs navigation and historical-boundary guidance |
| Documentation refresh audit | [DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md](DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md) | Repo-wide inventory and classification refresh |
| Architecture triage inventory | [architecture-inventory-2026-05-12.csv](audit/architecture-inventory-2026-05-12.csv) | Current architecture-file disposition overlay |
| CLI triage inventory | [cli-inventory-2026-05-12.csv](audit/cli-inventory-2026-05-12.csv) | Current CLI-file disposition overlay |
| Prior documentation state audit | [DOCUMENTATION_STATE_AUDIT_2026-04-27.md](DOCUMENTATION_STATE_AUDIT_2026-04-27.md) | Repo-wide docs classification after PR #1562, retained as historical baseline evidence |
| Setup | [SETUP_GUIDE.md](../guides/SETUP_GUIDE.md) | Local environment setup and dependency bring-up |
| Contribution workflow | [CONTRIBUTING.md](../../CONTRIBUTING.md) | Code, docs, issue, and PR expectations |
| Security | [SECURITY.md](../../SECURITY.md) | Security policy and reporting |

## Current Operational Docs

| Area | Canonical Document | Status |
| --- | --- | --- |
| Portal / orchestrator | [Portal + Orchestrator Quickstart](../guides/PORTAL_ORCHESTRATOR_QUICKSTART.md) | Maintained |
| Secure front door | [Portal Secure Front Door Quickstart](../guides/PORTAL_SECURE_FRONTDOOR_QUICKSTART.md) | Maintained |
| API / OpenAPI contracts | [API docs](../api/) | Maintained |
| Machine-mode metadata API | [Machine Mode Contract](../api/MACHINE_MODE_CONTRACT.md) | Maintained |
| Archive machine-mode API | [Archive Machine Mode Contract](../api/ARCHIVE_MACHINE_MODE_CONTRACT.md) | Maintained |
| Lux Depth V3 CLI | [Lux Depth V3 CLI Guide](../cli/LUX_DEPTH_V3_CLI_GUIDE.md) | Maintained |
| CLI entrypoints | [CLI Reference](../cli/CLI_REFERENCE.md) | Maintained |
| PBR CLI testing | [PBR CLI Testing Guide](../cli/PBR_CLI_TESTING_GUIDE.md) | Maintained |
| FastVLM advisory captioning runtime | [FastVLM Runtime](../runtimes/fastvlm.md) | Maintained |
| Orchestrator Postgres runtime (Phase 1.B/1.E) | [Orchestrator Postgres Runtime](../runtimes/orchestrator-postgres.md) | Maintained; durable `JobRepository` backend wired through `app.py`, opt-in via `TP_ORCHESTRATOR_STATE_BACKEND=postgres`; durable SSE replay remains separate |
| Lux Depth V3 troubleshooting | [Lux Depth V3 Troubleshooting](../guides/LUX_DEPTH_V3_TROUBLESHOOTING.md) | Maintained |
| Context-aware rendering | [Context-Aware Rendering](../guides/CONTEXT_AWARE_RENDERING.md) | Maintained |
| PBR processing | [PBR Processor Quickstart](../guides/PBR_PROCESSOR_QUICKSTART.md) | Maintained |
| Supported formats | [Supported File Formats](../guides/SUPPORTED_FILE_FORMATS.md) | Maintained |
| Design tokens reference | [Design tokens](../design/tokens.md) | Maintained (generated from `web/shared/shared-ui-tokens.css` + `web/secure-landing/portal-src/styles/tokens.css`) |

## Governance, CI, And Validation

| Area | Canonical Document | Status |
| --- | --- | --- |
| Documentation policy | [DOCUMENTATION_POLICY.md](DOCUMENTATION_POLICY.md) | Maintained |
| Documentation inventory | [documentation-inventory-2026-05-11.csv](audit/documentation-inventory-2026-05-11.csv) | Current repo-wide classification baseline; architecture overlay is tracked separately below |
| Architecture triage inventory | [architecture-inventory-2026-05-12.csv](audit/architecture-inventory-2026-05-12.csv) | Current disposition ledger for files formerly or currently under `docs/architecture` |
| CLI triage inventory | [cli-inventory-2026-05-12.csv](audit/cli-inventory-2026-05-12.csv) | Current disposition ledger for files formerly or currently under `docs/cli` |
| Repository organization | [REPO_ORGANIZATION.md](REPO_ORGANIZATION.md) | Maintained |
| Custom Agents | [CUSTOM_AGENT_GUIDE.md](../guides/CUSTOM_AGENT_GUIDE.md) | Maintained; live profiles are under `.github/agents/` |
| Agent quick reference | [AGENT_QUICK_REFERENCE.md](../reference/AGENT_QUICK_REFERENCE.md) | Maintained |
| Agent governance | [agent_governance.md](../architecture/agent_governance.md) | Maintained |
| Copilot instructions | [.github/copilot-instructions.md](../../.github/copilot-instructions.md) | Maintained repo-wide Copilot guidance |
| Claude Code instructions | [CLAUDE.md](../../CLAUDE.md) | Maintained repo-root guide for Claude Code; summarizes contracts, decomposition, marker discipline, and live agent profiles |
| Maintainer workflow reference | [AGENTS.md](../../AGENTS.md) | Maintained operator command reference (Make targets, validation scripts, AI skill policy) |
| Skill progression tracks | [SKILL_PROGRESS_TRACKS.md](../guides/SKILL_PROGRESS_TRACKS.md) | Maintained; maps recurring PR review themes to training drills |
| TODO governance | [TODO Inventory](../analysis/TODO_INVENTORY.md), [TODO Action Plan](../analysis/TODO_ACTION_PLAN.md), [TODO Inventory Quick Reference](../architecture/TODO_INVENTORY_QUICK_REF.md), [TODO Priority Schema](todo_priority_schema.yaml) | Maintained; scanner-governed baseline refreshed May 11, 2026 |
| CI workflow inventory | [WORKFLOW_MATRIX.md](../ci/WORKFLOW_MATRIX.md) | Maintained; current 30-workflow inventory |
| CI/CD workflow guide | [CI_CD_WORKFLOWS.md](../ci_cd/CI_CD_WORKFLOWS.md) | Maintained |
| Branch protection | [BRANCH_PROTECTION_SETUP.md](../ci/BRANCH_PROTECTION_SETUP.md) | Maintained |
| Dependency pinning | [ADR-032 Dependency Pinning Strategy](../architecture/ADR-032-dependency-pinning-strategy.md) | Maintained |
| Retired ML lock lanes | [Retired ML Lock Lanes - 2026-04-30](RETIRED_ML_LOCK_LANES_2026-04-30.md) | Maintained governance record for unsupported Linux/macOS Intel ML lanes |
| Production hardening gap (paid pilot) | [Production Hardening Gap - 2026-05-13](PRODUCTION_HARDENING_GAP_2026-05-13.md) | Paid-pilot baseline: what is already done, what is partial, what is net-new across Phases 1 through 7, plus pinned pilot acceptance commands |
| Test marker policy | [ADR-044 Test Marker Enforcement](../architecture/ADR-044-test-marker-enforcement.md) | Maintained |
| Security hardening report | [security_best_practices_report.md](security_best_practices_report.md) | Maintained status record |

## Architecture And Contracts

| Topic | Canonical Document | Status |
| --- | --- | --- |
| Architecture overview | [ARCHITECTURE.md](../architecture/ARCHITECTURE.md) | Maintained |
| Architecture index | [architecture/README.md](../architecture/README.md) | Maintained; lists canonical, promoted, review-required, moved historical, and delete-candidate architecture dispositions |
| Architecture triage ledger | [architecture-inventory-2026-05-12.csv](audit/architecture-inventory-2026-05-12.csv) | Current overlay for architecture-file disposition decisions |
| Depth backend unification | [ADR-019](../architecture/ADR-019-depth-backend-unification.md) | Maintained |
| DA3 non-commercial research tier | [ADR-0015](../architecture/adr-0015-da3-1-1-non-commercial-research-tier.md) | Maintained |
| Portal orchestrator roadmap | [PORTAL_ORCHESTRATOR_ROADMAP.md](../architecture/PORTAL_ORCHESTRATOR_ROADMAP.md) | Maintained planning doc |
| Portal front door roadmap | [PORTAL_FRONTDOOR_ROADMAP.md](../architecture/PORTAL_FRONTDOOR_ROADMAP.md) | Maintained planning doc |
| Portal UX/UI status snapshot | [Portal UX/UI Status Snapshot](../architecture/DNA_UX_UI_STRATEGY_REBASELINE_2026-04-08.md) | Current planning-context snapshot; does not choose the next implementation PR |
| APEX workflow design | [APEX_WORKFLOW_DESIGN.md](../architecture/APEX_WORKFLOW_DESIGN.md) | Maintained |
| Deterministic RAW ingest | [ADR-030](../architecture/ADR-030-phase2-deterministic-raw-ingest.md) | Maintained |
| Determinism harness spec | [SPEC-DH-001](../architecture/specifications/SPEC-DH-001.md) | Locked |

## APEX And Archive Gates

| Topic | Canonical Document | Status |
| --- | --- | --- |
| APEX governance | [APEX Governance Status](../apex/GOVERNANCE_STATUS.md) | Maintained |
| APEX contract | [APEX Contract](../apex/APEX_CONTRACT.md) | Maintained |
| APEX model-family characterization | [APEX Model Family Characterization Protocol](../validation/APEX_MODEL_FAMILY_CHARACTERIZATION_PROTOCOL.md) | Maintained |
| APEX canonical corpus bootstrap | [APEX Canonical Corpus Bootstrap](../validation/APEX_CANONICAL_CORPUS_BOOTSTRAP.md) | Maintained |
| Archive gates state audit | [Archive Gates A/B/C Audit](audit/archive-gates-2026-04-27.md) | Current audit evidence |
| Archive gate follow-up | [Wire Audit Pipeline Readiness](audit/follow-ups/ci-wire-audit-pipeline-readiness.md) | Follow-up scope |

## Historical And Archive Boundaries

| Area | Classification | Use |
| --- | --- | --- |
| `docs/750_picacho/`, `docs/analysis/`, `docs/projects/`, `docs/quality_analysis/`, `docs/visual_review/` | Historical or mixed project records | Dated project evidence and investigations; not current operator guidance unless linked above |
| `docs/depth_model/`, `docs/depth_pipeline/`, `docs/pipeline/`, `docs/pipeline_docs/` | Superseded or historical pipeline/depth material | Use current Lux Depth V3 docs and ADRs instead |
| `docs/deliverables/`, `docs/project-status/`, `docs/reports/`, `docs/status/`, `docs/summaries/`, `docs/session_summaries/`, `docs/sessions/`, `docs/historical/`, `docs/verification/` | Point-in-time reports | Audit context only |
| `docs/historical/architecture/` | Architecture historical records | Point-in-time architecture files moved out of current architecture navigation by the May 12 triage |
| `docs/historical/cli/` | CLI historical records | Point-in-time CLI implementation and coverage records moved out of current CLI navigation by the May 12 CLI alignment |
| `docs/pr_archive/`, `docs/pr_reports/`, `docs/pr_summaries/` | PR-specific records | Review and merge history only |
| `docs/pr_archive/architecture/` | Architecture PR/review evidence | PR-specific architecture records moved out of current architecture navigation by the May 12 triage |
| `docs/_archive/` | Archive-only | Retired or consolidated material |

## Maintenance Protocol

1. Check this map before creating new documentation.
2. Update the canonical document when one exists.
3. Put new documents in an approved top-level directory from
   [DOCUMENTATION_POLICY.md](DOCUMENTATION_POLICY.md).
4. Update this map when a new document becomes current guidance.
5. Label duplicate or stale material historical, archive it, or remove it from
   current navigation.

## Validation

```bash
make check-docs
make check-stale-docs
make check-doc-heading-links
python3 scripts/governance/check_docs_structure.py --all
```

Use `rg` for targeted stale-reference checks before merging documentation
changes.
