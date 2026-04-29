# Documentation Refresh Audit - 2026-04-29

**Scope:** Repo-wide documentation inventory and navigation refresh against
local repository `HEAD` at audit time.

**Baseline:** Current `main` plus this documentation refresh. The prior
product-state documentation baseline remains `main` through PR #1562
(April 27, 2026).

## Summary

This refresh inventories tracked documentation surfaces and aligns current
navigation without changing runtime behavior, route contracts, schemas, CLI
flags, selectors, Make targets, or validation semantics.

The full inventory is maintained in
[`documentation-inventory-2026-04-29.csv`](audit/documentation-inventory-2026-04-29.csv).

## Inventory Results

| Classification | Count | Meaning |
| --- | ---: | --- |
| `canonical` | 49 | Current source-of-truth documents linked from active navigation. |
| `current-support` | 267 | Supporting docs, schemas, READMEs, examples, or references that remain useful for current work. |
| `mixed` | 207 | Directories or files that need context before being treated as current guidance. |
| `historical` | 423 | Point-in-time project, PR, delivery, status, or validation evidence. |
| `archive-only` | 44 | Retired or consolidated evidence retained for audit context. |

High-volume directories remain consistent with the existing retention model:

- `docs/architecture/`: maintained ADR and architecture surface, with older PR
  reviews and completion reports classified historical.
- `docs/guides/`: mixed; current setup, portal, frontdoor, CLI support, and
  troubleshooting guides are canonical/current-support, while older project-era
  guides remain historical or mixed.
- `docs/historical/`, `docs/pr_archive/`, `docs/sessions/`, and
  `docs/project-status/`: historical by default.
- `.github/agents/`: live profile files are canonical; archived RAG/milestone
  material remains archive-only; RAG support docs are current-support but not
  live role-boundary instructions.

## Current Navigation

Current documentation entry points are:

- `README.md`
- `docs/README.md`
- `docs/governance/DOCUMENTATION_MAP.md`
- this refresh audit
- `docs/governance/DOCUMENTATION_STATE_AUDIT_2026-04-27.md` for the prior
  post-PR #1562 state classification

Current live agent/Copilot instruction surfaces remain:

- `.github/copilot-instructions.md`
- `.github/agents/README.md`
- `.github/agents/QUICK_START_v2.md`
- `.github/agents/transformation-portal-architect.md`
- `.github/agents/portal-app-steward.md`
- `.github/agents/transformation-portal-specialist.md`
- `docs/architecture/agent_governance.md`
- `docs/guides/CUSTOM_AGENT_GUIDE.md`
- `docs/reference/AGENT_QUICK_REFERENCE.md`

## Stale-Risk Findings

A repo-wide stale root-doc reference scan found 91 missing `docs/<file>`
references in inventory files. The majority are historical or archive-only
records that intentionally preserve old point-in-time paths. Current-facing
navigation is updated by this refresh instead of rewriting old evidence.

Known current-support stale-risk patterns that should be handled only when the
owning surface is next touched:

- ADR templates and older architecture records may include example `docs/*`
  paths that were never live repo files.
- Tool/sample READMEs may reference project-local `DOCS/` output directories;
  those are not repository documentation links.
- Historical records may cite pre-consolidation root-level docs paths.

## Follow-Ups

- Review `mixed` rows in the inventory before promoting any file into current
  navigation.
- Keep future navigation changes in `docs/governance/DOCUMENTATION_MAP.md`
  first; update `README.md` and `docs/README.md` only for current entry-point
  changes.
- Prefer directory-level historical notes for high-risk mixed directories over
  mass-editing point-in-time records.

## Validation

Required validation for this refresh:

```bash
git diff --check
python3 scripts/governance/check_docs_structure.py --all
make check-docs
make check-stale-docs
make check-todo-governance
make validate-ci
make docs
make ci-quick
make check-worktree
```
