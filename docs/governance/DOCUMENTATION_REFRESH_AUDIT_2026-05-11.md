# Documentation Refresh Audit - 2026-05-11

**Scope:** Repo-wide documentation baseline refresh against local repository
`main` at `9f7e25fc9` / PR #1721.

**Baseline:** Current `main` plus this documentation refresh. The prior
repo-wide documentation refresh audit dated 2026-04-29 remains retained as
point-in-time audit evidence.

## Summary

This refresh inventories tracked documentation surfaces and aligns current
navigation with repo state through PR #1721. It does not change runtime
behavior, route contracts, schemas, CLI flags, selectors, Make targets,
dependency locks, packages, workflow behavior, or validation semantics.

The full inventory is maintained in
[`documentation-inventory-2026-05-11.csv`](audit/documentation-inventory-2026-05-11.csv).

## Inventory Results

| Classification | Count | Meaning |
| --- | ---: | --- |
| `canonical` | 52 | Current source-of-truth documents linked from active navigation. |
| `current-support` | 331 | Supporting docs, schemas, READMEs, examples, or references that remain useful for current work. |
| `mixed` | 205 | Directories or files that need context before being treated as current guidance. |
| `historical` | 410 | Point-in-time project, PR, delivery, status, or validation evidence. |
| `archive-only` | 52 | Retired or consolidated evidence retained for audit context. |

The new inventory preserves the prior CSV schema:

```text
path,classification,current_owner,canonical_target,recommended_action,reason
```

## Material Changes Since The 2026-04-29 Audit

This audit updates current navigation for post-April 29 repo state without
rewriting historical records:

- Typed response-model coverage continued after PR #1562, including job
  lifecycle API follow-up work.
- FastVLM advisory captioning gained runtime guidance, managed portal readiness
  and diagnostics UX, and subprocess-isolated runtime validation surfaces.
- Portal/frontdoor work added modularized portal assets, production CSS layer
  governance, portal utility ownership checks, and browser-validation surfaces.
- Portal RUM lineage shipped from landing/login coverage through login and
  logout submit mirrors, including the PR #1716 approved attempt/success-only
  logout client mirror boundary.
- Telemetry policy work landed privacy approval, raw-log retention/deletion
  evidence, sink-path governance, independent front-door rollout controls,
  cohort-bucketing evaluation, and modernization evidence updates.
- Dependency maintenance landed #1718 (`urllib3` 2.7.0) and #1720 (Next.js
  16.2.6).
- PR #1721 converted the stale UX/UI rebaseline into a current portal UX/UI
  status snapshot while preserving the existing file path for link stability.

## Current Navigation

Current documentation entry points are:

- `README.md`
- `docs/README.md`
- `docs/governance/DOCUMENTATION_MAP.md`
- this refresh audit
- `docs/governance/audit/documentation-inventory-2026-05-11.csv`

Current live agent/Copilot instruction surfaces remain:

- `.github/copilot-instructions.md`
- `.github/agents/README.md`
- `.github/agents/QUICK_START_v2.md`
- `.github/agents/transformation-portal-architect.md`
- `.github/agents/portal-app-steward.md`
- `.github/agents/transformation-portal-specialist.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/architecture/agent_governance.md`
- `docs/guides/CUSTOM_AGENT_GUIDE.md`
- `docs/reference/AGENT_QUICK_REFERENCE.md`

## Stale-Risk Findings

The current docs validation gates pass before this refresh. Remaining
stale-risk is intentionally bounded:

- Historical reports, archived PR records, old session notes, and project-era
  analyses may retain old dates, paths, and conclusions as point-in-time
  evidence.
- Current navigation should be updated through `docs/governance/DOCUMENTATION_MAP.md`
  first, then mirrored into `README.md` and `docs/README.md` only when an
  entry-point change is needed.
- Mixed directories should be reviewed before promotion into current navigation.
  Do not mass-edit old "Last Updated" lines just to make dates look current.

## Current Validation State

Repo state observed for this refresh:

- `python3 scripts/governance/check_docs_structure.py --all` scans 878 tracked
  files under `docs/` after this refresh.
- `make check-docs`, `make check-stale-docs`, and
  `make check-doc-heading-links` pass for this refresh.
- `.github/workflows/` contains 30 tracked workflow YAML files.
- The documentation topology policy remains enforced by
  `scripts/governance/check_docs_structure.py`.

## Validation

Required validation for this refresh:

```bash
git diff --check
make check-docs
make check-stale-docs
make check-doc-heading-links
python3 scripts/governance/check_docs_structure.py --all
make check-todo-governance
make validate-ci
make docs
make ci-quick
make check-worktree
```
