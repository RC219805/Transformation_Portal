# Portal UX/UI plan - current status snapshot (verified 2026-05-12)

> File path retained for documentation-link stability. This content supersedes the
> 2026-04-08 UX/UI rebaseline. Runtime UX/frontdoor facts still reflect #1720;
> the repo documentation baseline was rechecked through `01dc1d816` / #1728.

## Snapshot metadata

- `main` / `origin/main`: `01dc1d816`
- Last UX/UI-relevant merged PR: #1716
- Latest operational dependency maintenance: #1718 urllib3, #1720 Next.js
- Latest repo documentation baseline: #1728
- Purpose: planning context only; this document does not choose the next
  implementation PR.

## Context

The previous version of this plan was written when #1681 was the latest relevant
merge. Since then, the Phase-1 front-door, portal telemetry, privacy-policy,
logout UX, governed smoke, and logout client mirror work have shipped.

The old "queued for Phase-1" list is now empty in practice. The remaining items
are either candidate Phase-2 implementation work or operational evidence/decision
work.

This snapshot is a status reference for the next PR-driver. It does not
recommend an implementation order or select a specific next PR.

## What's shipped since PR #1681

### Portal RUM telemetry lineage

| PR | Title |
| --- | --- |
| #1682 | RUM coverage on landing + login |
| #1684 | server-side `login_submit_*` events |
| #1689 | client `login_submit_attempt` mirror |
| #1694 | client `login_submit_failure` mirror |
| #1695 | client `login_submit_success` mirror |
| #1696 | server-side `logout_submit_*` events |
| #1697 | marker-cookie helpers DRY refactor |
| #1716 | client logout RUM mirror |

### Front-door + portal UX

| PR | Title |
| --- | --- |
| #1683 | document design tokens |
| #1686 | retry-after countdown UI |
| #1690 | front-door Playwright smoke fixtures |
| #1692 | portal Playwright `@portal-browser` suite |
| #1693 | managed smoke follow-up / fix |
| #1711 | managed portal Sign out button |
| #1713 | governed logout click-flow smoke |

### Telemetry policy package

| PR | Title |
| --- | --- |
| #1698 | refresh portal telemetry privacy sign-off packet |
| #1700 | record portal telemetry approval |
| #1705 | automate raw-log retention/deletion evidence |
| #1706 | enforce telemetry sink-path policy |
| #1707 | separate front-door RUM rollout controls |
| #1708 | evaluate cohort bucketing |
| #1710 | backfill `TP_FRONTDOOR_RUM_*` evidence gates |
| #1712 | refresh modernization evidence status |

### Operational / dependency maintenance

| PR | Title |
| --- | --- |
| #1718 | urllib3 2.7.0 security bump |
| #1720 | Next.js 16.2.6 security bump |

These are dependency/security maintenance items, not UX/UI roadmap items. #1719
is intentionally not listed because it closed unmerged.

## Candidate implementation backlog

The table below is unordered. It is a candidate backlog, not an implementation
sequence.

| Candidate | First artifact | Why first |
| --- | --- | --- |
| IndexedDB job persistence | Design note | Requires state schema, TTL, logout clearing, reload hydration, and SSE reconciliation decisions before code. |
| Auth proxy mode | ADR | Requires auth-model, direct-debug fallback, CSRF, and backend header-injection decisions before code. |
| WebP/AVIF dynamic artifact previews | Design + API contract | Requires backend transform endpoint, manifest schema, and `<picture>` / preview integration. |
| Portal bundle reduction | Measurement note | Requires current metafile/bundle accounting before further carve/refactor work. |

## Evidence / decision work

These remain open in the modernization RFC, but they close only with real pilot
data or human decision, not with a code PR.

| Gate | Required input |
| --- | --- |
| M1 visibility - pilot capture | Operator-run pilot RUM/event evidence |
| M4 pilot metrics | Real measured performance data |
| M5 viewer evidence | Pilot data for artifact-review workflow |
| ADR-050 rewrite-vs-no-rewrite evidence | Current evidence inventory and human decision |

## Standing telemetry/privacy constraints

From the `Approved with Conditions` block in
`docs/compliance/PORTAL_TELEMETRY_PRIVACY_SIGNOFF.md`:

1. Each pilot needs a named owner and explicit end date.
2. Raw JSONL paths must be outside the repo, outside `public/` / `static/`,
   access-restricted, and excluded from CI artifact upload.
3. Raw logs have a 14-day retention maximum after pilot end.
4. Backup retention must be documented if the host path is auto-backed-up.
5. Any new RUM event family, metadata key, marker cookie, `sessionStorage` key,
   rollout knob, sink behavior, or retention posture requires
   `PORTAL_TELEMETRY_PRIVACY_SIGNOFF.md` to be revised in the same PR before
   rollout expansion.

#1716 shipped only the approved attempt/success logout client mirror. Any later
expansion beyond that envelope, including client-side logout failure telemetry,
is a separate telemetry/privacy decision.

## Branch naming note

This snapshot does not establish a repo-wide branch naming convention. For
normal contributor work, follow the feature-branch guidance in
[CONTRIBUTING.md](../../CONTRIBUTING.md). For automation-assisted follow-ups,
use the branch prefix required by the active workflow, runbook, or existing PR
thread, and keep that choice local to the scoped work item.

## Critical files

### Governance

| Path |
| --- |
| `docs/compliance/PORTAL_TELEMETRY_PRIVACY_SIGNOFF.md` |
| `docs/architecture/PORTAL_OPERATOR_CONSOLE_MODERNIZATION_EVIDENCE.md` |
| `docs/decisions/ADR-050-portal-react-migration.md` |

### Runtime / front-door

| Path |
| --- |
| `web/secure-landing/lib/rum-client.js` |
| `web/secure-landing/lib/rum-emitter.js` |
| `web/secure-landing/portal-src/portal.template.js` |
| `portal.html` |

### Validation

| Path |
| --- |
| `scripts/validation/validate_frontdoor_browser_smoke.py` |
| `tests/validation/test_portal_smoke_scripts.py` |
| `config/portal_asset_budgets.json` |

## Freshness verification

Before using this snapshot for planning:

```bash
git fetch --prune origin
git log --oneline 01dc1d816..origin/main
.venv/bin/python scripts/validation/check_portal_asset_budgets.py
head -10 docs/compliance/PORTAL_TELEMETRY_PRIVACY_SIGNOFF.md
grep -A2 "## Remaining Open Gates" docs/architecture/PORTAL_OPERATOR_CONSOLE_MODERNIZATION_EVIDENCE.md
tail -25 docs/decisions/ADR-050-portal-react-migration.md
```

If any command shows material drift, regenerate this snapshot before using it
for PR planning.

## Out of scope

- Recommending a specific next implementation PR.
- Operational/pilot-data work that requires operator action.
- ML pipeline, ingest contract, archive-gate, or dependency-maintenance work.
- Runtime, telemetry, package, lockfile, asset-budget, or validation-script
  changes.
