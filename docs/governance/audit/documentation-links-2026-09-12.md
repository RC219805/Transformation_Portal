# Documentation Link Backlog — September 12, 2026

Baseline: `faa3758fc3b5180bccf264c9a1b1d1b5ef927c52`. Local target scans cover
maintained/classified support and changed Markdown, skipping fenced and indented
code blocks. Explicit changed/canonical inputs also go to the repository heading
checker. Scans do not verify external URLs, reference-style links, HTML links,
RST roles or dynamic/generated anchors. The repository heading checker has its
own Markdown slug rules and does not exclude every code example.

## Unresolved Original Targets

| Source | Missing original target | Investigation / acceptance criterion |
| --- | --- | --- |
| `docs/apex/README_GOVERNANCE.md` (two prose links) | `EXECUTIVE_SUMMARY.md` in that directory | Tracked `docs/performance/EXECUTIVE_SUMMARY.md` concerns the February depth-cache regression and is unrelated to this APEX governance plan. Exact-path history did not establish the original target. Recover original APEX summary before relinking. |
| `docs/architecture/TRANCHE_PHASE2_QUICKREF.md` | `TRANCHE_PHASE2_SUMMARY.md` | No tracked target or exact-name match in available `git log --all`. Obtain original summary/commit from the tranche owner. |
| `docs/performance/performance_ledger_v1.7_migration.md` | `ADR-023-performance-ledger.md` | No matching original tracked/history target. Other ADR-023 documents cannot be substituted solely by number. Recover intended decision record. |
| `tools/investigations/materials_v3/README.md` | `docs/project-status/PR934_EXTRACTION_STRATEGY.md` | No tracked target or exact-name history match. Recover the extraction plan from its author/PR evidence. |
| `docs/ci/APEX_RESEARCH_WORKFLOW_REPORT_20260207.md` | `ADR-019_IMPLEMENTATION_SUMMARY.md` | Additional historical report reference found during the expanded scan; no exact tracked/history target. Recover original summary before repairing. |

The six link occurrences above remain explicit backlog. Code-fence mentions of
the same filenames are examples and are counted separately, not as prose links.

## Confirmed Identity Repairs

APEX references to `ADR-026-APEX-governance-framework.md` now use its archived
copy under `docs/_archive/2026-Q1-consolidation/`. History records the original
APEX file in commit `08e78afea` and archived path in `fc12cb443`; the unrelated
current ADR-026 was not substituted. Other repaired targets retain exact
filenames and verified relocation history (including `b7bf9033b` architecture
triage and `d8b23ab3f` docs topology moves).

Raw scanner results and history commands are saved with the local validation
logs. Remaining inventory-only historical links are not certified by this scan.
