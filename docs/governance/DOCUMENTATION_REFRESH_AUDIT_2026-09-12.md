# Documentation Refresh Audit — September 12, 2026

Execution environment: the local Mac, Darwin arm64. Base commit:
`faa3758fc3b5180bccf264c9a1b1d1b5ef927c52` from freshly fetched `origin/main`.
Earlier workspace counts, patches, and validation reports were not used as
implementation evidence. Prior dated inventories are preserved.

## Scope And Review Depth

The [inventory](audit/documentation-inventory-2026-09-12.csv) includes every
tracked file under `docs/` plus tracked Markdown, reStructuredText, and MDX
elsewhere, including root guides, nested READMEs, and live agent instructions.
Generated build output, downloaded models, vendored/dependency directories,
and dependency lockfiles are excluded. Maintained generated documentation
(the TODO snapshot and design-token reference) remains in scope.

Classification is navigation/retention evidence, not paragraph verification.
The CSV distinguishes source-reviewed changes, navigation-only review,
historical-context review, generated snapshots, inventory-only classification,
and preserved historical evidence. `current_owner` names a maintenance area
from the actual path where no named owner is established; it does not assign
new people or privileges.

The resulting inventory has **1,028 files**: 65 canonical, 287 current-support,
192 mixed, 433 historical, and 51 archive-only. Review depth is recorded as
55 source-reviewed changes, 7 source-reviewed without changes, 41 navigation-only
reviews, 26 historical-context reviews, 3 generated snapshots, 438 inventory-only
classifications, and 458 preserved historical records. These categories describe
the recorded review work; they do not certify every paragraph in the inventory.

## Implementation Boundaries

Public setup, CLI/API examples, optional runtime installation, CI/test lanes,
agent guidance, execution authority, and operation/example boundaries were
checked against their current implementation. Source references and review
status are recorded per file in the inventory.

The reference machine-mode parser now handles top-level command errors before
routing command data. ComfyUI save/load now retains explicit graph edges and
resolves known numeric output slots using the source node contract. These are
separate source changes with subprocess and deterministic graph regressions.
Diagnostic help changes are separately reviewable and do not install runtimes.
An additional bounded heading-checker repair removes periods when deriving
GitHub heading anchors; regression tests cover numbered/punctuated headings and
duplicate normalized anchors. The valid original ADR-049 link is preserved.

Canonical `tp.execution.plan.v1` preparation and Lux consumption are implemented;
ADR-051 executor/publication activation gates remain in force. No optional
runtime, model checkpoint, managed service, credential, license acknowledgment,
remote setting, agent privilege, or dependency lock was changed by this refresh.

## Findings And Evidence

- [Finding ledger and acceptance criteria](audit/documentation-findings-2026-09-12.md)
- [Validation ledger and reproduction commands](audit/documentation-validation-2026-09-12.md)
- [Local-link backlog and scanner limits](audit/documentation-links-2026-09-12.md)
- [Prior May 11 audit](DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md)
- [Current documentation map](DOCUMENTATION_MAP.md)

Raw command logs, environment records, original checkout backup, patches, and
reviewer notes are saved outside tracked source under
`/private/tmp/tp-docs-refresh-20260912-handoff/`. The final local handoff records
commit/tree identities. Documentation and contract checks do not establish
whole-repository or production readiness.
