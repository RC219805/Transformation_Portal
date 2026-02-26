# Documentation Retention Policy

This policy defines where documentation belongs and how long it should be retained.

## Canonical Structure

- `docs/architecture/adr/`
- `docs/governance/`
- `docs/deployment/`
- `docs/cli/`
- `docs/development/`
- `docs/historical/`
- `docs/pr_archive/`

## Classification Rules

| Type | Location | Retention |
| --- | --- | --- |
| ADR | `docs/architecture/adr/` | Permanent |
| CLI reference | `docs/cli/` | Permanent |
| Deployment guide | `docs/deployment/` | Maintained |
| PR-specific | `docs/pr_archive/` | Historical |
| Execution logs | `docs/historical/` | Historical |
| Session notes | `docs/historical/` | Historical |
| Status reports | Forbidden outside `docs/historical/` | Historical |

## Deterministic Placement Rules

- Any document tied to a specific PR, commit hash, merge event, or fix rollout belongs in `docs/pr_archive/`.
- Any point-in-time execution output (for example push summaries, completion reports, session reports) belongs in `docs/historical/`.
- Filenames containing `SUMMARY`, `REPORT`, `COMPLETE`, or `STATUS` are not allowed outside `docs/historical/` and `docs/pr_archive/`.

## Enforcement

- CI enforces naming/placement with `scripts/governance/check_docs_structure.py`.
- The governance check validates changed docs files during CI so new sprawl cannot be introduced.
