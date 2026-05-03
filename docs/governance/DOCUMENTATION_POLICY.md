# Documentation Retention Policy

This policy defines where documentation belongs and how long it should be retained.

## Current vs Historical Guidance

Current operator guidance is discoverable from `README.md`, `docs/README.md`,
and `docs/governance/DOCUMENTATION_MAP.md`. A document outside those navigation
paths may still be useful evidence, but it is not current guidance unless the
document itself states that it is maintained and a current index links to it.

Historical documents may retain old dates, point-in-time status, former command
examples, and superseded project conclusions. They must not be linked from
current navigation as live runbooks. When a historical document is easy to
mistake for current guidance, add a short banner or directory-level README that
points readers back to the documentation map.

## Strict Topology Contract

- `docs/README.md` is the only allowed file directly under `docs/`.
- Every other file under `docs/` must be placed in an explicitly approved top-level directory.
- Creating a new top-level directory under `docs/` requires updating both this policy and `scripts/governance/check_docs_structure.py`.

Approved top-level directories:

- `docs/750_picacho/`
- `docs/_archive/`
- `docs/analysis/`
- `docs/apex/`
- `docs/api/`
- `docs/architecture/`
- `docs/archive/`
- `docs/brand/`
- `docs/ci/`
- `docs/ci_cd/`
- `docs/cli/`
- `docs/compliance/`
- `docs/contracts/`
- `docs/decisions/`
- `docs/deliverables/`
- `docs/deployment/`
- `docs/deprecation/`
- `docs/depth_model/`
- `docs/depth_pipeline/`
- `docs/development/`
- `docs/examples/`
- `docs/fixes/`
- `docs/governance/`
- `docs/guides/`
- `docs/historical/`
- `docs/implementation/`
- `docs/implementation_notes/`
- `docs/incidents/`
- `docs/investigations/`
- `docs/materials/`
- `docs/migration/`
- `docs/operations/`
- `docs/optimization/`
- `docs/performance/`
- `docs/pipeline/`
- `docs/pipeline_docs/`
- `docs/pr_archive/`
- `docs/pr_summaries/`
- `docs/processing/`
- `docs/project-status/`
- `docs/projects/`
- `docs/quality_analysis/`
- `docs/quick_references/`
- `docs/reference/`
- `docs/reports/`
- `docs/runtimes/`
- `docs/schemas/`
- `docs/session_summaries/`
- `docs/sessions/`
- `docs/spatial_ai/`
- `docs/status/`
- `docs/summaries/`
- `docs/validation/`
- `docs/verification/`
- `docs/version_history/`
- `docs/visual_review/`
- `docs/workflow/`
- `docs/workflows/`

## Classification Rules

| Type | Location | Retention |
| --- | --- | --- |
| ADR | `docs/architecture/adr/` | Permanent |
| CLI reference | `docs/cli/` | Permanent |
| Deployment guide | `docs/deployment/` | Maintained |
| Optional local runtime guide | `docs/runtimes/` | Maintained |
| Documentation maps and repo organization specs | `docs/governance/` | Maintained |
| PR-specific | `docs/pr_archive/` | Historical |
| Execution logs | `docs/historical/` | Historical |
| Session notes and deliverable snapshots | `docs/historical/` | Historical |
| Project-specific 2025 reports | `docs/750_picacho/`, `docs/projects/`, `docs/quality_analysis/`, `docs/visual_review/` | Historical |
| Superseded depth/pipeline evaluations | `docs/depth_model/`, `docs/depth_pipeline/`, `docs/pipeline/`, `docs/pipeline_docs/` | Historical unless promoted in `DOCUMENTATION_MAP.md` |

## Deterministic Placement Rules

- Any document tied to a specific PR, commit hash, merge event, or fix rollout belongs in `docs/pr_archive/`.
- Any point-in-time execution output (for example push summaries, completion reports, session reports) belongs in `docs/historical/`.
- Documentation indexes and organization policy docs belong in `docs/governance/`.
- The `docs/` root is reserved for `README.md` only, with no exceptions.
- Do not add `START_HERE`, `PIPELINE_V1.1.0`, or other dated project-era files
  to current navigation. Point readers to the documentation map and current
  Lux Depth V3 / portal guides instead.

## Enforcement

- CI enforces structural topology with `scripts/governance/check_docs_structure.py`.
- CI runs a strict changed-doc validation plus a repo-wide docs topology scan.
- Repo-wide docs validation runs with no grandfathered exceptions.
- Changed-doc validation fails immediately for any root-level `docs/*` violation.
