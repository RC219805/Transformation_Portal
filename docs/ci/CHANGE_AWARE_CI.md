# Change-Aware CI Quick Reference

Source-reviewed against [`build.yml`](../../.github/workflows/build.yml) on
2026-11-15. Preflight classifies changed PR paths inside the workflow; this is
separate from other workflows' own `on.paths` filters.

## Classification and execution

| Change / event | Main CI behavior |
| --- | --- |
| PR touching runtime/test/tool/setup/dependency/workflow paths | Full suite. |
| PR touching frontdoor/Worker/shared-token paths, public assets, portal markup/assets config, root Worker deployment manifests and `.nvmrc`, `build.yml`, frontdoor smoke/CSS parity validators and fixtures, or secure-frontdoor quickstart | Full suite plus `frontdoor-contract`. |
| Empty, incomplete, or capped PR file inventory | Full suite plus frontdoor contract. |
| Other PR paths, such as ordinary docs | `lightweight` and `dependency-constraints`; heavy jobs may skip. |
| Main push or manual dispatch | Full suite plus frontdoor contract. |

The full matcher includes `src/`, `apps/`, `tests/`, `scripts/`, `tools/`,
`config/`, `migrations/`, `policy/`, `schemas/`, the contract schemas under
`docs/schemas/`, `docs/archive/schemas/`, and `docs/compliance/schemas/`, all root
Python entrypoints (including `app.py`), `requirements*`, `pyproject.toml`,
`Makefile`, `mypy.ini`, `.pylintrc`, `.pre-commit-config.yaml`, `Dockerfile`,
`docker-compose.yml`, `.github/agents/`, `.github/workflows/`, `.github/actions/`,
`.github/dependabot.yml`, and `.github/copilot-instructions.md`. Therefore Markdown under these matched
directories is not a lightweight docs-only change. The secure-frontdoor
quickstart is also an explicit runtime-boundary trigger.

Renames classify both the current and previous filename. The paginated PR file
inventory must match the event's changed-file count; an empty list, count
mismatch, or the GitHub API's 3,000-file cap requests all gates. This preserves
lightweight validation only when the inventory supports that decision.

Enforcement's six unconditional checks start independently of its change
classifier; only ML and golden jobs wait for the corresponding classifier
outputs. The classifier alone requests read access to pull requests. The
documentation workflow cancels superseded builds for the same event and ref.
Job names and required-check composition remain unchanged. Layer 1 retains
`not ml and not slow and not benchmark`; the separately named Golden gate uses
`golden and not ml and not slow and not benchmark`. Empty Golden selections
fail rather than silently passing.

Pip cache identities include the transitive base lock reached through
`requirements-ci.txt` and `requirements.txt`. Core test caches also track the
archive-tool lock; ML caches track the installer, raw requirements, and package
extras. The build matrix separates cache publication by Python version and
test tier. Cache changes do not alter installed dependency policy or firewall
checkout trust.

The existing Frontdoor job verifies the declared npm 11.12.1 owner, installs
all three consumed Node lock closures, and runs Worker request, TypeScript,
parity and root/Worker dry-run checks before its existing frontend/browser
checks. The 15-minute cap, routes, selectors and authentication contracts stay
unchanged. Cache identities include every consumed owner lock.

`CI Gate` depends on `preflight`, `lightweight`, `dependency-constraints`,
`frontdoor-contract`, `docker-cpu-runtime`, `lint`, `typecheck`, `test`, and
`generate-manifest`.
Lightweight/dependency results must succeed. Full mode additionally requires
lint/typecheck/test/manifest success, and frontdoor success when requested.
The classified Docker runtime result is checked in every gate mode; an
unexpected skip or failure cannot pass through a lightweight result.
Inspect the preflight reason and earliest failing upstream job before diagnosing
a downstream skip or gate failure. No universal time saving is guaranteed.

## Manual full validation

```bash
gh workflow run build.yml --ref main
```

For post-CI firewall verification, dispatch `build.yml` on `main` or `develop`.
The firewall accepts only a successful same-repository push/manual upstream
run and checks out its exact SHA; it remains `workflow_run`-only.

## Local configuration checks

```bash
make validate-ci
```

Workflow pass/fail and remotely enforced required checks are separate. See
[branch-protection snapshot and verification](BRANCH_PROTECTION_SETUP.md),
[complete workflow inventory](WORKFLOW_MATRIX.md), and
[test strategy](../testing/STRATEGY.md). Documentation CI does not exercise every
runtime, service, model, or production deployment.
