# Change-Aware CI Quick Reference

Source-reviewed against [`build.yml`](../../.github/workflows/build.yml) on
2026-09-12. Preflight classifies changed PR paths inside the workflow; this is
separate from other workflows' own `on.paths` filters.

## Classification and execution

| Change / event | Main CI behavior |
| --- | --- |
| PR touching runtime/test/tool/setup/dependency/workflow paths | Full suite. |
| PR touching frontdoor paths, portal markup, portal asset manifest, frontdoor smoke validator, or secure-frontdoor quickstart | Full suite plus `frontdoor-contract`. |
| Other PR paths, such as ordinary docs | `lightweight` and `dependency-constraints`; heavy jobs may skip. |
| Main push or manual dispatch | Full suite plus frontdoor contract. |

The full matcher includes `src/`, `apps/`, `tests/`, `scripts/`, `tools/`,
`config/`, `requirements*`, `pyproject.toml`, `setup.py`, `conftest.py`,
`Makefile`, `mypy.ini`, `.pylintrc`, `.github/workflows/`, `.github/actions/`,
and `.github/copilot-instructions.md`. Therefore Markdown under `scripts/`,
`requirements/`, or `.github/workflows/` is not a lightweight docs-only change.
The secure-frontdoor quickstart is also an explicit runtime-boundary trigger.

`CI Gate` depends on `preflight`, `lightweight`, `dependency-constraints`,
`frontdoor-contract`, `lint`, `typecheck`, `test`, and `generate-manifest`.
Lightweight/dependency results must succeed. Full mode additionally requires
lint/typecheck/test/manifest success, and frontdoor success when requested.
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
