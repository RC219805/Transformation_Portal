# Follow-up: wire `make audit-pipeline-readiness` into CI

**Status:** scoped, not implemented
**Source:** `docs/governance/audit/archive-gates-2026-04-27.md` §7 #1
**Branch (when picked up):** new branch off `main`
**Estimated effort:** ~1–2 hours including a dry-run debug pass.

## Goal

Run the existing local readiness audit on every PR (and via manual
dispatch) so any contract regression in archive gates A/B/C surfaces
before merge. The harness already exists, runs against
`tests/fixtures/archive_small/` in seconds, and emits the deterministic
`tp.orchestrator.pipeline_readiness_audit.v1` JSON matrix — this work
just plugs it into CI.

## Acceptance criteria

1. New workflow `.github/workflows/audit_pipeline_readiness.yml`:
   - Triggers: `pull_request` and `push` to `main` with `paths:`
     filtering, plus `workflow_dispatch`.
   - Runs `make audit-pipeline-readiness` (or the script directly with
     `--json-output`) and fails on non-zero exit.
   - Uploads the readiness matrix JSON as a workflow artifact named
     `archive-gate-readiness-matrix` for diagnosability.
   - Wall-clock under ~5 minutes (mostly dependency install — the
     audit itself completes in under a second on the small fixture).
2. A green run on a PR that touches one of the watched paths.
3. A red run when an intentional regression is introduced — e.g.,
   temporarily remove `manifest_jsonl` from gate B's allowed inputs in
   a sandbox branch and confirm the workflow fails.
4. Workflow listed as a required check on `main` (branch protection
   update — note this is a maintainer/admin action, not a code
   change).

## Files touched

- **New:** `.github/workflows/audit_pipeline_readiness.yml`
- **Read-only references** (no edits): `Makefile:342-344`,
  `scripts/validation/audit_pipeline_readiness.py`,
  `requirements/base.txt`, `requirements/tools-archive.txt`,
  `pyproject.toml`.

## Reuse / pattern

Model directly on
`.github/workflows/machine_mode_contract_validation.yml` — same
contract-validation shape, same Python 3.11 + pip-cache pattern, same
pinned-SHA actions:

- `actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd` (v6)
- `actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405` (v6)
- `actions/upload-artifact` (use the SHA-pinned version already in
  use elsewhere in the repo — grep `.github/workflows/` to find it).

### Path filter (recommended starting set)

```yaml
paths:
  - 'app.py'
  - 'tools/archive_governance.py'
  - 'tools/archive_prereqs.py'
  - 'src/transformation_portal/**'
  - 'scripts/validation/audit_pipeline_readiness.py'
  - 'tests/fixtures/archive_small/**'
  - 'policy/archive/**'
  - 'docs/api/ARCHIVE_MACHINE_MODE_CONTRACT.md'
  - 'docs/schemas/machine_mode/**'
  - 'requirements/base.txt'
  - 'requirements/tools-archive.txt'
  - 'pyproject.toml'
  - '.github/workflows/audit_pipeline_readiness.yml'
```

### Install step

The audit imports `app.py`, which transitively pulls in FastAPI plus
the in-tree `transformation_portal` package. Empirically, the minimum
set is:

```bash
pip install -r requirements/base.txt -r requirements/tools-archive.txt
pip install -e .
```

(`requirements-ci.txt` works too if heavier deps are tolerable —
machine-mode-contract uses it.)

### Run step

```bash
mkdir -p artifacts
python scripts/validation/audit_pipeline_readiness.py \
    --json-output artifacts/archive-gate-readiness.json
# or: make audit-pipeline-readiness
```

Then upload `artifacts/archive-gate-readiness.json` via
`actions/upload-artifact`.

## Open questions for the implementer

1. **Required check vs. advisory?** Recommend required on `main` once
   the workflow has stabilized over a handful of green PR runs.
2. **Schedule trigger?** A nightly `schedule: cron` run would catch
   environmental drift (e.g., new pip resolutions) even on quiet
   days. Optional for v1.
3. **Aggregator integration.** `docs/architecture/ci_gate_pattern.md`
   describes a CI-gate aggregator pattern; check whether this new job
   should feed `ci-quality-firewall.yml` or stand alone. Stand-alone is
   fine for v1 — aggregation is a separate concern.
4. **Cache key.** Use the same `cache-dependency-path` pattern as
   `machine_mode_contract_validation.yml` (`requirements-ci.txt` +
   `pyproject.toml`); update if the install step diverges.

## Out of scope

- Persisting dated readiness matrices in the repo (deferred from the
  audit report's #2).
- Adding a non-default-rights fixture (deferred from #3).
- Any change to gate logic, the audit harness, the contract, or
  fixtures.

## Verification once implemented

- `act` or a draft PR that intentionally fails a contract check
  (e.g., introduce a bogus baseline expectation) — confirm workflow
  reports the failure and surfaces the JSON artifact.
- Confirm green run wall-clock and artifact contents match what
  `make audit-pipeline-readiness` produces locally.
- After a few clean runs, request branch-protection update to mark
  the workflow as required on `main`.
