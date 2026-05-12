# Transformation Portal Health Check Report

**Generated:** 2026-05-12
**Repository:** `REPO_ROOT` (`/path/to/Transformation_Portal`)
**Branch:** `main` tracking `origin/main`
**HEAD:** `a3cf8030c docs(governance): refresh todo inventory baseline (#1723)`
**Working Tree:** Clean at inspection time

---

## Report Boundary

This document is a point-in-time health snapshot for audit context under
`docs/fixes/`. It is not promoted as canonical operator guidance. Current
operator navigation remains in `README.md`, `docs/README.md`, and
`docs/governance/DOCUMENTATION_MAP.md`.

The snapshot reflects local inspection on May 12, 2026. Local dependency
health and documentation topology passed. The full canonical `make ci` lane
was not run for this report, so this report must not be read as an all-green
repository certification.

---

## Overall Status: Healthy With Bounded Maintenance Warnings

The current repository is a governed image and video processing platform for
luxury real estate rendering and architectural visualization. The checked
state is substantially newer than the old 2025 RAG-branch report this file
replaced.

Current health signals:

- Local Python dependency health passes with the repo `.venv`.
- Documentation topology validation passes for the current docs layout.
- The active health/readiness API contracts are implemented and covered by
  focused tests.
- Managed frontdoor and backend validation are Make-governed and Node/Python
  version-gated.
- The repository is large, approximately 91 GiB at inspection time (`du -sh`
  reported `91G`), which is a maintenance concern but not evidence of a
  code-health failure by itself.

Current validation boundary:

- Proven green for this refresh: dependency health and docs topology checks.
- Not run for this report: full `make ci`, browser smokes, and live pipeline
  validations.
- No current evidence supports the stale critical claims about missing Python
  dependencies, Python 3.14 incompatibility, or a missing `depth_pipeline/`
  runtime path.

---

## 1. Codebase Surface

### Backend And Portal

The FastAPI backend remains rooted in `app.py` and serves the portal and
orchestrator API surface:

- `GET /portal` serves the single-file portal UI.
- `GET /portal/bootstrap` exposes standalone portal auth mode for direct
  backend debugging.
- `GET /healthz` is a minimal raw liveness probe.
- `GET /ready` is raw readiness with optional verbose fields.
- `GET /v1/readiness` is the governed pipeline readiness envelope.

The managed Next frontdoor lives under `web/secure-landing` and enforces Node
22.x through its runtime guard and package engine contract.

### Health And Readiness Contracts

Current health contracts are intentionally split by consumer:

| Route | Shape | Purpose |
| --- | --- | --- |
| `/healthz` | Raw JSON with `ok` and `time` | Minimal liveness for frontdoors, probes, and load balancers |
| `/ready` | Raw JSON with `ok`, `time`, `version`, and optional verbose fields | Backend readiness without the orchestrator API envelope |
| `/v1/readiness` | `tp.orchestrator.readiness.v1` API envelope | Pipeline readiness for `lux-depth-v3` and archive gates A/B/C |

The raw `/healthz` and `/ready` shapes are contract-sensitive. Do not wrap
them in the versioned API envelope unless the external probe contract is
explicitly changed and tests are updated in the same patch.

### Current Major Subsystems

The current repo contains active surfaces for:

- Lux Depth V3 orchestration, depth backends, Materials V3 segmentation, PBR
  outputs, APEX validation, and run-card governance.
- Archive gates A/B/C with machine-mode and readiness coverage.
- Spatial AI reconstruction, segmentation, materials, and ingest contracts.
- Optional FastVLM advisory captioning through subprocess-isolated local
  runtime paths; captions are advisory and not quality-gate evidence.
- Managed frontdoor auth, local smoke credentials, session scaling, RUM
  telemetry controls, retry-after UI, and portal asset governance.
- Layered dependency lock governance, including generic Python locks and the
  target-owned Darwin arm64 ML lock lane.

---

## 2. Local Environment Snapshot

Observed local environment:

| Component | Observed State |
| --- | --- |
| Python | `.venv/bin/python` reports `Python 3.12.13` |
| Node | `node --version` reports `v22.22.2` |
| Python dependency health | `.venv/bin/python -m pip check` passed |
| Frontdoor Node contract | `web/secure-landing` declares `>=22 <23` |

Observed installed package examples:

| Package | Version |
| --- | --- |
| pytest | 9.0.3 |
| hypothesis | 6.152.1 |
| Pillow | 12.2.0 |
| tifffile | 2026.3.3 |
| fastapi | 0.136.1 |
| starlette | 1.0.0 |
| uvicorn | 0.46.0 |

These observations replace the stale 2025 claims that tests were blocked by
missing `hypothesis`, `Pillow`, `tifffile`, `tqdm`, `typer`, or `PyYAML`.

---

## 3. Test And Validation Surface

Current observed scale:

- About 606 Python test files under `tests/`.
- 30 GitHub workflow YAML files under `.github/workflows/`.
- 36 YAML/JSON config files under `config/`.

Important local validation targets:

```bash
make check-environment
make test-fast
make test-orchestrator-contract
make test-frontdoor-contract
make validate-portal-browser
make validate-frontdoor-browser
make ci
```

The canonical `make ci` target currently chains lint/governance checks,
dependency-lock checks, portal asset budget checks, fast tests, orchestrator
contracts, and managed frontdoor contracts. It was not run for this report.

Live browser and pipeline validations can be slower and environment-sensitive.
Classify failures by source before weakening product contracts:

- Product logic regression: deterministic contract or smoke failure in the
  current code.
- Stale test logic: assertion no longer matches an intentional product
  contract change.
- Environment/tooling failure: missing browser runtime, occupied ports,
  unavailable optional model runtime, network restriction, or sandbox issue.

---

## 4. Dependency Governance

The repo no longer uses ad-hoc dependency installation as the primary health
fix path. Use the governed Make targets and layered lockfiles:

```bash
make install-core
make repair-core-venv
make check-environment
```

Dependency policy is enforced through the `requirements/` layered inputs and
locks, lock ownership metadata, and validation scripts such as:

```bash
make check-requirements-lock-contract
make check-dependency-pinning
make check-ci-sync
```

ML installation remains capability-specific. Do not reintroduce a broad
umbrella ML install path without a trusted checked-in lockfile contract.

---

## 5. Documentation And Repository Structure

Documentation governance is active and strict:

- `docs/README.md` is the only maintained file allowed directly under `docs/`.
- Current navigation belongs in `docs/governance/DOCUMENTATION_MAP.md`.
- Historical and point-in-time reports remain available for audit context, but
  they are not current operator guidance unless promoted by the documentation
  map.
- `docs/fixes/HEALTH_CHECK_REPORT.md` remains a point-in-time report, not a
  canonical runbook.

Observed docs validation checks for this refresh:

```bash
make check-docs
make check-doc-heading-links
python3 scripts/governance/check_docs_structure.py --all
.venv/bin/python -m pytest -q tests/test_app_healthcheck_contract.py tests/api/v1/test_health_models.py tests/validation/test_environment_preflight.py
```

All passed during refresh validation.

---

## 6. Repository Size

The checkout size was observed at approximately 91 GiB (`du -sh` reported
`91G`):

```bash
du -sh .
```

This is a maintenance warning, not a direct code-health failure. Before
deleting anything, identify generated outputs and cache directories explicitly
and preserve any user-owned artifacts. Prefer repo-governed cleanup targets:

```bash
make clean
make clean-frontdoor
make clean-all
```

Use destructive cleanup only when the target directories are confirmed
generated and the user has approved removal.

---

## 7. Recommended Operator Commands

Use repo-governed commands instead of direct `pip install` remediation:

```bash
make install-core
make repair-core-venv
make check-environment
make test-fast
make test-orchestrator-contract
make test-frontdoor-contract
make validate-portal-browser
make validate-frontdoor-browser
make ci
```

For local full-stack bring-up:

```bash
make dev-write-env
source /tmp/tp-local-http-all-on.env
make dev-start
```

For managed frontdoor browser validation, keep the backend/frontdoor contract
intact and use the governed smoke:

```bash
make validate-frontdoor-browser
```

---

## Conclusion

### Assessment: Current Local Health Snapshot Is Positive, With Full-CI Boundary

The current codebase does not match the old stale report. It is on `main`,
uses Python 3.12 and Node 22, has active dependency governance, and exposes
typed health/readiness contracts with focused test coverage.

Proven green during this refresh:

- Python dependency health via `.venv/bin/python -m pip check`.
- Documentation organization via `make check-docs`.
- Documentation heading links via `make check-doc-heading-links`.
- Documentation topology via `python3 scripts/governance/check_docs_structure.py --all`.
- Focused health/preflight tests via `.venv/bin/python -m pytest -q tests/test_app_healthcheck_contract.py tests/api/v1/test_health_models.py tests/validation/test_environment_preflight.py`.

Still not proven by this report:

- Full canonical `make ci`.
- Live backend/frontdoor browser smokes.
- Live optional ML/runtime validations.

The correct next step for release or merge confidence is to run the applicable
governed validation lane, not to follow the obsolete ad-hoc dependency install
steps from the prior version of this report.
