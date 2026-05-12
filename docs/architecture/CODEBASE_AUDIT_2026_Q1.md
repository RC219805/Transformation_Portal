# Transformation Portal Codebase Audit

**Original audit window:** 2026 Q1
**Current refresh:** 2026-05-12
**Repository baseline inspected:** `main` at `a3cf8030c docs(governance): refresh todo inventory baseline (#1723)`
**Document classification:** current-support
**Scope:** Refresh the Q1 audit against the current codebase, documentation
baseline, and validation contracts without changing runtime behavior.

---

## Executive Summary

Transformation Portal is now best described as a governed image and video
processing platform for luxury real estate rendering, architectural
visualization, and editorial finishing. The March 2026 Q1 audit correctly
identified strong foundations in security, dependency governance, test
infrastructure, and performance discipline, but several facts in the original
scorecard are now stale.

Current posture as of this refresh:

- The browser-facing architecture is `Browser -> managed Next frontdoor ->
  FastAPI backend`. A Cloudflare Worker, when present, remains frontdoor-only
  and must not proxy directly to FastAPI.
- Health/readiness contracts are active governed surfaces: `/healthz`,
  `/ready`, and `/v1/readiness` preserve distinct wire shapes and test
  coverage.
- Documentation governance has moved from ad hoc cleanup to a May 11, 2026
  repo-wide documentation baseline and inventory.
- Testing and CI are substantially larger than the Q1 snapshot: roughly 606
  Python test files and 30 workflow YAML files were observed during this
  refresh.
- Architecture decomposition has progressed, but the repository still contains
  large integration files, especially `app.py` and `lux_depth_v3/orchestrator.py`.
- Full `make ci` was not run for this document refresh; conclusions below are
  based on static inspection plus documentation validation.

## Current Directional Scorecard

The old numerical scores are retained only as Q1 context. Current status should
be read directionally unless a specific validation command is listed as proven
green.

| Area | Current posture | Notes |
| --- | --- | --- |
| Security | Strong | Fail-closed path/auth posture, portal RUM controls, dependency scanning, and model/runtime governance are active. |
| Testing | Strong, broad | Test surface has grown to about 606 Python test files with contract, smoke, governance, security, and fixture coverage. |
| Dependency governance | Strong | Layered locks, ownership checks, retired ML lane stubs, and governed optional runtime installers remain central. |
| Documentation | Improved, governed | May 11 repo-wide refresh classified current, support, mixed, historical, and archive-only docs. |
| CI/CD | Mature, complex | 30 workflow YAML files and Make validation lanes cover CI, security, dependency, docs, portal, and pipeline checks. |
| Code architecture | Mixed | Decomposition modules exist, but large integration files still require careful ownership and focused refactors. |
| Performance governance | Strong but lane-specific | Quality firewall, benchmark isolation, APEX, ledger, and determinism surfaces remain active; full live performance status requires lane-specific runs. |

## Current System Boundaries

### Frontdoor And Backend

The current runtime topology is:

```text
Browser
  -> managed Next frontdoor (`web/secure-landing`)
  -> FastAPI backend (`app.py`)
  -> orchestrator jobs, pipeline CLIs, local runtimes, and artifacts
```

The frontdoor owns browser login/logout, Cloudflare Access posture, local smoke
credentials, session handling, proxy behavior, frontdoor `/healthz`, RUM
rollout controls, and portal bundle generation. Node 22.x is the enforced
frontdoor runtime contract.

The FastAPI backend owns direct-debug portal behavior, typed API envelopes,
pipeline readiness, job lifecycle, server-sent events, artifact serving, and
protected API-key routes.

### Health And Readiness

Current health contracts are intentionally split:

| Route | Contract |
| --- | --- |
| `/healthz` | Raw minimal liveness JSON with `ok` and `time`; not API-enveloped. |
| `/ready` | Raw backend readiness JSON with `ok`, `time`, `version`, and optional verbose fields. |
| `/v1/readiness` | `tp.orchestrator.readiness.v1` envelope with per-pipeline dispatch truth. |

The readiness matrix currently covers `lux-depth-v3` and archive gates A/B/C.
Transport success is not sufficient to prove dispatch readiness; per-pipeline
state can be `ready`, `degraded`, or `blocked`.

### Pipeline And Artifact Surfaces

Current governed surfaces include:

- Lux Depth V3 depth, materials, PBR, enhancement, run-card, and artifact
  workflows.
- APEX quality and model-family characterization surfaces.
- Archive gates A/B/C and machine-mode archive contracts.
- Spatial AI reconstruction, segmentation, materials, and ingest research
  surfaces.
- Optional FastVLM advisory captioning through subprocess-isolated runtime
  checks; captions remain advisory and are not quality-gate evidence.
- Provenance, manifests, attestation, content-addressed storage, run cards, and
  fixity evidence.

## Findings By Area

### 1. Security And Governance

Current strengths:

- Backend protected endpoints enforce API key policy when enabled.
- Trusted-host, proxy, content-length, rate-limit, artifact-path, and allowed
  root boundaries remain fail-closed surfaces.
- Frontdoor session handling and Cloudflare Access posture are isolated from
  direct backend debugging.
- RUM telemetry has explicit allow-lists, rollout controls, retention/deletion
  evidence, and sink-path governance.
- Dependency security is governed through lockfiles, banned dependency checks,
  pip-audit/security workflows, Bandit, CodeQL, and gitleaks.

Current risks:

- Security-sensitive code is spread across backend middleware, frontdoor
  routing/session helpers, validation scripts, and pipeline path guards. Keep
  future fixes narrow and contract-tested.
- Optional local model runtimes introduce filesystem and download boundaries;
  keep them under governed setup scripts and `.runtime/` contracts.

### 2. Testing

Current strengths:

- The repository now has about 606 Python test files.
- Coverage includes unit, contract, HTTP, browser-smoke script tests, security,
  dependency governance, docs governance, ingest, archive, APEX, Materials V3,
  spatial AI, and runtime fixtures.
- `pytest-rerunfailures` and `pytest-xdist` are present in the governed dev
  dependency layer.
- Frontdoor has Node test, build, CSS, utility-ownership, and Playwright smoke
  surfaces.

Current risks:

- Broad test volume increases triage cost. Failures should be grouped by root
  cause: product regression, stale test contract, or environment/tooling.
- Browser and optional-runtime validations can fail due to local Chrome,
  sandbox, port, model, or network state. Do not weaken product contracts until
  the failure class is proven.

Primary validation lanes:

```bash
make test-fast
make test-orchestrator-contract
make test-frontdoor-contract
make validate-portal-browser
make validate-frontdoor-browser
make ci
```

### 3. Dependency Management

Current strengths:

- `pyproject.toml` keeps broad package ranges while checked-in lockfiles enforce
  tested combinations.
- `requirements/` owns generic, dev, CI, security, tools-archive, and target
  ML lock surfaces.
- Darwin arm64 ML is the target-owned ML lane; Linux x86_64 and Darwin x86_64
  ML lanes are retired fail-closed stubs.
- Optional DA3, Depth Pro, RAW, SAM2, CoreML, and FastVLM runtimes have
  explicit setup or validation commands.

Current risks:

- Direct `pip install ...` guidance is not the supported remediation path for
  normal development health. Prefer Make targets and lock-governed installers.
- Dependency updates must preserve marker contracts, lock ownership, and
  current CI sync checks.

Primary commands:

```bash
make install-core
make repair-core-venv
make check-environment
make check-requirements-lock-contract
make check-dependency-pinning
make check-ci-sync
```

### 4. Documentation

Current strengths:

- The May 11 documentation refresh established current navigation and a
  repo-wide classification inventory.
- `docs/governance/DOCUMENTATION_MAP.md` is the current source of truth for
  maintained navigation.
- `docs/architecture/ARCHITECTURE.md` now reflects the current system topology
  and contract boundaries.
- Documentation topology is enforced by `scripts/governance/check_docs_structure.py`.

Current risks:

- Historical and mixed docs intentionally retain old dates, commands, and
  conclusions. Do not promote them as current guidance without checking the
  documentation map.
- Point-in-time reports should be refreshed only when they are classified as
  current-support or linked by current navigation.

Primary commands:

```bash
make check-docs
make check-stale-docs
make check-doc-heading-links
python3 scripts/governance/check_docs_structure.py --all
```

### 5. CI/CD

Current strengths:

- `.github/workflows/` contains 30 workflow YAML files.
- The local `make ci` target chains lint, serialization, YAML governance,
  pip-tools cache, requirements lock contract, dependency pinning, CI sync,
  portal asset budgets, fast tests, orchestrator contracts, and frontdoor
  contracts.
- CI includes security, dependency, documentation, ML, frontdoor, archive,
  determinism, APEX, and quality firewall surfaces.

Current risks:

- Workflow count and optional lanes make triage noisy. Keep suspected
  cancellations separate from confirmed root-cause failures.
- Scheduled/main failures can be stale relative to HEAD. Verify current source
  files and local validators before reopening already-fixed contract logic.

### 6. Code Architecture

Current strengths:

- Package boundaries are clearer around `api/v1`, `core`, `lux_depth_v3`,
  `depth`, `stage_graph`, `execution_graph`, `runtime`, `storage`,
  `spatial_ai`, `ingest`, `attestation`, `schemas`, and frontdoor code.
- ADR-043 introduced concrete decomposition targets and supporting
  `lux_depth_v3` modules such as `config_resolver.py`,
  `pipeline_coordinator.py`, `artifact_manager.py`, and `execution_engine.py`.
- Typed API models and schema envelopes reduce ambiguity at the route boundary.

Current risks:

- `app.py` is still a large integration module, observed at about 8.9K lines.
- `src/transformation_portal/lux_depth_v3/orchestrator.py` remains large,
  observed at about 6.9K lines, even with extracted supporting modules.
- Architecture work should continue as incremental, contract-preserving
  slices. Avoid broad rewrites of orchestrator, portal, or frontdoor surfaces.

Recommended refactor posture:

- Extract only when it reduces real coupling or isolates a testable contract.
- Keep public routes, schema names, CLI flags, selectors, and artifact paths
  stable unless the change explicitly updates them.
- Pair every behavior change with focused tests and the correct validation
  lane.

### 7. Performance And Determinism

Current strengths:

- APEX, quality firewall, performance ledger, benchmark markers, and
  determinism harnesses remain active governance surfaces.
- Run cards, manifests, provenance, content-addressed storage, and fixity
  checks support repeatable artifact review.

Current risks:

- Live performance status is lane-specific. Static docs refreshes should not
  claim benchmark health unless the relevant benchmark or APEX lane was run.
- Optional ML backends and local model runtimes can shift performance and
  availability; readiness should report those states explicitly.

## Updated Roadmap

| Priority | Workstream | Current recommendation |
| --- | --- | --- |
| P0 | Preserve contracts | Keep `/healthz`, `/ready`, `/v1/readiness`, job routes, frontdoor selectors, CLI flags, and schema names stable unless a change intentionally updates them with tests. |
| P1 | Architecture debt | Continue incremental `app.py` and `lux_depth_v3/orchestrator.py` decomposition around already-extracted modules. |
| P1 | Frontdoor validation | Keep Node 22, frontdoor health, browser smoke selectors, and portal bundle generation deterministic. |
| P1 | Dependency governance | Preserve lock ownership, marker contracts, retired ML lane stubs, and optional runtime setup boundaries. |
| P2 | Documentation hygiene | Maintain current navigation through the documentation map; classify historical material instead of mass-updating dates. |
| P2 | CI triage | Group failures by current root cause and distinguish stale scheduled failures from HEAD regressions. |
| P2 | Performance evidence | Treat benchmark/APEX/performance claims as current only after lane-specific validation. |

## Validation For This Refresh

This document refresh is documentation-only. The following commands should be
used to validate the edit:

```bash
make check-docs
make check-stale-docs
make check-doc-heading-links
python3 scripts/governance/check_docs_structure.py --all
git diff --check
```

Full product validation remains outside this refresh unless explicitly run:

```bash
make ci
make validate-portal-browser
make validate-frontdoor-browser
```

## Conclusion

The Q1 audit remains useful historical context, but the current codebase has
moved beyond several March assumptions. The repository now has a more explicit
managed-frontdoor topology, typed health/readiness contracts, broader docs and
CI governance, more test coverage, and stronger telemetry/dependency policy.

The main remaining engineering risk is not lack of governance; it is the size
and coupling of central integration surfaces. Future work should preserve the
current contracts while continuing narrow decomposition of `app.py`,
`lux_depth_v3/orchestrator.py`, and adjacent runtime/portal seams.
