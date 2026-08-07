# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Identity

Transformation Portal is a **governed** image/video processing platform for luxury real estate rendering and architectural visualization. It is **not** a generic CRUD/government/health-IT portal — do not import assumptions from those domains.

Authoritative behavioral rules live in `.github/copilot-instructions.md`, `AGENTS.md`, the live agent profiles under `.github/agents/` (`transformation-portal-architect.md`, `portal-app-steward.md`, `transformation-portal-specialist.md`), and `docs/architecture/agent_governance.md`. Read those before making non-trivial changes; the sections below summarize the parts that most often trip up automated edits.

## Common Commands

Always prefer Make targets — they encode the correct env, marker selection, and lock contract. Direct `pytest`/`pip` invocations are second choice.

### Setup
```bash
make venv                       # create/validate .venv (Python 3.11+, fail-closed)
make install-core               # pinned core runtime + dev tooling, editable install --no-deps, pip check
make repair-core-venv           # nuke/recreate .venv, reinstall pinned core, re-run pip check
make install-ml-core            # add target-owned ML baseline (Apple Silicon arm64 only; Linux/Intel macOS lanes are retired and fail closed)
make install-fastvlm-runtime    # optional FastVLM advisory captioning subprocess runtime (.runtime/fastvlm/)
make check-fastvlm-runtime      # verify FastVLM runtime + selected model roles (TP_FASTVLM_VALIDATE_MODELS=smoke,default by default)
make check-environment          # pre-flight (Python/Node/Chrome/ports/dep-health)
```
The umbrella `make install-ml` is **disabled** until a trusted umbrella ML lockfile exists. `install-ml-raw` is also fail-closed pending a trusted target-correct lockfile. Use the layered Make targets (`install-ml-core`, `install-ml-sam2`, `install-ml-coreml`) for current operator setup. Advanced Apple Silicon bootstrap-profile work can call `./scripts/bootstrap/install_ml_stack.sh --profile <core-cpu|core-mps|...>` directly. All current core profiles (`core-cpu`, `core-mps`) are Apple Silicon (`darwin-arm64`) only; `core-cuda` and the Linux/Intel macOS lanes are retired and fail closed.

### Tests
```bash
make test-fast                       # fast subset + Phase 6 smoke (default PR lane)
make test-novideo                    # full suite minus the luxury video master grader tests
make test-full                       # full suite (parallel if xdist installed)
make test-orchestrator-contract      # portal/orchestrator route + HTTP contract
make test-orchestrator-http-contract # HTTP-only orchestrator contract subset
make test-portal-contract            # portal runtime/browser contract subset
make test-frontdoor-contract         # web/secure-landing Node 22 contract/build
make test-archive-gate-contract      # archive Gates A/B/C readiness + HTTP
make test-integration                # DA3/HF live model loading (requires TP_RUN_HF_MODEL_TESTS=1, often HF_TOKEN)

# Managed-services (paid-pilot) contract gates — mockable/local unless env vars name live services:
make test-artifact-s3-contract              # ArtifactStore local + S3/moto by default; live S3 via TP_TEST_S3_URL/TP_TEST_S3_BUCKET
make test-orchestrator-postgres-contract    # JobRepository/JobEventStore against Postgres (TP_TEST_POSTGRES_URL)
make test-orchestrator-postgres-app-contract # app.py Postgres-backed job authority smoke
make test-worker-redis-contract             # QueueBroker contract against Redis (TP_TEST_REDIS_URL)
make test-frontdoor-redis-contract          # frontdoor Redis SessionStore (TP_FRONTDOOR_REDIS_URL)
make test-paid-pilot-services-contract      # full paid-pilot service-matrix gate (Postgres + Redis + S3)

# Direct pytest with marker selection:
pytest -v tests/ -ra -m "(unit or security or regression or golden or integration) and not slow" --maxfail=1
pytest -v tests/ -ra -m "ml and not slow" --maxfail=1
pytest -v tests/test_pbr_processor.py::TestName::test_method   # single test
```

Coverage targets (for governance / diff-coverage gates):
```bash
make coverage-report                 # full HTML+XML+terminal report (excludes ml/slow/benchmark)
make coverage-diff                   # diff coverage vs origin/main, 85% threshold
make coverage-fast-scope             # focused branch coverage on core/config + streaming
make coverage-package                # baseline for events/, storage/, runtime/, lux_depth_v3/, hardening/, app.py
```

### CI / Quality
```bash
make ci                    # lint + governance checks + test-fast + orchestrator + frontdoor contracts
make ci-full               # ./scripts/local_ci_check.sh (comprehensive)
make ci-quick              # ./scripts/local_ci_check.sh --quick
make lint                  # flake8 + pylint (advisory)
make lint-parity           # GitHub lint job locally with CI-pinned Python 3.12 env
make pre-commit            # pre-commit hooks with CI-aligned Black/isort
make install-hooks         # install git pre-commit and pre-push hooks
make quality-check         # lint + validate-ci + root placement check
make fix-quality / make check-quality   # auto-fix wrappers (scripts/auto_fix_quality.py)
```
`make ci` includes governance gates that often fail edits: `check-json-serialization` (no raw `json.dump(s)` outside approved modules), `check-yaml-governance` (no raw `yaml.safe_load` outside the preset loader), `check-python-headers` (PEP 263 cookies only), `check-piptools-cache`, `check-requirements-lock-contract`, `check-dependency-pinning`, `check-ci-sync`, `check-portal-asset-budgets`. Adjacent enforcement (outside `make ci`):
- `check-test-markers` (ADR-044 marker coverage) — pre-commit only.
- Tautological-test ban (`scripts/ci/check_no_tautological_tests.py`, blocks `assert True` and similar literals in `tests/`) — pre-commit + `.github/workflows/enforcement.yml`; use `# tautology-ok` for the rare intentional placeholder.
- `check-todo-governance` (`scripts/validation/scan_todo_inventory.py`) — `.github/workflows/enforcement.yml` only; flags ungoverned `TODO` / `FIXME` / `HACK` / `XXX` / `NotImplementedError` markers.
- `check-stale-docs` — `.github/workflows/build.yml` only.
- `check-doc-heading-links` — local-only `make` helper, not wired into pre-commit or CI; useful to run before pushing.

Prefer fixing the root cause over silencing these.

### Local dev stack
```bash
make dev-write-env                    # (re)writes /tmp/tp-local-http-all-on.env with TP_API_KEY/TP_BACKEND_API_KEY bound; use ./scripts/dev/write_local_env.sh --rotate to regenerate
make dev-start                        # full stack: env file → backend (with reload boundaries) → /ready wait → frontdoor; logs at /tmp/tp-{backend,frontdoor}.log
make dev-stop                         # kill listeners on dev ports (8000/3000/8001/3002) + orphan uvicorn parents
make run-backend-local                # FastAPI on 127.0.0.1:8000 with reload boundaries that exclude .runtime/output/tmp/tests/node_modules/.next; requires TP_API_KEY
make run-backend-local-noreload       # same backend without --reload (for full-stack smokes)
make run-frontdoor-local              # canonical local frontdoor on :3000 (refuses :3001 fallback)
make seed-frontdoor-user              # seeds /tmp/tp-frontdoor-users.json (smoke-admin / correct horse battery staple)
```

### Live validation (browser/HTTP smokes)
```bash
make validate-orchestrator-http       # against a running FastAPI origin
make validate-portal-browser          # spawns isolated backend, runs portal browser smoke
make validate-frontdoor-browser       # spawns backend + managed frontdoor, browser smoke
make validate-portal-lux-materials-live   # live Lux Materials V3 segmentation (EfficientSAM, optional SAM2)
make validate-portal-fastvlm-captioning-live   # live FastVLM advisory sidecar smoke (asserts used_for_quality_gate: false)
make validate-portal-css-layer-parity # production portal CSS layer contract vs post-#1592 baseline
make validate-frontdoor-deployment-gate   # manual shared-deployment frontdoor posture gate (Cloudflare + Vercel + FastAPI)
make audit-pipeline-readiness         # safe local 4-pipeline readiness audit
make check-vercel-env                 # validate Vercel/production frontdoor env vars (TP_VERCEL_ENV_FILE=..., TP_VERCEL_ENV_PRODUCTION=1 for prod-only)
```

### Dependency lockfiles
Edit the right `.in` source under `requirements/`, then regenerate. The umbrella ML and Linux/Darwin-x86_64 ML lanes are **retired and fail closed** — only Darwin arm64 ML and the generic layered locks are live.
```bash
make lock                                                  # top-level prod/ci/dev locks only (requirements*.lock.txt)
cd requirements && make compile LOCK_PYTHON_VERSION=3.11   # generic layered locks only
cd requirements && make compile-ml-darwin-arm64 LOCK_PYTHON_VERSION=3.11   # native arm64 only
python3 scripts/validation/check_requirements_lock_contract.py             # validate contract
```

### Database migrations (orchestrator durable state)
```bash
make db-upgrade                                # alembic upgrade head against TP_DATABASE_URL
make db-revision MESSAGE="<short description>" # alembic --autogenerate
```
Migrations live under `migrations/versions/` and target the `JobRepository` / `JobEventStore` Postgres schema. The Postgres backend activates when `TP_ORCHESTRATOR_STATE_BACKEND=postgres` and `TP_DATABASE_URL` is set; the memory backend is the default for local dev and core CI.

### Subprocess runtimes (DA3 / Depth Pro / RAW / FastVLM)
These run in **isolated venvs** that the orchestrator auto-discovers:
```bash
./scripts/setup/install_da3_runtime.sh         # ./.runtime/Depth-Anything-3/.venv-da3/bin/python
./scripts/setup/install_depth_pro_runtime.sh   # ./.venv-depth-pro/bin/python
./scripts/setup/install_raw_runtime.sh         # ./.venv-raw/bin/python
./scripts/setup/install_fastvlm_runtime.sh     # ./.runtime/fastvlm/.venv-fastvlm/bin/python (default models: smoke,default; --all-models also installs review)
```
Use `--da3-python`, `--depth-pro-python`, `--raw-python` flags only to override the auto-discovered runtime. Explicit `--depth-backend da3` is **strict**: failure to initialize is an actionable error, not a silent DA2 downgrade. FastVLM is enabled with `--vlm-captioning on` and is **advisory-only** — its output is never quality-gate evidence (sidecars must carry `used_for_quality_gate: false`).

## Architecture

### Two import surfaces — both are public

`src/transformation_portal/` is the main package. `src/tp/` is a **separate top-level import surface** for contract / fixity / phase tooling (`tp.crypto`, `tp.merkle`, `tp.phase4`). CI explicitly verifies both import paths in source-tree and wheel-installed contexts. **Never** collapse `tp` into `transformation_portal` or break either path.

### Lux Depth V3 is a decomposed orchestrator (do not re-monolithize)

`src/transformation_portal/lux_depth_v3/` is the flagship pipeline. Behavior is split across focused seams; new behavior should land in the right seam, not in `orchestrator.py`:
- `config_resolver.py` — preset/config normalization
- `pipeline_coordinator.py` — backend/stage resolution
- `execution_engine.py` — stage execution
- `artifact_manager.py` — output hashing/indexing/provenance assembly
- `validators/` — schema/run-card validation
- `orchestrator.py` — compatibility-facing surface (`EnhanceOrchestrator` re-export), **not** a dumping ground
- Backends: `da3_model_backend.py`, `da3_integration.py`, `coreml_backend.py`, `segmentation_backend.py` (the segmentation backend is itself split into `segmentation/` — `efficient_sam.py`, `sam2.py`, `sam_vit_h.py`, `registry.py`, `_cache.py`)
- Materials/PBR: `materials_v3*.py`, `pbr*.py`
- I/O & provenance: `io_atomic.py`, `manifest.py`, `provenance.py`, `reconstruction_manifest.py`, `run_card_contract.py`

Sibling subpackages under `src/transformation_portal/` worth knowing about: `api/v1/` (typed envelope foundation) and `api/routes/` (extracted FastAPI route seams such as `jobs.py`), `core/` (CAS DAG executor + security helpers), `events/`, `storage/`, `hardening/`, `rendering/` (4k pipeline stages and types), `spatial_ai/` (graph execution bridge, pipeline result models, segmentation cache), `attestation/`, `vlm_captioning/`, and `presence_security/` (governed presence/watermarking countermeasures with its own `presence-security` console-script CLI). Recent refactors have been extracting helpers from these areas — follow the same decomposition discipline rather than expanding existing modules.

Quality tier (`standard|premium|apex`) and `--preset` are **distinct** concepts. V2 enhancement is optional; backward-compat defaults and fail-fast validation must stay intact. Input discovery deliberately excludes derived artifacts and output dirs to prevent "depth-of-depth" loops — do not weaken this filter.

### Production hardening package (paid-pilot phase)

`src/transformation_portal/orchestrator/` is the durable-state surface for the paid-pilot roadmap (`docs/governance/PRODUCTION_HARDENING_GAP_2026-05-13.md` is the authoritative baseline; see also `docs/architecture/PORTAL_ORCHESTRATOR_ROADMAP.md`). It is a **first-class package**, not a helper bucket:

- `storage/` — `JobRepository`, `JobEventStore`, `JobRecord`, `JobEvent` Protocols (`base.py`) + `memory.py` (default) + `postgres.py` (SQLAlchemy 2.x async + Alembic). Backend selected via `TP_ORCHESTRATOR_STATE_BACKEND=memory|postgres` and `TP_DATABASE_URL`.
- `queue/` — `QueueBroker`, `JobEnqueueRequest`, `JobLease` Protocols + `memory.py` (default) + `redis.py` (server-side Lua atomicity, lease deadlines pinned to Redis TIME). Backend selected via `TP_ORCHESTRATOR_QUEUE_BACKEND=memory|redis` and `TP_REDIS_URL`.
- `artifact_store/` — `ArtifactStore` Protocol + `local.py` (filesystem, default) + `s3.py` (lazy boto3, supports MinIO/LocalStack via `TP_ARTIFACT_ENDPOINT_URL`). Backend selected via `TP_ARTIFACT_STORE=local|s3` plus `TP_ARTIFACT_BUCKET` / `TP_ARTIFACT_PREFIX` / `TP_ARTIFACT_REGION`.
- `worker.py` — in-process `WorkerRunner` pool spawned from the FastAPI lifespan; acquires leases via the broker and runs the orchestrator's `_run_job` body. **The legacy in-band `asyncio.create_task(_run_job(...))` dispatch path was removed in Phase 2.E — broker dispatch is now the only execution path.**
- `recovery.py` — `sweep_orphaned_jobs` runs on every FastAPI startup; jobs stranded as `queued|running` with no live worker handle are marked `worker_lost` (a distinct terminal state) with `error.retriable=True`.

Mirror surface on the managed frontdoor: `web/secure-landing/lib/session-store/` decomposes session persistence into a `contract.js` Protocol + `sqlite-store.js` (default, single-instance) + `redis-store.js` (multi-instance, ships `ioredis`). Backend selected via `TP_FRONTDOOR_SESSION_STORE=sqlite|redis` and `TP_FRONTDOOR_REDIS_URL`. `evaluateSessionScaling()` in `web/secure-landing/lib/session-scaling.js` is the readiness gate — `multi_instance` / `ephemeral_runtime` deployments fail closed unless `redis` is configured.

**Reuse, do not reinvent.** The roadmap explicitly designates these primitives as the integration points for Phase 1–7 paid-pilot work — extend them rather than introducing parallel abstractions. `EventStore` in `events/store.py`, the tenant primitives in `core/security/tenant.py`, and the attestation chain in `tp/phase4/` + `attestation/` are likewise already-shipped surfaces the gap doc enumerates.

### Portal HTTP surfaces

`app.py` (~10.8k lines) is the FastAPI origin. `portal.html` is the direct-debug HTML. `web/secure-landing/` is the **Node 22.x only** managed front door (Next.js) that splits the browser experience into `/`, `/login`, `/portal`. Authoritative routes:
- `GET /healthz` — managed front-door liveness
- `GET /ready` — backend liveness
- `GET /v1/readiness` — execution-readiness matrix for the four governed pipelines
- `/v1/*` and `/v2/*` — typed envelope contracts (`ApiEnvelope[T]` from `src/transformation_portal/api/v1/envelopes.py`). The route inventory contract requires `/v2/jobs/...` to mirror `/v1/jobs/...` method coverage; response schema names remain the existing `tp.orchestrator.*.v1` envelopes unless intentionally changed.
- Job lifecycle persists through `JobRepository` first; `JOBS` only carries runtime handles. Repository failures return redacted `503 JOB_REPOSITORY_UNAVAILABLE` rather than falling back to stale process cache.

Hardening that must remain intact when editing `app.py`: allowed-root path validation, API key + trusted-host enforcement, request size / concurrency / rate limits, pipeline allowlists, typed validation for archive-gate flows. **Fail closed, not open.** Recent decomposition has extracted helpers (`path_security`, `sam2_checkpoint_security`, `asset_bundle`) and route seams (`api/routes/jobs.py`) under the ADR-045 / ADR-046 / ADR-047 monolith-decomposition pattern — keep extracting along seams rather than re-monolithizing.

### Contract families (treat as binding)

| Contract | Schema/Anchor | Notes |
|---|---|---|
| Ingest | `v1.0.2` | exit-code semantics, deterministic file-derived fields, audit-grade provenance |
| Machine-mode JSON | `tp.meta.machine.v1` | stable keys: `schema`, `command`, `success`, `exit_code`, `data`, `error`. **Distinct** from evidence canonicalization. |
| Evidence / attestation | `tools/`, `tp.merkle`, `tp.phase4` | canonicalization + Merkle/signature + detached attestation; keep layered separately from machine-mode wire output |
| Run card | `transformation_portal.lux_depth_v3.run_card_contract` | governed deliverable shape |
| PBR / presets | `config/`, `lux_depth_v3.pbr_presets` | stable presets are contract-bearing; canary/experimental paths are flexible but documented |

Schemas live in `schemas/` + `docs/schemas/`. Update validators, tests, schemas, and docs **in the same change**.

### Version planes (do not collapse)

The repo intentionally maintains separate version numbers — repo/release baseline, `pyproject.toml` package version, `transformation_portal.__version__`, `lux_depth_v3.__version__`, ingest schema version, machine/evidence/attestation schema IDs, and preset/feature versions. Bump **only the plane you are changing**.

## Conventions That Bite

### Lazy imports are mandatory
Core CI and wheel-smoke paths must work **without** torch/transformers/diffusers/rawpy/coreml. Do not move heavy imports to `__init__.py` or top-level CLI/help paths. ML dependencies must degrade gracefully when absent.

### Cold-zone coverage program
`docs/testing/COLD_ZONE_COVERAGE_PROGRAM.md` defines per-package/per-file branch-coverage floors for historically under-tested modules (`events/`, `storage/`, `runtime/`, `lux_depth_v3/`, `hardening/`, `app.py`). Enforcement gates:
- `scripts/ci/check_per_package_coverage.py` — per-package line-coverage floors.
- `scripts/ci/check_per_package_branch_coverage.py` — per-package branch-coverage floors.
- `scripts/ci/check_cold_zone_touched_files.py` — diff-based cold-zone touched-file ratchet vs `origin/main`.

These run inside `.github/workflows/build.yml` after `coverage.xml` is produced. Coverage gains land via small ratchet PRs — never weaken a floor without an ADR.

### Test marker discipline (ADR-031, ADR-044)
Markers in `pyproject.toml`: `unit`, `security`, `regression`, `golden`, `integration`, `ml`, `slow`, `benchmark`, `stress`. ML deps are **not installed** in core CI. Required ML test patterns:

```python
# Pattern A: module-level import guard (preferred)
try:
    import transformers, torch
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

@pytest.mark.ml
@pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies required")
class TestX: ...

# Pattern B: inline
@pytest.mark.ml
def test_y():
    torch = pytest.importorskip("torch")
```

**Anti-pattern:** `@patch("transformers.CLIPModel")` at decoration time imports `transformers` during collection and breaks offline CI. Always guard the import first. Pre-commit runs `scripts/check_ml_test_isolation.sh` and `scripts/validation/check_test_markers.py` to enforce this.

### File placement (auto-organize is enforced)
The repo enforces root-file hygiene via `scripts/setup/pre-commit-check.sh`. Default homes:
- `scripts/validation/`, `scripts/verification/`, `scripts/bootstrap/`, `scripts/setup/`, `scripts/pipelines/`, `scripts/diagnostics/`, `scripts/runbooks/`, `scripts/ci/`, `scripts/governance/`, `scripts/security/`
- `tools/` for governed CLIs (archive/performance/evidence)
- `docs/<area>/` for analysis, plans, ADRs, governance
- `tests/<area>/` mirrors src layout

Do not add new root files unless the placement is genuinely operational (e.g., `Makefile`, top-level config). Pre-commit will reject misplaced files.

### Subprocess & filesystem
Use `pathlib.Path`, normalize/validate untrusted paths, enforce allowlisted roots, atomic writes (helpers in `lux_depth_v3/io_atomic.py`). For subprocesses: `subprocess.run([...], check=True, capture_output=True, text=True)`, explicit argv, no `shell=True`, set timeouts, surface stderr meaningfully.

### Backend / license governance
- `da3` is the default commercial-safe production depth backend.
- `da3-research` / `depth_pro` require `non_commercial_ok=True` + (for Depth Pro) `accept_apple_depth_pro_research_license=True`.
- Depth Pro lives in its own venv (constraint conflicts with main repo stack: pinned `torch==2.13.0`, `torchvision==0.28.0`, `numpy==1.26.4`).
- Backend resolution metadata (`requested_backend`, `resolved_backend`, `resolution_status`, `resolution_reason`) is part of every manifest — preserve it.
- Banned dependency: `realesrgan` (unmaintained); enforced by `requirements/constraints.txt` + `scripts/security/verify_banned_dependencies.py`.

### Custom-agent surface
`.github/agents/` defines three live profiles (use the narrowest one):
- `@transformation-portal-architect` — contracts, dependency policy, CI/CD, security posture, docs topology
- `@portal-app-steward` — managed browser boundary (front door, portal shell, manifest-backed assets, browser validation)
- `@transformation-portal-specialist` — backend/orchestrator, Lux Depth, archive, ingest, machine-mode, governed non-browser execution

`.github/agents/_archive/` and `.github/agents/rag_system/_archive/` are historical — never treat them as live.

## When You Edit X, Also Update Y

- CLI flags / Makefile / validation scripts / GitHub workflows / dependency locks → update `AGENTS.md`, `README.md`, relevant `docs/`, CLI help text.
- Schema fields → update validators, fixtures, schema docs in `docs/schemas/`, and contract tests in the same PR.
- `app.py` request/response shapes → update typed OpenAPI models, contract tests under `tests/test_app_orchestrator_*`, and `docs/api/`.
- Lux Depth V3 deliverables/naming → update `docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md` and run-card schema.
- Portal asset bundles → run `make check-portal-asset-budgets`; update budget contract if intentional.
- Front-door portal sources → `cd web/secure-landing && npm run build:portal` to regenerate `public/portal-assets/portal.js`.
- `orchestrator/storage/` schema → add an Alembic migration under `migrations/versions/`, run `make db-upgrade`, and update `tests/orchestrator/test_postgres_*` + `docs/runtimes/orchestrator-postgres.md`.
- `orchestrator/queue/` or `orchestrator/artifact_store/` Protocols → keep the memory backend + the live backend (Redis/S3) test parity intact and update the paid-pilot env example at `docs/deployment/paid-pilot.env.example`.
- New paid-pilot env var → add it to `docs/deployment/paid-pilot.env.example`, the `test-paid-pilot-services-contract` gate, and `docs/deployment/managed_paid_pilot_staging_runbook.md`.
- Frontdoor `session-store/` backend → mirror the Protocol in `contract.js`, keep parity with `evaluateSessionScaling()` readiness reasons, and update `web/secure-landing/tests/` contract suites.

## Canonical Worktree Discipline

`origin/main` is the only standing source of truth. Desktop siblings like `Transformation_Portal__fastapi` or `Transformation_Portal__upload` are temporary git worktrees, **not** independent repos. When consolidating parallel work, fast-forward local `main` to `origin/main`, create a single integration branch from that updated base, and replay commits there — do not treat sibling worktrees as long-lived branches of record.

## Documentation Authority

`docs/governance/DOCUMENTATION_MAP.md` plus the May 11, 2026 refresh audit (`docs/governance/DOCUMENTATION_REFRESH_AUDIT_2026-05-11.md`) are the current navigation. Two roadmaps and one gap doc anchor active engineering work:

- `docs/governance/PRODUCTION_HARDENING_GAP_2026-05-13.md` — authoritative paid-pilot baseline (what's already shipped, what's net-new, which existing primitives to reuse for Phases 1–7).
- `docs/architecture/PORTAL_ORCHESTRATOR_ROADMAP.md` — FastAPI orchestrator + portal HTTP surface roadmap.
- `docs/architecture/PORTAL_FRONTDOOR_ROADMAP.md` — managed Next.js frontdoor roadmap.

Recent ADRs that materially affect routine edits (under `docs/architecture/`):
- ADR-043 (orchestrator decomposition pattern), ADR-044 (test marker enforcement), ADR-045 (monolith-decomposition residuals governance), ADR-046 (`path_security` extraction), ADR-047 (SAM2 checkpoint security extraction), ADR-048 (Materials V3 production integration), ADR-049 (plugin manifest trust).

Older project reports under `docs/` are retained for audit context but are **not** live guidance unless the map promotes them.

## Decision Defaults

When unsure: prefer existing contracts over local convenience, decomposition over monolith expansion, explicit typing/validation over inference, lazy imports over eager heavy imports, offline deterministic tests over networked tests, additive backward-compatible changes over silent semantic drift, small PRs with docs/tests over large "cleanup" rewrites. Changes that touch contracts, schemas, portal behavior, archive governance, evidence/attestation, paid-pilot managed-services contracts (`JobRepository` / `QueueBroker` / `ArtifactStore` / `SessionStore`), or performance thresholds are **governed work** — not routine refactoring.
