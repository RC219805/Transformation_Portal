# Production Hardening Gap Audit - 2026-05-13

**Document Status:** Active baseline for paid-pilot hardening
**Last Updated:** 2026-05-15
**Maintainer:** Repository Architect
**Related Docs:** `docs/testing/COLD_ZONE_COVERAGE_PROGRAM.md`, `docs/architecture/PORTAL_ORCHESTRATOR_ROADMAP.md`, `docs/architecture/PORTAL_FRONTDOOR_ROADMAP.md`, `docs/deployment/PRODUCTION_READINESS.md`, `docs/operations/portal_telemetry_retention.md`
**Related Scripts:** `scripts/ci/check_per_package_branch_coverage.py`, `scripts/ci/cold_zone_report.py`, `scripts/ci/check_per_package_coverage.py`, `tools/portal_telemetry_retention.py`
**Related ADRs:** ADR-031 (test dependency isolation), ADR-044 (marker enforcement)

---

## 1. Why this exists

A multi-phase "Production Hardening Plan For Paid Pilot" was proposed against
the current `main`. Its summary is directionally correct but understates how
much hardening work is already merged and how much is genuinely greenfield.
This document is the empirical baseline: what exists, what is partial, what is
net-new, and which existing primitives must be reused rather than reinvented.

The intended outcome is a controlled paid pilot, not full self-serve SaaS.
Phase 0 of the plan is to pin this baseline; Phases 1 through 7 build on it.

## 2. Methodology

The baseline is the result of a three-agent read-only code audit performed on
2026-05-13 against `origin/main` at commit `2e4dba8`. The audit covered:

- Job orchestrator, queue, and worker execution surfaces.
- Frontdoor session storage and artifact serving.
- Tenant isolation, telemetry retention, deployment posture, evidence and
  attestation, and commercial primitives.

Findings below cite verbatim file paths and class or function names. Where the
audit recorded "not present", that means a code search returned no
implementation; it does not mean the surface is forbidden.

## 3. What is already done

The plan treats several Phase 0 items as residual work. They are landed.

| Item | Plan status | Actual status | Evidence |
| --- | --- | --- | --- |
| Cold-zone coverage program merged | Plan: "PR #1746 now merged" | Confirmed | `docs/testing/COLD_ZONE_COVERAGE_PROGRAM.md` (#1746), baseline at `docs/testing/cold_zone_baseline_2026-05-12.md` |
| Cold-zone instrumentation | Plan: Phase 0 deliverable | Already merged via #1748 | `scripts/ci/cold_zone_report.py`, `scripts/ci/check_per_package_branch_coverage.py` |
| Branch-aware coverage reporting | Plan: Phase 0 deliverable | Already merged via #1748 | `scripts/ci/check_per_package_branch_coverage.py` |
| VLM lazy-import seam | Cold-zone doc lists as Phase 0 precondition | Resolved via #1748 | See section 5.2 below |
| `/v1/jobs` and `/v2/jobs` envelope shapes | Plan: "preserve" | Both exist today | `app.py` `POST/GET /v1/jobs*` and `POST/GET /v2/jobs*`; typed models in `src/transformation_portal/api/v1/jobs.py` |
| Typed orchestrator envelope models | Plan: implicit | Live | `src/transformation_portal/api/v1/envelopes.py` `ApiEnvelope[T]`, `ErrorEnvelope` |
| Attestation chain (Merkle, DSSE, in-toto, sigstore) | Plan: implicit | Live | `src/tp/phase4/`, `src/tp/crypto/merkle.py`, `src/transformation_portal/attestation/{detached,dsse,gpg,sigstore,verify,run_card_intoto}.py` |
| Containerized backend | Plan: Phase 5 | Docker + compose live; managed services net-new | `Dockerfile` (cpu/gpu/apple-silicon stages), `docker-compose.yml`, `docs/deployment/PRODUCTION_READINESS.md` |

The actual residual Phase 0 work is therefore narrow:

1. This gap doc itself.
2. Pinned acceptance command list for paid pilot sign-off (section 7).
3. No further VLM seam work required.
4. No further branch-aware coverage script work required.

## 4. Existing primitives to reuse, not reinvent

The plan declares several net-new abstractions whose precursors already exist
in the repository. Phase 1 through Phase 7 work should extend these rather
than introduce parallel implementations.

| Plan abstraction | Existing primitive | Location | Reuse note |
| --- | --- | --- | --- |
| `JobEventStore` | `EventStore` (per-event `.json` files written under date directories `<storage>/YYYY-MM-DD/<event_id>.json`; `Event` dataclass with `id, type, timestamp, data, metadata, user, correlation_id`) | `src/transformation_portal/events/store.py` | Currently orphaned. Wire to job lifecycle for Phase 1 instead of creating a new abstraction. |
| Phase 7 organization, RBAC, and quota model | `TenantContext`, `TenantPolicy`, `TenantManager`, `TenantAwareFSGuard`, `create_tenant_sandbox()` (per-tenant `gpu_quota`, `network_allowed`, `allowed_node_types`) | `src/transformation_portal/core/security/tenant.py` | Fully tested but unwired in the request path. Phase 7 should ground the user, organization, and quota model on these primitives. |
| Phase 6 audit-event table | `tp.phase4.provenance_capture`, `tp.crypto.merkle`, `attestation/{detached,dsse,verify,run_card_intoto}.py` | `src/tp/phase4/`, `src/transformation_portal/attestation/` | Cryptographic provenance is authoritative. The new operational audit table must mirror events, not duplicate the attestation chain. |
| `ArtifactStore` (Phase 4) | `JobArtifactIndexResult`, `_artifact_fingerprint`, `_validate_resolved_job_artifact_path`, `ArtifactPathOutsideJobOutputDirError`; `ArtifactManager.compute_merkle_root(...)` (method) and module-level `compute_artifact_merkle_root(...)` | `src/transformation_portal/portal/job_artifacts.py`, `src/transformation_portal/lux_depth_v3/artifact_manager.py` | Phase 4.A landed: `src/transformation_portal/orchestrator/artifact_store/` reuses these helpers behind the `ArtifactStore` Protocol so the local + S3 backends share the validated path-traversal, fingerprint, content-type, and Merkle-input behaviour. Phase 4.B landed via PR #1776: `app.py` now uses the store for artifact delivery, signed S3 redirects, lifecycle metadata, deletion, cleanup, and readiness. The post-merge hardening follow-up preserves S3 response-header semantics, uses real bucket readiness, expands destructive-route/mirror tests, and keeps failed artifact cleanup retryable. |
| Phase 3 frontdoor session readiness gate | `evaluateSessionScaling()` (returns `ok: false` for `multi_instance` and `ephemeral_runtime` with `*_requires_external_session_store` reason codes) | `web/secure-landing/lib/session-scaling.js` | Add a `redis` branch that flips the gate to `ok: true` rather than rewriting the readiness check. |
| Phase 0 production-gap doc | This document | `docs/governance/PRODUCTION_HARDENING_GAP_2026-05-13.md` | Treat as the authoritative reference cited by Phase 1 through Phase 7 PRs. |

## 5. Net-new surface (Phases 1 through 7)

The majority of Phases 1 through 7 is genuinely net-new. The audit found no
implementations or precursors for the items below.

### 5.1 Phase 1 - durable jobs

**Updated 2026-05-14: Phase 1.A through Phase 1.E have landed.**

Already done:

- `JobRepository` and `JobEventStore` Protocols + `JobRecord` dataclass + memory backend (Phase 1.A, PR #1756).
- `PostgresJobRepository` + `PostgresJobEventStore` + SQLAlchemy 2.x async ORM + Alembic initial migration + docker-compose Postgres service + `make db-upgrade` / `make db-revision` / `make test-orchestrator-postgres-contract` (Phase 1.B, PR #1758). Backend selected via `TP_ORCHESTRATOR_STATE_BACKEND=memory|postgres` and `TP_DATABASE_URL`. See `docs/runtimes/orchestrator-postgres.md` for the operator runbook.
- Pessimistic restart recovery (Phase 1.C, PR #1759). `src/transformation_portal/orchestrator/recovery.py:sweep_orphaned_jobs` runs on every FastAPI startup via `_orchestrator_lifespan` in `app.py`: any job that the repository still records as `queued` or `running` but that no live worker in the runtime registry is executing is marked `failed` with `error.code = "worker_lost_on_restart"`. SSE late-clients now see a terminal `done` after a restart instead of hanging. Memory backend: deterministic no-op (per-process state). Postgres backend: durable. The lifespan also disposes the repository's connection pool on shutdown.
- Runtime Pydantic envelope validation on `/v[12]/jobs` success paths (Phase 1.D). The four success-path returns (`_create_job`, `_list_jobs`, `_get_job`, `_cancel_job`) construct an `ApiEnvelope[T]` Pydantic model at runtime and serialize via `model_dump(by_alias=True, exclude_unset=True)`. Bad field types or missing required fields now raise at construction instead of slipping through as malformed JSON. Byte-identity with the legacy `_api_envelope` helper is pinned by `tests/orchestrator/test_envelope_runtime_validation.py` (12 tests: side-by-side model-vs-helper canonical JSON equality + HTTP-level wire-shape pins across `v1` and `v2`). SSE event payloads stay manual; the error path (`_error_response`) is unchanged.
- `app.py` JobRepository cutover (Phase 1.E). Job list/detail/cancel/artifact/delete/events now load durable state from `JobRepository` first and overlay only runtime handles from `JOBS`; create writes a `JobRecord` before broker enqueue; runner state/progress/log/artifact mutations persist through repository methods; cleanup avoids artifact-unaware `repo.cleanup_expired()` and preserves Phase 4.B delete-retry metadata. Repository failures return redacted `503 JOB_REPOSITORY_UNAVAILABLE` instead of falling back to stale process cache.

Still net-new (follow-up commits / PRs):

- `JobCreateRequest` exists at `src/transformation_portal/api/v1/jobs.py:130` but is still "defined, not yet wired" as a handler parameter. Wiring it would collapse the specific `_create_job` error reason codes (e.g. `"unsupported_pipeline"`) into a generic `"request_validation_failed"`; that trade-off is intentionally deferred per the module docstring's recommendation and is the next layer's decision.
- Durable SSE replay through `JobEventStore` remains a separate event-store cutover. Phase 1.E makes the event route repository-backed for existence/state; it does not make event history restart-replayable.

### 5.2 Phase 2 - worker split

**Updated 2026-05-13: Phases 2.A, 2.B, 2.C, 2.D, and 2.E have landed.**

Already done:

- `QueueBroker` Protocol + `JobEnqueueRequest` / `JobLease` dataclasses + `MemoryQueueBroker` (Phase 2.A). Backend selected via `TP_ORCHESTRATOR_QUEUE_BACKEND=memory|redis`. The broker contract pins lease/heartbeat semantics (`acquire_lease` / `extend_lease` / `release_lease` / `reclaim_expired_leases`), pre-lease and in-flight cancellation, and admission-collision detection. A worker runner skeleton (`src/transformation_portal/orchestrator/worker.py`) wraps the broker in a poll-with-backoff supervisor loop, a heartbeat coroutine, cooperative `CancelledByOrchestrator` handling, and a `JobExecutor` callable seam now wired in Phase 2.C.
- `RedisQueueBroker` (Phase 2.B) backed by `redis>=5` async client + a docker-compose `redis` service (AOF on, `noeviction` policy, healthcheck) + `make test-worker-redis-contract`. Atomicity for acquire / extend / release / reclaim / cancel is implemented as server-side Lua so admission collisions and lease handoff never produce partial state, and lease deadlines are pinned to the Redis server clock (via `redis.call('TIME')`) so a multi-host fleet shares a single source of truth. Per-test `key_prefix` isolation lets pytest-xdist and shared-tenant Redis deployments coexist. The Phase 2.A contract test suite now runs against both backends; Redis activates when `TP_TEST_REDIS_URL` is set, mirroring the Phase 1.B Postgres pattern.
- `app.py` broker-mediated dispatch (Phases 2.C–2.E). `_create_job` enqueues a `JobEnqueueRequest` instead of calling `asyncio.create_task(_run_job(...))`; an in-process `WorkerRunner` pool (`MAX_CONCURRENT_JOBS` workers) spawned in the FastAPI lifespan acquires leases and runs the existing `_run_job` body via `_orchestrator_job_executor`. `_request_cancel` routes through `broker.cancel`: queued (pre-lease) jobs are dropped from the broker FIFO and the orchestrator publishes the terminal cancelled-done event itself; leased (in-flight) jobs surface `LeaseStatus.cancelled` to the worker's next heartbeat, which trips an `asyncio.Event` bridged into `Job.cancel_requested` so the existing `_run_job` readline loop terminates the subprocess. The Phase 1.C restart sweep stays wired; broker close runs after worker stop on lifespan shutdown. Phase 2.E removed the legacy in-band `asyncio.create_task(_run_job(...))` fallback and the `TP_ORCHESTRATOR_USE_QUEUE_BROKER` env var — broker dispatch is the only execution path; `_dispatch_job` fail-closes with a 503 `QUEUE_UNAVAILABLE` if broker construction or `enqueue` raises.
- `worker_lost` state + retry classification + broker-reclaim reconciler (Phase 2.D). The Job state machine gains a distinct `worker_lost` terminal state (`queued|running|succeeded|partial|failed|canceled|worker_lost`) so operators can distinguish executor failures (the work itself is broken) from broker-level failures (the worker died holding a lease — the job payload is intact). Both the Phase 1.C `sweep_orphaned_jobs` restart sweep and a new in-process reclaim reconciler driven by `broker.reclaim_expired_leases(now=server_time())` produce `state=worker_lost` with an explicit `error.retriable=True` flag; executor-level failures (`RUNNER_EXIT_NONZERO` / `RUNNER_NOT_FOUND` / `RUNNER_ERROR`) carry `error.retriable=False`. The reclaim reconciler runs every `TP_ORCHESTRATOR_RECLAIM_SWEEP_INTERVAL_SECONDS` (default 5s); the executor adapter has a terminal-state guard so a stale broker re-queue cannot re-spawn a job that was already marked `worker_lost`. Worker tunables accept the canonical `TP_WORKER_*` names (matching `worker.py:_config_from_env`) with the legacy `TP_ORCHESTRATOR_WORKER_*` form honored as a fallback for backward compatibility.

Still net-new (follow-up commits / PRs on this track):

- Multi-host worker deployment (separate worker process, separate Dockerfile target, observable lease-reclaim metrics). Tracked under Phase 5 pilot deployment.

### 5.3 Phase 3 - external sessions

**Updated 2026-05-13: Phase 3.A and Phase 3.B have landed.**

Already done:

- `SessionStore` contract + factory + `SqliteSessionStore` + `RedisSessionStore` (Phase 3.A). Lives under `web/secure-landing/lib/session-store/` with one module per concern: `contract.js` (JSDoc-typed wire shape), `sqlite-store.js` (refactor of the pre-Phase-3 SQLite logic — schema and statements byte-identical), `redis-store.js` (ioredis-backed implementation), and `index.js` (factory keyed off `TP_FRONTDOOR_SESSION_STORE=sqlite|redis`, default `sqlite`). Env vars: `TP_FRONTDOOR_SESSION_STORE`, `TP_FRONTDOOR_REDIS_URL`, `TP_FRONTDOOR_REDIS_KEY_PREFIX` (default `tp:frontdoor:`).
- `evaluateSessionScaling()` gate flip (Phase 3.A). Accepts `sessionStoreBackend` and returns `ok: true` for `multi_instance` / `ephemeral_runtime` modes when `sessionStoreBackend === "redis"`, while keeping the `single_instance` + SQLite default green. Invalid mode declarations still fail closed; unknown backend values fall back to SQLite gate semantics so a misconfigured store can never silently unlock the gate.
- `sessions.js` cut-over (Phase 3.B, this PR). The frontdoor's session helpers now delegate to `getSessionStore()` rather than touching `better-sqlite3` directly. Persistence-touching helpers (`createAnonymousSession`, `getSessionById`, `getSessionFromRequest`, `rotateAuthenticatedSession`, `destroySession`, `isLoginThrottled`, `recordLoginAttempt`, `revokeSessionOnAccessFailure`) became async; pure-cookie / CSRF helpers (`setSessionCookie`, `clearSessionCookie`, `validateCsrfToken`, `getRemoteAddress`) stay sync since they never touched the store. Every Next.js route handler that imports the helpers (`app/login`, `app/logout`, `app/portal`, `app/portal/bootstrap`, `app/v1/[...path]`) was updated to `await`. The audit-event payloads and the SQLite schema / SQL surface are byte-identical to the pre-Phase-3 implementation — the only observable change is the new backend selection and the async signatures.
- `ioredis` runtime dep (Phase 3.B, this PR). `web/secure-landing/package.json` pins `ioredis: ^5.7.0` (lock resolves to 5.10.1) and `npm run build` succeeds with the dep present. The `redis-store.js` module's lazy `await import("ioredis")` pattern is unchanged — production multi-instance deployments set the env vars and the runtime activates the Redis path automatically.
- Test coverage. 250 frontdoor cases pass via `npm test`, including the 11 `tests/session-store.test.mjs` cases (SqliteSessionStore round-trip + RedisSessionStore login-throttle path through an injected fake client) and the 9 `tests/session-scaling.test.mjs` cases (gate semantics for sqlite + redis + invalid modes + unknown backends). The 13-case `tests/auth-flow.test.mjs` integration test exercises the async surface of `createAnonymousSession` / `rotateAuthenticatedSession` / `recordLoginAttempt` / `isLoginThrottled` end-to-end against the SQLite store. The 132-case `tests/routes.test.mjs` HTTP integration suite covers every async-bound route handler in the new shape.

Still net-new (follow-up tracks):

- Operator runbook for the multi-instance Redis deployment (env vars, Cloudflare Workers preview pin, session-replay smoke against the redis branch). Tracked under Phase 5 pilot deployment rather than Phase 3.
- `RedisSessionStore.persistSession` and `touchSession` set TTL on every write; a follow-up could swap this for `EXPIRE` after a `WATCH`/`MULTI` round-trip to make the absolute timeout strictly monotonic under tab-noise. Not required for the paid-pilot blast radius; logged here as a known follow-up.

### 5.4 Phase 4 - artifact lifecycle

**Updated 2026-05-13: Phase 4.A has landed. Updated 2026-05-14:
Phase 4.B delivery lifecycle landed via PR #1776; the post-merge
hardening follow-up is integrated here.**

Already done:

- `ArtifactStore` Protocol + factory + `LocalArtifactStore` + `S3ArtifactStore` (Phase 4.A). Lives under `src/transformation_portal/orchestrator/artifact_store/` with one module per concern: `base.py` (the async `ArtifactStore` ABC + `ArtifactObjectMetadata` dataclass + `ArtifactNotFoundError` / `ArtifactPathValidationError` / `ArtifactStoreError` exceptions), `local.py` (filesystem backend; reuses `portal.job_artifacts._artifact_content_type`, `ARTIFACT_FINGERPRINT_MAX_BYTES`, and legacy exception semantics; preserves the pre-Phase-4 traversal guarantees via equivalent normalization logic plus safe single-component `job_id` validation; uses temp-file + fsync + `os.replace` writes; persists explicit content-type overrides in a local metadata sidecar), `s3.py` (boto3-backed S3 backend with lazy `import boto3` so the local lane pays no S3 client construction cost; key layout `{prefix}/jobs/{job_id}/{relative_path}`; sha256 computed on a streamed `GetObject` body without materialising the full bytes in memory; `list_for_job` reads S3 object metadata so `content_type` matches `write_bytes` / `head`; single deletes return `0` for missing keys and bulk delete uses `list_objects_v2` + `delete_objects` 1000-key batches), and `__init__.py` (factory keyed off `TP_ARTIFACT_STORE=local|s3`, default `local`).
- Env vars `TP_ARTIFACT_STORE` (`local`|`s3`, default `local`), `TP_ARTIFACT_BUCKET` (required when backend is `s3`), `TP_ARTIFACT_PREFIX` (default `tp/artifacts`), `TP_ARTIFACT_ENDPOINT_URL` (optional for MinIO / LocalStack / R2), `TP_ARTIFACT_REGION`, `TP_ARTIFACT_LOCAL_ROOT` (filesystem root for the local backend; default `$XDG_STATE_HOME/transformation-portal/artifacts` when set, otherwise the platform temp directory).
- Parametrized contract suite. `tests/orchestrator/test_artifact_store_contract.py` runs against `LocalArtifactStore` (always) and `S3ArtifactStore` (default branch uses `moto`'s `mock_aws` so the S3 contract is exercised in every CI lane without live AWS / MinIO credentials; `TP_TEST_S3_URL` + `TP_TEST_S3_BUCKET` switch the same suite to a real S3-compatible endpoint, mirroring Phase 1.B Postgres and Phase 2.B Redis patterns). Coverage: write/head round-trip, explicit `content_type` round-trip through head/list, open_bytes streaming, list_for_job sort, path-traversal validation for relative paths and malicious job IDs across write/head/open/list/delete, symlinked local job-root escape rejection, missing single-delete `0` semantics, single + bulk delete, Merkle parity with `compute_artifact_merkle_root`, oversized-fingerprint `skipped_size` contract.
- Dependency wiring. `boto3>=1.34,<2` added to `pyproject.toml` runtime metadata and `requirements/base.in` (lazy-loaded inside `S3ArtifactStore` so the local lane pays no import cost at orchestrator start-up). `moto[s3]>=5.0,<6` added to `requirements/dev.in` for the parametrized test branch.
- App delivery lifecycle (Phase 4.B, PR #1776). `app.py` now mirrors indexed terminal-job artifacts into the configured `ArtifactStore` with `write_file`, serves local artifacts through `open_bytes`, keeps legacy filesystem fallback only for already-indexed local jobs that predate mirroring, and routes S3-backed artifacts through authenticated orchestrator URLs that mint short-lived 307 redirects after auth/job/path/item validation. Persisted `Job.artifacts.items` keep the existing `url`, `download_url`, `path`, `relative_path`, `content_type`, `mime_type`, preview, display-hint, size, and fingerprint fields; presigned URLs are not stored in status payloads.
- Lifecycle and cleanup (Phase 4.B, PR #1776). `job.artifacts.lifecycle` is additive top-level metadata with `created_at`, `expires_at`, `deleted_at`, `retention_policy`, `artifact_store_backend`, deletion status/count/error fields, and mirror status. Authenticated `DELETE /v1/jobs/{job_id}/artifacts` and `/v2/jobs/{job_id}/artifacts` delete terminal-job artifacts idempotently through the store, retain artifact item metadata for support, return `410 ARTIFACT_DELETED` after deletion, and reject active jobs with conflict. The cleanup loop attempts artifact deletion before job removal and records failures without crashing.
- Readiness (Phase 4.B, PR #1776). `/ready` reports `artifact_store` with `backend`, `configured`, `prefix`, and `signed_urls`; S3 configuration failures are redacted as `artifact_store_unavailable`.
- Post-merge ArtifactStore hardening follow-up. S3 presigned delivery now passes response-header overrides so final object responses preserve the legacy `Content-Type`, non-preview attachment `Content-Disposition`, and `Cache-Control: no-store` semantics. Readiness uses a real S3 `head_bucket` bucket/access probe instead of a synthetic presign. Destructive artifact deletion routes have explicit auth-failure coverage for `/v1` and `/v2`; S3 first-access mirroring from legacy local artifacts is covered, including mirror-failure `503 ARTIFACT_STORE_UNAVAILABLE` behavior; and expired job cleanup keeps retry metadata when artifact deletion fails instead of removing the job record.

Still net-new after Phase 4 closeout:

- Live MinIO / LocalStack / managed-S3 smoke for pilot deployment acceptance. CI uses `moto` for deterministic contract coverage; live endpoint validation remains a Phase 5 deployment gate.

### 5.5 Phase 5 - pilot deployment model

**Updated 2026-05-15: Phase 5.A local Compose validation has landed.**

- `Dockerfile` provides `base`, `cpu`, and `gpu` runtime stages with health
  probes. The `apple-silicon` target remains as a CPU-image compatibility alias;
  native Apple Silicon ML installs are governed by the Darwin arm64 ML lock, not
  by ad hoc Docker `.[ml]` installs.
- `docker-compose.yml` provides CPU, GPU, worker, and monitor services.
- `make test-paid-pilot-services-contract` exists as the opt-in managed
  services smoke gate.
- Local Compose validation passed at merge commit
  `07a3e8e847dee4a6e1ccf46d6dcd80b612fe3753`.
- The gate composes Postgres `JobRepository`, Redis `QueueBroker`, Redis
  frontdoor `SessionStore`, S3-compatible `ArtifactStore`, artifact
  fetch/delete lifecycle, and `worker_lost` recovery.
- The provider-neutral managed staging procedure is documented in
  `docs/deployment/managed_paid_pilot_staging_runbook.md`.
- Still pending: managed-provider/staging validation against provider
  Postgres, Redis, and S3-compatible endpoints; Helm charts; Terraform;
  Kubernetes manifests; separate multi-host worker deployment; and a CI deploy
  pipeline beyond container build.
- `docs/deployment/PRODUCTION_READINESS.md` exists but does not cover managed
  services.

### 5.6 Phase 6 - production evidence

- No `prometheus_client`, no OpenTelemetry, no `structlog`. Logging is stdlib
  `logging` plus a healthcheck-noise filter at `app.py:120`.
- No queue-depth, job-latency, failure-rate, or artifact-write metrics.
- No alert thresholds and no dashboard templates.
- The only audit log today is `FSGuard.audit_log` (filesystem accesses to a
  JSONL file at `src/transformation_portal/core/security/fs_guard.py:100`).
  This is not an audit surface for jobs or the orchestrator.
- Cryptographic provenance and attestation are complete (see section 4); the
  operational audit-event table is the net-new piece.

### 5.7 Phase 7 - commercial primitives

- No `User`, `Organization`, `ApiKey`, `Quota`, `Billing`, or `Subscription`
  models in the codebase.
- API-key auth is single-tenant only: `TP_API_KEY` and `TP_BACKEND_API_KEY`.
- Tenant primitives in `src/transformation_portal/core/security/tenant.py`
  exist but are not wired (see section 4); they should be the foundation for
  this phase.

## 6. Plan refinements

The following refinements address ambiguity or under-specification in the
original plan; they do not change its phase order.

1. **`/v2/jobs` is live, not aspirational.** Reframe Phase 1 wording from
   "preserve `/v1/jobs`, `/v2/jobs` envelope shapes" to "preserve the
   currently-live `/v1` and `/v2` shapes." All four route families
   (`POST/GET/cancel/artifacts`) exist today on a shared in-memory backend.
2. **Pydantic envelopes are OpenAPI-only.** Phase 1 must explicitly switch
   the typed envelopes to runtime-validated request and response models in
   the same change as the storage cutover; otherwise the "wire shape stable"
   claim is unenforceable.
3. **Audit-event table boundary.** Phase 6 should specify that the new
   operational audit table is a queryable mirror of events, while
   `tp.phase4` and the Merkle and attestation chain remain authoritative for
   evidence. Operational and cryptographic audit must not collide.
4. **Run-card and reconstruction-manifest continuity in Phase 4.**
   `lux_depth_v3.artifact_manager.compute_artifact_merkle_root` builds Merkle
   roots over filesystem layouts. The S3 swap must keep the Merkle inputs
   deterministic; that means stable iteration order, stable canonical relative
   paths, and explicit handling of object keys versus directory prefixes.
5. **Frontdoor session migration.** The plan's assumption "no migration
   required" is correct **only** if no paid users have logged in before
   cutover. The Phase 3 PR must include a documented cutover step that either
   invalidates SQLite sessions or migrates them.
6. **Tenant primitives are the floor for Phase 7.** Phase 7 must build on
   `TenantContext`, `TenantPolicy`, `TenantManager`, and `TenantAwareFSGuard`
   rather than introducing a parallel customer model.

## 7. Pinned acceptance commands for paid pilot sign-off

The following command list is the gate that must pass before the paid pilot
opens. Each command has a current ownership and expected outcome.

### 7.1 Currently passing

```bash
make ci                                # advisory lint + governance + test-fast + orchestrator + frontdoor contracts
make lint-parity                       # CI-aligned Black/isort/flake8/pylint
make test-orchestrator-contract        # in-memory job orchestrator contract
make test-frontdoor-contract           # Node 22 frontdoor + Next.js build
make validate-ci                       # workflow / concurrency / gitleaks / dependabot contracts
make audit-pipeline-readiness          # 4-pipeline local readiness audit
make check-portal-asset-budgets        # portal asset size gates
```

These are the existing gates and should not regress as Phases 1 through 7
land. They cover the in-memory orchestrator, the SQLite frontdoor, the
filesystem artifact path, and the workflow and dependency governance.

### 7.2 Durable service component gates

The Postgres, Redis, frontdoor Redis, and S3 component targets now exist. They
must pass before provider-backed paid-pilot admission:

| Phase | Make target | What it must prove |
| --- | --- | --- |
| 1 | `test-orchestrator-postgres-contract` | `JobRepository` contract identical for memory and Postgres backends; restart recovery; cancel semantics; artifact index parity. Durable SSE replay remains a separate event-store cutover. |
| 2 | `test-worker-redis-contract` | Queue lease and heartbeat expiry; duplicate-consumer protection; `worker_lost` marking; cancellation honored across queue boundary. |
| 3 | `test-frontdoor-redis-contract` | Multi-instance and ephemeral readiness green paths; CSRF preserved across instances; throttle parity; session TTL semantics. |
| 4 | `test-artifact-s3-contract` | S3-compatible write, read, delete, signed-URL expiry, checksum mismatch rejection, path-traversal rejection; deterministic Merkle inputs. |

### 7.3 Paid-pilot managed-services gate

```bash
make test-paid-pilot-services-contract
```

Phase 5.A local Compose validation for this gate is complete at
`07a3e8e847dee4a6e1ccf46d6dcd80b612fe3753`. Managed-provider validation is
pending until the same target passes against provider-managed Postgres, Redis
queue storage, Redis frontdoor session storage, and S3-compatible artifact
storage.

### 7.4 Live validation gates

```bash
make validate-frontdoor-browser        # frontdoor browser smoke (current: SQLite path)
make validate-portal-browser           # portal browser smoke (current: in-memory path)
```

After Phases 1 through 4 land, these should be re-run against the durable
backend rather than the local in-memory and SQLite defaults.

### 7.5 Evidence and attestation

```bash
python3 scripts/security/verify_banned_dependencies.py
python3 scripts/validation/check_requirements_lock_contract.py
python3 scripts/validation/check_dependency_pinning.py
```

These cover the supply-chain and dependency-pinning gates. They should pass
on every pilot-track PR.

## 8. VLM lazy-import seam status

**Status: resolved.**

The cold-zone program (`docs/testing/COLD_ZONE_COVERAGE_PROGRAM.md` section
2.1 and 4.3) lists the VLM lazy-import seam as a Phase 0 precondition for
cold-zone PRs 1 through 3. PR #1748 landed the seam:

- `src/transformation_portal/vlm/__init__.py` now uses PEP 562 `__getattr__`
  with a `_LAZY_EXPORTS` mapping; no eager re-exports.
- `src/transformation_portal/vlm/llava.py` wraps `import torch` in
  `try / except (ImportError, OSError)` and exposes `TORCH_AVAILABLE`.

Verification:

```bash
python -c "import sys; \
  assert 'torch' not in sys.modules; \
  import transformation_portal.vlm; \
  assert 'torch' not in sys.modules"
```

The cold-zone program doc still lists the seam as Phase 0 work; that doc
should be updated separately to mark section 2.1 and 4.3 as resolved when
the next cold-zone PR lands. No further hardening-track work is required for
the seam itself.

## 9. Open decisions before Phase 1 starts

These decisions are deliberately not made in this gap doc; they require
operator input.

1. **Deployment target.** Managed Postgres, Redis, and S3-compatible storage
   per provider (AWS, GCP, Cloudflare R2, Fly, Render, Railway, other). The
   choice changes Phase 4 endpoint defaults, IAM model, and Phase 5 IaC
   approach.
2. **Frontdoor host.** Vercel-managed Next.js (current default in
   `web/secure-landing/`) or self-hosted standalone Next.js. The choice
   changes the Phase 3 session-store rollout shape.
3. **Pilot cutover policy for SQLite frontdoor sessions.** Invalidate or
   migrate (see section 6 item 5).
4. **Audit-event table backing.** Same Postgres instance as the job store, a
   separate database, or a managed log store. This affects retention and
   querying.
5. **Customer entitlement model.** Contract-based or quota-based for the
   pilot; both can ground on `TenantPolicy`. This affects Phase 7 sequencing.

These should be answered before Phase 1 implementation starts so the work
does not require mid-phase rework.

---

*Phase 0 baseline pinned 2026-05-13. Update this doc when a Phase 1 through
Phase 7 PR materially changes the audited state.*
