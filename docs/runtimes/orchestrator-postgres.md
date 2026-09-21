# Orchestrator Postgres Runtime (Phase 1.B/1.E)

**Document Status:** Active operator runbook for the Postgres-backed
orchestrator state. The memory backend remains the default; Postgres is
opt-in via env vars.
**Last Updated:** 2026-09-12
**Related Docs:**
- `docs/governance/PRODUCTION_HARDENING_GAP_2026-05-13.md` (the Phase 0
  baseline that introduced this work)
- `src/transformation_portal/orchestrator/storage/base.py` (the
  `JobRepository` / `JobEventStore` contract this backend implements)
- `migrations/versions/0001_initial_orchestrator_schema.py` (the schema
  this layer manages)

## When to use this

By default the orchestrator stores job state in the in-process memory
`JobRepository` and loses it on restart. Phase 1.B introduced a durable
Postgres backend, and Phase 1.E wired `app.py` so `JobRepository` is the
authoritative state surface while `JOBS` only carries runtime handles.
With the Postgres backend enabled:

- With the memory queue, restart recovery marks orphaned active jobs
  `worker_lost` with `error.code = worker_lost_on_restart`. With the Redis
  locator queue, database-clock lease expiry owns recovery and preserves
  live claims belonging to other hosts.
- Multiple orchestrator workers can share state (a precondition for
  Phase 2 horizontal workers).
- Job list/detail/cancel/artifact routes read durable job rows instead
  of falling back to process-local cache.
- Reconnecting SSE clients can replay stored per-job events after a
  restart by supplying `Last-Event-ID`.

The wire shape of `/v1/jobs` and `/v2/jobs` is unchanged. Enable the runtime
only after setting the database URL, applying migrations, and preserving the
required API-key environment in each backend/worker terminal. With the Postgres backend enabled,
successfully persisted events receive a per-job monotonic sequence,
emitted as the SSE `id`. A reconnect to
`/v1/jobs/{job_id}/events` or `/v2/jobs/{job_id}/events` with a
non-negative ASCII-decimal `Last-Event-ID` receives retained events with
a higher sequence before live delivery; `Last-Event-ID: 0` requests all
retained events. Without a replay cursor, the route emits current state.
Postgres live streams poll committed history and emit only committed terminal
events; they do not synthesize a terminal event after a replay failure. See
[bounded event replay](../deployment/paid_pilot_services.md#bounded-event-replay)
for cursor gaps, polling, and retention behavior.

## Environment variables

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `TP_ORCHESTRATOR_STATE_BACKEND` | `memory` \| `postgres` | `memory` | Backend selector. |
| `TP_DATABASE_URL` | SQLAlchemy URL | unset | Required when backend is `postgres`. Example: `postgresql+asyncpg://tp:tp_dev_password@127.0.0.1:5432/transformation_portal`. |
| `TP_TEST_POSTGRES_URL` | SQLAlchemy URL | unset | Test-only. When set, `tests/orchestrator/test_repository_contract.py` runs against Postgres in addition to memory. **Never** point at production: the conftest drops/recreates the schema between cases. |

## Bring-up: local docker-compose

```bash
# 1. Start Postgres
docker compose up -d postgres

# 2. Export the database URL (or put it in your .env)
export TP_DATABASE_URL=postgresql+asyncpg://tp:tp_dev_password@127.0.0.1:5432/transformation_portal

# 3. Apply migrations
make db-upgrade

# 4. (Optional) verify the schema
docker compose exec postgres psql -U tp -d transformation_portal -c '\d+ jobs'

# 5. In this same terminal, set your existing backend API key
: "${TP_API_KEY:?Set TP_API_KEY before starting the backend}"
export TP_ORCHESTRATOR_STATE_BACKEND=postgres
make run-backend-local
```

## Schema overview

| Table | Purpose | Primary key |
| --- | --- | --- |
| `jobs` | One row per orchestrator job; mirrors the persistent slice of the legacy `app.py:Job`. JSONB columns hold `request`, `effective_request`, `logs_tail`, `artifacts`, `run_summary`, `error`. `version` increments on every update for optimistic concurrency. | `id` |
| `job_artifacts` | One row per artifact-lookup entry. Mirrors `Job.artifact_lookup`. | `(job_id, path)` |
| `job_events` | Bounded SSE replay history with per-job monotonic `seq`. | `id` (autoincrement BIGINT) |
| `job_event_sequences` | Monotonic counters independent of retained replay rows. | `job_id` |
| `dispatch_attempts` | Immutable attempt identity, once-only claim fence, generation pointer, and cleanup marker. | `job_id` |
| `admission_capacity` | Locked global and tenant active-job counters. | `scope` |
| `dispatch_plans`, `operational_records`, `committed_generations` | Immutable canonical execution and publication evidence. | Digest or immutable record identifier |
| `operational_outbox` | Pending dispatch and event delivery intents; removed after acknowledgement. | `id` |

Indices: `jobs(created_at)`, `jobs(state)`, `jobs(finished_at)`,
`job_artifacts(job_id)`, `job_events(job_id)`, and a unique
`job_events(job_id, seq)`.

The three job tables are created by
`migrations/versions/0001_initial_orchestrator_schema.py`. Apply the complete
migration chain through `0007_photography_bindings`: 0002 adds operational
audit, 0003 adds admission/dispatch/publication authority, 0004 bounds replay
with monotonic counters, 0005 tracks private attempt/staging cleanup, 0006
adds an indexed periodic scrub cursor for late writes from orphaned subprocesses,
and 0007 adds immutable physical bindings for opt-in managed V5 photography.
Do not stop at an earlier revision.

Existing V1 attempts retain null physical bindings. Migration 0007 refuses
downgrade while any V5 attempt remains, including terminal tombstones; retain a
compatible API/worker release or repair forward. See the
[managed V5 guide](../reference/LUX_DEPTH_V5.md#managed-job-execution) for activation
and request contracts.

## Migrations

The orchestrator schema is managed by Alembic under `migrations/`.
The env file (`migrations/env.py`) reads `TP_DATABASE_URL` and uses
the async-Alembic recipe so the same setup works for asyncpg.

```bash
# Apply all pending migrations.
make db-upgrade

# Create a new revision after editing the ORM models.
make db-revision MESSAGE="add foo column to jobs"

# Inspect history.
.venv/bin/python -m alembic -c migrations/alembic.ini history
```

## Concurrency

`update` and `append_logs` select the job row with `FOR UPDATE` and hold its
row lock through mutation and commit; `append_log` delegates to `append_logs`.
`set_artifacts` instead uses a version-guarded compare-and-set with bounded
retries, raising `RepositoryError` after exhausted conflicts. Read-only
`get`/`list` do not take this update lock. See
[`postgres.py`](../../src/transformation_portal/orchestrator/storage/postgres.py).

For Redis dispatch, operational transactions own state and artifact publication;
database triggers reject direct projection mutations. The distributed authority
and its service-backed validation are described below. Legacy memory-queue
jobs retain the repository behavior described above.

## Tests

The Postgres fixtures **drop and recreate tables** through `repo.reset()`.
Use a separate disposable database, never the database serving local user jobs
or a shared/production environment. Set `TP_TEST_POSTGRES_URL` to that database
and apply migrations there explicitly:

```bash
# Memory backend only (offline, no Postgres required).
make test-orchestrator-contract

# Operator must supply a disposable test database URL first.
: "${TP_TEST_POSTGRES_URL:?Set this to a disposable Postgres database}"
TP_DATABASE_URL="$TP_TEST_POSTGRES_URL" make db-upgrade
make test-orchestrator-postgres-contract
make test-orchestrator-postgres-app-contract
```

Without `TP_TEST_POSTGRES_URL`, repository-contract collection uses the memory
backend only and the app-authority smoke skips Postgres. A passing offline lane
therefore supplies no live Postgres evidence. The app smoke also stubs job
dispatch; it tests durable route authority rather than executing a model.

## Production posture

| Concern | Recommendation |
| --- | --- |
| Provider | Any managed Postgres 16+ that supports `JSONB`. Section 9 of the gap doc is the canonical decision tree. |
| Connection pool | `AsyncEngine` defaults with `pool_pre_ping=True`, `pool_recycle=300`, five-second connection/command/statement/lock bounds. There is no `TP_DB_POOL_SIZE` runtime option; changing pool tuning requires a separate implementation and validation. |
| Backups | Out of scope for Phase 1.B; document in the provider's runbook. The orchestrator never assumes durability beyond commit. |
| Schema changes | Always go through Alembic and `make db-revision` so the migration graph stays linear. |
| Secret rotation | `TP_DATABASE_URL` is read once at engine construction; a rotation requires a process restart. The memory-queue restart sweeper marks orphaned jobs `worker_lost_on_restart`; Redis dispatch retains live claims and recovers expired leases using the database clock. |
| Repository unavailable recovery | `JOB_REPOSITORY_UNAVAILABLE` is fail-closed and redacted. If it appears after fixing `TP_DATABASE_URL`, restart the backend process because repository construction failure is latched in process state. |

## Known limits in this layer

- `logs_tail` is stored on the `jobs` row as a JSONB array, bounded
  by the legacy `LOG_TAIL_LIMIT`. A full log table is intentionally
  deferred to Phase 2/6.
- Artifact files are handled by the `ArtifactStore` abstraction
  (local or S3-compatible); `job_artifacts` holds the lookup map while
  `jobs.artifacts.lifecycle` stores mirror/delete/retention metadata.
- `JOBS` and `EVENT_SUBSCRIBERS` remain process-local by design. `JOBS`
  stores live subprocess handles and cancellation tasks, while
  `EVENT_SUBSCRIBERS` stores live-delivery queues; neither is a durable
  fallback when repository reads or writes fail.
- The memory event store loses replay history on restart. Postgres retains at
  most `TP_ORCHESTRATOR_EVENT_RETENTION_PER_JOB` events per job (default 4096),
  pruning within the append transaction while preserving a monotonic counter.
  Telemetry persistence can contain gaps; it remains operational recovery
  rather than complete audit evidence. Distributed terminal events commit
  with their authoritative outcome. See [bounded event replay](../deployment/paid_pilot_services.md#bounded-event-replay).
- Job deletion removes its event history and counter. Orphan event namespaces
  remain independently bounded by the same per-job cap; operators should
  monitor the number of orphan namespaces as well as total replay storage.

---

*Phase 1.B Postgres backend - introduced 2026-05-13. Phase 1.E app
cutover - introduced 2026-05-14. Durable SSE replay wiring - introduced
2026-05-30. Update this doc when the wiring or any subsequent phase
changes the operator surface.*

## Distributed dispatch and generation authority

Redis deployments now use the versioned `:dispatch:v1:` queue namespace and
Postgres operational records. The memory broker retains its local execution
behavior. Before cutting over, stop legacy producers and workers, drain their
raw-command queue, and revoke its Redis credentials. Run `make db-upgrade`,
then give the new producer/worker credentials access only to the versioned
locator namespace. `TP_REDIS_KEY_PREFIX` supplies the prefix before
`:dispatch:v1:`; the default namespace is `tp:dispatch:v1:`. Existing raw-command
Redis messages are deliberately not consumed by the new workers.

Set `TP_ORCHESTRATOR_STATE_BACKEND=postgres`,
`TP_ORCHESTRATOR_QUEUE_BACKEND=redis`, `TP_DATABASE_URL`, `TP_REDIS_URL`, and
`TP_ORCHESTRATOR_EXECUTION_ROOT`. The execution root must be a protected shared
directory at the same absolute path on API and worker hosts, owned by the
service account with mode `0700`, outside tenant-writable directories. Its
parent must exist. Missing or unsafe execution storage rejects admission.
Both `TP_MAX_CONCURRENT_JOBS` and `TP_PILOT_MAX_ACTIVE_JOBS_PER_TENANT` must be
positive. All API hosts must use matching limits: conflicting values fail
closed. The database stores global and tenant counters, locks them in that
order, and admits the job, canonical plan, immutable attempt/dispatch, and
outbox entry in one transaction. The tenant cap applies to the `default`
tenant when the pilot control plane is disabled. Operator changes to stored
limits require a coordinated database change while admission is stopped;
never lower a limit below its active count.

An accepted HTTP job remains queued through a Redis outage. Its committed
outbox retries the same locator; queued database records also repair Redis
loss after an acknowledged delivery. Workers atomically claim each dispatch
once, then reconstruct a fixed local entrypoint from the exact canonical plan
bytes. Existing Lux/archive jobs use `tp.execution.plan.v1`; opt-in V5 photography
uses `tp.execution.plan.v4` plus separately digested immutable physical bindings.
Broker payloads contain only immutable job,
attempt, dispatch, tenant, plan-digest, and version fields. Neither broker
payloads nor operational records contain an executable command. The shared job
execution service selects the versioned photography adapter for V5 and preserves
the current Lux executor for V1; archive operations use
closed, operation-specific configurations with the current archive runner.
This does not activate the separately gated Spatial/CAS executor convergence.

Input and output roots must be available at the same authorized absolute
paths on API and worker hosts. Workers recheck current root and tenant policy
before spawning. Only archive `rights-apply`'s `policy_yaml` may additionally
use the governed `policy/archive` directory; manifest and data inputs remain
tenant-owned, and current global allowed roots still apply. Native executors
write into protected execution storage;
bounded regular outputs are atomically exported through pinned directory
descriptors into a unique `.tp-attempts` directory under the requested root.
Replacing that directory or an ancestor cannot redirect native output writes.
The private workspace name derives from the immutable admitted output locator.
Normal cleanup runs after child reaping; terminal database records and the
periodic scrub also remove interrupted or late-recreated private workspaces.
Do not change execution-root configuration until admitted jobs are drained and
terminal cleanup has completed on the old shared root. Direct local plan
consumers use private per-user temporary storage when no root is configured;
an interrupted direct invocation requires cleanup of its exact workspace
before retrying the same output locator.
Each worker has a unique holder identifier;
a database sequence supplies its epoch. Only a successful Redis heartbeat
can renew the database fence. An independent monotonic deadline signals
cancellation when either Redis or Postgres stalls; it does not wait for socket
cancellation to finish. Pending network work prevents another lease acquisition,
and connection/command timeouts bound cleanup after transport failures.
Database-clock expiration terminalizes the attempt as `worker_lost` and releases its counters once. Expired locators are
removed from Redis rather than requeued, and tombstones reject duplicate or
late deliveries. An explicit retry submits a new job. API startup never
classifies another host's live claim as an orphan.
Retryable executor hydration/startup failures retain the live claim and broker
lease for this expiry path without committing a synthetic `RUNNER_ERROR`.

`GenerationPublisher` stages immutable objects inside the existing artifact
store. Local publication uses fsynced files/directories and atomic no-replace
links; S3 publication requires conditional `PutObject` (`If-None-Match: *`).
The publisher copies regular, non-symlink output files through bounded
buffers, pins each parent directory without following symlinks, rejects
FIFO/special-file sources without blocking, and verifies staged lengths and SHA-256 digests before committing. The current limits are
`TP_MAX_INDEXED_ARTIFACTS` files (default 200, minimum 1), 4 GiB per file,
16 GiB per generation, and 1 MiB for its closed canonical manifest. Configure
the same count limit on API and worker hosts; indexing, export, and manifest
publication share it. Increasing the count does not increase the byte limits.
Generation manifests and operational records are
append-only. The database transaction verifies the running holder, epoch,
tenant, and unexpired database lease; only then does it commit the manifest,
reader pointer, terminal projection, event, and outbox. Raw output directories
and staged objects are never artifact-route authority.

Artifact routes resolve only the committed generation pointer. Failed or
stale publication leaves staged objects invisible. Explicit deletion and
retention revoke the pointer before deleting bytes; the immutable record
retains the reason. A failed physical deletion is retryable and remains
invisible. Redis delivery failures after a committed cancellation or deletion
do not undo the operation. SSE replay reads committed event history, including
terminal outcomes written in the same publication transaction.

After a child is reaped and publication finishes, cleanup removes only the
recorded private attempt directory, using descriptor-relative traversal that
cannot follow a swapped ancestor. A bounded reconciler reserves up to 100 due
terminal jobs per pass through an indexed cursor, checking each at most once
per hour. Reservations advance before filesystem I/O, so a failed check cannot
starve later jobs. Successful cleanup is recorded separately from the check
cursor; explicit deletion can retry immediately. If no committed generation
pointer exists, the scrub also deletes that job's invisible staging objects;
otherwise committed objects are preserved. A changed pointer prevents marking
cleanup complete. The periodic scrub removes late files recreated after an
earlier cleanup by a stranded child. It is best-effort cleanup: worker-host
supervisors must still reap subprocesses when the worker or host dies. The
scrub does not replace process supervision or add per-check immutable records.
Immutable evidence and dispatch tombstones intentionally outlive mutable job
history and require an explicit evidence-retention policy. S3 credentials used for
generation writes must enforce the dedicated prefix and conditional-create
policy; avoid granting clients direct write access to committed objects.

The dedicated service test database must end in `_fencing` or `_test`; its
contents are truncated by the following test lane. Supply an isolated Redis
instance/prefix and optional S3 endpoint/bucket with standard AWS credentials:

```bash
TP_DISPATCH_TEST_DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST/dispatch_test \
TP_DISPATCH_TEST_REDIS_URL=redis://HOST:6379/0 \
TP_DISPATCH_TEST_S3_ENDPOINT=http://HOST:9000 \
TP_DISPATCH_TEST_S3_BUCKET=dispatch-test \
PYTHONPATH=src:. .venv/bin/pytest tests/orchestrator/test_dispatch_authority_services.py -q
```

The suite exercises independent-process admission contention, duplicate
locators, an abruptly killed lease holder, stale publication rejection,
transactional terminal delivery, fresh-session database mutation guards,
real conditional S3 writes, and HTTP admission through an independent native
archive worker to committed artifact download. These bounded fixtures are
correctness evidence; they are not production TIFF performance evidence.
