# Orchestrator Postgres Runtime (Phase 1.B/1.E)

**Document Status:** Active operator runbook for the Postgres-backed
orchestrator state. The memory backend remains the default; Postgres is
opt-in via env vars.
**Last Updated:** 2026-05-14
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

- Restart no longer destroys in-flight jobs (the Phase 1.C sweeper
  marks orphans `worker_lost_on_restart`).
- Multiple orchestrator workers can share state (a precondition for
  Phase 2 horizontal workers).
- Job list/detail/cancel/artifact routes read durable job rows instead
  of falling back to process-local cache.

The wire shape of `/v1/jobs` and `/v2/jobs` is unchanged. The runtime
swap is a single env flip. Durable SSE replay through `JobEventStore`
is intentionally separate from this cutover; the event route now checks
repository-backed job existence/state, but replay history is still a
later event-store wiring change.

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

# 5. Start the orchestrator with the durable backend
export TP_ORCHESTRATOR_STATE_BACKEND=postgres
make run-backend-local
```

## Schema overview

| Table | Purpose | Primary key |
| --- | --- | --- |
| `jobs` | One row per orchestrator job; mirrors the persistent slice of the legacy `app.py:Job`. JSONB columns hold `request`, `effective_request`, `logs_tail`, `artifacts`, `run_summary`, `error`. `version` increments on every update for optimistic concurrency. | `id` |
| `job_artifacts` | One row per artifact-lookup entry. Mirrors `Job.artifact_lookup`. | `(job_id, path)` |
| `job_events` | Append-only SSE event history with per-job monotonic `seq` so a reconnecting client can resume. | `id` (autoincrement BIGINT) |

Indices: `jobs(created_at)`, `jobs(state)`, `jobs(finished_at)`,
`job_artifacts(job_id)`, `job_events(job_id)`, and a unique
`job_events(job_id, seq)`.

All three tables are created by the initial Alembic migration
`migrations/versions/0001_initial_orchestrator_schema.py`.

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

Every `JobRepository.update`, `append_log`, and `set_artifacts` call
runs an optimistic-concurrency check against `jobs.version` and
retries up to three times on conflict before raising
`RepositoryError`. This pairs well with the asyncio single-loop
model used by the FastAPI orchestrator and is robust enough for the
horizontal-worker model that Phase 2 will introduce.

For row-level reads the backend uses `select` without `FOR UPDATE`;
read-modify-write paths re-fetch + re-compare-and-set inside a single
async transaction.

## Tests

```bash
# Memory backend only (offline, no Postgres required).
make test-orchestrator-contract

# Memory + Postgres backends (parametrized).
docker compose up -d postgres
make db-upgrade
TP_TEST_POSTGRES_URL=postgresql+asyncpg://tp:tp_dev_password@127.0.0.1:5432/transformation_portal \
  make test-orchestrator-postgres-contract
```

`tests/orchestrator/conftest.py` auto-skips the Postgres branch when
`TP_TEST_POSTGRES_URL` is unset, so the offline `make test-fast` and
`make ci` lanes stay green without a database.

## Production posture

| Concern | Recommendation |
| --- | --- |
| Provider | Any managed Postgres 16+ that supports `JSONB`. Section 9 of the gap doc is the canonical decision tree. |
| Pool size | `AsyncEngine` default plus `pool_pre_ping=True` and `pool_recycle=300`. Tune with `TP_DB_POOL_SIZE` once Phase 6 metrics are wired. |
| Backups | Out of scope for Phase 1.B; document in the provider's runbook. The orchestrator never assumes durability beyond commit. |
| Schema changes | Always go through Alembic and `make db-revision` so the migration graph stays linear. |
| Secret rotation | `TP_DATABASE_URL` is read once at engine construction; a rotation requires a process restart. The restart sweeper (Phase 1.C, follow-up) will mark any orphaned `running` jobs `failed` with `error.code = worker_lost_on_restart`. |

## Known limits in this layer

- `logs_tail` is stored on the `jobs` row as a JSONB array, bounded
  by the legacy `LOG_TAIL_LIMIT`. A full log table is intentionally
  deferred to Phase 2/6.
- Artifact files are handled by the `ArtifactStore` abstraction
  (local or S3-compatible); `job_artifacts` holds the lookup map while
  `jobs.artifacts.lifecycle` stores mirror/delete/retention metadata.
- `JOBS` remains process-local by design. It stores live subprocess
  handles, cancellation tasks, and subscriber queues only; it is not a
  durable fallback when repository reads or writes fail.
- Durable SSE replay is not wired through `JobEventStore` yet. Late
  clients still receive the current process-local replay behavior; a
  restart-safe event replay cutover is a separate PR.

---

*Phase 1.B Postgres backend - introduced 2026-05-13. Phase 1.E app
cutover - introduced 2026-05-14. Update this doc
when the wiring or any subsequent phase changes the operator surface.*
