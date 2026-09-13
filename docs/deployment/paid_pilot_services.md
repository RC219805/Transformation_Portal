# Paid-Pilot Managed-Services Smoke Gate

Phase 5.A adds an opt-in validation gate for the current paid-pilot backend topology:

- Postgres-backed orchestrator `JobRepository`
- Redis-backed orchestrator `QueueBroker`
- Redis-backed managed-frontdoor `SessionStore`
- S3-compatible `ArtifactStore`

This is not production infrastructure-as-code. It is a deterministic smoke gate for proving that the existing durable components compose under explicit service endpoints.

## Managed Provider / Staging Validation

This document defines the paid-pilot smoke gate and local Compose validation
path. For provider-managed staging validation, use
[`managed_paid_pilot_staging_runbook.md`](managed_paid_pilot_staging_runbook.md).

Local Compose validation proves the stack against disposable local Postgres,
Redis, and MinIO services. Managed-provider validation is separate and remains
pending until the same gate passes against provider-managed Postgres, Redis
queue storage, Redis frontdoor session storage, and S3-compatible artifact
storage.

## Local Compose Services

Bring up the local disposable services:

```bash
set -a
. ./docs/deployment/paid-pilot.env.example
set +a
docker compose up -d postgres redis
docker compose --profile paid-pilot up -d minio minio-create-bucket
make db-upgrade
```

The local MinIO lane binds to loopback by default:

```text
TP_ARTIFACT_ENDPOINT_URL=http://127.0.0.1:9000
TP_TEST_S3_URL=http://127.0.0.1:9000
TP_ARTIFACT_BUCKET=tp-artifacts-pilot
TP_TEST_S3_BUCKET=tp-artifacts-pilot
AWS_ACCESS_KEY_ID=tp_minio
AWS_SECRET_ACCESS_KEY=tp_minio_password
```

The credentials in `docs/deployment/paid-pilot.env.example` are development-only defaults. Managed environments must use provider-issued credentials and secret storage.

## Managed-Service Env Mapping

Set these selectors and URLs before running the paid-pilot gate:

```text
TP_ORCHESTRATOR_STATE_BACKEND=postgres
TP_DATABASE_URL=postgresql+asyncpg://...
TP_TEST_POSTGRES_URL=postgresql+asyncpg://...
TP_ORCHESTRATOR_QUEUE_BACKEND=redis
TP_ORCHESTRATOR_EXECUTION_ROOT=/srv/transformation-portal/execution
TP_REDIS_URL=redis://...
TP_TEST_REDIS_URL=redis://...
TP_FRONTDOOR_SESSION_STORE=redis
TP_FRONTDOOR_REDIS_URL=redis://...
TP_ARTIFACT_STORE=s3
TP_ARTIFACT_BUCKET=...
TP_ARTIFACT_ENDPOINT_URL=...
TP_TEST_S3_URL=...
TP_TEST_S3_BUCKET=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
TP_ARTIFACT_REGION=...   # optional; live S3-compatible tests default to us-east-1 when unset
```

`TP_TEST_POSTGRES_URL`, `TP_TEST_REDIS_URL`, and `TP_TEST_S3_BUCKET` must point at disposable validation services. The paid-pilot integrated smoke intentionally remaps the app-facing `TP_DATABASE_URL`, `TP_REDIS_URL`, `TP_ARTIFACT_ENDPOINT_URL`, and `TP_ARTIFACT_BUCKET` to those test endpoints inside the test process before it runs destructive setup/cleanup. The Postgres smoke resets the test schema, Redis tests delete keys under isolated prefixes, and S3 tests delete objects under isolated prefixes.

## Validation Commands

Component gates:

```bash
make test-orchestrator-postgres-contract
make test-orchestrator-postgres-app-contract
make test-worker-redis-contract
make test-artifact-s3-contract
make test-frontdoor-redis-contract
```

Integrated pilot gate:

```bash
make test-paid-pilot-services-contract
```

Managed-provider staging should use the clean launcher so local development env
does not leak into the gate:

```bash
TP_MANAGED_PAID_PILOT_ENV_FILE=/tmp/tp-managed-staging.env \
make run-managed-paid-pilot-gate
```

The clean launcher runs `make db-upgrade`, the component gates above, and then
the integrated paid-pilot smoke. When `--evidence-out` is supplied through
`MANAGED_PAID_PILOT_GATE_ARGS`, the generated redacted note records each step,
result, and exit code.

The integrated smoke submits a real `/v1/jobs` request, enqueues through
Redis, executes through the in-process worker pool with a tiny generated
subprocess, persists terminal state in Postgres, mirrors artifacts to
S3-compatible storage, verifies repository-backed artifact fetch/delete
semantics after `app.JOBS.clear()`, and proves a separate abandoned active row
sweeps to `worker_lost`.
It also proves persisted SSE replay with `Last-Event-ID: 0` after the runtime
job cache is cleared.

## Multi-Host Worker Mode

Local/default backend operation keeps the in-process worker pool enabled.
Managed pilot deployments that run workers on separate hosts can disable that
pool on backend hosts and start explicit worker processes:

```bash
TP_ORCHESTRATOR_IN_PROCESS_WORKERS_ENABLED=0 make run-backend-local-noreload
TP_ORCHESTRATOR_STATE_BACKEND=postgres \
TP_ORCHESTRATOR_QUEUE_BACKEND=redis \
TP_DATABASE_URL=postgresql+asyncpg://... \
TP_REDIS_URL=redis://... \
make run-orchestrator-worker
```

The worker process consumes the existing Redis broker, uses the existing
Postgres job repository and event store, finalizes artifacts through the
configured artifact store, and drains on `SIGINT`/`SIGTERM`.

Opt-in distributed authority requires the full migration chain through
`0003_dispatch_authority`, `0004_bounded_event_replay`, and
`0005_generation_cleanup`. Follow the
[Postgres orchestrator runbook](../runtimes/orchestrator-postgres.md) for
configuration, authority, recovery, and rollback. Workers revalidate the frozen
plan's input, output, model/runtime, and secondary data paths against their
current policy before creating an attempt directory or starting a subprocess.

If a broker heartbeat fails, the worker signals cancellation to its executor
immediately because lease ownership cannot be confirmed. The executor must
observe that signal and stop its subprocess; the worker waits for executor
cleanup before releasing the lease. Release failures remain visible in worker
logs and leave the lease for the existing reclaim sweep. Acquisition failures
use the normal capped polling backoff so the worker can recover when Redis
returns; they never trigger execution without a confirmed lease.

## Pilot Tenant, Admission, And Audit Mode

Tenant mode is opt-in and preserves the single-tenant compatibility path when
disabled:

```text
TP_PILOT_CONTROL_PLANE_ENABLED=1
# Set on both backend and frontdoor, using the same dedicated random secret:
TP_FRONTDOOR_IDENTITY_SECRET=<at-least-32-random-UTF-8-bytes>
# Backend only: canonical verified Access email to one tenant:
TP_PILOT_ACTOR_TENANTS_JSON={"acme@example.com":"pilot_acme","beta@example.com":"pilot_beta"}
TP_PILOT_ALLOWED_TENANTS=pilot_acme,pilot_beta
TP_PILOT_ALLOWED_PIPELINES=lux-depth-v3
TP_PILOT_MAX_ACTIVE_JOBS_PER_TENANT=2
```

Enable `TP_PILOT_CONTROL_PLANE_ENABLED=1` on both backend and frontdoor. Backend
API-key authentication remains required. The frontdoor signs the authenticated
Access/session actor after the existing CSRF checks, using a dedicated secret
that must differ from the backend API key. The assertion expires after 60 seconds
(with 5 seconds of future clock tolerance) and binds the HTTP method and exact
encoded path/query. Keep clocks synchronized and the backend origin restricted
to the authenticated frontdoor/service network; use TLS between hosts. Identity
assertions must never be logged or exposed to the browser.

The backend derives membership from `TP_PILOT_ACTOR_TENANTS_JSON`; no browser
header or request-body tenant selector grants membership. Unknown actors receive
403. Browser-provided actor, assertion and tenant headers are stripped by the
frontdoor, including a configured legacy `TP_PILOT_TENANT_HEADER`. Direct
conflicting tenant headers receive 403. Missing/invalid assertions receive 401.
Missing, malformed or ambiguous mappings and missing/weak/reused identity secrets
fail closed with 503; `/ready` reports `ok: false` and `/v1/readiness` reports an
auth configuration error before job submission. Frontdoor startup preflight also
rejects missing/weak/reused identity secrets when tenant mode is enabled.

Migration: disabled mode remains the existing single-tenant path. To enable
isolation, provision membership and the shared identity secret on the backend and
frontdoor together, then enable tenant mode on both. Existing pilot clients that
send only an API key and `x-tp-tenant-id` must move to the authenticated frontdoor;
there is no unsigned selector fallback. Existing tenant-tagged jobs retain their
namespace. Untagged historical jobs are not visible in tenant mode. Rotate the
identity secret on both services together; existing sessions can obtain fresh
assertions without changing tenant membership.

Job list/detail/events/artifact/cancel surfaces are tenant-filtered, artifact store
keys are prefixed with the tenant id, active-job quota is enforced per tenant,
pipeline entitlement uses `TenantPolicy.allowed_node_types`, and
all explicitly supplied job-data paths (including archive manifests, indexes,
archive roots, policies, output overrides and Lux sidecars) must remain inside
the tenant workspace or tenant CAS roots. Checks run before config preview can
inspect file content, and both snake-case and camel-case aliases are checked.
Explicit model/interpreter paths must be in the tenant namespace or existing
server-owned model roots: SAM2 uses its trusted `models/sam2` and `checkpoints`
roots; FastVLM uses its configured runtime root. They retain all downstream
checksum and runtime provenance checks. Named FastVLM roles remain server
selected. This also prevents model selectors from probing another tenant's
files during readiness checks. Tenant clients must copy custom policies/data
into their namespace instead of referencing another tenant.

Audit v1 writes append-only operational events to the same Postgres database as
job state. Run `make db-upgrade` before enabling tenant/audit mode so the
`operational_audit_events` table exists. Audited actions currently include
tenant admission decisions, job create/cancel, artifact fetch/delete, and config
preview, plus the protected `/v1/readiness` operator readiness surface.

## Bounded Event Replay

Postgres keeps at most `TP_ORCHESTRATOR_EVENT_RETENTION_PER_JOB` events per job
on each committed append (default `4096`, matching the memory store). The value
must be a positive integer; zero is rejected. Use the same value on all API and
worker hosts. This bounds replay history, not the separate append-only operational
audit/evidence stores. Normal expired-job cleanup removes replay rows and their
counter; it preserves immutable dispatch tombstones and refuses active dispatches.

Run `make db-upgrade` with writers stopped before deploying this version. Migration
`0004_bounded_event_replay` seeds one monotonic counter per existing job from its
highest event sequence and widens event sequences to bigint. The next append
prunes that job's retained history to the configured cap; old inactive histories
remain until normal job retention removes them. The counter survives pruning, so
neither a restart nor removal of replay rows can reuse a reconnect cursor. Event
allocation, insertion and pruning share the caller transaction, including fenced
terminal publication. Rollback restores all three. The migration's downgrade
keeps the widened sequence column to avoid narrowing valid cursor values.

Postgres SSE streams poll committed events at one-second intervals, including
changes published on other hosts. Each stream sends a sequence at most once and
uses only committed `done` events. A temporary replay failure retries without
synthesizing terminal success. Database connections are released after reading a
bounded batch, before data is sent to a slow client. Memory mode retains its
existing local-queue behavior.

`Last-Event-ID` remains a non-negative cursor, with `0` requesting all retained
events. When retention has removed intervening events, the stream emits the SSE
comment `: replay-gap requested_after=<cursor> first_available=<seq>` before the
oldest available event. Comments introduce no new event name or payload envelope;
native EventSource ignores them, while raw stream consumers can observe the gap.
Clients can also detect the sequence jump and refresh the current job detail.
Replay is recovery assistance, never complete audit evidence. If the cursor
already covers a committed terminal event, the stream closes without duplicating
it.

## Startup Order

1. Provision or start Postgres, Redis, and S3-compatible storage.
2. Create the S3 bucket.
3. Export the service env vars.
4. Run `make db-upgrade` against `TP_DATABASE_URL`.
5. Start the backend and frontdoor processes.
6. Run the component gates.
7. Run `make test-paid-pilot-services-contract`.

## Rollback

For a failed pilot deployment:

1. Stop new job admission at the frontdoor or load balancer.
2. Drain active work before stopping worker processes. In distributed mode,
   abandoned attempts must reach fenced terminal state through Postgres expiry
   reconciliation before rolling back; legacy restart sweeping does not own
   durable attempts.
3. Roll all backend/frontdoor/worker images together to a validated release that
   understands the installed dispatch schema, signed tenant identity, canonical
   plans and generation pointers. A legacy raw-command release is not a compatible
   rollback target after this cutover; keep admission closed and repair forward
   if no compatible release exists. Do not translate durable attempts back into
   raw queue commands or drop their immutable records to enable rollback.
4. Re-run `make db-upgrade` only if the rollback release requires it.
5. Re-run the component gates and integrated pilot gate before reopening admission.

Keep the original shared execution root mounted and configured until both active
work and terminal private-workspace cleanup have finished.

## Backup And Restore

Postgres is authoritative for job state. Use provider-native point-in-time restore or snapshots before deployments that change schema or retention behavior.

Redis owns delivery leases and frontdoor sessions. In distributed mode,
Postgres owns admission, attempt fences, and artifact visibility; Redis locators
cannot authorize work by themselves. After restoring an old Redis snapshot,
reconcile expired attempts and resend committed pending outbox deliveries from
Postgres before reopening admission. Do not rewrite durable attempts through
legacy restart recovery. Treat Redis persistence as operational durability,
not long-term audit history.

S3-compatible storage owns artifact bytes. Bucket lifecycle policies must not delete objects before the orchestrator retention cleanup has either deleted artifacts intentionally or marked deletion retry metadata.

## Secret Rotation

Rotate in this order:

1. Add the new secret to the managed service.
2. Deploy backend/frontdoor processes with the new env var values.
3. Confirm `/ready` and the paid-pilot gates pass.
4. Revoke the old secret from the managed service.
5. Restart processes that may have latched client construction failures.

## Known Fail-Closed Errors

- `JOB_REPOSITORY_UNAVAILABLE`: Postgres repository construction or operation failed. The repository construction failure is latched in process state; restart the backend after fixing `TP_DATABASE_URL` or credentials.
- `QUEUE_UNAVAILABLE`: Redis queue broker construction or enqueue failed. Do not fall back to in-process dispatch.
- `ARTIFACT_STORE_UNAVAILABLE`: S3-compatible artifact store construction/readiness failed. Artifact fetch/delete routes must fail closed before mutating storage.
- Frontdoor Redis session failures: managed-mode frontdoor session reads/writes fail closed rather than silently downgrading multi-instance deployments to local SQLite.

## Non-Goals

This gate does not add Terraform, Helm, metrics dashboards, billing, or
production secret management. Opt-in distributed admission is atomic across
instances through Postgres capacity reservations and fenced attempts.
Durable SSE replay is a supported `Last-Event-ID` job-events contract.
Multi-host workers and tenant/audit mode are opt-in runtime surfaces;
provider-specific deployment manifests remain separate follow-up work.
