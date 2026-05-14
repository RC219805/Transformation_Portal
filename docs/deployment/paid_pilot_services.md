# Paid-Pilot Managed-Services Smoke Gate

Phase 5.A adds an opt-in validation gate for the current paid-pilot backend topology:

- Postgres-backed orchestrator `JobRepository`
- Redis-backed orchestrator `QueueBroker`
- Redis-backed managed-frontdoor `SessionStore`
- S3-compatible `ArtifactStore`

This is not production infrastructure-as-code. It is a deterministic smoke gate for proving that the existing durable components compose under explicit service endpoints.

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
TP_ARTIFACT_REGION=...
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

The integrated smoke submits a real `/v1/jobs` request, enqueues through Redis, executes through the in-process worker pool with a tiny generated subprocess, persists terminal state in Postgres, mirrors artifacts to S3-compatible storage, verifies repository-backed artifact fetch/delete semantics after `app.JOBS.clear()`, and proves a separate abandoned active row sweeps to `worker_lost`.

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
2. Stop worker processes after active leases drain, or mark abandoned active rows via restart recovery.
3. Roll the backend/frontdoor process images back to the last validated release.
4. Re-run `make db-upgrade` only if the rollback release requires it.
5. Re-run the component gates and integrated pilot gate before reopening admission.

## Backup And Restore

Postgres is authoritative for job state. Use provider-native point-in-time restore or snapshots before deployments that change schema or retention behavior.

Redis owns queue leases and frontdoor sessions. Treat Redis persistence as operational durability, not long-term audit history. If Redis is restored from an old snapshot, reconcile active jobs through restart recovery before enabling workers.

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

This gate does not add Terraform, Helm, durable SSE replay, globally atomic multi-instance admission, metrics, audit events, tenancy, billing, or production secret management. Those remain separate follow-up phases.
