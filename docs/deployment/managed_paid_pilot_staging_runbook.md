# Managed Paid-Pilot Staging Runbook

## Status

Phase 5.A local Compose validation is complete at merge commit
`07a3e8e847dee4a6e1ccf46d6dcd80b612fe3753`.

Managed-provider validation is not yet complete. It is complete only after
`make test-paid-pilot-services-contract` passes against provider-managed
Postgres, Redis queue storage, Redis frontdoor session storage, and
S3-compatible artifact storage.

## Acceptance Baseline

The local acceptance baseline is PR #1784, squash-merged as
`07a3e8e847dee4a6e1ccf46d6dcd80b612fe3753`.

The historical local acceptance record is
[`phase5a_paid_pilot_live_acceptance_2026-05-14.md`](phase5a_paid_pilot_live_acceptance_2026-05-14.md).
It records the first live-gate failures, the stabilization fixes, and the
passing local Compose gate. This runbook is the provider/staging repeat of that
same gate.

## Scope

This runbook validates the currently landed paid-pilot service composition:

- Postgres-backed orchestrator `JobRepository`
- Redis-backed orchestrator `QueueBroker`
- Redis-backed managed-frontdoor `SessionStore`
- S3-compatible `ArtifactStore`
- Backend route contracts, artifact fetch/delete lifecycle, and
  `worker_lost` recovery

## Non-Goals

This runbook does not add or validate Terraform, Helm, Kubernetes manifests,
globally atomic multi-instance admission, billing, metrics dashboards, or
provider-specific worker deployment manifests. Durable job-event replay is part
of the landed orchestrator surface and is covered by the integrated gate.
Multi-host workers and tenant/audit mode are opt-in runtime surfaces.

## Managed-Service Topology

Use one staging deployment and separate disposable validation resources:

| Surface | Staging resource | Disposable validation resource |
| --- | --- | --- |
| Job state | Managed Postgres database for `TP_DATABASE_URL` | Empty managed Postgres database for `TP_TEST_POSTGRES_URL` |
| Queue broker | Managed Redis for `TP_REDIS_URL` | Disposable Redis DB or keyspace for `TP_TEST_REDIS_URL` |
| Frontdoor sessions | Managed Redis for `TP_FRONTDOOR_REDIS_URL` | Same provider class; use staging-safe keys only |
| Artifacts | S3-compatible bucket for `TP_ARTIFACT_BUCKET` | Disposable bucket for `TP_TEST_S3_BUCKET` |

The validation resources must be empty or explicitly disposable before the gate
runs.

## Environment Matrix

Use provider-neutral values only. Store real values in the staging secret
manager, not in the repository.

| Variable | Provider-neutral value |
| --- | --- |
| `TP_ORCHESTRATOR_STATE_BACKEND` | `postgres` |
| `TP_DATABASE_URL` | `postgresql+asyncpg://<user>:<password>@<host>:<port>/<database>` |
| `TP_TEST_POSTGRES_URL` | `postgresql+asyncpg://<user>:<password>@<test-host>:<port>/<test-database>` |
| `TP_ORCHESTRATOR_QUEUE_BACKEND` | `redis` |
| `TP_REDIS_URL` | `rediss://<host>:<port>/<db>` |
| `TP_TEST_REDIS_URL` | `rediss://<test-host>:<port>/<db>` |
| `TP_FRONTDOOR_SESSION_STORE` | `redis` |
| `TP_FRONTDOOR_REDIS_URL` | `rediss://<session-host>:<port>/<db>` |
| `TP_ARTIFACT_STORE` | `s3` |
| `TP_ARTIFACT_ENDPOINT_URL` | `https://<s3-compatible-endpoint>` |
| `TP_TEST_S3_URL` | `https://<test-s3-compatible-endpoint>` |
| `TP_ARTIFACT_BUCKET` | `<staging-artifact-bucket>` |
| `TP_TEST_S3_BUCKET` | `<disposable-test-artifact-bucket>` |
| `TP_ARTIFACT_REGION` | Optional; set to `<region>` when the endpoint requires an explicit region |
| `AWS_ACCESS_KEY_ID` | `<redacted>` |
| `AWS_SECRET_ACCESS_KEY` | `<redacted>` |

Use `redis://` only when the provider endpoint is intentionally non-TLS inside
a private network. Prefer `rediss://` for managed staging endpoints.

The artifact store accepts an unset `TP_ARTIFACT_REGION`. The paid-pilot
contract defaults live S3-compatible tests to `us-east-1` when the variable is
absent.

If the managed endpoint requires an explicit region, export it before running
the gates:

```bash
export TP_ARTIFACT_REGION='<region>'
```

## Disposable Test Resource Requirements

`TP_TEST_POSTGRES_URL`, `TP_TEST_REDIS_URL`, and `TP_TEST_S3_BUCKET` must point
to disposable validation resources. They must never point at production
databases, queues, buckets, or customer data.

The paid-pilot smoke resets the test Postgres schema, deletes Redis keys under
isolated prefixes, and deletes S3 objects under isolated prefixes. Treat every
`TP_TEST_*` target as destructive.

## Preflight Checklist

- The checkout is on the commit intended for staging validation.
- The virtual environment is installed and active.
- Managed Postgres, Redis, and S3-compatible services are provisioned.
- The disposable Postgres database is empty or approved for reset.
- The disposable Redis DB or keyspace contains no production data.
- The disposable S3 bucket contains no production data.
- Staging secrets are loaded from the provider secret manager into a private
  gate-only env file outside the repository.
- Backend and frontdoor processes can reach each managed endpoint.

## Startup Order

1. Provision or select the managed Postgres, Redis, and S3-compatible
   resources.
2. Create the disposable validation database, Redis namespace, and S3 bucket.
3. Write a private env file outside the repository, usually
   `/tmp/tp-managed-staging.env`.
4. Run the clean-process preflight.
5. Run the clean-process gate, which applies migrations and then invokes
   `make test-paid-pilot-services-contract`.
6. Capture validation evidence before opening paid-pilot admission.

## Clean Env File

Create a gate-only env file outside the repository. Do not use the local
`/tmp/tp-local-http-all-on.env` development profile.

```bash
umask 077
touch /tmp/tp-managed-staging.env
chmod 600 /tmp/tp-managed-staging.env
${EDITOR:-vi} /tmp/tp-managed-staging.env
```

The file must contain real provider values only. The clean launcher rejects
missing values, placeholder-like values, wrong selectors, local-development
variables, and unsafe staging/test overlap for the destructive Postgres and S3
resources.

Do not include local-development variables such as `TP_API_KEY`,
`TP_BACKEND_API_KEY`, `TP_FASTAPI_ORIGIN`, `TP_ALLOW_LOCAL_ACCESS_BYPASS`,
`TP_FRONTDOOR_SESSION_DB`, `TP_FRONTDOOR_SESSION_SCALING_MODE`,
`TP_PORTAL_DIRECT_DEBUG_COHORT_KEY`, telemetry log paths, upload staging vars,
or FastVLM runtime vars.

## Clean Env Preflight

Run the preflight from any shell. The launcher re-execs itself through `env -i`
with only `HOME`, `PATH`, `USER`, and `SHELL`, then sources the private env file:

```bash
TP_MANAGED_PAID_PILOT_ENV_FILE=/tmp/tp-managed-staging.env \
MANAGED_PAID_PILOT_GATE_ARGS=--preflight-only \
make run-managed-paid-pilot-gate
```

The expected preflight output is:

```text
missing: []
placeholder-like: []
wrong selectors: {}
leaked local-dev vars: []
unsafe managed/test overlap: {}
Managed paid-pilot clean-env preflight passed.
```

Any non-empty bucket is an environment setup blocker, not a product failure.

## Component Gates

Load provider-neutral staging and disposable test values:

```bash
source .venv/bin/activate

export TP_ORCHESTRATOR_STATE_BACKEND=postgres
export TP_DATABASE_URL='postgresql+asyncpg://<redacted>'
export TP_TEST_POSTGRES_URL='postgresql+asyncpg://<redacted-disposable-test-db>'

export TP_ORCHESTRATOR_QUEUE_BACKEND=redis
export TP_REDIS_URL='rediss://<redacted>'
export TP_TEST_REDIS_URL='rediss://<redacted-disposable-test-redis>'

export TP_FRONTDOOR_SESSION_STORE=redis
export TP_FRONTDOOR_REDIS_URL='rediss://<redacted-session-redis>'

export TP_ARTIFACT_STORE=s3
export TP_ARTIFACT_ENDPOINT_URL='https://<redacted-s3-compatible-endpoint>'
export TP_TEST_S3_URL='https://<redacted-test-s3-compatible-endpoint>'
export TP_ARTIFACT_BUCKET='<redacted-staging-artifact-bucket>'
export TP_TEST_S3_BUCKET='<redacted-disposable-test-bucket>'
export AWS_ACCESS_KEY_ID='<redacted>'
export AWS_SECRET_ACCESS_KEY='<redacted>'

make test-orchestrator-postgres-contract
make test-orchestrator-postgres-app-contract
make test-worker-redis-contract
make test-artifact-s3-contract
make test-frontdoor-redis-contract
```

The component gates isolate failures to one service seam before the integrated
gate runs.

## Integrated Paid-Pilot Gate

Run the full provider/staging composition gate with the same private env file:

```bash
TP_MANAGED_PAID_PILOT_ENV_FILE=/tmp/tp-managed-staging.env \
make run-managed-paid-pilot-gate
```

The launcher runs `make db-upgrade` against staging `TP_DATABASE_URL`, not
`TP_TEST_POSTGRES_URL`, before invoking `make test-paid-pilot-services-contract`.
Do not point `TP_DATABASE_URL` at the disposable test database.

Add `--evidence-out /tmp/tp-managed-paid-pilot-acceptance.md` through
`MANAGED_PAID_PILOT_GATE_ARGS` when a redacted acceptance note is needed. The
output must live outside the repository. Use
[`managed_provider_acceptance_note_template.md`](managed_provider_acceptance_note_template.md)
as the manual review template before promoting evidence into docs.

The integrated gate must prove `/v1/jobs` creation through Redis broker
enqueue, worker subprocess completion, Postgres-backed terminal state after
`app.JOBS.clear()`, durable SSE replay with `Last-Event-ID`, S3-compatible
artifact redirect, artifact delete lifecycle, `410 ARTIFACT_DELETED` after
delete, and abandoned active-row recovery to `worker_lost`.

## Evidence Capture

Record the following in the staging acceptance note or release record:

- Git commit under test.
- Provider names, regions, and service versions where available.
- Redacted env selector summary.
- Clean env preflight result.
- `make db-upgrade` result.
- Each component gate result.
- `make test-paid-pilot-services-contract` result.
- Any skipped lane and the reason.
- Artifact bucket, Redis, and Postgres validation resource cleanup result.

Do not record secrets, real usernames, private hostnames, customer identifiers,
or bucket names unless the release record is stored in an approved private
operations system.

## Rollback

For a failed staging validation:

1. Keep paid-pilot admission closed.
2. Stop new job submission at the frontdoor or load balancer.
3. Stop workers after active leases drain, or let restart recovery mark
   abandoned jobs `worker_lost`.
4. Roll backend and frontdoor images back to the last validated release.
5. Restore Postgres only when the failed deployment applied a bad migration or
   corrupted staging state.
6. Re-run component gates and `make test-paid-pilot-services-contract` before
   reopening admission.

## Backup And Restore

Postgres is authoritative for job state. Enable provider-native snapshots or
point-in-time restore before staging migrations and before paid-pilot cutover.

Redis owns queue leases and frontdoor sessions. Treat Redis persistence as
operational durability, not long-term audit history. If Redis is restored from
an older snapshot, reconcile active jobs through restart recovery before
enabling workers.

S3-compatible storage owns artifact bytes. Bucket lifecycle policies must not
delete objects before orchestrator retention cleanup has either intentionally
deleted artifacts or recorded deletion retry metadata.

## Secret Rotation

Rotate secrets in this order:

1. Add the new provider secret.
2. Deploy backend and frontdoor processes with the new env values.
3. Confirm `/ready`, frontdoor health, and component gates are green.
4. Run `make test-paid-pilot-services-contract`.
5. Revoke the old provider secret.
6. Restart processes that may have latched client construction failures.

## Fail-Closed Triage

| Error or symptom | First seam to inspect |
| --- | --- |
| `JOB_REPOSITORY_UNAVAILABLE` | `TP_DATABASE_URL`, credentials, network policy, migrations, and backend restart after fixing a latched repository construction failure |
| `QUEUE_UNAVAILABLE` | `TP_REDIS_URL`, queue credentials, Redis TLS mode, key prefix, lease acquire/reclaim behavior |
| Frontdoor Redis session failure | `TP_FRONTDOOR_SESSION_STORE`, `TP_FRONTDOOR_REDIS_URL`, session TTL, throttle keys, multi-instance readiness |
| `ARTIFACT_STORE_UNAVAILABLE` | S3 endpoint, bucket, region, credentials, path-style support, object permissions |
| Artifact fetch returns non-redirect | Auth, artifact metadata, S3 presign settings, response-header overrides |
| Artifact refetch does not return `410 ARTIFACT_DELETED` | Artifact lifecycle persistence, deletion metadata, repository write path |
| Active abandoned row does not become `worker_lost` | Restart sweep, worker registry, Postgres repository state |

Do not weaken fail-closed behavior to make staging pass. Fix the failing seam
or its environment configuration.

## Known Limits

- Provider validation is manual and opt-in.
- Local Phase 5.A validation is complete; provider validation is pending.
- No Terraform, Helm, Kubernetes manifests, or deployment pipeline are added by
  this runbook.
- Durable SSE replay is landed and must remain covered by the integrated gate.
- Multi-host worker execution is available as an opt-in worker process;
  provider deployment manifests remain follow-up work.
- Tenant admission, artifact-prefix isolation, per-tenant quota, and Postgres
  audit logging are opt-in pilot runtime surfaces.
- Observability dashboards, billing, and provider-specific IaC remain future
  phases.

## Follow-Up Gates

After provider validation passes, record a separate managed-provider acceptance
note with the exact commit, provider resource classes, redacted env selector
summary, command results, skipped lanes, and remaining limits.

Do not mark the managed paid pilot validated until the provider-backed
`make test-paid-pilot-services-contract` result is captured.
