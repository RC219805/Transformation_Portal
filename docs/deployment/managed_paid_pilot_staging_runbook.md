# Managed Paid-Pilot Staging Runbook

## Status

Phase 5.A local Compose validation is complete at merge commit
`07a3e8e847dee4a6e1ccf46d6dcd80b612fe3753`.

Managed-provider validation is not yet complete. It is complete only after
`make run-managed-paid-pilot-gate` passes against provider-managed
Postgres, Redis queue storage, Redis frontdoor session storage, and
S3-compatible artifact storage.

## Acceptance Baseline

The local acceptance baseline is PR #1784, squash-merged as
`07a3e8e847dee4a6e1ccf46d6dcd80b612fe3753`.

The historical local acceptance record is
[`phase5a_paid_pilot_live_acceptance_2026-05-14.md`](phase5a_paid_pilot_live_acceptance_2026-05-14.md).
It records the first live-gate failures, the stabilization fixes, and the
passing local Compose gate. This runbook repeats that gate on staging and adds
the orchestrator authority contracts.

## Scope

This runbook validates the currently landed paid-pilot service composition:

- Postgres-backed orchestrator `JobRepository`
- Redis-backed orchestrator `QueueBroker`
- Redis-backed managed-frontdoor `SessionStore`
- S3-compatible `ArtifactStore`
- Backend route contracts, artifact fetch/delete lifecycle, and
  `worker_lost` recovery
- Managed dispatch authority, including a tenant job on an external worker,
  and durable keyed job submissions

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
| Job state | Managed Postgres database for `TP_DATABASE_URL` | Disposable reset database for `TP_TEST_POSTGRES_URL` and separate dedicated `*_test` database for `TP_DISPATCH_TEST_DATABASE_URL` |
| Queue broker | Managed Redis for `TP_REDIS_URL` | Disposable, non-staging Redis database or keyspace for `TP_TEST_REDIS_URL` |
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
| `TP_DISPATCH_TEST_DATABASE_URL` | `postgresql+asyncpg://<user>:<password>@<test-host>:<port>/<dispatch-test-database>_test` |
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

For all three Postgres URLs, use the `postgresql+asyncpg` scheme, a host,
an optional valid port, and a literal database name in the path. Connection
parameters such as TLS settings are allowed. Provide the database and endpoint
in the URL itself.

The artifact store accepts an unset `TP_ARTIFACT_REGION`. The paid-pilot
contract defaults live S3-compatible tests to `us-east-1` when the variable is
absent.

If the managed endpoint requires an explicit region, export it before running
the gates:

```bash
export TP_ARTIFACT_REGION='<region>'
```

## Disposable Test Resource Requirements

`TP_TEST_POSTGRES_URL`, `TP_DISPATCH_TEST_DATABASE_URL`, `TP_TEST_REDIS_URL`,
and `TP_TEST_S3_BUCKET` must point to disposable validation resources. Keep
them separate from staging, production, and customer data.

The paid-pilot smoke resets the schema at `TP_TEST_POSTGRES_URL`. The service
authority phase migrates `TP_DISPATCH_TEST_DATABASE_URL` and resets test data
there. Use distinct database names for staging, reset, and dispatch even when
the endpoints use different hosts or aliases. The dispatch database name must
end in `_test`. Both phases reuse disposable `TP_TEST_REDIS_URL` and clean up
only their own key prefixes. The integrated smoke deletes S3 objects under its
own prefix.

## Preflight Checklist

- The checkout is on the commit intended for staging validation.
- The virtual environment is installed and active.
- Managed Postgres, Redis, and S3-compatible services are provisioned.
- Both disposable Postgres databases are empty or approved for reset.
- The disposable Redis endpoint is separate from the staging queue and
  frontdoor session stores and contains only test data.
- The disposable S3 bucket contains no production data.
- Staging secrets are loaded from the provider secret manager into a private
  gate-only env file outside the repository.
- Backend and frontdoor processes can reach each managed endpoint.

## Startup Order

1. Provision or select the managed Postgres, Redis, and S3-compatible
   resources.
2. Create the two disposable validation databases, Redis resource, and S3
   bucket.
3. Write a private env file outside the repository, usually
   `/tmp/tp-managed-staging.env`.
4. Run the clean-process preflight.
5. Run the full clean-process gate to verify live Postgres identities, apply
   migrations, and run the component, authority, and integrated gates.
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
variables, invalid Postgres targets, and unsafe staging/test overlap for the
destructive Postgres and S3 resources.

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

This mode validates configuration without contacting providers. The expected
preflight output is:

```text
missing: []
placeholder-like: []
wrong selectors: {}
leaked local-dev vars: []
invalid Postgres targets: {}
unsafe managed/test overlap: {}
Managed paid-pilot clean-env preflight passed.
```

Resolve any reported configuration issue before running the full gate.

## Component Gates

Load provider-neutral staging and disposable test values:

```bash
source .venv/bin/activate

export TP_ORCHESTRATOR_STATE_BACKEND=postgres
export TP_DATABASE_URL='postgresql+asyncpg://<redacted>'
export TP_TEST_POSTGRES_URL='postgresql+asyncpg://<redacted-disposable-test-db>'
export TP_DISPATCH_TEST_DATABASE_URL='postgresql+asyncpg://<user>:<password>@<test-host>:<port>/<dispatch-database>_test'

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

python scripts/validation/run_orchestrator_service_contracts.py provider-plain
make test-orchestrator-postgres-app-contract
make test-worker-redis-contract
make test-artifact-s3-contract
make test-frontdoor-redis-contract
```

The component gates isolate failures to one service seam before the integrated
gate runs. The `provider-plain` phase limits the reset database to the legacy
Postgres suites. Migrated-schema photographic review tests are outside the
provider gate.

## Integrated Paid-Pilot Gate

Run the full provider/staging composition gate with the same private env file:

```bash
TP_MANAGED_PAID_PILOT_ENV_FILE=/tmp/tp-managed-staging.env \
make run-managed-paid-pilot-gate
```

The launcher first connects to all three Postgres targets and verifies the
database identities with read-only queries. It also compares the disposable
Redis target to both staging Redis targets. These checks must pass before any
migration. The launcher then runs `make db-upgrade` against staging
`TP_DATABASE_URL`, runs the component gates, and invokes the service authority
contracts on the dedicated `TP_DISPATCH_TEST_DATABASE_URL` and disposable
`TP_TEST_REDIS_URL`. The integrated paid-pilot smoke runs last.

Add `--evidence-out /tmp/tp-managed-paid-pilot-acceptance.md` through
`MANAGED_PAID_PILOT_GATE_ARGS` when a redacted acceptance note is needed. The
output must live outside the repository and records each launcher step with a
redacted command, result, and exit code. Use
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
- Destructive-test database safety preflight result.
- `make db-upgrade` result.
- Each component gate result.
- Orchestrator service authority contracts result.
- Integrated paid-pilot smoke result.
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
6. Re-run the full provider gate before reopening admission.

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
4. Run the full provider gate.
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

Record a passing provider-backed `make run-managed-paid-pilot-gate` result
before marking the managed paid pilot validated.
