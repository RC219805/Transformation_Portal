# Phase 5.A Paid-Pilot Live Acceptance - 2026-05-14

## Status

Local Compose acceptance passed on follow-up fix commit `5e0632279`.

The merged Phase 5.A gate commit `cd9bf5a679d040c2af14acd8e97dbe7ee9ed11d5` was used as the baseline, but the first live run exposed three gate defects:

- Redis lease deadlines returned lower precision than the ZSET score used for expiry.
- Postgres per-job concurrent updates and log appends could exhaust optimistic retries under the contract's 50-way contention tests.
- The integrated smoke could observe terminal runtime state before artifact finalization and could let lifecycle-only durable metadata hide cached indexed artifacts.

Those defects are fixed in `5e0632279`; the live gate passes with those fixes.

## Local Service Surface

Environment loaded from `docs/deployment/paid-pilot.env.example`.

Runtime tooling:

- Docker CLI: `29.5.0`
- Docker Compose: `5.1.3`
- Colima: `0.10.1`
- Node.js: `v22.22.2`

Local services:

| Service | Image | Image digest |
| --- | --- | --- |
| Postgres | `postgres:16-alpine` | `sha256:890480b08124ce7f79960a9bb16fe39729aa302bd384bfd7c408fee6c8f7adb7` |
| Redis | `redis:7-alpine` | `sha256:6ab0b6e7381779332f97b8ca76193e45b0756f38d4c0dcda72dbb3c32061ab99` |
| MinIO | `minio/minio:latest` | `sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e` |
| MinIO client | `minio/mc:latest` | `sha256:a7fe349ef4bd8521fb8497f55c6042871b2ae640607cf99d9bede5e9bdf11727` |

`docker-compose --profile paid-pilot ps` reported Postgres, Redis, and MinIO healthy on loopback ports `5432`, `6379`, `9000`, and `9001`.

## Validation

Migration:

```text
make db-upgrade
PASS - Alembic connected to Postgres and reported transactional DDL.
```

Focused seam checks:

```text
TP_TEST_REDIS_URL=redis://127.0.0.1:6379/0 .venv/bin/python -m pytest -q tests/orchestrator/test_queue_contract.py -m unit -k server_time_shares_domain
PASS - 2 passed, 36 deselected

TP_TEST_POSTGRES_URL=postgresql+asyncpg://tp:tp_dev_password@127.0.0.1:5432/transformation_portal .venv/bin/python -m pytest -q tests/orchestrator/test_repository_contract.py -m unit -k "concurrent_updates or concurrent_append_log"
PASS - 4 passed, 49 deselected

.venv/bin/python -m pytest -q tests/test_app_orchestrator_runtime.py -k overlay_runtime_state
PASS - 4 passed, 261 deselected

TP_RUN_PAID_PILOT_SERVICES_CONTRACT=1 .venv/bin/python -m pytest -q tests/orchestrator/test_paid_pilot_services_contract.py -m unit -k backend_services_compose
PASS - 1 passed, 1 deselected
```

Full gates:

```text
make test-paid-pilot-services-contract
PASS
- Postgres repository contract: 271 passed, 2 skipped
- Postgres app authority smoke: 7 passed
- Redis QueueBroker contract: 38 passed
- ArtifactStore local + live S3/MinIO contract: 140 passed
- Frontdoor Redis SessionStore contract: 2 passed
- Integrated paid-pilot backend smoke: 2 passed

make test-orchestrator-contract
PASS - 720 passed, 9 skipped

git diff --check
PASS
```

## Acceptance Result

Phase 5.A is locally validated on the follow-up fix branch. The gate proves the composed local pilot stack:

- Postgres `JobRepository`
- Redis `QueueBroker`
- Redis frontdoor `SessionStore`
- S3-compatible MinIO `ArtifactStore`
- `/v1/jobs` create through Redis broker
- worker subprocess completion
- Postgres-backed terminal state after `app.JOBS.clear()`
- S3 redirect artifact fetch
- artifact delete and `410 ARTIFACT_DELETED`
- abandoned active row sweep to `worker_lost`

## Known Limits

This is local Compose validation only. Managed-service validation must rerun the same gate with provider Postgres, Redis, and S3-compatible endpoints. The gate is still manual and opt-in; durable SSE replay is now a supported `Last-Event-ID` job-events contract and must remain covered by the integrated gate. Provider-specific Terraform, Helm, globally atomic multi-instance admission, broad observability, billing, and production secret management remain follow-up work. Multi-host workers and tenant/audit mode are opt-in runtime surfaces.

## Merged Baseline

PR #1784 was squash-merged as
`07a3e8e847dee4a6e1ccf46d6dcd80b612fe3753`.

This preserves the historical first-run failure evidence above while recording
the merged local Phase 5.A acceptance baseline. Managed-provider validation
remains pending and must be recorded separately after the same gate passes
against provider endpoints.
