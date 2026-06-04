# Managed Provider Paid-Pilot Acceptance Note Template

Use this template for redacted provider acceptance evidence. The managed gate
can generate a starter note with:

```bash
TP_MANAGED_PAID_PILOT_ENV_FILE=/tmp/tp-managed-staging.env \
MANAGED_PAID_PILOT_GATE_ARGS="--evidence-out /tmp/tp-managed-paid-pilot-acceptance.md" \
make run-managed-paid-pilot-gate
```

Keep the note outside the repository unless it has been reviewed and scrubbed.

## Provider Surface

- Validation date:
- Git commit:
- Backend host:
- Frontdoor host:
- Provider Postgres: configured / not configured
- Provider Redis queue: configured / not configured
- Provider Redis frontdoor sessions: configured / not configured
- Provider S3-compatible artifacts: configured / not configured

Do not include secrets, private hostnames, customer identifiers, usernames, or
bucket names. Use provider/environment labels instead.

## Gate Evidence

- Clean env preflight:
- `make db-upgrade`:
- `make test-orchestrator-postgres-contract`:
- `make test-orchestrator-postgres-app-contract`:
- `make test-worker-redis-contract`:
- `make test-artifact-s3-contract`:
- `make test-frontdoor-redis-contract`:
- Integrated paid-pilot smoke:

The integrated gate must prove `/v1/jobs` creation through Redis broker
enqueue, worker completion, terminal state from Postgres after
`app.JOBS.clear()`, persisted SSE replay with `Last-Event-ID`, S3-compatible
artifact fetch/delete lifecycle, `410 ARTIFACT_DELETED` after delete, and
abandoned active-row recovery to `worker_lost`.

## Follow-Ups

- Provider validation:
- Multi-host worker rollout:
- Tenant/admission/audit mode:
- Observability:
- Billing/entitlements:
- Infrastructure-as-code:
