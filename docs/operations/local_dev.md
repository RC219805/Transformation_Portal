# Local Development Runbook

This runbook describes the canonical local Transformation Portal stack: how to
generate the shared API key, start the backend with safe reload boundaries,
launch the managed frontdoor, and tear everything down deterministically.

## Quick start

```bash
# 1. Generate /tmp/tp-local-http-all-on.env (idempotent; reuses an existing key).
./scripts/dev/write_local_env.sh

# 2. In every shell that runs the backend or the frontdoor:
source /tmp/tp-local-http-all-on.env

# 3. Start the backend (terminal A).
make run-backend-local

# 4. Start the frontdoor (terminal B).
make seed-frontdoor-user           # only the first time
make run-frontdoor-local
```

If `TP_API_KEY` (backend) and `TP_BACKEND_API_KEY` (frontdoor) drift apart, the
frontdoor preflight refuses to start with the diagnostic:

```
Frontdoor preflight failed: backend protected probe returned 401.
TP_BACKEND_API_KEY does not match backend TP_API_KEY.
```

To rotate the key, run `./scripts/dev/write_local_env.sh --rotate` and re-source
the file in both terminals.

## Why `make run-backend-local`?

Running raw `uvicorn app:app --reload` watches **everything** under the working
tree, so writes under `.runtime/fastvlm/`, `output/`, or `tests/` (created
during normal pipeline operation) trigger reloads in the middle of active jobs.

`make run-backend-local` invokes Uvicorn with explicit `--reload-dir`/`--reload-exclude`
flags so only `app.py` and `src/` trigger restarts. Generated runtime
directories, the front door's `.next/` build, `node_modules`, the virtualenv,
test artifacts, and `tmp/` are excluded.

For a full-stack smoke run where reloading is undesirable (e.g. before invoking
`validate-portal-browser`), use `make run-backend-local-noreload`.

## Verifying readiness

```bash
# Process liveness (no auth check).
curl -i http://127.0.0.1:8000/healthz

# Authenticated readiness (proves TP_API_KEY is correctly wired through the
# frontdoor). Returns 503 with reason=backend_auth_mismatch on key drift.
curl -s http://127.0.0.1:3000/healthz | jq '.checks.backend'
```

## Cloudflare tunnel

For ad-hoc tunneling during local demos, prefer a named Cloudflare tunnel over
a `trycloudflare.com` quick tunnel — quick tunnels rotate hostnames and have
been observed to fail intermittently over QUIC. See
`docs/operations/cloudflared_tunnel.md`.

When using any tunnel, append the tunnel hostname to `TP_TRUSTED_HOSTS` in the
canonical env file before restarting the backend; otherwise the Trusted-Host
middleware rejects the proxied request as `Invalid host header`.

## Related documentation

- Vercel/production frontdoor env: `docs/operations/frontdoor_vercel_env.md`
- Cloudflare tunnel setup: `docs/operations/cloudflared_tunnel.md`
- Frontdoor portal asset bundles (CLAUDE.md): regenerate with
  `cd web/secure-landing && npm run build:portal` after editing
  `portal-src/portal.template.js`.
