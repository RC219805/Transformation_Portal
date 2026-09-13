# Local Development Runbook

This runbook describes the canonical local Transformation Portal stack: how to
generate the shared API key, start the backend with safe reload boundaries,
launch the managed frontdoor, and identify the scope of process shutdown.

## Quick start

```bash
# 1. Generate /tmp/tp-local-http-all-on.env (idempotent; reuses an existing key).
./scripts/dev/write_local_env.sh

# 2. In every shell that runs the backend or the frontdoor:
source /tmp/tp-local-http-all-on.env

# 3. Start the backend (terminal A).
make run-backend-local

# 4. Start the frontdoor (terminal B, after backend /ready succeeds).
source /tmp/tp-local-http-all-on.env
./scripts/setup/ensure_node_version.sh
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

## Managed launcher and local credentials

`make dev-start` writes and sources the shared environment, invokes
`stop_local_stack.sh`, starts the backend, waits for `/ready`, and starts the
frontdoor. It does not seed users automatically. Run `make seed-frontdoor-user`
once before using the launcher; its defaults are the development-only
`smoke-admin` fixture and the password configured by `TP_FRONTDOOR_PASSWORD`
(or the Makefile fixture default). Use the seeded fixture for local login.

The launcher replaces listeners on ports 8000, 3000, 8001, and 3002. Its
shutdown helper also targets matching orphan Uvicorn processes and can escalate
from TERM to KILL. `make dev-stop` and launcher cleanup have the same scope;
inspect existing work before invoking them. Separate Make invocations above
are useful when processes should remain under individual terminal control.

Launcher logs are `/tmp/tp-backend.log` and `/tmp/tp-frontdoor.log`; override
with `TP_DEV_BACKEND_LOG` and `TP_DEV_FRONTDOOR_LOG`. Direct Make launches
write to the invoking terminal unless redirected.

The environment writer preserves an existing API key, but rewrites the file's
managed defaults. It does not preserve arbitrary edits. Key rotation requires
restarting both processes with the new shared environment. Startup flags and
server defaults do not rewrite an existing browser-local Build draft; inspect
and save its settings through the UI. Profiles are actor-scoped in managed mode.

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

Readiness establishes service/configuration probes only. A completed job and
its verified artifacts are separate evidence; no model inference is established
by either health URL.

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
