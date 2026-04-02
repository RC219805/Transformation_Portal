# Portal Secure Front Door Quickstart

## Topology

The secure front door is a separate Node app in `web/secure-landing/`.

- Browser traffic goes to the front door on one origin.
- The front door proxies `/portal` and `/v1/*` to FastAPI server-to-server.
- The backend API key stays on the front door and is never exposed to the browser in managed mode.
- The FastAPI origin remains the system of record for `GET /`, `GET /ready`, and `/v1/*`.

In production, place the front door behind Cloudflare Tunnel + Access and keep the FastAPI origin off the public browser path.

## Required Environment

Create front-door env vars from `web/secure-landing/.env.example`.

```bash
export TP_FASTAPI_ORIGIN="http://127.0.0.1:8000"
export TP_BACKEND_API_KEY="replace-with-strong-backend-token"
export TP_FRONTDOOR_USERS_JSON='[{"username":"admin","password_hash":"$argon2id$...","access_email":"admin@example.com","role":"admin"}]'
export TP_FRONTDOOR_SESSION_DB="/tmp/transformation-portal-frontdoor-sessions.db"
export TP_ALLOW_LOCAL_ACCESS_BYPASS=1
```

Notes:
- `TP_FRONTDOOR_USERS_JSON` is the v1 credential source and must contain Argon2id hashes.
- `TP_ALLOW_LOCAL_ACCESS_BYPASS=1` is for local development only and is honored only when `NODE_ENV=development`.
- Production login expects both a valid Cloudflare Access identity and a matching username/password pair.

FastAPI still needs its own backend secret for machine and proxy authentication:

```bash
export TP_API_KEY="replace-with-strong-backend-token"
```

## Local Development

Start the FastAPI origin first:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Start the front door in a second shell:

```bash
cd web/secure-landing
npm install
npm run dev
```

Open `http://127.0.0.1:3000/`.

Route ownership:
- `GET /` redirects to `/portal` when authenticated or `/login` otherwise.
- `GET /login` serves the separate login page with the video background.
- `GET /portal` proxies the existing FastAPI portal UI.
- `GET /portal/bootstrap` returns the managed-mode bootstrap contract for the browser UI.
- `/v1/*` stays same-origin at the front door and is proxied to FastAPI with server-side secret injection.
- `GET /healthz` reports front-door readiness plus backend reachability.

## Portal Modes

The operator UI now supports two modes:

- `managed`
  - Served through the front door.
  - Hides the API key input.
  - Clears stored browser API keys.
  - Sends CSRF headers on unsafe requests.
  - Uses same-origin `/v1/*` without browser-visible backend credentials.
- `direct_debug`
  - Served directly from the FastAPI origin for local troubleshooting.
  - Keeps the existing API-key workflow.
  - Must not be treated as the normal production browser path.

FastAPI now exposes `GET /portal/bootstrap` for standalone `direct_debug` startup. The front door exposes its own `GET /portal/bootstrap` for managed mode.

## Cloudflare Production Notes

- Put the front door behind Cloudflare Tunnel + Access.
- Keep the FastAPI origin non-public or otherwise inaccessible to end-user browsers.
- Ensure Cloudflare Access identity reaches the front door origin, and validate that identity at the origin unless Tunnel is already enforcing Access there.
- Do not route normal browser traffic directly to FastAPI.

## Validation

Front door checks:

```bash
cd web/secure-landing
npm test
npm run build
```

FastAPI contract gate:

```bash
make test-orchestrator-contract
```
