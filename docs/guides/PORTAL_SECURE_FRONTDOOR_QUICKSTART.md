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
export TP_FRONTDOOR_USERS_FILE="/absolute/path/to/frontdoor-users.json"
export TP_FRONTDOOR_SESSION_DB="/tmp/transformation-portal-frontdoor-sessions.db"
export TP_CF_ACCESS_TEAM_DOMAIN="https://your-team.cloudflareaccess.com"
export TP_CF_ACCESS_AUD="replace-with-access-application-aud"
export TP_ALLOW_LOCAL_ACCESS_BYPASS=0
```

Notes:
- `TP_FRONTDOOR_USERS_FILE` is the v1 credential source and should point to a secret-managed JSON file containing an array of `{ username, password_hash, access_email, role }`.
- `TP_FRONTDOOR_USERS_JSON` remains available only as a local-dev and test fallback when a file is not supplied.
- `TP_CF_ACCESS_TEAM_DOMAIN` must point at the Cloudflare Access team domain used to mint `Cf-Access-Jwt-Assertion`.
- `TP_CF_ACCESS_AUD` must match the Access application audience tag for this front door.
- `TP_ALLOW_LOCAL_ACCESS_BYPASS=1` is for local development only and is honored only when `NODE_ENV=development`.
- Production login expects a valid `Cf-Access-Jwt-Assertion`, a matching username/password pair, and issuer/audience validation against the configured Access team domain and audience tag.
- Development uses an HTTP-safe `tp_session` cookie. Production uses `__Host-tp_session` with `Secure`.

## Runtime Requirements

The front door is a Node app.

- Recommended runtime: Node `22.x` LTS
- Supported install/build range: `>=20.9.0 <21 || >=22 <26`
- The package now enforces this during `npm install`

If you use `nvm`, the app includes `.nvmrc`:

```bash
cd web/secure-landing
nvm use
```

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
- `managed_unavailable`
  - Fail-closed managed state used when `/portal/bootstrap` cannot establish a valid managed session.
  - Keeps the API key input disabled and hidden.
  - Clears stored browser API keys and blocks privileged dispatch actions until managed auth recovers.
  - Treats `401/403` bootstrap failures as session/auth failures and redirects back to `/login`.
- `direct_debug`
  - Served directly from the FastAPI origin for local troubleshooting.
  - Keeps the existing API-key workflow.
  - Must not be treated as the normal production browser path.

FastAPI now exposes `GET /portal/bootstrap` for standalone `direct_debug` startup. The front door exposes its own `GET /portal/bootstrap` for managed mode.

## Cloudflare Production Notes

- Put the front door behind Cloudflare Tunnel + Access.
- Keep the FastAPI origin non-public or otherwise inaccessible to end-user browsers.
- Production authentication must be driven by `Cf-Access-Jwt-Assertion`, not by convenience headers such as `cf-access-authenticated-user-email` or `x-access-email`.
- The front door verifies the Access JWT against `${TP_CF_ACCESS_TEAM_DOMAIN}/cdn-cgi/access/certs`, requires the normalized issuer derived from `TP_CF_ACCESS_TEAM_DOMAIN`, requires `TP_CF_ACCESS_AUD`, and treats the JWT `email` claim as the canonical Access identity.
- Convenience headers may be logged or inspected after successful JWT validation, but they are not trusted as identity on their own and are never valid authentication fallbacks.
- If Cloudflare Tunnel is used, require origin-side Access enforcement as well:

```yaml
originRequest:
  access:
    required: true
    teamName: <your-team-name>
    audTag:
      - <Access-application-audience-tag>
```

- Keep app-side JWT verification enabled even when Tunnel origin enforcement is active.
- Do not route normal browser traffic directly to FastAPI.
- If the front door is hosted on Vercel, Cloudflare must still front the user-facing hostname, the app must continue to verify the Access JWT, and deployment or preview URLs must be protected with equivalent controls such as Vercel Deployment Protection.
- The v1 session store is SQLite-backed. Production should assume a single-instance deployment or shared persistent storage for `TP_FRONTDOOR_SESSION_DB`; do not place the session database on ephemeral disk in a horizontally scaled setup.

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
