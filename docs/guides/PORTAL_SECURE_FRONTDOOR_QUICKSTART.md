# Portal Secure Front Door Quickstart

## Topology

The secure front door is a separate Node app in `web/secure-landing/`.

- Browser traffic goes to the front door on one origin.
- The front door proxies `/portal` and `/v1/*` to FastAPI server-to-server.
- The backend API key stays on the front door and is never exposed to the browser in managed mode.
- The front door serves `GET /` directly; FastAPI remains the backend system of record for `GET /ready` and `/v1/*`.
- `GET /healthz` is the managed front-door health contract; FastAPI `GET /ready` remains the backend readiness contract and is not mirrored under `/api/*` by default.

In production, place the front door behind Cloudflare Tunnel + Access and keep the FastAPI origin off the public browser path.

## Required Environment

Create front-door env vars from `web/secure-landing/.env.example`.

```bash
export TP_FASTAPI_ORIGIN="http://127.0.0.1:8000"
export TP_BACKEND_API_KEY="replace-with-strong-backend-token"
export TP_FRONTDOOR_USERS_FILE="/absolute/path/to/frontdoor-users.json"
export TP_FRONTDOOR_SESSION_DB="/tmp/transformation-portal-frontdoor-sessions.db"
export TP_FRONTDOOR_SESSION_SCALING_MODE="single_instance"
export TP_CF_ACCESS_TEAM_DOMAIN="https://your-team.cloudflareaccess.com"
export TP_CF_ACCESS_AUD="replace-with-access-application-aud"
export TP_ALLOW_LOCAL_ACCESS_BYPASS=0
```

Notes:
- `TP_FRONTDOOR_USERS_FILE` is the v1 credential source and should point to a secret-managed JSON file containing an array of `{ username, password_hash, access_email, role }`.
- `TP_FRONTDOOR_USERS_JSON` remains available only as a local-dev and test fallback when a file is not supplied.
- `TP_CF_ACCESS_TEAM_DOMAIN` must point at the Cloudflare Access team domain used to mint `Cf-Access-Jwt-Assertion`.
- `TP_CF_ACCESS_AUD` must match the Access application audience tag for this front door.
- `TP_FRONTDOOR_SESSION_SCALING_MODE` should stay `single_instance` for the current SQLite-backed front door. Declaring `multi_instance` or `ephemeral_runtime` intentionally fails readiness until a real external session store exists.
- `TP_ALLOW_LOCAL_ACCESS_BYPASS=1` is for local development only and is honored only when `NODE_ENV=development`.
- Production login expects a valid `Cf-Access-Jwt-Assertion`, a matching username/password pair, and issuer/audience validation against the configured Access team domain and audience tag.
- Development uses an HTTP-safe `tp_session` cookie. Production uses `__Host-tp_session` with `Secure`.

## Runtime Requirements

The front door is a Node app.

- Required runtime: Node `22.x`
- Install, dev, test, build, and start now fail fast outside Node 22.x.
- Runtime guardrails also verify native addons like `better-sqlite3` and `argon2` before the front door boots.

If you use `nvm`, the app includes `.nvmrc`:

```bash
cd web/secure-landing
nvm use 22
```

FastAPI still needs its own backend secret for machine and proxy authentication:

```bash
export TP_API_KEY="replace-with-strong-backend-token"
export TP_BACKEND_API_KEY="$TP_API_KEY"
```

## Local Development

Start the FastAPI origin first:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Start the front door in a second shell:

```bash
make run-frontdoor-local
```

Open `http://localhost:3000/`.

The canonical launcher:
- exports the known-good local managed env (`NODE_ENV=development`, `TP_ALLOW_LOCAL_ACCESS_BYPASS=1`)
- pins the supported SQLite session posture with `TP_FRONTDOOR_SESSION_SCALING_MODE=single_instance`
- reuses `TP_API_KEY` as `TP_BACKEND_API_KEY` when needed
- verifies FastAPI readiness first
- refuses to start if `localhost:3000` is already occupied instead of letting Next.js drift to `:3001`

Route ownership:
- `GET /` serves the public Dynamic Neural Access homepage, even for authenticated operators.
- `GET /login` serves the separate login page with the video background and boots the anonymous session cookie that binds the hidden CSRF token before credential submission.
- `GET /portal` proxies the existing FastAPI portal UI.
- `GET /portal/video/*` proxies the portal background video asset with cache-friendly headers.
- `GET /portal/bootstrap` returns the managed-mode bootstrap contract for the browser UI.
- `/v1/*` stays same-origin at the front door and is proxied to FastAPI with server-side secret injection.
- `GET /healthz` reports front-door readiness plus backend reachability.
- `GET /healthz` now returns structured readiness checks under `checks.backend`, `checks.access_config`, `checks.user_source`, `checks.session_store`, and `checks.session_scaling`; required production failures return `503`.

Static front-door assets:
- `/brand/dna-symbol-dark.svg` and `/brand/dna-symbol-light.svg` for compact brand marks on dark and light front-door surfaces
- `/brand/dna-lockup-dark.svg` and `/brand/dna-lockup-light.svg` for explicit full-logo lockups on homepage hero and login shells
- `/video/dna-loop.mp4` as the canonical branded loop for homepage and login

Managed portal-served brand assets:
- `/portal/assets/brand/dna-symbol-dark.svg`
- `/portal/assets/brand/dna-symbol-light.svg`

Brand contract notes:
- Front-door assets use explicit `symbol` and `lockup` variants instead of the older generic `dna-mark-*` naming.
- Portal-served assets are limited to mirrored `symbol` variants in this tranche and must remain allowlisted through `config/portal_asset_manifest.json`.

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
- Vercel deployment or preview-URL protection is distinct from production-domain coverage. Configure the appropriate Vercel Deployment Protection scope for the environment you are validating; protecting only preview URLs is not sufficient for a production-domain rollout.
- Treat this posture as a predeploy requirement before any internet-reachable staging or production environment, not as a post-launch hardening task.
- The v1 session store remains SQLite-backed and currently supports only `TP_FRONTDOOR_SESSION_SCALING_MODE=single_instance`.
- If your deployment target requires `multi_instance` or `ephemeral_runtime`, treat that as a blocked requirement until a dedicated external session store is introduced; the front door now fails `/healthz` under those modes on purpose.

## Validation

Front door checks:

```bash
make test-frontdoor-contract
```

Manual shared-deployment posture gate:

```bash
TP_FRONTDOOR_GATE_ENVIRONMENT="staging" \
TP_FRONTDOOR_GATE_FRONTDOOR_URL="https://portal.example.com" \
TP_FRONTDOOR_GATE_CF_ACCESS_TEAM_DOMAIN="https://your-team.cloudflareaccess.com" \
TP_FRONTDOOR_GATE_VERCEL_DEPLOYMENT_URL="https://portal-preview.vercel.app" \
TP_FRONTDOOR_GATE_CONFIRM_FASTAPI_NON_PUBLIC=1 \
make validate-frontdoor-deployment-gate
```

Notes:
- This gate is a manual predeploy control. It does not reconfigure Cloudflare or Vercel for you.
- The gate validates edge posture at rollout time. It does not replace app-side Cloudflare Access JWT verification.
- If FastAPI has a public URL, set `TP_FRONTDOOR_GATE_FASTAPI_PUBLIC_URL` instead of `TP_FRONTDOOR_GATE_CONFIRM_FASTAPI_NON_PUBLIC=1`.
- The gate fails closed for ambiguous frontdoor or Vercel responses; only clearly protected responses pass.

Browser smoke with isolated local backend + managed front door:

```bash
TP_FRONTDOOR_BASE_URL="http://localhost:3000" \
TP_FRONTDOOR_USERNAME="<username>" \
TP_FRONTDOOR_PASSWORD="<password>" \
make validate-frontdoor-browser
```

Equivalent direct commands remain available if you want to run the underlying
checks manually:

```bash
cd web/secure-landing
npm test
TP_NEXT_DIST_DIR=.next-build-verify npm run build
TP_NEXT_DIST_DIR=.next-build-verify npm run start

TP_FRONTDOOR_BASE_URL="http://localhost:3000" \
TP_FRONTDOOR_USERNAME="<username>" \
TP_FRONTDOOR_PASSWORD="<password>" \
python scripts/validation/validate_frontdoor_browser_smoke.py
```

For local managed smoke validation, use `http://localhost:3000` rather than
`http://127.0.0.1:3000`; the development front door normalizes same-origin CSRF
checks to `localhost`.

For release validation, prefer running the browser smoke against
`TP_NEXT_DIST_DIR=.next-build-verify npm run start` after building with the
same `TP_NEXT_DIST_DIR=.next-build-verify` value, not only against `next dev`.
The start wrapper launches the standalone build output and preserves the
production-like cookie posture. Local HTTP login validation still needs
HTTPS (or equivalent) if you want to exercise the full auth flow outside `next dev`.

FastAPI contract gate:

```bash
make test-orchestrator-contract
```

Manual GitHub predeploy workflow:
- Use `.github/workflows/frontdoor-deployment-gate.yml` via `workflow_dispatch` for shared staging or production rollouts.
- Bind `staging` runs to `frontdoor-staging` and `production` runs to `frontdoor-production`.
- Configure GitHub environment reviewers so deployment approval happens before the live gate executes.

Direct-debug portal browser smoke:

```bash
python scripts/validation/validate_portal_browser_smoke.py
```
