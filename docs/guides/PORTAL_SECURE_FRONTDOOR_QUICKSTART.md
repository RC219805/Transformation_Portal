# Portal Secure Front Door Quickstart

See [Portal Edge Hardening Implementation Standard](../architecture/PORTAL_EDGE_HARDENING_IMPLEMENTATION_STANDARD.md)
for the repo-owned cache, header, auth-boundary, and validation baseline that
governs this quickstart.

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
The repository root `.env.example` is the Docker/FastAPI template; it is not a
replacement for the front-door template because the Node app has separate proxy,
Cloudflare Access, user-source, and session-store settings.

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

## Optional RUM Pilot Knobs

M1 measurement is additive, default-off, and protected by both a hard enable flag
and a deterministic rollout percentage.

```bash
export TP_PORTAL_RUM_ENABLED=0
export TP_PORTAL_RUM_ROLLOUT_PERCENT=0
export TP_FRONTDOOR_RUM_ENABLED=0
export TP_FRONTDOOR_RUM_ROLLOUT_PERCENT=0
export TP_PORTAL_RUM_LOG_PATH="/absolute/path/to/portal-rum.jsonl"
export TP_PORTAL_EVENT_LOG_PATH="/absolute/path/to/portal-events.jsonl"
```

Notes:
- `TP_PORTAL_RUM_ENABLED=0` keeps `/v1/portal/rum` in success/no-op mode and keeps `features.rumTelemetry=false` on both bootstrap surfaces.
- `TP_PORTAL_RUM_ROLLOUT_PERCENT=0` keeps managed portal/bootstrap RUM disabled even when the hard flag is on.
- `TP_FRONTDOOR_RUM_ENABLED=0` independently disables landing, login, logout, and front-door payloads submitted through the RUM proxy after the shared hard flag is on.
- `TP_FRONTDOOR_RUM_ROLLOUT_PERCENT=0` samples out front-door RUM even when both hard flags are on.
- `TP_PORTAL_RUM_LOG_PATH` is optional. When set, FastAPI appends PII-free JSONL records for the pilot summary CLI.
- `TP_PORTAL_EVENT_LOG_PATH` is optional. When set, FastAPI appends PII-free portal event JSONL for viewer, review, and stream telemetry evidence.
- Direct-debug rollout stability reuses `TP_PORTAL_DIRECT_DEBUG_COHORT_KEY`.
- Managed rollout stability uses the authenticated actor identity already present on the front door; raw usernames and emails are not stored in the RUM sink.
- Bounded pilot rollout is governed by the `Approved with Conditions` sign-off packet at `docs/compliance/PORTAL_TELEMETRY_PRIVACY_SIGNOFF.md`; revise that packet before adding any new telemetry event family, metadata key, marker cookie, storage key, rollout knob, sink behavior, or retention posture.

## Optional Review Surface Pilot Knobs

The review-surface split and in-portal viewer stay additive and default-off.
Both rollouts use the same deterministic cohort hashing path as the RUM pilot.

```bash
export TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT=0
export TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT=0
```

Notes:
- `TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT=0` keeps `features.artifactViewerModal=false` on both bootstrap surfaces and leaves review actions on the raw-link path.
- `TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT=0` keeps `features.reviewSurfaceDeferred=false`, so the split review asset is prefetched after bootstrap instead of deferred to review/operate entry.
- Roll back either pilot by setting the corresponding percentage back to `0` and redeploying the managed front door and/or backend origin that serves `/portal/bootstrap`.
- Cohort expansion should move in bounded steps and record the operator owner, rollout date, rollback owner, and target percentage in the deployment notes for the change.

## Optional FastVLM Captioning Pilot Knobs

FastVLM advisory captioning controls stay additive and default-off. The portal
feature appears only when the backend enable flag is on and the rollout cohort
matches.

```bash
export TP_PORTAL_FASTVLM_CAPTIONING_ENABLED=0
export TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT=0
```

Notes:
- `TP_PORTAL_FASTVLM_CAPTIONING_ENABLED=0` keeps `features.fastVlmCaptioning=false` on `/portal/bootstrap` and keeps captioning controls hidden.
- `TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT=0` keeps the UI hidden even when the hard flag is on.
- Enabled cohorts can request FastVLM advisory sidecars through config preview and job creation; disabled cohorts receive the `captioning_feature_disabled` field error.
- FastVLM remains subprocess-only metadata. Missing runtime paths are preview warnings, not quality-gate failures, and captions never satisfy Materials V3 or run-card validation gates.

## Optional Staged Upload Pilot Knobs

Staged uploads stay additive and default-off. The browser feature appears only
when the backend enable flag is on and the rollout cohort matches.

```bash
export TP_PORTAL_UPLOAD_STAGING_ENABLED=0
export TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT=0
export TP_PORTAL_UPLOAD_ROOT="/tmp/transformation-portal/uploads"
export TP_PORTAL_MAX_UPLOAD_REQUEST_BYTES=1048576
export TP_PORTAL_UPLOAD_MAX_FILES=256
export TP_PORTAL_UPLOAD_MAX_FIELDS=32
export TP_PORTAL_UPLOAD_MAX_PART_BYTES=1048576
export TP_PORTAL_UPLOAD_TTL_SECONDS=86400
export TP_PORTAL_UPLOAD_CAPTURE_METADATA_ENABLED=0
```

Notes:
- `TP_PORTAL_UPLOAD_STAGING_ENABLED=0` keeps `features.stagedUploads=false` on both bootstrap surfaces and returns typed `404 not found` from `POST /v1/uploads/staging`.
- `TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT=0` keeps the UI hidden even when the backend route is enabled.
- `TP_PORTAL_UPLOAD_ROOT` must stay within `TP_ALLOWED_INPUT_ROOTS`.
- `TP_PORTAL_MAX_UPLOAD_REQUEST_BYTES` is route-specific and does not change the existing `/v1/jobs` request-size ceiling.
- `TP_PORTAL_UPLOAD_CAPTURE_METADATA_ENABLED=0` keeps the capture metadata artifact on the empty-array path until the extraction pilot is explicitly enabled.

## Local Development

Start the FastAPI origin first:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Seed the canonical reusable local login fixture if you want a stable localhost sign-in:

```bash
make seed-frontdoor-user
```

Start the front door in a second shell:

```bash
make seed-frontdoor-user
make run-frontdoor-local
```

Open `http://localhost:3000/`.

The canonical launcher:
- exports the known-good local managed env (`NODE_ENV=development`, `TP_ALLOW_LOCAL_ACCESS_BYPASS=1`)
- pins the supported SQLite session posture with `TP_FRONTDOOR_SESSION_SCALING_MODE=single_instance`
- reuses `TP_API_KEY` as `TP_BACKEND_API_KEY` when needed
- auto-seeds `/tmp/tp-frontdoor-users.json` with `smoke-admin` / `correct horse battery staple` when no explicit `TP_FRONTDOOR_USERS_FILE` or `TP_FRONTDOOR_USERS_JSON` is configured
- verifies FastAPI readiness first
- refuses to start if `localhost:3000` is already occupied instead of letting Next.js drift to `:3001`

The canonical local credential bootstrap:
- `make seed-frontdoor-user` writes a single-user JSON fixture to `TP_FRONTDOOR_USERS_FILE` when you override it, or to `/tmp/tp-frontdoor-users.json` by default
- `TP_FRONTDOOR_USERS_FILE`, `TP_FRONTDOOR_USERNAME`, and `TP_FRONTDOOR_PASSWORD` override the built-in local bootstrap defaults; they are not required for the canonical local path
- it defaults `access_email` to `<username>@local.invalid` and `role` to `admin`
- it overwrites stale local fixture content instead of relying on fragile inline `node -e` snippets

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

Managed bootstrap and managed `/v1/*` responses now echo `traceparent`. The
front door forwards a browser-supplied `traceparent` upstream unchanged, and
FastAPI mints one when the browser does not supply a valid value.

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

Validation prerequisites for every local managed frontdoor run:

- Switch the frontdoor shell to Node `22.x` first. `web/secure-landing` ships
  `.nvmrc` with `22`, and install/test/build/start all fail fast outside that
  runtime.
- If the shell previously used another Node major, rebuild the native modules
  under Node `22.x` before running frontdoor checks:

```bash
cd web/secure-landing
nvm use 22
npm rebuild better-sqlite3 argon2
```

- Run local browser smoke against `http://localhost`, not `http://127.0.0.1`,
  because the browser-smoke harness defaults to `http://localhost:3000` and
  same-origin CSRF validation requires the exact origin to match, so
  `127.0.0.1` and `localhost` are different origins.
- Treat Node `25.x` failures as unsupported-runtime/tooling failures unless the
  same command also fails under Node `22.x`.

Front door checks:

```bash
cd web/secure-landing
nvm use 22
cd ../..
make test-frontdoor-contract
```

Portal contract checks:

```bash
make test-portal-contract
```

When `TP_PORTAL_RUM_LOG_PATH` is configured and the pilot has collected samples,
summarize the JSONL sink with:

```bash
python tools/portal_rum_summary.py --input /absolute/path/to/portal-rum.jsonl
```

The summary groups by auth mode, route, view, and cohort bucket, then prints p75
LCP/INP/CLS, bootstrap timings, queue request timings, and SSE reconnect counts.

For the RFC evidence package, summarize RUM plus optional viewer-event evidence with:

```bash
python tools/portal_modernization_evidence.py \
  --rum-log /absolute/path/to/portal-rum.jsonl \
  --event-log /absolute/path/to/portal-events.jsonl \
  --operator-hours 8
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
cd web/secure-landing
nvm use 22
cd ../..
make validate-frontdoor-browser
```

That target launches isolated local backend and managed front-door runtimes and
auto-seeds the canonical smoke credentials for the managed front-door runtime it
creates. No manual username/password exports are required for the standard local
path.

Equivalent direct commands remain available if you want to run the underlying
checks manually:

```bash
cd web/secure-landing
nvm use 22
npm test
TP_NEXT_DIST_DIR=.next-build-verify npm run build
```

If you want to launch the standalone build locally without Cloudflare Access,
do that in a separate shell after the backend is ready and the local credential
fixture exists:

```bash
make seed-frontdoor-user
cd web/secure-landing
NODE_ENV=development \
TP_ALLOW_LOCAL_ACCESS_BYPASS=1 \
TP_FASTAPI_ORIGIN=http://127.0.0.1:8000 \
TP_BACKEND_API_KEY="${TP_BACKEND_API_KEY:-$TP_API_KEY}" \
TP_FRONTDOOR_USERS_FILE=/tmp/tp-frontdoor-users.json \
TP_FRONTDOOR_SESSION_DB=/tmp/transformation-portal-frontdoor-sessions-standalone.db \
TP_FRONTDOOR_SESSION_SCALING_MODE=single_instance \
TP_NEXT_DIST_DIR=.next-build-verify \
npm run start
```

For local standalone runs, keep `NODE_ENV=development` and
`TP_ALLOW_LOCAL_ACCESS_BYPASS=1`; otherwise the front door treats Cloudflare
Access as required and `/healthz` fails closed until `TP_CF_ACCESS_TEAM_DOMAIN`
and `TP_CF_ACCESS_AUD` are configured.

If you want standalone validation with the production auth posture instead of
the local bypass, keep `NODE_ENV=production`, supply
`TP_CF_ACCESS_TEAM_DOMAIN` + `TP_CF_ACCESS_AUD`, and run it behind HTTPS (or an
equivalent trusted edge) so the secure-cookie flow remains valid.

If you want browser smoke against an already running managed front door, run it
from the repo root and point it at that instance instead of
`--spawn-local-frontdoor`:

```bash
TP_FRONTDOOR_BASE_URL="https://portal.example.com" \
TP_FRONTDOOR_USERNAME="replace-with-operator-username" \
TP_FRONTDOOR_PASSWORD="replace-with-operator-password" \
python scripts/validation/validate_frontdoor_browser_smoke.py
```

If you point the browser smoke at an already running or non-local managed
front-door instead of `--spawn-local-frontdoor`, continue to pass
`TP_FRONTDOOR_USERNAME` and `TP_FRONTDOOR_PASSWORD` explicitly.

For local managed smoke validation, use `http://localhost:3000` rather than
`http://127.0.0.1:3000`; the browser smoke defaults to that origin and the
front door's same-origin CSRF validation requires an exact origin match, so
`127.0.0.1:3000` and `localhost:3000` are different origins.

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
