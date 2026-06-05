# Portal Edge Hardening Implementation Standard

Date: 2026-04-13 (v2)

## Summary

This document replaces the generic external `Flask + Vanilla JavaScript + Entra`
hardening plan with a repo-owned implementation standard for the actual browser
boundary in this codebase.

The repo-owned request path is:

`public ingress/WAF -> web/secure-landing (Next.js managed frontdoor) -> app.py (FastAPI origin)`

This standard preserves the current auth, route, selector, proxy, cache,
session, and validation contracts. It does not approve an Entra access-token
migration. Any future Entra adoption must be handled as a separate tracked
change.

## Terminology

- **Public ingress / upstream edge**: external edge (for example Azure Front Door or WAF)
- **Managed frontdoor**: `web/secure-landing`
- **Origin**: FastAPI (`app.py`)

## Boundary And Route Ownership

The managed browser boundary starts at `web/secure-landing/`.

- `GET /` is the public DNA homepage rendered by the managed frontdoor.
- `GET /login` is the managed sign-in shell and anonymous-session/CSRF entry
  point.
- `POST /login` is the operator credential handoff path.
- `GET /portal` proxies the FastAPI operator shell through the frontdoor.
- `GET /portal/bootstrap` is the managed browser bootstrap/auth contract.
- `GET /portal/assets/*` proxies the checked-in allowlisted operator assets.
- `GET /portal/video/*` proxies the managed video asset path.
- `GET /healthz` is the managed frontdoor readiness contract.
- `/v1/*` stays same-origin at the frontdoor and is proxied server-to-server to
  FastAPI.
- FastAPI `GET /ready` remains the backend readiness contract.

Operational assumptions:

- An upstream public ingress may sit in front of the managed frontdoor.
- Any upstream edge policy must preserve the application contracts below.
- FastAPI is not a normal public browser entry.
- The managed frontdoor is the authoritative browser boundary.

## Cache And Compression Standard

Caching and compression are route-specific. Do not apply blanket edge policies.

| Route | Owner | Cache policy | Compression policy |
| --- | --- | --- | --- |
| `/` | Next.js frontdoor | `public, max-age=300, must-revalidate` | Edge compression allowed when the outer ingress supports it |
| `/login` | Next.js frontdoor | `no-store` | No edge caching; compression is optional but not a design goal |
| `/portal` | Next.js proxy to FastAPI | `no-store` | No edge caching |
| `/portal/bootstrap` | Next.js frontdoor | `no-store` | No edge caching |
| `/portal/assets/*` | FastAPI asset manifest via Next.js proxy | Preserve origin and manifest contract; fingerprinted assets may be immutable, stale or unversioned fallbacks stay `no-store` | Do not force compression on range-sensitive or fallback paths |
| `/portal/video/*` | Next.js proxy | `public, max-age=86400` | Do not blanket-compress video or range content |
| `/healthz` | Next.js frontdoor | `no-store` | No edge caching |
| `/v1/*` | FastAPI API via Next.js proxy | `no-store` or stricter | No edge caching |
| FastAPI direct browser paths | FastAPI origin | Not a public browser cache surface | Not a public browser cache surface |

Implementation rules:

- Do not edge-cache authenticated, session-bearing, or operator-specific
  responses.
- Treat `/login`, `/portal`, `/portal/bootstrap`, `/healthz`, and `/v1/*` as
  non-cacheable surfaces.
- Preserve the existing portal asset manifest contract instead of inventing new
  wildcard extension routes.
- If Azure Front Door is used upstream, route or pattern configuration should
  map to the concrete paths above rather than generic `*.css` or `*.js`
  patterns.

## Current Security Baseline

### Trusted Host And Proxy Trust

FastAPI host validation is enforced in `app.py` through `TrustedHostMiddleware`.

The managed frontdoor does not implement a trusted-hop model. It enforces an
equivalence rule:

- Forwarded host and proto values are accepted only when they match the
  already-derived request URL surface.
- Any mismatch between forwarded headers and the resolved request URL must be
  ignored.
- Redirect construction and same-origin checks must use the resolved request
  URL, not forwarded overrides.

Required posture:

- Keep `TP_ENABLE_TRUSTED_HOSTS=1` outside exceptional troubleshooting.
- Keep `TP_TRUSTED_HOSTS` aligned with the actual deployed hostnames.
- Do not introduce hop-count or proxy-chain trust logic.
- Do not treat forwarded headers as authoritative input.
- Preserve the existing equivalence-based validation behavior in the managed
  frontdoor.

### Security Headers And CSP

The repo already emits enforced CSP and security headers from both the FastAPI
origin and the managed frontdoor.

Required posture:

- Preserve the current enforced CSP posture for shipped routes.
- Use `Content-Security-Policy-Report-Only` only when trialing a policy delta.
- Keep HSTS and any outer-edge header normalization at the public ingress.
- Do not regress to inline script or style allowances without an explicit,
  reviewed exception.

### Auth Boundary

Current-state auth is unchanged:

- Public homepage: unauthenticated.
- Login shell (`GET /login`): publicly renderable managed entry surface that
  mints an anonymous session and CSRF binding.
- Credential handoff (`POST /login`): requires verified Access context unless
  the explicit local development bypass is active.
- Authenticated managed surfaces: `/portal`, `/portal/bootstrap`, and `/v1/*`
  require a current verified Access context and fail closed on auth failure.
- Browser session: server-side frontdoor session with CSRF-bound unsafe
  requests. SQLite is the single-instance local/default store; Redis is the
  supported hosted external store.
- API proxy: the frontdoor injects the backend secret server-to-server; browser
  code does not receive the backend API key in managed mode.
- Local bypass: allowed only in explicit development flows described below.

This baseline does not introduce Entra access-token validation into the current
browser path.

### Cloudflare Access Enforcement

Cloudflare Access is required for managed credential handoff and authenticated
managed surfaces whenever the explicit local development bypass is not active.

Required posture:

- `GET /login` remains renderable without a verified Access JWT and may mint an
  anonymous session.
- `POST /login` must require verified Access before operator credential handoff
  continues, unless the explicit local development bypass is active.
- `/portal`, `/portal/bootstrap`, and `/v1/*` must require current verified
  Access and fail closed on missing or invalid Access state.
- Team domain and audience validation must be configured whenever bypass is not
  active.

Local development exception:

- Local bypass is enabled only when both of the following are true:
  - `NODE_ENV=development`
  - `TP_ALLOW_LOCAL_ACCESS_BYPASS=1`
- The bypass is for explicit non-public local development only.
- Production and other managed environments must not treat
  `TP_ALLOW_LOCAL_ACCESS_BYPASS=1` by itself as sufficient to bypass Access.

### Session And Cookie Contract

The browser session is server-side. The cookie, store, scaling, and timeout
contract is environment-aware.

Cookie naming and transport:

- Production cookie name: `__Host-tp_session`
- Non-production local managed-development cookie name: `tp_session`
- Production cookie transport: `Secure`
- Local managed-development cookie transport: not `Secure`, so local HTTP
  development remains functional
- Cookie attributes:
  - `HttpOnly`
  - `SameSite=Lax`
  - `Path=/`
- The cookie must not set a `Domain` attribute.

Session behavior:

- `GET /login` may mint an anonymous session before credentials are posted so
  CSRF is bound to a server-side session from the start.
- Successful login must rotate the session identifier before redirecting to
  `/portal`.
- Idle timeout is enforced.
- Absolute timeout is enforced.
- The cookie lifetime is bounded by the absolute timeout.

Current shipped timeout defaults:

- Idle timeout: 8 hours
- Absolute timeout: 24 hours

CSRF:

- All state-changing requests must require a valid CSRF token.
- Requests without a valid token must fail.

Do not introduce alternate session semantics without updating contract tests and
deployment documentation in the same change.

### Runtime And Request Limits

- `web/secure-landing/` is a Node 22-only application for install, dev, test,
  build, and start.
- FastAPI request limits and path or root allowlists remain enforced in
  `app.py`.
- Do not weaken the current session-scaling guardrails: SQLite is limited to
  single-instance posture, while hosted `multi_instance` and
  `ephemeral_runtime` posture require the Redis session store.

### Direct-Origin Denial

The FastAPI origin is not a normal public browser entry.

Required posture:

- Keep normal browser traffic on the managed frontdoor.
- Treat direct FastAPI exposure as an exception that requires compensating
  controls and explicit documentation.
- Preserve the deployment gate that checks protected frontdoor posture,
  protected Worker or Vercel deployment URL posture, and the non-public FastAPI
  assumption.

## SSRF And Outbound Fetch Policy

This repo does not currently treat arbitrary user-supplied URL fetching as a
first-class frontdoor feature. If an outbound fetch feature is added later, the
minimum standard is:

- explicit hostname allowlist
- HTTPS only
- DNS and IP classification before connect
- redirects off by default
- short connect and read timeouts
- network egress restrictions for private, loopback, link-local, and metadata
  targets
- a pinned-client pattern when the transport must connect only to previously
  validated addresses

Do not add a convenience `requests.get(user_url)`-style helper and call the
surface hardened.

## Validation And Release Gates

The canonical validation path for this boundary is:

```bash
make test-orchestrator-contract
make test-frontdoor-contract
make validate-frontdoor-browser
```

Use the shared-deployment posture gate when validating a real
internet-reachable environment:

```bash
make validate-frontdoor-deployment-gate
```

Minimum acceptance checks:

### Host And Proxy

- Untrusted FastAPI host header returns `400`.
- Frontdoor redirects ignore untrusted host overrides.
- Frontdoor redirect and same-origin behavior preserve the equivalence-based
  proxy trust contract.

### Cache Headers

- `/` returns `Cache-Control: public, max-age=300, must-revalidate`
- `GET /login` returns `Cache-Control: no-store`
- `/portal` returns `Cache-Control: no-store`
- `/portal/bootstrap` returns `Cache-Control: no-store`
- `/healthz` returns `Cache-Control: no-store`
- `/v1/*` returns `Cache-Control: no-store` or stricter
- SSE or streaming endpoints, when present, return
  `Cache-Control: no-store, no-transform`
- `/portal/video/*` returns `Cache-Control: public, max-age=86400`
- `/portal/assets/*` preserves the documented manifest and upstream cache
  contract

### Auth And Session

- `GET /login` renders the managed sign-in shell and may mint an anonymous
  session without verified Access.
- `POST /login` rejects credential handoff without verified Access when bypass
  is not active.
- `/portal`, `/portal/bootstrap`, and `/v1/*` fail closed when current Access
  verification is missing or invalid.
- Production ignores `TP_ALLOW_LOCAL_ACCESS_BYPASS=1` without the explicit
  development gate.
- Local bypass works only when `NODE_ENV=development` and
  `TP_ALLOW_LOCAL_ACCESS_BYPASS=1`.
- Successful login rotates the session.
- Production authenticated cookies use `__Host-tp_session`.
- Local managed-development cookies use `tp_session`.
- Session cookies are `HttpOnly`, production cookies are `Secure`, and cookies
  are `SameSite=Lax`.
- Session idle and absolute expiry remain enforced.
- Managed login, session, and CSRF behavior remains intact.

### Runtime And Surface Ownership

- FastAPI is not directly accessible as a normal browser surface.
- Frontdoor test and build runs use Node 22.x.
- Browser auth and API proxy behavior remain same-origin at the managed
  frontdoor.

## Observability

- Existing structured audit logging must be preserved.
- Do not bypass `audit.js` or `managed-failure.js`.
- Do not weaken existing managed-surface failure classification without updating
  the standard and tests in the same change.

## Future Entra Migration Appendix

Entra is not part of the current baseline. If adopted later, it must be tracked
as an explicit auth migration that:

- defines whether Entra replaces or layers with Cloudflare Access and operator
  credential handoff
- updates `/login`, `/portal/bootstrap`, and proxy auth semantics deliberately
- updates browser smokes, contract tests, deployment gates, and quickstarts in
  the same change
- records rollback and coexistence behavior before rollout

Until that work is approved, treat Entra token validation guidance as future
architecture, not as current implementation instruction.

## Governance

This document is the authoritative implementation standard for the managed
browser boundary in this repo.

Any deviation must record:

- route or surface
- reason
- compensating control
- owner
