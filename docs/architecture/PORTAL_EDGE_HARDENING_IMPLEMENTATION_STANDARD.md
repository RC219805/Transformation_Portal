# Portal Edge Hardening Implementation Standard

Date: 2026-11-25 (v3)

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

- **Public ingress / upstream edge**: external edge (for example Azure Front Door, a WAF, or the
  Cloudflare Worker edge proxy in `cloudflare/transformationportal-worker/`)
- **Public origin**: the one origin browsers use to reach the managed frontdoor
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

The managed frontdoor does not implement a trusted-hop model. It resolves the
origin it answers as in one of two ways, and never from a forwarded header.

Configured public origin. When `TP_FRONTDOOR_PUBLIC_ORIGIN` holds a value that
is honoured, that origin is the resolved request origin for every request:

- The value is one exact origin, `https://<host>[:<port>]`, written as a browser
  sends it in `Origin`: lower case, a DNS name of at least two labels, with no
  credentials, path, trailing slash, query, fragment, wildcard, IP literal,
  loopback name, or default port. Nothing is normalised; a value that would
  need it is rejected.
- `Host`, `Forwarded`, `X-Forwarded-Host`, `X-Forwarded-Proto`, and the request
  URL derived by the runtime do not affect the resolved origin. Only the path,
  query, and fragment come from the request.
- A browser that reaches the frontdoor under any other origin fails the
  same-origin check and is redirected to the configured origin.
- A malformed value, or any value while the local development bypass is
  active, is not honoured. The frontdoor then behaves as if the variable were
  unset, and `/healthz`, the startup preflight, and `make check-vercel-env`
  report the failure.

No configured public origin. The frontdoor keeps the equivalence rule:

- Forwarded host and proto values are accepted only when they match the
  already-derived request URL surface.
- Any mismatch between forwarded headers and the resolved request URL must be
  ignored.

In both cases, redirect construction and same-origin checks must use the
resolved request origin, not forwarded overrides.

Required posture:

- Keep `TP_ENABLE_TRUSTED_HOSTS=1` outside exceptional troubleshooting.
- Keep `TP_TRUSTED_HOSTS` aligned with the actual deployed hostnames. It governs
  the `Host` header FastAPI receives, which is the `TP_FASTAPI_ORIGIN` host, not
  the public origin.
- Set `TP_FRONTDOOR_PUBLIC_ORIGIN` wherever browsers reach the frontdoor on an
  origin other than the one its runtime derives for itself. That includes the
  Cloudflare Worker edge proxy and any tunnel or reverse proxy in front of a
  self-hosted frontdoor. Set it only for the environment that serves that
  origin.
- Do not introduce hop-count or proxy-chain trust logic.
- Do not treat forwarded headers as authoritative input, and do not use one to
  choose between origins.
- Preserve the equivalence-based validation behavior for deployments that do
  not configure a public origin.

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
- Session activity updates only the current timing fields atomically. A
  delayed request cannot recreate a session removed by logout, rotation, or
  expiration. Redis touches preserve the existing key expiration and current
  identity, role, and CSRF fields. The Redis provider and its ACL must support
  server-side `EVAL` and `SET ... KEEPTTL` for these atomic updates.

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
- Without `TP_FRONTDOOR_PUBLIC_ORIGIN`, frontdoor redirect and same-origin
  behavior preserve the equivalence-based proxy trust contract.
- With an honoured `TP_FRONTDOOR_PUBLIC_ORIGIN`, redirects target that origin
  and unsafe requests pass only when `Origin` (or the `Referer` origin when
  `Origin` is absent) equals it, whatever `Host` and forwarded headers say.
- A malformed `TP_FRONTDOOR_PUBLIC_ORIGIN`, or one set while the local
  development bypass is active, fails `/healthz` and is not honoured.

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

### Decision Record: Configured Public Origin (2026-11-25)

Decided by the repository owner on 2026-11-25. This record covers the change to
"Trusted Host And Proxy Trust" in v3.

- **Surface**: redirect construction for `/login`, `/logout`, and `/portal`;
  same-origin checks on `POST /login`, `POST /logout`, unsafe `/v1/*` methods,
  and `POST /v1/portal/rum`.
- **Reason**: a self-hosted Next.js runtime derives the request URL from its
  own listener, and a Vercel build derives it from the deployment `Host`.
  Behind the Cloudflare Worker edge proxy, a tunnel, or a reverse proxy,
  neither is the origin the browser is on. Redirects left the public origin and
  unsafe requests failed the same-origin check. This is finding F-01 in the
  [2026-11-18 forensic audit](../analysis/REPOSITORY_FORENSIC_AUDIT_2026-11-18.md),
  since reproduced against the production standalone build.
- **Change**: `TP_FRONTDOOR_PUBLIC_ORIGIN` names one origin the frontdoor
  answers as. It replaces how the resolved request origin is derived. It does
  not make any forwarded header an input.
- **Compensating controls**: the value is operator configuration on a closed
  grammar; no request header selects or alters it; it is not honoured while the
  local development bypass is active; `/healthz`, the startup preflight, and
  `make check-vercel-env` fail on a value that is not honoured; the CSRF token,
  session cookie, and Cloudflare Access JWT verification are unchanged.
- **Residual risk**: a well-formed but wrong origin sends browsers to that host
  and blocks sign-in, so review the accepted origin and repeat a sign-in through
  the public hostname after each change. With the variable set, only that one
  origin can sign in. Hosted behavior is established only by a real deployment,
  not by repository tests.
- **Alternatives rejected**: honouring `X-Forwarded-Host` when it equals a
  configured host, which uses a forwarded header to choose the origin and fails
  when an intermediary rewrites it; deriving same-origin from Fetch Metadata
  with relative redirects, which changes the CSRF signal and every redirect
  site.
- **Owner**: Repository Architect.
