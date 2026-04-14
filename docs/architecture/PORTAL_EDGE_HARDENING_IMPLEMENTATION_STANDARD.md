# Portal Edge Hardening Implementation Standard

Date: 2026-04-13

## Summary

This document replaces the generic external `Flask + Vanilla JavaScript + Entra`
hardening plan with a repo-owned implementation standard for the actual browser
boundary in this codebase.

The repo-owned request path is:

`public ingress/WAF -> web/secure-landing (Next.js managed frontdoor) -> app.py (FastAPI origin)`

This standard preserves the current auth, route, selector, proxy, and validation
contracts. It does not approve an Entra access-token migration. Any future Entra
adoption must be handled as a separate tracked change.

## Boundary And Route Ownership

The managed browser boundary starts at `web/secure-landing/`.

- `GET /` is the public DNA homepage rendered by the managed frontdoor.
- `GET /login` is the managed operator sign-in shell and anonymous-session/CSRF
  entry point.
- `GET /portal` proxies the FastAPI operator shell through the frontdoor.
- `GET /portal/bootstrap` is the managed browser bootstrap/auth contract.
- `GET /portal/assets/*` proxies the checked-in allowlisted operator assets.
- `GET /portal/video/*` proxies the managed video asset path.
- `GET /healthz` is the managed frontdoor readiness contract.
- `/v1/*` stays same-origin at the frontdoor and is proxied server-to-server to
  FastAPI.
- FastAPI `GET /ready` remains the backend readiness contract.

Operational assumption:

- An upstream public ingress such as Azure Front Door may sit in front of the
  managed frontdoor.
- Any upstream edge policy must preserve the application contracts below.
- Existing Cloudflare Access verification and deployment protection remain part
  of the current managed auth posture where configured.

## Cache And Compression Standard

Caching and compression are route-specific. Do not apply blanket edge policies.

| Route | Owner | Cache policy | Compression policy |
| --- | --- | --- | --- |
| `/` | Next.js frontdoor | `public, max-age=300, must-revalidate` | Edge compression allowed when the outer ingress supports it |
| `/login` | Next.js frontdoor | `no-store` | No edge caching; compression is optional but not a design goal |
| `/portal` | Next.js proxy to FastAPI | `no-store` | No edge caching |
| `/portal/bootstrap` | Next.js frontdoor | `no-store` | No edge caching |
| `/portal/assets/*` | FastAPI asset manifest via Next.js proxy | Preserve origin `Cache-Control`; fingerprinted assets may be immutable, stale/unversioned fallbacks stay `no-store` | Do not force compression on range-sensitive or fallback paths |
| `/portal/video/*` | Next.js proxy | Preserve upstream cache policy | Do not blanket-compress video or range content |
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
- If Azure Front Door is used upstream, route/pattern configuration should map
  to the concrete paths above rather than generic `*.css` or `*.js` patterns.

## Current Security Baseline

### Trusted Host And Proxy Trust

FastAPI host validation is enforced in `app.py` through `TrustedHostMiddleware`.
The managed frontdoor independently ignores untrusted forwarded host/proto
overrides when constructing browser redirects and same-origin URLs.

Required posture:

- Keep `TP_ENABLE_TRUSTED_HOSTS=1` outside exceptional troubleshooting.
- Keep `TP_TRUSTED_HOSTS` aligned with the actual deployed hostnames.
- Do not trust arbitrary `X-Forwarded-Host` or `X-Forwarded-Proto` input.
- Do not introduce proxy-hop trust changes without updating contract tests and
  deployment docs in the same change.

### Security Headers And CSP

The repo already emits enforced CSP and security headers from both the FastAPI
origin and the managed frontdoor.

Required posture:

- Preserve the current enforced CSP posture for shipped routes.
- Use `Content-Security-Policy-Report-Only` only when trialing a policy delta.
- Keep HSTS and any outer-edge header normalization at the public ingress.
- Do not regress to inline script/style allowances without an explicit,
  reviewed exception.

### Auth Boundary

Current-state auth is unchanged:

- Public homepage: unauthenticated.
- Login shell: managed entry path protected by Cloudflare Access validation
  where configured, plus operator username/password handoff.
- Browser session: SQLite-backed frontdoor session with CSRF-bound unsafe
  requests.
- API proxy: frontdoor injects the backend secret server-to-server; browser
  code does not receive the backend API key in managed mode.
- Local bypass: allowed only in explicit development flows.

This baseline does not introduce Entra access-token validation into the current
browser path.

### Runtime And Request Limits

- `web/secure-landing/` is a Node 22-only application for install, dev, test,
  build, and start.
- FastAPI request limits and path/root allowlists remain enforced in `app.py`.
- Do not weaken the current session-scaling guardrails: the shipped posture is
  still single-instance SQLite unless an external session store is introduced.

### Direct-Origin Denial

The FastAPI origin is not a normal public browser entry.

Required posture:

- Keep normal browser traffic on the managed frontdoor.
- Treat direct FastAPI exposure as an exception that requires compensating
  controls and explicit documentation.
- Preserve the deployment gate that checks protected frontdoor posture,
  protected deployment URL posture, and the non-public FastAPI assumption.

## SSRF And Outbound Fetch Policy

This repo does not currently treat arbitrary user-supplied URL fetching as a
first-class frontdoor feature. If an outbound fetch feature is added later, the
minimum standard is:

- explicit hostname allowlist
- HTTPS only
- DNS and IP classification before connect
- redirects off by default
- short connect/read timeouts
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

Use the shared-deployment posture gate when validating a real internet-reachable
environment:

```bash
make validate-frontdoor-deployment-gate
```

Minimum acceptance checks:

- untrusted FastAPI host header returns `400`
- frontdoor redirects ignore untrusted host overrides
- `/`, `/login`, `/portal`, `/portal/bootstrap`, `/healthz`, and `/v1/*`
  preserve the documented cache posture
- managed login/session/CSRF behavior remains intact
- Cloudflare Access-managed enforcement remains fail-closed where configured
- frontdoor test/build runs under Node 22.x

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
