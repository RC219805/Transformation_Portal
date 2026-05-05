# Vercel Frontdoor Environment Checklist

The managed frontdoor (`web/secure-landing/`) deployed on Vercel reads its
configuration entirely from environment variables. When any of the variables
below are missing or stale, `/healthz` returns `503` with a structured `reason`
field that names the offender. This checklist enumerates every variable, how to
verify it, and the diagnostic the frontdoor emits when it is wrong.

The single most common production incident is a drift between
`TP_BACKEND_API_KEY` (frontdoor) and `TP_API_KEY` (backend); the resulting
`401` is surfaced as `503 config_failure` to every portal client. Treat the
two values as one secret, set them together, rotate them together.

## Required variables

| Variable | Required in | Purpose | `/healthz` reason if missing/wrong |
|----------|-------------|---------|-------------------------------------|
| `TP_BACKEND_ORIGIN` | all envs | URL of the FastAPI origin (named Cloudflare tunnel hostname in production). | `backend_unreachable` |
| `TP_FASTAPI_ORIGIN` | all envs | Alias used by `/healthz` and the v1 proxy. Must equal `TP_BACKEND_ORIGIN` unless an explicit split is intended. | `backend_unreachable` |
| `TP_BACKEND_API_KEY` | all envs | API key the frontdoor presents to the backend. **Must equal backend `TP_API_KEY`.** | `missing_backend_api_key` (when empty) / `backend_auth_mismatch` (when wrong) |
| `TP_FRONTDOOR_USERS_JSON` *or* `TP_FRONTDOOR_USERS_FILE` | all envs | At least one configured user. JSON form is `[{"username":"…","password_hash":"…","access_email":"…","role":"admin"}]`. | `no_configured_users` |
| `TP_FRONTDOOR_SESSION_SCALING_MODE` | all envs | `single_instance` for SQLite, or a multi-instance value backed by an external store. Vercel's stateless runtime needs an external store for multi-instance mode. | `multi_instance_requires_external_session_store` / `invalid_session_scaling_mode` |
| `TP_FRONTDOOR_SESSION_DB` | when `single_instance` | Path to the SQLite session DB. On Vercel, prefer an external store. | `session_store_unavailable` |
| `TP_CF_ACCESS_TEAM_DOMAIN` | production | Cloudflare Access team domain (`https://<team>.cloudflareaccess.com`). Validated by the preflight; required when `TP_ALLOW_LOCAL_ACCESS_BYPASS` is unset. | `missing_access_team_domain` |
| `TP_CF_ACCESS_AUD` | production | Cloudflare Access JWT audience for the protected application. | `missing_access_audience` |

## Optional variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TP_FRONTDOOR_PREFLIGHT_DISABLE` | `0` | Hard escape hatch for the startup preflight. Use only for emergencies; never set in production. |
| `TP_ALLOW_LOCAL_ACCESS_BYPASS` | unset | Development-only bypass for Cloudflare Access verification. **Never set in production.** |
| `TP_NEXT_DIST_DIR` | unset | Override Next.js build output directory; rarely needed on Vercel. |

## Verification

After updating Vercel project environment variables and redeploying, confirm
the frontdoor reports green readiness:

```
curl -s https://<frontdoor-host>/healthz | jq '{ok, frontend, backend, checks}'
```

Expected shape when healthy:

```
{
  "ok": true,
  "frontend": "ready",
  "backend": { "ok": true, "status": 200 },
  "checks": {
    "backend": {
      "ok": true,
      "configured": true,
      "status": 200,
      "auth_status": 200,
      "reason": null
    },
    "access_config": { "ok": true, "mode": "cloudflare_access", "teamDomainConfigured": true, "audienceConfigured": true },
    "user_source":   { "ok": true, "userCount": <n> },
    "session_store": { "ok": true, "configured": true },
    "session_scaling": { "ok": true }
  }
}
```

If `checks.backend.reason` is non-null, the table above identifies the missing
configuration. `auth_status: 401` (with `status: 200`) means the upstream is
healthy but the configured `TP_BACKEND_API_KEY` does not match the backend's
`TP_API_KEY` — rotate them together.

## Related runbooks

- Local development: `scripts/dev/write_local_env.sh` writes a canonical
  `/tmp/tp-local-http-all-on.env` with the two API keys bound together. Source
  it in any shell that runs the backend or the frontdoor.
- Cloudflare tunnel: prefer a **named** tunnel (`cloudflared tunnel create …`)
  over a `trycloudflare.com` quick tunnel for production frontdoor origins.
  Quick tunnels rotate hostnames on every restart and are unstable over QUIC.
