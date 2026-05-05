# Cloudflare Tunnel Runbook

Cloudflare tunnels publish the local FastAPI backend through Cloudflare's edge
without exposing inbound ports. Two flavours exist:

1. **Named tunnel** — the operator creates a persistent tunnel and routes a
   stable DNS hostname to it. Production frontdoor origins **must** use this.
2. **Quick tunnel** — `cloudflared tunnel --url …` provisions an ephemeral
   `*.trycloudflare.com` hostname that rotates every restart. Suitable for
   ad-hoc demos only.

The diagnosis logs show that quick tunnels degrade under sustained load (QUIC
session timeouts, DNS resolver failures). The wrapper script
`scripts/dev/run_cloudflared.sh` forces HTTP/2 instead of QUIC and prefers the
named-tunnel mode whenever the operator provides the env vars.

## Named tunnel setup (one time, per environment)

```bash
# 1. Authenticate cloudflared with the Cloudflare account.
cloudflared tunnel login

# 2. Create the tunnel. The credentials JSON is stored under ~/.cloudflared/.
cloudflared tunnel create transformation-portal-backend

# 3. Route a stable DNS hostname to the tunnel.
cloudflared tunnel route dns transformation-portal-backend backend.example.com
```

## Running the named tunnel

```bash
export CLOUDFLARED_TUNNEL_NAME=transformation-portal-backend
export CLOUDFLARED_TUNNEL_HOSTNAME=backend.example.com

./scripts/dev/run_cloudflared.sh
```

The script writes `${TP_CLOUDFLARED_HOST_FILE:-/tmp/tp-cloudflared-host}` so
that `scripts/setup/run_frontdoor_local.sh` and any future `dev-start`
orchestrator can append the tunnel hostname to `TP_TRUSTED_HOSTS`.

After the tunnel is up, point the frontdoor at the public hostname:

```bash
export TP_BACKEND_ORIGIN="https://${CLOUDFLARED_TUNNEL_HOSTNAME}"
export TP_FASTAPI_ORIGIN="${TP_BACKEND_ORIGIN}"
```

Then update Vercel project env vars per `docs/operations/frontdoor_vercel_env.md`
to use the same origin.

## Running a quick tunnel (development only)

```bash
unset CLOUDFLARED_TUNNEL_NAME
./scripts/dev/run_cloudflared.sh
```

The script:
- Starts `cloudflared tunnel --protocol http2 --url http://127.0.0.1:8000`.
- Watches stdout for the assigned `*.trycloudflare.com` hostname.
- Writes the hostname (without the `https://` prefix) to the sentinel file.

## Trusted Host middleware

The FastAPI app rejects requests whose `Host` header is not in
`TP_TRUSTED_HOSTS` (`app.py:802-806`). The frontdoor launcher reads the
sentinel file at startup and appends the tunnel hostname automatically. If you
launch the backend manually, append the hostname yourself:

```bash
export TP_TRUSTED_HOSTS="localhost,127.0.0.1,::1,testserver,${CLOUDFLARED_TUNNEL_HOSTNAME}"
make run-backend-local
```

Without this step, requests through the tunnel return `400 Invalid host header`.

## Troubleshooting

- **Quick tunnel hostname rotates** — that is expected; named tunnels solve it.
- **`failed to refresh DNS local resolver`** — almost always transient; the
  HTTP/2 protocol switch in `run_cloudflared.sh` reduces the failure rate.
- **`Invalid host header` on the backend** — the tunnel hostname is missing
  from `TP_TRUSTED_HOSTS`. Re-source the env file or restart with the value
  set.
