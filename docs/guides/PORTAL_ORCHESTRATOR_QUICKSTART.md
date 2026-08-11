# Portal + Orchestrator Quickstart

## Start the Service

Standalone FastAPI origin:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/` for direct backend debugging, or use the secure front door quickstart for the managed browser path:
- [Portal Secure Front Door Quickstart](PORTAL_SECURE_FRONTDOOR_QUICKSTART.md)

## Security Controls

Set an API key for protected job endpoints (`/v1/jobs*`).
By default, the orchestrator now enforces API-key auth for job endpoints and returns `AUTH_CONFIGURATION_ERROR` if `TP_API_KEY` is unset.

```bash
export TP_API_KEY="replace-with-strong-token"
```

Additional hardening knobs:

```bash
export TP_RATE_LIMIT_PER_MINUTE=120
export TP_MAX_REQUEST_BYTES=1048576
export TP_ALLOWED_ORIGINS="http://localhost,http://127.0.0.1:8000"
```

Local-dev only opt-outs (not recommended for shared environments):

```bash
export TP_ENFORCE_JOB_API_KEY=0
export TP_ENABLE_API_DOCS=1
export TP_READY_VERBOSE=1
```

## API Contract Notes

As of PR #1562, the health/readiness routes are backed by typed OpenAPI
response models. That change documents the contract shape for generated clients;
it does not change the existing response bodies for `/healthz`, `/ready`, or
`/v1/readiness`.

## Response envelope

```json
{
  "schema": "tp.orchestrator.*.v1",
  "success": true,
  "data": {},
  "error": null
}
```

## Error object

```json
{
  "code": "INVALID_ARGUMENT",
  "message": "input_dir and output_dir are required",
  "details": {
    "field": "payload"
  }
}
```

## Endpoints

- `GET /ready` returns a shallow backend liveness signal.
- `GET /healthz` returns the backend's minimal liveness signal; the secure front
  door exposes its own `/healthz` route with the managed readiness checks.
- `GET /v1/readiness` returns the operator-truth execution readiness matrix for `lux-depth-v3`, `archive-gate-a`, `archive-gate-b`, and `archive-gate-c`.
- `GET /portal/bootstrap` returns the standalone portal bootstrap contract for `direct_debug` mode.
- `GET /v1/presets?pipeline=lux-depth-v3` dynamic UI preset catalog.
- `POST /v1/uploads/staging` stages multipart browser uploads under a governed `input_dir` and returns the staged input path plus baseline/capture/receipt artifact paths.
- `POST /v1/jobs` submit allowlisted job request.
- `GET /v1/jobs` bounded recent job snapshots (for refresh/recovery).
- `GET /v1/jobs/{id}` detailed job status (`logs_tail`, `error`, `artifacts`).
- `POST /v1/jobs/{id}/cancel` request cancellation.
- `GET /v1/jobs/{id}/events` SSE events: `state`, `log`, `progress`, `artifact`, `done`.

## Readiness Semantics

- `GET /ready` and `GET /healthz` are liveness probes. They tell you the service answered, not that a given pipeline is dispatchable.
- `GET /v1/readiness` is the execution-readiness contract. It reports per-pipeline `ready`, `degraded`, or `blocked` state plus `missing_prerequisites`, canonical command mapping, runner details, and safe operator notes.
- `lux-depth-v3` reports `base` readiness and a separate `canary_status`; canary unavailability does not block the safe local execution lane.
- `archive-gate-a` is normally `degraded` until an archive index is supplied.
- `archive-gate-b` and `archive-gate-c` are blocked by default until a rights-manifest JSONL is available.

## Run Gate A End-to-End

`archive-gate-a` maps to the canonical archive command `fixity-scan`.
In the portal build flow, `Input Dir` is forwarded as `--archive-root` automatically and `Output Dir` becomes the orchestration output root. Beyond those standard paths, the extra Gate A field you need to provide is an existing `Archive Index Path`.

Safe local fixture inputs already checked into this repo:

```bash
ARCHIVE_ROOT=./tests/fixtures/archive_small/archive_root
ARCHIVE_INDEX=./tests/fixtures/archive_small/archive_index_normalized.csv.gz
OUTPUT_DIR=/tmp/gate-a-smoke
```

### Direct CLI

This is the fastest no-UI smoke and is safe to run repeatedly:

```bash
.venv/bin/python tools/archive_governance.py --json fixity-scan \
  --archive-index "$ARCHIVE_INDEX" \
  --archive-root "$ARCHIVE_ROOT" \
  --out-dir "$OUTPUT_DIR" \
  --workers 1 \
  --no-validate-schemas
```

Expected Gate A artifacts:

- `$OUTPUT_DIR/hash_manifest.csv.gz`
- `$OUTPUT_DIR/hash_summary.json`
- `$OUTPUT_DIR/merkle_roots.json`

### HTTP Orchestrator

Start a clean backend first. If `127.0.0.1:8000` is already occupied, use another local port such as `8001`.

```bash
export TP_API_KEY="contract-secret"
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Submit Gate A directly through the orchestrator:

```bash
curl -sS \
  -H "Authorization: Bearer $TP_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST http://127.0.0.1:8001/v1/jobs \
  -d '{
    "pipeline": "archive-gate-a",
    "args": {
      "input_dir": "./tests/fixtures/archive_small/archive_root",
      "output_dir": "/tmp/gate-a-smoke-http",
      "archive_command": "fixity-scan",
      "archive_index": "./tests/fixtures/archive_small/archive_index_normalized.csv.gz"
    }
  }'
```

Poll the returned job id with `GET /v1/jobs/{id}` until `state` becomes `succeeded`.

### Portal UI

Open the build view:

```text
http://127.0.0.1:8001/portal?view=build
```

Then configure the form exactly like this:

1. Pipeline: `archive-gate-a`
2. Input Dir: `./tests/fixtures/archive_small/archive_root`
3. Output Dir: `/tmp/gate-a-smoke-portal`
4. Archive Index Path: `./tests/fixtures/archive_small/archive_index_normalized.csv.gz`
5. Leave the canonical command as `fixity-scan`

What the portal should show once configured:

- the `Archive Index Path` field is visible
- the CLI preview includes `--archive-command "fixity-scan"` and `--archive-index "...archive_index_normalized.csv.gz"`
- the missing-index warning clears
- the job runs to `Succeeded`
- run details list three indexed artifacts

If the form still shows the missing-index warning:

- that warning is expected when `Archive Index Path` is blank
- Gate A is not ready yet
- fill `Archive Index Path` with `./tests/fixtures/archive_small/archive_index_normalized.csv.gz`
- keep `Input Dir` pointed at `./tests/fixtures/archive_small/archive_root`
- the equivalent command will then include `--archive-index`

### Browser-Saved Build Profiles

Build profiles are stored only in the current browser and are scoped to the
resolved portal actor. Unsaved restored drafts are protected: choosing another
profile opens an explicit discard confirmation instead of replacing the draft.

Current managed profiles use the signed-in Access email and portal username as
one actor scope, so separate configured usernames remain isolated even when
they share an Access identity. Older portal versions used either one
browser-wide storage key or an Access-email-only managed key. In standalone
`direct_debug` mode, only the browser-wide legacy store migrates automatically
to the direct-debug scope when no scoped profiles exist. In managed mode, open
**Manage saved profile** and use **Import Legacy Profiles**; the two-step claim
prevents either ambiguous legacy store from being assigned silently. When a
legacy profile name collides with a current actor profile, the current profile
wins.

An unsaved managed draft created before the composite actor scope is not
claimed or deleted automatically. The Portal blocks Build edits and asks the
current user to **Claim & Recover Draft** or **Discard Draft Permanently**;
background persistence remains paused until that explicit choice succeeds.
If browser storage rejects a recovery write, the legacy draft stays preserved
and the recovery dialog remains open.

## SSE Authentication Note

There are now two supported browser paths:

- Managed front door mode:
  - the browser talks only to the front door on one origin
  - `/portal/bootstrap` returns `authMode: "managed"`
  - the browser does not hold the backend API key
  - unsafe requests use CSRF plus same-origin checks
- Standalone `direct_debug` mode:
  - the browser is pointed directly at the FastAPI origin
  - `/portal/bootstrap` returns `authMode: "direct_debug"`
  - the existing API-key workflow remains available for local debugging

When `TP_API_KEY` is configured for standalone `direct_debug` mode:
- fetch-based endpoints use `Authorization: Bearer <token>` or `x-api-key`.
- SSE stream auth should use headers (`Authorization` or `x-api-key`).
- Query-string SSE auth (`?api_key=<token>`) is disabled by default and can be re-enabled only with `TP_ALLOW_SSE_QUERY_API_KEY=1`.

## Validation Commands

```bash
make test-orchestrator-contract
make test-orchestrator-http-contract
make test-portal-contract
make validate-orchestrator-http
make validate-portal-browser
make audit-pipeline-readiness
```

`make validate-portal-browser` now preflights `POST /v1/config-preview` before
launching Chrome. The Make target now launches an isolated local backend by
default; if you want to aim the smoke at an already-running backend instead,
call `python scripts/validation/validate_portal_browser_smoke.py` directly
without `--spawn-local-backend`. When the backend is running in `direct_debug`
mode with `TP_API_KEY` configured, export the same `TP_API_KEY` in the shell
that runs the browser smoke or it will fail early with an explicit
preview-auth error.

Direct pytest equivalent:

```bash
pytest -q tests/test_app_orchestrator_runtime.py tests/test_app_orchestrator_contract_http.py tests/validation/test_portal_smoke_scripts.py
```

Front-door validation:

```bash
cd web/secure-landing
nvm use 22
npm test
npm run build
```

Expected contract gate outcomes:
- `/v1/*` success and failure responses use typed envelope (`schema`, `success`, `data`, `error`).
- `/healthz`, `/ready`, and `/v1/readiness` remain wire-compatible while exposing typed OpenAPI response models.
- `/v1/readiness` keeps transport success (`200`) separate from per-pipeline `ready` / `degraded` / `blocked` execution truth.
- Validation failures return `400` with `error.code=INVALID_ARGUMENT`.
- Oversized request paths return `413` with typed error envelope.
- With `TP_API_KEY` set, `/v1/jobs*` and `/v1/jobs/{id}/events` enforce auth.
- SSE lifecycle includes `state`, `log`, `progress`, `artifact`, and terminal `done` events.
- Live smokes (`validate-*`) exercise the running service and browser path; `test-*contract` targets stay fixture/contract-only.
