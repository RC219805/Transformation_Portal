# Portal + Orchestrator Quickstart

## Start the Service

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/`.

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

- `GET /ready` returns `{ok,time,version}` by default; set `TP_READY_VERBOSE=1` for extended runtime/security fields.
- `GET /v1/presets?pipeline=lux-depth-v3` dynamic UI preset catalog.
- `POST /v1/jobs` submit allowlisted job request.
- `GET /v1/jobs` bounded recent job snapshots (for refresh/recovery).
- `GET /v1/jobs/{id}` detailed job status (`logs_tail`, `error`, `artifacts`).
- `POST /v1/jobs/{id}/cancel` request cancellation.
- `GET /v1/jobs/{id}/events` SSE events: `state`, `log`, `progress`, `artifact`, `done`.

## SSE Authentication Note

When `TP_API_KEY` is configured:
- fetch-based endpoints use `Authorization: Bearer <token>` or `x-api-key`.
- SSE stream auth should use headers (`Authorization` or `x-api-key`).
- Query-string SSE auth (`?api_key=<token>`) is disabled by default and can be re-enabled only with `TP_ALLOW_SSE_QUERY_API_KEY=1`.

## Validation Commands

```bash
make test-orchestrator-contract
```

Direct pytest equivalent:

```bash
pytest -q tests/test_app_orchestrator_runtime.py tests/test_app_orchestrator_contract_http.py
```

Expected contract gate outcomes:
- `/v1/*` success and failure responses use typed envelope (`schema`, `success`, `data`, `error`).
- Validation failures return `400` with `error.code=INVALID_ARGUMENT`.
- Oversized request paths return `413` with typed error envelope.
- With `TP_API_KEY` set, `/v1/jobs*` and `/v1/jobs/{id}/events` enforce auth.
- SSE lifecycle includes `state`, `log`, `progress`, `artifact`, and terminal `done` events.
