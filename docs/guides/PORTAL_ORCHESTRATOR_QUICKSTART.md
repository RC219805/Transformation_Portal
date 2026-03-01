# Portal + Orchestrator Quickstart

## Start the Service

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/`.

## Optional Security Controls

Set an API key to protect job endpoints (`/v1/jobs*`):

```bash
export TP_API_KEY="replace-with-strong-token"
```

Optional hardening knobs:

```bash
export TP_RATE_LIMIT_PER_MINUTE=120
export TP_MAX_REQUEST_BYTES=1048576
export TP_ALLOWED_ORIGINS="http://localhost,http://127.0.0.1:8000"
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

- `GET /ready` health and runtime security settings.
- `GET /v1/presets?pipeline=lux-depth-v3` dynamic UI preset catalog.
- `POST /v1/jobs` submit allowlisted job request.
- `GET /v1/jobs` bounded recent job snapshots (for refresh/recovery).
- `GET /v1/jobs/{id}` detailed job status (`logs_tail`, `error`, `artifacts`).
- `POST /v1/jobs/{id}/cancel` request cancellation.
- `GET /v1/jobs/{id}/events` SSE events: `state`, `log`, `progress`, `artifact`, `done`.

## SSE Authentication Note

When `TP_API_KEY` is configured:
- fetch-based endpoints use `Authorization: Bearer <token>` or `x-api-key`.
- EventSource stream can use `?api_key=<token>` query parameter.

## Validation Commands

```bash
pytest -q tests/test_app_orchestrator_runtime.py
```

Optional local CI subset:

```bash
make test-fast
```
