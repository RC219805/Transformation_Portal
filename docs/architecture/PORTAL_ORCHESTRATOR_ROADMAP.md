# Portal Orchestrator Roadmap (Re-Baselined)

Date: 2026-03-01
Scope: `app.py` + the root `portal.html` shell generated from
`web/secure-landing/portal-src/portal.template.js` and deferred portal surfaces.
Current-state refresh: 2026-05-12.

## Objective
Re-baseline the portal roadmap against current repository reality, then focus only on remaining high-impact gaps.

## Recommendation Matrix

| Report Recommendation | Status | Evidence | Action |
|---|---|---|---|
| Build backend wrapper (`/ready`, jobs, status, cancel, SSE) | Shipped | `app.py` provides `/ready`, `POST /v1/jobs`, `GET /v1/jobs/{id}`, `POST /v1/jobs/{id}/cancel`, `GET /v1/jobs/{id}/events` | Keep and extend |
| Build queue/run/cancel UX | Shipped | `portal.html` and `web/secure-landing/portal-src/portal.template.js` cover queue rendering, cancel actions, and EventSource stream handling, with larger review/operate surfaces carved into deferred modules | Keep and optimize |
| Add CSP/security hardening | Partial | UI has CSP meta; backend had security headers, auth/rate-limit controls | Added server CSP header + stronger auth scope |
| Dynamic preset discovery API | Pending (now implemented) | `GET /v1/presets?pipeline=<name>` added in `app.py` | Integrated in UI |
| Recover history across refresh | Pending (now implemented) | `GET /v1/jobs` added; UI recovery flow added | Validate with focused tests |
| Artifact indexing and surfacing | Pending (now implemented) | Artifact indexing in runner + SSE `artifact` event + UI artifact panel | Validate with focused tests |
| Typed error envelope consistency | Completed in Phase 5A | `/v1/*` validation, auth/rate-limit, middleware body-size, and routed `HTTPException` paths return typed envelope | Maintain with contract gate |

## Current Contract Surfaces

### Endpoints
- `GET /ready`
- `GET /v1/presets?pipeline=<name>`
- `POST /v1/jobs`
- `GET /v1/jobs`
- `GET /v1/jobs/{id}`
- `GET /v1/jobs/{id}/artifacts/{artifact_path}`
- `POST /v1/jobs/{id}/cancel`
- `GET /v1/jobs/{id}/events` (SSE events: `state`, `log`, `progress`, `artifact`, `done`)
- `/v2/jobs` parity routes for create/list/detail/artifacts/cancel/events.
  The route inventory contract requires `/v2/jobs/...` to mirror `/v1/jobs/...`
  method coverage; response schema names remain the existing
  `tp.orchestrator.*.v1` envelopes unless intentionally changed.

### Envelope
All JSON `/v1` orchestrator API endpoints and the `/v2/jobs` parity surface use
this envelope for application-level success and failure, including middleware and
routed `HTTPException` paths. Non-API routes like `/ready` keep native FastAPI
response shapes.

```json
{
  "schema": "tp.orchestrator.*.v1",
  "success": true,
  "data": {},
  "error": null
}
```

Error shape:

```json
{
  "code": "INVALID_ARGUMENT",
  "message": "human-readable summary",
  "details": {}
}
```

## Execution Phases (Fast-Ship)

### Phase 0: Re-baseline
- Completed by this document.

### Phase 1: API completion
- Completed:
  - Presets endpoint.
  - Job list endpoint.
  - Expanded job payload (`error`, `artifacts`, `events_url`).
  - Artifact indexing and SSE artifact events.

### Phase 2: UI contract alignment
- Completed:
  - Presets fetched from backend (fallback retained).
  - Job recovery via `GET /v1/jobs`.
  - Artifact panel from status + SSE.
  - Typed backend error presentation in UI.

### Phase 3: Security/hardening
- Completed:
  - API key requirement expanded to job read/list/events when configured.
  - Server CSP header added.
  - Typed error envelopes for validation and auth/rate-limit cases.

### Phase 4: Validation and docs
- Completed in Phase 5A:
  - Route-level contract suite added: `tests/test_app_orchestrator_contract_http.py`.
  - Runtime coverage retained/expanded: `tests/test_app_orchestrator_runtime.py`.
  - Quickstart/README updated with executable contract gate commands.

## Release Sign-Off Traceability Matrix

| Acceptance Item | Implementation Evidence | Automated Evidence |
|---|---|---|
| `GET /v1/presets` drives lux-depth preset selector | `app.py:list_presets`, `portal-src/portal.template.js:fetchPresetsForPipeline` + `applyPipelinePresetOptions` | `tests/test_app_orchestrator_contract_http.py::test_presets_contract_for_lux_depth_pipeline` |
| Refresh recovers job list and reconnects streams | `app.py:list_jobs` + `job_events`, `portal-src/portal.template.js:recoverJobs` + `startJobEventStream` | `tests/test_app_orchestrator_contract_http.py::test_jobs_list_and_detail_include_recovery_fields` |
| Artifact events populate artifact panel | `app.py:_index_job_artifacts` + SSE `artifact` event, `portal-src/portal.template.js:upsertArtifact` + `web/secure-landing/portal-src/review-surface-deferred.js:renderArtifactPanel` | `tests/test_app_orchestrator_contract_http.py::test_job_events_stream_emits_state_log_progress_artifact_done`, `tests/test_app_orchestrator_contract_http.py::test_artifact_indexing_truncation_visible_via_job_status` |
| API key blocks unauthorized read/list/event access | `app.py:security_layer` + `_has_valid_api_key` | `tests/test_app_orchestrator_contract_http.py::test_v1_routes_enforce_api_key_for_reads_and_events` |
| Validation errors return typed envelope + 400 | `app.py:create_job` validation path | `tests/test_app_orchestrator_contract_http.py::test_invalid_job_payload_returns_typed_invalid_argument` |
| Oversized request paths return typed envelope + 413 | `app.py:_enforce_content_length_limit`, `app.py:http_exception_handler` | `tests/test_app_orchestrator_contract_http.py::test_oversized_v1_request_returns_typed_413_envelope` |
| Regression behavior for `/ready`, submit/cancel, and SSE/log/progress intact | `app.py` endpoint surfaces unchanged | `tests/test_app_orchestrator_runtime.py::test_run_job_is_async_and_does_not_block_event_loop`, `tests/test_app_orchestrator_runtime.py::test_cancel_request_terminates_running_job`, `tests/test_app_orchestrator_runtime.py::test_sse_broadcast_delivers_events_to_multiple_subscribers`, `tests/test_app_orchestrator_contract_http.py::test_ready_keeps_non_enveloped_shape` |

## Release Acceptance Checklist
- [x] `GET /v1/presets` drives lux-depth preset selector.
- [x] Refresh recovers job list and reconnects streams for running jobs.
- [x] Artifact events populate artifact panel for completed jobs.
- [x] `/v2/jobs` route parity preserves the `/v1/jobs` job lifecycle surface.
- [x] API key blocks unauthorized read/list/event access when `TP_API_KEY` is set.
- [x] Validation errors return typed envelope and `400` semantics.
- [x] Oversized payload paths on `/v1/*` return typed envelope and `413` semantics.
- [x] Regression behavior for `/ready`, job submit/cancel, and SSE progress/logs remains intact.
