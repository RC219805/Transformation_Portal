# Portal Orchestrator Roadmap (Re-Baselined)

Date: 2026-03-01  
Scope: `app.py` + `portal.html` (single-file UI + FastAPI wrapper)

## Objective
Re-baseline the portal roadmap against current repository reality, then focus only on remaining high-impact gaps.

## Recommendation Matrix

| Report Recommendation | Status | Evidence | Action |
|---|---|---|---|
| Build backend wrapper (`/ready`, jobs, status, cancel, SSE) | Shipped | `app.py` provides `/ready`, `POST /v1/jobs`, `GET /v1/jobs/{id}`, `POST /v1/jobs/{id}/cancel`, `GET /v1/jobs/{id}/events` | Keep and extend |
| Build queue/run/cancel UX | Shipped | `portal.html` has queue rendering, cancel actions, and EventSource stream handling | Keep and optimize |
| Add CSP/security hardening | Partial | UI has CSP meta; backend had security headers, auth/rate-limit controls | Added server CSP header + stronger auth scope |
| Dynamic preset discovery API | Pending (now implemented) | `GET /v1/presets?pipeline=<name>` added in `app.py` | Integrated in UI |
| Recover history across refresh | Pending (now implemented) | `GET /v1/jobs` added; UI recovery flow added | Validate with focused tests |
| Artifact indexing and surfacing | Pending (now implemented) | Artifact indexing in runner + SSE `artifact` event + UI artifact panel | Validate with focused tests |
| Typed error envelope consistency | Partial (now improved) | Validation/read/auth/rate-limit responses now use typed envelope for API paths | Continue migration for remaining plain errors over time |

## Current Contract Surfaces

### Endpoints
- `GET /ready`
- `GET /v1/presets?pipeline=<name>`
- `POST /v1/jobs`
- `GET /v1/jobs`
- `GET /v1/jobs/{id}`
- `POST /v1/jobs/{id}/cancel`
- `GET /v1/jobs/{id}/events` (SSE events: `state`, `log`, `progress`, `artifact`, `done`)

### Envelope
All orchestrator API endpoints use:

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
- In progress:
  - Focused runtime tests updated/expanded for new endpoints and artifact indexing.
  - Quickstart + API notes added (`docs/guides/PORTAL_ORCHESTRATOR_QUICKSTART.md`).

## Release Acceptance Checklist
- [ ] `GET /v1/presets` drives lux-depth preset selector.
- [ ] Refresh recovers job list and reconnects streams for running jobs.
- [ ] Artifact events populate artifact panel for completed jobs.
- [ ] API key blocks unauthorized read/list/event access when `TP_API_KEY` is set.
- [ ] Validation errors return typed envelope and `400` semantics.
- [ ] Regression behavior for `/ready`, job submit/cancel, and SSE progress/logs remains intact.
