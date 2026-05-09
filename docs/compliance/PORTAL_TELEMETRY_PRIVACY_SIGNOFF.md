# Portal Telemetry Privacy Sign-Off Packet

**Status:** Pending Human Approval
**Date:** 2026-04-14
**Last revised:** 2026-05-09
**Owner:** Frontdoor / Platform
**Review Required From:** Security / Privacy

This packet inventories the current portal telemetry contract and the proposed pilot retention posture. It is a preparation document for human approval. It is not itself an approval record.

The 2026-05-09 revision adds eight RUM event families that landed in #1682, #1684, and #1696, and discloses the two server-set marker cookies introduced by the client-side login submit RUM mirror series (#1689 / #1694 / #1695). No metric values, sanitizer rules, or rollout knobs were changed; only the inventory below was extended to match what is in the repo.

## Purpose

Approve or reject the telemetry schema, retention posture, and disposal procedure used for bounded portal modernization pilots.

## Current Repo-Backed Data Collection

### Portal RUM Sink

Endpoint: `/v1/portal/rum`

Current schema fields captured by repo code:

- `schema`
- `timestamp`
- `event_type`
- `route`
- `view`
- `metric`
- `value`
- `unit`
- `metadata` after token-safe sanitization
- `trace_id`
- `cohort_bucket`
- `auth_mode`

Current supported event families:

- `portal_shell_rendered`
- `bootstrap_ready`
- `first_view_interactive`
- `core_web_vital`
- `queue_request`
- `sse_reconnect`
- `landing_rendered` *(added in #1682)*
- `login_rendered` *(added in #1682)*
- `login_submit_attempt` *(added in #1684; client mirror in #1689)*
- `login_submit_success` *(added in #1684; client mirror in #1695)*
- `login_submit_failure` *(added in #1684; client mirror in #1694)*
- `logout_submit_attempt` *(added in #1696)*
- `logout_submit_success` *(added in #1696)*
- `logout_submit_failure` *(added in #1696)*

Allowed event-type / metric / unit triples are pinned by `PORTAL_ALLOWED_RUM_EVENT_TYPES`, `PORTAL_ALLOWED_RUM_METRICS`, and `PORTAL_ALLOWED_RUM_UNITS` in `app.py`. Coverage is split across the repo: `tests/test_app_orchestrator_contract_http.py` round-trips the HTTP sink contract for `queue_request`, `core_web_vital`, and the login/logout submit families; `tests/test_app_orchestrator_runtime.py` pins the portal bundle emissions for `portal_shell_rendered`, `first_view_interactive`, `queue_request`, and `sse_reconnect`; `web/secure-landing/tests/rum-client.test.mjs`, `web/secure-landing/tests/login-rum.test.mjs`, and `web/secure-landing/tests/logout-rum.test.mjs` pin the managed frontdoor render and submit emitters. Widening the inventory requires paired schema and coverage updates in the same change.

### RUM Marker Cookies

The client-side login submit RUM mirrors (#1689 / #1694 / #1695) require two server-set cookies so the browser can correlate a submit attempt with its terminal outcome across the cross-page redirect. Both cookies are user-device state and are inventoried here so reviewers can assess them alongside the RUM payload schema.

| Cookie name | Path | Max-Age | Sample value | Set when | Cleared when |
|---|---|---|---|---|---|
| `tp_login_submit_failure` | `/login` | 60 s | `csrf` / `configuration` / `access` / `throttled` / `invalid` | POST `/login` returns a 303 with `?error=<code>` | Immediate `/login` GET after the failure redirect reads + clears; a later successful POST also clears any stale failure marker server-side |
| `tp_login_submit_success` | `/` | 60 s | `1` (fixed presence marker) | POST `/login` returns a 303 to `/portal` | Portal bundle reads + clears on first `/portal` load |

Properties of both markers:

- `httpOnly: false` — the inline rum-client.js script and the portal bundle both need to read these from JavaScript.
- `secure` — tracks `config.sessionCookieSecure` (true in production HTTPS, false in local-dev HTTP loopbacks only).
- `sameSite: lax` — lets the cookie ride the same-origin 303 redirect from POST to GET.
- 60-second max-age — caps stale-tab false positives; longer than the 99th-percentile submit-to-render latency, shorter than a casual reopen.

Cookie values are deterministic and bounded:

- The failure cookie value is one of the five values in `LOGIN_RUM_FAILURE_CODES` (`csrf`, `configuration`, `access`, `throttled`, `invalid`). It is not derived from user input.
- The success cookie value is the fixed string `1`. The actual submit-to-render duration is computed from a `sessionStorage` breadcrumb (`tpLoginSubmitStartedAt`) the client wrote at submit time; the cookie itself carries no timing information.

Cookie clear discipline (`clearRumMarkerCookie` in `lib/rum-client.js`) writes an empty value with `expires=epoch` on the original `Path` so the browser drops the cookie immediately on receipt of the response. Cross-marker hygiene is enforced: a successful login also clears any stale failure marker, and a failed login clears any stale success marker, so the two markers can never both be live for the same submit.

The logout route (#1696) currently sets no marker cookies on the user device. Server-side `logout_submit_*` events fire from `/logout` directly without a client-mirror handshake, because the portal exposes no logout button today; a future client-mirror PR for logout would extend this section accordingly.

### Portal Event Sink

Endpoint: `/v1/portal/events`

Current review-relevant event families:

- `artifact_viewer_opened`
- `artifact_viewer_fallback`
- `artifact_opened`
- `artifact_compared`
- `run_details_opened`
- `stream_reconnected`

Current review-relevant metadata keys emitted by repo code:

- `job_id`
- `pipeline`
- `media_kind`
- `artifact_fingerprint`
- `viewer_mode`
- `fallback_reason`

## Sanitization and Excluded Data

Repo-backed sanitization currently guarantees:

- metadata keys must be token-safe
- metadata string values must be token-safe
- unsupported keys and values are dropped
- raw paths are dropped
- plain-text usernames are not persisted in the RUM sink
- plain-text access emails are not persisted in the RUM sink

### Bounded Metadata Keys for the Login / Logout Submit Families

The login-submit and logout-submit event families carry only the metadata keys below. The token-safe sanitizer in `app.py:_portal_sanitize_metadata` would drop any other key or value, but listing the closed enums here gives reviewers an exact ceiling on what flows through the sink.

- `failure_code` — set on `login_submit_failure` and `logout_submit_failure` only.
  - Login enum: `csrf`, `configuration`, `access`, `throttled`, `invalid` (`LOGIN_RUM_FAILURE_CODES` in `lib/rum-emitter.js`).
  - Logout enum: `csrf` only (`LOGOUT_RUM_FAILURE_CODES` in `lib/rum-emitter.js`); the route has one failure surface today, audited as `csrf_failure`.
- `source` — set to the literal string `"client"` by the browser-side mirrors only (#1689 / #1694 / #1695). Lets dashboards distinguish a server-side emission from a client-side mirror so the two are not double-counted.

Neither key carries user input: every value comes from a closed-enum constant in repo code.

The pilot is not intended to capture:

- message content
- source file paths
- plain-text email addresses
- plain-text usernames
- browser-entered secrets
- free-form notes

## Rollout and Sink Controls

Current rollout and sink knobs:

```bash
export TP_PORTAL_RUM_ENABLED=0
export TP_PORTAL_RUM_ROLLOUT_PERCENT=0
export TP_PORTAL_RUM_LOG_PATH="/absolute/path/to/portal-rum.jsonl"
export TP_PORTAL_EVENT_LOG_PATH="/absolute/path/to/portal-events.jsonl"
```

Observed sink behavior:

- both sinks are optional
- both sinks append JSONL records when enabled
- the repo does not currently implement automatic retention, rotation, or deletion
- the sinks must therefore be treated as short-lived pilot artifacts, not long-term ledgers

## Proposed Pilot Retention and Disposal Procedure

Proposed for approval:

1. Keep raw pilot JSONL logs only in an operator-owned restricted path.
2. Retain raw JSONL for no longer than 14 calendar days after pilot end.
3. Extract aggregate evidence with `tools/portal_modernization_evidence.py`.
4. Preserve only the aggregate evidence output needed for RFC or rollout review.
5. Delete raw JSONL once the aggregate evidence has been attached and reviewed.

## Questions Requiring Human Approval

- Is the current schema sufficiently minimized for bounded pilot use?
- Is the proposed 14-day raw-log retention window acceptable?
- Are additional masking or field removals required before any rollout expansion?
- Is the current disposal procedure sufficient, or is a stronger deletion control required?

## Approval Block

Pending reviewer completion:

- Reviewer:
- Role:
- Review date:
- Decision:
- Conditions or required changes:
