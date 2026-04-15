# Portal Telemetry Privacy Sign-Off Packet

**Status:** Pending Human Approval
**Date:** 2026-04-14
**Owner:** Frontdoor / Platform
**Review Required From:** Security / Privacy

This packet inventories the current portal telemetry contract and the proposed pilot retention posture. It is a preparation document for human approval. It is not itself an approval record.

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
