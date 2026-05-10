# Portal Operator Console Modernization Evidence

**Status:** Working evidence register
**Date:** 2026-04-14
**Last revised:** 2026-05-10
**Owner:** Frontdoor / Platform

This note tracks the repo-owned evidence package for the portal modernization RFC. It does not create human approval or invent pilot results. It records what the repo can prove now, how pilot measurements should be collected, and which gates remain open.

## Repo-Backed Evidence Available Now

### Boundary and Rollout Controls

- Deterministic rollout helpers back portal RUM, the artifact viewer modal, and deferred review-surface loading in `app.py`.
- The managed frontdoor quickstart already documents the rollout knobs for portal RUM and review-surface pilots.

### Measurement Path

- `/v1/portal/rum` accepts the portal RUM schema, records a stable cohort bucket, and emits trace correlation fields.
- `tools/portal_rum_summary.py` remains the RUM-only summary tool.
- `tools/portal_modernization_evidence.py` summarizes RUM plus optional portal event logs for repo-measurable M1, M4, and M5 evidence.

### M2 Accessibility Contract

- The accepted M2 contract is the existing repo-native browser-probe coverage for `/login` and `/portal`, plus manual keyboard and reduced-motion verification.
- This RFC refresh does not require `axe` or another broader automated suite to keep M2 implemented.

### M3 Resilience Contract

- Managed `returnTo` validation, same-route recovery, and transient build-draft restore are already covered by browser and contract validation.

### M4 Performance Contract

- Portal asset budgets are enforced in repo validation.
- Deferred review loading and related portal performance work are shipped, but M4 still needs measured pilot evidence before formal closure.

### M5 Current Shipped Scope

- Deferred review loading
- Modal artifact viewer
- Keyboard-only navigation and zoom controls
- Integrity metadata visibility and fingerprint copy
- Explicit non-preview fallback states
- Viewer open and fallback telemetry

Optional review-time segmentation refinement is not part of the shipped scope captured by this evidence note.

## Pilot Collection Procedure

1. Enable the pilot sinks:

```bash
export TP_PORTAL_RUM_ENABLED=1
export TP_PORTAL_RUM_ROLLOUT_PERCENT=100
export TP_FRONTDOOR_RUM_ENABLED=1
export TP_FRONTDOOR_RUM_ROLLOUT_PERCENT=100
export TP_PORTAL_RUM_LOG_PATH="/absolute/path/to/portal-rum.jsonl"
export TP_PORTAL_EVENT_LOG_PATH="/absolute/path/to/portal-events.jsonl"
```

`TP_PORTAL_RUM_ENABLED` is the shared master kill switch and `TP_PORTAL_RUM_ROLLOUT_PERCENT` only governs managed portal/bootstrap RUM. The independent `TP_FRONTDOOR_RUM_*` gates added in #1707 must also be set or `make validate-frontdoor-browser` will silently produce zero landing/login/logout samples in the JSONL log even though the pilot appears to run.

2. Run the current browser validation backbone:

```bash
make validate-frontdoor-browser
make validate-portal-browser
```

3. Summarize the pilot logs:

```bash
python tools/portal_modernization_evidence.py \
  --rum-log /absolute/path/to/portal-rum.jsonl \
  --event-log /absolute/path/to/portal-events.jsonl \
  --operator-hours 8
```

4. Attach the command output and the pilot date range to the rollout notes or architecture review packet.

## Evidence Slots

| Gate | Repo-Owned Input | Current State |
| --- | --- | --- |
| M1 visibility | `portal_rum_summary.py`, `portal_modernization_evidence.py` | Implemented, waiting for pilot capture |
| M1 telemetry sign-off | `docs/compliance/PORTAL_TELEMETRY_PRIVACY_SIGNOFF.md` | `Approved with Conditions` for bounded portal/front-door pilots |
| M4 pilot metrics | `portal_modernization_evidence.py` plus browser validation | Open |
| M5 viewer evidence | `portal_modernization_evidence.py` plus browser validation | Open |
| ADR-050 rewrite evidence | `docs/decisions/ADR-050-portal-react-migration.md` | Open |

## Remaining Open Gates

- Pilot measurements for CWV, queue latency, SSE reconnect rate, and artifact-viewer success
- Privacy packet revision before any new RUM event family, metadata key, marker cookie, storage key, rollout knob, sink behavior, or retention posture
- ADR-050 evidence that compares rewrite vs no-rewrite delivery, quality, and developer-experience tradeoffs
