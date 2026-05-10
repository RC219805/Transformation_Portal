# Portal Telemetry Cohort Bucketing Evaluation

Status: evaluation for #1703
Last updated: 2026-05-10
Runtime behavior changed: no

## Purpose

This document evaluates whether portal telemetry should keep using the current
stable rollout bucket behavior or move to a less linkable cohort strategy. It is
an evaluation artifact only. It does not change RUM schema, event families,
emission paths, rollout controls, sinks, retention tooling, cookies,
sessionStorage, sanitizer behavior, or login/logout submit semantics.

## Current Behavior Inventory

The backend portal rollout helper in `app.py` derives a cohort key from the
portal actor with this precedence:

1. `username`
2. `accessEmail`
3. `role`
4. `TP_PORTAL_DIRECT_DEBUG_COHORT_KEY`, defaulting to `direct-debug`

The stable bucket helper lowercases and trims the cohort key, hashes it with
SHA-256, takes the first 32 bits of the digest, and reduces it modulo 100.
Missing or empty keys return the sentinel bucket `100`, which keeps those
requests outside rollout checks that use `bucket < percent`.

The portal RUM record stores only `cohort_bucket`, not the raw cohort key or the
full hash. The existing privacy packet correctly classifies the bucket as coarse
pseudonymous rollout telemetry because the value is stable for the same
pre-hash input.

The same stable bucket shape also appears in the managed frontdoor JavaScript
surface. `web/secure-landing/lib/rollout.js` hashes a normalized key to a
`0..99` bucket with `100` reserved for an empty key. Current uses include:

- Managed portal/bootstrap feature rollouts keyed from the authenticated
  session's username, access email, or role.
- Front-door RUM request sampling keyed from the request traceparent after
  #1707 added independent front-door RUM controls.

The front-door rollout-control work did not change backend `cohort_bucket`
semantics. It separated enablement and sampling controls so landing, login,
logout, and front-door-classified `/v1/portal/rum` payloads can be governed
independently from managed portal/bootstrap RUM.

## Evaluation Criteria

- Privacy and linkability: whether the bucket can be linked across sessions,
  surfaces, pilots, or exports.
- Rollout stability: whether the same actor remains consistently sampled in or
  out during a rollout window.
- Operational complexity: key management, rotation, migration, and incident
  response cost.
- Migration risk: compatibility with existing tests, aggregate evidence,
  dashboards, and pilot comparisons.

## Options

| Option | Privacy and linkability | Rollout stability | Operational complexity | Migration risk |
| --- | --- | --- | --- | --- |
| Keep current SHA-256 modulo-100 bucket from username, email, role, or direct-debug key | Lowest improvement. The raw input is not stored, but a stable bucket remains linkable at coarse granularity and the input sources include identity-derived values. | Strong. Existing rollout decisions are deterministic and already covered by tests. | Low. No key custody or migration process. | Lowest runtime risk, but leaves the #1703 privacy concern unresolved. |
| Opaque server-owned cohort key | Strongest general-purpose improvement. The rollout subject no longer needs username, email, or role as the pre-hash input. | Strong if the opaque key is stable for the intended actor or account lifetime. | Medium. Requires creation, storage, reset, and backfill policy for the opaque key. | Medium. Needs a migration plan and tests proving old identity-derived inputs are not used. |
| Keyed HMAC bucket over the current subject | Good improvement over plain SHA-256 because offline comparison requires the secret key. Still depends on the selected subject, so username/email use remains sensitive. | Strong. The same subject and key produce stable rollout decisions. | Medium. Requires secret custody, rotation policy, and key-version handling. | Medium. Can preserve rollout distribution but needs careful key rotation semantics. |
| Rotating salt strategy | Good time-bounded unlinkability when rotation windows are short. | Weaker. Rotation can move users between rollout cohorts unless the window is aligned with pilot boundaries. | Medium to high. Requires rotation schedule, evidence records, and rollback semantics. | Medium to high. Can disrupt longitudinal pilot comparisons and rollout stability. |
| Role-only or coarse cohort | Reduces direct identity-derived inputs. | Weak for sampling fairness because roles are low-cardinality and can cluster many users into the same bucket. | Low. Easy to reason about and operate. | Medium. May distort rollout distribution and make pilot evidence less representative. |
| No cohort bucket for front-door RUM | Strong for front-door linkability reduction because landing/login/logout telemetry would avoid persistent cohort grouping. | Depends on rollout implementation. Front-door sampling can still use request-local traceparent without storing the cohort bucket. | Low to medium. Requires schema or evidence-policy decision if the backend keeps one RUM schema. | Medium. Existing aggregate summaries and tests may expect `cohort_bucket` on all RUM records. |

## Recommendation

Implementation is warranted before expanding portal telemetry beyond the current
governed pilot posture, but the implementation should be a separate approved
runtime PR. This evaluation PR should remain documentation-only.

Recommended implementation path for a later PR:

- Retire username, access email, and role as pre-hash inputs for telemetry
  cohort bucketing where a stable actor bucket is still needed.
- Prefer an opaque server-owned cohort key for managed portal/bootstrap RUM and
  feature rollout decisions that need actor stability.
- Use a keyed HMAC bucket only as a migration bridge or when a stable external
  subject must be retained. Do not persist the HMAC digest, HMAC key, key
  version details, or raw subject in RUM records.
- Avoid rotating salts for normal rollout control unless Security/Privacy
  explicitly prioritizes short-window unlinkability over cohort stability.
- Avoid role-only bucketing for sampled rollout decisions because it is too
  coarse for stable percentage rollouts. It can remain an aggregate reporting
  dimension only if it is already otherwise allowed.
- Evaluate omitting `cohort_bucket` for front-door RUM records, or replacing it
  with a non-persistent request-local sampling decision, because #1707 now gives
  front-door RUM independent controls from managed portal/bootstrap RUM.

## Later Implementation Guardrails

Any behavior-changing PR that follows this evaluation should:

- Preserve `TP_PORTAL_RUM_ENABLED` as the shared master kill switch unless a
  later policy decision explicitly changes it.
- Preserve `TP_PORTAL_RUM_ROLLOUT_PERCENT` for managed portal/bootstrap RUM and
  `TP_FRONTDOOR_RUM_ENABLED` / `TP_FRONTDOOR_RUM_ROLLOUT_PERCENT` for
  front-door RUM unless the policy decision explicitly changes them.
- Update `docs/compliance/PORTAL_TELEMETRY_PRIVACY_SIGNOFF.md` when runtime
  semantics change.
- Add tests proving the selected cohort input no longer uses username, access
  email, or role when those inputs are retired.
- Keep raw identities, full hashes, HMAC digests, salts, keys, and key versions
  out of RUM records and raw-log retention evidence.
- Include migration evidence for aggregate reporting so old and new pilot data
  are not compared as if they used identical cohort semantics.

## Open Decision

Security/Privacy should choose one of these implementation decisions before a
runtime PR starts:

- Use opaque server-owned cohort keys for portal/bootstrap telemetry.
- Use keyed HMAC as a bridge while opaque cohort keys are introduced.
- Remove or replace `cohort_bucket` for front-door RUM only.
- Leave runtime behavior unchanged and document acceptance of the current coarse
  pseudonymous bucket risk.
