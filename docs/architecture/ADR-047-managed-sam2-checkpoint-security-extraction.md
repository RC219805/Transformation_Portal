# ADR-047: Managed SAM2 Checkpoint Security Extraction Contract

**Status:** Implemented
**Date:** 2026-05-06
**Status changed:** 2026-05-16 (Proposed → Implemented; extraction shipped as `src/transformation_portal/portal/sam2_checkpoint_security.py` with `app.py` re-export shims preserved per the compatibility-only contract)
**Decision Makers:** Architect (review) + Specialist (implementation)
**Replaces:** None
**Supersedes:** None
**Related:** [ADR-045 Monolith Decomposition Residuals](ADR-045-monolith-decomposition-residuals.md), [ADR-046 App Path Security Helper Extraction Contract](ADR-046-app-path-security-helper-extraction.md), [Monolith Decomposition Targets](MONOLITH_DECOMPOSITION_TARGETS.md), [Dependency ML Alert Triage](../governance/DEPENDABOT_ML_ALERT_TRIAGE_2026-04-16.md)

---

## Executive Summary

Target 2C may extract managed SAM2 checkpoint validation and checksum-cache helpers only after this contract is reviewed. The future destination module is `src/transformation_portal/portal/sam2_checkpoint_security.py`. The extraction must be compatibility-only: no behavior changes to repo-controlled missing checkpoint acceptance, external checksum trust, bounded cache eviction, reason codes, preview errors, dispatch errors, or SAM2 backend semantics.

`app.py` remains the legacy compatibility surface. Existing private helper access through `app.py` must continue to work for tests and internal callers.

---

## Context

Managed portal/orchestrator SAM2 paths are security-sensitive because they admit model checkpoint files into a server-side execution path. Current behavior deliberately accepts repo-controlled SAM2 paths before the artifact exists locally, but external checkpoint files are accepted only when their SHA-256 digest appears in the app-owned trusted SHA set.

Target 2B extracted pure path-security helpers behind ADR-046. Target 2C is the next adjacent app decomposition slice, but its trust-source boundary must be explicit before extraction because the runtime currently keeps SAM2 trusted checkpoint digests in app-owned constants rather than loading them from `config/model_lock_manifest.yaml`.

---

## Decision

Target 2C extraction is allowed only under these constraints:

1. `src/transformation_portal/portal/sam2_checkpoint_security.py` owns pure SAM2 checkpoint validation and checksum-cache helper logic and must not import `app.py`.
2. `app.py` wraps or re-exports the legacy private helper and model names so existing imports and monkeypatch-based tests continue to work.
3. `_PortalValidationReasonError` stays in `app.py`; the extracted module must use its own private validation error carrying a stable `reason`.
4. `ALLOWED_INPUT_ROOTS`, `MANAGED_SAM2_TRUSTED_ROOTS`, `MANAGED_SAM2_TRUSTED_SHA256`, `MANAGED_SAM2_CHECKSUM_MAX_BYTES`, `_MANAGED_SAM2_CHECKSUM_CACHE_LOCK`, and public error-envelope mapping remain app-owned runtime configuration for Target 2C.
5. The extracted module must accept app-owned trust/config values by parameter or wrapper injection.
6. The current app-owned trusted SHA semantics are the Target 2C extraction contract. Migrating SAM2 checkpoint trust into `config/model_lock_manifest.yaml` requires a separate ADR/update and migration PR.
7. No call-site rewrites outside `app.py` are allowed in the extraction PR unless a compatibility test proves an existing public contract already imports the new module directly.

The extraction surface approved for Target 2C is:

- `ManagedSam2CheckpointValidationResult`
- `_ManagedSam2ChecksumCacheEntry`
- `_Sam2CacheKey`
- `_ManagedSam2BoundedChecksumCache`
- `_managed_sam2_reason_message`
- `_managed_sam2_checksum_cache_key`
- `_clear_managed_sam2_checksum_cache`
- `_hash_file_sha256`
- `_cached_managed_sam2_checksum_result`
- `_resolve_managed_sam2_checkpoint_validation`
- `_validate_managed_sam2_checkpoint_path`

---

## Security Review Checklist

The Target 2C extraction PR must show explicit review evidence for:

- Repo-controlled missing checkpoint paths remain accepted only under trusted SAM2 roots.
- External checkpoint files remain rejected unless their SHA-256 digest is trusted.
- Oversized external checkpoint files fail before hashing with `checkpoint_file_too_large`.
- Invalid path values, outside-root escapes, and non-file paths preserve existing reason codes.
- Preview and dispatch paths preserve matching `sam2_checkpoint_path` error codes.
- The bounded checksum cache preserves key fields, hash reuse, cache clearing, and FIFO eviction behavior.
- The extracted module does not import `app.py` and cannot silently read app globals.
- Any future model-lock manifest migration is reviewed separately and does not ride along with the extraction PR.

---

## Implementation Plan

1. Prep PR: land this ADR, update the Target 2C readiness row, and add extraction-readiness tests that still import through `app.py`.
2. Extraction PR: add `portal/sam2_checkpoint_security.py`, delegate from `app.py`, keep app-owned runtime configuration and `_PortalValidationReasonError` in `app.py`, and preserve all listed helper names.
3. Follow-up only if approved: migrate SAM2 checkpoint trust into a governed manifest source with explicit tests for manifest loading, fallback behavior, and error semantics.

---

## Success Metrics

- Existing SAM2 checkpoint preview, dispatch, and app runtime tests stay green before and after extraction.
- `app.py` remains the compatibility import surface for all listed private helper and model names.
- No route shape, selector, API envelope, CLI argument, output filename, reason code, or public error-message behavior changes.
- The extraction PR cites this ADR and passes ADR-045 governance gates.

---

## References

- `app.py` managed SAM2 checkpoint validation helpers and callers.
- `tests/test_app_orchestrator_runtime.py`
- `tests/test_app_orchestrator_contract_http.py`
- `docs/governance/DEPENDABOT_ML_ALERT_TRIAGE_2026-04-16.md`
