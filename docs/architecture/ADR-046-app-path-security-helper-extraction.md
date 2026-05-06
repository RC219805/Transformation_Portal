# ADR-046: App Path Security Helper Extraction Contract

**Status:** Proposed
**Date:** 2026-05-06
**Decision Makers:** Architect (review) + Specialist (implementation)
**Replaces:** None
**Supersedes:** None
**Related:** [ADR-045 Monolith Decomposition Residuals](ADR-045-monolith-decomposition-residuals.md), [Monolith Decomposition Targets](MONOLITH_DECOMPOSITION_TARGETS.md), [Portal Edge Hardening Implementation Standard](PORTAL_EDGE_HARDENING_IMPLEMENTATION_STANDARD.md)

---

## Executive Summary

Target 2B may extract `app.py` path resolution and filesystem trust helpers only after this security contract is reviewed. The destination module is `src/transformation_portal/portal/path_security.py`. The extraction must be compatibility-only: no behavior changes to allowed-root resolution, symlink handling, trusted entry walking, upload-root resolution, typed validation reasons, or public error envelopes.

`app.py` remains the legacy compatibility surface. Existing private helper access through `app.py` must continue to work for tests and internal callers.

---

## Context

`app.py` is still the largest monolith and contains security-sensitive path handling used by upload staging, archive dispatch, FastVLM runtime validation, artifact lookup, and job-output resolution. Target 2A moved the pure portal asset bundle seam. Target 2B is higher risk because these helpers decide what paths the service may read, write, create, or serve.

ADR-045 provides the general decomposition pattern, but it explicitly does not override module-specific app hardening. This ADR is the module-specific gate for Target 2B.

---

## Decision

Phase 2B extraction is allowed only under these constraints:

1. `src/transformation_portal/portal/path_security.py` owns pure path-security helpers and must not import `app.py`.
2. `app.py` re-exports or wraps the legacy private helper names so existing imports and monkeypatch-based tests continue to work.
3. `_PortalValidationReasonError` stays in `app.py` for Phase 2B. Moving it requires a later ADR/update that covers the full app validation-reason surface.
4. `PORTAL_UPLOAD_ROOT`, `ALLOWED_INPUT_ROOTS`, `ALLOWED_OUTPUT_ROOTS`, and `ALLOWED_PATH_ROOTS` remain app-owned runtime configuration for Phase 2B.
5. No call-site rewrites outside `app.py` are allowed in the extraction PR unless a compatibility test proves an existing public contract already imports the new module directly.

The extraction surface approved for Phase 2B is:

- `_normalize_root_path`
- `_default_allowed_path_roots`
- `_env_path_roots`
- `_resolve_untrusted_request_path`
- `_validate_path_against_roots`
- `_resolve_allowed_request_path`
- `_path_is_within_root`
- `_trusted_allowed_entry`
- `_trusted_existing_dir`
- `_trusted_creatable_dir`
- `_ensure_safe_regular_file_path`

`_resolved_portal_upload_root` remains an `app.py` compatibility wrapper that uses app-owned runtime configuration and the extracted resolver.

---

## Security Review Checklist

The Phase 2B extraction PR must show explicit review evidence for:

- Traversal rejection: empty values, tilde shorthand, NUL bytes, and outside-root escapes.
- Symlink escape rejection for existing files, existing directories, runtime paths, and artifact paths.
- Missing path behavior for required existing files/directories versus creatable output directories.
- Creatable output directory behavior, including no pre-admission directory creation.
- Archive path callers: archive index, archive root, manifests, reports, policy files, bag directories, and rights sidecars.
- Upload path callers: portal upload root resolution and staged upload input directories.
- FastVLM path callers: runtime root, Python executable, model selector paths, and non-blocking missing-runtime status.
- Artifact-serving callers: job output directory resolution, artifact lookup, preview proxy paths, and attachment filenames.
- Typed error compatibility: invalid-path and outside-root reasons remain stable in internal exceptions and public envelopes.

---

## Implementation Plan

1. Prep PR: land this ADR, update the Target 2B readiness row, and add extraction-readiness tests that still import through `app.py`.
2. Extraction PR: add `portal/path_security.py`, import helpers into `app.py`, keep `_PortalValidationReasonError` and runtime config in `app.py`, and preserve all helper names.
3. Follow-up only if needed: evaluate whether broader validation-reason objects can move without changing app-wide typed error behavior.

---

## Success Metrics

- Existing `app.py` path/security tests stay green before and after extraction.
- `app.py` remains the compatibility import surface for all listed private helper names.
- No route shape, selector, API envelope, CLI argument, output filename, or public error-code behavior changes.
- The extraction PR cites this ADR and passes ADR-045 governance gates.

---

## References

- `app.py` path helper region and callers.
- `tests/test_app_security.py`
- `tests/test_portal_backend_hardening.py`
- `docs/architecture/MONOLITH_DECOMPOSITION_TARGETS.md`
