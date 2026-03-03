# Security Best Practices Report

Date: 2026-03-03  
Repository: `Transformation_Portal`  
Scope: API service (`app.py`), core runtime (`src/`), and operational tooling (`scripts/`, `tools/`).

## Executive Summary

This review found **6 actionable security findings**:
- **2 High**
- **3 Medium**
- **1 Low**

The two highest-priority issues are:
1. Resource-exhaustion exposure in the job API (no default rate limit and no concurrency cap).
2. Unrestricted filesystem paths accepted from job API payloads.

The codebase also has several strong controls already in place (API key enforcement on job routes, request size limits, trusted host/CORS defaults, and use of `hmac.compare_digest`).

## Findings

## High Severity

### H-001: Job API is vulnerable to resource exhaustion by default
**Evidence**
- `app.py:95` (`RATE_LIMIT_PER_MINUTE` defaults to `0`)
- `app.py:401-402` (rate limiting disabled when value is `<= 0`)
- `app.py:1268-1287` (each accepted job immediately spawns a subprocess task, no concurrency ceiling)

**Why this matters**
An authenticated client can enqueue unlimited expensive jobs and exhaust CPU/RAM/process limits, causing denial of service.

**Secure-by-default improvements**
1. Set a non-zero default rate limit (for example, `TP_RATE_LIMIT_PER_MINUTE=60`).
2. Add `TP_MAX_CONCURRENT_JOBS` and queue overflow protection (`429` / `503` when saturated).
3. Add per-API-key quotas and global process budget enforcement.
4. Add tests for abuse scenarios (burst job creation, long-running process floods).

---

### H-002: `/v1/jobs` accepts unrestricted filesystem paths from request payloads
**Evidence**
- `app.py:684-687` (`_path_arg` returns arbitrary path strings)
- `app.py:703-760` (archive command arguments consume those paths without root allowlisting)
- `app.py:981-989` (`input_dir` and `output_dir` copied directly into command args)

**Why this matters**
With valid API credentials, a caller can direct the pipeline to read/write arbitrary locations accessible to the service account. This increases blast radius for credential compromise and weakens least-privilege boundaries.

**Secure-by-default improvements**
1. Introduce explicit allowed roots (`TP_ALLOWED_INPUT_ROOTS`, `TP_ALLOWED_OUTPUT_ROOTS`).
2. Canonicalize and validate all user-provided paths against allowlisted roots before command construction.
3. Reject symlink escapes and parent traversal after path resolution.
4. Add contract tests that assert rejection of `/etc`, home-directory escapes, and symlink pivots.

## Medium Severity

### M-001: SQL uses interpolated `LIMIT` values instead of strict parameterization
**Evidence**
- `src/transformation_portal/metrics/ledger.py:327-329`
- `src/transformation_portal/metrics/comparator.py:237-243`

**Why this matters**
Current call sites mostly pass integers, but interpolation leaves a latent SQL-injection footgun if these functions are reused with less-trusted inputs.

**Secure-by-default improvements**
1. Force `limit` to `int` and clamp to a safe range (for example, `1..1000`).
2. Use `LIMIT ?` with SQLite parameter binding.
3. Add tests that verify non-integer values are rejected.

---

### M-002: Timestamp CLI allows insecure `http://` TSA endpoints
**Evidence**
- `tools/timestamp_merkle_signature.py:225-227` (`http` and `https` are both accepted)

**Why this matters**
Timestamp requests over plaintext HTTP are vulnerable to interception and tampering.

**Secure-by-default improvements**
1. Require `https://` by default.
2. Add explicit `--allow-insecure-http` escape hatch for local test-only workflows.
3. Emit a high-visibility warning when insecure mode is used.

---

### M-003: Model loading/downloading is not consistently revision-pinned and fail-closed
**Evidence**
- `src/transformation_portal/style_transfer/reference_encoder.py:83,87`
- `src/transformation_portal/style_transfer/ip_adapter.py:101-109`
- `src/transformation_portal/vlm/llava.py:126,145`
- `scripts/install_models_auto.py:246`
- `scripts/download_samples.py:147-148`

**Why this matters**
Unpinned model revisions and optional checksum verification increase supply-chain risk (silent model drift or compromised upstream artifacts).

**Secure-by-default improvements**
1. Require immutable model revisions (commit SHA pins) for all `from_pretrained`/snapshot pulls.
2. Centralize trusted model IDs + revisions + digests in a lock manifest.
3. Fail closed on missing checksum/digest in all downloader scripts.
4. Add CI enforcement to block unpinned model sources in runtime code paths.

## Low Severity

### L-001: Default CSP still allows inline script/style execution
**Evidence**
- `app.py:101-103` (`'unsafe-inline'` in `script-src` and `style-src`)

**Why this matters**
If an XSS vector is introduced elsewhere, inline allowances increase exploitability.

**Secure-by-default improvements**
1. Move to nonce/hash-based CSP for scripts/styles.
2. Self-host frontend assets where possible and remove inline exceptions.
3. Add CSP regression tests for production configuration.

## Positive Controls Observed

1. Job endpoints enforce API key auth by default (`app.py:90`, `app.py:1168-1178`).
2. Request body size protections exist at header and stream levels (`app.py:440-497`).
3. Constant-time API key comparison is used (`app.py:434`).
4. Security headers and TrustedHost middleware are configured (`app.py:1097`, `app.py:111-119`).

## Recommended Fix Order

1. H-001 (rate/concurrency guards for job execution)
2. H-002 (filesystem root restrictions for all job path arguments)
3. M-001 (parameterize and bound SQL `LIMIT`)
4. M-002 (HTTPS-only timestamp transport by default)
5. M-003 (model revision pinning + fail-closed artifact verification)
6. L-001 (CSP hardening)
