# Security Best Practices Report

Date: 2026-03-03
Repository: `Transformation_Portal`
Scope: API (`app.py`), runtime modules (`src/`), and operational tooling (`scripts/`, `tools/`).

## Executive Summary

This review found **3 actionable findings**:
- **0 High**
- **2 Medium**
- **1 Low**

High-severity items H-001 and H-002 are remediated as of 2026-03-03:
1. RFC3161 timestamp handling now enforces HTTPS by default and verifies responses cryptographically via `openssl ts -verify` against trusted CA inputs.
2. Runtime/install paths for CLIP, FLUX, and LLaVA now resolve immutable pinned revisions from the model-lock manifest, and placeholders were replaced with pinned SHAs.

The API layer itself has good hardening progress (path root allowlists, request size limits, API-key enforcement, default rate limit and concurrency caps).

## Findings

## High Severity (Remediated)

### H-001: RFC3161 timestamp trust is cryptographically verified end-to-end (Remediated 2026-03-03)

**Evidence**
- `tools/timestamp_merkle_signature.py` enforces `https://` TSA URLs by default, with explicit `--allow-insecure-http` for local test endpoints.
- `tools/timestamp_merkle_signature.py` performs OpenSSL-backed RFC3161 verification (`openssl ts -verify`) against the original query and a trusted CA file/path or system trust store.
- `tests/test_merkle_timestamp_cli.py` now covers cryptographic verification failure and redirect rejection behavior in addition to HTTPS gating.

**Impact**
Risk materially reduced for default flows. Remaining risk is operational: trust-store provisioning must stay controlled and `--allow-insecure-http` should remain test-only.

**Remediation delivered**
1. HTTPS-by-default TSA transport policy.
2. Cryptographic verification of TSA responses prior to writing detached `.tsr`.
3. Negative tests for invalid cryptographic responses.

---

### H-002: Remote model supply chain controls are substantially hardened for targeted loaders (Remediated 2026-03-03)

**Evidence**
- Runtime loaders now resolve and apply pinned revisions for:
  - `src/transformation_portal/style_transfer/ip_adapter.py`
  - `src/transformation_portal/vlm/llava.py`
  - `src/transformation_portal/segmentation/clip_classifier.py`
- Installer checks now resolve strict pinned revisions before HuggingFace snapshot/model access:
  - `scripts/install_models_auto.py`
- Previously added CLIP/FLUX/LLaVA manifest placeholders are replaced with pinned 40-char SHAs:
  - `config/model_lock_manifest.yaml`

**Impact**
Supply-chain drift risk is materially reduced for covered model paths through immutable revision resolution, installer strict checks, and pinned manifest state.

**Remediation delivered**
1. Pinned revision enforcement in targeted runtime loaders and installer checks.
2. Placeholder lock entries for CLIP/FLUX/LLaVA replaced with immutable SHAs.
3. Existing CI controls continue to validate HuggingFace revision hygiene.

## Medium Severity

### M-001: Sample downloader bypasses integrity checks when checksum is absent

**Evidence**
- All current sample registry entries lack checksums:
  - `scripts/download_samples.py:63,71,82,90,101,109`
- Verification logic explicitly skips validation when checksum is missing:
  - `scripts/download_samples.py:145-148`

**Impact**
Downloaded sample artifacts are accepted without tamper detection when checksum metadata is missing (which is the current default state).

**Secure-by-default improvements**
1. Make `sha256` mandatory for every downloadable sample entry.
2. Fail closed when checksum is missing or malformed.
3. Optionally sign the sample manifest itself and verify signature before download.

---

### M-002: URL fetchers rely on `urlretrieve` without centralized scheme/host policy

**Evidence**
- `scripts/download_samples.py:171`
- `scripts/download_depth_models.py:120`
- `scripts/install_models.py:237`
- `scripts/download_sam2_checkpoint.py:83`

**Impact**
While most current URLs are HTTPS constants, the project lacks a shared, explicit allowlist policy (scheme/host) for download endpoints, increasing future misconfiguration risk.

**Secure-by-default improvements**
1. Centralize download policy helper: enforce `https` and optional allowed-host list.
2. Reject non-HTTPS URLs unless explicitly overridden with a test-only flag.
3. Reuse one hardened downloader utility across scripts.

## Low Severity

### L-001: Default CSP still permits inline scripts/styles

**Evidence**
- `app.py:177-180` includes `'unsafe-inline'` in `script-src` and `style-src`.

**Impact**
If any XSS path appears in the UI surface, inline allowances increase exploitability and reduce CSP value.

**Secure-by-default improvements**
1. Move to nonce/hash-based CSP for scripts and styles.
2. Reduce inline usage in `portal.html` and self-host assets where possible.
3. Add CSP regression checks in tests/CI for production config.

## Positive Controls Observed

1. API path arguments are validated against allowed roots (`app.py:133-141`, `app.py:172-174`, `app.py:1220-1222`).
2. Job API key auth is enforced by default (`app.py:160`, `app.py:1405-1419`).
3. Request size enforcement is present at both header and stream layers (`app.py:522-553`, `app.py:556-581`).
4. Job admission now includes default rate-limit and concurrency protections (`app.py:165-167`, `app.py:1521-1530`).

## Recommended Fix Order

1. M-001 (mandatory sample checksums; fail-closed)
2. M-002 (centralized secure downloader policy)
3. L-001 (CSP nonce/hash hardening)
