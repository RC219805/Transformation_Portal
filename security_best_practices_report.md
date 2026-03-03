# Security Best Practices Report

Date: 2026-03-03  
Repository: `Transformation_Portal`  
Scope: API (`app.py`), runtime modules (`src/`), and operational tooling (`scripts/`, `tools/`).

## Executive Summary

This review found **5 actionable findings**:
- **2 High**
- **2 Medium**
- **1 Low**

The highest-risk issues are:
1. RFC3161 timestamp handling accepts unsigned/untrusted trust chains in practice (structure checked, cryptographic trust not verified).
2. Multiple runtime and install paths still load remote HuggingFace artifacts without immutable revision pinning, while the lock manifest remains placeholder-based and strict enforcement is opt-in.

The API layer itself has good hardening progress (path root allowlists, request size limits, API-key enforcement, default rate limit and concurrency caps).

## Findings

## High Severity

### H-001: RFC3161 timestamp trust is not cryptographically verified end-to-end

**Evidence**
- `tools/timestamp_merkle_signature.py:160-184` parses DER structure + status only.
- `tools/timestamp_merkle_signature.py:268-277` accepts response and writes `.tsr` without CMS signature/certificate chain verification.
- `tools/timestamp_merkle_signature.py:225-227` allows both `http://` and `https://` TSA URLs.
- `tools/verify_evidence_bundle_manifest.py:84-86` later validates only hash equality of the `.tsr` blob, not token authenticity.

**Impact**
An attacker controlling network path or TSA endpoint can provide a forged or untrusted timestamp token that still passes local workflow checks, undermining notarization evidence quality.

**Secure-by-default improvements**
1. Require `https://` TSA URLs by default; keep explicit `--allow-insecure-http` only for local testing.
2. Verify RFC3161 CMS signature, TSA cert chain, EKU (`timeStamping`), message imprint, and nonce before accepting a token.
3. Pin trusted TSA cert(s) or root set in configuration for deterministic verification.
4. Add negative tests: wrong nonce, wrong imprint, expired/untrusted cert, bad CMS signature.

---

### H-002: Remote model supply chain is not fail-closed by default

**Evidence**
- Runtime loading without immutable revisions:
  - `src/transformation_portal/style_transfer/ip_adapter.py:101-108`
  - `src/transformation_portal/vlm/llava.py:126,145`
  - `src/transformation_portal/segmentation/clip_classifier.py:149-151`
- Setup/install flows download snapshots without revision pinning:
  - `scripts/install_models_auto.py:193,246,275`
- Model lock manifest entries are placeholders (not 40-char SHAs):
  - `config/model_lock_manifest.yaml:9-39`
- Strict lock mode is opt-in (`False` unless env set):
  - `src/transformation_portal/core/security/model_lock.py:66-70`

**Impact**
Model revisions can drift or be replaced upstream; compromised remote artifacts can be consumed without deterministic pinning, increasing supply-chain risk.

**Secure-by-default improvements**
1. Require revision SHA pins for every `from_pretrained` / `snapshot_download` call in production paths.
2. Replace placeholder lock entries with verified commit SHAs.
3. Enable strict model lock mode in CI and production (`TP_STRICT_MODEL_LOCK=1`) and fail builds/runtime on unpinned models.
4. Add CI lint rule that blocks new unpinned HuggingFace loads in `src/` and release scripts.

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

1. H-001 (RFC3161 trust verification + HTTPS enforcement)
2. H-002 (complete revision pinning + strict model lock enablement)
3. M-001 (mandatory sample checksums; fail-closed)
4. M-002 (centralized secure downloader policy)
5. L-001 (CSP nonce/hash hardening)
