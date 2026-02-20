# Security Best Practices Report

Date: 2026-02-20
Repository: Transformation_Portal
Scope: Python application code (`src/`), operational scripts (`scripts/`), dependency/install flows, and GitHub Actions release workflow.

## Executive Summary

This review found 7 security findings: 2 High, 4 Medium, and 1 Low.

The highest-risk issues are:
1. Dynamic plugin loading that executes Python from user/environment-controlled paths.
2. `pickle.load()` deserialization in runtime code paths.

Most findings are secure-by-default hardening opportunities (supply chain integrity, trust boundaries, and least privilege) rather than confirmed in-the-wild exploitation.

## Findings

## High Severity

### H-001: Dynamic plugin discovery executes code from user/env-controlled paths
**Evidence**
- `src/transformation_portal/plugins/loader.py:149`
- `src/transformation_portal/plugins/loader.py:157`
- `src/transformation_portal/plugins/loader.py:415`
- `src/transformation_portal/plugins/registry.py:212`
- `src/transformation_portal/plugins/registry.py:220`
- `src/transformation_portal/plugins/registry.py:177`

**Why this matters**
Plugin discovery can load and execute arbitrary Python modules from `~/.transformation_portal/plugins` and `TRANSFORMATION_PORTAL_PLUGINS`. If these locations are writable by an attacker (or misconfigured), code execution occurs as soon as discovery is run.

**Secure-by-default improvements**
1. Default discovery to built-in plugins only; require explicit opt-in for user/env plugin paths.
2. Add signed plugin manifests (or hash allowlist) and refuse unsigned plugins by default.
3. Isolate plugin execution in a subprocess with reduced privileges.
4. Emit high-visibility audit logs when external plugin paths are enabled.

---

### H-002: Unsafe deserialization via `pickle.load()` in runtime paths
**Evidence**
- `src/transformation_portal/style_transfer/reference_encoder.py:243`
- `src/transformation_portal/depth/utils/cache.py:321`
- `src/transformation_portal/depth/utils/cache.py:368`

**Why this matters**
`pickle.load()` can execute attacker-controlled code during deserialization. Current comments assume files are self-generated, but this is not a cryptographic trust guarantee.

**Secure-by-default improvements**
1. Replace pickle artifacts with safer formats (`.npz`, JSON + typed arrays, or `safetensors` where applicable).
2. If legacy pickle must remain, require integrity verification (HMAC/signature) before load.
3. Restrict cache directories to owner-only permissions (`0700`) and reject symlinks/hardlinks.
4. Add explicit trust-boundary documentation: never load external/untrusted files.

## Medium Severity

### M-001: SQL query assembly interpolates `LIMIT` directly
**Evidence**
- `src/transformation_portal/metrics/ledger.py:327`
- `src/transformation_portal/metrics/ledger.py:329`
- `src/transformation_portal/metrics/comparator.py:237`
- `src/transformation_portal/metrics/comparator.py:242`

**Why this matters**
Current call sites pass integers, but APIs allow direct string interpolation into SQL for `LIMIT`. This creates future misuse risk if inputs expand to less-trusted sources.

**Secure-by-default improvements**
1. Enforce strict type/range validation for `limit` (`int`, `>=1`, upper bound).
2. Use parameterized `LIMIT ?` for SQLite.
3. Keep SQL fragments from fixed internal allowlists only.

---

### M-002: Model download paths do not consistently enforce checksum verification
**Evidence**
- `scripts/download_depth_models.py:91`
- `scripts/download_sam2_checkpoint.py:41`
- `scripts/install_models.py:72`
- `scripts/install_models.py:79`
- `scripts/install_models.py:190`
- `scripts/install_models.py:191`
- `src/transformation_portal/spatial_ai/segmentation/sam2_backend.py:979`

**Why this matters**
HTTPS and host allowlists reduce risk but do not guarantee artifact integrity. Without mandatory digest/signature checks, compromised upstream assets can be accepted.

**Secure-by-default improvements**
1. Require SHA-256 (or signature) for every downloaded artifact; fail closed if absent.
2. Keep checksum manifest under version control and rotate only via reviewed PRs.
3. Verify checksum before atomic replace and log expected/actual digest.
4. Prefer signed release provenance (e.g., Sigstore/TUF-style trust metadata).

---

### M-003: PyPI release workflow relies on long-lived API token secrets
**Evidence**
- `.github/workflows/submit-pypi.yml:95`
- `.github/workflows/submit-pypi.yml:96`
- `.github/workflows/submit-pypi.yml:137`
- `.github/workflows/submit-pypi.yml:138`

**Why this matters**
If CI secrets are exposed, attackers can publish compromised packages. Trusted Publishing (OIDC) removes persistent publish secrets and is the safer default.

**Secure-by-default improvements**
1. Migrate to PyPI Trusted Publishing (OIDC) and remove `PYPI_API_TOKEN`/`TEST_PYPI_API_TOKEN`.
2. Add `id-token: write` only for publish jobs.
3. Gate publish jobs with protected environments and required approvals.

---

### M-004: Security helper modules exist but are not enforced across command/path call sites
**Evidence**
- `src/transformation_portal/utils/security.py:46`
- `src/transformation_portal/utils/security.py:234`
- `src/transformation_portal/core/security/path.py:44`

No direct usage found in code search for these helpers.

**Why this matters**
Security controls implemented but not adopted create a false sense of coverage; high-risk call sites can bypass intended protections.

**Secure-by-default improvements**
1. Make security wrappers the default entry path for file and subprocess operations.
2. Add lint/check to block new raw `subprocess.run()`/path usage in sensitive modules.
3. Add targeted tests proving traversal/injection rejection behavior.

## Low Severity

### L-001: Atomic write helpers force world-readable file permissions (`0644`)
**Evidence**
- `src/transformation_portal/lux_depth_v3/io_atomic.py:79`
- `src/transformation_portal/lux_depth_v3/io_atomic.py:218`

**Why this matters**
Generated artifacts may be readable by other local users on shared systems. This is a confidentiality hardening gap.

**Secure-by-default improvements**
1. Default to `0600` and provide explicit opt-in for shared outputs.
2. Make output permission mode configurable via environment/config.
3. Document deployment guidance for multi-user hosts.

## Positive Controls Observed

1. YAML parsing predominantly uses `yaml.safe_load()` (good deserialization practice).
2. Several download paths enforce HTTPS and host allowlists.
3. Banned dependency constraints are present for known-risk packages (`basicsr`, `realesrgan`, etc.).
4. Workflows generally define `permissions:` blocks rather than relying on broad defaults.

## Recommended Fix Order

1. H-001 (plugin trust boundary hardening).
2. H-002 (`pickle` replacement or cryptographic integrity gate).
3. M-002 (mandatory artifact verification for downloads).
4. M-003 (PyPI Trusted Publishing migration).
5. M-001 and M-004 (defense-in-depth and consistency hardening).
6. L-001 (permission defaults).

## Notes and Assumptions

1. Some risky code paths may be CLI/admin-only; severities assume realistic shared-dev/CI environments.
2. SQL interpolation findings are currently low exploitability in observed call paths but are still best-practice violations worth fixing before interfaces expand.
