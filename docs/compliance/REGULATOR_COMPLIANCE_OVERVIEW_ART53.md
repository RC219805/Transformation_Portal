# Regulator-Facing Compliance Overview

**Profile ID:** `EU-AI-ACT-ART53-GPAI-V1`
**Applies To:** General-Purpose AI model providers
**Version:** 1.0

---

## Executive Summary

This system provides a deterministic, cryptographically verifiable training-data traceability and disclosure mechanism aligned with Article 53 of Regulation (EU) 2024/1689.

It separates:

1. **Integrity Layer** — file-level hashing, Merkle binding, signature, timestamp, bundle root.
2. **Compliance Layer** — regulator-facing summary artifacts bound to the bundle root.

All public disclosures are cryptographically anchored to `bundle_root_sha256`.

---

## Evidence Architecture

### File-Level Integrity

- SHA-256 manifest of all corpus files

### Corpus Binding

- Merkle root over file digests

### Authenticity

- Detached signature of Merkle root

### Existence Proof

- RFC 3161 timestamp

### Bundle-Level Canonical Root

- Deterministic projection of manifest fields
- Cross-runtime parity enforced in CI
- Projection invariants frozen via ADR-035

---

## Regulatory Export Mode

The export subsystem:

- Validates bundle integrity
- Validates governed risk metadata
- Validates source taxonomy classification
- Emits deterministic JSON and Markdown
- Embeds verification instructions
- Enforces strict schema validation

The export artifact:

- Does not disclose file-level inventory
- Discloses structured source categories
- Discloses TDM compliance posture
- Discloses opt-out handling declarations
- Binds narrative to bundle root

---

## Copyright & TDM Disclosure

Export includes explicit declarations of:

- Whether TDM opt-out detection was performed
- Signal classes honored (for example `robots.txt`, HTTP headers)
- Whether removal process is documented
- Whether removal deltas affect bundle root

---

## Confidentiality Posture

The public export:

- Provides summary-level transparency
- Preserves trade secrets
- Enables targeted regulator verification
- Supports controlled disclosure under Article 78

---

## Verification Procedure

Regulators may verify integrity by running:

```bash
python tools/verify_evidence_bundle_manifest.py \
  --bundle-manifest evidence_bundle_manifest.json \
  --bundle-dir <bundle_dir>
```

```bash
python tools/regulatory_export.py \
  --bundle-manifest evidence_bundle_manifest.json \
  --risk-metadata risk_metadata.json \
  --source-taxonomy source_taxonomy.json \
  --out-json regulatory_export.json \
  --out-markdown regulatory_export.md
```

Expected exit code: `0`.
