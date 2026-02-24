# Regulator-Facing Compliance Overview

**Profile ID:** `EU-AI-ACT-ART53-GPAI-V1`
**Applies To:** Systems requiring Article 53-style traceability and disclosure controls (deployment-context dependent)
**Version:** 1.0

---

## Executive Summary

This system provides a deterministic, cryptographically verifiable training-data traceability and disclosure mechanism aligned with Article 53-style expectations under Regulation (EU) 2024/1689.

It separates:

1. **Integrity Layer** — file-level hashing, Merkle binding, signature, timestamp, bundle root.
2. **Compliance Layer** — regulator-facing summary artifacts bound to the bundle root.

All public disclosures are cryptographically anchored to `bundle_root_sha256`.

---

## Positioning

This document defines a capability alignment profile for deterministic evidence
integrity and disclosure exports.
Applicability of Article 53 obligations depends on deployment context and
operator role.
The controls in this repository can support compliant deployments where those
obligations apply.
This document does not assert that this repository operates a foundation-model
training service.
This document does not assert that this repository operates a production
inference service.

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
- Projection invariants frozen via `docs/architecture/ADR-035-bundle-root-anchoring-invariants.md`

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

## Control Taxonomy

Integrity Controls:

- SHA-256 artifact hashing
- Merkle bundle root construction
- Detached signature verification
- RFC 3161 timestamp anchoring

Traceability Controls:

- Phase version embedding in bundle metadata
- Contract version binding in manifests/specs
- Hash manifest totals and digest cross-checking

Transparency Controls:

- Deterministic training-data summary exports
- Structured source taxonomy disclosures
- Risk and copyright metadata declarations

Governance Controls:

- ADR-035 invariant freeze for bundle-root projection (`docs/architecture/ADR-035-bundle-root-anchoring-invariants.md`)
- LOCKED export rendering contract in `docs/compliance/SPEC-REGEXPORT-001.md`
- Strict schema controls (`additionalProperties: false`)

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

- Defaults to summary-level disclosure
- Preserves trade secrets
- Retains file-level evidence in bundle artifacts
- Supports competent-authority review under applicable confidentiality safeguards (including Article 78)

---

## Verification Procedure

Contract-stable verification example:

```bash
python tools/regulatory_export.py \
  --bundle-manifest <BUNDLE_DIR>/evidence_bundle_manifest.json \
  --risk-metadata <RISK_METADATA_JSON> \
  --source-taxonomy <SOURCE_TAXONOMY_JSON> \
  --out-json <OUTPUT_DIR>/regulatory_export.json \
  --out-markdown <OUTPUT_DIR>/regulatory_export.md
```

Run `python tools/regulatory_export.py --help` for full interface; the example
above is contractually stable per `docs/compliance/SPEC-REGEXPORT-001.md`.

Expected exit code: `0`.

---

## Final Compliance Position

With v3.5.0, this repository is aligned with Article 53-style traceability and
disclosure expectations and is capable of supporting compliant deployments where
obligations apply.
