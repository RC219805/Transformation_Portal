# Release Notes — v3.5.0

**Release Date:** 2026-02-24
**Tag:** `v3.5.0`
**Scope:** Completion of deterministic integrity + compliance export architecture

---

## Overview

Version 3.5.0 introduces **Regulatory Export Mode**, a deterministic, schema-governed compliance layer built on top of the frozen Phase 3.4.1 bundle-root contract.

This release completes the transformation from engineering-grade integrity tooling to a capability-aligned, regulator-verifiable AI evidence infrastructure.

---

## What’s Included

### Phase 3.3 — Evidence Bundle

- Portable evidence bundle manifest
- Cross-artifact digest enforcement
- Strict schema validation

### Phase 3.4 — Bundle Root Anchoring

- Canonical bundle root (`bundle_root_sha256`)
- Projection freeze via ADR
- Notarization exclusion from preimage

### Phase 3.4.1 — Cross-Runtime Determinism Hardening

- 3.11 vs 3.12 parity CI gate
- Golden-root drift detection
- ADR-035: frozen anchoring invariants (`docs/architecture/ADR-035-bundle-root-anchoring-invariants.md`)

### Phase 3.5 — Regulatory Export Mode

- `tools/regulatory_export.py`
- LOCKED specification: `docs/compliance/SPEC-REGEXPORT-001.md`
- Article 53 compliance profile document
- Canonical export artifact names: `regulatory_export.json` and `regulatory_export.md`
- Strict schemas:
  - `risk_metadata.schema.json`
  - `source_taxonomy.schema.json`
- Deterministic JSON + Markdown export
- Root-bound disclosure artifacts
- Redaction-aware design
- CI-safe, deterministic output

---

## Security & Integrity

- No changes to core evidence bundle format
- No changes to root projection contract
- No regressions in determinism enforcement
- CodeQL scanning enabled in CI for this repository (as of tag `v3.5.0`)
- Cross-runtime root parity enforced in CI
- Exit-code contracts frozen and documented

---

## Determinism Statement

Regulatory export artifacts are deterministic for identical inputs in the same
runtime configuration (validated by regression tests). Cross-runtime parity
enforcement in CI currently covers the bundle-root contract (Python 3.11/3.12).
No nondeterministic timestamps or environment-derived fields are introduced.

---

## Compliance Capabilities

v3.5.0 enables:

- Regulator-usable public summary
- Explicit TDM/copyright disclosures
- Structured source taxonomy
- Cryptographic binding to bundle root
- Deterministic rendering
- Verification commands embedded in artifact
- Summary-default confidentiality posture aligned with Article 78 safeguards

---

## Backward Compatibility

- Fully backward-compatible with 3.3 and 3.4 artifacts
- Export layer is additive
- No changes to existing CLI contracts

---

## Maturity Level

This release establishes:

> Cryptographically anchored, runtime-stable, schema-governed compliance infrastructure.

Compliance Maturity Position:

- Cryptographic binding of compliance artifacts
- Obligation-to-control traceability
- Export-layer reproducibility
- Structured regulator-readable documentation

This release does not constitute:

- Legal certification
- Third-party audit attestation
- Notified body approval

Applicability of Article 53 obligations remains deployment-context dependent;
this release does not assert that this repository itself operates a
general-purpose AI training or inference service.
