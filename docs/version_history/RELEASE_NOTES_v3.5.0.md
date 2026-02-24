# Release Notes — v3.5.0

**Release Date:** 2026-02-24
**Tag:** `v3.5.0`
**Scope:** Completion of deterministic integrity + compliance export architecture

---

## Overview

Version 3.5.0 introduces **Regulatory Export Mode**, a deterministic, schema-governed compliance layer built on top of the frozen Phase 3.4.1 bundle-root contract.

This release completes the transformation from engineering-grade integrity tooling to compliance-grade, regulator-verifiable AI evidence infrastructure.

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
- ADR-035: frozen anchoring invariants

### Phase 3.5 — Regulatory Export Mode

- `tools/regulatory_export.py`
- LOCKED specification: `SPEC-REGEXPORT-001.md`
- Article 53 compliance profile document
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
- CodeQL clean
- Cross-runtime root parity enforced in CI
- Exit-code contracts frozen and documented

---

## Determinism Statement

All regulatory export artifacts produced in v3.5.0 remain byte-for-byte stable
under identical inputs across supported Python versions (3.11 and 3.12).
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
- Confidentiality posture aligned with Article 78

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
