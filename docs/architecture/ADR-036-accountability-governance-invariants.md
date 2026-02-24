# ADR-036 Accountability Governance Invariants

## Status
Proposed

## Date
2026-02-24

## Executive Summary

Phase 3.6 introduces CPPA-oriented accountability artifacts as a governance
layer above the frozen Phase 3.4.1 integrity substrate and the Phase 3.5
regulatory export contract. This ADR freezes how governance records are
schema-validated, root-bound, and exported without changing existing bundle
root projection semantics.

## Context

Phase 3.3-3.5 already provides deterministic evidence bundling, canonical
bundle root anchoring, and Article 53 disclosure exports.

California CPPA regulations effective on 2026-01-01 introduce additional
programmatic obligations for:

- privacy risk assessment records,
- cybersecurity audit governance records,
- ADMT governance declarations where applicable.

These obligations are document- and retention-heavy and must be tied to the
same immutable evidence root to preserve reproducibility and audit defensibility.

## Decision

### D1. Governance Layer Is Additive

Phase 3.6 governance artifacts MUST be additive and MUST NOT modify:

- evidence bundle schema contracts,
- bundle root preimage projection,
- existing Phase 3.5 JSON/Markdown output contracts,
- existing exit-code contracts for prior phases.

### D2. Governance Schemas Are Strict

The following schemas are governance contracts and MUST remain strict
(`additionalProperties: false`):

- `docs/compliance/schemas/risk_assessment_report.schema.json`
- `docs/compliance/schemas/cybersecurity_audit_record.schema.json`
- `docs/compliance/schemas/admt_governance.schema.json`

### D3. Governance Exports Must Be Root-Bound

Any Phase 3.6 governance export MUST include:

- `bundle_root_sha256`,
- digest binding for each supplied governance artifact,
- deterministic serialization rules aligned with Phase 3.5 conventions
  (UTF-8, stable key ordering, trailing LF).

### D4. Date and Digest Field Discipline Is Frozen

Governance records MUST use:

- ISO-8601 calendar date strings (`YYYY-MM-DD`),
- lowercase 64-character SHA-256 digests for externally referenced reports.

### D5. CPPA Deadlines Are Governance Metadata, Not Root Inputs

Transition deadlines (for example, 2027 and 2028-2030 windows) MAY be included
in governance narrative/export fields, but MUST NOT be introduced into bundle
root projection inputs.

### D6. CI Must Enforce Schema Validity

CI MUST validate compliance schemas on every PR path, including docs-only
changes, so schema drift cannot bypass required checks.

## Alternatives Considered

- Embedding CPPA governance fields directly into Phase 3.5 export payload:
  rejected to avoid expanding the locked Article 53 contract.
- Storing governance assertions without schema controls:
  rejected because uncontrolled free-form metadata weakens audit traceability.
- Deferring governance validation to manual review only:
  rejected because deterministic compliance posture requires automated gates.

## Implementation Plan

- Add strict governance schemas under `docs/compliance/schemas/`.
- Extend `tools/regulatory_export.py` with an optional governance export mode
  that binds governance records to the existing bundle root.
- Add CI schema validation gate for compliance schemas in the lightweight PR
  path.
- Add regression tests for governance export behavior and failure modes.

## Success Metrics

- Governance schema changes fail CI when invalid.
- Governance exports are deterministic for identical inputs.
- Governance export digest binding remains stable and verifiable.
- Phase 3.5 outputs remain unchanged when governance mode is not requested.

## Enforcement

Enforcement occurs through:

- strict JSON schema files in `docs/compliance/schemas/`,
- runtime validation and deterministic serialization in
  `tools/regulatory_export.py`,
- compliance schema gate in `.github/workflows/build.yml`,
- regression tests in `tests/test_regulatory_export.py`.

## Consequences

Positive:

- CPPA governance artifacts become first-class, portable, and root-bound.
- Multi-year retention artifacts remain reproducible and verifiable.
- Existing integrity and export contracts remain stable.

Trade-off:

- Additional schema governance overhead and change-control discipline are
  required for compliance artifacts.

## References

- `docs/architecture/ADR-035-bundle-root-anchoring-invariants.md`
- `docs/compliance/SPEC-REGEXPORT-001.md`
- `docs/compliance/schemas/risk_assessment_report.schema.json`
- `docs/compliance/schemas/cybersecurity_audit_record.schema.json`
- `docs/compliance/schemas/admt_governance.schema.json`
