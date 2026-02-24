# Compliance Schema Version Policy

This policy governs version evolution for Phase 3.6+ governance schemas:

- `docs/compliance/schemas/risk_assessment_report.schema.json`
- `docs/compliance/schemas/cybersecurity_audit_record.schema.json`
- `docs/compliance/schemas/admt_governance.schema.json`

## Version Field

Each governance schema MUST define:

- top-level `$schema_version` in `MAJOR.MINOR.PATCH` format
- top-level `additionalProperties: false`

Current required value: `$schema_version = 1.0.0`.

## Bump Rules

### MAJOR

Breaking structural changes (for example, removed fields, stricter required
sets, incompatible enum/domain changes) require:

- MAJOR bump to `$schema_version`
- new or updated ADR documenting the contract change
- update to regression tests and fixture payloads

### MINOR

Backward-compatible additive changes (optional fields only) require:

- MINOR bump to `$schema_version`
- updates to fixture payloads and validation tests

### PATCH

Non-structural clarifications (description text, comments, formatting-only)
require:

- PATCH bump to `$schema_version`
- no schema contract behavior changes

## CI Enforcement

`scripts/validation/validate_compliance_schemas.py` is the enforcement gate.
It MUST fail when governance schemas:

- omit `$schema_version`
- use non-semver version values
- drift from the pinned governance schema version
- omit top-level `additionalProperties: false`
