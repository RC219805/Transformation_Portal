# Compliance Docs

Phase 3.5 and Phase 3.6 compliance-layer artifacts live here.

## Primary Regulator Entry Point

Start here:

1. `REGULATOR_COMPLIANCE_OVERVIEW_ART53.md`
2. `ARTICLE53_TRACEABILITY_MATRIX.md`
3. `EU_AI_ACT_ART53_PROFILE.md`
4. Verification command contract in the overview document
5. Canonical export artifact names: `regulatory_export.json`, `regulatory_export.md`

## Files

### Phase 3.5 — Export + Article 53 Binder

1. `SPEC-REGEXPORT-001.md` - LOCKED deterministic export contract.
2. `EU_AI_ACT_ART53_PROFILE.md` - compliance profile for Article 53-style exports.
3. `REGULATOR_COMPLIANCE_OVERVIEW_ART53.md` - regulator-facing compliance overview.
4. `ARTICLE53_TRACEABILITY_MATRIX.md` - Annex-level traceability mapping for Article 53.
5. `schemas/risk_metadata.schema.json` - governed risk metadata schema.
6. `schemas/source_taxonomy.schema.json` - governed source taxonomy schema.

### Phase 3.6 — Accountability Governance Layer (CPPA)

7. `CROSS_REGIME_TRACEABILITY_ART53_CPPA.md` - cross-regime mapping (EU Article 53 + CPPA 2026).
8. `schemas/risk_assessment_report.schema.json` - CPPA-aligned privacy risk assessment governance schema.
9. `schemas/cybersecurity_audit_record.schema.json` - CPPA-aligned cybersecurity audit governance schema.
10. `schemas/admt_governance.schema.json` - CPPA ADMT governance declaration schema.

### Phase 3.7 — Governance Hardening

11. `SCHEMA_VERSION_POLICY.md` - schema version discipline (MAJOR/MINOR/PATCH) and CI enforcement requirements.
