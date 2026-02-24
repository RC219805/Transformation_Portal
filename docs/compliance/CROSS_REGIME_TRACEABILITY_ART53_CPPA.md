# Cross-Regime Traceability Mapping: EU Article 53 and CPPA 2026

## Scope

This document maps the existing Phase 3.3-3.5 integrity/export controls and
the Phase 3.6 governance extension to:

- EU AI Act Article 53 transparency expectations, and
- California CPPA CCPA regulations effective on 2026-01-01.

## CPPA Transition Deadlines

- ADMT obligations apply for in-scope uses beginning 2026-01-01, with
  pre-2027 significant-decision transitional deadline no later than 2027-01-01.
- Privacy risk assessments are required before initiating in-scope processing;
  pre-2026 processing that continues after 2026 must be completed by
  2027-12-31.
- Cybersecurity audit first reporting deadlines begin 2028-2030 by tier,
  with five-year audit record retention requirements.

## Coverage Summary

### Already delivered in Phase 3.3-3.5

- Deterministic evidence manifest and digest binding.
- Canonical `bundle_root_sha256` anchoring and cross-runtime parity.
- Schema-governed regulatory export artifacts with deterministic JSON/Markdown.
- Governed risk metadata and source taxonomy inputs.

### Added in Phase 3.6

- `risk_assessment_report.schema.json` for structured CPPA risk assessment data.
- `cybersecurity_audit_record.schema.json` for audit governance record binding.
- `admt_governance.schema.json` for ADMT notice/opt-out governance declarations.
- Optional governance export path in `tools/regulatory_export.py`:
  `--governance-export`.
- CI schema validation gate for compliance schemas in lightweight PR checks.
- ADR-036 accountability invariants for additive governance layering.

## Architectural Constraint

Phase 3.6 is strictly additive. It does not alter:

- bundle root projection rules,
- Phase 3.5 export contracts,
- prior exit-code contracts.

All governance records are bound to the same bundle root so later CPPA program
artifacts remain verifiable against the original evidence substrate.
