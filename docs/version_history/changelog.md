# Changelog

## 2026-02-24

### Release: v3.5.0
**Type:** Documentation Milestone
**Runtime Impact:** None
**Integrity Contract Change:** None

- Documented and named the existing deterministic Regulatory Export Mode as the compliance-layer capstone on top of frozen Phase 3.4.1 integrity contracts (no runtime changes).
- Added formal release notes: `docs/version_history/RELEASE_NOTES_v3.5.0.md`.
- Added regulator overview and Annex-level traceability matrix:
  - `docs/compliance/REGULATOR_COMPLIANCE_OVERVIEW_ART53.md`
  - `docs/compliance/ARTICLE53_TRACEABILITY_MATRIX.md`

### Phase 3.6 — Accountability Governance Layer (CPPA)
**Type:** Feature / Governance Layer
**Runtime Impact:** Additive (optional export path + schema gate)
**Integrity Contract Change:** None (root projection unchanged)

- Added Phase 3.6 governance schemas:
  - `docs/compliance/schemas/risk_assessment_report.schema.json`
  - `docs/compliance/schemas/cybersecurity_audit_record.schema.json`
  - `docs/compliance/schemas/admt_governance.schema.json`
- Added ADR-036 accountability invariants:
  - `docs/architecture/ADR-036-accountability-governance-invariants.md`
- Added cross-regime traceability mapping (EU Article 53 + CPPA 2026):
  - `docs/compliance/CROSS_REGIME_TRACEABILITY_ART53_CPPA.md`
- Extended `tools/regulatory_export.py` with optional governance export:
  - `--governance-export`
- Added lightweight CI compliance-schema validation gate:
  - `scripts/validation/validate_compliance_schemas.py`

## 2026-01-29

**Security & Bug Fixes:**
- Fixed CVE-2026-0994 by bumping protobuf to 6.34.0 (baf69e04)
- Hardened workflow token permissions across all GitHub Actions workflows (baf69e04)
- Fixed duplicate `permissions:` block in quality-gate workflow (aa555e0a)
- Fixed InputMetadata positional args bug in lux_depth_v3 orchestrator (0fe68a41)
  - Resolved silent metadata corruption where `True` was incorrectly assigned to `image_size_bytes`
  - Changed to keyword-based construction for safer dataclass initialization

**Dependency Updates:**
- virtualenv: 20.35.4 → 20.36.1 (cc8b3bae)

**Known Issues:**
- Issue #761: quality-gate workflow contains local-only git commit (optional cleanup, non-urgent)

## 2025-10-03

- Enhanced `.github/copilot-instructions.md` with best practice sections following GitHub Copilot coding agent guidelines:
  - Added Repository Structure section with visual directory tree
  - Added Getting Started section with prerequisites and setup instructions
  - Added Troubleshooting section with common issues and solutions
  - Added Additional Resources section with links to internal documentation
  - Added Code Examples section with practical snippets for common tasks

## 2025-07-04
- Standardized README anchors and terminology for the tooling sections, including consistent tone-mapping language and nested table-of-contents links.

## 2025-07-03
- Reconciled README guidance so the table of contents, section anchors, and terminology match the merged tooling pull requests.

## 2025-07-02

- Added integrated comprehensive dataset for Picacho Lane project under Client Deliverables.
