# Changelog

## 2026-02-24

- Added Phase 3.6 governance schemas for `risk_assessment_report`, `cybersecurity_audit_record`, and `admt_governance`.
- Added ADR-036 accountability invariants for root-bound governance artifacts.
- Extended `tools/regulatory_export.py` with optional `--governance-export` packaging for CPPA governance records.
- Added lightweight CI compliance-schema validation gate for docs-only and full-code PR paths.

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

## 2025-07-02

- Added integrated comprehensive dataset for Picacho Lane project under Client Deliverables.

## 2025-07-03
- Reconciled README guidance so the table of contents, section anchors, and terminology match the merged tooling pull requests.

## 2025-07-04
- Standardized README anchors and terminology for the tooling sections, including consistent tone-mapping language and nested table-of-contents links.
