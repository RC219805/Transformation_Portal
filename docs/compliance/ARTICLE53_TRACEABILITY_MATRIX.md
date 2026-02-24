# Annex-Level Traceability Matrix (Article 53 Alignment)

Below is a structured mapping of system controls to Article 53 obligations.

All values in the `SHA256 Binding Location` column use exact JSON Pointer
(`RFC 6901`) or JSONPath notation relative to the artifact named in each row.

---

## Article 53(1)(a) — Technical Documentation

| Obligation | Control | Artifact | Artifact Path | SHA256 Binding Location | Deterministic Binding |
| --- | --- | --- | --- | --- | --- |
| Training data traceability | File-level SHA-256 manifest | `hash_manifest.csv.gz` | `<bundle_dir>/hash_manifest.csv.gz` | `/hash_manifest_sha256` (in `evidence_bundle_manifest.json`) | Yes (`bundle_root_sha256` projection) |
| Dataset integrity | Merkle root binding | `merkle_roots.json` | `<bundle_dir>/merkle_roots.json` | `/roots_sha256` (in `evidence_bundle_manifest.json`) | Yes (`bundle_root_sha256` projection) |
| Authenticity | Detached signature | `merkle_roots.sig.json` | `<bundle_dir>/merkle_roots.sig.json` | `/signature_sha256` (in `evidence_bundle_manifest.json`) | Yes (`bundle_root_sha256` projection) |
| Time-of-existence | RFC 3161 timestamp | `merkle_roots.tsr` or `merkle_roots.sig.tsr` | `<bundle_dir>/*.tsr` | `/timestamp_sha256` (in `evidence_bundle_manifest.json`) | Yes (`bundle_root_sha256` projection) |
| Whole-bundle fingerprint | Canonical bundle root | `bundle_root_sha256` | `/bundle_root_sha256` (in `evidence_bundle_manifest.json`) | `/bundle_root_sha256` (in `evidence_bundle_manifest.json`) | Yes (frozen by `docs/architecture/ADR-035-bundle-root-anchoring-invariants.md`) |
| Determinism guarantee | Cross-runtime CI parity | 3.4.1 parity gate | `.github/workflows/determinism-gate.yml` | `/bundle_root_sha256` (in `evidence_bundle_manifest.json`) | Yes (3.11/3.12 parity) |
| Governance freeze | Projection and exit-code invariants | ADR-035 | `docs/architecture/ADR-035-bundle-root-anchoring-invariants.md` | `/bundle_root_sha256` (projection contract authority) | Yes (change-controlled) |

---

## Article 53(1)(c) — Copyright Compliance

| Requirement | Control | Artifact | Artifact Path | SHA256 Binding Location | Deterministic Binding |
| --- | --- | --- | --- | --- | --- |
| Copyright policy | Policy ID + version declaration | `risk_metadata.json` | `/content_rights/policy_id`, `/content_rights/policy_version` | `/artifact_digests/risk_metadata_sha256` (in `regulatory_export.json`) | Yes (digest-bound in export) |
| TDM opt-out compliance | Explicit detection + signal classes | `risk_metadata.json` | `/copyright_compliance/tdm_opt_out_detection`, `$.copyright_compliance.signals_supported[*]` | `/artifact_digests/risk_metadata_sha256` (in `regulatory_export.json`) | Yes (digest-bound in export) |
| Removal process | Structured declaration | `risk_metadata.json` | `/copyright_compliance/removal_process_documented` | `/artifact_digests/risk_metadata_sha256` (in `regulatory_export.json`) | Yes (digest-bound in export) |
| Root impact disclosure | Structured declaration | `risk_metadata.json` | `/copyright_compliance/removal_deltas_affect_root` | `/artifact_digests/risk_metadata_sha256` (in `regulatory_export.json`) | Yes (digest-bound in export) |

---

## Article 53(1)(d) — Public Training Summary

| Requirement | Control | Artifact | Artifact Path | SHA256 Binding Location | Deterministic Binding |
| --- | --- | --- | --- | --- | --- |
| High-level summary | Deterministic export JSON + Markdown | `regulatory_export.json`, `regulatory_export.md` | `/training_data_summary` (JSON), summary sections (Markdown) | `/bundle_binding/bundle_root_sha256`, `/artifact_digests/hash_manifest_sha256`, `/artifact_digests/hash_summary_sha256` (in `regulatory_export.json`) | Yes (fixed rendering contract) |
| Source categories | Governed source taxonomy schema | `source_taxonomy.json` | `$.sources[*].category` | `/artifact_digests/source_taxonomy_sha256` (in `regulatory_export.json`) | Yes (digest-bound in export) |
| Web-scrape disclosure | Required crawler/period/domains fields | `source_taxonomy.json` | `$.sources[*].crawler`, `$.sources[*].collection_period`, `$.sources[*].top_domains` | `/artifact_digests/source_taxonomy_sha256` (in `regulatory_export.json`) | Yes (digest-bound in export) |
| No file-level disclosure | Public-summary redaction semantics | `docs/compliance/SPEC-REGEXPORT-001.md` | Section 6 (Redaction and Disclosure Semantics) | `/bundle_binding/bundle_root_sha256` (in `regulatory_export.json`) | Yes (spec-frozen) |
| Binding to evidence | Bundle root anchoring | `regulatory_export.json` | `/bundle_binding/bundle_root_sha256` | `/bundle_binding/bundle_root_sha256` (in `regulatory_export.json`) | Yes (projection-stable root) |

---

## Article 78 — Confidentiality

| Obligation | Control | Artifact | Artifact Path | SHA256 Binding Location | Deterministic Binding |
| --- | --- | --- | --- | --- | --- |
| Trade secret protection | Summary-level disclosure by default | `regulatory_export.json`, `docs/compliance/SPEC-REGEXPORT-001.md` | Confidentiality + redaction sections | `/bundle_binding/bundle_root_sha256` (in `regulatory_export.json`) | Yes (fixed export semantics) |
| Competent authority review capability | Summary-default export with retained file-level evidence under applicable safeguards | `regulatory_export.json`, `evidence_bundle_manifest.json` | Verification command contract + manifest digest fields | `/bundle_binding/bundle_root_sha256` (in `regulatory_export.json`), `/hash_manifest_sha256`, `/hash_summary_sha256` (in `evidence_bundle_manifest.json`) | Yes (deterministic tooling + digest checks) |
| Targeted inspection capability | Root-bound artifact set with manifest verification | Bundle artifact set | Manifest + digest fields | `/bundle_root_sha256`, `/hash_manifest_sha256`, `/hash_summary_sha256` (in `evidence_bundle_manifest.json`) | Yes (cross-runtime parity gate) |

---

## Determinism & Change Control

| Control | Enforcement |
| --- | --- |
| Projection freeze | `docs/architecture/ADR-035-bundle-root-anchoring-invariants.md` |
| Cross-runtime parity | CI |
| Golden-root drift detection | CI |
| Export rendering freeze | `docs/compliance/SPEC-REGEXPORT-001.md` |
| Schema strictness | `additionalProperties: false` |
| Version bump discipline | LOCKED spec rules |

---

## Obligations Out of Scope

The following operational obligations are outside this repository context:

- Foundation-model training execution and orchestration
- Model weight publication/distribution controls
- Production inference service operation controls

This repository provides evidence integrity and disclosure-export capabilities.
Where operational obligations apply, they are satisfied in the deployment
system of record.

---

## Final Compliance Position

With v3.5.0, the system provides:

- Cryptographically anchored training corpus evidence
- Deterministic cross-runtime verification
- Structured compliance disclosures
- Explicit copyright posture
- Regulator-verifiable narrative artifacts
- Governance-frozen integrity invariants

These controls are aligned with Article 53-style expectations and are capable
of supporting compliant deployments where obligations apply. Applicability
depends on deployment context; this document does not assert that this
repository itself operates a general-purpose AI training or inference service.
