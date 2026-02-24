# Annex-Level Traceability Matrix (Article 53 Alignment)

Below is a structured mapping of system controls to Article 53 obligations.

---

## Article 53(1)(a) — Technical Documentation

| Obligation | Control | Artifact | Artifact Path | SHA256 Binding Location | Deterministic Binding |
| --- | --- | --- | --- | --- | --- |
| Training data traceability | File-level SHA-256 manifest | `hash_manifest.csv.gz` | Bundle artifact | `evidence_bundle_manifest.json.hash_manifest_sha256` | Yes (`bundle_root_sha256` projection) |
| Dataset integrity | Merkle root binding | `merkle_roots.json` | Bundle artifact | `evidence_bundle_manifest.json.roots_sha256` | Yes (`bundle_root_sha256` projection) |
| Authenticity | Detached signature | `merkle_roots.sig.json` | Bundle artifact | `evidence_bundle_manifest.json.signature_sha256` | Yes (`bundle_root_sha256` projection) |
| Time-of-existence | RFC 3161 timestamp | `.tsr` | Bundle artifact | `evidence_bundle_manifest.json.timestamp_sha256` | Yes (`bundle_root_sha256` projection) |
| Whole-bundle fingerprint | Canonical bundle root | `bundle_root_sha256` | Manifest root field | `evidence_bundle_manifest.json.bundle_root_sha256` | Yes (frozen by ADR-035) |
| Determinism guarantee | Cross-runtime CI parity | 3.4.1 parity gate | `.github/workflows/determinism-gate.yml` | Golden fixture + runtime parity checks | Yes (3.11/3.12 parity) |
| Governance freeze | Projection and exit-code invariants | ADR-035 | `docs/architecture/ADR-035-bundle-root-anchoring-invariants.md` | ADR-controlled contract | Yes (change-controlled) |

---

## Article 53(1)(c) — Copyright Compliance

| Requirement | Control | Artifact | Artifact Path | SHA256 Binding Location | Deterministic Binding |
| --- | --- | --- | --- | --- | --- |
| Copyright policy | Policy ID + version declaration | `risk_metadata.json` | `content_rights.*` | `regulatory_export.json.artifact_digests.risk_metadata_sha256` | Yes (digest-bound in export) |
| TDM opt-out compliance | Explicit detection + signal classes | `risk_metadata.json` | `copyright_compliance.tdm_opt_out_detection`, `signals_supported` | `regulatory_export.json.artifact_digests.risk_metadata_sha256` | Yes (digest-bound in export) |
| Removal process | Structured declaration | `risk_metadata.json` | `copyright_compliance.removal_process_documented` | `regulatory_export.json.artifact_digests.risk_metadata_sha256` | Yes (digest-bound in export) |
| Root impact disclosure | Structured declaration | `risk_metadata.json` | `copyright_compliance.removal_deltas_affect_root` | `regulatory_export.json.artifact_digests.risk_metadata_sha256` | Yes (digest-bound in export) |

---

## Article 53(1)(d) — Public Training Summary

| Requirement | Control | Artifact | Artifact Path | SHA256 Binding Location | Deterministic Binding |
| --- | --- | --- | --- | --- | --- |
| High-level summary | Deterministic export JSON + Markdown | `regulatory_export.json`, `regulatory_export.md` | Export output set | `bundle_binding.bundle_root_sha256` + export artifact digests | Yes (fixed rendering contract) |
| Source categories | Governed source taxonomy schema | `source_taxonomy.json` | `sources[].category` | `regulatory_export.json.artifact_digests.source_taxonomy_sha256` | Yes (digest-bound in export) |
| Web-scrape disclosure | Required crawler/period/domains fields | `source_taxonomy.json` | `sources[].crawler`, `collection_period`, `top_domains` | `regulatory_export.json.artifact_digests.source_taxonomy_sha256` | Yes (digest-bound in export) |
| No file-level disclosure | Public-summary redaction semantics | `SPEC-REGEXPORT-001.md` | Redaction/disclosure sections | LOCKED spec contract | Yes (spec-frozen) |
| Binding to evidence | Bundle root anchoring | `evidence_bundle_manifest.json` | `bundle_root_sha256` | `regulatory_export.json.bundle_binding.bundle_root_sha256` | Yes (projection-stable root) |

---

## Article 78 — Confidentiality

| Obligation | Control | Artifact | Artifact Path | SHA256 Binding Location | Deterministic Binding |
| --- | --- | --- | --- | --- | --- |
| Trade secret protection | Summary-level disclosure only | `regulatory_export.json`, `SPEC-REGEXPORT-001.md` | Confidentiality + redaction sections | Root-bound export structure | Yes (fixed export semantics) |
| Controlled regulator access | Explicit verification procedure | `REGULATOR_COMPLIANCE_OVERVIEW_ART53.md` | Verification command section | Verifies `evidence_bundle_manifest.json` + root bindings | Yes (commanded deterministic tooling) |
| Targeted inspection capability | Root-bound artifact set with manifest verification | Bundle artifact set | Manifest + digest fields | `bundle_root_sha256` and per-artifact SHA256 fields | Yes (cross-runtime parity gate) |

---

## Determinism & Change Control

| Control | Enforcement |
| --- | --- |
| Projection freeze | ADR-035 |
| Cross-runtime parity | CI |
| Golden-root drift detection | CI |
| Export rendering freeze | SPEC-REGEXPORT-001 |
| Schema strictness | `additionalProperties: false` |
| Version bump discipline | LOCKED spec rules |

---

## Obligations Out of Scope

The following obligations are out of scope for this repository context:

- Foundation-model training execution and orchestration
- Model weight publication/distribution controls
- Production inference service operation controls

This repository covers evidence integrity and compliance export controls for
traceability/disclosure artifacts; obligations tied to model training operations
must be satisfied in the corresponding training/inference system of record.

---

## Final Compliance Position

With v3.5.0, the system provides:

- Cryptographically anchored training corpus
- Deterministic cross-runtime verification
- Structured compliance disclosures
- Explicit copyright posture
- Regulator-verifiable narrative artifacts
- Governance-frozen integrity invariants

This meets traceability and integrity expectations under Article 53 and establishes a defensible compliance architecture.
