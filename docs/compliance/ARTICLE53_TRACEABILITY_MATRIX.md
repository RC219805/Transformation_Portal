# Annex-Level Traceability Matrix (Article 53 Alignment)

Below is a structured mapping of system controls to Article 53 obligations.

---

## Article 53(1)(a) — Technical Documentation

| Obligation | Implementation | Artifact |
| --- | --- | --- |
| Training data traceability | File-level SHA-256 manifest | `hash_manifest.csv.gz` |
| Dataset integrity | Merkle root binding | `merkle_roots.json` |
| Authenticity | Detached signature | `merkle_roots.sig.json` |
| Time-of-existence | RFC 3161 timestamp | `.tsr` |
| Whole-bundle fingerprint | Canonical bundle root | `bundle_root_sha256` |
| Determinism guarantee | Cross-runtime CI parity | 3.4.1 gate |
| Governance freeze | ADR-035 | Architecture docs |

---

## Article 53(1)(c) — Copyright Compliance

| Requirement | Implementation |
| --- | --- |
| Copyright policy | `content_rights.policy_id` |
| TDM opt-out compliance | `tdm_opt_out_detection` + signal list |
| Removal process | `removal_process_documented` |
| Root impact disclosure | `removal_deltas_affect_root` |

---

## Article 53(1)(d) — Public Training Summary

| Requirement | Implementation |
| --- | --- |
| High-level summary | Deterministic export JSON + Markdown |
| Source categories | `source_taxonomy.json` |
| Web-scrape disclosure | crawler + period + top domains |
| No file-level disclosure | Enforced by design |
| Binding to evidence | `bundle_root_sha256` |

---

## Article 78 — Confidentiality

| Obligation | Implementation |
| --- | --- |
| Trade secret protection | Summary-level disclosure only |
| Controlled regulator access | Verification commands provided |
| Targeted inspection capability | Root-bound artifacts |

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

## Final Compliance Position

With v3.5.0, the system provides:

- Cryptographically anchored training corpus
- Deterministic cross-runtime verification
- Structured compliance disclosures
- Explicit copyright posture
- Regulator-verifiable narrative artifacts
- Governance-frozen integrity invariants

This meets traceability and integrity expectations under Article 53 and establishes a defensible compliance architecture.
