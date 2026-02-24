# SPEC-REGEXPORT-001: Regulatory Export Mode

**Status:** LOCKED
**Version:** 1.0.0
**Date:** 2026-02-24
**Owner:** Transformation Portal Architect
**Scope:** Public/compliance export layer bound to Phase 3.4 bundle roots

---

## 1. Purpose

Define deterministic, regulator-facing export artifacts for Article 53-style
disclosure while preserving trade-secret controls and cryptographic verifiability.

---

## 2. Normative Requirements

The key words "MUST", "MUST NOT", "SHOULD", and "MAY" are normative.

1. Export output MUST be cryptographically bound to `bundle_root_sha256`.
2. Export inputs MUST include governed `risk_metadata.json` and
   `source_taxonomy.json`.
3. Input metadata MUST be strict-schema validated
   (`additionalProperties: false` contracts).
4. `hash_manifest.csv.gz` and `hash_summary.json` digests MUST match the
   digests declared in `evidence_bundle_manifest.json`.
5. Export rendering MUST be deterministic for identical inputs.

---

## 3. Required Inputs

1. `evidence_bundle_manifest.json` with Phase 3.4 root fields present.
2. `hash_manifest.csv.gz`.
3. `hash_summary.json`.
4. `risk_metadata.json` conforming to
   `docs/compliance/schemas/risk_metadata.schema.json`.
5. `source_taxonomy.json` conforming to
   `docs/compliance/schemas/source_taxonomy.schema.json`.

---

## 4. Output Contracts

### 4.1 JSON Export

`tools/regulatory_export.py` MUST emit JSON with:

1. `export_mode_version`
2. `compliance_profile_id`
3. `bundle_binding` (including `bundle_root_sha256`)
4. `artifact_digests`
5. `training_data_summary`
6. `content_rights`
7. `copyright_compliance`
8. `verification_commands`
9. confidentiality/legal statements

### 4.2 Markdown Export

When requested, Markdown output MUST include:

1. compliance profile and root binding
2. source category summary
3. top-N origin-drive and extension tables
4. web-scraped collection summary (crawler/domain/period if declared)
5. explicit verification commands
6. confidentiality and Article 78 notice

---

## 5. Canonical Rendering Rules

1. JSON MUST be UTF-8.
2. JSON MUST use `sort_keys=true`.
3. JSON MUST use two-space indentation and `(",", ": ")` separators.
4. JSON MUST end with a trailing LF.
5. Markdown section order MUST be fixed (no dynamic section insertion).
6. Markdown tables MUST be stably ordered by:
   - descending count
   - lexicographic key tie-breaker
7. Numeric rendering MUST be locale-independent.
8. Boolean rendering in Markdown MUST be lowercase `true`/`false`.
9. Top-list truncation MUST be deterministic and explicitly controlled by `--top-n`.

---

## 6. Redaction and Disclosure Semantics

1. Public export MUST NOT disclose file-level training inventory.
2. Public export MUST summarize source categories and high-level provenance
   instead of full corpus membership.
3. Integrity claims MUST remain independently verifiable from published digests.
4. Confidential details MAY be disclosed to competent authorities under Article 78.

---

## 7. Strict vs Public Modes

1. Strict validation is REQUIRED for all metadata and manifest contracts.
2. Public export mode is the default representation and contains only
   disclosure-safe aggregates.
3. Any future non-public mode MUST:
   - preserve bundle-root binding
   - remain schema-governed
   - be documented in a versioned spec update

---

## 8. CLI and Exit Codes

`tools/regulatory_export.py`:

1. `0`: export generated successfully
2. `31`: validation/build failure
3. `32`: write failure

---

## 9. Change Control

Because this spec is LOCKED:

1. Clarifications MAY be added without changing output semantics.
2. Any schema, rendering, binding, or digest contract change REQUIRES:
   - version increment of this spec
   - architecture review approval
   - CI/test updates that prove deterministic behavior
