# EU AI Act Article 53 Compliance Profile

**Profile ID:** `EU-AI-ACT-ART53-GPAI-V1`
**Status:** Proposed
**Date:** 2026-02-24
**Owner:** Transformation Portal Architect

---

## 1. Objective

Define the minimum artifact set and verification posture required to produce a
public, regulator-usable export summary without disclosing proprietary corpus
details.

---

## 2. Required Artifact Set

1. `evidence_bundle_manifest.json` (Phase 3.3/3.4)
2. `bundle_root_sha256` and root contract fields (Phase 3.4)
3. cross-runtime parity enforcement (Phase 3.4.1)
4. `risk_metadata.json`
5. `source_taxonomy.json`
6. regulatory export JSON (`tools/regulatory_export.py`)
7. optional regulatory export Markdown (`tools/regulatory_export.py --out-markdown`)

---

## 3. Binding Requirements

1. `bundle_root_sha256` MUST be present and verified.
2. `hash_manifest_sha256` and `hash_summary_sha256` MUST match bundle-manifest
   values.
3. `risk_metadata.json` and `source_taxonomy.json` digests MUST be embedded in
   the final export artifact.

---

## 4. Copyright/TDM Disclosure Requirements

The profile requires explicit declaration of:

1. whether TDM opt-out detection was performed
2. which signal classes were honored
3. whether removal process exists
4. whether removal deltas affect the current bundle root

---

## 5. Source Taxonomy Requirements

The profile requires category-level source classification including:

1. source category
2. provenance type
3. license class
4. synthetic flag
5. TDM compliance note

For `web_scraped` entries, require:

1. crawler identifier
2. collection period
3. top domains

---

## 6. Verification Commands

```bash
python tools/verify_evidence_bundle_manifest.py \
  --bundle-manifest /path/to/evidence_bundle_manifest.json \
  --bundle-dir /path/to/bundle_dir
```

```bash
python tools/regulatory_export.py \
  --bundle-manifest /path/to/evidence_bundle_manifest.json \
  --risk-metadata /path/to/risk_metadata.json \
  --source-taxonomy /path/to/source_taxonomy.json \
  --out-json /path/to/regulatory_export.json \
  --out-markdown /path/to/regulatory_export.md
```

Expected exit code for both commands: `0`.
