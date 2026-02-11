# HuggingFace Model Revision Pinning Guide

## Purpose

Phase 1.1 (Item 5) introduces a governance posture for pinning HuggingFace model revisions for reproducibility.

**Why this matters:**
- `revision: main` can change between runs → different model weights → different outputs
- Pinned commits ensure **provenance guarantees** and **reproducible research**
- Aligns with "Spatial AI Foundation" determinism requirements

---

## Current Status

Presets have two acceptable states:

1. **Stable/canary presets** may omit `revision` (loader defaults to `main`) until a verified commit hash is provided.
2. **Experimental presets** may use explicit placeholders to force manual verification before use.

```yaml
# Experimental-only placeholder (must be replaced before real use)
revision: "NEEDS_VERIFICATION_0000000000000000000000"
```

**⚠️ Placeholders are NOT valid commits** — they intentionally force manual verification.

---

## How to Verify and Update

### Step 1: Visit HuggingFace Model Repository

For DA3 1.1 Nested Giant Large:
```
https://huggingface.co/depth-anything/DA3-NESTED-GIANT-LARGE-1.1/commits/main
```

### Step 2: Find Latest Stable Commit

- Look for the most recent commit on the `main` branch
- Click on the commit to view its full SHA hash (40 hex characters)
- Example format: `a1b2c3d4e5f6789012345678901234567890abcd`

### Step 3: Update Preset Files

Replace `NEEDS_VERIFICATION_0000000000000000000000` with the actual commit hash in:

1. `config/presets/apex_research.yaml` (fallback section)
2. `config/presets/apex_research_canary.yaml` (model section)
3. `config/presets/experimental/apex_research_ultra.yaml` (ensemble model entry)

### Step 4: Document the Verification

Add a comment noting:
- Date of verification
- Commit hash source
- Who verified it

Example:
```yaml
# Verified 2026-02-11 from https://huggingface.co/depth-anything/DA3-NESTED-GIANT-LARGE-1.1/commits/main
# Latest stable commit as of verification date
revision: "a1b2c3d4e5f6789012345678901234567890abcd"
```

---

## Testing After Updates

After pinning to real commit hashes:

1. **Validate HuggingFace revisions policy:**
   ```bash
   python scripts/validation/validate_hf_revisions.py
   ```

2. **Run full test suite:**
   ```bash
   pytest tests/spatial_ai/ingest/test_linear_decoder.py -v
   ```

3. **Verify preset loading:**
   ```bash
   pytest tests/depth/backends/ -k preset -v
   ```

---

## Maintenance

**When to update commit hashes:**
- Model repository releases a new version with bug fixes
- Security patches to model weights
- Intentional upgrade to newer model variant

**Always:**
- Document the reason for changing the pinned commit
- Run regression tests after updating
- Update `CHANGELOG.md` with model version changes

---

## Rationale (from Phase 1.1 architectural review)

> "If this is 'Spatial AI Foundation,' reproducibility matters. `revision: main` is nondeterministic.
> Pin commit hashes. Otherwise provenance guarantees are incomplete."

This aligns with the repository's governance-first positioning and scientific rigor standards.

---

## See Also

- `docs/architecture/ADR-026-apex-research-ultra.md` (Spatial AI Foundation requirements)
- `scripts/validation/validate_hf_revisions.py` (CI enforcement script)
- Phase 1.1 implementation plan (session checkpoints)
