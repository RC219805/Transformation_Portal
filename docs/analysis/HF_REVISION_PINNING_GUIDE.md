# HuggingFace Model Revision Pinning Guide

**Purpose**: Pin HuggingFace model revisions to specific commit SHAs for reproducibility
**ADR**: ADR-032 (Dependency Pinning Strategy)
**Phase**: 1.1 Item 5

---

## Why Pin Revisions?

**Problem**: Using `main` branch or `null` revision means:
- Model behavior can change without code changes
- Non-reproducible results across time
- Difficult to debug when models update
- Violates ADR-032 strict pinning requirements

**Solution**: Pin to specific 40-character commit SHAs

---

## Current Status

### Production Presets (Must Be Pinned)
- ✅ `apex_research.yaml` - Uses fallback with TODO marker (acceptable for now)
- ✅ `apex_research_canary.yaml` - Uses fallback with TODO marker
- ✅ Other production presets - No HF dependencies

### Experimental Presets (Placeholders OK)
- ⚠️ `apex_research_ultra.yaml` - NEEDS_VERIFICATION placeholder
- ⚠️ `sam2_segmentation.yaml` - NEEDS_VERIFICATION placeholder
- ⚠️ `gaussian_splat_3d.yaml` - NEEDS_VERIFICATION placeholder
- ⚠️ `material_pbr.yaml` - NEEDS_VERIFICATION placeholder

---

## How to Pin Revisions

### Step 1: Find Model on HuggingFace

Example for DA3 1.1:
```
URL: https://huggingface.co/depth-anything/DA3-NESTED-GIANT-LARGE-1.1
```

### Step 2: Navigate to Commits

1. Click "Files and versions" tab
2. Click "Commits" sub-tab
3. Find the verified release commit

### Step 3: Copy Commit SHA

Look for 40-character hex string like:
```
2c21ea8f4d3c5b9a7e6f8d1a2b3c4d5e6f7a8b9c
```

**Important**: Use the **full 40-character SHA**, not abbreviated.

### Step 4: Update Preset YAML

Replace:
```yaml
revision: "NEEDS_VERIFICATION_0000000000000000000000"
```

With:
```yaml
revision: "2c21ea8f4d3c5b9a7e6f8d1a2b3c4d5e6f7a8b9c"  # Verified 2026-02-18
```

### Step 5: Validate

```bash
python scripts/validation/validate_hf_revisions.py
```

---

## Models Needing Verification

| Model | HuggingFace Repo | Status | Notes |
|-------|------------------|--------|-------|
| DA3 1.1 Nested Giant Large | `depth-anything/DA3-NESTED-GIANT-LARGE-1.1` | ⚠️ TODO | Fallback in apex_research |
| SAM2 Hiera Large | `facebook/sam2-hiera-large` | ⚠️ Placeholder | Experimental only |
| 3D Gaussian Splatting | `graphdeco-inria/gaussian-splatting` | ⚠️ Placeholder | Experimental only |
| MaterialGAN v2 | (Not on HF) | ❌ N/A | Research model |
| NVDIFFREC | (Not on HF) | ❌ N/A | Research model |

---

## Production Readiness Checklist

For a model to be promoted from experimental to production:

- [ ] HuggingFace revision pinned to commit SHA
- [ ] Checkpoint SHA256 verified and documented
- [ ] License compliance verified
- [ ] Integration tests passing
- [ ] Performance benchmarks recorded
- [ ] Documentation updated

---

## Validation Script

**Location**: `scripts/validation/validate_hf_revisions.py`

**Usage**:
```bash
# Check all presets
python scripts/validation/validate_hf_revisions.py

# Skip experimental presets
python scripts/validation/validate_hf_revisions.py --experimental-ok

# Get fix guidance
python scripts/validation/validate_hf_revisions.py --fix
```

---

## Best Practices

1. **Always verify commits manually** - Don't trust automated tools
2. **Document verification date** - Add comment with date
3. **Test after pinning** - Ensure model still loads
4. **Update CHANGELOG** - Record version changes
5. **Keep placeholders in experimental** - Until full verification

---

## References

- ADR-032: Dependency Pinning Strategy
- HuggingFace Hub API: https://huggingface.co/docs/hub/api
- DA3 GitHub: https://github.com/ByteDance-Seed/depth-anything-3
- Latest DA3 commit (as of 2026-02): `2c21ea8` (GitHub, not HF)

---

## Notes

**HuggingFace Commit SHAs vs GitHub Commit SHAs**:
- HuggingFace repos have their own commit history (for model uploads)
- This is separate from the source code GitHub repo
- Pin to HF commit SHA for model weights reproducibility
- Pin to GitHub commit SHA for code reproducibility

**When to Update Pins**:
- Major model releases (1.0 → 2.0)
- Critical bug fixes in model
- Security patches
- Never for minor tweaks (stay stable)
