# HuggingFace Model Pinning - Phase 2 Spatial AI

**Status:** Experimental → Stable Transition
**Policy:** [ADR-024 - HuggingFace Model Revision Pinning](../architecture/decisions/ADR-024-hf-model-revisions.md)

This document tracks HuggingFace model revisions used in Phase 2 Spatial AI and documents the verification process for promoting experimental presets to stable.

---

## Policy Summary

Per ADR-024, all HuggingFace model revisions must be:
1. **Explicitly pinned** to a specific commit SHA
2. **Verified** for correctness and safety
3. **Documented** with verification date and methodology
4. **Tracked** in version control

Experimental presets may use placeholder revisions (`NEEDS_VERIFICATION_0000...`) until proper verification is completed.

---

## Phase 2.1: Segmentation (SAM2)

### Model: facebook/sam2-hiera-large

**Repository:** https://huggingface.co/facebook/sam2-hiera-large

**Current Status:** EXPERIMENTAL (Placeholder Revision)

**Revision:**
```python
# experimental preset (luxury_estate_experimental.yaml)
revision = "NEEDS_VERIFICATION_0000000000000000000000000000000000000000"
```

**Verification Required Before Stable:**

1. **Navigate to commits:**
   ```
   https://huggingface.co/facebook/sam2-hiera-large/commits/main
   ```

2. **Select stable commit:**
   - Avoid commits labeled "experimental" or "wip"
   - Prefer commits with release tags
   - Verify commit includes model weights (sam2.1_hiera_large.pt)

3. **Verification steps:**
   ```bash
   # Download specific revision
   git clone https://huggingface.co/facebook/sam2-hiera-large
   cd sam2-hiera-large
   git checkout <COMMIT_SHA>

   # Verify model file exists and has expected size
   ls -lh sam2.1_hiera_large.pt
   # Expected: ~900MB

   # Verify config matches expectations
   cat config.json

   # Test model loads correctly
   python scripts/test_sam2_revision.py --revision <COMMIT_SHA>
   ```

4. **Quality checks:**
   - Run segmentation on test suite
   - Compare outputs with known-good baseline
   - Verify no degradation in stability scores
   - Check memory usage is within bounds

5. **Document verification:**
   ```yaml
   # In luxury_estate_stable.yaml
   segmentation:
     backend: sam2
     model_id: facebook/sam2-hiera-large
     revision: "abc123..."  # Verified commit SHA
     verified_date: "2024-02-11"
     verified_by: "username"
     verification_method: "visual inspection + test suite"
   ```

**Test Script:** `scripts/validation/test_sam2_revision.py`
```python
"""Test SAM2 revision for quality and compatibility."""
import torch
from transformers import SamModel, SamProcessor

def test_revision(revision: str):
    """Test a specific SAM2 revision."""
    print(f"Testing revision: {revision}")

    # Load model
    model = SamModel.from_pretrained(
        "facebook/sam2-hiera-large",
        revision=revision,
    )
    processor = SamProcessor.from_pretrained(
        "facebook/sam2-hiera-large",
        revision=revision,
    )

    # Test inference
    from PIL import Image
    import numpy as np

    test_image = Image.fromarray(
        (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    )

    inputs = processor(test_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"✅ Model loads and runs successfully")
    print(f"Output keys: {outputs.keys()}")
    print(f"Memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return True

if __name__ == "__main__":
    import sys
    revision = sys.argv[1] if len(sys.argv) > 1 else "main"
    test_revision(revision)
```

---

## Phase 2.2: Materials (NVDIFFREC)

### Status: Not Yet Implemented

**Backend:** Heuristic (no HF dependency)

**Neural Backend (Planned):**
- Model: TBD (custom NVDIFFREC checkpoint or similar)
- Will require same verification process when implemented
- Document here when neural backend is added

---

## Phase 2.3: Reconstruction (3DGS)

### Model: graphdeco-inria/gaussian-splatting

**Repository:** https://github.com/graphdeco-inria/gaussian-splatting
*(Note: This is a GitHub repo, not HuggingFace. May not require HF revision pinning.)*

**Current Status:** EXPERIMENTAL

**License:** Inria research-only license
⚠️ **License Restriction:** May not be suitable for production use without Inria permission.

**Revision:**
```python
# If using HF mirror:
# revision = "NEEDS_VERIFICATION_0000000000000000000000000000000000000000"

# If using GitHub directly:
git_sha = "NEEDS_VERIFICATION"  # Pin to specific commit
```

**Verification Required:**

1. **Review license compatibility:**
   - Verify Inria license terms for commercial use
   - May need alternative (e.g., Nerfstudio, gsplat)

2. **Test reconstruction quality:**
   - Run on test dataset with known geometry
   - Verify convergence and visual quality
   - Check memory/performance characteristics

3. **Document alternative backends:**
   - Consider `nerfstudio` (Apache 2.0 license)
   - Consider `gsplat` (Apache 2.0 license)
   - Document trade-offs

**Alternative Models (Production-Friendly):**

| Model | License | Status |
|-------|---------|--------|
| nerfstudio/nerfacto | Apache 2.0 | Recommended |
| gsplat | Apache 2.0 | Evaluation |
| 3DGS (Inria) | Research-only | Experimental |

---

## Verification Workflow

### Step 1: Identify Model Revision

```bash
# For HuggingFace models
cd ~/hf_models
git clone https://huggingface.co/<model_id>
cd <model_id>
git log --oneline -20  # Find stable commit

# Note the commit SHA
```

### Step 2: Test Revision

```bash
# Run verification script
python scripts/validation/test_sam2_revision.py <COMMIT_SHA>

# Run full test suite with pinned revision
SPATIAL_AI_SAM2_REVISION=<COMMIT_SHA> pytest tests/spatial_ai/segmentation/ -v

# Check performance
SPATIAL_AI_SAM2_REVISION=<COMMIT_SHA> pytest tests/spatial_ai/test_phase2_performance.py::TestSAM2Performance -v
```

### Step 3: Visual Inspection

```bash
# Generate segmentation on test images
python examples/spatial_ai/segment_image.py \
    --image tests/fixtures/test_image.jpg \
    --revision <COMMIT_SHA> \
    --output /tmp/test_output

# Compare with baseline
diff /tmp/test_output baseline_outputs/
```

### Step 4: Document Verification

Update preset config:
```yaml
# config/spatial_ai/presets/luxury_estate_stable.yaml
segmentation:
  backend: sam2
  model_id: facebook/sam2-hiera-large
  revision: "abc123def456..."  # ← Verified commit SHA

  # Verification metadata
  verified:
    date: "2024-02-11"
    by: "username"
    method: "test_suite + visual_inspection"
    notes: "Tested on 100 luxury estate images, all passed quality checks"
```

### Step 5: Update Documentation

Add entry to verification ledger:
```markdown
## Verification Ledger

| Model | Revision | Date | Verifier | Method | Status |
|-------|----------|------|----------|--------|--------|
| sam2-hiera-large | abc123... | 2024-02-11 | username | test suite + visual | ✅ Verified |
```

---

## Automation (Planned)

Future improvements:
1. **Automated testing:** CI job to test new revisions
2. **Regression detection:** Compare outputs with baseline
3. **Performance tracking:** Benchmark new revisions
4. **Security scanning:** Check for malicious code in checkpoints

**Script:** `scripts/ci/verify_hf_revisions.sh`
```bash
#!/bin/bash
# Verify all HF revisions in config files

set -e

echo "Verifying HuggingFace model revisions..."

# Extract revisions from config files
REVISIONS=$(grep -r "revision:" config/spatial_ai/presets/*.yaml | grep -v "NEEDS_VERIFICATION")

# Test each revision
for revision in $REVISIONS; do
    echo "Testing $revision..."
    python scripts/validation/test_sam2_revision.py "$revision"
done

echo "✅ All revisions verified"
```

---

## Current Status Summary

| Phase | Model | Revision Status | Production Ready |
|-------|-------|----------------|------------------|
| 2.1 Segmentation | SAM2 | ⚠️ Experimental | No - needs verification |
| 2.2 Materials | Heuristic | ✅ N/A (no model) | Yes |
| 2.2 Materials | Neural | 🔄 Not implemented | No |
| 2.3 Reconstruction | 3DGS | ⚠️ Experimental + License | No - needs alternative |

**Blocking Issues for Production:**
1. SAM2 revision must be verified and pinned
2. 3DGS license incompatible - need alternative backend
3. Neural materials not yet implemented

**Recommended Actions:**
1. Pin SAM2 to latest stable release (verify outputs first)
2. Implement `nerfstudio` backend as 3DGS alternative (Apache 2.0)
3. Keep heuristic materials as production backend until neural tested

---

## References

- [ADR-024: HuggingFace Model Revision Pinning](../architecture/decisions/ADR-024-hf-model-revisions.md)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)
- [Model Card Best Practices](https://huggingface.co/docs/hub/model-cards)
- [Security Best Practices](https://huggingface.co/docs/hub/security)

---

**Last Updated:** 2024-02-11
**Next Review:** Before promoting to stable presets
