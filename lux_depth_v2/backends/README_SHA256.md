# Model Cache - SHA256 Verification

## Overview

The model cache system provides secure, verified downloads of ONNX models with SHA256 integrity checking. This prevents corrupted downloads and ensures reproducibility.

## Model Registry

Models are registered in `DEFAULT_MODELS` with:
- **url**: Download URL (typically HuggingFace)
- **sha256**: Expected SHA256 hash (None if not yet computed)
- **size_mb**: Approximate file size

## SHA256 Workflow

### Initial Setup (One-time)

When a model is first downloaded without a SHA256 hash:

1. **Download** the model (manually or via `get_model_path(..., auto_download=True)`)
2. **Compute SHA256** using the utility script:
   ```bash
   python scripts/utilities/compute_model_sha256.py weights/efficientsam/ --generate-registry
   ```
3. **Update registry** in `model_cache.py` with the computed hash
4. **Commit** the updated registry to version control

### Example

```bash
# Download model (if not already cached)
python -c "from lux_depth_v2.backends.model_cache import get_model_path; \
  get_model_path('efficientsam_ti_vit_s', auto_download=True)"

# Compute SHA256
python scripts/utilities/compute_model_sha256.py weights/efficientsam/efficientsam_ti_vit_s.onnx

# Output:
# ✓ efficientsam_ti_vit_s.onnx
#   SHA256: a1b2c3d4e5f6...
#   Size: 41.2 MB

# Update model_cache.py:
DEFAULT_MODELS = {
    "efficientsam_ti_vit_s": {
        "url": "https://...",
        "sha256": "a1b2c3d4e5f6...",  # ← Add computed hash here
        "size_mb": 41.2,
    },
}
```

### Verification on Download

Once SHA256 is registered:
- All subsequent downloads are **automatically verified**
- Download fails if hash mismatch (corrupted file or wrong version)
- No manual verification needed

## Security Benefits

1. **Integrity**: Detect corrupted downloads or man-in-the-middle attacks
2. **Reproducibility**: Ensure all users download identical model files
3. **Version pinning**: Detect unexpected model version changes
4. **Audit trail**: SHA256 hashes in version control provide tamper-proof record

## Testing

Model cache has comprehensive tests in `lux_depth_v2/tests/test_model_cache.py`:
- SHA256 computation correctness
- Download with verification
- Atomic file writes (temp + rename)
- Error handling and cleanup

Run tests:
```bash
pytest lux_depth_v2/tests/test_model_cache.py -v
```

## Troubleshooting

### "SHA256 verification disabled" warning

This warning appears when downloading a model with `sha256: None` in the registry. It's safe but means the download is not verified.

**Solution**: Compute SHA256 and update registry (see workflow above).

### "SHA256 mismatch" error

This error indicates:
- File was corrupted during download
- URL points to a different model version
- Model was updated upstream without registry update

**Solution**:
1. Delete cached file: `rm weights/efficientsam/<model>.onnx`
2. Re-download
3. If error persists, URL may have changed - verify upstream source

### HuggingFace authentication errors

Some HuggingFace models require authentication. If download fails:

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Download model
huggingface-cli download yunyangx/efficientvit-sam efficientsam_ti_s_encoder.onnx \
  --local-dir weights/efficientsam/

# Compute SHA256
python scripts/utilities/compute_model_sha256.py weights/efficientsam/
```

## References

- [Model Cache Implementation](model_cache.py)
- [SHA256 Utility Script](../../scripts/utilities/compute_model_sha256.py)
- [Test Suite](../tests/test_model_cache.py)
- [HuggingFace Model Hub](https://huggingface.co/yunyangx/efficientvit-sam)
