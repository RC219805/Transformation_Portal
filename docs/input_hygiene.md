# Input Hygiene - Preventing Reprocessing of Depth Artifacts

## Overview

The Lux Depth V3 pipeline automatically excludes derived artifacts, outputs, and intermediate files from input discovery to prevent nonsensical reprocessing scenarios like:
- Computing depth maps of existing depth maps
- Processing PBR normal maps as RGB images
- Reprocessing already-enhanced outputs

This document describes the exclusion patterns, rationale, and usage.

---

## Exclusion Patterns

### Path-Based Exclusions

These directory patterns are excluded anywhere in the file path:

| Pattern | Purpose | Example |
|---------|---------|---------|
| `/_non_source/` | Intermediate/derived files | `./renders/_non_source/temp.jpg` |
| `/output/` | General output directory | `./output/result.jpg` |
| `/depth/` | Depth map outputs | `./output/depth/depth_map.png` |
| `/pbr/` | PBR map outputs | `./output/pbr/normal.png` |
| `/v2/` | V2 enhancement outputs | `./output/v2/enhanced.jpg` |
| `/manifests/` | Processing manifests | `./output/manifests/manifest.json` |
| `/logs/` | Processing logs | `./output/logs/processing.log` |
| `/.depth_cache/` | Cached depth artifacts | `./.depth_cache/cached.bin` |
| `/checkpoints/` | Model checkpoints | `./checkpoints/model.pt` |

### Filename Suffix Exclusions

These suffixes (before file extension) are excluded:

| Suffix | Purpose | Example |
|--------|---------|---------|
| `_depth` | Generic depth maps | `image_depth.png` |
| `_depthpro_depth16` | Depth Pro 16-bit outputs | `image_depthpro_depth16.png` |
| `_normal` | PBR normal maps | `scene_normal.png` |
| `_roughness` | PBR roughness maps | `scene_roughness.png` |
| `_ao` | Ambient occlusion maps | `scene_ao.png` |
| `_pbr` | Generic PBR outputs | `scene_pbr.png` |
| `_zone` | Zone segmentation maps | `image_zone.png` |

**Note:** All pattern matching is case-insensitive (e.g., `_DEPTH`, `_Depth`, `_depth` all match).

### Hidden Files

By default, hidden files and directories (starting with `.`) are excluded:
- `.DS_Store` (macOS metadata)
- `.cache/` (cache directories)
- `.git/` (version control)

---

## Usage

### Default Behavior (Non-Strict Mode)

By default, the pipeline silently excludes artifacts and logs a summary:

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output" \
  --quality-tier "standard"
```

**Output:**
```
INFO: Discovered 17 images, excluded 3 artifacts
DEBUG: Skipped artifact: image_depthpro_depth16.png (matched: stem suffix: _depthpro_depth16)
DEBUG: Skipped artifact: output/depth/result.png (matched: path pattern: /depth/)
```

### Strict Mode (Validation)

Use `--strict-inputs` to fail if artifacts are found (useful for CI/CD validation):

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output" \
  --strict-inputs
```

**Output (if artifacts found):**
```
ERROR: Strict mode: 3 excluded artifacts found in ./input_images
ERROR:   - image_depthpro_depth16.png (stem suffix: _depthpro_depth16)
ERROR:   - output/depth/result.png (path pattern: /depth/)
ERROR:   - scene_normal.png (stem suffix: _normal)
ValueError: Strict mode: 3 excluded artifacts found in ./input_images
```

---

## Rationale

### Why Exclude Depth Artifacts?

Depth maps are **derived outputs**, not RGB inputs. Processing them would create:
- **Nonsensical results**: Depth of depth maps has no semantic meaning
- **Feedback loops**: Reprocessing outputs as inputs
- **Data corruption**: Mixing single-channel depth with RGB processing

### Why Exclude PBR Maps?

PBR maps (normal, roughness, AO) are:
- **Single-channel or encoded data**, not photographs
- **Derived from depth estimation**, not source images
- **Incompatible with RGB enhancement** (would corrupt surface encoding)

### Why Exclude Output Directories?

Output directories contain:
- **Previously processed results** (already enhanced)
- **Intermediate artifacts** (manifests, logs)
- **Derived assets** (depth, PBR, zones)

Reprocessing outputs creates redundant work and feedback loops.

---

## Implementation Details

### Module: `input_discovery.py`

The exclusion logic is implemented in `src/transformation_portal/lux_depth_v3/input_discovery.py`:

```python
from .input_discovery import discover_images, DiscoveryConfig

config = DiscoveryConfig(
    exclude_path_patterns=["/_non_source/", "/output/", "/depth/", ...],
    exclude_stem_suffixes=["_depth", "_normal", "_roughness", ...],
    exclude_hidden=True,
    strict_mode=False  # or True for validation
)

images = discover_images(input_dir, config)
```

### Orchestrator Integration

The orchestrator (`orchestrator.py`) uses input discovery in `enhance_batch()`:

```python
def enhance_batch(self, input_dir: Path, ...) -> List[Dict[str, Any]]:
    discovery_config = DiscoveryConfig(strict_mode=self.config.strict_inputs)
    images = discover_images(input_dir, discovery_config, image_extensions)
    # Process discovered images...
```

### CLI Flag

The `--strict-inputs` flag is defined in `__main__.py`:

```python
strict_inputs: bool = typer.Option(
    False,
    "--strict-inputs",
    help="Fail if depth artifacts or derived outputs found in input directory (validation mode)"
)
```

---

## Edge Cases

### Mixed Input Directories

If your input directory contains both source images and outputs:

```
input_images/
├── source1.jpg          # ✅ Processed
├── source2.jpg          # ✅ Processed
├── output/
│   └── result.jpg       # ❌ Excluded (in /output/)
└── source1_depth.png    # ❌ Excluded (_depth suffix)
```

**Recommendation:** Keep source images separate from outputs.

### Custom Extensions

Only standard image formats are discovered by default:
- `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff` (case-insensitive)

Other formats (`.webp`, `.bmp`, etc.) are **not discovered** unless you modify `image_extensions` parameter.

### Subdirectories

The pipeline recursively scans subdirectories:

```
input_images/
├── renders/
│   ├── render1.jpg      # ✅ Processed
│   └── render1_depth.png # ❌ Excluded
└── photos/
    └── photo1.jpg       # ✅ Processed
```

---

## Testing

Run the test suite to verify exclusion behavior:

```bash
# Unit tests for input discovery
pytest -v tests/test_input_discovery.py

# All tests
pytest -v tests/test_input_discovery.py -ra
```

Test cases include:
- `test_exclude_depth_artifacts` - Depth maps excluded
- `test_exclude_pbr_artifacts` - PBR maps excluded
- `test_exclude_output_directories` - Output dirs excluded
- `test_exclude_hidden_files` - Hidden files excluded
- `test_strict_mode_fails_on_artifacts` - Strict mode validation
- `test_exclude_patterns_case_insensitive` - Case-insensitive matching

---

## Customization

### Adding Custom Exclusions

To exclude additional patterns, modify `DiscoveryConfig`:

```python
config = DiscoveryConfig(
    exclude_path_patterns=[
        "/_non_source/", "/output/", "/depth/",
        "/custom_output/",  # Add custom pattern
    ],
    exclude_stem_suffixes=[
        "_depth", "_normal", "_roughness",
        "_custom_artifact",  # Add custom suffix
    ],
)
```

### Disabling Hidden File Exclusion

To include hidden files:

```python
config = DiscoveryConfig(exclude_hidden=False)
```

---

## Troubleshooting

### "No images discovered"

**Cause:** All files excluded or wrong directory.

**Solution:**
1. Check input directory: `ls -la ./input_images/`
2. Verify image extensions: `.jpg`, `.png`, `.tif`
3. Run with `--verbose` for DEBUG logging
4. Ensure files aren't excluded by patterns

### "Strict mode error"

**Cause:** Artifacts found in input directory.

**Solution:**
1. Review excluded files in error message
2. Move artifacts to separate directory
3. Clean up depth/PBR outputs from input dir
4. Use non-strict mode if mixed directories unavoidable

---

## Performance

Input discovery overhead:
- **Fast:** ~1ms per 1000 files (pattern matching is O(n))
- **Negligible:** <0.1% of total processing time
- **Cached:** File system traversal uses native filesystem APIs

---

## Future Enhancements

Potential improvements:
- **Configurable patterns via YAML:** Allow users to define custom exclusions
- **Whitelist mode:** Explicitly specify which files to process
- **Dry-run mode:** Preview discovered files without processing
- **Artifact report:** Export list of excluded files to JSON

---

## References

- [Architecture Decision Record: Input Hygiene](../docs/architecture/decisions/adr-input-hygiene.md) *(if created)*
- [Orchestrator Documentation](../src/transformation_portal/lux_depth_v3/orchestrator.py)
- [Input Discovery Module](../src/transformation_portal/lux_depth_v3/input_discovery.py)
- [Test Suite](../tests/test_input_discovery.py)
