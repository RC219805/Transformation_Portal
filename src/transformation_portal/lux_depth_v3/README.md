# Lux Depth V3 - Orchestrated Depth + Enhancement Pipeline

Lux Depth V3 is a production-grade orchestrator for depth-aware image processing with optional AI-powered enhancement.

## Overview

The Lux Depth V3 pipeline provides:
- **Depth Estimation** using the canonical `da3` backend surface
- **PBR Map Generation** (normal, roughness, ambient occlusion)
- **Materials V3** surface-aware finishing
- **V2 Enhancement** (optional AI-powered refinement)
- **APEX Quality Tier** support for governed production and research workflows

## Quick Start

### Commercial Production (PBR-Only)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/commercial" \
  --quality-tier "apex" \
  --depth-backend "da3" \
  --model-key "da3-metric" \
  --depth-device "mps" \
  --pbr "on" \
  --enable-v2 "off" \
  --run-card-version "v2" \
  --output-bit-depth 16
```

### Commercial Production (With Enhancement)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/enhanced" \
  --quality-tier "apex" \
  --depth-backend "da3" \
  --model-key "da3-metric" \
  --depth-device "mps" \
  --pbr "on" \
  --materials-v3 "on" \
  --output-bit-depth 16
```

### Research Experiment

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/research" \
  --preset "depth-anything-v3.1-research-m4" \
  --model-key "da3-research" \
  --non-commercial-ok "true" \
  --quality-tier "apex" \
  --pbr "on"
```

## Key Concepts

### V2 Enhancement (Optional)

The V2 enhancement stage is **optional** and enabled by default for backward compatibility.

**To disable V2 enhancement:**
```bash
--enable-v2 "off"
```

**Why disable V2?**
- PBR-only workflows (depth + maps only)
- Enhancement script not available
- Custom post-processing pipelines
- Faster iteration during development

**V2 is independent:** All other pipeline features (depth, PBR, Materials V3) work without V2.

### Quality Tier vs Preset

**Use `--quality-tier` for most workflows:**
- `standard` - Fast/draft quality
- `premium` - Balanced quality
- `apex` - Maximum quality for production

**Use `--preset` for specialized scenarios:**
- Research model configurations
- Fine-tuned parameter combinations
- Non-commercial depth models
- Curated presets such as `premium`, `default`, or `depth-anything-v3.1-research-m4`
- Unmapped preset strings are preserved as metadata labels unless they match a real preset

### Model Selectors

- `da3-research` - explicit research DA3 selector; requires `--non-commercial-ok "true"`
- `da3` - deprecated compatibility alias for `da3-research`; do not use it in new commands
- `da3-metric` - Apache-2.0 DA3 selector for the current Lux V3 relative-depth surface
- `da3-base` / `da3-small` - registry-supported experimental selectors, hidden from public CLI help until smoke-tested

Programmatic typed presets preserve their historical model mappings:
`Preset.DEFAULT`, `Preset.ARCHITECTURAL_INTERIOR`, and
`Preset.LUXURY_ESTATE` select `da3-research` and require
`non_commercial_ok=True`; `Preset.ARCHITECTURAL_EXTERIOR` selects
`da3-base`. Omitting the typed preset and all model selectors remains the
commercial-safe `da3-metric` default.

**Recommendation:** Start with `--quality-tier`, add `--preset` only when needed.

## Common Workflows

### PBR-Only (No Enhancement)

**Use Case:** Generate depth and PBR maps for 3D workflows, game engines, or technical visualization.

```bash
lux-depth-v3 \
  --input-dir "./input" \
  --output-dir "./output/pbr_only" \
  --quality-tier "apex" \
  --pbr "on" \
  --enable-v2 "off" \
  --depth-device "mps"
```

**Outputs:**
- `depth/<input-key>_depth.png` - 16-bit depth map
- `depth/<input-key>_depth.npy` - Float32 depth array when requested
- `depth/<input-key>_depth_metadata.json` - Depth provenance and statistics
- `pbr/<input-key>_normal.png` - Normal map
- `pbr/<input-key>_roughness.png` - Roughness map
- `pbr/<input-key>_ao.png` - Ambient occlusion map
- `manifests/<input-key>_combined.json` - Per-image processing manifest
- `manifests/batch_<batch-id>.json` - Prepared-plan-bound batch manifest
- `manifests/execution_evidence_<batch-id>.json` - Detached completion record
- `run_card_<batch-id>.json` - Reproducibility card when enabled
- `run_card_<batch-id>.self.json` - Run-card self-integrity sidecar when enabled

### Client Deliverable (APEX)

**Use Case:** Maximum quality output for client deliverables and final production.

```bash
lux-depth-v3 \
  --input-dir "./input" \
  --output-dir "./output/client" \
  --quality-tier "apex" \
  --depth-device "cuda" \
  --pbr "on" \
  --materials-v3 "on" \
  --enable-v2 "off" \
  --cache-depth "on" \
  --run-card-version "v2" \
  --output-bit-depth 16
```

**Outputs:**
- `depth/<input-key>_depth.png` - 16-bit depth map
- `depth/<input-key>_depth.npy` - Float32 depth array when requested
- `depth/<input-key>_depth_metadata.json` - Depth provenance and statistics
- All PBR maps listed above
- Enhanced-image paths reported by the batch result and combined manifest when
  the configured stages retain them
- `manifests/<input-key>_combined.json` - Unconditional processing manifest
- `manifests/batch_<batch-id>.json` - Prepared-plan-bound batch manifest
- `manifests/execution_evidence_<batch-id>.json` - Detached completion record
- `run_card_<batch-id>.json` - Reproducibility card when enabled
- `run_card_<batch-id>.self.json` - Run-card self-integrity sidecar when enabled

`<input-key>` contains the normalized source extension and a stable path hash.
Consume the paths returned by `enhance_batch` and recorded in the combined
manifest instead of reconstructing filenames from the source stem.

### Run Card Trust Layers

Use `--run-card-version "v2"` for production trust decisions. Run Card v2 replaces the legacy batch hash with a transparency-style CT Merkle artifact tree while keeping v1 verification support for historical bundles. Per-artifact inclusion proofs are now opt-in via `--run-card-include-proofs on` so large batches do not pay the proof-size and proof-build cost by default.

Common operator commands:

```bash
python scripts/verify_run_card_integrity.py ./output/client/run_card_batch.json --check-canonical-json

python tools/sign_run_card_attestation.py \
  --run-card ./output/client/run_card_batch.json \
  --format both \
  --key-id "release-signer"

python tools/verify_run_card_attestation.py \
  --run-card ./output/client/run_card_batch.json \
  --require-native \
  --require-dsse

python scripts/validation/assess_run_card_release.py \
  ./output/client/run_card_batch.json \
  --require-native-attestation \
  --require-dsse-attestation
```

`tools/sign_run_card_attestation.py` can also emit an optional Sigstore bundle sidecar when `cosign` is available locally. Offline verification of the run card, artifact tree, native detached attestation, and DSSE binding does not depend on Rekor or network access.

### Fast Iteration (Development)

**Use Case:** Quick validation during development.

```bash
lux-depth-v3 \
  --input-dir "./input" \
  --output-dir "./output/dev" \
  --quality-tier "standard" \
  --depth-device "cpu" \
  --pbr "off" \
  --enable-v2 "off"
```

## Input Discovery

The pipeline automatically excludes derived artifacts from input discovery to prevent nonsensical reprocessing:

### Excluded Artifacts

- **Depth maps:** `*_depth.png`, `*_depthpro_depth16.png`
- **PBR maps:** `*_normal.png`, `*_roughness.png`, `*_ao.png`
- **Output directories:** `depth/`, `pbr/`, `v2/`, `manifests/`, `logs/`
- **Hidden files:** `.DS_Store`, `.cache/`
- **Intermediate directories:** `_non_source/`

### Default Behavior

The pipeline silently excludes artifacts and logs a summary:

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output" \
  --quality-tier "standard"
```

**Output:**
```
INFO: Discovered 17 images, excluded 3 artifacts
```

### Validation Mode (Strict)

Use `--strict-inputs` to fail if artifacts are found (useful for CI/CD validation):

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output" \
  --strict-inputs
```

**Error (if artifacts found):**
```
ERROR: Strict mode: 3 excluded artifacts found in ./input_images
ERROR:   - image_depthpro_depth16.png (matched: _depthpro_depth16)
ERROR:   - output/depth/result.png (matched: /depth/)
```

**Why exclude artifacts?** Processing depth maps as RGB inputs creates nonsensical results (depth of depth), feedback loops, and data corruption.

**Full documentation:** [docs/guides/input_hygiene.md](../../../../docs/guides/input_hygiene.md)

## Troubleshooting

### "Script not found" Error

**Error:**
```
ERROR: V2 enhancement script not found: scripts/enhance_image.py
```

**Fix:** Add `--enable-v2 "off"` to your command.

**Why:** V2 is enabled by default and validates the enhancement script at startup. This is **correct fail-fast design** to prevent wasted processing. For PBR-only workflows, disable V2.

### RAW Inputs Require Optional RAW Support

**Error:**
```
RAW inputs detected but canonical RAW ingest is unavailable because rawpy is not installed.
```

**Fix:** Bootstrap the repo-local RAW runtime and let the pipeline auto-discover
`./.venv-raw/bin/python` for RAW batches:
```bash
./scripts/setup/install_raw_runtime.sh
```

You can also override the interpreter explicitly with:
```bash
--raw-python "./.venv-raw/bin/python"
```

**Why:** Canonical RAW ingest stays deterministic and fail-closed. RAW batches are rejected before dispatch when `rawpy` is unavailable so the pipeline does not start a run that cannot satisfy the ingest contract.

### More Help

- **Full Troubleshooting Guide:** [docs/guides/LUX_DEPTH_V3_TROUBLESHOOTING.md](../../../../docs/guides/LUX_DEPTH_V3_TROUBLESHOOTING.md)
- **CLI Reference:** [docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md](../../../../docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md)
- **CLI Help:** `lux-depth-v3 --help`

## Architecture

```
Input Images
    ↓
Depth Estimation (`da3` or `depth_pro`)
    ↓
PBR Generation (Normal, Roughness, AO) [Optional]
    ↓
Materials V3 (Surface-aware finishing) [Optional]
    ↓
V2 Enhancement (AI-powered refinement) [Optional]
    ↓
Output Deliverables
```

### Key Components

- **`__main__.py`** - CLI entry point with typer
- **`orchestrator.py`** - Main pipeline orchestration
- **`pbr_processor.py`** - PBR map generation
- **`materials_v3.py`** - Surface-aware finishing
- **`v2_runner.py`** - V2 enhancement execution
- **`config.py`** - Configuration and presets

## License Compliance

### Commercial-Safe (Default)

**DA3 (`da3` backend)**
- ✅ Commercial use allowed
- ✅ No license flags required
- Recommended for production

### Research-Only (Explicit Opt-In)

**Depth Anything V3.1** (CC BY-NC 4.0)
```bash
--preset "depth-anything-v3.1-research-m4" \
--model-key "da3-research" \
--non-commercial-ok "true"
```

**Apple Depth Pro** (AMLR Research License)
```bash
--depth-backend "depth_pro" \
--depth-pro-python "./.venv-depth-pro/bin/python" \
--non-commercial-ok "true" \
--accept-apple-depth-pro-research-license "true"
```

The CLI **enforces license compliance** at startup to prevent accidental violations.
For safe installation, bootstrap `depth-pro` with
`./scripts/setup/install_depth_pro_runtime.sh` and keep it in a dedicated NumPy 1.x environment
and point the main pipeline at it with `--depth-pro-python` or
`TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON`.
When explicit `depth_pro` runs are launched from a repo checkout, the pipeline
also auto-discovers the repo-local contract path `./.venv-depth-pro/bin/python`
when that interpreter exists.
Legacy aliases such as `depth_anything_v3` remain accepted as inputs, but emitted manifests, run cards, and logs normalize to `da3`.

## Performance Optimization

### GPU Acceleration

**NVIDIA CUDA:**
```bash
--depth-device "cuda"
```

**Apple Silicon (M1/M2/M3/M4):**
```bash
--depth-device "mps"
```

### Depth Caching

Enable content-addressable caching for faster iterations:
```bash
--cache-depth "on"
```

Cached depth maps are reused across runs, dramatically speeding up parameter exploration.

### Batch Processing

The pipeline automatically processes all images in `--input-dir` recursively. Supported formats:
- `.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.webp`, `.bmp`

### Performance Tuning (Advanced)

Control parallelism and resource usage:

```bash
# Limit CPU/I/O workers
--max-workers 4

# Limit GPU workers (MPS/CUDA)
--max-gpu-workers 2

# Enable strict image verification (CI/ingest validation)
--verify-images
```

**Default behavior:**
- GPU/MPS: 2 workers (VRAM-conservative)
- CPU: Auto-detect (CPU count - 1)

## Precision Guardrails

The pipeline maintains **16-bit precision** throughout the depth processing chain to prevent quality degradation.

### Design Principles

1. **Quantize only at write boundaries** - Internal processing uses float32/float64
2. **No intermediate uint8 conversions** - Prevents banding and precision loss
3. **Scale-aware filtering** - Bilateral filter adapts to depth value range
4. **Crop/pad over resample** - Dimension enforcement preserves pixel fidelity

### Implementation Details

**Bilateral Filter (v2.0+):**
- Processes float32 depth directly (no uint8 quantization)
- Auto-scales sigmaColor based on depth range
- Optional RGB-guided joint bilateral (cv2.ximgproc)

**Preprocessing:**
- Returns float32 RGB [0, 1] for inference
- Dimension enforcement via center crop + edge pad
- No quality-degrading resampling for 1-13px adjustments

**Write Path:**
- 16-bit PNG output maintains full dynamic range
- Optional float32 NPY for maximum precision PBR
- Quantization strategies: linear, percentile, adaptive

### Avoiding Common Pitfalls

❌ **Don't do this:**
```python
# Lossy uint8 round-trip
depth_u8 = (depth * 255).astype(np.uint8)
filtered = cv2.bilateralFilter(depth_u8, ...)
depth = filtered.astype(np.float32) / 255.0
# Result: 256 discrete levels, visible banding
```

✅ **Do this instead:**
```python
# Direct float32 processing
filtered = cv2.bilateralFilter(depth.astype(np.float32), ...)
# Result: Smooth gradients, full precision maintained
```

**For more details:** See `src/transformation_portal/lux_depth_v3/postprocessing.py` and `preprocessing.py`

## Output Structure

```
output_dir/
├── depth/
│   ├── <input-key>_depth.png # Depth visualization
│   ├── <input-key>_depth.npy # Optional float depth
│   └── <input-key>_depth_metadata.json # Depth provenance and statistics
├── pbr/                      # PBR maps (when --pbr on)
│   ├── <input-key>_normal.png
│   ├── <input-key>_roughness.png
│   └── <input-key>_ao.png
├── v2/                       # Enhanced images and V2 reports
├── temp/                     # Retained intermediates when requested
├── manifests/
│   ├── <input-key>_combined.json
│   ├── batch_<batch-id>.json
│   └── execution_evidence_<batch-id>.json # Detached completion record
├── run_card_<batch-id>.json  # Reproducibility card when requested
└── run_card_<batch-id>.self.json # Run-card self-integrity sidecar
```

## Python API

### Basic Usage

```python
from pathlib import Path
from transformation_portal.lux_depth_v3 import EnhanceConfig, EnhanceOrchestrator
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution

config = EnhanceConfig(
    quality_tier="apex",
    enable_v2=False,  # Disable V2 for PBR-only
    generate_pbr=True,
    depth_device="mps"
)

input_root = Path("./input_images").resolve()
input_files = sorted(input_root.glob("*.jpg")) + sorted(input_root.glob("*.png"))
prepared = prepare_lux_execution(config, input_root, input_files)
orchestrator = EnhanceOrchestrator.from_prepared(prepared, Path("./output"))

results = orchestrator.enhance_batch(
    prepared.input_root,
    input_files=list(prepared.input_files),
)
```

### PBR-Only Processing

```python
from pathlib import Path
from transformation_portal.lux_depth_v3 import PBRProcessor, get_preset

pbr_paths = PBRProcessor.from_cached_depth(
    depth_path=Path("output/depth/scene_depth.npy"),
    config=get_preset("premium").to_pbr_config(),
    output_dir=Path("output/pbr"),
    base_name="scene",
)

# pbr_paths contains: normal, roughness, ao
```

## Development

### Running Tests

```bash
# Core tests (fast)
pytest tests/lux_depth_v3/ -v -m "not ml and not slow"

# ML tests (requires ML dependencies)
pytest tests/lux_depth_v3/ -v -m "ml"
```

### Linting

```bash
# From repository root
flake8 src/transformation_portal/lux_depth_v3/
pylint src/transformation_portal/lux_depth_v3/
```

## Roadmap

### Near-Term Enhancements

**Uncertainty-Guided Refinement** (v2.1)
- Pixel-wise confidence estimation from depth backend
- Selective smoothing based on uncertainty masks
- QA tooling with uncertainty visualization
- Manifest integration for downstream quality gates

**Multi-View Consistency** (v2.2)
- Detect view groups via EXIF + similarity embeddings
- Cross-view depth consistency enforcement
- Optional refinement stage for real estate multi-shot workflows
- Artifact tiering (original + refined depth)

### Research-Track Features

**Planar Priors for Architectural Scenes** (v3.0)
- Plane segmentation for indoor/architectural content
- Depth snapping to detected planes (walls, floors, ceilings)
- Sharp occlusion boundaries, reduced "rubber wall" artifacts
- Plane masks for PBR/material processing

**Multi-Input Geometry Path** (v3.1+)
- First-class support for multi-view depth estimation
- Depth Anything 3 style spatially consistent geometry
- Integration with existing cacheable/manifest-driven architecture

### Quality of Life

- Preset editor/validator tool
- Live preview mode for parameter tuning
- Benchmark suite for regression detection
- Container/serverless deployment patterns

**Contributions welcome!** See [CONTRIBUTING.md](../../../../CONTRIBUTING.md)

## Additional Resources

- **Troubleshooting Guide:** [docs/guides/LUX_DEPTH_V3_TROUBLESHOOTING.md](../../../../docs/guides/LUX_DEPTH_V3_TROUBLESHOOTING.md)
- **CLI Guide:** [docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md](../../../../docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md)
- **Architecture:** [docs/architecture/ARCHITECTURE.md](../../../../docs/architecture/ARCHITECTURE.md)
- **Main README:** [README.md](../../../../README.md)
