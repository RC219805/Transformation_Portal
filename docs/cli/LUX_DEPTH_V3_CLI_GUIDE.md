# Lux Depth V3 CLI - APEX Command Variants

This document provides usage examples for the `lux-depth-v3` CLI with APEX quality tier support, including the current Apache-2.0 DA3 selector for the Lux V3 relative-depth surface and the research-only variants.

## Table of Contents
- [Installation](#installation)
- [Apache APEX Mode](#apache-apex-mode)
- [Research-Only APEX+ Variants](#research-only-apex-variants)
- [Command Options Reference](#command-options-reference)
- [Quality Tiers](#quality-tiers)
- [Output Deliverables](#output-deliverables)

## Installation

Use the repository-managed virtual environment and Make targets:

```bash
source .venv/bin/activate
make install-core
make check-environment
```

The `lux-depth-v3` command is installed into `.venv/bin/`. When the virtual
environment is active, invoke it directly:

```bash
lux-depth-v3 --help
```

When the virtual environment is not active, use the explicit path or module
entrypoint:

```bash
.venv/bin/lux-depth-v3 --help
.venv/bin/python -m transformation_portal.lux_depth_v3 --help
```

Optional model runtimes are installed through repo-governed scripts instead of
ad-hoc package installs:

```bash
make install-ml-core
./scripts/setup/install_da3_runtime.sh
./scripts/setup/install_depth_pro_runtime.sh
./scripts/setup/install_raw_runtime.sh
./scripts/setup/install_fastvlm_runtime.sh
make check-fastvlm-runtime
```

## Apache APEX Mode

For the current Lux V3 relative-depth surface, the Apache-2.0 APEX path uses `--model-key "da3-metric"` — also the out-of-the-box default when no `--model-key` is given (repair 1.2, #2066). The bare `da3` model selector is deprecated: it still resolves the research model and requires `--non-commercial-ok "true"`; use `--model-key "da3-research"` explicitly.

### Basic APEX Command

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/lux_depth_v3_apex" \
  --preset "premium" \
  --quality-tier "apex" \
  --depth-backend "da3" \
  --model-key "da3-metric" \
  --materials-v3 "on" \
  --pbr "on" \
  --cache-depth "on" \
  --output-bit-depth 16 \
  --emit-run-card "on" \
  --run-card-version "v2" \
  --overwrite
```

### APEX with GPU Acceleration (CUDA)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/apex_cuda" \
  --preset "premium" \
  --quality-tier "apex" \
  --model-key "da3-metric" \
  --depth-device "cuda" \
  --materials-v3 "on" \
  --pbr "on" \
  --output-bit-depth 16
```

### APEX with Apple Silicon (MPS)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/apex_mps" \
  --preset "premium" \
  --quality-tier "apex" \
  --model-key "da3-metric" \
  --depth-device "mps" \
  --materials-v3 "on" \
  --pbr "on" \
  --cache-depth "on"
```

## Research-Only APEX+ Variants

**Important**: Research-only variants use non-commercial models that require explicit license acknowledgement. Only use these if you comply with the respective license restrictions.

### Variant A: Depth Anything V3.1 (CC BY-NC 4.0)

Depth Anything V3.1 provides state-of-the-art depth estimation but is restricted to non-commercial use under CC BY-NC 4.0.

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/lux_depth_v3_apex_da31" \
  --preset "depth-anything-v3.1-research-m4" \
  --quality-tier "apex" \
  --model-key "da3-research" \
  --non-commercial-ok "true" \
  --depth-device "mps" \
  --materials-v3 "on" \
  --pbr "on" \
  --cache-depth "on" \
  --output-bit-depth 16 \
  --emit-run-card "on" \
  --run-card-version "v2" \
  --overwrite
```

**License**: CC BY-NC 4.0 (Non-Commercial)
**Use Cases**: Research, academic projects, non-commercial portfolio work

### Variant A2: Explicit DA3 Research Model

The deprecated `da3` model selector resolves to the research nested DA3 checkpoint and requires non-commercial acknowledgement; prefer the explicit `da3-research`. With no selector, the default is the Apache-2.0 `da3_metric` model.

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/lux_depth_v3_apex_da3_research" \
  --preset "premium" \
  --quality-tier "apex" \
  --depth-backend "da3" \
  --model-key "da3-research" \
  --non-commercial-ok "true" \
  --materials-v3 "on" \
  --pbr "on"
```

**License**: CC BY-NC 4.0 (Non-Commercial)
**Output Contract**: Current Lux V3 relative-depth surface

### Variant B: Apple Depth Pro (AMLR Research License)

Apple Depth Pro provides high-quality depth estimation but requires both non-commercial acknowledgement and explicit Apple license acceptance.

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/lux_depth_v3_apex_depthpro" \
  --preset "premium" \
  --quality-tier "apex" \
  --depth-backend "depth_pro" \
  --non-commercial-ok "true" \
  --accept-apple-depth-pro-research-license "true" \
  --depth-device "mps" \
  --materials-v3 "on" \
  --pbr "on" \
  --cache-depth "on" \
  --output-bit-depth 16 \
  --emit-run-card "on" \
  --run-card-version "v2" \
  --overwrite
```

**License**: AMLR (Apple Machine Learning Research)
**Use Cases**: Research only, requires explicit license acceptance
**Requirements**: Both `--non-commercial-ok` and `--accept-apple-depth-pro-research-license` must be set to `true`

## Command Options Reference

### Required Options

- `--input-dir PATH`: Input directory containing images to process (required)
- `--output-dir PATH`: Output directory for all artifacts (required)

### Pipeline Configuration

- `--preset TEXT`: Curated preset or metadata label (default: `premium`)
  - Curated examples: `premium`, `depth-anything-v3.1-research-m4`, `default`
  - Unmapped values are preserved as labels and do not become first-class typed presets
- `--quality-tier TEXT`: Quality tier (default: `standard`)
  - Options: `standard`, `premium`, `apex`

### Depth Backend Configuration

- `--depth-backend TEXT`: Depth estimation backend
  - Options: `da3` (default backend family), `depth_pro` (research-only)
  - Explicit `depth_pro` runs report `model_variant: "apple/ml-depth-pro"` in effective config, manifests, depth cache fingerprints, and run-card fingerprints.
- `--model-key TEXT`: Canonical model selector within the chosen backend family
  - Public options in this release: `da3`, `da3-research`, `da3-metric`
  - `da3-research`: Explicit research DA3 selector; requires `--non-commercial-ok "true"`
  - `da3`: Deprecated compatibility alias for `da3-research`; do not use it in new commands
  - `da3-metric`: Apache-2.0 DA3 selector for the current Lux V3 relative-depth surface
  - Python API typed presets keep their historical mappings: `Preset.DEFAULT`,
    `Preset.ARCHITECTURAL_INTERIOR`, and `Preset.LUXURY_ESTATE` select
    `da3-research` (requiring `non_commercial_ok=True`), while
    `Preset.ARCHITECTURAL_EXTERIOR` selects `da3-base`. With no typed preset
    or model selector, the default remains `da3-metric`.
- `--depth-device TEXT`: Device for depth inference (default: `cpu`)
  - Options: `cpu`, `cuda`, `mps`
- `--da3-python PATH`: Override the isolated DA3 subprocess interpreter.
- `--depth-pro-python PATH`: Override the isolated Depth Pro subprocess interpreter.
- `--raw-python PATH`: Override the isolated RAW subprocess interpreter.

### Feature Toggles

- `--materials-v3 TEXT`: Enable Materials V3 surface-aware finishing (default: `off`)
  - Options: `on`, `off`, `true`, `false`, `yes`, `no`, `1`, `0`
- `--pbr TEXT`: Enable PBR map generation (normal, roughness, AO) (default: `off`)
  - Options: Same as above
- `--save-float-depth TEXT`: Persist float depth assets for downstream PBR/DOF workflows.
- `--cache-depth TEXT`: Enable content-addressable depth cache (default: `off`)
  - Options: Same as above

### Segmentation, SAM2, and Advisory Captioning

- `--enable-segmentation TEXT`: Enable governed segmentation for Materials V3 masks.
- `--segmentation-backend TEXT`: Select `stub`, `efficientsam`, `sam2`, or `sam_vit_h`.
- `--strict-segmentation`: Fail closed on segmentation backend errors instead of falling back to `stub`.
- `--segmentation-cache TEXT`: Select segmentation cache policy (`off` or `read_write`).
- `--sam2-model-size TEXT`: Select SAM2 model size (`base` or `large`) when `--segmentation-backend sam2` is used.
- `--sam2-checkpoint-path PATH`: Override the SAM2 checkpoint path.
- `--sam2-tiling-enabled`: Enable deterministic SAM2 tiling for large images.
- `--sam2-tile-size-px INTEGER`, `--sam2-overlap-px INTEGER`, `--sam2-global-pass-longest-side INTEGER`, `--sam2-max-concurrency INTEGER`: Tune deterministic SAM2 tiled segmentation.
- `--sam2-points-per-side INTEGER`, `--sam2-points-per-batch INTEGER`, `--sam2-pred-iou-thresh FLOAT`, `--sam2-stability-score-thresh FLOAT`, `--sam2-crop-n-layers INTEGER`: Tune the SAM2 automatic mask generator.
- `--vlm-captioning TEXT`: Enable optional advisory FastVLM captioning.
- `--vlm-captioning-backend TEXT`: Select advisory captioning backend (`fastvlm`).
- `--vlm-captioning-model TEXT`: Select `default`, `review`, `smoke`, or an explicit local model path.
- `--vlm-captioning-proxy-format TEXT`: Select advisory proxy image format (`png` or `jpeg`).
- `--vlm-captioning-max-side-px INTEGER`: Bound the longest side for advisory proxy images.
- `--fastvlm-python PATH`: Override the isolated FastVLM subprocess interpreter.
- `--fastvlm-mlx-vlm-dir PATH`: Override the isolated `mlx-vlm` checkout.
- `--fastvlm-timeout-seconds INTEGER`: Bound FastVLM subprocess runtime.

FastVLM captioning is advisory only. Its output is not quality-gate evidence.

### V2 Enhancement Controls

- `--enable-v2 TEXT`: Enable V2 enhancement stage (default: `on`)
  - Options: `on`, `off`, `true`, `false`, `yes`, `no`, `1`, `0`
  - Set to `off` to completely skip V2 enhancement (no validation, no execution)
  - Useful for PBR-only workflows or when the enhancement script is not available
- `--v2-preset TEXT`: V2 enhancement preset (default: `default`)
  - Options: `default`, `none`, or custom preset names
  - Set to `none` to skip V2 processing while keeping validation
  - Only used when `--enable-v2` is `on`

### Output Deliverables

- `--output-bit-depth INTEGER`: Encode enhanced images as 8-bit PNG (default) or 16-bit TIFF
- `--emit-run-card TEXT`: Emit run card for reproducibility (default: `on`)
- `--run-card-version TEXT`: Run card contract version (default: `v1`)
  - Options: `v1`, `v2`
  - Use `v2` for production trust decisions and detached attestation workflows

`--emit-marketing` and `--emit-report` are deprecated compatibility
flags and will be removed in the next major release. Either value emits a
visible warning. No marketing artifact is produced, and the combined processing
manifest is always emitted.

`--emit-master16` and `--emit-upscaled16` are deprecated compatibility
aliases for `--output-bit-depth 16`. They emit one warning per invocation,
do not create separate deliverables, and will be removed in the next major release.

### License Acknowledgements

- `--non-commercial-ok TEXT`: Acknowledge non-commercial license restrictions (default: `false`)
  - Required for: `da3` / `da3-research`, Depth Anything V3.1, Depth Pro
- `--accept-apple-depth-pro-research-license TEXT`: Accept Apple Depth Pro research license (default: `false`)
  - Required for: Depth Pro backend

### Processing Flags

- `--overwrite`: Force reprocessing even if outputs exist
- `--keep-intermediates`: Preserve intermediate artifacts for audit/debugging
- `--force-depth`: Force depth recomputation (ignore cache)
- `--strict-inputs TEXT`: Fail closed on unsupported or invalid inputs.
- `--verify-images TEXT`: Verify image inputs before processing.
- `--allow-semantic-fallback TEXT`: Allow configured semantic fallback paths.
- `--max-workers INTEGER`: Bound CPU/I/O worker threads.
- `--max-gpu-workers INTEGER`: Bound GPU/MPS inference workers.

### RAW Ingest

- `--raw-ingest-mode TEXT`: Decode mode (`auto`, `force_rawpy`, `force_preview`).
- `--raw-wb-mode TEXT`: White-balance mode (`camera`).
- `--raw-demosaic TEXT`: rawpy demosaic algorithm name (default `AHD`).
  The CLI/orchestrator perform a syntactic check (must be a valid
  `rawpy.DemosaicAlgorithm` member name - uppercase letters, digits, and
  underscores; must start with a letter). The actual semantic check happens
  in the RAW decode subprocess, which fails closed with the list of members
  exposed by the installed LibRaw build (typical members: `AHD`, `AAHD`,
  `AMAZE`, `DCB`, `DHT`, `LINEAR`, `LMMSE`, `MODIFIED_AHD`, `PPG`, `VNG`;
  some builds also expose `AFD`, `VCD`, `VCD_MODIFIED_AHD`). Different
  algorithms produce different pixels - the choice is captured in the
  Phase II ingest fingerprint for reproducibility.

### Planning

- `--plan`: Resolver-only mode. Runs the same argument parsing, configuration
  and model/license resolution, input selection, and cross-field validation a
  real run performs, then prints the resolved invocation
  (`tp.lux.resolved_invocation.v1`) as canonical JSON and exits — without
  loading models, creating the output directory, or writing any files. A
  command that would fail validation or resolution fails `--plan` with the
  same error and exit code. The emitted plan includes the authoritative
  resolved model contract, the planned backend and candidate fallback chain
  (the backend actually executed at runtime is recorded in manifests, not in
  the plan), license acknowledgements, planned stages, requested artifacts,
  the selected input files, and the configuration fingerprint. With no model
  selector, `resolved_model.requested_selector` is `"default"` and
  `resolved_model.resolution_reason` names `da3_metric`; emitted run cards
  repeat those values as `model_contract.requested_model_selector` and
  `model_contract.resolution_reason`.

### Logging

- `--verbose` / `-v`: Enable verbose logging
- `--quiet` / `-q`: Suppress all output except errors
- `--log-level TEXT`: Set log level (DEBUG, INFO, WARNING, ERROR)

## Quality Tiers

### Standard
- Default quality level
- Optimized for speed and moderate quality
- Suitable for drafts and previews

### Premium
- High-quality processing
- Balanced speed and quality
- Suitable for professional work

### APEX
- Maximum quality processing
- Full orchestrator path with all features enabled
- Includes PBR map generation, Materials V3, and all deliverables
- Suitable for final production and client deliverables

#### APEX Gate Policy

APEX mode enforces fail-closed quality gates with two explicit recovery paths:

- **Depth fallback auto-upgrade.** When `quality_tier=apex` is selected, `depth_fallback` is auto-upgraded from the default `"fail"` to `"v2-auto"`. Flat-distribution scenes that fail both DA3 (`APEX_DEPTH_PLATEAU`) and DA2 (`APEX_DEPTH_SATURATION_LOW`) recover via the V2 stage with independent depth instead of failing the batch. The run card records the full attempt history.
  - **Opt out:** there is no standalone `--depth-fallback` flag in this CLI. `depth_fallback="apex-strict"` is a value consumed by config-loaded or programmatic flows that construct `EnhanceConfig` directly (for example via the Python API or a YAML preset). Set it there to suppress the auto-upgrade; the validator accepts the value, `EnhanceConfig` canonicalizes it to `"fail"`, and the apex run keeps fail-closed depth.
- **Materials V3 soft-passthrough on confidence-only blocks.** When masks are detected and every implemented pixel op is blocked solely by `below_confidence_threshold`, the strict gate emits the output without applying pixel ops and surfaces a non-fatal `APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE` warning instead of `APEX_MATERIALS_PIXEL_OPS_EMPTY`. The warning lands in the run card under `result_summary[].segmentation_status.pixel_ops_passthrough` and `.warnings`.
  - Mixed blocker sets (e.g. `missing_material_confidence`, `unsupported_confidence_score_type`, `below_coverage_threshold`) still fail closed.
  - Mask handoff, mask generation, confidence gating, and pixel-op execution are distinct states. Run cards expose `result_summary[].segmentation_status.materials_summary` with `masks_generated`, `mask_count`, `pixel_ops_applied`, `pixel_ops_applied_count`, `blocked_count`, and `passthrough_code`.
  - If SAM2 consumes at least 90% of a per-image runtime, masks exist, and Materials V3 applies zero pixel ops, the run card adds advisory `performance_warnings[]` code `APEX_MATERIALS_SEGMENTATION_DOMINATES_NO_PIXEL_OPS`. This is cost telemetry only; it does not fail the run or change enhancement behavior.
  - To carry the soft-pass through APEX promotion, derive the per-candidate evidence from each per-image manifest with `transformation_portal.evals.apex_evidence_bundle.derive_materials_v3_evidence_from_manifest`, dump it to JSON, and pass it via `--candidate-evidence materials_v3:<asset_id>=<path>`.

## Output Deliverables

When APEX mode is enabled with governed outputs, the following artifacts are generated:

### Depth Assets
- `*_depth.png`: 16-bit PNG depth map (quantized for compatibility)
- `*_depth.npy`: Float32 depth array (high-precision, used for PBR)

### PBR Maps (when `--pbr on`)
- `*_normal.png`: Normal map for lighting calculations
- `*_roughness.png`: Roughness map for material appearance
- `*_ao.png`: Ambient occlusion map for contact shadows

### Enhanced Images
- `*_v2_enhanced.png` / `*_materials_v3_enhanced.png`: 8-bit output
- `*_v2_enhanced.tif` / `*_materials_v3_enhanced.tif`: 16-bit output

### Metadata
- `*_combined.json`: Unconditional processing manifest with provenance
- `*_run_card.json`: Run card for reproducibility tracking (when `--emit-run-card on`)
- `*.attestation.native.json`: Repo-native detached attestation for a v2 run card (when signed)
- `*.attestation.dsse.json`: DSSE + in-toto detached attestation sidecar (when signed)
- `*.attestation.dsse.sigstore.bundle.json`: Optional Sigstore verification bundle (when signed with `cosign`)

V2 reports expose material handoff and actual V2 material adjustment separately under `enhancement_metadata`:

- `material_masks_supplied`: `true` when at least one normalized material mask entry reached V2.
- `material_masks_supplied_count`: count of normalized mask entries handed to V2.
- `v2_material_adjustments_applied`: `true` only when V2 applied at least one supported, non-empty material adjustment with material strength enabled.
- `materials_applied`: deprecated boolean compatibility alias for `v2_material_adjustments_applied`; do not interpret it as supplied mask keys.

For deterministic V2 TIFF output, ICC profiles are preserved when available, including TIFF tag `34675` fallback extraction. EXIF remains stripped; reports use `metadata_preservation_mode: "partial"` when ICC is preserved and EXIF is intentionally not written.

## Run Card Verification and Signing

Use the offline verifier for both v1 and v2 bundles:

```bash
.venv/bin/python scripts/verify_run_card_integrity.py ./path/to/run_card.json --check-canonical-json
```

For v2 run cards, detached attestation helpers are available:

```bash
.venv/bin/python tools/sign_run_card_attestation.py \
  --run-card ./path/to/run_card.json \
  --format both \
  --gpg \
  --gpg-key-id "<PRIMARY_GPG_FINGERPRINT>" \
  --key-id "<PRIMARY_GPG_FINGERPRINT>"

.venv/bin/python tools/verify_run_card_attestation.py \
  --run-card ./path/to/run_card.json \
  --require-native \
  --require-dsse \
  --gpg
```

For native clear-sign attestations, `--key-id` must resolve through the local
GPG keyring to exactly one primary fingerprint and must match the primary key
reported by the verified signature. Logical labels such as `release-signer`
that are not GPG-resolvable now fail closed. Verification also compares the
GPG-extracted cleartext with the exact canonical attestation preimage; a valid
signature over unrelated content is rejected.

When you need policy-based release gating instead of low-level integrity checks, use:

```bash
.venv/bin/python scripts/validation/assess_run_card_release.py \
  ./path/to/run_card.json \
  --require-v2 \
  --require-native-attestation \
  --require-dsse-attestation \
  --gpg
```

## Example Workflows

### Workflow 1: Draft Preview (Fast)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/draft" \
  --preset "default" \
  --quality-tier "standard" \
  --depth-device "cpu"
```

### Workflow 2: Client Deliverable (APEX Commercial)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/client_deliverable" \
  --preset "premium" \
  --quality-tier "apex" \
  --depth-device "cuda" \
  --model-key "da3-metric" \
  --materials-v3 "on" \
  --pbr "on" \
  --cache-depth "on" \
  --output-bit-depth 16 \
  --emit-run-card "on"
```

### Workflow 3: Research Experiment (APEX+ Non-Commercial)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/research" \
  --preset "depth-anything-v3.1-research-m4" \
  --quality-tier "apex" \
  --model-key "da3-research" \
  --non-commercial-ok "true" \
  --depth-device "cuda" \
  --materials-v3 "on" \
  --pbr "on"
```

### Workflow 4: PBR-Only (Skip V2 Enhancement)

For workflows that only need depth and PBR maps without AI-powered enhancement:

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/pbr_only" \
  --quality-tier "apex" \
  --depth-device "mps" \
  --model-key "da3-metric" \
  --pbr "on" \
  --enable-v2 "off" \
  --output-bit-depth 16
```

**Key Points:**
- `--enable-v2 off` completely disables the V2 enhancement stage
- Faster processing (skips enhancement script execution)
- Still produces high-quality depth maps and PBR outputs
- Useful for technical workflows requiring only geometric data

### Workflow 5: Commercial APEX with Quality-Tier Focus

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/commercial_apex" \
  --quality-tier "apex" \
  --depth-device "mps" \
  --model-key "da3-metric" \
  --materials-v3 "on" \
  --pbr "on" \
  --cache-depth "on" \
  --output-bit-depth 16
```

**Note:** Using `--quality-tier apex` automatically enables appropriate features for commercial production. The `--preset` flag is optional and provides additional fine-tuning.

## Understanding V2 Enhancement

The V2 enhancement stage is an **optional** AI-powered enhancement step that applies after depth estimation and PBR processing. It is controlled by two flags:

- `--enable-v2`: Master switch to enable/disable the entire V2 stage (default: `on`)
- `--v2-preset`: Preset configuration for V2 enhancement or `none` to skip (default: `default`)

### When V2 Is Enabled (Default Behavior)

When `--enable-v2 on` (the default), the pipeline:
1. Validates that the enhancement script exists at `scripts/enhance_image.py`
2. Executes the script for each processed image
3. Applies AI-powered refinements configured by `--v2-preset`

### How to Disable V2 Enhancement

There are **two ways** to disable V2 enhancement:

**Method 1: Disable the V2 Stage Entirely**
```bash
--enable-v2 "off"
```
This completely skips V2 validation and execution. Best for PBR-only workflows.

**Method 2: Set V2 Preset to None**
```bash
--v2-preset "none"
```
This keeps V2 enabled but with no preset applied (effectively a no-op).

### Quality Tier vs Preset

These flags serve **different purposes**:

**`--quality-tier`** (standard|premium|apex)
- Controls **output quality level** across the entire pipeline
- Affects processing resolution, precision, and deliverable formats
- Determines which features are enabled by default
- Examples: `standard` (fast/draft), `premium` (balanced), `apex` (maximum quality)

**`--preset`** (named configuration)
- Provides **named combinations** of parameters for specific scenarios
- Fine-tunes pipeline behavior for particular depth models or use cases
- Curated examples: `premium`, `depth-anything-v3.1-research-m4`, `default`
- Unmapped values are preserved as metadata labels unless they match a real preset
- Can override quality-tier defaults when specified

**Recommendation**: Start with `--quality-tier` for most workflows. Use `--preset` only when you need specific model configurations or research-only features.

## Troubleshooting

### "Script not found" Error

**Error Message:**
```
ERROR: V2 enhancement script not found: scripts/enhance_image.py
```

**Cause:** The V2 enhancement stage is enabled (default), but the placeholder script is missing or not executable.

**Solutions:**

1. **Disable V2 Enhancement** (recommended for PBR-only workflows):
   ```bash
   --enable-v2 "off"
   ```

2. **Set V2 Preset to None**:
   ```bash
   --v2-preset "none"
   ```

3. **Ensure Script Exists**: Verify `scripts/enhance_image.py` exists and is executable:
   ```bash
   ls -l scripts/enhance_image.py
   chmod +x scripts/enhance_image.py
   ```

**Why This Happens:** The pipeline validates all required scripts at startup when V2 is enabled. This is **correct fail-fast design** to prevent wasted processing time.

### "Input directory does not exist"
Ensure the `--input-dir` path is correct and the directory exists.

### "Depth Pro backend requires --non-commercial-ok true"
When using `--depth-backend depth_pro`, you must set `--non-commercial-ok true` and `--accept-apple-depth-pro-research-license true`.

### "No images found in [directory]"
The input directory must contain at least one supported image format:
- `.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.webp`

### Common Configuration Mistakes

**Mistake: Mixing quality-tier with incompatible presets**
```bash
# Avoid this - conflicts between tier and preset
--quality-tier "standard" --preset "depth-anything-v3.1-research-m4"
```
Solution: Let presets override quality-tier, or use quality-tier alone.

**Mistake: Forgetting to disable V2 for PBR-only workflows**
```bash
# Avoid this - V2 is enabled by default but you want PBR-only
--pbr "on" --quality-tier "apex"
```
Solution: Add `--enable-v2 "off"` when you only need depth and PBR outputs.

**Mistake: Using research models without license acknowledgement**
```bash
# Avoid this - explicit research selection without its required acknowledgement
--model-key "da3-research"
```
Solution: Keep the explicit `--model-key "da3-research"` selection and add
`--non-commercial-ok "true"` for the non-commercial model.

### Missing ML Dependencies
If you see warnings about missing torch, transformers, or coremltools:
```bash
make install-ml-core
./scripts/setup/install_da3_runtime.sh
```

## Additional Resources

- [Current CLI reference](CLI_REFERENCE.md)
- [PBR CLI testing guide](PBR_CLI_TESTING_GUIDE.md)
- [Lux Depth V3 troubleshooting](../guides/LUX_DEPTH_V3_TROUBLESHOOTING.md)
- [Architecture Decision Record: Depth Backend Unification](../architecture/ADR-019-depth-backend-unification.md)
- [Architecture Decision Record: RAW Ingest](../architecture/ADR-030-phase2-deterministic-raw-ingest.md)
