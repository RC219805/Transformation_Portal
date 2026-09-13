# Lux Depth V3 Troubleshooting Guide

Source-reviewed troubleshooting for the Lux Depth V3 orchestrator (2026-09-12).
Use [setup guidance](SETUP_GUIDE.md) for core and isolated-runtime boundaries.
A successful plan/readiness check is not a completed inference run or artifact validation.

## Table of Contents
- [Quick Diagnosis](#quick-diagnosis)
- [V2 Enhancement Issues](#v2-enhancement-issues)
- [Configuration Mistakes](#configuration-mistakes)
- [Flag Usage Guide](#flag-usage-guide)
- [Performance Issues](#performance-issues)
- [Input/Output Issues](#inputoutput-issues)
- [License and Research Models](#license-and-research-models)

---

## Quick Diagnosis

### Symptom: "Script not found" Error

**Error Message:**
```
ERROR: V2 enhancement script not found: scripts/enhance_image.py
```

**Quick Fix:**
```bash
# Add this flag to your command:
--enable-v2 "off"
```

**Why This Works:** The V2 enhancement stage is **optional** and enabled by default. Setting `--enable-v2 off` completely disables V2 validation and execution. This is the correct approach for PBR-only workflows that don't need AI-powered enhancement.

**Alternative Fix:**
```bash
# Keep V2 enabled but skip preset:
--v2-preset "none"
```

**See:** [V2 Enhancement Issues](#v2-enhancement-issues) for detailed explanation.

---

### Symptom: Confusion About quality-tier vs preset

**Question:** "Which flag should I use, `--quality-tier` or `--preset`?"

**Answer:**

**Use `--quality-tier` to select the quality policy:**
- Selects `standard`, `premium`, or `apex` policy
- Does not automatically enable PBR, Materials V3, segmentation, or captioning
- With Materials V3 enabled, `apex` requires explicit segmentation, a real backend, and `--strict-segmentation`

**Use `--preset` only when you need:**
- Specific depth model configurations (e.g., Depth Anything V3.1)
- Research-only model variants
- Fine-tuned parameter combinations for specialized scenarios

**Example - Simple Commercial Workflow (Recommended):**
```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output" \
  --quality-tier "apex" \
  --depth-device "mps" \
  --pbr "on" \
  --enable-v2 "off"
```

**Example - Research Workflow (Advanced):**
```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output" \
  --quality-tier "apex" \
  --preset "depth-anything-v3.1-research-m4" \
  --model-key "da3-research" \
  --non-commercial-ok "true"
```

---

## V2 Enhancement Issues

### Understanding V2 Enhancement

The V2 enhancement stage is an **optional** AI-powered refinement step that:
- Applies **after** depth estimation and PBR processing
- Uses the maintained subprocess entry point at `scripts/enhance_image.py`
- Is **enabled by default** for backward compatibility
- Can be completely disabled without affecting other pipeline features

### V2 Enhancement Design Philosophy

**Fail-Fast Validation Is Correct Design:**
- The orchestrator validates required scripts at startup (when V2 is enabled)
- This prevents wasted processing time on large batches
- If V2 is disabled, no validation occurs
- This is **intentional behavior**, not a bug

### When to Disable V2 Enhancement

**PBR-Only Workflows:**
```bash
--enable-v2 "off"
```
Use this when you only need depth maps and PBR outputs (normal, roughness, AO) without AI enhancement.

**Missing Enhancement Script:**
```bash
--enable-v2 "off"
```
Use this if `scripts/enhance_image.py` is not available or not yet implemented.

**Custom Post-Processing:**
```bash
--enable-v2 "off"
```
Use this if you plan to apply your own enhancement pipeline after Lux Depth V3 processing.

### Two Ways to Disable V2

**Method 1: Disable Entirely (Recommended for PBR-Only)**
```bash
--enable-v2 "off"
```
- Completely skips V2 stage
- No validation performed
- No script execution
- Faster processing
- Best for technical workflows

**Method 2: Set Preset to None**
```bash
--enable-v2 "on" --v2-preset "none"
```
- Resolves to a disabled V2 stage in the prepared execution plan
- Skips V2 script validation and execution, like `--enable-v2 off`
- Does not disable validation for other selected stages

### V2 Enhancement Workflow Integration

```
Input Images
    ↓
Depth Estimation (Depth Anything V3 or Depth Pro)
    ↓
PBR Generation (Normal, Roughness, AO)
    ↓
Materials V3 (Optional, surface-aware finishing)
    ↓
V2 Enhancement (Optional, AI-powered refinement) ← **This stage is optional**
    ↓
Output Deliverables
```

**Key Insight:** All stages before V2 are **independent** and fully functional. V2 is purely additive.

---

## Configuration Mistakes

### Mistake 1: Conflicting Quality-Tier and Preset

**Problem:**
```bash
# Review the combined model/default selection and independent quality policy
--quality-tier "standard" --preset "depth-anything-v3.1-research-m4"
```

**Solution:**
```bash
# ✅ Use quality-tier alone for most workflows
--quality-tier "apex"

# ✅ OR use preset alone for specialized configurations
--preset "depth-anything-v3.1-research-m4" --model-key "da3-research" --non-commercial-ok "true"
```

**Explanation:** Preset and quality tier are separate inputs. Typed presets resolve
model/postprocessing defaults; explicit model choices override preset model
selection. The quality-tier value remains the requested policy. Unknown preset
strings can be retained as metadata labels; they do not automatically load an
arbitrary YAML configuration. Inspect `--plan` for the resolved stages and identity.

---

### Mistake 2: Enabling PBR Without Disabling V2

**Problem:**
```bash
# ❌ V2 is enabled by default, will fail if script missing
--pbr "on" --quality-tier "apex"
```

**Solution:**
```bash
# ✅ Explicitly disable V2 for PBR-only workflows
--pbr "on" --quality-tier "apex" --enable-v2 "off"
```

**Explanation:** V2 enhancement is enabled by default. Always explicitly disable it for PBR-only workflows.

---

### Mistake 3: Research Models Without License Acknowledgement

**Problem:**
```bash
# ❌ Explicit research selection missing its required license acknowledgement
--model-key "da3-research"
```

**Solution:**
```bash
# ✅ Acknowledge non-commercial license
--model-key "da3-research" \
--non-commercial-ok "true"
```

**Explanation:** Research models require explicit license acknowledgement. The CLI enforces this for compliance.

---

### Mistake 4: Mixing Commercial and Research Flags

**Problem:**
```bash
# ❌ Unnecessary research flags for commercial workflow
--quality-tier "apex" \
--depth-backend "da3" \
--non-commercial-ok "true"
```

**Solution:**
```bash
# ✅ Use the Apache DA3 selector for commercial workflows
--quality-tier "apex" \
--depth-backend "da3" \
--model-key "da3-metric"
```

**Explanation:** the `da3` model selector (deprecated) resolves the research model and requires the non-commercial acknowledgement. Use `--model-key "da3-metric"` (or simply omit `--model-key` — it is the default as of repair 1.2, #2066) for the Apache-2.0 DA3 path, or `--model-key "da3-research"` explicitly for the research model.

---

## Flag Usage Guide

### Essential Flags (Always Needed)

```bash
--input-dir PATH          # Required: Source images directory
--output-dir PATH         # Required: Output artifacts directory
```

### Quality Control

```bash
--quality-tier TIER       # Recommended: standard|premium|apex
--preset NAME             # Advanced: Curated preset or metadata label
```

### V2 Enhancement (Optional, Default: ON)

```bash
--enable-v2 on|off        # Master switch for V2 stage
--v2-preset NAME|none     # V2 preset or 'none' to skip
```

**Default Behavior:** V2 is **ON** by default. Disable explicitly for PBR-only workflows.

### Feature Toggles (On/Off Flags)

```bash
--pbr on|off              # Enable PBR map generation
--materials-v3 on|off     # Enable surface-aware finishing
--cache-depth on|off      # Enable depth caching
```

### Output Encoding and Deliverables

```bash
--output-bit-depth 8|16   # 8-bit PNG (default) or 16-bit TIFF
--emit-run-card on|off    # Reproducibility card (default: on)
--save-float-depth on|off # Optional canonical .npy depth (default: off)
```

`--emit-marketing` and `--emit-report` are deprecated compatibility flags
scheduled for removal in the next major release. No marketing artifact is
created, and the combined processing report is always emitted.

### Hardware Acceleration

```bash
--depth-device cpu|cuda|mps  # Device for depth inference
```

### Research Models (Optional)

```bash
--non-commercial-ok true              # Required for CC BY-NC 4.0 models
--accept-apple-depth-pro-research-license true  # Required for Depth Pro
```

---

## Performance Issues

### Symptom: Slow Processing on GPU-Capable Hardware

**Problem:** Pipeline is using CPU despite having CUDA/MPS available.

**Solution:**
```bash
# For NVIDIA GPUs:
--depth-device "cuda"

# For Apple Silicon (M1/M2/M3/M4):
--depth-device "mps"
```

**Verification:**
```bash
# Use the selected worker Python, not the main environment, for this import check:
.runtime/Depth-Anything-3/.venv-da3/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

---

### Symptom: Out of Memory (OOM)

**Problem:** Pipeline crashes with memory errors during depth estimation.

**Solutions:**

1. **Inspect the failing stage and input size.** A cache hit can avoid repeated
   depth inference only when the complete runtime identity is authorized.
   `--cache-depth on` does not reduce memory for a first uncached run.

2. **Reduce Batch Size:** Process images in smaller batches.

3. **Lower Quality Tier:**
   ```bash
   --quality-tier "premium"  # Instead of "apex"
   ```

4. **Use CPU Instead of GPU:**
   ```bash
   --depth-device "cpu"
   ```
   Slower but uses system RAM instead of VRAM.

---

### Symptom: Depth Pro exits before inference with missing `jsonschema`

`RUNNER_EXIT_NONZERO` is the orchestrator's exit summary. Inspect the selected
job's Logs tab for the first backend exception. If the isolated Depth Pro
worker reports `ModuleNotFoundError: No module named 'jsonschema'`, its runtime
is missing the validator required by `tp.execution.plan.v1`.

For an otherwise working `.venv-depth-pro`, repair the missing dependency and
verify it without replacing the existing ML packages:

```bash
./.venv-depth-pro/bin/python -m pip install "jsonschema==4.26.0"
./.venv-depth-pro/bin/python -m pip check
PYTHONPATH=src ./.venv-depth-pro/bin/python -m transformation_portal.depth.backends.depth_pro_worker --check --checkpoint checkpoints/depth_pro.pt --device cpu
```

The full `./scripts/setup/install_depth_pro_runtime.sh` installer also installs
this dependency and now validates the packaged execution-plan schema during
readiness. A fresh runtime installation replaces the venv. Keep the existing
research-license acknowledgments and canonical-plan validation intact; synthetic
fallback does not repair a production runtime.

Strict input discovery intentionally excludes hidden folders and generated
output paths. Place validation sources under a normal input directory such as
`input_images/local_managed_validation/`, rather than `.runtime/` or `output/`.
Historical failed jobs remain failed after a repair; verify a new job and its
indexed outputs before rerunning a larger batch.

### Symptom: OpenMP Runtime Collision on macOS (OMP Error #15)

**Error Message:**
```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

**Cause:** On macOS, multiple copies of the OpenMP runtime (libomp.dylib) are being loaded into the same process. This commonly occurs when:
- Using a separate Python environment for Depth Pro (`.venv-depth-pro`)
- Different PyTorch builds between environments link different OpenMP libraries
- Homebrew-installed libomp conflicts with wheel-bundled libomp

**Solution:**

The pipeline has been updated to defer torch imports during backend initialization,
which prevents this collision in most cases. However, if you still encounter this error:

1. **Ensure consistent libomp across environments:**
   - Rebuild `.venv-depth-pro` with `./scripts/setup/install_depth_pro_runtime.sh`
   - Keep the Depth Pro subprocess on the repo-owned pin (`torch==2.13.0`, `torchvision==0.28.0`, `numpy==1.26.4`) instead of mirroring the main repo runtime
   - Use the subprocess isolation mode (configured via `--depth-pro-python`)

2. **Inspect the actual failing process and imported libraries.** Suppressing
   duplicate OpenMP checks can hide the collision; repair the selected runtime
   and repeat its readiness/inference check instead of treating suppression as
   a production fix.

**When can this still occur?**
- If custom code imports torch before invoking the pipeline
- If a third-party library in your environment eagerly imports torch
- If you're running outside the standard orchestrator flow

**Technical Background:**
The `DepthProBackend` no longer imports torch during construction—it defaults to
CPU and defers device capability checks to compute() time. This prevents the
collision when the Depth Pro subprocess loads its own libomp from `.venv-depth-pro`.

---

## Input/Output Issues

### Symptom: "No images found in [directory]"

**Problem:** Input directory exists but no images detected.

**Supported Formats:**
- `.jpg`, `.jpeg` (standard JPEG)
- `.png` (8-bit or 16-bit PNG)
- `.tiff`, `.tif` (TIFF, any bit depth)
- `.webp` (WebP format)

**Solutions:**

1. **Check File Extensions:**
   ```bash
   ls -la ./input_images/
   ```
   Ensure files have supported extensions (case-insensitive).

2. **Check Subdirectories:**
   The pipeline searches subdirectories recursively. Ensure images are in the specified directory or its children.

3. **Convert Unsupported Formats:**
   ```bash
   # Convert BMP to JPEG:
   for f in *.bmp; do convert "$f" "${f%.bmp}.jpg"; done
   ```

---

### Symptom: "Input directory does not exist"

**Problem:** Specified `--input-dir` path is incorrect.

**Solutions:**

1. **Use Absolute Paths:**
   ```bash
   --input-dir "/Users/username/projects/input_images"
   ```

2. **Use Relative Paths from CWD:**
   ```bash
   --input-dir "./input_images"
   ```

3. **Verify Path:**
   ```bash
   ls -ld ./input_images
   ```

---

## License and Research Models

### Commercial Model Selection

**DA3 (`da3` backend):**
```bash
--depth-backend "da3" --model-key "da3-metric"  # Apache-2.0 model
```
- The registry identifies `da3-metric` as Apache-2.0; comply with that license.
- No research acknowledgement is required for this selector.
- The backend name alone is not a license grant; `da3-research` has separate terms.

---

### Research-Only Models (Restricted)

**Depth Anything V3.1 (CC BY-NC 4.0):**
```bash
--preset "depth-anything-v3.1-research-m4" \
--model-key "da3-research" \
--non-commercial-ok "true"
```
- ❌ Commercial use **not allowed**
- ✅ Research and academic use only
- Requires explicit license acknowledgement

**Apple Depth Pro (AMLR Research License):**
```bash
--depth-backend "depth_pro" \
--non-commercial-ok "true" \
--accept-apple-depth-pro-research-license "true"
```
- ❌ Commercial use **not allowed**
- ✅ Research use only
- Requires **two** license acknowledgements

---

### License Enforcement

The CLI **enforces license compliance** at startup:
- Research models require explicit `--non-commercial-ok true`
- Apple Depth Pro requires both flags
- Missing flags result in immediate error (fail-fast design)

**This is intentional design** to prevent accidental license violations.

---

## Common Error Messages

### "V2 enhancement script not found"

**Cause:** V2 is enabled but `scripts/enhance_image.py` is missing.

**Fix:** Add `--enable-v2 "off"` to your command.

**See:** [V2 Enhancement Issues](#v2-enhancement-issues)

---

### "Depth Pro backend requires --non-commercial-ok true"

**Cause:** Using Depth Pro without acknowledging non-commercial license.

**Fix:** Add both required flags:
```bash
--non-commercial-ok "true" \
--accept-apple-depth-pro-research-license "true"
```

---

### "Preset 'X' requires --non-commercial-ok true"

**Cause:** Using research preset without license acknowledgement.

**Fix:** Add license flag:
```bash
--non-commercial-ok "true"
```

---

### "Invalid quality tier 'X'"

**Cause:** Typo or unsupported quality tier.

**Valid Options:**
- `standard` (fast/draft)
- `premium` (balanced)
- `apex` (maximum quality)

**Fix:** Correct the spelling or choose a valid tier.

---

## Best Practices

### For Commercial Production

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/commercial" \
  --quality-tier "apex" \
  --depth-device "mps" \
  --pbr "on" \
  --materials-v3 "on" \
  --enable-segmentation on \
  --segmentation-backend efficientsam \
  --strict-segmentation \
  --enable-v2 "off" \
  --cache-depth "on" \
  --output-bit-depth 16
```

**Key Points:**
- Use `--quality-tier apex` for maximum quality
- Disable V2 if only depth/PBR needed
- Enable caching for faster iterations
- Emit all required deliverables

---

### For Research Experiments

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/research" \
  --preset "depth-anything-v3.1-research-m4" \
  --model-key "da3-research" \
  --non-commercial-ok "true" \
  --quality-tier "apex" \
  --depth-device "cuda" \
  --pbr "on" \
  --cache-depth "on"
```

**Key Points:**
- Explicitly acknowledge non-commercial license
- Use research presets for state-of-the-art models
- Document license restrictions in outputs

---

### For Fast Iteration (Development)

```bash
lux-depth-v3 \
  --input-dir "./input_images" \
  --output-dir "./output/dev" \
  --quality-tier "standard" \
  --depth-device "cpu" \
  --pbr "off" \
  --enable-v2 "off"
```

**Key Points:**
- Use `standard` quality for speed
- Disable optional features (PBR, V2)
- Focus on core pipeline validation

---

## Additional Resources

- [Lux Depth V3 CLI Guide](../cli/LUX_DEPTH_V3_CLI_GUIDE.md) - Complete command reference
- [Architecture Decision Record: Depth Backend Unification](../architecture/ADR-019-depth-backend-unification.md)
- [PBR CLI Testing Guide](../cli/PBR_CLI_TESTING_GUIDE.md)
- [Main README](../../README.md) - Repository overview

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Check verbose logs:**
   ```bash
   lux-depth-v3 [your-flags] --verbose
   ```

2. **Verify installation:**
   ```bash
   .venv/bin/python -m pip list | grep transformation-portal
   ```

3. **Check the failing runtime:**
   ```bash
   make check-environment
   ```
   This covers the core environment. Use the selected isolated worker and its
   installer/validator for DA3, Depth Pro, or FastVLM failures; a main-environment
   ML install does not repair another interpreter.

4. **Review configuration:**
   The pipeline emits a run card (`*_run_card.json`) with full configuration details for debugging.
