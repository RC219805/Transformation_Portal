# FastVLM Advisory Captioning Runtime

FastVLM is integrated as an optional advisory captioning sidecar for local Lux
Depth V3 runs. It is not a quality gate and is not part of the main Python
runtime.

## Runtime Isolation

FastVLM MUST stay outside `Transformation_Portal/.venv`. Use this local layout:

```text
.runtime/fastvlm/
  .venv-fastvlm/
  ml-fastvlm/
  mlx-vlm/
  checkpoints/
    FastVLM-0.5B-fp16/
    FastVLM-1.5B-int8/
    FastVLM-7B-int4/
  test_images/
  reports/
```

The main pipeline calls FastVLM by subprocess only:

```bash
python -m mlx_vlm.generate \
  --model "$MODEL" \
  --image "$IMAGE" \
  --prompt "$PROMPT" \
  --max-tokens 120 \
  --temperature 0.0
```

No production code imports `mlx_vlm`, MLX, or CoreML packages.

## Model Policy

Use these model roles for local operation:

```text
default_caption_model = apple/FastVLM-1.5B-int8
review_caption_model  = apple/FastVLM-7B-int4
smoke_caption_model   = apple/FastVLM-0.5B-fp16
```

The default is `apple/FastVLM-1.5B-int8` because the May 2, 2026 benchmark
showed clean flat-schema adherence, roughly 2.5 GB peak memory, and about
200-209 generated tokens/sec. The `apple/FastVLM-7B-int4` model produced the
best captions but used roughly 4.95 GB peak memory and about 99-102 generated
tokens/sec. The benchmark report was written to
`.runtime/fastvlm/reports/fastvlm_caption_benchmark_20260502_142926.txt`.

## Input Image Policy

Canonical TIFF/RAW assets remain authoritative. FastVLM consumes deterministic
8-bit RGB proxies by default:

```text
source TIFF/JPEG/PNG/RAW preview -> RGB PNG proxy -> FastVLM subprocess
```

Direct TIFF inference is tolerated for local diagnostics, but PNG proxy
normalization is the production input policy for reproducibility.

## Governance

FastVLM output MUST be treated as advisory metadata.

FastVLM output MUST NOT be used as a quality gate.

FastVLM output MUST NOT satisfy Materials V3 segmentation or material-confidence
requirements.

Canonical TIFF/RAW assets MUST remain authoritative; FastVLM proxy images are
derived runtime inputs only.

All sidecars use `role: advisory` and `used_for_quality_gate: false`. Run-card
validation fails closed if a captioning status claims
`used_for_quality_gate: true`.

## Portal Feature Gate

Portal controls for FastVLM advisory captioning are hidden by default. Enable
them only through the backend bootstrap feature flag:

```bash
export TP_PORTAL_FASTVLM_CAPTIONING_ENABLED=1
export TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT=100
```

`/portal/bootstrap` exposes the cohort decision as
`features.fastVlmCaptioning`. Disabled cohorts cannot dispatch
`vlm_captioning_enabled=true`; config preview returns the
`captioning_feature_disabled` field error instead.

When enabled, the portal maps the captioning controls to the existing Lux Depth
V3 flags and shows the resulting command preview, effective config, readiness
warnings, and review-side advisory caption panel. Missing FastVLM runtime paths
remain non-blocking preview warnings because captions are metadata only and do
not satisfy quality gates.

## Local Setup

Use the governed runtime installer from the repository root:

```bash
./scripts/setup/install_fastvlm_runtime.sh
```

The default install prepares the source clones, isolated virtual environment,
and `smoke,default` model roles. Add the review model explicitly when needed:

```bash
./scripts/setup/install_fastvlm_runtime.sh --models smoke,default,review
```

The installer is manifest-backed and fail-closed:

```bash
./scripts/setup/install_fastvlm_runtime.sh --dry-run --models smoke,default
./scripts/setup/install_fastvlm_runtime.sh --verify-only --models smoke
./scripts/validation/validate_fastvlm_runtime.py --verify-only --models smoke
```

The manifest lives at `config/fastvlm_runtime_manifest.json` and pins:

```text
apple/ml-fastvlm@592b4add3c1c8a518e77d95dc6248e76c1dd591f
Blaizzy/mlx-vlm@1884b551bc741f26b2d54d68fa89d4e934b9a3de
```

The isolated Python dependency set is pinned in
`config/fastvlm_runtime_requirements.txt`; the installer installs that file
first and then installs the pinned `mlx-vlm` checkout with `--no-deps` so the
manifest and requirements file remain the source of truth.

Model downloads are limited to the allowlisted FastVLM roles, pinned Hugging
Face revisions, and SHA-256-checked required files. Partial downloads, unsafe
paths, symlink escapes, unpinned revisions, checksumless artifacts, and checksum
mismatches are rejected before a checkpoint is promoted into
`.runtime/fastvlm/checkpoints/`.

Make targets are available for operator workflows:

```bash
make install-fastvlm-runtime
make check-fastvlm-runtime
TP_PORTAL_FASTVLM_CAPTIONING_ENABLED=1 \
TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT=100 \
make validate-portal-fastvlm-captioning-live
```

## Standalone Smoke Test

Run the local command against one source image:

```bash
python -m transformation_portal.vlm_captioning \
  --input-image /path/to/source.tif \
  --output-dir /path/to/vlm_captioning_out \
  --model-path /Users/richardcheetham/Desktop/Transformation_Portal/.runtime/fastvlm/checkpoints/FastVLM-1.5B-int8 \
  --fastvlm-python /Users/richardcheetham/Desktop/Transformation_Portal/.runtime/fastvlm/.venv-fastvlm/bin/python \
  --mlx-vlm-dir /Users/richardcheetham/Desktop/Transformation_Portal/.runtime/fastvlm/mlx-vlm \
  --proxy-format png \
  --max-side-px 1600
```

Expected outputs:

```text
vlm_captioning_out/
  image_proxy.png
  vlm_captioning.sidecar.json
  vlm_captioning.raw.txt
```

## Lux Depth V3 Use

Captioning is off by default:

```bash
python -m transformation_portal.lux_depth_v3 \
  --input-dir /Users/richardcheetham/Desktop/Transformation_Portal/input_images \
  --output-dir /Users/richardcheetham/Desktop/Transformation_Portal/output/lux_depth_v3_with_fastvlm \
  --quality-tier apex \
  --vlm-captioning on \
  --vlm-captioning-backend fastvlm \
  --vlm-captioning-model default \
  --vlm-captioning-proxy-format png \
  --fastvlm-python /Users/richardcheetham/Desktop/Transformation_Portal/.runtime/fastvlm/.venv-fastvlm/bin/python \
  --fastvlm-mlx-vlm-dir /Users/richardcheetham/Desktop/Transformation_Portal/.runtime/fastvlm/mlx-vlm
```

The run card records `captioning_status.role = advisory` and
`captioning_status.used_for_quality_gate = false`.
