# Transformation Portal CLI Reference

Last updated: 2026-05-12

This is the current operator reference for repository CLI entrypoints. It is a
live support document, not a historical release report. Prefer the Makefile and
checked-in lock contracts over ad-hoc `pip install` instructions.

## Environment Baseline

```bash
source .venv/bin/activate
make install-core
make check-environment
```

The project declares Python `>=3.11`; the active local `.venv` used for this
refresh is Python 3.12. Console scripts are installed into `.venv/bin/`. If the
virtual environment is not activated, call scripts with `.venv/bin/<name>`.

Use these quick checks before relying on a CLI surface:

```bash
.venv/bin/python -m transformation_portal version
.venv/bin/python -m transformation_portal --help
.venv/bin/lux-depth-v3 --help
.venv/bin/depth-aware-dof --help
.venv/bin/presence-security --help
```

`python -m transformation_portal info` imports optional runtime probes and may
surface warnings for optional ML stacks that are not installed. Treat those as
environment signals, not as proof that the core CLI is broken.

## Entrypoint Map

| Entrypoint | Current role | Invocation |
| --- | --- | --- |
| Root package CLI | Recipe-driven processing and repo metadata | `.venv/bin/python -m transformation_portal ...` |
| Lux Depth V3 | Canonical image/depth/APEX pipeline | `.venv/bin/lux-depth-v3 ...` |
| PBR helper CLI | Generate PBR maps from depth assets | `.venv/bin/python -m transformation_portal.lux_depth_v3.pbr_cli ...` |
| Depth-aware DOF | Apply depth-aware focus rendering from an image and `.npy` depth | `.venv/bin/depth-aware-dof ...` |
| Presence Security | Sessionized Presence Compiler parameters, manifest anchor payloads, and watermark helpers | `.venv/bin/presence-security ...` |
| TIFF batch processor | Batch 16-bit TIFF finishing | `.venv/bin/luxury-tiff-batch ...` |
| Compatibility Typer CLIs | Legacy direct command groups for render/process/analyze helpers | `.venv/bin/transform-render`, `.venv/bin/transform-process`, `.venv/bin/transform-analyze` |

## Root Package CLI

The root CLI is implemented as a module entrypoint:

```bash
.venv/bin/python -m transformation_portal --help
```

Current commands:

- `process`: recipe-driven batch processing.
- `list-recipes`: list available recipe YAML files.
- `validate-recipe`: validate a recipe file.
- `version`: print package version.
- `info`: print runtime capability information.

Example:

```bash
.venv/bin/python -m transformation_portal process \
  --input "input_images/*.jpg" \
  --recipe config/recipes/signature_estate.yaml \
  --output output/signature_estate \
  --mode auto \
  --log-level info
```

Recipe helpers:

```bash
.venv/bin/python -m transformation_portal list-recipes --dir config/recipes
.venv/bin/python -m transformation_portal validate-recipe config/recipes/signature_estate.yaml --verbose
```

## Presence Security

The Presence Security CLI is maintained under
`transformation_portal.presence_security`:

```bash
.venv/bin/presence-security params --session "demo-session" --locale US_EN
.venv/bin/python -m transformation_portal.presence_security --help
```

Current commands:

- `params`: emit deterministic sessionized parameters.
- `anchor`: emit SHA3 anchor payload hashes for manifest, hero, web, and the
  combined hero+web asset bundle.
- `watermark`: embed LSB or DCT manifest/session watermarks in an image.

Schemas and examples live under `docs/schemas/presence/` and
`docs/contracts/examples/tp.presence.*.example.json`.

## Lux Depth V3

The canonical Lux Depth V3 CLI is documented in
[LUX_DEPTH_V3_CLI_GUIDE.md](LUX_DEPTH_V3_CLI_GUIDE.md). Required options are
`--input-dir` and `--output-dir`.

Commercial APEX runs should use the Apache-2.0 DA3 selector:

```bash
.venv/bin/lux-depth-v3 \
  --input-dir input_images \
  --output-dir output/lux_depth_v3_apex \
  --preset premium \
  --quality-tier apex \
  --depth-backend da3 \
  --model-key da3-metric \
  --materials-v3 on \
  --pbr on \
  --cache-depth on \
  --emit-master16 on \
  --emit-upscaled16 on \
  --emit-marketing on \
  --emit-report on \
  --emit-run-card on \
  --run-card-version v2 \
  --overwrite
```

Research-only model selectors such as `da3`, `da3-research`, and
`depth_pro` require explicit non-commercial license acknowledgement. Depth Pro
also requires Apple Depth Pro research-license acknowledgement.

Current option groups include:

- Depth backend selection: `--depth-backend`, `--model-key`,
  `--depth-device`, `--da3-python`, `--depth-pro-python`.
- RAW subprocess control: `--raw-python`, `--raw-ingest-mode`,
  `--raw-wb-mode`, `--raw-demosaic`.
- Materials and PBR: `--materials-v3`, `--pbr`, `--save-float-depth`,
  `--cache-depth`, `--enable-segmentation`, `--segmentation-backend`,
  `--strict-segmentation`, `--segmentation-cache`, and SAM2 controls.
- Advisory captioning: `--vlm-captioning`, `--vlm-captioning-backend`,
  `--vlm-captioning-model`, `--vlm-captioning-proxy-format`,
  `--vlm-captioning-max-side-px`, `--fastvlm-python`,
  `--fastvlm-mlx-vlm-dir`, and `--fastvlm-timeout-seconds`.
- Reproducibility: emit flags, `--emit-run-card`, `--run-card-version`,
  `--keep-intermediates`, and run-card signing/verifier tools.
- Safety and execution: `--strict-inputs`, `--verify-images`,
  `--allow-semantic-fallback`, `--max-workers`, `--max-gpu-workers`,
  `--overwrite`, and `--force-depth`.

Repo-governed runtime setup commands are documented in `AGENTS.md` and include:

```bash
make install-ml-core
./scripts/setup/install_da3_runtime.sh
./scripts/setup/install_depth_pro_runtime.sh
./scripts/setup/install_raw_runtime.sh
./scripts/setup/install_fastvlm_runtime.sh
make check-fastvlm-runtime
```

## PBR Helper CLI

The PBR helper is a Typer app under the Lux Depth V3 package:

```bash
.venv/bin/python -m transformation_portal.lux_depth_v3.pbr_cli --help
.venv/bin/python -m transformation_portal.lux_depth_v3.pbr_cli generate --help
```

Single depth asset:

```bash
.venv/bin/python -m transformation_portal.lux_depth_v3.pbr_cli generate \
  --depth output/frame_depth.npy \
  --output output/pbr \
  --preset premium \
  --base-name frame
```

Batch depth directory:

```bash
.venv/bin/python -m transformation_portal.lux_depth_v3.pbr_cli generate \
  --depth-dir output/depth \
  --output output/pbr \
  --pattern "*_depth.*" \
  --preset standard \
  --manifest output/pbr/manifest.json \
  --json
```

Current presets are `standard`, `premium`, `draft`, `wood`, `metal`, `glass`,
`stone`, and `fabric`. Useful safety flags include `--dry-run`, `--fail-fast`,
`--max-files`, and `--overwrite/--no-overwrite`.

## Depth-Aware DOF

Use the dedicated console script for single-image depth-aware focus rendering:

```bash
.venv/bin/depth-aware-dof \
  --source input/frame.tiff \
  --depth-npy output/depth/frame_depth.npy \
  --metadata output/depth/frame_metadata.json \
  --out-dir output/dof \
  --preview-long-edge 2400
```

Optional controls include `--protect-mask`, `--sky-mask`, `--edge-mask`,
`--focus-depth`, `--focus-roi X Y W H`, and `--depth-convention` when metadata
does not provide the depth direction.

## TIFF Batch Processor

Use `luxury-tiff-batch` for batch TIFF finishing:

```bash
.venv/bin/luxury-tiff-batch input/source_tiffs output/tiff_finished \
  --preset signature \
  --profile quality \
  --recursive \
  --suffix _finished \
  --workers 4
```

Current presets are `architectural`, `golden_hour_courtyard`, `signature`, and
`twilight`. Current profiles are `balanced`, `performance`, and `quality`.

## Compatibility Typer CLIs

These entrypoints remain available for targeted helper flows:

```bash
.venv/bin/transform-render lux --input input.jpg --output output/lux --prompt "luxury interior" --upscale
.venv/bin/transform-render depth --input input.jpg --output output/depth --preset interior
.venv/bin/transform-process material --input input.jpg --output output/material.jpg --strength 0.7
.venv/bin/transform-process video --input input.mp4 --output output/graded.mp4 --preset signature_estate
.venv/bin/transform-process tif --input input/source_tiffs --output output/tiff_finished --preset signature --recursive
.venv/bin/transform-analyze workflow
```

Keep these compatibility commands narrow. The canonical production operator path
for current depth/material deliverables is `lux-depth-v3`.

## Validation

For docs-only CLI reference updates:

```bash
git diff --check
make check-docs
make check-stale-docs
make check-doc-heading-links
python3 scripts/governance/check_docs_structure.py --all
```

For PBR CLI behavior:

```bash
.venv/bin/python -m pytest -q tests/test_pbr_cli.py tests/test_pbr_cli_contract.py
```

For broader repo confidence:

```bash
make check-environment
make test-fast
make test-orchestrator-contract
make test-frontdoor-contract
make ci
```

Do not describe the repository as fully healthy unless the requested canonical
lane, especially `make ci`, has actually passed.
