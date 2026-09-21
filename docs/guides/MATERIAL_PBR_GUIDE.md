# Material PBR Integration Guide

**Status:** Maintained heuristic PBR API guidance
**Last source review:** 2026-09-21

## Current Capability

`MaterialBackend` generates six approximate texture maps from linear RGB,
optionally using depth and a known material hint: albedo, normal, roughness,
metallic, ambient occlusion, and height. The currently executable backend in
this single-image API is `heuristic`, using CPU image-processing operations.
These estimates do not establish measured physical material properties or
photographic acceptance.

PBRFusion integration is not implemented. `backend="pbr_fusion"` resolves to
heuristic execution with `runtime_missing` when its runtime path is absent, or
`integration_missing` when `PBRFUSION_PATH` exists. Setting that variable does
not enable GPU inference. NVDIFFREC and MaterialGAN requests report
`input_contract_mismatch` because this API lacks their richer capture inputs.
Use strict backend selection when a fallback would be unacceptable.

[MaterialsV4](MATERIALS_V4.md) is the opt-in successor for material evidence and
bounded photographic response. It does not replace physical PBR reconstruction
or the depth-derived three-map [PBRProcessor](PBR_PROCESSOR_QUICKSTART.md).

## Setup

Run from the repository root:

```bash
make venv
make install-core
make check-environment
```

Use `.venv/bin/python` for scripts. Heuristic generation requires the core
NumPy/SciPy environment and no model download. Optional OpenCV availability
changes albedo filtering; record `metadata.bilateral_enabled` and the runtime
when comparing outputs. Do not interpret device selection as a GPU capability
claim for this backend.

## Quick Start

Supply an already decoded linear RGB `float32` array in `[0, 1]`. A plain cast
of an sRGB JPEG to floating point does not linearize its color values.

```python
from transformation_portal.spatial_ai.materials import MaterialBackend

backend = MaterialBackend(backend="heuristic", device="cpu")
result = backend.generate_pbr_textures(
    rgb=linear_rgb,           # [H, W, 3], float32, linear RGB in [0, 1]
    depth=depth_map,           # optional [H, W] float32 array, aligned to RGB
    material_hint="wood",     # optional known material category
)

albedo = result.albedo
normal = result.normal
roughness = result.roughness
metallic = result.metallic
ao = result.ambient_occlusion
height = result.height

if result.metadata is not None:
    print(result.metadata.backend)
    print(result.metadata.backend_decision)
    print(result.metadata.bilateral_enabled)
```

Omit `depth` and `material_hint` when unavailable. Hints select heuristic
roughness/metallic behavior; they are not inferred material labels. Provide
`mask=region_mask` for an aligned, nonempty boolean region. Reuse one backend
instance for a batch and retain source identity, color interpretation, depth,
mask, material hint, configuration, and generation metadata with each result.

## Preset Selection

`config/presets/material_pbr.yaml` and `material_pbr_canary.yaml` are materials
configuration artifacts. Their version labels and old invocation comments do
not prove that a neural backend executes. They are not values accepted by the
V2 enhancement script's `--preset` argument.

For the separate V2 image-finishing workflow, use a named preset and the actual
positional input / `--output-dir` interface:

```bash
.venv/bin/python scripts/enhance_image.py input/scene.png \
  --preset architectural \
  --output-dir output/scene_finishing \
  --device cpu
```

Accepted V2 names are `default`, `luxury_estate`, `architectural`, and `none`.
This command finishes an image; use the `MaterialBackend` API above to generate
six PBR maps. See [CLI Reference](../cli/CLI_REFERENCE.md) for pipeline routing.

## Explicit Generation Configuration

The material API accepts `MaterialGenerationConfig` independently of V2 presets:

```python
from transformation_portal.spatial_ai.materials.contracts import MaterialGenerationConfig

config = MaterialGenerationConfig(
    backend="heuristic",
    device="cpu",
    normal_strength=1.0,
    ao_intensity=0.7,
    strict_backend=True,
)
result = backend.generate_pbr_textures(rgb=linear_rgb, config=config)
```

Strict mode raises `BackendResolutionError` if the requested backend cannot
execute. A request for an unavailable neural backend must not be reported as
successful neural generation merely because heuristic maps were returned.

## Output Contract

| Field | Shape | Range / meaning |
| --- | --- | --- |
| `albedo` | `[H, W, 3]` | float32 `[0, 1]`, filtered RGB estimate |
| `normal` | `[H, W, 3]` | float32 `[-1, 1]`, gradient-derived normals |
| `roughness` | `[H, W]` | float32 `[0, 1]`, heuristic roughness |
| `metallic` | `[H, W]` | float32 `[0, 1]`, heuristic metalness |
| `ambient_occlusion` | `[H, W]` | float32 `[0, 1]`, larger means less occluded |
| `height` | `[H, W]` | float32 `[0, 1]`, normalized depth/luminance estimate |
| `properties` | `MaterialProperties` | Aggregate generated-map properties |
| `metadata` | Optional `PBRGenerationMetadata` | Executed backend, decision, parameters, filtering and timing |

Normal-map values require the appropriate conversion when exporting to unsigned
image formats. Check output shape, finite values, and ranges before use. Contract
success proves the array interface, not physical reconstruction accuracy.

## Troubleshooting

- **PBRFusion unavailable:** inspect `metadata.backend_decision`. An existing
  path still reports `integration_missing`; installing packages does not complete
  the unimplemented adapter. Select `heuristic` explicitly or fail closed with
  `strict_backend=True`.
- **Invalid V2 preset:** pass a supported name such as `architectural`, not a
  materials YAML filename. The V2 script also requires a positional input and
  `--output-dir`.
- **Weak normals:** check depth alignment and gradients, input resolution, and
  `normal_strength`. Flat input regions contain little geometric evidence.
- **Dark AO:** inspect supplied depth and reduce `ao_intensity` as appropriate;
  do not treat the result as measured occlusion.
- **Different albedo results across hosts:** inspect OpenCV availability and
  `bilateral_enabled`; compare under the same governed runtime.

## Validation And Acceptance

```bash
.venv/bin/pytest tests/spatial_ai/materials/test_material_backend.py -q
.venv/bin/pytest tests/test_documentation_operator_examples.py -q
```

These exercise local contracts and documented command admission. Historical
measurements in [Phase 5 PBR baselines](../performance/PHASE5_PBR_BASELINES.md)
are hardware/runtime-specific evidence, not current universal latency or quality
guarantees. Use the [Performance Gate Policy](../performance/GATE_POLICY.md)
and representative source images before accepting outputs for production.

A future neural PBR successor needs an implemented adapter, governed isolated
runtime and model identity, suitable capture inputs, explicit failure handling,
output verification, and independent photographic/performance acceptance. No
current installation command establishes that successor.

## Source References

- [Backend resolution and generation](../../src/transformation_portal/spatial_ai/materials/material_backend.py)
- [Texture and configuration contracts](../../src/transformation_portal/spatial_ai/materials/contracts.py)
- [Heuristic implementation](../../src/transformation_portal/spatial_ai/materials/heuristic_fallback.py)
- [V2 named presets](../../src/transformation_portal/lux_depth_v3/v2_presets.py)
- [MaterialsV4 candidate](MATERIALS_V4.md)
- [PBRProcessor quickstart](PBR_PROCESSOR_QUICKSTART.md)
