# LuxDepthV6 retained-evidence grading

LuxDepthV6 is an opt-in, standalone finishing successor for a completed,
independently verified LuxDepthV5 generation. It reconstructs a conservative
photographic baseline from the retained original pixels and native depth,
applies explicit color controls in floating point, and renders a separate SDR
delivery. It does not start new model inference or authorize a managed portal
job. LuxDepthV3 remains the production default and rollback path.

V6 establishes reproducible processing and output verification. Representative
photographic quality, calibrated color accuracy, and production acceptance remain
unestablished. Its display transforms are local versioned recipes, not ACES,
AgX, camera profiling, or HDR delivery.

## Source requirements

Keep the complete V5 output directory described in the
[V5 operator guide](LUX_DEPTH_V5.md). V6 verifies its canonical execution plan,
completion evidence, full artifact inventory, array geometry, and replayed image
products before admission. Verification binds retained bytes; it does not
independently authenticate those pixels against the original camera file.

A V5 generation with applied Materials responses is rejected: that generation
does not retain all masks needed to replay those operations independently.
Abstained Materials responses must also have zero reported changes. Calibration,
source ICC bytes, alpha, unavailable sky, and native-depth validity remain bound
to the V5 source evidence.

V6 uses the upstream strength, clarity, and precision settings. Guided
reconstruction additionally requires persistent native surfaces. A photograph
with any nonopaque alpha sample abstains from depth reconstruction because the
upstream model proxy does not account for transparency. Explicit grading still
applies to its fully opaque pixels; nonopaque pixels remain unchanged in the
unbounded graded master. Display rendering can change RGB for delivery while
preserving the alpha samples.

## Plan, run, and verify

Run from the repository root with the supported core environment. V6 uses the
existing NumPy, Pillow, SciPy, tifffile, and imagecodecs dependencies; it requires
neither a new model download nor a new ML runtime.

Choose an existing V5 generation and a new, disjoint V6 output directory whose
parent already exists:

```bash
TP_V6_PARENT=/absolute/output-v5
TP_V6_OUTPUT=/absolute/output-v6

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth_v6 \
  --input-dir "$TP_V6_PARENT" --output-dir "$TP_V6_OUTPUT" \
  --exposure-stops 0.25 --saturation 1.05 --render perceptual_srgb --plan

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth_v6 \
  --input-dir "$TP_V6_PARENT" --output-dir "$TP_V6_OUTPUT" \
  --exposure-stops 0.25 --saturation 1.05 --render perceptual_srgb

PYTHONPATH=src ./.venv/bin/python -m transformation_portal.lux_depth_v6 \
  --input-dir "$TP_V6_PARENT" --output-dir "$TP_V6_OUTPUT" --verify
```

`--plan` reads and verifies the parent but creates no output. Running the same
controls against the same retained parent and processing implementation produces
the same canonical plan. The CLI prints that plan with a trailing newline; its
reported SHA-256 covers the canonical JSON bytes without that output newline.
Execution prints `plan_sha256`; supply that value as
`--expected-plan-sha256 <digest>` with `--verify` when verifying an exact intended
plan. The existing output is allowed only for verification. A failed attempt
keeps its partial directory for inspection and requires a new output directory
for a subsequent run.

Installing the current source also exposes `lux-depth-v6`. The module invocation
above remains available when an existing checkout environment has not refreshed
its console scripts. Python callers can prepare once and execute that exact
immutable carrier:

```python
from pathlib import Path

from transformation_portal.lux_depth_v6 import GradeRecipe, LuxDepthV6Request, prepare
from transformation_portal.lux_depth_v6.pipeline import run

prepared = prepare(
    LuxDepthV6Request(
        Path("/absolute/output-v5"),
        Path("/absolute/output-v6"),
        grade=GradeRecipe(exposure_stops=0.25, saturation=1.05),
    )
)
result = run(prepared)
```

## Color controls

| CLI control | Default | Accepted range and meaning |
| --- | --- | --- |
| `--exposure-stops` | `0` | `-8` to `8`; multiplies linear RGB by `2**stops` |
| `--white-balance R G B` | `1 1 1` | Each gain `0.25` to `4`; explicit RGB multipliers, without hidden normalization or Kelvin inference |
| `--contrast` | `1` | `0.25` to `4`; signed luminance power around the pivot, preserving RGB ratios; zero luminance is unchanged |
| `--pivot` | `0.18` | `0.001` to `1`; linear-light contrast pivot |
| `--saturation` | `1` | `0` to `2`; scales Oklab chroma while retaining the named space's lightness and hue before rendering |
| `--render` | `perceptual_srgb` | `perceptual_srgb`, `soft_srgb`, or `clip_srgb`; selects the separate SDR transform |
| `--shoulder` | `0.8` | `0.1` to `0.95`; highlight shoulder threshold; inactive for explicit clipping |

Exposure and white balance run first, then luminance contrast, then chroma.
Default grading controls preserve reconstructed master pixel bytes exactly,
including signed zero. The float master retains negative and above-white values;
operations that exceed finite float32 representation fail instead of clipping
the master. Non-finite values, booleans masquerading as numbers, unknown recipe
fields, and unsupported recipe versions are rejected.

Oklab uses the [published 2021 reference transform](https://bottosson.github.io/posts/oklab/).
Numerical support for signed RGB does not establish perceptual accuracy for
negative values or near-cancelling cone responses. White-balance gains should
come from an intentional photographic decision or a suitable neutral reference;
V6 does not estimate an illuminant automatically.

The default `perceptual_srgb` renderer applies a smooth shoulder to Oklab
lightness expressed as a brightness-like cubic quantity, then contracts chroma
at fixed Oklab lightness and hue to fit the display gamut. Its bounded numerical
search is part of the recipe. This is a named-space numerical contract rather
than a universal perceptual guarantee. In-gamut values below the shoulder remain
unchanged. `soft_srgb` provides the earlier linear-luminance shoulder with chroma
contraction toward neutral; it does not promise perceptual hue constancy.
`clip_srgb` provides an explicit independent-channel clipping comparison.

The working space remains extended linear sRGB. Unsupported input profiles and
RAW highlight clipping in the original V5 ingest cannot be repaired by labeling
the pixels differently or by increasing output bit depth. V6 does not add a
wide-gamut ingest transform or recover missing sensor or native-depth detail.

## Outputs and replay

Each input has separate artifacts:

| Artifact | Authority |
| --- | --- |
| `baseline.npy` | Float32 photographic baseline reconstructed from retained original pixels and native depth |
| `master.npy` | Unbounded float32 graded RGB; precision authority before display rendering |
| `display.npy` | Bounded float32 display-linear sRGB used by both delivery encoders |
| `alpha.npy` | Optional unchanged straight-alpha samples |
| `delivery.tif` | 16-bit encoded sRGB TIFF with embedded profile and normalized orientation |
| `preview.png` | Bounded 8-bit sRGB browser derivative, maximum edge 1600 pixels; reduction handles premultiplied alpha |
| `photograph.json` | Baseline/master/display descriptors and reconstruction, grade, render, and encoding receipts |

The generation root contains `plan.json` and, only after all products pass
semantic replay, `evidence.json`. The plan binds parent evidence, recipes,
resource ceilings, processing source hashes, and dependency versions.
Verification checks exact expected products and independently regenerates their
bytes from the retained V5 parent. Replacing an image and merely recomputing its
inventory hash cannot authorize the changed pixels. Processing-source or bound
dependency drift requires a new plan; verification under another implementation
is not silently treated as equivalent.

Verification securely snapshots the completion record again after replay. A
record that changes, disappears, or gains a link alias during verification is
rejected instead of returning the completion bytes cached at entry.

The processing identity covers the selected Python source files and dependency
version strings used by this recipe. It is a same-environment replay boundary,
not an attestation of every installed wheel byte, native library, operating
system component, or numerical kernel. Matching version strings alone do not
prove equivalence across different binary builds or hosts; independently verify
the products in the intended environment before claiming equivalence.

Keep the complete V5 parent alongside the V6 result. V6 is not a standalone
replacement for its upstream provenance bundle. An `ImageMaster` returned by
`render_master` is marked `color_domain="display_linear_srgb"`; passing it back
to grading or rendering is rejected to prevent a second display transform.
Low-level APIs are `grade_master(master, GradeRecipe(...))` and
`render_master(master, RenderRecipe(...))` in `lux_depth_v6.color`; each returns
an immutable master and a measured receipt.

## Resource and validation boundaries

Defaults are 64 GiB retained input, 64 GiB output, 100 million pixels per image,
16 GiB admitted memory, and 3600 seconds execution time. The conservative V6
memory admission is `pixels * 256 + 256 MiB` per image; the upstream verification
also accounts for its proxy and ICC storage. Output admission reserves every
float product, TIFF, bounded PNG, metadata, plan, and completion before writing.
These ceilings are frozen and rechecked by plan parsing and source admission.
They are conservative numerical admission bounds, not hard operating-system
memory isolation. Cancellation and deadlines are observed at processing and
verification checkpoints.
These checks are cooperative: they do not preempt an individual NumPy/SciPy or
encoding operation in progress, and are not a hard process-isolation deadline.

Use the focused validation gate:

```bash
make test-lux-depth-v6-contract
```

Synthetic contracts establish arithmetic, identity, failure handling, and
replay behavior. Production promotion additionally needs representative interior
and exterior photographs, controlled color references, paired full-frame and
100% inspection, native runtime acceptance, and measured performance. Preserve
V3 and the retained V5 source until those independent acceptance gates pass.
