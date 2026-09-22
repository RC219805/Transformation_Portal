# LuxDepthV4/V5 forensics and the LuxDepthV6 successor

**Audit date:** 2026-09-22 UTC
**Audited baseline:** `b3759f7e3d8150399fba960163264e1d11b67462`
**Reviewed implementation workspace:** `/private/tmp/tp-lux-depth-v6-pr`
**Status:** Additive standalone candidate; production and photographic acceptance
are not established. V3 remains the production baseline and rollback path.

## Decision and scope

LuxDepthV6 is an evidence-bound photographic reconstruction and grading successor
to V5. It consumes a retained, independently verified V5 execution bundle,
reconstructs a more conservative depth response from its original float master
and native depth, grades in extended linear sRGB, and creates a separate bounded
SDR rendition. It does not replace the depth model, recover missing geometry,
activate managed V6 dispatch, or reinterpret historical V5 outputs.

This is the narrow successor supported by the audit: preserve the native model
and execution authority that already work, repair demonstrated alpha and
reconstruction defects, and separate creative grading from display encoding.
Selecting a better inference model or promising universally superior photographs
requires evidence this implementation does not supply.

Evidence labels used below:

- **Reproduced:** a deterministic local experiment demonstrates the behavior.
- **Source-proven:** the inspected implementation establishes the behavior.
- **Historical, inspected:** retained records were read during this audit;
  their original model execution is not a fresh execution of the current head.
- **Implemented:** source and focused regression coverage exist in this
  workspace; this label alone does not establish the full validation ladder.
- **Unmeasured:** a required photographic, physical, native, or operational
  acceptance claim lacks qualifying evidence.

The work does not provide a zero-bug guarantee. Passing numerical contracts and
replaying artifacts establish narrower properties than fidelity to a real scene
or agreement with an expert colorist.

## Findings and dispositions

### F1 — PNG color-key transparency was discarded

**Reproduced; implemented shared decoder repair.** RGB and grayscale PNG files
can carry transparency in a `tRNS` color-key chunk without an RGBA/LA channel.
The previous decoder read the RGB/gray samples directly and derived alpha only
from channel count. Such files therefore lost their transparent region before
photographic processing.

The shared V4 decoder now materializes PNG color-key transparency before
orientation and channel normalization. This also protects V5 and future V5
inputs used by V6. Explicit-channel alpha behavior remains supported; palette
and unsupported precision inputs have not been silently admitted.

Source: [decode_master](../../src/transformation_portal/lux_depth_v4/photography.py),
PNG ingest at lines 202–207. Regression:
`test_png_color_key_transparency_survives_ingest_and_delivery` in
[photographic tests](../../tests/lux_depth_v4/test_photography.py).
Previously generated opaque bundles cannot acquire missing alpha by replay;
regenerate affected inputs from the original PNG.

### F2 — Hidden RGB influences the alpha-unaware depth proxy

**Source-proven; implemented conservative V6 abstention.**
[create_proxy](../../src/transformation_portal/lux_depth_v4/photography.py),
lines 255–268, resizes RGB without an alpha-aware compositing policy. Protecting
nonopaque output pixels later does not remove hidden RGB from the proxy or
reverse its effect on neighboring opaque pixels and global model inference.

V6 therefore abstains from all depth finishing if any source alpha sample is
not exactly one. Its reconstruction receipt records
`alpha_unaware_upstream_proxy`; the original float pixels remain the baseline.
The grading layer separately protects nonopaque source RGB, and delivery keeps
the alpha channel. This does not claim alpha-aware native inference has been
implemented.

Source: [reconstruct_baseline](../../src/transformation_portal/lux_depth_v6/reconstruction.py),
lines 49–83; [reconstruction regressions](../../tests/lux_depth_v6/test_reconstruction.py).
A future inference recipe must bind its alpha/compositing policy, invalidate the
appropriate proxy/cache identity, and pass hidden-RGB metamorphic tests before
this abstention can be relaxed.

### F3 — An isolated native outlier was amplified by RGB guidance

**Reproduced; implemented versioned remediation.** The legacy guided recipe
required a complete 4×4 neighborhood whose values belonged to two depth groups,
but did not require both groups to persist beyond the central samples. One
incorrect native depth sample could therefore qualify as a second surface.

In a 112×112 analytic scene with a 28×28 native depth grid, a single far-depth
outlier on a flat near plane overlapped an unrelated 8×8 RGB texture square.
Legacy guidance expanded the outlier into 64 full-resolution samples. Within
the 20×20 diagnostic region, mean absolute relative-depth error rose from
**0.04 for bilinear interpolation to 0.16 for legacy guidance**, and the peak
rose from 0.765625 to 1.0. Native samples were unchanged. Reproduction and
receipts are retained under `/private/tmp/lux-successor-depth-audit/` as
`repro_isolated_outlier.py` and `repro_isolated_outlier.json`.

The explicit `guided_bilinear_v3` helper requires each group to occupy at least
four of the 16 native samples, including at least two outside the central 2×2.
The isolated outlier now retains bilinear interpolation. A true two-plane step
and a supported one-pixel-wide native strip remain eligible. Affine/gradual
slopes, holes, missing border support, and ambiguous color matches abstain.

Source: [_native_discontinuity_support](../../src/transformation_portal/lux_depth_v5/photography.py),
lines 124–168; [versioned regressions](../../tests/lux_depth_v5/test_photography.py).
V5's existing `guided_bilinear` recipe, default CLI, and Plan V4 semantics remain
unchanged so historical evidence remains replayable. V6 derives its own aligned
depth from the original/native carriers and records the new recipe. Coherent
multi-sample model errors may still resemble real surfaces; this guard is not
physical-depth confidence or a recovery algorithm for absent geometry.

### F4 — Applied Materials responses cannot be independently replayed spatially

**Source-proven; implemented admission restriction.** V5 bundles with applied
Materials responses do not retain the masks needed to reconstruct that spatial
operation against the new depth baseline. Reusing their finished master would
hide the old response inside a purportedly reconstructed V6 image.

V6 admission requires an explicit Materials abstention and zero changed-pixel
and delta reports, then invokes the V5 semantic verifier. Applied or nonzero
responses fail closed. This is a deliberate unsupported input, not evidence
that MaterialsV4 itself is defective.

Source: [_inspect_image](../../src/transformation_portal/lux_depth_v6/source.py),
the `materials` admission block. A future compatible carrier must include
source-bound masks, operation authority, uncertainty/protection semantics, and
replayable spatial receipts before material-aware V6 editing is enabled.

### F5 — Integrity is not independent source authenticity or sensor fidelity

**Source-proven boundary; remaining acceptance work.** The V5 verifier explicitly
does not authenticate source-camera pixels. V6 verifies canonical plans,
inventories, retained arrays, internal source/master/proxy identities, numeric
reconstruction, and output products. Hash consistency cannot independently prove
that a self-consistent bundle came from a particular camera exposure, checkpoint
execution, or externally trusted acquisition process.

Sources: [V5 verifier contract](../../src/transformation_portal/lux_depth_v5/evidence.py),
module docstring; [V6 source admission](../../src/transformation_portal/lux_depth_v6/source.py)
and [V6 output verification](../../src/transformation_portal/lux_depth_v6/evidence.py).
Acquisition authenticity needs retained original bytes plus independently trusted
provenance/attestation and an appropriate threat model. Those are separate from
the implemented bundle-consistency boundary.

Wide-gamut and RAW limits also remain upstream. Automatic ICC resolution
recognizes the contracted sRGB profile; it is not a general high-precision ICC
transform. An explicit `input_color` declaration identifies how bytes are to be
interpreted and does not convert Display P3/Adobe RGB into sRGB. The governed RAW
decoder records `LibRaw clip; no highlight reconstruction`. V6 cannot recover
sensor channels or color volume already discarded upstream.

Sources: [_resolve_color](../../src/transformation_portal/lux_depth_v4/photography.py),
lines 74–99; [RAW decode metadata](../../src/transformation_portal/lux_depth_v4/raw.py),
lines 288–291. A camera input transform is a distinct operation in the
[ACES input-transform model](https://docs.acescentral.com/system-components/input-transforms/);
adopting such a governed transform would require new input/runtime identity and
native reference evidence rather than relabeling the current sRGB master.

### F6 — Native resolution and production quality are not established

**Historical, inspected; broader acceptance unmeasured.** Retained V5 native
records cover the same 29,998,176-pixel, **8-bit-derived** photograph at target
sizes 518 and 1008. The native grids were 350×518, including padding, and
672×1008. The records report 17 verified artifacts for each and zero changed
protected pixels. The separate uint16 ramp was reported as precision-retention
evidence in its original record, not a high-bit-depth photographic acceptance set.

The retained analytic quality report contains one two-surface independence
group. It reports no absolute metric accuracy, insufficient independent groups
for uncertainty, and no production promotion. Its single ordinal annotation
does not meet that report's minimum of two valid pairs. The 1008 setting remains
an opt-in comparison; finer apparent railings in one image do not establish the
best default for windows, mirrors, foliage, interiors, fine ornament, or RAW.

Evidence: `/private/tmp/lux-depth-v5-pr-review/native/resolution-results.json`,
`quality/protected-pixels.json`, and `quality/report.json`; the prior
[V5 implementation record](LUX_DEPTH_V5_IMPLEMENTATION_VALIDATION_2026-09-20.md).
These records were inspected, not represented as fresh current-head inference.

**Current audit-time verification, separate from fresh inference:** the V6 source
admission path reverified the retained `photo-1008` parent: 17 artifacts totaling
1,565,568,067 bytes, with a 4472×6708 source, in 11.61 seconds on this host.
The admitted source digest was
`6c907d04f8942cbce5bb46c00577530c2b3f38628dd202e4d69ee223bee23ff3`.
This establishes current bundle admission/replay for those bytes; it does not
rerun the model, establish camera authenticity, or measure photographic quality.

The earlier `/private/tmp/lux-depth-v5-validation/native/accepted/ramp-cold`
bundle failed the same admission with
`V5 aligned depth descriptor differs from reconstructed semantics`.
It is excluded from current proof. Its historical record is retained without
weakening the verifier to accept the outdated descriptor. Both terminal results
are recorded in `/private/tmp/lux-v6-validation/parent-verification.json`.

### F7 — Ordinary product writes have a pathname race boundary

**Source-proven; V6 uses descriptor-confined publication.** The V4 executor,
also used by V5, writes ordinary arrays through `Path.open("xb")` and TIFFs
through a pathname-based temporary file. It observes the resulting artifact
after writing. A concurrent replacement of the destination directory can
therefore redirect a write before subsequent namespace/inventory checks detect
the change. This is a local filesystem race, not evidence of a remote exploit
or successful publication of a corrupted generation.

V6 encodes products into bounded byte carriers and writes every product through
the existing pinned directory-descriptor helper. It rechecks the pinned
namespace and requires semantic replay before successful completion. Tests
cover output-root replacement, source links, and failures during publication.
The earlier executor's pathname behavior is recorded here; this standalone
successor does not silently change historical V4/V5 execution semantics.

Sources: [V4 executor](../../src/transformation_portal/lux_depth_v4/pipeline.py),
array/delivery publication around lines 447–456;
[V6 executor](../../src/transformation_portal/lux_depth_v6/pipeline.py) and
[publication regressions](../../tests/lux_depth_v6/test_pipeline.py).

### F8 — Delivery clipping and creative grading need separate authority

**Source-proven; implemented separate V6 master/display products.** The shared
V4/V5 TIFF writer retains an extended float master but clips each encoded RGB
channel to the SDR interval before uint16 quantization. Sixteen output bits do
not preserve out-of-gamut color relationships or recover clipped highlights.
V6 keeps the unbounded graded master, renders a separate display-linear image,
and encodes that image to TIFF and PNG with measured receipts. Exposure,
white-balance gains, contrast, and chroma are explicit frozen controls rather
than effects hidden in delivery encoding.

Sources: [write_delivery](../../src/transformation_portal/lux_depth_v4/photography.py),
lines 379–415; [V6 color processing](../../src/transformation_portal/lux_depth_v6/color.py)
and [V6 product encoders](../../src/transformation_portal/lux_depth_v6/products.py).
The display transform necessarily reduces the available color volume for SDR;
its acceptance is separate from preservation of the float master.

### F9 — Completion evidence must remain stable through verification

**Reproduced in the initial V6 candidate; repaired during independent PR review.**
The initial verifier cached `evidence.json` before replay and could return those
cached bytes after the completion file was changed during the long operation.
Pinning the directory alone did not detect that child-file change.

The final verifier securely re-snapshots the completion file after replay with
its original admitted byte bound and requires its size/hash record to remain
identical. Four regressions cover changed bytes, removal, symlink replacement,
and introduction of a hardlink alias. Each is rejected rather than returning
successful verification of stale completion evidence. This is a final observed
snapshot boundary, not a promise that an external writer cannot later alter a
previously verified directory.

Sources: [V6 verifier](../../src/transformation_portal/lux_depth_v6/evidence.py)
and `test_completion_must_remain_bound_after_semantic_replay` in the
[pipeline regressions](../../tests/lux_depth_v6/test_pipeline.py).

## Implemented V6 processing boundary

1. Bound the V5 input inventory, byte/pixel/memory limits, canonical plan and
   completion records; verify its semantic products before admission.
2. Load retained original pixels, native depth, sky/support evidence, and the
   exact proxy; reconstruct conservative depth finishing with the upstream
   strength/clarity settings and the versioned guidance/alpha restrictions.
3. Apply explicit exposure, RGB-gain white balance, signed-luminance contrast,
   and Oklab chroma controls to an extended float32 linear-sRGB master. Defaults
   preserve its pixel bits; nonopaque RGB is protected.
4. Render a separate bounded SDR product. The default `perceptual_srgb` recipe
   compresses a lightness measure and contracts Oklab chroma at fixed lightness
   and hue using a fixed-iteration gamut search. TIFF/PNG encoding does not
   replace the retained extended master.
5. Bind processing source/dependency identities and replay expected products
   before writing successful completion evidence. Retain the V5 parent for
   independent V6 verification. Standalone V6 authority does not authorize a
   new model run or managed job.

Sources: [source.py](../../src/transformation_portal/lux_depth_v6/source.py),
[reconstruction.py](../../src/transformation_portal/lux_depth_v6/reconstruction.py),
[color.py](../../src/transformation_portal/lux_depth_v6/color.py),
[plan.py](../../src/transformation_portal/lux_depth_v6/plan.py),
[pipeline.py](../../src/transformation_portal/lux_depth_v6/pipeline.py), and
[evidence.py](../../src/transformation_portal/lux_depth_v6/evidence.py).

The renderer is a versioned local SDR recipe, not an ACES implementation or HDR
output transform. The [Oklab reference](https://bottosson.github.io/posts/oklab/)
provides the named matrices and perceptual model; preserving its numerical hue
does not establish universal observer, HDR, or photographic appearance fidelity.
ACES treats display gamut/luminance as explicit
[output parameters](https://docs.acescentral.com/system-components/output-transforms/parameters/)
and specifies distinct [tone mapping](https://docs.acescentral.com/system-components/output-transforms/technical-details/tone-mapping/)
and [JMh gamut compression](https://docs.acescentral.com/system-components/output-transforms/technical-details/gamut-compression/).
Those references motivate keeping grading and rendering separate; they do not
certify this local recipe as equivalent.

A provisional `soft_srgb` RGB-neutral-axis renderer showed substantial Oklab hue
drift in an independent positive-color stress probe: at 2× exposure its measured
95th-percentile hue difference was 11.04 degrees and maximum 15.67 degrees; at
4× the figures were 14.31 and 16.87 degrees. The comparison excluded nearly
neutral results, where hue is undefined. This was decision evidence for adding
the constant-Oklab-hue default, not a defect attributed to that final default.
`soft_srgb` and direct `clip_srgb` remain explicit comparison modes.

## Acceptance matrix and remaining work

| Surface | Implemented evidence boundary | Required before promotion |
| --- | --- | --- |
| Source fidelity | Float master, alpha, ICC identity and internal source bindings; repaired PNG color-key alpha | Retained high-bit-depth/RAW originals, gray/color charts, camera and illuminant diversity, independently reviewed transforms |
| Depth reconstruction | Versioned outlier abstention, native preservation, sky/support masks, calibrated-derivative semantics | Held-out registered/surveyed depth, thin structures, occlusions, sloped planes, reflective/transparent scenes, crop/border coverage |
| Native inference | Existing V5 device/precision/runtime recipe remains authoritative | Repeated uncached CPU/MPS runs, declared numeric tolerances, FP32/FP16 and 518/1008 paired quality/performance comparisons |
| Alpha | Any nonopaque source disables V6 depth finishing; grading protects nonopaque RGB | Versioned alpha-aware proxy/inference policy and hidden-RGB invariance before enabling depth edits |
| Grading | Explicit bounded controls; extended master retained; analytic neutral/chroma/finite checks | Paired expert photographic review, skin/neutral/material references, highlight texture, local contrast/halo review, approved creative targets |
| SDR rendering | Named local transform, bounded output and independently replayed products | Saturated highlight ramps, hue/chroma continuity, banding, neutral tracking, calibrated display and 16-bit export review |
| Materials | Applied spatial responses rejected at V6 admission | Frozen replayable masks/operations and protected-pixel/uncertainty acceptance |
| Evaluation | Paired linear RGB/luminance errors, in-gamut Oklab distance/chroma/hue with explicit coverage, range/headroom accounting | Predeclared thresholds, independent scene groups, held-out references, matched valid support and statistical uncertainty |
| Integrity and operation | Canonical plans, inventories, budget admission, source rechecks and semantic product replay | Canonical repository gates, adversarial mutation/cancellation/resource testing, installed-package tests and any separately authorized service deployment |

The paired [color evaluator](../../src/transformation_portal/lux_depth_v6/evaluation.py)
is an agreement instrument. Its Oklab distance is unscaled Euclidean distance,
**not CIEDE2000**. It restricts perceptual measures to mutually in-gamut SDR
pairs, reports coverage, and leaves near-neutral hue unavailable. Extended
samples remain represented in linear error and headroom summaries. Counts of
samples equal to zero or one do not independently prove clipping. No metric
automatically judges a creative style or promotes a model.

Future wide-gamut scene-referred ingest, a governed ACES/OCIO integration, camera
chromatic adaptation, RAW highlight reconstruction, calibrated depth confidence,
spatial material replay, and representative performance selection are additional
scoped work. None is implied by the V6 name, its test count, or a rendered TIFF.

## Validation record

These results apply to the candidate source in the named implementation
workspace, based on the audited commit above. They are local results, not hosted
CI, managed-service deployment, or photographic release acceptance.

| Command | Terminal result |
| --- | --- |
| `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 make test-lux-depth-v6-contract` | 949 passed; includes V6, V5, V4, MaterialsV4, and shared evidence/plan contracts |
| `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 make test-lux-depth-v5-managed-contract` | 252 passed; offline managed regression lane |
| `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/pytest tests/lux_depth_v6 tests/structural -q` | 161 passed |
| `PATH=/private/tmp/tp-node22-npm11-bin:$PATH TP_LINT_PYTHON=/Users/richardcheetham/Desktop/Transformation_Portal/.venv-lint/bin/python PYTHONDONTWRITEBYTECODE=1 make ci` | Passed, including fast tests, orchestrator contracts, and frontdoor contracts; orchestrator lane reported 1498 passed and 57 skipped |
| `PATH=/private/tmp/tp-node22-npm11-bin:$PATH PYTHONDONTWRITEBYTECODE=1 make validate-ci` | Passed |
| `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/mypy --config-file=mypy.ini src/transformation_portal/lux_depth_v6 src/transformation_portal/lux_depth_v4/photography.py src/transformation_portal/lux_depth_v5/photography.py` | Passed, 12 source files |
| `PYTHONDONTWRITEBYTECODE=1 make check-documentation-catalog test-documentation-contract check-doc-heading-links` | 279 tests passed; catalog and heading links passed |
| `PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python scripts/governance/check_docs_structure.py --all` | Passed, 956 files scanned |

The skipped tests do not establish acceptance for the optional services they
exclude. Dedicated Postgres/Redis/S3 deployment gates were not run for this
standalone candidate. The final CI log is
`/private/tmp/lux-v6-pr-ci.log`; focused and combined successor/managed logs are
`/private/tmp/lux-v6-pr-focused.log` and
`/private/tmp/lux-v6-pr-contract.log`.

Initial sandbox restrictions on process inspection and native package build
were environment failures; the same supported paths passed with the required
host access. A new CLI serialization-guard violation was repaired to use the
repository canonical JSON writer before the successful CI run. The obsolete
historical ramp remains rejected as documented above.
The fresh PR worktree's initial frontend build rejected an external
`node_modules` symlink. Installing the pinned dependencies inside the worktree
resolved that environment issue; the subsequent complete `make ci` passed.

The `0.7.0` wheel was built with `python -m build --wheel --no-isolation`,
installed with `pip install --no-deps --no-compile --target` into the isolated
`/private/tmp/lux-v6-pr-validation/installed-wheel` directory, and exercised from
`/private/tmp`. All 12 processing-identity modules resolved from that installed
target. The installed `lux-depth-v6` command passed help, read-only planning on
the real parent, and independent verification of the completed V6 image. Its
plan digest matched the source checkout exactly. Planning created no output.
This proves the tested native macOS/Python 3.12 wheel with the available
dependencies, not an untested cross-platform binary closure. The exact commands,
wheel hash, module paths, and terminal results are in
`/private/tmp/lux-v6-pr-validation/wheel-validation.json` and
`check_installed_wheel.py`.

### Full-resolution photographic replay

The following commands completed successfully against the retained, verified
4472×6708 parent. This is fresh V6 reconstruction, rendering, encoding, and
semantic replay; the upstream V5 model inference is retained evidence.

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python \
  -m transformation_portal.lux_depth_v6 \
  --input-dir /private/tmp/lux-depth-v5-pr-review/native/photo-1008 \
  --output-dir /private/tmp/lux-v6-pr-validation/photo-1008-neutral

PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python \
  -m transformation_portal.lux_depth_v6 \
  --input-dir /private/tmp/lux-depth-v5-pr-review/native/photo-1008 \
  --output-dir /private/tmp/lux-v6-pr-validation/photo-1008-neutral --verify \
  --expected-plan-sha256 87c1727078a398a4406e805a38822a0e9efed466e3b65a0be66e6cf41dd93032
```

The neutral grade retained extended master values from -0.000550399 to
1.062902689: 71,356 channel samples below zero and 120,461 above one. There were
182,757 pixels outside the SDR cube, or 0.60923% of this image. These values are
retained before the separate display transform; they are not evidence of sensor
highlight recovery. Independent array comparison found zero different V5/V6
master pixels on this scene and exact bitwise identity between the V6 baseline
and neutral graded master. The versioned outlier correction therefore does not
change this photograph; its corrective effect is established by the separate
analytic regression. The V6 display is finite within `[0, 1]` and differs from
the graded master at 229,318 pixels.

Full-frame previews and source/V5/V6 comparisons were inspected, including the
fountain detail crop. Additional native-resolution fountain, railing, and lamp
crop sheets are retained under `/private/tmp/lux-v6-validation/` with
`photographic-comparison.png` and `numerical-visual-report.json`. This single
8-bit-derived photograph supports a smoke inspection only. The other inspected
local photographs include Adobe RGB profiles; relabeling them as sRGB would not
provide valid high-precision color-fidelity evidence.

After the completion-stability repair, fresh execution and independent replay
passed under the new plan digest above. All six per-image products, including
the float arrays, TIFF, PNG, and photograph receipt, were byte-identical to the
initial candidate's products. The earlier numerical measurements and visual
comparisons therefore still describe these image bytes. New plan/completion
identity is tracked separately in
`/private/tmp/lux-v6-pr-validation/native-validation.json`; per-product hashes
and equivalence are recorded in `product-equivalence.json` in that directory.

### Saturated-color rendering probe

A fixed 2,400-sample positive-color probe measured Oklab hue before and after
SDR rendering. The fixture uses NumPy generator seed `991`, a 40×60 RGB array
sampled uniformly from `[0.01, 1]`, normalized by each pixel's maximum channel.
It excludes input chroma at or below 0.01 and output chroma at or
below 0.001, where hue becomes weak or undefined. At 2× exposure,
`perceptual_srgb` retained 2,388 eligible pairs with a 95th-percentile hue
difference of 0.000191 degrees and a maximum of 0.000389 degrees. At 4×,
2,390 pairs gave 0.000532 and 0.001085 degrees respectively. The same probe's
`soft_srgb` comparison retained fewer eligible pairs and produced
95th-percentile differences of 9.23 and 12.83 degrees respectively. Eligibility
differs because desaturation can remove a pair from hue measurement.

This checks the named-space constant-hue arithmetic on those samples. It is not
a calibrated observer study, a test of negative-RGB appearance, a CIEDE2000
score, or proof of world-class grading. The report retains total and eligible
counts so loss of chroma cannot silently improve the measured hue result.
The exact fixture, versions, source/data hashes, and original measurements are
persisted in `numerical-visual-report.json`; running
`/private/tmp/lux-v6-validation/reproduce_numeric_report.py` independently
reproduced all three master-range groups and all eight hue-result entries.
