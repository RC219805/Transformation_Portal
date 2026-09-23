# LuxDepthV6 depth inference and reconstruction forensics

**Audit date:** 2026-09-23.
**Audited baseline:** `41a626ccf991e06a1218b08670f6f137c616b6fb`.
**Implementation workspace:** `/private/tmp/tp-lux-v6-depth-20260923`.
**Fresh PR checkout:** `/private/tmp/tp-lux-v6-managed-pr-20260923`, based on the
same clean `origin/main` baseline.
**Acceptance:** synthetic measurements, source inspection, one retained
photograph's reconstruction/replay compatibility, and one fresh native managed
composite execution plus a final native HTTP/external-worker publication trial.
Physical model accuracy,
representative photographic quality, and production acceptance remain
unestablished. V3 remains the production baseline.

## Scope and evidence authority

The audit follows canonical image decoding, RGB model-proxy construction,
governed DA3 inference, native evidence, and V6 depth reconstruction. Standalone
V6 consumes an independently verified V5 bundle. Managed V6 freezes original
photograph inference and finishing in one admitted composite, retaining that
V5 bundle within the generation. Neither path obtains additional model detail
by creating a larger derivative raster.

Evidence labels distinguish **reproduced** numerical behavior, **source-proven**
implementation behavior, and **unmeasured** physical or photographic accuracy.
Synthetic references below are exact fixture definitions, not camera captures,
surveyed depth, RGB-edge ground truth, or model benchmark results. The initial
synthetic and retained-source audit did not load a model. The later native
smoke below executed the pinned real model using its existing local checkpoint;
that single-scene result is recorded separately.

## F1: Disconnected native noise can pass the previous persistence check

**Reproduced; versioned reconstruction repair.** The previous V6
`guided_bilinear_v3` recipe requires both native depth groups to contain at least
four samples in a complete 4×4 neighborhood, including two samples outside the
central 2×2 footprint. Counts alone do not establish connected surfaces.

A 28×28 alternating depth raster containing values 1 and 2 satisfies those
counts despite each same-depth sample being disconnected from its orthogonal
neighbors. Expanding the corresponding RGB texture fourfold into a 112×112
linear-RGB master, with intensities 0.15 and 0.60, permits the previous RGB
guidance to select false surfaces. Against the deliberately constant relative
depth reference 0.5, its RMSE increases from **0.168526786 to 0.343975496** and
5,000 master pixels are refined. This is injected native noise; the experiment
does not imply that DA3 normally produces checkerboards.

The new `guided_bilinear_v4` recipe additionally requires each central native
sample to belong to a four-connected same-depth component with at least four
samples, including at least two in the outer ring. Connectivity is checked
within the complete native neighborhood. It rejects this disconnected noise
and produces **bitwise-identical bilinear depth**, with zero refined pixels.
It retains supported straight-step refinement in the separate step fixture.
This is conservative rejection of unsupported sharpening; the remaining
bilinear error is not removed and no missing geometry is recovered.

Source: [versioned alignment](../../src/transformation_portal/lux_depth_v5/photography.py)
and [V6 reconstruction](../../src/transformation_portal/lux_depth_v6/reconstruction.py).
Historical V5 and V6 recipes retain their distinct identities; new recipe
selection must not reinterpret a previously frozen plan.

## F2: Full-size raster dimensions do not establish inference resolution

**Source-proven; explicitly preserved limitation.**
[Proxy construction](../../src/transformation_portal/lux_depth_v4/photography.py)
downsamples the canonical linear-sRGB image, encodes an RGB8 sRGB proxy, and
adds right/bottom edge padding to multiples of 14. Padding is excluded from
surface support and relative-depth normalization.

For a 6000×3375 source, dimensions in width×height order are:

| Requested long edge | Unpadded image support | Padded model raster |
| --- | --- | --- |
| 518 | 518×291 | 518×294 |
| 1008 | 1008×567 | 1008×574 |

A 6000×3375 reconstructed map derived from the first row still depends on
518×291 native image support. Target 1008 requires new governed inference,
distinct prepared configuration and cache identity, and independent comparison
evidence. This change does not silently raise the native inference default or
describe interpolation as additional measured detail.

The installed, pinned DA3 `InputProcessor` was exercised directly at 28×14,
56×42, 518×294, and 1008×574 with deterministic RGB8 input. Its output was
exactly equal to the input converted to float32, divided by 255, and ImageNet
normalized: there was **no additional spatial resizing**. This processor-only
probe did not instantiate the model.

## F3: Existing precision, sky, and geometry protections are retained

**Source-proven and focused tests passed; no new inference defect demonstrated.**
The [V5 worker](../../src/transformation_portal/lux_depth_v5/worker.py) checks
the pinned source/API contract, float32 weights, preprocessing tensor geometry,
and the returned native grid. Explicit `fp32` disables outer autocast, while
`fp16` requests mixed precision with the upstream float32 depth head. Storage
remains float32. The worker seeds inference after lazy model initialization.

The worker uses the raw model sky threshold **≥0.3**, matching the region
affected by upstream monocular sky substitution; the upstream convenience
prediction's ≥0.5 mask would omit part of that region. The
[depth evidence contract](../../src/transformation_portal/core/depth_evidence.py)
separates finite-positive numeric validity, in-frame support, known sky, and
usable surface validity. Missing sky evidence abstains from surface use.

The 52 focused backend/evidence tests passed. A separate bounded synthetic
probe set sky depths to 1,000,000 and changed padded depths to 3,000,000;
usable aligned relative depth remained unchanged and invalid output samples
retained their zero sentinel. The committed harness includes a constant plane
with three padded rows set to 1,000; all recipes retain zero relative error.

These checks justify preserving the governed inference recipe. They do not
establish cross-device bitwise determinism, model accuracy, or quality on
untested photographs. A zero-bug guarantee is not supported by this evidence.

## F4: ICC admission failures occur before depth inference

**Source-proven; existing admission boundary.** The
[photographic decoder](../../src/transformation_portal/lux_depth_v4/photography.py)
accepts its bounded sRGB/linear-sRGB contract and fails closed on unsupported
ICC profiles in automatic mode. `input_color="srgb"` declares the pixel
interpretation; it does not convert Adobe RGB samples through their profile.
An Adobe RGB source must undergo an actual color-managed conversion before
being submitted as sRGB. This reconstruction change does not perform that
conversion or recover a failed source ingest.

Likewise, hidden RGB in a nonopaque image can affect the existing alpha-unaware
model proxy. V6 continues to abstain from depth finishing for any nonopaque
source; this audit does not claim alpha-aware model inference.

## Opt-in V6 depth-map integration

Independent integration review reproduced a second defect in verification:
caller-supplied tighter memory limits constrained parent admission but did not
constrain the later reconstruction/replay reservation. Verification now applies
the minimum of caller and frozen limits to that reservation before loading the
parent. Both V1 and V2 regression cases reject an insufficient 128 MiB budget;
the frozen plan and source identity remain unchanged.

The successor enables this reconstruction through `--depth-maps`, represented
by `LuxDepthV6Request.depth_maps=DepthMapRecipe()` and the versioned
`tp.lux.grade.plan.v2` contract. Its default depth refinement is
`guided_bilinear_v4`; `--depth-refinement` explicitly selects `bilinear`,
`guided_bilinear_v3`, or `guided_bilinear_v4` and requires `--depth-maps`.
Without the new flag, the established standalone v1 plan remains unchanged.
Managed `lux-depth-v6` jobs always include depth products and use the new
composite `tp.execution.plan.v5` authority. V3 remains the production default.

[Depth-map publication](../../src/transformation_portal/lux_depth_v6/depth_maps.py)
exports the unchanged native arrays and separate validity/support/sky evidence,
master-grid relative float32 arrays and TIFF, a bounded 16-bit grayscale PNG
with a separate validity PNG, and source-bound JSON metadata. At most 13
products are emitted per image, including the optional calibrated-meter array.
Meter values require retained camera calibration and remain inferred values,
not measured scene distances. Zero is both a possible valid near-depth value
and the invalid sentinel; consumers must use the associated validity mask.

One reconstructed alignment feeds both photographic depth finishing and the
published maps. The descriptor records native padded/unpadded dimensions,
reconstructed master dimensions, exact reconstruction identity, calibration,
and the absence of any new native-detail claim. A 16-bit scalar preview does
not add inference precision or replace the float arrays as numeric authority.

## Managed portal, worker, and publication integration

The opt-in `lux-depth-v6` pipeline accepts original photographs through the
existing configuration-preview and job routes. A separate
`TP_LUX_V6_MANAGED_ENABLED` flag controls API/worker admission independently of
V5 activation. The closed request freezes exposure, RGB gains, contrast, pivot,
saturation, SDR render/shoulder, depth refinement, and the bounded V5 inference
controls. Materials inputs and physical runtime/cache overrides are rejected.
Current preview/readiness, server path allowlists, tenant authorization, and
Postgres/Redis dispatch remain required.

`tp.execution.plan.v5` embeds the exact V5 inference plan and freezes finishing
before inference. `tp.job.photography.bindings.v1` retains physical input/runtime
bindings under migration `0007_photography_bindings`. API, workers, and packaged
schema must be deployed together. The generation retains complete `source-v5/`
and `v6/` namespaces, an outer plan, and a complete outer evidence record.
Publication replays finishing, verifies retained evidence, and checks expected
hashes under the dispatch fence. Review associates the photographic TIFF with
its same-input draft PNG; scalar depth and retained evidence have separate roles.

The total output limit covers both retained stages. It is clamped to publisher
policy, reserves 17 MiB for outer records, and splits the remainder equally
between inference output and V6 output. V6's source limit equals the retained
inference budget. Pixel admission is capped at 100 million, and finishing
rechecks the measured geometry against its conservative memory/output bounds.
This preserves server publication ceilings without treating a low byte budget
as permission to omit provenance or depth products.

Integration audit reproduced a completion-rebinding defect: replacing the TIFF
and its inner/outer inventory immediately after successful semantic replay
could otherwise admit unverified bytes. The managed collector now compares the
later completion snapshot with the exact verified completion carrier before
using its inventory. The execution-return boundary receives the same binding,
and regression tests exercise both replacement windows. Managed publication
also avoids repeating complete V5 source verification when the admitted source
limits have already been proven identical.

A further admission regression allowed whole-number floats in outer resource
and publication fields because JSON Schema treats them as integers and Python
numeric equality accepts them against inner integer fields. Finishing then
rejected the value after inference. The outer parser now requires exact integer
types before any execution; 20 adversarial float/boolean cases cover all ten
fields. A separate portal correction keeps ready V5/V6 previews from showing
the archive dispatch instruction.

Independent PR review found that the shared `parse_execution_plan` entry
point still dispatched only V1 through V4. Managed worker dispatch already
accepted V5, but other core callers rejected its bytes. A lazy V5 dispatch
branch now preserves canonical bytes through both byte and text entry points;
regression coverage also retains the earlier plan-version behavior.

The [operator guide](../reference/LUX_DEPTH_V6.md#managed-portal-and-api-execution)
documents flags, strict request controls, startup, output paths, and distinct
standalone versus managed resource semantics. These source contracts do not
alone prove a live deployment or production photographic acceptance.

## Deterministic measurement harness

[The executable harness](../../scripts/analysis/audit_lux_depth_v6_reconstruction.py)
requires the ordinary core Python environment and imports no model runtime.
From the repository root:

```bash
./.venv/bin/python scripts/analysis/audit_lux_depth_v6_reconstruction.py \
  --output /private/tmp/tp-lux-v6-depth-forensics-20260923.json
./.venv/bin/python -m pytest tests/test_audit_lux_depth_v6_reconstruction.py \
  -q --override-ini addopts=''
./.venv/bin/python -m pytest tests/lux_depth_v5/test_backend.py \
  tests/core/test_depth_evidence.py -q --override-ini addopts=''
```

The isolated checkout was exercised with the existing Desktop
`.venv/bin/python` and its own source import root; no environment installation
was required. The disposable JSON report is not a tracked generated artifact.
The report carries native before/after hashes, reference and validity hashes,
float32 storage, source/unpadded/padded geometry, and explicit synthetic and
unestablished-acceptance labels. No timestamps or random sampling enter it.

Measured MAE/RMSE in **relative-depth units, not meters**:

| Synthetic case | Bilinear | Guided v3 | Guided v4 |
| --- | --- | --- | --- |
| Constant plane with contaminated padding | 0 / 0 | 0 / 0 | 0 / 0 |
| Affine ramp with unrelated RGB edge | 0.000000026 / 0.000000038 | same | same |
| Supported straight depth/RGB step | 0.008928571 / 0.052822141 | 0.004942602 / 0.039300930 | same as v3 |
| Perpendicular RGB texture/depth boundaries | 0.008928571 / 0.052822141 | same | same |
| Checkerboard native noise, constant reference | 0.134088010 / 0.168526786 | 0.283561862 / 0.343975496 | same as bilinear |
| Isolated native outlier, local region | 0.040000000 / 0.131250000 | same | same |

All recipes preserved each fixture's native depth SHA-256. The outlier score
uses rows 32:52, columns 32:52; the other cases score the complete valid master.
`refined_master_pixels` covers the full master, including the legitimate far
plane in the outlier fixture. `differs_from_bilinear_in_score_region` is bounded
to the scoring region.

The report also counts horizontal/vertical adjacent-pair depth differences
of at least 0.1, matching exact pair locations with no tolerance. Those counts
are diagnostic threshold measurements, not certified scene boundaries. For
the checkerboard's constant reference, all detected edges are false; guided
v3 produces 9,612 such pairs, while bilinear and v4 produce 9,412. The v4 repair
prevents extra amplification and leaves the original noisy evidence intact.

The harness's 11 focused tests passed, including analytical metric checks,
invalid-endpoint exclusion, malformed raster rejection, deterministic reports,
native preservation, the non-vacuous checkerboard regression, and direct CLI
execution from outside the repository. These are bounded numerical contracts;
representative calibrated photography and physical depth accuracy remain
separate acceptance work. The later native smoke is independent of these
synthetic measurement contracts.

## Remaining accuracy and resolution acceptance

### Local validation

The retained-depth phase, before managed integration, passed **1,040 tests**:

```bash
make test-lux-depth-v6-contract \
  PY=/Users/richardcheetham/Desktop/Transformation_Portal/.venv/bin/python
```

The first sandboxed attempt was blocked by macOS process inspection in nine
existing subprocess-watchdog tests. The complete gate passed with process
inspection permitted; no watchdog contract was weakened. Additional green
checks were `make ci-quick`, `make test-fast` (77 tests), documentation contracts
(279 tests), documentation catalog and heading links, V6 mypy, formatting,
pre-commit hooks, and `git diff --check`. Local commands used the supported
Desktop environment on `PATH` and `PYTHONPATH=src:.` where necessary; the
isolated worktree did not install or upgrade the application environment.

The worktree cleanliness gate reports the intended uncommitted implementation
and documentation files. Generated experiment outputs remain outside the
repository. Those historical counts describe the retained-depth phase, not the
later expanded managed suite. They do not establish hosted CI, live service
dispatch, or production acceptance. Current managed validation commands are
`make test-lux-depth-v6-managed-contract` and
`make test-lux-depth-v6-managed-services`; the service lane requires explicitly
configured test Postgres/Redis and uses controlled inference.

### Retained real-scene replay

The retained `DJI_20251119163919_0092_D.tif` V5 generation was independently
verified and processed through depth-enabled V6, followed by a separate CLI
verification of the exact plan digest. The original generation's undeclared
Finder `.DS_Store` caused inventory rejection. A disposable copy containing
only hash-verified declared artifacts plus the unchanged completion record was
used; the original generation was not modified and the verifier was not relaxed.

The real-scene result preserved native depth bytes and produced a 4032×3024
map from 518×388 native image support (518×392 including padding). Surface
validity covered 12,191,712 of 12,192,768 master pixels (99.991339%); 1,056
pixels remained invalid. The stricter reconstruction matched the retained V5
relative-depth raster exactly: that scene's existing selections already met
the additional restrictions. This demonstrates compatibility, not improved
physical accuracy or additional inferred detail. Camera calibration and metric
depth were unavailable. This retained-source comparison performed no fresh
model inference or reference-depth comparison; the separate fresh smoke follows.

Disposable local evidence:

- V6 output: `/private/tmp/tp-lux-v6-depth-real-20260923`
- Retained verified parent copy: `/private/tmp/tp-v6-retained-parent-20260923`
- Canonical V6 plan SHA-256: `89ff110dc8be0ed8c253e80abfcfe6a2d01d448ab2c92131d2e26177d64b50dd`
- Visual comparison: `/private/tmp/tp-v6-depth-real-comparison.png`
- Full-grid numeric comparison: `/private/tmp/tp-v6-depth-real-comparison.json`

The visual comparison shows photograph, V5/V6 depth on the same scale, validity,
and a signed difference raster. It displays every fourth sample for readability;
reported numerical measurements cover the complete original grids.

### Fresh native managed composite smoke

**Executed and independently verified; one scene.** The managed composite
accepted the original canonical TIFF `DJI_20251119163919_0092_D.tif`, whose
73,196,052 source bytes match the retained V5 input digest:
`c56d4249808239110ff1ca4a9db1f3c815b98b05e51c65ee6ee0d6d5b5deb0d8`.
The source was interpreted with explicit `input_color="srgb"`; this assumption
is not a measured color-calibration result.

The run executed `depth-anything/DA3METRIC-LARGE` at pinned revision
`4010e39f3634a45bc60553321fb49fb760bd594e` on MPS with explicit FP32 and target
518. The existing governed DA3 runtime and local checkpoint were used without
a model download. Finishing used neutral grading, `perceptual_srgb` with
shoulder 0.8, and `guided_bilinear_v4` depth reconstruction.

The report records **85.7247 seconds execution**, then **8.0073 seconds independent
verification**, **40 artifacts**, and **1,328,028,726 total bytes**. Inspection
confirmed 4032×3024 RGB uint16 photographic TIFF with embedded sRGB ICC,
1600×1200 RGB draft PNG with ICC, 4032×3024 float32 relative-depth TIFF without
ICC, 1600×1200 grayscale 16-bit depth preview with its validity mask, and
392×518 native float32 depth. Meter output was absent because camera
calibration was unavailable.

Disposable evidence:

- Output: `/private/tmp/tp-v6-managed-native-20260923/generation`
- Run report: `/private/tmp/tp-v6-managed-native-20260923/smoke-report.json`
- Artifact inspection: `/private/tmp/tp-v6-managed-native-20260923/artifact-inspection.json`
- Exact composite plan SHA-256: `5cef442edec6db98c6ffb34f10b999165cdf9552749dcdad64a9c95de26fd84c`

This proves one local raw-photograph-to-complete-V6 execution and independent
semantic verification at the recorded processing identity. It does not prove
a hosted frontend deployment, the live Postgres/Redis delivery path, performance
across scenes/devices, camera color accuracy, or physical depth accuracy. The
run report explicitly retains `physical_accuracy="not_established"`. Processing
source changes require a new plan; later source versions cannot silently
reinterpret this completed generation.

### Final native HTTP and external-worker delivery

After the exact-integer and shared-parser fixes, the fresh PR checkout exercised the real
`POST /v1/config-preview` and `POST /v1/jobs` routes, a dedicated disposable
Postgres test database, a unique Redis namespace, and the external worker.
It ran the same canonical TIFF with explicit sRGB, MPS FP32, and target 518.
No controlled inference fixture was used: execution evidence records
`synthetic=false`, `pytorch_mps`, zero cache hits, and one cache miss.

The job succeeded with **40 published artifacts / 1,328,055,188 bytes**.
Measured dispatch-to-download time was **104.4330 seconds**; the complete smoke
test took **105.48 seconds**. HTTP downloads verified the photographic TIFF as
4032×3024 RGB uint16 with sRGB ICC, the sRGB PNG draft, and the float32 depth
TIFF. Publication independently replayed and verified the complete generation
under the database dispatch fence. Processing identity matched the final
source, and the external worker and its Redis namespace were cleaned up.

- Job: `job_36a08b8e15d24be68b9f5de366878d97`
- Plan SHA-256: `b1ed64acb5078b4e146ffdbf3f8e11b740893656d05e332ab2586a2f9e8731aa`
- Evidence root: `/private/tmp/tp-v6-pr-native-final-20260923/test_v6_external_worker_keeps_0`
- Report: `native-service-report.json` under that root

The earlier direct composite plan predates the exact-integer source change;
it remains historical evidence at its recorded processing identity. This final
trial establishes local native API/worker/publication integration. It does not
establish hosted deployment or representative physical depth accuracy. Browser
verification separately exercised hydrated V6 controls and Review, including a
disposable fixture displaying these native image bytes; its job lifecycle was
simulated. The user's existing live Desktop app was not replaced.

### Final local validation

Commands ran from the isolated worktree with the existing Desktop core Python,
`PYTHONPATH=src:.`, and official Node 22.22.3 on `PATH` where needed. Final
processing identity was compared again with the native published plan after
formatting and the adapter's type-only variable rename.

| Gate | Result |
| --- | --- |
| `make test-lux-depth-v6-contract` | 1,078 passed |
| `make test-lux-depth-v6-managed-contract` | 139 passed |
| Combined V5/V6 managed service tests | 8 passed with real Postgres/Redis and controlled inference |
| Native V6 API/external-worker service smoke | 1 passed with real governed inference and HTTP downloads |
| `make ci` | Passed, including 77 fast tests, 1,566 orchestrator tests with 63 optional-service skips, 347 Node tests, and production frontend build |
| `make test-portal-contract` | 399 passed |
| Hydrated frontend UX browser suite | 58 passed; final V6 retake and native-image rendering fixture also passed |
| `make ci-quick` | Passed |
| `make pre-commit` plus explicit new/changed-file hooks | Passed; one test-file formatting correction was applied before rerun |
| `make test-documentation-contract` | 279 passed |
| Catalog, heading links, `make validate-ci`, `git diff --check` | Passed |
| Mypy on new managed plan, API, adapter, orchestration and publication modules | Passed across seven files |

`make check-worktree` is expected to report the intentional uncommitted patch;
experimental outputs are outside the repository. The initial sandbox failure
was macOS process inspection/Metal access, not a product failure; authorized
runs passed without weakened assertions. Hosted CI and a deployed live Desktop
upgrade were not performed. The change remains opt-in and reviewable. The same gates were rerun in the
fresh PR checkout; new shared-parser byte/text regressions account for the
final successor count. The new native trial used a separate test database,
unique Redis namespace, fresh inference cache, and explicit import-root checks.

### Independent accuracy acceptance

The next acceptance experiment requires representative, independent scenes
with registered depth or surveyed distance references, recorded camera
calibration, and explicit reference uncertainty. Evaluate indoor edges,
diagonal boundaries, repeated texture, thin structures, glass, mirrors, sky,
and low-contrast surfaces separately. Source color conversion must be known;
unsupported ICC assignment is not a substitute for conversion.

Compare frozen native inference at 518 and 1008 using the same governed model,
source bytes, declared precision, and calibration; measure memory and latency
alongside depth error. Keep native-grid measurements separate from resampled
master-grid results. The existing
[reference-bound depth evaluator](../../src/transformation_portal/lux_depth_v5/evaluation.py)
can record metric/relative, boundary, ordinal, and independent-group evidence
when suitable references are available. Synthetic interpolation improvements
alone do not justify promoting a resolution, changing the model, claiming
physical accuracy, or promoting opt-in managed V6 to the production default.
