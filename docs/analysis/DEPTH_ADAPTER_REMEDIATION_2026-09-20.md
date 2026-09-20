# Depth adapter remediation before LuxDepthV5 push — 2026-09-20

This follow-up repairs the unresolved adapter defects identified in the
[depth forensic analysis](DEPTH_FORENSIC_AND_LUX_DEPTH_V5_SUCCESSOR_2026-09-20.md).
LuxDepthV5 retains its governed DA3 Metric baseline and opt-in interface. These
repairs do not promote a replacement model or establish physical depth accuracy.

## Correctness changes

| Finding | Repair and regression evidence |
| --- | --- |
| Validated Depth Pro checkpoint differed from the loader's default | Copy the upstream configuration and pass the resolved validated checkpoint to the loader; clear a cached pre-load hash before validation. Tests assert the actual loader argument and unchanged upstream defaults. |
| Depth Pro focal information disappeared | Preserve returned focal length, horizontal FOV, and estimated/supplied/unavailable provenance. An explicit stage `focal_length_px` artifact is validated, passed to inference, and bound into its cache key. |
| DA2 used an 8-bit display image as depth | Both Transformers consumers use float `predicted_depth`. Tests retain 1,024 distinct levels, preserve singleton spatial axes, and reject missing/multiple predictions. |
| Callable manual DA2 model received a PIL image | Select the manual path by its processor, then pass processed tensors and postprocess float depth to the source image grid. CPU and MPS dispatch, unequal source/model grids, and singleton spatial axes are exercised with deterministic doubles. |
| Direct uint16 input wrapped modulo 256 | Scale the full unsigned 16-bit range to 8-bit with nearest rounding, including big-endian arrays. Preserve uint8 values, including images containing only 0 and 1. |
| Relative maps were multiplied by 10 and called meters | Metric fusion uses only metric contributors; legacy execution reports excluded relative models, while canonical execution rejects incompatible planned membership. No scale calibration is invented. |
| Different native grids could not be stacked | Restore eligible full-frame maps to the original image grid with float bilinear sampling. Reject incompatible original geometry and invalid metric samples. |
| Zero-weight synthetic outputs still biased real predictions | Compute statistics, weight normalization, fusion, and primary metadata from genuine contributors only. Tests add 100 m and 1e30 synthetic outliers and require byte-identical real output and statistics. |
| Corrected code could reuse old outputs | Rotate the DA2, Depth Pro, and ensemble legacy cache recipes; give the legacy DA2 pipeline a new recipe namespace. A disk-cache regression proves old entries are bypassed, preserved, and replaced by reusable new entries. |

The Depth Pro API was checked against the installed pinned Apple source revision
`9efe5c1def37a26c5367a71df664b18e1306c708`. Backend and worker metadata already
consume stage focal fields, so they now retain the returned estimate. Supplied
intrinsics transport through the backend/CLI remains unsupported; this patch
does not add implicit EXIF inference or an unbound execution-plan input. See the
[Depth Pro guide](../depth_pipeline/DEPTH_PRO_QUICKSTART.md).

All-relative ensembles retain the existing percentile normalization and remain
relative. Their agreement statistics do not become calibrated accuracy
probabilities. Synthetic/fallback predictions must have no influence on genuine
contributor statistics, weights, or output, even when retained for diagnostics.

The combined validation also exposed a pre-existing test-order failure: a fake
ML-stack import replaced the live module and parent-package binding permanently.
The test helper now restores both through `monkeypatch`, preserving enum and
class identity for subsequent tests. Product runtime behavior is unchanged by
this test-isolation repair.

## Validation and remaining acceptance

Validation logs, frozen native results, and visual comparisons are retained
outside Git in `/private/tmp/depth-remediation-20260920`.

| Gate | Terminal result |
| --- | --- |
| Combined adapter/cache/worker tests, command below | 640 passed, 2 skipped |
| `PYTHONDONTWRITEBYTECODE=1 make test-lux-depth-v5-contract` | 751 passed with native process supervision |
| Final DA2 guard and V5 fixture regression run | 83 passed |
| `PATH=/private/tmp/tp-node22-npm11-bin:$PATH PYTHONDONTWRITEBYTECODE=1 make ci` | Passed, including the frontdoor build |
| `make pre-commit`, with repo venv and Node 22 on PATH | Passed |
| Mypy on DA2/Depth Pro/ensemble backends, V3 inference, and Depth Pro stage | Passed, 5 source files |

Combined adapter command from the repository root:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest \
  tests/test_ml_dependency_health.py tests/test_da2_inference.py \
  tests/test_da2_backend.py tests/test_model_lock.py \
  tests/test_inference_api_simplified.py \
  tests/lux_depth_v3/test_model_registry_resolution.py \
  tests/lux_depth_v3/test_depth_pro_runtime_contract.py \
  tests/depth/backends tests/unit/depth tests/test_cache_validation.py \
  tests/test_depth_postprocessing.py tests/test_phase3_optimizations.py -q
```

The two skips are an existing mock-import test and unavailable optional Numba.
An additional advisory type scan of legacy `depth/pipeline.py`,
`depth/utils/cache.py`, and `depth/models/depth_anything_v2.py` still reports
62 existing errors, compared with 67 on the unmodified baseline. A shadow-file
comparison found no added diagnostic. That broader scan is not green; its
existing typing debt is separate from the passing contract surfaces and
canonical CI. Evidence is in `mypy-comparison.json` and the corresponding logs.

The initial sandboxed contract run failed nine process-supervision tests on
macOS process-table permissions; its native rerun passed all 751. The first
MPS preflight likewise required native Metal access. Initial lint errors from
NumPy inference in two V5 tests were corrected by explicit array conversion and
a tuple shape, preserving their assertions. No supervision, model, or evidence
guard was weakened.

## Photographic resolution evidence

Fresh serialized governed DA3 Metric FP32 MPS runs processed the same retained
6,708×4,472 photograph at strength 0.25 and clarity 0.1, each with independent
verification of all 17 artifacts. The source is an 8-bit-derived TIFF, distinct
from the earlier high-bit-depth ramp validation.

| Target | Native grid | In-frame samples | Protected master pixels | Changed protected pixels | Execution / including verification |
| --- | --- | --- | --- | --- | --- |
| 518 | 350×518, including padding | 178,710 | 4,258,942 | 0 | 75.73 / 84.33 seconds |
| 1008 | 672×1008, no padding | 677,376 | 4,267,910 | 0 | 77.32 / 86.54 seconds |

The 518 native NPY bytes exactly match the pre-repair photographic run. At 1008,
the inspected railing crop resolves several straight bars that are largely
absent at 518, and fountain boundaries are more distinct. Fine railing ornament
and small background structure remain incomplete. The contact sheet applies
the same display contrast range to both depth crops in each row; it does not
modify the stored arrays. Full-frame comparisons retain the normalized range.

This supports documenting `--target-size 1008` as an opt-in comparison setting.
It does not justify changing the 518 default, claiming measured geometric
accuracy, or treating the single-run timings as performance acceptance. Files:
`native/resolution-results.json`, `quality/protected-pixel-checks.json`, and
`quality/resolution-review.png` under the evidence root above.

Adapter regressions use controlled runtime doubles, not model downloads or
ground-truth scene depth. Passing them establishes the corrected data path,
cache behavior, and failure contracts. It does not establish native DA2/Depth
Pro quality, commercial model-policy eligibility, or new physical calibration.

The retained photographic resolution probe has no measured depth reference and
only one scene. It cannot establish production-quality superiority, uncertainty
calibration, or a repeatable performance baseline. Held-out scene references,
representative RAW/alpha/material corpora, and service-backed publication remain
separate release-acceptance gates described in the
[V5 validation report](LUX_DEPTH_V5_IMPLEMENTATION_VALIDATION_2026-09-20.md).
