# LuxDepthV5 implementation and validation — 2026-09-20

LuxDepthV5 is implemented as an opt-in candidate on V4's governed execution
machinery. DA3 Metric remains the baseline. The candidate is locally validated;
model replacement, physical scene accuracy, comparative production quality, and
performance acceptance remain unestablished.

See the [operator guide](../reference/LUX_DEPTH_V5.md),
[Execution Plan V4](../reference/EXECUTION_PLAN_V4.md), and preceding
[forensic analysis](DEPTH_FORENSIC_AND_LUX_DEPTH_V5_SUCCESSOR_2026-09-20.md).

The subsequent [pre-push adapter remediation](DEPTH_ADAPTER_REMEDIATION_2026-09-20.md)
records repairs to the adjacent depth backends and fresh validation on this
candidate branch.

## Implemented behavior

- A separate `lux-depth-v5` CLI/request/prepared-execution boundary consumes
  `tp.execution.plan.v4`. V3/V4 remain independent. Shared V4 changes are private
  lifecycle, worker-transport, executor, verifier, publication, and CLI hooks.
- Native samples remain unchanged. Separate masks describe finite-positive
  numeric validity, in-frame support, model sky substitution, and usable surface
  validity. Unknown sky evidence causes abstention. Normalization excludes
  padding, invalid samples, and sky. The metric baseline has no contracted
  accuracy-confidence output.
- FP32 and FP16 policies control the actual pinned DA3 forward path. FP16 is
  explicitly mixed precision: weights, depth head, and stored output remain
  float32. Receipts and native cache identity include the selected policy.
- Bounded guided reconstruction selects between existing surface samples only
  at supported native discontinuities with strong RGB agreement. Ambiguous
  pixels and affine/gradual native slopes retain bilinear interpolation. The
  aligned artifact is computed once and passed as a declared graph
  output, then independently reconstructed by the verifier.
- Depth exposure and clarity are bounded and attenuated by valid interpolation
  support. Unusable, protected, and nonopaque pixels are unchanged. Preview maps
  use complete valid neighborhoods and explicit neutral values.
- The `identity-v5` native cache binds proxy/source/model/runtime/device/precision
  and recipe identity while allowing downstream finishing, refinement, and
  camera-calibration changes to reuse inference. MaterialsV4 remains a separate
  explicit response. Legacy companion material masks are rejected in V5.
- Evidence V3 verification reconstructs masks, alignment, calibration, finishing,
  previews, and TIFF delivery. Rehashed semantic forgeries fail. Managed output
  retains existing publisher admission, inventory, lease, and dispatch fences.
- A bounded, reference-aware evaluator reports metric, separately fitted
  relative, ordinal, and depth-boundary measures. It never treats RGB edges as
  ground truth or automatically promotes a model.

## Proven local gates

The isolated checkout was `/private/tmp/tp-depth-successor-forensic`, branch
`codex/lux-depth-v5`, based on main `3a0744c40ee932c1a27c5b072231ea401af9ca5d`.
The Desktop checkout's unrelated `AGENTS.md` edit was preserved byte-for-byte.
Validation logs and photographic artifacts remain outside Git under
`/private/tmp/lux-depth-v5-validation`.

| Gate | Terminal result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 make test-lux-depth-v5-contract` | 751 passed |
| Legacy execution/identity/worker contracts, command below | 287 passed |
| `PATH=/private/tmp/tp-node22-npm11-bin:$PATH PYTHONDONTWRITEBYTECODE=1 make ci` | Passed, including Python/frontdoor tests and build |
| `make pre-commit`, with repo `.venv/bin` and Node 22 on PATH | Passed with all new files staged |
| `make validate-ci` | Passed |
| Mypy, 30 V4/V5/core source files | Passed |
| Installed wheel, with explicit governed model manifest | V5 entrypoint, packaged schemas, core dispatch, read-only planning passed |
| Documentation structure and heading links | Passed |
| Staged/unstaged whitespace checks | Passed |

Legacy contract command:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest \
  tests/core/test_execution_plan.py \
  tests/core/test_execution_identity_v3.py \
  tests/core/test_execution_plan_v2.py \
  tests/stage_graph/test_stage_registry.py \
  tests/lux_depth_v3/test_execution_plan_adapter.py \
  tests/lux_depth_v3/test_execution_lifecycle.py \
  tests/lux_depth_v3/test_prepared_execution_callsites.py \
  tests/depth/backends/test_execution_plan_workers.py -q
```

Static checks include the new source and tests, not merely previously tracked
V4 files. The wheel was built from a disposable copy of the current source,
installed separately, and imported outside the checkout. Model policy is an
external governance input; planning without that policy correctly failed closed.

## Fresh governed MPS execution

The lock-aligned native runtime was
`/private/tmp/tp-lux-v4-da3-runtime/.venv-da3/bin/python`. Every successful run
used the actual governed DA3 Metric model and independent V5 output verification.
No controlled test worker, synthetic depth fallback, or bypassed integrity check
was used for these runs.

| Scenario | Depth cache | Verified artifacts | Observed result |
| --- | --- | --- | --- |
| FP32 cold, 28×42 uint16 ramp | Miss | 16 | All 1,176 source levels retained with zero finishing strength |
| Different strength/clarity and bilinear reconstruction | Hit | 16 | Same native bytes; finishing recomputed |
| Added calibration and preview maps | Hit | 21 | Same native bytes; calibrated derivative and validity-aware previews verified |
| Independent uncached FP32 repetition | Miss | 16 | Native NPY bytes identical to cold run; 1,176 levels retained |
| FP16 mixed-precision run | Miss | 16 | Distinct precision identity; 1,176 levels retained |
| 6,708×4,472 photograph, FP32, strength 0.25/clarity 0.1 | Miss | 17 | Full-resolution master and 16-bit TIFF independently verified |

On the synthetic ramp, FP16 versus FP32 maximum absolute native-depth difference
was `0.0003795624`, with mean relative difference `0.0001425493`. These are
observations on one input, not accepted precision tolerances or physical errors.

The photograph contained 29,998,176 pixels and came from the retained explicit
sRGB conversion fixture with **8-bit source precision**. It is separate from the
16-bit ramp test. Inference used 345×518 real proxy content on a padded 350×518
grid. Of 178,710 in-frame samples, 25,373 were classified as sky. Depth finishing
protected 4,258,942 full-resolution pixels; direct comparison found zero changed
pixels in that protected set.

The guide refined 64,803 pixels and retained bilinear interpolation for 101,027
ambiguous candidates. Full-frame and 100% railing/fountain crops were inspected.
Photographic finishing was modest, with no observed geometric image warping.
The depth raster still lacks fine railing geometry; this run does not establish
recovery of unsupported detail. TIFF encoding clipped 0.12544% of RGB samples
above range and 0.07931% below range; the linear master retains those values.

Photograph execution took 79.52 seconds, or 88.35 seconds including independent
verification, on this host. These single-run timings include governed setup and
I/O and are **not performance acceptance**.

## Evaluated synthetic reconstruction

The frozen analytic two-surface fixture was evaluated through the new CLI with
`relative_alignment=none`. Bilinear mean absolute relative-depth error was
`0.0089285714`; guided error was `0.0044642857`. Transition width fell from four
pixels to two, retaining the correct boundary and identical native samples.
Boundary F1 was 1.0 for both at the declared one-pixel tolerance; ordinal accuracy
was also 1.0 for both. Thus the measured gain is narrower reconstruction error,
not an invented boundary-F1 improvement.

The report uses one declared independence group, so uncertainty intervals remain
unavailable. Its `production_acceptance` and promotion result remain
`not_established`. Evidence: `quality/manifest.json`, `quality/report.json`, and
the preceding probe under `/private/tmp/depth-forensic-20260920/contracts`.

## Environment failures and remaining limits

Early process tests were blocked by macOS sandbox process-table restrictions;
the complete suite passed with authorized native supervision. The isolated
frontdoor initially had the wrong ambient Node and missing native dependencies;
Node 22 plus the locked `npm ci` resolved the gate. A staged hook's index-lock
permission failure was resolved by running the standard hook with worktree
metadata access.

Early native attempts correctly failed on stale/relative-filename bytecode and
changing validation namespaces. Another repeat stopped on worker runtime drift
during parallel tooling. Absolute-path bytecode regeneration, fixed namespaces,
and serialized native validation produced the successful results above. Integrity
guards were retained; failed attempts remain recorded separately from successes.

No local gate remains failing. Held-out paired photography, measured scene depth,
uncertainty calibration, broad FP32/FP16 and CPU/MPS comparisons, representative
RAW/alpha/material native corpora, service-backed publication acceptance, and
repeatable performance evaluation remain release-acceptance work. This candidate
does not replace the existing production executor or establish a better model.


## Fresh-main review before publication

The review-ready branch `codex/lux-depth-v5-reviewed` was created from a clean
checkout of freshly fetched main `3a0744c40ee932c1a27c5b072231ea401af9ca5d`.
The preceding candidate was transferred without the Desktop checkout's unrelated
`AGENTS.md` edit. The measurements above describe the earlier candidate; fresh
review evidence is recorded separately under `/private/tmp/lux-depth-v5-pr-review`.

Independent inference, execution-contract, and depth-quality reviews identified
and repaired three additional issues:

- DA2's manual fallback now uses the processor's float-depth postprocessor to
  restore the source image grid before normalization. Unequal inference/source
  dimensions and singleton spatial axes have focused regressions.
- Guided reconstruction now requires a complete valid 4×4 native neighborhood
  supporting two locally stable raw-depth groups. Affine slopes, gradual ramps,
  narrow valid bands, and missing outer support abstain. This conservative test
  does not prove a physical discontinuity; it may leave fine details unrefined.
  The alignment recipe is `bounded_two_surface_rgb_selection_v2`.
- Boundary scoring applies identical complete 3×3 overlap support to predicted
  and reference edges. Boundaries outside evaluable neighborhoods are excluded;
  inadequate reference support is reported as unavailable. The evaluator recipe
  `reference_bound_depth_metrics_v2` is included in its recipe hash.

A cold/warm inference regression exercises the actual worker seeding and forward
path with lazy initialization that consumes Torch RNG. Both FP32 and FP16 retain
identical samples for identical pixels across cold/warm execution and different
PNG encodings, despite intervening RNG work; changed pixels change the samples.
Removing the post-load reseed makes both regressions fail.


The broad core gate also caught the missing console-entrypoint snapshot and
required additive SemVer update. Package metadata now advances from 0.5.0 to
0.6.0, and the console-script snapshot was regenerated with the governed updater.
All 27 structural tests pass after that repair. No release tag or publication is
part of this change.

Fresh focused gates pass: 764 V4/V5/material contracts; 646 adapter/cache/worker
tests with two declared skips; 287 legacy execution contracts; and mypy across
35 V4/V5/core/adapter source files. Canonical `make ci`, full pre-commit including
secrets scanning, `make validate-ci`, documentation structure, and heading links
pass. The three legacy DA2 pipeline/model/cache modules retain 62 nonblocking
mypy errors versus 67 on the main baseline, with no new diagnostics.

The revised reconstruction/evaluator CLI on the same analytic two-surface
family reports mean absolute relative error 0.00892857 for bilinear and
0.00542092 for guided, with the center transition width reduced from four pixels
to two. Stricter outer-neighborhood support intentionally leaves additional
pixels bilinear compared with the earlier recipe. One synthetic independence
group still provides neither physical-depth nor production-quality acceptance.


The fresh staged 0.6.0 wheel was built in a disposable copy and installed into an
isolated target. All 3,475 copied source files matched the staged Git blobs.
Outside the checkout, the installed entrypoint, packaged Plan V4/Evidence V3
schemas, core plan dispatch, and CLI/API canonical planning parity passed.
Planning launched no inference subprocess and created no output/cache namespace.
The wheel SHA-256 is
`d800a92f3df0c392fd01807bc00c01b75169f18f09dbf113b258ff697dbbf9ba`.
Exact packaging commands and receipts are in `logs/package-commands.txt` and
`logs/package-smoke.json` under the fresh evidence directory.


The broad core coverage command recorded 14,516 passed, 90 skipped, 943
deselected, and one failure: the console-script snapshot/version omission repaired
above. Coverage was 71.78%, exceeding its 30% gate. The full coverage run was not
repeated after that metadata-only repair; the entire structural suite passed and
canonical `make ci` passed again with final 0.6.0 metadata. No unresolved test
failure was identified. Optional dependency, checkpoint, platform, and manual
checks remain represented by their declared skips in `logs/core-suite.log`.

Fresh governed FP32 MPS execution on the retained 29,998,176-pixel, 8-bit-derived
photograph passed independent verification of 17 artifacts at each target size:

| Target | Native grid | Refined master pixels | Protected pixels | Changed protected pixels | Execution / including verification |
| --- | --- | --- | --- | --- | --- |
| 518 | 350×518, including bottom padding | 12,305 | 4,258,942 | 0 | 79.48 / 88.74 seconds |
| 1008 | 672×1008 | 14,792 | 4,267,910 | 0 | 80.91 / 91.27 seconds |

Protected-pixel equality was checked against the source master using raw float32
bit patterns. Native-depth file hashes at both targets exactly match the earlier
adapter-remediation photo runs; the refinement repair changes downstream
reconstruction, not inference samples. Full-frame and 100% railing/fountain crops
were inspected in `quality/resolution-review.png`: the higher inference size
retains more straight railing structure and tighter fountain silhouettes, while
fine ornament remains unresolved. These observations do not justify changing the
default target or establish physical-depth, production-quality, or performance
acceptance. Complete receipts are in `native/resolution-results.json` and
`quality/protected-pixels.json` under the fresh evidence directory.
