# Cold-Zone Coverage Program

**Document Status:** Active proposal
**Last Updated:** 2026-05-25
**Related Docs:** `docs/testing/STRATEGY.md`, `docs/testing/test_coverage_improvement_plan.md`
**Related Scripts:** `scripts/ci/check_per_package_coverage.py`, `scripts/ci/check_per_package_branch_coverage.py`
**Related ADRs:** ADR-031 (test dependency isolation), ADR-044 (marker enforcement)
**Related audit items:** N-2 (audit `PORTAL_AUDIT_REPO_WIDE_2026-05-18.md` finding #2 — fragmented coverage)

> This document supersedes the unrevised "cold-zone testing optimization strategy"
> draft circulated 2026-05-12. It folds in a feasibility audit of the proposed
> targets and corrects five assumptions that would have blocked execution.

---

## 1. Why this exists

The repository's test suite has substantial volume but uneven distribution.
A handful of high-LOC, decision-heavy modules carry minimal direct coverage and
are exercised mostly through broad smoke tests. This program closes those gaps
deterministically (no model downloads, no GPU, no network in the default lane)
and ratchets per-package floors so the gains don't silently regress.

The program optimizes four metrics jointly. **Coverage percentage alone is not
the goal.**

| Metric                      | Why                                                       | Gate style                                |
| --------------------------- | --------------------------------------------------------- | ----------------------------------------- |
| Cold-zone **line coverage** | Surfaces previously-untested files                        | Per-package/file floors in `check_per_package_coverage.py` |
| **Branch coverage**         | These modules are decision-heavy                          | Branch-aware report; targeted assertions  |
| **Runtime budget**          | Prevent coverage gains from bloating CI                   | Per-test and per-lane caps                |
| **Runtime isolation**       | No accidental ML/model/network work in core CI            | Strict pytest markers + import seams      |

---

## 2. Feasibility audit (2026-05-12)

Five strategy assumptions failed audit and have been revised. The rest hold.

### 2.1 VLM import seam is broken at the package level

- `src/transformation_portal/vlm/__init__.py` eagerly imports `LLaVAProcessor`,
  `SceneAnalyzer`, `QualityValidator`.
- `src/transformation_portal/vlm/llava.py` line 21 does unconditional top-level
  `import torch` (no `try`/`except`).
- `quality_validator.py` and `scene_analyzer.py` both eagerly do
  `from transformation_portal.vlm.llava import LLaVAProcessor`.

Any `import transformation_portal.vlm` therefore triggers `import torch`. The
existing tests in `tests/vlm/test_{llava,quality_validator,scene_analyzer}.py`
all gate on `pytest.importorskip("torch")` and are marked `@pytest.mark.ml`,
so they only run in the ML lane today.

**Revision:** the lazy-import refactor is a precondition for any cpu/core VLM
coverage work and must land in **PR 0**, not PR 3.

### 2.2 VLM is brownfield, not greenfield

`tests/vlm/test_llava.py`, `test_quality_validator.py`, and
`test_scene_analyzer.py` already exist (all `@pytest.mark.ml`). Targets like
"70%+ initial" must be re-framed as deltas over a measured baseline.

### 2.3 Proposed test paths don't match repo convention

| Originally proposed                | Actual existing layout                                                                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/plugins/`                   | `tests/unit/plugins/{test_plugin_manager,test_plugin_registry,test_plugin_validator}.py` plus `tests/security/test_plugin_loading_security.py`    |
| `tests/depth/test_tools_*.py`      | `tests/test_depth_tools.py` (root) plus `tests/unit/depth/*`                                                                                      |
| `tests/streaming/`                 | `tests/test_streaming_stages.py`, `tests/test_streaming_async_pipeline.py` (root)                                                                 |
| `tests/reconstruction/`            | `tests/spatial_ai/reconstruction/` (16 files at audit time, exists)                                                                               |

**Revision:** new tests land under `tests/unit/<area>/` or alongside existing
peers. No new top-level test directories.

### 2.4 Reconstruction (former PR 7) is mostly covered already

Source is at `src/transformation_portal/spatial_ai/reconstruction/` (not a
top-level `reconstruction/`). `tests/spatial_ai/reconstruction/` already has
16 test files at audit time, including `test_contracts.py`,
`test_gaussian_backend.py`, `test_gaussian_rasterizer.py`,
`test_nvdiffrec_backend.py`, `test_lazy_imports.py`, `test_export_ply.py`,
`test_phase6a_verification.py`, `test_rasterizer_contract.py`, and
`test_reconstruction_mvp.py`. Golden snapshot at
`tests/golden/reconstruction/tiny_scene_cpu.json`.

**Revision:** reconstruction is demoted from a new-contracts PR to a
**measurement-driven fill-gaps** follow-up that runs after PR 0 baseline lands.

### 2.5 No baseline numbers exist locally

No `coverage.xml` is committed. Targets like "55%" / "70%" are guesses until
`make coverage-report` runs and per-file numbers are recorded.

**Revision:** PR 0 generates and commits a cold-zone baseline report; floors
are set in the **next** PR after the baseline is reviewed.

### 2.6 Confirmed (strategy was correct)

- Branch coverage **is** collected (`pyproject.toml`
  `[tool.coverage.run] branch = true`), but
  `scripts/ci/check_per_package_coverage.py` enforces line coverage only
  (reads `lines-covered` / `lines-valid`).
- Current floors: `src/tp/` 40%, `lux_depth_v3/validators/` 70%,
  `lux_depth_v3/` 30% (excludes validators). No floors yet for
  plugins/stage_graph/vlm/depth/streaming.
- All target source files exist with the approximate LOC profile assumed.
  Treat these as audit sizing figures and refresh them before opening the
  implementation PRs that use them for scope/risk decisions:

  | File                              | Approx. LOC |
  | --------------------------------- | ----------: |
  | `streaming/stages.py`             |      ~1,240 |
  | `depth/tools.py`                  |      ~1,110 |
  | `plugins/loader.py`               |        ~650 |
  | `vlm/quality_validator.py`       |        ~560 |
  | `vlm/scene_analyzer.py`          |        ~400 |
  | `vlm/llava.py`                   |        ~370 |
  | `stage_graph/policy.py`          |        ~340 |
  | `stage_graph/stages/depth.py`    |        ~180 |

  `streaming/stages.py` at roughly 1.2k LOC is larger than a 6-file slice will
  cover cleanly; PR 6 needs further decomposition.

---

## 3. Revised PR order

| PR  | Scope                                                                    | Initial line target | Notes |
| --: | ------------------------------------------------------------------------ | ------------------: | ----- |
|   0 | Instrumentation: cold-zone report + branch-aware mode + **VLM lazy-import seam** |                 N/A | Precondition for everything else |
|   1 | `plugins/loader.py` security + lifecycle                                 |               70%+ | Security-sensitive, deterministic |
|   2 | `stage_graph/policy.py` matrix                                            |               85%+ | Pure policy logic; fastest stable branch coverage |
|   3 | VLM parser tests (cpu/core after PR 0 seam)                              |             70-80% | Was PR 3, unchanged in order but unblocked by PR 0 |
|   4 | `stage_graph/stages/depth.py` fallback boundaries                         |               70%+ | Mockable model boundary |
|   5 | `depth/tools.py` behavioral slices                                        |             55-65% | ~1.1k LOC; slice by behavior |
|   6 | `streaming/stages.py` async lifecycle (subset 1)                          |             40-50% | ~1.2k LOC; subset first, finish in 6b |
|  6b | `streaming/stages.py` subset 2 (worker pool + remaining stages)           |             55-65% | Split out of PR 6 due to size |
|   7 | Reconstruction / Gaussian: **fill measured gaps only**                    | delta over baseline | Demoted from new-contracts work |

`stage_graph/policy.py` (PR 2) jumps ahead of VLM parsers because it has no
optional-runtime dependency surface and produces the fastest stable branch
coverage gains.

---

## 4. PR 0 — instrumentation and import seam

PR 0 is the bedrock. It has three deliverables in a single PR.

### 4.1 Cold-zone coverage report

New tool: `scripts/ci/cold_zone_report.py`. Reads `coverage.xml`, emits both
JSON (machine-readable) and Markdown (review-readable) tables with, per file:

- Line coverage percent
- Branch coverage percent
- Missed line ranges
- Missed branch count
- Recommended marker lane (`unit`, `security`, `ml`, `integration`)

Commits the **first run output** as
`docs/testing/cold_zone_baseline_YYYY-MM-DD.md`, using the ISO date when the
baseline is generated. Reviewers see actual numbers before any floor is
proposed.

### 4.2 Branch-aware enforcement (companion mode)

Either extend `scripts/ci/check_per_package_coverage.py` with a `--branches`
flag that reads `branches-covered` / `branches-valid`, **or** add
`scripts/ci/check_per_package_branch_coverage.py` as a sibling. Do not call
the existing line-oriented enforcement "branch ratchet" until a script
actually reads branch-rate data from Cobertura.

### 4.3 VLM lazy-import seam

Concrete refactor:

- `src/transformation_portal/vlm/__init__.py` — drop eager re-exports.
  Replace with `__getattr__` lazy loader or PEP 562 `__all__`-driven lazy
  imports.
- `src/transformation_portal/vlm/llava.py` — wrap `import torch` and
  `from transformers import ...` in `try`/`except ImportError` blocks,
  setting `TORCH_AVAILABLE` / `LLAVA_AVAILABLE` flags. Module must import
  cleanly with no `torch` installed.
- `src/transformation_portal/vlm/quality_validator.py` — replace top-level
  `from transformation_portal.vlm.llava import LLaVAProcessor` with
  `TYPE_CHECKING`-guarded import for typing and lazy import inside methods
  that actually instantiate a processor.
- Same treatment for `scene_analyzer.py`.

Acceptance test (cpu/core, no `ml` marker):

```python
def test_vlm_importable_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    importlib.reload(transformation_portal.vlm)
    # imports without raising; LLaVAProcessor exists but instantiation
    # raises an actionable ImportError pointing at LLAVA_INSTALL_GUIDANCE.
```

### 4.4 Acceptance gate for PR 0

- `cold_zone_report.py` runs cleanly against a committed `coverage.xml`.
- Companion branch-aware enforcement exists and is dry-run wired (no floors
  applied yet).
- `import transformation_portal.vlm` succeeds in an environment without
  `torch`.
- All existing `tests/vlm/test_*.py` continue to pass in the `ml` lane.
- At least one new test under `tests/vlm/` proves cpu/core importability.
- No floor changes; no claim of "branch ratchet" in CI text.

---

## 5. PR 1 — `plugins/loader.py` security + lifecycle

### Test placement

`tests/unit/plugins/test_loader_external_paths.py`
`tests/unit/plugins/test_loader_manifest_loading.py`
`tests/unit/plugins/test_loader_file_plugins.py`
`tests/unit/plugins/test_loader_lifecycle.py`
`tests/unit/plugins/test_loader_dependency_errors.py`

(Matches existing `tests/unit/plugins/test_plugin_{manager,registry,validator}.py`.)

### Must-cover contracts

| Contract                             | Required tests                                                                 |
| ------------------------------------ | ------------------------------------------------------------------------------ |
| External plugins disabled by default | Env plugin path ignored; direct `add_search_path()` rejects non-builtin path   |
| External plugins explicitly enabled  | Env var and constructor opt-in both allow external search paths                |
| Manifest loading                     | `plugin.json`, `pyproject.toml`, malformed JSON, missing entry point           |
| Entry point failure modes            | Invalid `module:Class`, missing class, class not subclassing `PluginInterface` |
| Dependency checks                    | Missing dependency becomes load error; installed dependency passes             |
| File plugin discovery                | Skips `_private.py`; loads concrete plugin class                               |
| Lifecycle                            | `unload_plugin()` calls `cleanup()`, removes cache, removes `sys.modules`      |
| Singleton                            | `get_global_loader()` stable across repeated access                            |

### Acceptance gate

- `plugins/loader.py` line coverage at least the lower of (70%, baseline + 25
  points).
- All branch misses on security-relevant paths reviewed manually in PR.
- All new tests marked `unit` or `security`; none touch real filesystem
  plugin paths outside `tmp_path`.

---

## 6. PR 2 — `stage_graph/policy.py` matrix

### Test placement

`tests/stage_graph/test_policy_matrix.py` (alongside existing
`tests/stage_graph/test_{enhancement,materials,upscaling}_stage*.py`).

### Test matrix

| Component                            | Required matrix                                                                        |
| ------------------------------------ | -------------------------------------------------------------------------------------- |
| `DevicePolicy.select_device()`       | coreml-depth preference, cuda priority, mps fallback, cpu fallback, `prefer_gpu=False` |
| `DevicePolicy.can_use_batch()`       | tiny batch pass, large batch fail, memory headroom boundary                            |
| `QualityPolicy.apply_preset()`       | draft, standard, high, production exact field expectations                             |
| `CachingPolicy.should_cache_stage()` | depth, material, segmentation, enhance, upscale, unknown, disabled                     |
| `PolicyEngine.create_policy()`       | preset + scene + config overrides compose in correct order                             |
| `_detect_devices()`                  | missing torch, cuda true, mps true, missing psutil, missing coremltools                |
| `_apply_config()`                    | cpu override, cache dir, cache enabled, upscale factor, enhancement strength, workers  |

### Acceptance gate

- `stage_graph/policy.py` line coverage ≥85% and branch coverage ≥75%.
- All enum values exercised.
- All device branches exercised via `monkeypatch` / dependency injection.
- Zero real `torch` / `coremltools` / `psutil` dependency at test time.

---

## 7. PR 3 — VLM parser coverage in cpu/core (post-PR 0)

PR 0 must have landed.

### Test placement

`tests/vlm/test_quality_validator_parser.py`
`tests/vlm/test_scene_analyzer_parser.py`
`tests/vlm/test_vlm_import_without_ml_runtime.py` (the PR 0 acceptance test
moves here if it lived elsewhere).

These are **separate files** from the existing `@pytest.mark.ml` tests so the
core lane and ml lane stay independent.

### Quality validator tests

| Contract                  | Required tests                                              |
| ------------------------- | ----------------------------------------------------------- |
| Importability             | Module imports without ML runtime (PR 0 contract)            |
| Processor injection       | Fake processor returns fixed responses                      |
| Detailed score parsing    | `Score: 8/10`, `8.5/10`, missing score default              |
| Thresholds                | pass/warning/fail boundaries                                |
| Strict mode               | any failed aspect fails strict validation                   |
| Artifact extraction       | halo, banding, blur, synthetic, overexposed                 |
| Recommendation extraction | uppercase header, lowercase header, bullets, numbered items |
| Enhancement comparison    | improvement, regression, new artifacts                      |

### Scene analyzer tests

| Contract        | Required tests                                                              |
| --------------- | --------------------------------------------------------------------------- |
| Space type      | interior, exterior, aerial, unknown                                         |
| Room type       | kitchen, bathroom, bedroom, living, dining, office, pool, courtyard, entry  |
| Style           | Mediterranean/Spanish, modern, coastal, luxury estate, unknown              |
| Materials       | marble, wood, glass, metal, stone, fabric                                   |
| Luxury features | high ceilings, designer fixtures, ocean view, smart home                    |
| Lighting        | explicit `LIGHTING:` section and fallback keyword                           |
| Recommendations | kitchen, bathroom, bedroom, pool, Mediterranean, aerial                     |

### Acceptance gate

- New parser tests marked `unit` (not `ml`).
- `src/transformation_portal/vlm/` line coverage ≥70%, with the new cpu/core
  tests contributing measurably (verify by running the `not ml` lane in
  isolation).
- No `torch`, `transformers`, model download, or network access in cpu/core
  tests.

---

## 8. PR 4 — `stage_graph/stages/depth.py` fallback boundaries

### Test placement

`tests/stage_graph/test_depth_stage_contracts.py`

### Required tests

| Contract            | Required tests                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------- |
| Missing input       | `compute()` fails with explicit missing image error                                       |
| Cache key           | deterministic for same image/config; changes with model size/version; `no_image` fallback |
| Model load fallback | fake transformer import failure yields placeholder path                                   |
| Placeholder output  | correct shape, `float32`, normalized range                                                |
| Fake real model     | fake pipeline returns depth; output normalized to `[0,1]`                                 |
| Bad model output    | inference exception returns constant fallback                                             |
| Input scaling       | float `[0,1]` and uint8 paths both convert cleanly                                        |

### Acceptance gate

- `stage_graph/stages/depth.py` line coverage ≥70%.
- No `transformers` dependency in cpu/core.
- All fallback paths exercised.

---

## 9. PR 5 — `depth/tools.py` behavioral slices

### Test placement

Alongside existing `tests/test_depth_tools.py`, **or** as new files under
`tests/unit/depth/`:

`tests/unit/depth/test_tools_cache_retry.py`
`tests/unit/depth/test_tools_validation.py`
`tests/unit/depth/test_tools_discovery.py`
`tests/unit/depth/test_tools_depth_and_masks.py`
`tests/unit/depth/test_tools_effects.py`
`tests/unit/depth/test_tools_batch_driver.py`

### Runtime budget

| Test group             | Fixtures                  | Per-test budget |
| ---------------------- | ------------------------- | --------------: |
| Cache/retry/validation | No image files            |            < 1s |
| Discovery              | `tmp_path` filenames only |            < 1s |
| Depth/mask loading     | Tiny PNG/TIFF fixtures    |            < 2s |
| Effects                | `8x8` / `16x16` arrays    |            < 2s |
| Batch driver           | 1-3 tiny files            |            < 5s |

### Must-cover branches

| Subarea             | Critical branches                                                   |
| ------------------- | ------------------------------------------------------------------- |
| `BoundedCache`      | eviction, copy-on-read, clear/stats                                 |
| Retry decorator     | retry on `OSError`, no retry on non-I/O, max-attempt failure        |
| `validate_color`    | 0-1, 0-255, wrong length, out of range                              |
| Discovery           | recursive, priority tag ordering, extension case handling, no match |
| Depth normalization | percentile, histogram, linear, resize, cache hit                    |
| Mask loading        | missing, `L`, `RGBA`, `RGB`, corrupt file fallback                  |
| Haze/clarity/DOF    | shape, range, mask protection, quality modes                        |
| Batch behavior      | skip missing, fail missing, partial success                         |

### Acceptance gate

- `depth/tools.py` line coverage ≥55% after first PR; ratchet upward
  separately.
- No test image larger than `32x32` unless explicitly `@pytest.mark.integration`.
- No multiprocessing in the default lane unless deterministic.

---

## 10. PR 6 / 6b — `streaming/stages.py` async lifecycle

Split because the source is roughly 1.2k LOC.

### PR 6 — subset 1

`tests/unit/streaming/test_image_data.py`
`tests/unit/streaming/test_image_load_stage.py`
`tests/unit/streaming/test_image_save_stage.py`

(Or as root peer files beside existing `tests/test_streaming_stages.py`;
placement must not create a new top-level `tests/streaming/` directory.)

| Stage                   | Required tests                                                                 |
| ----------------------- | ------------------------------------------------------------------------------ |
| `ImageData`             | shape/dtype for valid array and `None`                                         |
| `ImageLoadStage`        | PNG, TIFF fallback, 16-bit conversion, optional EXIF failure ignored           |
| `ImageSaveStage`        | float TIFF, float PNG, output metadata, output dir creation                    |

### PR 6b — subset 2

`tests/unit/streaming/test_depth_estimation_stage_contracts.py`
`tests/unit/streaming/test_material_response_stage.py`
`tests/unit/streaming/test_stage_lifecycle.py`

| Stage                   | Required tests                                                                 |
| ----------------------- | ------------------------------------------------------------------------------ |
| `DepthEstimationStage`  | fake torch device detection, synthetic opt-in, fail-closed unavailable backend |
| `MaterialResponseStage` | RGB fixture, grayscale fixture, metadata, depth attenuation                    |
| Lifecycle               | owned worker pool startup/shutdown; injected worker pool not owned             |

### Acceptance gate (combined PR 6 + 6b)

- `streaming/stages.py` line coverage ≥55%.
- All default-lane tests marked `unit`.
- No real model load.
- Async tests deterministic under pytest's asyncio auto mode.

---

## 11. PR 7 — reconstruction / Gaussian fill-gaps

Reconstruction is at `src/transformation_portal/spatial_ai/reconstruction/`,
with 16 test files under `tests/spatial_ai/reconstruction/` at audit time.
PR 7 is **not** a new-contracts PR; it fills measured gaps from PR 0's
baseline.

### Workflow

1. After PR 0 lands, generate a reconstruction-scoped report.
2. List the top 5 uncovered branches across `gaussian_backend.py`,
   `gaussian_rasterizer.py`, `geometric_validator.py`, `mesh_exporter.py`,
   `nvdiffrec_backend.py`, `scene_builder.py`, `export_ply.py`,
   `contracts.py`, `protocol.py`.
3. Add **targeted** tests under the existing `tests/spatial_ai/reconstruction/`
   directory. Reuse fixtures from `test_reconstruction_mvp.py` and
   `test_coverage_boost.py`.
4. Backend gating tests (standard tier rejected / research tier accepted /
   license ack required) belong under `tests/spatial_ai/reconstruction/` as
   contracts, not in a new file tree.

### Acceptance gate

- Reconstruction cold-module line coverage rises by at least the delta
  necessary to match the per-package floor proposed in the post-baseline
  ratchet PR (number set after PR 0 measurement).
- Backend gating tests assert standard/research/license behavior **without**
  invoking CUDA, COLMAP, NVDiffRec, or real Gaussian splatting runtime.
- Real runtime-backed tests remain separately marked `ml` or `integration`.

---

## 12. Ratchet policy

Two floors, both required.

### 12.1 Stable package floor (CI-enforced)

Added to `PACKAGE_FLOORS` in `scripts/ci/check_per_package_coverage.py`
**after** the baseline report lands, set 5-10 points below the measured
stable coverage. Ratchet upward after two consecutive stable CI runs.

Post-PR7 floors landed after baseline review. A second stability ratchet landed
after repeated green CI runs and a fresh 2026-05-13 required CI coverage
snapshot. Floors remain conservative: prefixes with stable required-lane
headroom were raised, while prefixes with cross-lane variance were held or
given additional buffer until the next measured ratchet.

```python
PackageFloor("src/transformation_portal/plugins/", 48.0)
PackageFloor("src/transformation_portal/stage_graph/", 74.0)
PackageFloor("src/transformation_portal/vlm/", 69.0)
PackageFloor("src/transformation_portal/depth/", 57.0)
PackageFloor("src/transformation_portal/streaming/", 53.0)
PackageFloor(
    "src/transformation_portal/spatial_ai/reconstruction/",
    42.0,
)
```

| Package prefix | Required CI line coverage snapshot | Enforced floor |
| --- | ---: | ---: |
| `src/transformation_portal/plugins/` | 51.40% | 48.0% |
| `src/transformation_portal/stage_graph/` | 77.66% | 74.0% |
| `src/transformation_portal/vlm/` | 73.49% | 69.0% |
| `src/transformation_portal/depth/` | 59.95% | 57.0% |
| `src/transformation_portal/streaming/` | 56.72% | 53.0% |
| `src/transformation_portal/spatial_ai/reconstruction/` | 43.62% | 42.0% |

Post-baseline branch floors also landed after dry-run baseline review, then
received the same stability ratchet:

```python
BranchFloor("src/transformation_portal/plugins/", 36.0)
BranchFloor("src/transformation_portal/stage_graph/", 63.0)
BranchFloor("src/transformation_portal/vlm/", 55.0)
BranchFloor("src/transformation_portal/depth/", 42.0)
BranchFloor("src/transformation_portal/streaming/", 29.0)
BranchFloor(
    "src/transformation_portal/spatial_ai/reconstruction/",
    47.0,
)
```

| Package prefix | Required CI branch coverage snapshot | Enforced floor |
| --- | ---: | ---: |
| `src/transformation_portal/plugins/` | 39.86% | 36.0% |
| `src/transformation_portal/stage_graph/` | 66.77% | 63.0% |
| `src/transformation_portal/vlm/` | 58.82% | 55.0% |
| `src/transformation_portal/depth/` | 44.19% | 42.0% |
| `src/transformation_portal/streaming/` | 31.97% | 29.0% |
| `src/transformation_portal/spatial_ai/reconstruction/` | 49.77% | 47.0% |

The depth branch floor was raised after PR #1794 merged at
`31361b27526b438dbc2220ada3e4f2f4145807de` and full CI run `25949723087`
confirmed `src/transformation_portal/depth/` at 601/1360 branches (44.19%)
in both core coverage lanes.

### 12.2 Touched-file rule (per-PR, reviewer-enforced)

For any PR touching a cold-zone file:

- Coverage on the touched file must not decrease.
- New lines must be covered unless explicitly justified.
- New untested branches require reviewer sign-off in the PR body.

The core coverage lane now runs
`scripts/ci/check_cold_zone_touched_files.py coverage.xml --compare-ref origin/main`
after the package line and branch coverage ratchets. The check reports line
coverage, branch coverage, missed line ranges, and missed branch counts for any
touched cold-zone source file. It fails only when a touched cold-zone file is
missing from `coverage.xml`; percentage regressions and newly untested branches
remain review decisions documented in the PR body.

---

## 13. Test quality bar

### Required

- Tiny deterministic fixtures.
- No network, no model downloads, no real GPU in cpu/core.
- Explicit marker on every new test file.
- Branch intent legible from test names.
- Failure messages prove contract behavior, not implementation trivia.

### Discouraged

- Tests that only `import` modules for coverage.
- Broad end-to-end tests used to cover tiny branches.
- Snapshot tests on volatile LLM/VLM text.
- Real plugin loading outside `tmp_path`.
- Large fixture images in unit tests.
- Hidden dependency on local ML packages.

---

## 14. CI lane mapping

| Lane          | Includes                                                       | Excludes                          |
| ------------- | -------------------------------------------------------------- | --------------------------------- |
| `cpu/core`    | unit, security, parser, policy, mock backend contracts          | model downloads, GPU, network     |
| `ml`          | `torch`, `transformers`, SAM2, LLaVA smoke/availability         | large e2e unless `integration`   |
| `integration` | real image fixtures, end-to-end stage chains                    | default PR blocker unless selected |
| `golden`      | deterministic artifact byte/hash checks                         | unstable visual/model outputs     |
| `benchmark`   | performance budgets                                             | coverage-driven assertions        |

After PR 0 lands, **importability without optional ML runtime** is an
explicit contract for VLM parser modules. Other optional-runtime modules
adopt the same contract when their PR opens.

---

## 15. Success criteria by milestone

### After PR 0-2

- Cold-zone coverage report tool exists and runs in CI.
- Branch-aware enforcement script exists (dry-run, no floors yet).
- VLM modules importable without `torch`.
- `plugins/loader.py` has security + lifecycle coverage.
- `stage_graph/policy.py` has matrix + branch coverage.
- No CI runtime regression over baseline > 5%.

### After PR 3-4

- VLM parser modules covered in cpu/core, not just `ml`.
- `stage_graph/stages/depth.py` fallback paths covered.

### After PR 5-6/6b

- `depth/tools.py` no longer depends on broad smoke tests for confidence.
- `streaming/stages.py` async lifecycle covered.
- Image fixtures remain tiny and deterministic.

### After PR 7

- Reconstruction cold-spot gaps closed without runtime dependency.
- Real-runtime tests still isolated in `ml` / `integration` lanes.

---

## 16. Out of scope

- ~~Changing the global `--cov-fail-under` floor.~~ **Superseded 2026-05-25 by audit item N-2** — the 25 → 30 ratchet is now in scope (see §17 below). The original prohibition stood when this doc was self-contained; the audit places the floor inside the cold-zone program's responsibility.
- Touching `lux_depth_v3/` core seams beyond what existing per-package floors
  cover.
- Rewriting `app.py` or `orchestrator.py` coverage strategy (covered by
  `docs/testing/test_coverage_improvement_plan.md`).
- ML-lane GPU validation (`integration` and `ml` lanes remain governed
  separately).

---

## 17. N-2 — ML sampled coverage and global floor ratchet

Tracks backlog item N-2 (audit finding #2, fragmented coverage). Two
deliverables, sequenced.

### 17.1 ML sampled coverage on cold packages (landed)

The ML PR lane in `.github/workflows/build.yml` now runs an advisory,
non-blocking `pytest --cov` over `src/transformation_portal/vlm` and
`src/transformation_portal/spatial_ai/segmentation` after the main test
step completes. The step is gated with `continue-on-error: true`; its
output is uploaded as the `coverage-ml-sampled-<python-version>`
artifact (30-day retention) for use as evidence in §17.2. The artifact
also carries `coverage-ml-sampled-status.txt`, so marker-zero or other
advisory pytest exits remain visible even when no coverage XML is
produced.

Selection: `pytest -m "ml and not slow and not integration and not
benchmark"` over the two test trees only — keeps the sample focused on
the cold packages without re-running the full ML suite under tracing
overhead.

### 17.2 Global `--cov-fail-under` ratchet (deferred)

The build.yml core lane currently sets `--cov-fail-under=25`. The
audit's acceptance criterion is to ratchet that to `30` once the
cold-zone baseline shows stable margin on the affected packages.

**Ratchet trigger (must hold before flipping):**

1. Two consecutive main-branch core CI runs report combined line
   coverage ≥ 35% (a five-point margin over the new 30 floor).
2. The `coverage-ml-sampled-*` artifact from a main-branch run shows
   the ML-lane sampled packages above their existing per-package
   floors (`vlm ≥ 69%` per §12.1, no regression on
   `spatial_ai/segmentation`).
3. No open PR is reducing the core lane's combined coverage.

When all three hold, a follow-up PR flips
`COV_FLAGS="... --cov-fail-under=30"` in build.yml and records the date
the new floor took effect in §17.3 below.

### 17.3 Ratchet history

| Date | Floor before | Floor after | Trigger evidence |
|---|---:|---:|---|
| 2026-05-25 | 25 | 25 | N-2 instrumentation landed; ratchet awaiting two consecutive ≥35% main runs (see §17.2). |

---

## 18. Open questions before PR 0

1. Should the cold-zone report be committed per run, or regenerated on demand?
   Recommendation: commit baseline once, regenerate on demand thereafter and
   diff in CI.
2. Branch-aware enforcement as a flag on `check_per_package_coverage.py`, or a
   sibling script? Recommendation: sibling script — keeps the line-oriented
   contract documented and stable.
3. Does the VLM lazy-import seam require any downstream caller updates (e.g.
   code paths that assume `transformation_portal.vlm.LLaVAProcessor` is
   importable without instantiation)? Audit during PR 0; surface affected
   call sites in the PR description.

---

## 19. Floor additions — 2026-06-06 (app.py + orchestrator durable-state)

A coverage audit on 2026-06-06 found two governed surfaces that were already
*measured* by the core-tier coverage step (`.github/workflows/build.yml` runs
`--cov=app`) but had **no enforced floor**, so they could silently regress:

1. **`app.py`** — the FastAPI origin holding the most security-critical
   hardening (allowed-root validation, API-key / trusted-host enforcement,
   request size / concurrency / rate limits, pipeline allowlists).
2. **The paid-pilot durable-state backends** —
   `orchestrator/storage/`, `orchestrator/queue/`, `orchestrator/artifact_store/`
   (the `JobRepository` / `QueueBroker` / `ArtifactStore` Protocol surfaces).
   Their Postgres/Redis/S3 paths only get full exercise behind the opt-in
   live-service contract gates, so the core-lane rollup leans on the
   in-memory/local implementations — which is exactly the contract coverage a
   floor protects.

Floors were set conservatively below a 2026-06-06 core-lane snapshot
(`(unit or security or regression or golden or integration) and not ml and not
slow and not benchmark`) to absorb cross-lane variance, per the
"conservative starter, ratchet upward after a confirming CI run" discipline.

| Prefix | Measured line | Line floor | Measured branch | Branch floor |
|---|---:|---:|---:|---:|
| `app.py` | 83.8% | 76.0% | 75.3% | 66.0% |
| `orchestrator/storage/` | 68.0% | 60.0% | 51.4% | 44.0% |
| `orchestrator/queue/` | 67.6% | 58.0% | 43.8% | 36.0% |
| `orchestrator/artifact_store/` | 65.9% | 58.0% | 54.7% | 46.0% |

Line floors live in `scripts/ci/check_per_package_coverage.py`; branch floors
in `scripts/ci/check_per_package_branch_coverage.py`. Ratchet upward once the
live-service lanes (Postgres/Redis/S3 contract gates) are folded into the
floor-bearing coverage run.

### 19.1 File-level ratchets from behavioral test fills (2026-06-06)

Two live, pure-Python modules were lifted out of the cold zone with
deterministic behavioral tests and pinned with file-level floors:

| File | Before | After (line/branch) | Floor (line/branch) | Tests |
|---|---:|---|---:|---|
| `metrics/ledger.py` | ~30% line | 98.8% / 93% | 90% / 82% | `tests/test_metrics_ledger.py` |
| `comfyui/workflow_builder.py` | 0% line | 100% / 100% | 90% / 80% | `tests/test_comfyui_workflow_builder.py` |
| `hardening/` (`universal.py`) | 81% / 67% | 98% / 100% | 90% / 85% | `tests/security/test_universal_hardening.py` |
| `storage/cas_store.py` | 78.5% / 59% | 96% / 89% | 88% / 80% | `tests/storage/test_cas_store_lifecycle.py` |
| `orchestrator/worker.py` | 76% / 71% | 100% / 100% | 90% / 85% | `tests/orchestrator/test_worker_runner_unit.py` |

`comfyui/workflow_builder.py` required a precondition: `comfyui/__init__.py`
eagerly imported `custom_nodes` (top-level `import torch`), so the pure builder
could not be imported in the core lane. The package import now eagerly exposes
only pure workflow construction primitives while keeping runtime custom nodes
and the executor behind lazy `__getattr__` access, so `workflow_builder` /
`workflow_templates` import torch-free while `custom_nodes` / `executor` load
only on first access.
