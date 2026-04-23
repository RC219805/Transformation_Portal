# Materials V3 Response Planner Audit

**Date:** 2026-04-23
**Status:** Confirmed defects, focused hardening recommended
**Scope:** `materials_v3.py`, `materials_v3_response.py`, `materials_v3_taxonomy.py`, adjacent pixel-ops/config/tests/CI surfaces

## Executive Summary

The current Materials V3 response planner has three confirmed correctness defects in the local checkout:

1. `(H, W, 1)` masks crash `compute_edge_signals`.
2. `sky` is marked as canary material in taxonomy but rejected by the planner as `not_in_canary_set`.
3. `refinement_strategy` handling is internally inconsistent: taxonomy defines `none`, config comments mention `disabled`, and the planner still allows `glass` refinement when given `none`.

The right remediation is not a rewrite. Make a narrow planner hardening pass: canonicalize masks before morphology, cache image gradients once per response plan, drive refinement eligibility from taxonomy plus config, sort material iteration, and split refinement telemetry from pixel-op telemetry. Depth-guided refinement should remain a follow-up feature after these contract fixes land.

## Evidence Base

This audit is grounded in the current local repository, not connector inventory or uploaded copies. The directly audited files are:

| File | Relevant responsibility |
| --- | --- |
| `src/transformation_portal/lux_depth_v3/materials_v3.py` | Engine entry point; attaches raw masks to planner stats and accepts `depth_map`. |
| `src/transformation_portal/lux_depth_v3/materials_v3_response.py` | Response planning, edge-signal computation, refinement and pixel-op decision summaries. |
| `src/transformation_portal/lux_depth_v3/materials_v3_taxonomy.py` | Refinement strategy enum and default material metadata. |
| `src/transformation_portal/lux_depth_v3/pixel_ops_executor.py` | Existing canonical mask behavior for executor-side mask handling. |
| `tests/materials/*` | Current Materials V3 regression coverage. |
| `.github/workflows/*` and `requirements/security.txt` | Existing security/dependency validation posture. |

Local probes confirmed the headline behavior:

- Calling `compute_edge_signals` with an `(8, 8, 1)` mask raised `IndexError: tuple index out of range` inside SciPy morphology.
- Calling `_decide_refinement("sky", ...)` with sufficient coverage/confidence/edge support returned `reason: "not_in_canary_set"`.
- Calling `_decide_refinement("glass", ...)` with `refinement_strategy="none"` returned `should_refine_edges: True` and `strategy: "canary"`, while the config comment currently documents `(canary, disabled)` and the taxonomy enum defines `canary`, `all`, and `none`.

## Findings

| Severity | Finding | Current behavior | Required change |
| --- | --- | --- | --- |
| Critical | Planner mask canonicalization gap | `materials_v3.py` passes raw masks into `compute_edge_signals`; the planner uses 2D morphology directly. | Canonicalize planner masks to 2D float32 before boundary extraction and fail closed on unsupported shapes. |
| High | Taxonomy/planner drift | `DEFAULT_MATERIAL_METADATA["sky"]["canary"]` is `True`, but `_decide_refinement` hard-codes `{glass, foliage, water}`. | Make canary eligibility read from taxonomy. |
| High | `refinement_strategy` ignored / config contract drift | `RefinementStrategy` in taxonomy supports `canary`, `all`, and `none`, but planner logic always behaves as canary and returns `"strategy": "canary"`; current config documentation/usage is also inconsistent about whether `disabled` is meant to alias `none`. | Resolve strategy from config and taxonomy, enforce `none`, `canary`, and `all` semantics, and document or normalize whether `disabled` is a supported alias for `none`. |
| Medium | Repeated full-image gradient work | Sobel gradients are recomputed once per material. | Compute normalized gradient magnitude once per response plan and reuse it for every mask. |
| Medium | Summary telemetry conflates reasons | `skipped_reasons_histogram` is populated only from pixel-op reasons. | Split refinement reasons from pixel-op reasons while preserving the existing summary envelope. |
| Low | Material order depends on upstream insertion order | `generate_response_plan` iterates `per_class_stats.items()`. | Iterate sorted material keys for deterministic response-plan ordering. |
| Low | `depth_map` is accepted but unused | `MaterialsV3Engine.process(..., depth_map=None)` does not consume it. | Document as follow-up or emit metadata such as `depth_guidance_available`; do not mix depth-guided refinement into the critical fix. |

### 1. Planner Mask Canonicalization Gap

The executor already contains `_canonical_mask` for `(H, W)`, `(H, W, 1)`, and `(1, H, W)` masks. The response planner does not share that contract. `compute_edge_signals` thresholds whatever mask it receives and then calls `scipy.ndimage.binary_dilation` with a 2D structure. For singleton-channel 3D masks, this is a confirmed runtime crash.

Minimum safe fix:

- Move the canonical mask helper to a planner-accessible module or duplicate it in `materials_v3_response.py` if avoiding import cycles is simpler.
- Return 2D `float32` masks for supported shapes.
- Raise `ValueError` with a clear message for unsupported mask dimensionality.
- Add planner-specific tests for `(H, W)`, `(H, W, 1)`, `(1, H, W)`, invalid 3D, and 4D masks.

### 2. Taxonomy And Strategy Drift

The taxonomy says `sky` is a canary material. The planner does not consult taxonomy, so `sky` is blocked before refinement eligibility can be evaluated. Separately, the taxonomy enum exposes `canary`, `all`, and `none`, while the config comment documents `(canary, disabled)`. `_decide_refinement` ignores both surfaces and hard-codes the response strategy to `"canary"`.

Minimum safe fix:

- Resolve `strategy = str(getattr(config, "refinement_strategy", "canary")).lower()`.
- Decide and document whether `disabled` is a supported alias for `none`; if supported, normalize it before decision logic, and if not supported, reject it as invalid.
- For `none`, always return ineligible with a stable reason such as `strategy_disabled`.
- For `canary`, allow only materials whose taxonomy metadata has `canary: True`.
- For `all`, allow all present materials subject to coverage/confidence/boundary/edge gates.
- For unknown strategy values, fail closed by treating the strategy as disabled or returning `invalid_refinement_strategy`; do not silently widen eligibility.

### 3. Gradient Recompute Hotspot

The edge-alignment metric is reasonable, but the implementation recomputes grayscale conversion and Sobel magnitude for the whole image for every present material. That makes response planning scale with material count for work that is actually image-global.

Minimum safe fix:

- Add a helper that computes normalized gradient magnitude once per `generate_response_plan` call.
- Pass the cached gradient field into `compute_edge_signals`.
- Keep performance claims as estimates until a benchmark baseline is recorded.

Expected impact is likely meaningful on multi-material megapixel images, but the exact CPU savings must be benchmarked.

## Corrected CI And Security Assessment

The earlier broad recommendation to add dependency scanning is stale for this checkout. The repo already has:

- Dependency Review workflow at `.github/workflows/dependency-review.yml`.
- Bandit and pip-audit execution in `.github/workflows/ci.yml`.
- Governed security tool pins in `requirements/security.txt`.
- Benchmark markers and benchmark workflows that keep heavy performance checks outside fast PR gates.

Do not add a separate Materials V3 CI workflow unless the existing lanes cannot express the needed checks. Prefer focused tests under `tests/materials/` and rely on the current CI/security infrastructure.

## Recommended Implementation Patch

Keep the patch bounded to response planning plus focused tests:

1. Add planner mask canonicalization and normalized gradient helper.
2. Change `compute_edge_signals` to accept the cached gradient field and operate only on canonical 2D masks.
3. Make `_decide_refinement` taxonomy/config driven.
4. Sort `per_class_stats` iteration in `generate_response_plan`.
5. Split `skipped_reasons_histogram` into:
   - `refinement`
   - `pixel_ops`
6. Preserve the top-level response-plan schema keys: `version`, `config_summary`, `per_class`, and `summary`.

Do not implement depth-guided refinement in this pass. At most, emit explicit metadata that depth guidance was available so the unused API surface is visible without changing refinement behavior.

## Required Tests

Add focused response-planner tests covering:

- `compute_edge_signals` accepts 2D masks.
- `compute_edge_signals` accepts `(H, W, 1)` masks.
- `compute_edge_signals` accepts `(1, H, W)` masks.
- Invalid 3D masks fail closed with a clear `ValueError`.
- `refinement_strategy="none"` disables all refinement.
- `refinement_strategy="disabled"` is either normalized to `none` or rejected with the same documented fail-closed behavior used for invalid strategies.
- `refinement_strategy="canary"` allows taxonomy canaries, including `sky`.
- `refinement_strategy="all"` allows non-canary materials through normal gates.
- Response-plan material ordering is stable and sorted.
- Summary histograms separately report refinement and pixel-op reasons.

Repo-aligned validation commands:

```bash
.venv/bin/python -m pytest -q tests/materials
.venv/bin/python -m pytest -q tests/materials -k "response_plan or refinement or mask"
make test-fast
```

## Prioritized Roadmap

| Priority | Work | Rationale |
| --- | --- | --- |
| P0 | Planner mask canonicalization | Fixes a confirmed runtime crash. |
| P0 | Config/taxonomy-driven refinement policy | Restores the documented contract for sky and resolves the `none` versus `disabled` strategy drift. |
| P1 | Cached gradient field | Removes repeated full-image work with low implementation risk. |
| P1 | Deterministic ordering and split histograms | Improves reproducibility and triage without changing execution semantics. |
| P2 | Depth-guided refinement | Valuable feature work, but not part of the critical correctness fix. |

## Acceptance Criteria

The hardening work is complete only when:

- The Materials V3 focused test suite passes.
- The mask-shape regression tests prove `(H, W, 1)` and `(1, H, W)` no longer crash the planner.
- Strategy tests prove `none`, `canary`, and `all` behave distinctly, and prove the documented behavior for `disabled`.
- `sky` eligibility follows taxonomy.
- Response-plan ordering is deterministic.
- Fast local validation is green, or any remaining failure is clearly classified as product logic, stale test logic, or environment/tooling.
