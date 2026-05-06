# Monolith Decomposition — Ranked Seam Targets

**Status:** Active inventory (2026-05-04)
**Authority:** Companion to [ADR-043 Orchestrator Decomposition](ADR-043-orchestrator-decomposition.md) and [ADR-045 Monolith Decomposition Residuals](ADR-045-monolith-decomposition-residuals.md).
**Source of pressure:** [Development Roadmap Q2 2026 §4 "Orchestrator Residual Slimming & Boundary Enforcement"](DEVELOPMENT_ROADMAP_2026_Q2.md).
**Predecessor audit:** [TODO Inventory 2026-05-04 Refresh](../analysis/TODO_INVENTORY.md).

---

## Why this list exists

The 2026-05-04 audit confirmed there are **zero `TODO|FIXME|XXX|HACK` markers** in the five largest source modules:

| File | LOC | Markers | Governance signal |
|---|---:|---:|---|
| `app.py` | 10,039 | 0 | CLAUDE.md: known monolith; tested by feature surface |
| `src/transformation_portal/lux_depth_v3/orchestrator.py` | 7,257 | 0 | ADR-043 complete; Q2 roadmap residual-slimming target |
| `src/transformation_portal/lux_depth_v3/segmentation_backend.py` | 2,519 | 0 | None active |
| `src/transformation_portal/pipelines/rendering_4k_pipeline.py` | 2,380 | 0 | None active |
| `src/transformation_portal/spatial_ai/orchestration/pipeline.py` | 2,121 | 0 (1 governed phase gate at line 1774) | Single-view reconstruction phase gate only |

Without source-code TODOs to crystallize the work, this document is the persistent target list. Each candidate names a seam, an extraction destination, and the verification pattern to use. **No extraction is performed by this document — only the prioritization.**

---

## Ranking criteria

A seam is ranked higher when it satisfies more of:

1. **Governance pressure exists** (an ADR, roadmap, or contract demands it).
2. **Destination module already exists** so backward-compatible re-exports are cheap.
3. **Surface is non-public-API** (no FastAPI route or contract envelope changes).
4. **Existing tests cover the boundary** so regressions surface immediately.
5. **Seam is structurally bounded** (continuous line range, single responsibility).

Risk is rated by inverse blast-radius: contracts > security boundaries > model-lock semantics > pure data/caching helpers.

---

## Top 3 Targets

### Target 1 — `lux_depth_v3/orchestrator.py` residual slimming

**Why first:** Only target with explicit Q2 roadmap pressure; all destination modules already exist (config_resolver, pipeline_coordinator, execution_engine, artifact_manager, validators); ADR-043 establishes the phased pattern; backward-compat re-exports already in place at the facade.

Three sub-seams, each independently extractable. Pick one per PR, ~200 LOC each, in line with the roadmap ratchet.

**Seam 1A — Config fingerprint helpers** → `lux_depth_v3/config_resolver.py`

| Symbol | Location |
|---|---|
| `_model_variant` | `orchestrator.py:583` |
| `_resolve_backend_model_id` | `orchestrator.py:734` |
| `_build_depth_cache_fingerprint` | `orchestrator.py:697` |
| `_build_materials_fingerprint_payload` | `orchestrator.py:1082` |
| `_build_pbr_fingerprint_payload` | `orchestrator.py:1089` |
| `_build_apex_depth_gate_fingerprint_payload` | `orchestrator.py:1096` |
| `_build_depth_cache_payload` | `orchestrator.py:1103` |
| `compute_config_fingerprint` | `orchestrator.py:1114` |
| `_finalize_run_card_config_fingerprint` | `orchestrator.py:1122` |
| `_build_run_card_config_fingerprint` | `orchestrator.py:1135` |

Risk: low. Pure-function helpers operating on already-resolved config; the existing `ConfigFingerprint` data class is the natural seam contract.

**Seam 1B — Artifact reload / coercion helpers** → `lux_depth_v3/artifact_manager.py`

| Symbol | Location |
|---|---|
| `_load_existing_manifest` | `orchestrator.py:1321` |
| `_coerce_output_paths` | `orchestrator.py:1341` |
| `_normalize_v2_status` | `orchestrator.py:1350` |
| `_restore_materials_v3_from_manifest` | `orchestrator.py:1366` |
| `_preserved_v2_result_from_manifest` | `orchestrator.py:1401` |
| `_normalize_backend_provenance` | `orchestrator.py:1430` |
| `_has_expanded_stage_a_fingerprint` | `orchestrator.py:1435` |
| `_segmentation_mask_artifact_path` | `orchestrator.py:2615` |

Risk: low–medium. Touches manifest schema; covered by `tests/test_backend_selection.py` and run-card contract tests.

**Seam 1C — Backend resolution & state** → `lux_depth_v3/pipeline_coordinator.py`

| Symbol | Location |
|---|---|
| `_initialize_depth_backend` | `orchestrator.py:589` |
| `_resolve_runtime_backend_chain` | `orchestrator.py:677` |
| `_expected_output_depth_units_for_backend` | `orchestrator.py:688` |
| `_default_model_id_for_backend` | `orchestrator.py:716` |
| `_derive_model_id_from_backend_instance` | `orchestrator.py:723` |
| `_resolve_backend_model_artifact` | `orchestrator.py:770` |
| `_seed_depth_attempts_from_selection_fallback` | `orchestrator.py:900` |
| `_get_or_create_depth_backend` | `orchestrator.py:947` |
| `_set_active_depth_state` | `orchestrator.py:995` |
| `_build_backend_metadata_for_attempts` | `orchestrator.py:1006` |

Risk: medium. Touches backend selection metadata that flows into manifests (`BackendSelectionMetadata` at `manifest.py:164–241`). Existing tests at `tests/test_backend_selection.py` plus `_capture_backend_metadata` (already at `orchestrator.py:1297`) prove the seam contract.

**Verification per sub-seam:** `make test-orchestrator-contract` + `pytest tests/test_backend_selection.py -v`. Each PR is expected to drop ≥150 LOC from `orchestrator.py` while keeping it a thin facade.

---

### Target 2 — `app.py` first slice: portal asset bundle

**Why second:** `app.py` is the largest monolith but also the riskiest (typed envelope contracts, security boundary, rate limits). The portal-asset bundle is a clean first slice: pure data + caching, no FastAPI routes, no auth or path-validation logic.

**Seam:** Extract to `src/transformation_portal/portal/asset_bundle.py` (new module).

| Symbol | Location |
|---|---|
| `PortalAssetSpec` | `app.py:86` |
| `PortalAssetBundle` | `app.py:92` |
| `PortalRenderedTextAsset` | `app.py:102` |
| `_load_portal_asset_manifest` | `app.py:301` |
| `_portal_asset_signature` | `app.py:339` |
| `_fingerprint_bytes` | `app.py:344` |
| `_portal_asset_route_path` | `app.py:348` |
| `_portal_asset_versioned_url` | `app.py:353` |
| `_render_portal_template` | `app.py:357` |
| `_portal_direct_asset_signature` | `app.py:370` |
| `_build_portal_direct_asset_fingerprint` | `app.py:375` |
| `_get_portal_direct_asset_fingerprint` | `app.py:379` |
| `_portal_css_dependency_asset_names` / `_portal_css_signature` / `_build_portal_css_asset` / `_get_portal_css_asset` | `app.py:383–420` |
| `_portal_html_signature` / `_build_portal_asset_bundle` / `_get_portal_asset_bundle` | `app.py:424–470` |
| `_requested_portal_asset_fingerprint` / `_portal_asset_cache_control` / `_portal_asset_etag` / `_portal_asset_request_etag_matches` / `_portal_asset_not_modified_response` | `app.py:474–501` |

Approx LOC: ~415. Re-export from `app.py` as `from transformation_portal.portal.asset_bundle import *` to preserve internal callers.

**Verification:** `make check-portal-asset-budgets`, `make validate-portal-css-layer-parity`, `pytest tests/test_app_orchestrator_contract_http.py -v`, `pytest tests/validation/test_portal_smoke_scripts.py -v`.

**Deferred slices (track here, do not promote until 2A lands):**
- *2B — Path resolution & security helpers* (`app.py:428–667` -> `src/transformation_portal/portal/path_security.py`). High risk: hardening surface; allowed-roots, symlink safety, untrusted-path validation. Landed behind the ADR-046 compatibility contract.
- *2C — SAM2 checkpoint security helpers* (`app.py` -> `src/transformation_portal/portal/sam2_checkpoint_security.py`). Medium risk: model-lock semantics and bounded cache eviction. Landed behind the ADR-047 SAM2 checkpoint trust contract; manifest migration remains a separate future decision.

---

### Target 3 — `lux_depth_v3/segmentation_backend.py` per-backend split

**Why third:** Cleanest natural seams (four top-level classes, no shared mutable state besides cache helpers); no governance pressure; useful precedent before tackling `pipelines/rendering_4k_pipeline.py`.

**Layout:** new package `src/transformation_portal/lux_depth_v3/segmentation/`.

| New module | Source range |
|---|---|
| `segmentation/_cache.py` | helpers at `segmentation_backend.py:76–532` (`_build_segmentation_cache_key`, `_read_cached_material_masks`, `_write_cached_material_masks`, `_material_confidence_metadata`, `_mask_checksum`, etc.) |
| `segmentation/stub.py` | `StubBackend` (`segmentation_backend.py:532–565`) |
| `segmentation/efficient_sam.py` | `EfficientSAMBackend` (`segmentation_backend.py:565–1449`) |
| `segmentation/sam2.py` | `SAM2SegmentationBackend` (`segmentation_backend.py:1449–1803`) — note it currently subclasses `EfficientSAMBackend`; preserve that import edge |
| `segmentation/sam_vit_h.py` | `SAMVitHBackend` (`segmentation_backend.py:1803–2110`) |
| `segmentation/registry.py` | `_get_sam_vit_h_instance`, `_get_backend_instance`, `segment_materials`, `get_last_segmentation_runtime_metadata` (`segmentation_backend.py:2110–2519`) |

`segmentation_backend.py` becomes a thin re-export shim for backward compatibility.

**Verification:** `pytest tests/test_segmentation*.py -v`, `make validate-portal-lux-materials-live` (live SAM2 only on Apple Silicon arm64), and the existing materials V3 contract tests.

---

## Lower-priority candidates (not yet ranked)

Track here but do **not** promote above the top 3 until at least target 1 ships an extraction.

- `pipelines/rendering_4k_pipeline.py` (2,380 LOC). Three plausible seams: config dataclasses (155–389), GPU memory + quality assessor (390–815), pipeline execution (1264–2257). No current governance pressure.
- `spatial_ai/orchestration/pipeline.py` (2,121 LOC). Two plausible seams: `PipelineConfig` (378–632) and `SpatialAIPipeline` execution (633–2121). The single-view reconstruction phase gate at line 1774 stays in place — it is governed and unrelated to decomposition.

---

## Seam status table

This table is the canonical progress ledger for extraction PRs that draw from the ranked targets above. Update the row for the shipped seam only; ranking changes require a separate governance refresh.

| Seam | Source module | Destination | Status | Latest evidence |
|---|---|---|---|---|
| Target 1A — Config fingerprint helpers | `lux_depth_v3/orchestrator.py` | `lux_depth_v3/config_resolver.py` | Landed | This PR: config fingerprint helpers extracted |
| Target 1B — Artifact reload / coercion helpers | `lux_depth_v3/orchestrator.py` | `lux_depth_v3/artifact_manager.py` | Landed | This PR: manifest reload/coercion helpers extracted |
| Target 1C — Backend resolution & state | `lux_depth_v3/orchestrator.py` | `lux_depth_v3/pipeline_coordinator.py` | Landed | This PR: backend initialization and runtime attempt-state helpers extracted |
| Target 2A — Portal asset bundle | `app.py` | `src/transformation_portal/portal/asset_bundle.py` | Landed | This PR: portal asset bundle helpers extracted |
| Target 2B — Path resolution & security helpers | `app.py` | `src/transformation_portal/portal/path_security.py` | Landed | This PR: path security helpers extracted behind ADR-046 app compatibility wrappers |
| Target 2C — SAM2 checkpoint security helpers | `app.py` | `src/transformation_portal/portal/sam2_checkpoint_security.py` | Landed | This PR: SAM2 checkpoint security helpers extracted behind ADR-047 app compatibility wrappers |
| Target 3A — Segmentation cache helpers | `lux_depth_v3/segmentation_backend.py` | `lux_depth_v3/segmentation/_cache.py` | Landed | This PR: segmentation cache helpers extracted |
| Target 3B — Stub backend | `lux_depth_v3/segmentation_backend.py` | `lux_depth_v3/segmentation/stub.py` | Landed | This PR: stub backend extracted |
| Target 3C — EfficientSAM backend | `lux_depth_v3/segmentation_backend.py` | `lux_depth_v3/segmentation/efficient_sam.py` | Landed | This PR: EfficientSAM backend extracted |
| Target 3D — SAM2 backend | `lux_depth_v3/segmentation_backend.py` | `lux_depth_v3/segmentation/sam2.py` | Landed | This PR: SAM2 backend extracted |
| Target 3E — SAM ViT-H backend | `lux_depth_v3/segmentation_backend.py` | `lux_depth_v3/segmentation/sam_vit_h.py` | Landed | This PR: SAM ViT-H backend extracted |
| Target 3F — Segmentation registry | `lux_depth_v3/segmentation_backend.py` | `lux_depth_v3/segmentation/registry.py` | Landed | This PR: segmentation registry extracted |
| Target 4A — Rendering config/result types | `pipelines/rendering_4k_pipeline.py` | `pipelines/rendering_4k/types.py` | Landed | This PR: rendering config/result types extracted; ranking unchanged pending governance refresh |
| Target 4B — Rendering pure image stage functions | `pipelines/rendering_4k_pipeline.py` | `pipelines/rendering_4k/stages.py` | Landed | This PR: rendering pure image stage functions extracted; ranking unchanged pending governance refresh |
| Target 4C — Rendering quality and memory helpers | `pipelines/rendering_4k_pipeline.py` | `pipelines/rendering_4k/quality.py` | Landed | This PR: rendering quality assessor and GPU memory manager extracted; ranking unchanged pending governance refresh |
| Target 4D — Rendering pipeline orchestration | `pipelines/rendering_4k_pipeline.py` | `pipelines/rendering_4k/pipeline.py` | Landed | This PR: rendering pipeline orchestration extracted behind legacy CLI/import shim; ranking unchanged pending governance refresh |
| Target 5A — Spatial AI pipeline config | `spatial_ai/orchestration/pipeline.py` | `spatial_ai/orchestration/config.py` | Landed | This PR: spatial pipeline config model and validation helpers extracted; ranking unchanged pending governance refresh |
| Target 5B — Spatial AI pipeline result models | `spatial_ai/orchestration/pipeline.py` | `spatial_ai/orchestration/results.py` | Landed | This PR: spatial pipeline result models and summary serializers extracted; ranking unchanged pending governance refresh |
| Target 5C — Spatial AI segmentation cache helpers | `spatial_ai/orchestration/pipeline.py` | `spatial_ai/orchestration/segmentation_cache.py` | Landed | This PR: segmentation cache helpers extracted; ranking unchanged pending governance refresh |
| Target 5D — Spatial AI graph execution bridge | `spatial_ai/orchestration/pipeline.py` | `spatial_ai/orchestration/graph_pipeline.py` | Landed | This PR: graph execution bridge extracted; ranking unchanged pending governance refresh |

---

## Process — extraction PRs draw from this list

1. Pick a single seam from §"Top 3 Targets". Open a small PR titled `refactor(<area>): extract <seam> per ADR-045`.
2. Follow the ADR-043 phased pattern: add new module + tests first, switch internal callers, leave a re-export shim, only then strip dead code from the source.
3. Cite this document and ADR-045 in the PR description; update this file's status table when a seam lands.
4. Run the listed verification gates plus `python scripts/validation/scan_todo_inventory.py --check-governance` to confirm no ungoverned TODOs slip in.

This document is itself read-only governance — extraction PRs may update only the seam status table, not the ranking.
