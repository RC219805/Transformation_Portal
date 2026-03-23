# ADR-042: Scene Group Contract for Multi-View Reconstruction

**Status:** IMPLEMENTED
**Date:** 2026-03-04
**Updated:** 2026-03-23
**Owner:** @RC219805
**Implementation Status:**
- Phase A (scene_id + images): ✅ Complete
- Phase B (cameras field + eligibility): ✅ Complete (2026-03-23)
- Reconstruction feature gate: ✅ Complete (config flag documented)

## Executive Summary

This ADR defines the scene-group contract for future multi-view reconstruction in
`lux_depth_v3` while preserving current per-image behavior by default.

The contract standardizes:
- immutable `SceneGroup` shape (`scene_id`, ordered `images`, optional `cameras`)
- deterministic `scene_id` generation and ordering invariants
- camera resolution precedence (`explicit > EXIF > synthetic`)
- reconstruction eligibility requirements and failure handling
- explicit reconstruction feature gate (`lux_depth_v3.enable_reconstruction=false` by default)

## Context

`lux_depth_v3` currently processes images independently:

`image -> depth -> materials segmentation -> enhancement`

Recent groundwork introduced:
- deterministic segmentation mask artifacts
- `SceneGroup` scaffold
- inert orchestration bridge in `enhance_batch`

These changes enable a future stage:

`scene_group (images >= 2, cameras) -> reconstruction`

The repository does not yet define a formal scene grouping contract for:
- grouping semantics
- camera metadata source and precedence
- deterministic ordering guarantees
- reconstruction eligibility and failure behavior

Without this contract, reconstruction integration risks interface churn and non-deterministic behavior.

## Decision

Define a scene-group contract as the canonical handoff boundary between per-image processing and scene-level reconstruction.

### SceneGroup structure

Target Phase B contract shape:

```python
@dataclass(frozen=True)
class SceneGroup:
    scene_id: str
    images: Tuple[Path, ...]
    cameras: Optional[Tuple[CameraParams, ...]] = None
```

Current Phase A/Phase A.6 scaffold shape (already implemented) includes only:
- `scene_id`
- `images`

The optional `cameras` field is introduced in Phase B.

Field semantics:
- `scene_id`: deterministic scene identifier
- `images`: ordered, immutable tuple of image paths
- `cameras`: optional camera tuple aligned one-to-one with `images`

## Deterministic Ordering

Grouping must preserve deterministic ordering:
- orchestrators MUST normalize input discovery order before grouping (for example `images = sorted(images)`)
- `build_scene_groups` MUST treat incoming `images` as already normalized and MUST NOT reorder image paths within a scene
- `scene_id` derivation must be deterministic for equivalent input sets

### Scene Identifier Generation

Default `scene_id` generation MUST use a SHA1 hash of normalized relative image paths.

Algorithm:
1. normalize each image path relative to `dataset_root`
2. convert to POSIX string representation
3. sort paths lexicographically
4. join paths with newline separators
5. compute SHA1 digest of the resulting string and keep the first 12 hex chars

Reference:

```python
import hashlib

scene_id = hashlib.sha1(
    "\n".join(
        sorted(str(path.relative_to(dataset_root).as_posix()) for path in images)
    ).encode("utf-8"),
    usedforsecurity=False,
).hexdigest()[:12]
```

Rationale:
- prevents collisions across directories (for example `foo/a.jpg` vs `bar/a.jpg`)
- deterministic across machines
- stable under identical datasets

Implementations MAY provide an explicit `scene_id_strategy` hook, but the default
algorithm above MUST be used unless explicitly configured otherwise.

## Validity Requirements

When reconstruction is enabled, reconstruction eligibility is evaluated
**after camera resolution** using the precedence policy in this ADR.

A scene is reconstruction-eligible only if:
- `len(scene.images) >= 2`
- `scene.cameras is not None`
- `len(scene.cameras) == len(scene.images)`

If eligibility is not met, per-image pipeline behavior continues and reconstruction is skipped.

Missing camera metadata alone does not make a scene ineligible; synthetic
camera fallback may satisfy eligibility if a valid camera tuple is produced.

## Camera Source Precedence

Camera parameters are resolved with strict precedence:
1. explicit metadata file (manifest/sidecar)
2. EXIF-derived intrinsics/extrinsics where available
3. synthetic defaults

Rule:
- explicit > EXIF > synthetic

## Failure Modes

| Condition | Behavior |
|---|---|
| Single image scene | Process image normally; skip reconstruction |
| Missing camera metadata | Resolve synthetic camera parameters; continue if valid cameras are produced |
| Invalid camera set | Mark scene invalid, emit warning, skip reconstruction |

## Pipeline Stage Order

Future pipeline shape with reconstruction enabled:

`discover_images -> build_scene_groups -> per-image processing -> scene-level reconstruction`

Current behavior remains per-image equivalent until reconstruction is explicitly enabled.

## Consequences

Positive:
- deterministic grouping and stable scene identity
- explicit reconstruction preconditions
- clean boundary between image and scene stages
- backward-compatible with current `lux_depth_v3` behavior

Tradeoffs:
- introduces camera metadata surface and precedence policy
- requires additional contract and integration tests

## Implementation Plan

Phase B implementation sequence:
1. extend `build_scene_groups()` with concrete grouping rules
2. add camera metadata loader with precedence enforcement
3. gate reconstruction execution behind explicit feature control

Constraint:
- no behavior change to current pipelines until reconstruction is enabled

## Reconstruction Feature Gate

Scene-level reconstruction is controlled by configuration flag:

`lux_depth_v3.enable_reconstruction`

Status:
- this flag is part of the Phase B implementation plan and is not present in current Phase A/Phase A.6 code

Default value:
- `enable_reconstruction = false`

Behavior:
- when `false`:
  - scene groups are computed
  - per-image pipeline executes unchanged
  - reconstruction stage is skipped
- when `true`:
  - reconstruction is attempted for eligible scenes
  - scenes that fail eligibility fall back to per-image behavior

## Alternatives Considered

- implicit grouping inside reconstruction stage
  - rejected: non-deterministic and difficult to test
- external dataset manifest as mandatory input
  - rejected: too heavy for default pipeline usage
- filename-only grouping heuristics
  - rejected: brittle and environment-dependent

## Validation

Add contract-focused tests:

`tests/lux_depth_v3/test_scene_group_contract.py`

Minimum coverage:
- deterministic grouping and ordering invariants
- camera precedence resolution
- reconstruction eligibility gating
- failure-mode behavior (skip/fallback/warn)

## Success Metrics

- Determinism: identical input scene sets generate identical `scene_id` values across machines.
- Compatibility: with `enable_reconstruction=false`, output ordering and count remain unchanged vs current pipeline.
- Eligibility correctness: reconstruction runs only for scenes meeting contract requirements.
- Failure isolation: invalid scenes do not break batch processing and are explicitly reported in logs/metadata.

## References

- [PR #1110: scene-group scaffold + orchestration seam](https://github.com/RC219805/Transformation_Portal/pull/1110)
- [Scene grouping scaffold](../../src/transformation_portal/lux_depth_v3/scene_groups.py)
- [ADR Directory Guidance](README.md)

## Follow-up

After this ADR is accepted, the next safe implementation step is to extend `build_scene_groups()` with deterministic grouping rules before introducing reconstruction-stage orchestration changes.
