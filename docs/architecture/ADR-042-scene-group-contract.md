# ADR-042: Scene Group Contract for Multi-View Reconstruction

Status: Proposed  
Date: 2026-03-04  
Owner: @RC219805

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

```python
@dataclass(frozen=True)
class SceneGroup:
    scene_id: str
    images: Tuple[Path, ...]
    cameras: Optional[Tuple[CameraParams, ...]] = None
```

Field semantics:
- `scene_id`: deterministic scene identifier
- `images`: ordered, immutable tuple of image paths
- `cameras`: optional camera tuple aligned one-to-one with `images`

## Deterministic Ordering

Grouping must preserve deterministic ordering:
- input image discovery order is normalized via `sorted(input_images)`
- grouping logic must not reorder image paths within a scene
- `scene_id` derivation must be deterministic for equivalent input sets

## Validity Requirements

When reconstruction is enabled, a scene is reconstruction-eligible only if:
- `len(scene.images) >= 2`
- `scene.cameras is not None`
- `len(scene.cameras) == len(scene.images)`

If eligibility is not met, per-image pipeline behavior continues and reconstruction is skipped.

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
| Missing cameras | Fallback to synthetic camera parameters |
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

## Follow-up

After this ADR is accepted, the next safe implementation step is to extend `build_scene_groups()` with deterministic grouping rules before introducing reconstruction-stage orchestration changes.
