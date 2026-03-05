# Phase 6 Backend to Rasterizer API Contract

## Scope

This document defines the explicit tensor contract between:

- `GaussianBackend` (`src/transformation_portal/spatial_ai/reconstruction/gaussian_backend.py`)
- `render_gaussians` / `render_gaussians_fast`
  (`src/transformation_portal/spatial_ai/reconstruction/gaussian_rasterizer.py`)

Contract validators live in:

- `src/transformation_portal/spatial_ai/reconstruction/protocol.py`

## Contract Version

- Current contract version: `1.0`

## Input Contract (Backend -> Rasterizer)

Required tensors and shapes:

- `positions`: `(N, 3)` float
- `colors`: `(N, 3)` float
- `scales`: `(N, 3)` float
- `rotations`: `(N, 4)` float quaternion form `[w, x, y, z]`
- `opacities`: `(N, 1)` float
- `intrinsics`: `(3, 3)` float
- `extrinsics`: `(4, 4)` float
- `image_size`: `(H, W)` positive integers

Invariants:

- All tensors must share a single device.
- All tensors must be finite (`torch.isfinite(...).all()`).
- Batch dimension `N` must match across splat tensors.

## Output Contract (Rasterizer -> Backend)

- `rendered`: `(H, W, 3)` float tensor
- Must be finite and shape-consistent with `image_size`

## Enforcement

Runtime validation hooks:

- `GaussianBackend._optimize(...)` validates payload once before optimization.
- `GaussianBackend.render_view(...)` validates rendered output before numpy conversion.

Test coverage:

- `tests/spatial_ai/reconstruction/test_rasterizer_contract.py`

## Change Policy

Any contract shape/device/dtype change must include all of:

1. Contract validator update (`protocol.py`)
2. Contract test update (`test_rasterizer_contract.py`)
3. This document update with a version bump when behavior is breaking
