"""Backend-to-rasterizer tensor contract for reconstruction.

This module defines explicit validation for the boundary between
`gaussian_backend.py` and `gaussian_rasterizer.py`.

Goals:
- Keep tensor-shape expectations explicit and testable.
- Catch interface drift early with focused contract tests.
- Provide one place to document image/camera/tensor invariants.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

CONTRACT_VERSION = "1.0"
NATIVE_DEFAULT_RECIPE = "isotropic_full_batch_v1"
NATIVE_RECIPES = (NATIVE_DEFAULT_RECIPE, "anisotropic_shuffled_rgb_v1", "anisotropic_shuffled_rgbd_v1")


def resolve_saved_native_recipe(scene_metadata: Mapping[str, Any], splat_metadata: Mapping[str, Any]) -> str | None:
    """Resolve explicit native identity before rendering or export side effects.

    Unmarked legacy scenes keep their historical behavior. Historical native
    scenes lacking recipe tags use the original isotropic recipe. Every present
    recipe tag, including saved optimizer tags, must agree; this pure helper
    must not import Torch so ordinary PLY export remains dependency-light.
    """
    provenance = scene_metadata.get("reconstruction_provenance", {})
    splat_optimizer = splat_metadata.get("native_optimizer", {})
    containers = [scene_metadata, splat_metadata]
    for value in (provenance, splat_optimizer):
        if isinstance(value, Mapping):
            containers.append(value)
    provenance_optimizer = provenance.get("optimizer", {}) if isinstance(provenance, Mapping) else {}
    if isinstance(provenance_optimizer, Mapping):
        containers.append(provenance_optimizer)
    recipes = [container["native_recipe"] for container in containers if "native_recipe" in container]
    implementations = [scene_metadata.get("implementation")]
    if isinstance(provenance, Mapping):
        implementations.append(provenance.get("implementation"))
    if not recipes and "torch_native" not in implementations:
        return None
    if not all(isinstance(value, Mapping) for value in (provenance, splat_optimizer, provenance_optimizer)):
        raise ValueError("Saved native recipe metadata containers must be mappings")
    if any(value is not None and value != "torch_native" for value in implementations):
        raise ValueError("Saved native recipe has a conflicting implementation descriptor")
    if any(not isinstance(recipe, str) or recipe not in NATIVE_RECIPES for recipe in recipes):
        raise ValueError("Saved native recipe descriptors are unsupported")
    recipe = recipes[0] if recipes else NATIVE_DEFAULT_RECIPE
    if any(value != recipe for value in recipes):
        raise ValueError("Saved native recipe descriptors disagree")
    return recipe


def _require_torch():
    """Import torch lazily so this module can be imported in no-torch lanes."""
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in no-torch lanes
        raise RuntimeError("Torch is required to validate reconstruction rasterizer contract.") from exc
    return torch


def _ensure_float_tensor(torch_mod, name: str, value: Any, shape: tuple[int, ...]) -> None:
    """Validate tensor type, floating dtype, and fixed-dimension shape."""
    if not isinstance(value, torch_mod.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim != len(shape):
        raise ValueError(f"{name} must have rank {len(shape)}, got shape {tuple(value.shape)}.")
    for axis, expected in enumerate(shape):
        if expected != -1 and value.shape[axis] != expected:
            raise ValueError(f"{name} has invalid shape {tuple(value.shape)}. " f"Expected axis {axis} size {expected}.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype, got {value.dtype}.")


def validate_backend_rasterizer_payload(
    *,
    positions: Any,
    colors: Any,
    scales: Any,
    rotations: Any,
    opacities: Any,
    intrinsics: Any,
    extrinsics: Any,
    image_size: Sequence[int],
) -> None:
    """Validate tensors passed from GaussianBackend into the rasterizer.

    Contract:
    - `positions`, `colors`, `scales`: (N, 3)
    - `rotations`: (N, 4)
    - `opacities`: (N, 1)
    - `intrinsics`: (3, 3)
    - `extrinsics`: (4, 4)
    - All tensors on same device and finite
    - image_size: (H, W), positive ints
    """
    torch_mod = _require_torch()

    _ensure_float_tensor(torch_mod, "positions", positions, (-1, 3))
    _ensure_float_tensor(torch_mod, "colors", colors, (-1, 3))
    _ensure_float_tensor(torch_mod, "scales", scales, (-1, 3))
    _ensure_float_tensor(torch_mod, "rotations", rotations, (-1, 4))
    _ensure_float_tensor(torch_mod, "opacities", opacities, (-1, 1))
    _ensure_float_tensor(torch_mod, "intrinsics", intrinsics, (3, 3))
    _ensure_float_tensor(torch_mod, "extrinsics", extrinsics, (4, 4))

    batch_size = positions.shape[0]
    for name, tensor in {
        "colors": colors,
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities,
    }.items():
        if tensor.shape[0] != batch_size:
            raise ValueError(f"{name} batch size mismatch: expected {batch_size}, got {tensor.shape[0]}.")

    devices = {
        positions.device,
        colors.device,
        scales.device,
        rotations.device,
        opacities.device,
        intrinsics.device,
        extrinsics.device,
    }
    if len(devices) != 1:
        sorted_devices = [str(device) for device in sorted(devices, key=str)]
        raise ValueError(
            "Backend↔rasterizer contract requires all tensors on one device. " f"Found devices: {sorted_devices}."
        )

    if len(image_size) != 2 or not all(isinstance(value, int) for value in image_size):
        raise TypeError(f"image_size must be a 2-tuple of ints, got {image_size!r}.")
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError(f"image_size must contain positive values, got {tuple(image_size)}.")

    for name, tensor in {
        "positions": positions,
        "colors": colors,
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
    }.items():
        if not torch_mod.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values.")


def validate_rasterizer_output(*, rendered: Any, image_size: Sequence[int]) -> None:
    """Validate rasterizer output contract.

    Output contract:
    - rendered: (H, W, 3) float tensor
    - finite values only
    """
    torch_mod = _require_torch()

    if len(image_size) != 2 or not all(isinstance(value, int) for value in image_size):
        raise TypeError(f"image_size must be a 2-tuple of ints, got {image_size!r}.")
    expected_shape = (image_size[0], image_size[1], 3)

    if not isinstance(rendered, torch_mod.Tensor):
        raise TypeError(f"rendered must be a torch.Tensor, got {type(rendered).__name__}.")
    if rendered.shape != expected_shape:
        raise ValueError(f"rendered shape mismatch: expected {expected_shape}, got {tuple(rendered.shape)}.")
    if not rendered.is_floating_point():
        raise TypeError(f"rendered must use a floating-point dtype, got {rendered.dtype}.")
    if not torch_mod.isfinite(rendered).all():
        raise ValueError("rendered contains non-finite values.")
