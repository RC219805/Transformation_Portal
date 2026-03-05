"""Contract tests for the backend↔rasterizer boundary."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is required for rasterizer contract tests")
pytestmark = [pytest.mark.ml, pytest.mark.slow]

from transformation_portal.spatial_ai.reconstruction.gaussian_rasterizer import (  # pylint: disable=wrong-import-position
    compute_rgb_loss,
    render_gaussians,
    render_gaussians_fast,
)
from transformation_portal.spatial_ai.reconstruction.protocol import (  # pylint: disable=wrong-import-position
    CONTRACT_VERSION,
    validate_backend_rasterizer_payload,
    validate_rasterizer_output,
)


def _build_payload(num_gaussians: int = 12, image_size: tuple[int, int] = (32, 48)):
    h, w = image_size
    positions = torch.randn(num_gaussians, 3, dtype=torch.float32)
    positions[:, 2] = positions[:, 2].abs() + 3.0
    colors = torch.sigmoid(torch.randn(num_gaussians, 3, dtype=torch.float32))
    scales = torch.rand(num_gaussians, 3, dtype=torch.float32) * 0.2 + 0.05
    rotations = torch.zeros(num_gaussians, 4, dtype=torch.float32)
    rotations[:, 0] = 1.0
    opacities = torch.rand(num_gaussians, 1, dtype=torch.float32) * 0.3 + 0.6
    intrinsics = torch.tensor(
        [
            [70.0, 0.0, w / 2.0],
            [0.0, 70.0, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    extrinsics = torch.eye(4, dtype=torch.float32)
    return positions, colors, scales, rotations, opacities, intrinsics, extrinsics, image_size


def test_contract_version_is_declared():
    """Contract should expose a stable version token for docs/tests."""
    assert CONTRACT_VERSION == "1.0"


def test_backend_payload_contract_accepts_valid_tensors():
    """Valid backend tensors should satisfy the payload contract."""
    positions, colors, scales, rotations, opacities, intrinsics, extrinsics, image_size = _build_payload()
    validate_backend_rasterizer_payload(
        positions=positions,
        colors=colors,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_size=image_size,
    )


def test_backend_payload_contract_rejects_batch_mismatch():
    """Batch dimensions must remain aligned across all splat tensors."""
    positions, colors, scales, rotations, opacities, intrinsics, extrinsics, image_size = _build_payload()
    bad_rotations = rotations[:-1]
    with pytest.raises(ValueError, match="batch size mismatch"):
        validate_backend_rasterizer_payload(
            positions=positions,
            colors=colors,
            scales=scales,
            rotations=bad_rotations,
            opacities=opacities,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_size=image_size,
        )


def test_render_output_contract_for_full_and_fast_paths():
    """Both render entrypoints should satisfy the same output contract."""
    positions, colors, scales, rotations, opacities, intrinsics, extrinsics, image_size = _build_payload(num_gaussians=32)
    validate_backend_rasterizer_payload(
        positions=positions,
        colors=colors,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_size=image_size,
    )

    rendered_full = render_gaussians(
        positions=positions,
        colors=colors,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_size=image_size,
        device="cpu",
    )
    validate_rasterizer_output(rendered=rendered_full, image_size=image_size)

    rendered_fast = render_gaussians_fast(
        positions=positions,
        colors=colors,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_size=image_size,
        max_gaussians=16,
        device="cpu",
    )
    validate_rasterizer_output(rendered=rendered_fast, image_size=image_size)


def test_gradient_flow_respects_contract():
    """Contracted tensors should still support autograd in rasterizer loss."""
    positions, colors, scales, rotations, opacities, intrinsics, extrinsics, image_size = _build_payload(num_gaussians=10)
    positions.requires_grad_(True)
    colors.requires_grad_(True)
    scales.requires_grad_(True)
    rotations.requires_grad_(True)
    opacities.requires_grad_(True)

    validate_backend_rasterizer_payload(
        positions=positions,
        colors=colors,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_size=image_size,
    )

    rendered = render_gaussians(
        positions=positions,
        colors=colors,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_size=image_size,
        device="cpu",
    )
    target = torch.zeros_like(rendered)
    loss = compute_rgb_loss(rendered, target)
    loss.backward()

    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert colors.grad is not None and torch.isfinite(colors.grad).all()
    assert scales.grad is not None and torch.isfinite(scales.grad).all()
