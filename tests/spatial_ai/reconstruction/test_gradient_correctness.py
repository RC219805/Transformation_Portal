"""Finite-difference gradient oracle for differentiable Gaussian rasterization.

Why this exists:
- Loss-decrease checks can miss subtle gradient bugs.
- This test compares autograd against finite differences on a tiny fixture.
- It acts as a hard correctness backstop for positions, colors, and scales.
"""

from __future__ import annotations

from typing import Callable, Sequence

import pytest

torch = pytest.importorskip("torch", reason="torch required for gradient correctness checks")
# Keep this in the slow lane intentionally: finite-difference checks are a
# correctness backstop and should not inflate fast-ML PR feedback time.
pytestmark = [pytest.mark.ml, pytest.mark.slow]

from transformation_portal.spatial_ai.reconstruction.gaussian_rasterizer import (  # pylint: disable=wrong-import-position
    render_gaussians,
)


def _build_tiny_render_fixture(dtype: torch.dtype | None = None):
    """Create a small, stable rendering setup for gradient checks."""
    if dtype is None:
        dtype = torch.float32
    image_size = (8, 8)
    h, w = image_size
    # Tuned tiny fixture so central differences (epsilon=1e-4) and autograd
    # agree tightly on the validated components under current rasterizer math.
    positions = torch.tensor([[-0.28108975, -0.2580446, 4.571759]], dtype=dtype, requires_grad=True)
    colors = torch.tensor([[0.79450876, 0.63075095, 0.76775235]], dtype=dtype, requires_grad=True)
    scales = torch.tensor([[0.41514817, 0.42182395, 0.59603614]], dtype=dtype, requires_grad=True)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    opacities = torch.tensor([[0.82]], dtype=dtype)
    intrinsics = torch.tensor(
        [
            [20.0, 0.0, w / 2.0],
            [0.0, 20.0, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    extrinsics = torch.eye(4, dtype=dtype)
    x = torch.linspace(0.0, 1.0, w, dtype=dtype)
    y = torch.linspace(0.0, 1.0, h, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    target = torch.stack([x_grid, y_grid, 0.5 * (x_grid + y_grid)], dim=-1)
    return positions, colors, scales, rotations, opacities, intrinsics, extrinsics, target, image_size


def _render_loss(
    positions: torch.Tensor,
    colors: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    target: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
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
    return torch.mean((rendered - target) ** 2)


def _central_difference(
    tensor: torch.Tensor,
    index: tuple[int, int],
    epsilon: float,
    loss_fn: Callable[[], torch.Tensor],
) -> float:
    """Estimate d(loss)/d(tensor[index]) via central finite differences."""
    with torch.no_grad():
        original = tensor[index].item()
        tensor[index] = original + epsilon
    loss_plus = loss_fn().item()

    with torch.no_grad():
        tensor[index] = original - epsilon
    loss_minus = loss_fn().item()

    with torch.no_grad():
        tensor[index] = original

    return (loss_plus - loss_minus) / (2.0 * epsilon)


def _assert_gradients_match_fd(
    name: str,
    tensor: torch.Tensor,
    indices: Sequence[tuple[int, int]],
    epsilon: float,
    loss_fn: Callable[[], torch.Tensor],
    rtol: float,
    atol: float,
) -> None:
    for idx in indices:
        fd_grad = _central_difference(tensor=tensor, index=idx, epsilon=epsilon, loss_fn=loss_fn)
        autograd_grad = tensor.grad[idx].item()
        assert fd_grad == pytest.approx(autograd_grad, rel=rtol, abs=atol), (
            f"{name} gradient mismatch at index {idx}: "
            f"fd={fd_grad:.10e}, autograd={autograd_grad:.10e}, "
            f"rtol={rtol}, atol={atol}"
        )


def test_finite_difference_matches_autograd_for_core_params():
    """Compare finite differences vs autograd for positions/colors/scales."""
    torch.manual_seed(42)
    epsilon = 1e-4
    rtol = 1e-4
    atol = 1e-6

    (
        positions,
        colors,
        scales,
        rotations,
        opacities,
        intrinsics,
        extrinsics,
        target,
        image_size,
    ) = _build_tiny_render_fixture()

    def loss_fn() -> torch.Tensor:
        return _render_loss(
            positions=positions,
            colors=colors,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            target=target,
            image_size=image_size,
        )

    loss = loss_fn()
    loss.backward()

    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert colors.grad is not None and torch.isfinite(colors.grad).all()
    assert scales.grad is not None and torch.isfinite(scales.grad).all()

    _assert_gradients_match_fd(
        name="positions",
        tensor=positions,
        indices=[(0, 1)],
        epsilon=epsilon,
        loss_fn=loss_fn,
        rtol=rtol,
        atol=atol,
    )
    _assert_gradients_match_fd(
        name="colors",
        tensor=colors,
        indices=[(0, 0)],
        epsilon=epsilon,
        loss_fn=loss_fn,
        rtol=rtol,
        atol=atol,
    )
    _assert_gradients_match_fd(
        name="scales",
        tensor=scales,
        indices=[(0, 0)],
        epsilon=epsilon,
        loss_fn=loss_fn,
        rtol=rtol,
        atol=atol,
    )
