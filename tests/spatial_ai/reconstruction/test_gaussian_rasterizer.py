"""Unit tests for Gaussian rasterizer (Phase 6A)."""

import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is required for gaussian rasterizer tests")
pytestmark = pytest.mark.ml

torch.manual_seed(0)
# CI-friendly defaults: avoid oversubscription on shared runners.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

H, W = 48, 64
IMAGE_SIZE = (H, W)
FX = FY = 100.0
CX = W / 2.0
CY = H / 2.0

from transformation_portal.spatial_ai.reconstruction.gaussian_rasterizer import (
    compute_3d_covariance,
    compute_rgb_loss,
    evaluate_gaussian_2d,
    project_gaussians_2d,
    quaternion_to_rotation_matrix,
    render_gaussians,
    render_gaussians_fast,
    render_gaussians_native,
)

# Compositing weight of three stacked splats at the centre pixel, nearest first:
# a1, (1 - a1) * a2, (1 - a1) * (1 - a2) * a3 for opacities 0.6, 0.5 and 0.9.
STACKED_CENTER_RGB = (0.6, 0.4 * 0.5, 0.4 * 0.5 * 0.9)


def _stacked_splats(input_order):
    """Three splats on the optical axis, all centred on the centre pixel.

    Nearest to farthest they are red, green and blue, so each colour channel of the
    centre pixel is the compositing weight of one splat. ``input_order`` permutes the
    tensors, because the result must not depend on the order the caller supplies.
    """
    index = torch.tensor(input_order)
    positions = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0], [0.0, 0.0, 6.0]])
    colors = torch.eye(3)
    # Same 10 px screen-space sigma for every splat (scale * FX / depth).
    scales = torch.tensor([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])
    opacities = torch.tensor([[0.6], [0.5], [0.9]])
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    intrinsics = torch.tensor([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
    return positions[index], colors[index], scales[index], rotations[index], opacities[index], intrinsics, torch.eye(4)


class TestQuaternionRotation:
    """Test quaternion to rotation matrix conversion."""

    def test_identity_quaternion(self):
        """Test identity quaternion [1, 0, 0, 0] produces identity matrix."""
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        R = quaternion_to_rotation_matrix(q)

        expected = torch.eye(3).unsqueeze(0)
        assert torch.allclose(R, expected, atol=1e-6)

    def test_batch_quaternions(self):
        """Test batch conversion."""
        # Identity and 90-degree rotation around z-axis
        q = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # Identity
                [0.7071, 0.0, 0.0, 0.7071],  # 90° around z
            ]
        )
        R = quaternion_to_rotation_matrix(q)

        assert R.shape == (2, 3, 3)

        # Check first is identity
        assert torch.allclose(R[0], torch.eye(3), atol=1e-4)

    def test_quaternion_normalization(self):
        """Test that non-normalized quaternions work (auto-normalized)."""
        q = torch.tensor([[2.0, 0.0, 0.0, 0.0]])  # Not normalized
        R = quaternion_to_rotation_matrix(q)

        # Should still be orthogonal (det = ±1)
        det = torch.det(R[0])
        assert torch.abs(det - 1.0) < 0.1 or torch.abs(det + 1.0) < 0.1


class TestCovariance:
    """Test 3D covariance computation."""

    def test_isotropic_gaussian(self):
        """Test isotropic Gaussian (spherical)."""
        scales = torch.tensor([[1.0, 1.0, 1.0]])
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity

        cov = compute_3d_covariance(scales, rotations)

        # Should be identity matrix (sphere)
        expected = torch.eye(3).unsqueeze(0)
        assert torch.allclose(cov, expected, atol=1e-6)

    def test_anisotropic_gaussian(self):
        """Test anisotropic Gaussian (ellipsoid)."""
        scales = torch.tensor([[2.0, 1.0, 0.5]])
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        cov = compute_3d_covariance(scales, rotations)

        # Should be diagonal with squared scales
        expected = torch.diag(torch.tensor([4.0, 1.0, 0.25])).unsqueeze(0)
        assert torch.allclose(cov, expected, atol=1e-6)


class TestProjection:
    """Test 3D to 2D projection."""

    def test_simple_projection(self):
        """Test projection of points in front of camera."""
        # Single point at (0, 0, 5) in camera space
        positions = torch.tensor([[0.0, 0.0, 5.0]])
        scales = torch.tensor([[0.1, 0.1, 0.1]])
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        # Simple pinhole camera
        intrinsics = torch.tensor([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])

        # Identity extrinsics (camera at origin)
        extrinsics = torch.eye(4)

        image_size = IMAGE_SIZE

        mean_2d, cov_2d, depths, valid_mask = project_gaussians_2d(
            positions, scales, rotations, intrinsics, extrinsics, image_size, use_rotation=False
        )

        # Should project to image center
        assert torch.allclose(mean_2d[0], torch.tensor([CX, CY]), atol=1.0)

        # Depth should be 5.0
        assert torch.allclose(depths[0], torch.tensor(5.0), atol=1e-6)

        # Should be visible
        assert valid_mask[0] == True

    def test_behind_camera_culling(self):
        """Test that points behind camera are culled."""
        # Point behind camera (negative z)
        positions = torch.tensor([[0.0, 0.0, -1.0]])
        scales = torch.tensor([[0.1, 0.1, 0.1]])
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        intrinsics = torch.eye(3)
        intrinsics[0, 0] = intrinsics[1, 1] = FX
        intrinsics[0, 2] = CX
        intrinsics[1, 2] = CY

        extrinsics = torch.eye(4)
        image_size = IMAGE_SIZE

        _, _, _, valid_mask = project_gaussians_2d(
            positions, scales, rotations, intrinsics, extrinsics, image_size, use_rotation=False
        )

        # Should be culled
        assert valid_mask[0] == False

    def test_out_of_bounds_culling(self):
        """Test that points outside image bounds are culled."""
        # Point way off to the side
        positions = torch.tensor([[100.0, 0.0, 5.0]])
        scales = torch.tensor([[0.1, 0.1, 0.1]])
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        intrinsics = torch.eye(3)
        intrinsics[0, 0] = intrinsics[1, 1] = FX
        intrinsics[0, 2] = CX
        intrinsics[1, 2] = CY

        extrinsics = torch.eye(4)
        image_size = IMAGE_SIZE

        _, _, _, valid_mask = project_gaussians_2d(
            positions, scales, rotations, intrinsics, extrinsics, image_size, use_rotation=False
        )

        # Should be out of bounds
        assert valid_mask[0] == False


class TestGaussianEvaluation:
    """Test 2D Gaussian evaluation."""

    def test_center_evaluation(self):
        """Test Gaussian is maximum at center."""
        # Single Gaussian at (100, 100)
        mean_2d = torch.tensor([[100.0, 100.0]])
        cov_2d = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        cov_2d_inv = torch.inverse(cov_2d)

        # Pixel grid around center
        pixel_coords = torch.zeros(3, 3, 2)
        for i in range(3):
            for j in range(3):
                pixel_coords[i, j] = torch.tensor([99.0 + j, 99.0 + i])

        weights = evaluate_gaussian_2d(pixel_coords, mean_2d, cov_2d_inv)

        # Center pixel should have highest weight
        assert weights[0, 1, 1] > weights[0, 0, 0]
        assert weights[0, 1, 1] > weights[0, 2, 2]

    def test_gaussian_falloff(self):
        """Test Gaussian falls off with distance."""
        mean_2d = torch.tensor([[50.0, 50.0]])
        cov_2d = torch.tensor([[[4.0, 0.0], [0.0, 4.0]]])  # Wider Gaussian
        cov_2d_inv = torch.inverse(cov_2d)

        # Points at different distances
        pixel_coords = torch.tensor([[[50.0, 50.0], [52.0, 50.0], [55.0, 50.0]]])  # Distance 0, 2, 5

        weights = evaluate_gaussian_2d(pixel_coords, mean_2d, cov_2d_inv)

        # Should decrease with distance
        assert weights[0, 0, 0] > weights[0, 0, 1]
        assert weights[0, 0, 1] > weights[0, 0, 2]


class TestRendering:
    """Test full rendering pipeline."""

    def test_render_single_gaussian(self):
        """Test rendering a single Gaussian."""
        # Single red Gaussian at image center
        positions = torch.tensor([[0.0, 0.0, 5.0]])
        colors = torch.tensor([[1.0, 0.0, 0.0]])  # Red
        scales = torch.tensor([[0.5, 0.5, 0.5]])
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        opacities = torch.tensor([[1.0]])

        intrinsics = torch.tensor([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])

        extrinsics = torch.eye(4)

        image_size = IMAGE_SIZE

        with torch.inference_mode():
            rendered = render_gaussians(
                positions, colors, scales, rotations, opacities, intrinsics, extrinsics, image_size, device="cpu"
            )

        # Check output shape
        assert rendered.shape == (H, W, 3)

        # Check dtype
        assert rendered.dtype == torch.float32

        # Check no NaN or inf
        assert not torch.isnan(rendered).any()
        assert not torch.isinf(rendered).any()

        # Check center pixel has some red
        center_pixel = rendered[H // 2, W // 2]
        assert center_pixel[0] > 0.1  # Red channel should be bright

    def test_render_multiple_gaussians(self):
        """Test rendering multiple Gaussians with occlusion."""
        # Two Gaussians: red in back, blue in front
        positions = torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, 5.0]])  # Blue closer

        colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # Red, Blue

        scales = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        opacities = torch.tensor([[1.0], [0.8]])  # Blue mostly opaque

        intrinsics = torch.tensor([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])

        extrinsics = torch.eye(4)

        image_size = IMAGE_SIZE

        with torch.inference_mode():
            rendered = render_gaussians(
                positions, colors, scales, rotations, opacities, intrinsics, extrinsics, image_size, device="cpu"
            )

        # Check output is valid
        assert rendered.shape == (H, W, 3)
        assert not torch.isnan(rendered).any()
        assert not torch.isinf(rendered).any()

        # Check that center has some color (not completely black)
        center_pixel = rendered[H // 2, W // 2]
        assert center_pixel.sum() > 0.1, "Center should have visible color"

        # Both splats are centred on this pixel (footprint weight 1): the near blue splat
        # contributes its full opacity and lets 20 % of the opaque red splat behind it through.
        assert torch.allclose(center_pixel, torch.tensor([0.2, 0.0, 0.8]), atol=1e-5)

        # Near splats occlude far ones whatever order the caller supplies them in, and every
        # entry point (full, capped training render, native reference) composites the same way.
        for input_order in itertools.permutations(range(3)):
            scene = _stacked_splats(input_order)
            for render in (render_gaussians, render_gaussians_fast, render_gaussians_native):
                with torch.inference_mode():
                    stacked_center = render(*scene, IMAGE_SIZE)[H // 2, W // 2]
                assert torch.allclose(
                    stacked_center, torch.tensor(STACKED_CENTER_RGB), atol=1e-5
                ), f"{render.__name__} composited input order {input_order} as {stacked_center.tolist()}"

    def test_render_empty_scene(self):
        """Test rendering with no visible Gaussians."""
        # All Gaussians behind camera
        positions = torch.tensor([[0.0, 0.0, -5.0]])
        colors = torch.tensor([[1.0, 0.0, 0.0]])
        scales = torch.tensor([[0.1, 0.1, 0.1]])
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        opacities = torch.tensor([[1.0]])

        intrinsics = torch.eye(3)
        intrinsics[0, 0] = intrinsics[1, 1] = FX
        intrinsics[0, 2] = CX
        intrinsics[1, 2] = CY

        extrinsics = torch.eye(4)
        image_size = IMAGE_SIZE

        rendered = render_gaussians(
            positions, colors, scales, rotations, opacities, intrinsics, extrinsics, image_size, device="cpu"
        )

        # Should be black image
        assert torch.allclose(rendered, torch.zeros_like(rendered))

    def test_render_values_clamped(self):
        """Test that rendered values are in [0, 1]."""
        positions = torch.tensor([[0.0, 0.0, 5.0]])
        colors = torch.tensor([[2.0, -1.0, 0.5]])  # Out of range colors
        scales = torch.tensor([[0.5, 0.5, 0.5]])
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        opacities = torch.tensor([[1.0]])

        intrinsics = torch.tensor([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])

        extrinsics = torch.eye(4)
        image_size = IMAGE_SIZE

        rendered = render_gaussians(
            positions, colors, scales, rotations, opacities, intrinsics, extrinsics, image_size, device="cpu"
        )

        # All values should be in [0, 1]
        assert (rendered >= 0.0).all()
        assert (rendered <= 1.0).all()


class TestFastRendering:
    """Test fast rendering with Gaussian culling."""

    def test_render_fast_culling(self):
        """Test that fast rendering culls distant Gaussians."""
        # Many Gaussians at different depths
        N = 25
        positions = torch.rand(N, 3) * 10
        positions[:, 2] = torch.arange(N, dtype=torch.float32) * 0.1 + 1.0

        colors = torch.rand(N, 3)
        scales = torch.ones(N, 3) * 0.1
        rotations = torch.zeros(N, 4)
        rotations[:, 0] = 1.0  # Identity quaternions
        opacities = torch.ones(N, 1) * 0.5

        intrinsics = torch.tensor([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])

        extrinsics = torch.eye(4)
        image_size = IMAGE_SIZE

        # Render with culling
        rendered = render_gaussians_fast(
            positions, colors, scales, rotations, opacities, intrinsics, extrinsics, image_size, max_gaussians=15
        )

        assert rendered.shape == (H, W, 3)
        assert not torch.isnan(rendered).any()

        # The cap keeps the closest *visible* splats: splats behind the camera or projecting
        # off-screen are nearer than the visible pair below but must not take its slots.
        hidden = torch.tensor([[0.0, 0.0, -0.5]] * 4 + [[1.0, 0.0, 0.5]] * 4)  # behind / off-screen
        visible = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 6.0]])  # near red in front of far blue
        positions = torch.cat([hidden, visible])
        colors = torch.cat([torch.ones(8, 3), torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])])
        scales = torch.cat([torch.full((8, 3), 0.1), torch.tensor([[0.2] * 3, [0.6] * 3])])
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(10, 1)
        opacities = torch.full((10, 1), 0.9)
        scene = (positions, colors, scales, rotations, opacities, intrinsics, extrinsics, image_size)
        with torch.inference_mode():
            full = render_gaussians(*scene)
            capped_to_visible = render_gaussians_fast(*scene, max_gaussians=2)
            nearest_only = render_gaussians_fast(*scene, max_gaussians=1)
        # Two slots hold exactly the two visible splats, so the capped render is the full render.
        assert torch.equal(capped_to_visible, full)
        assert torch.allclose(full[H // 2, W // 2], torch.tensor([0.9, 0.0, 0.1 * 0.9]), atol=1e-5)
        # One slot goes to the nearest visible splat: the near red splat wins, the far blue one is cut.
        assert torch.allclose(nearest_only[H // 2, W // 2], torch.tensor([0.9, 0.0, 0.0]), atol=1e-5)


class TestLossComputation:
    """Test loss functions."""

    def test_rgb_loss(self):
        """Test RGB reconstruction loss."""
        rendered = torch.ones(16, 16, 3) * 0.5
        target = torch.ones(16, 16, 3) * 0.7

        loss = compute_rgb_loss(rendered, target)

        # MSE should be 0.04 (0.2^2)
        expected = torch.tensor(0.04)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_rgb_loss_perfect_match(self):
        """Test loss is zero for perfect match."""
        rendered = torch.rand(100, 100, 3)
        target = rendered.clone()

        loss = compute_rgb_loss(rendered, target)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


class TestGradientFlow:
    """Test gradient backpropagation."""

    def test_gradients_flow(self, monkeypatch):
        """Test that gradients flow through rendering."""
        # Simple scene
        positions = torch.tensor([[0.0, 0.0, 5.0]], requires_grad=True)
        colors = torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True)
        scales = torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
        opacities = torch.tensor([[1.0]], requires_grad=True)

        extrinsics = torch.eye(4)
        image_size = (32, 32)
        # Keep camera principal point aligned to the test render size.
        intrinsics = torch.tensor([[FX, 0.0, image_size[1] / 2.0], [0.0, FY, image_size[0] / 2.0], [0.0, 0.0, 1.0]])

        # Render
        rendered = render_gaussians(
            positions, colors, scales, rotations, opacities, intrinsics, extrinsics, image_size, device="cpu"
        )

        # Compute loss
        target = torch.ones_like(rendered) * 0.8
        loss = compute_rgb_loss(rendered, target)

        # Backprop
        loss.backward()

        # Check gradients exist and are valid
        assert positions.grad is not None
        assert colors.grad is not None
        assert scales.grad is not None

        # Check no NaN or inf in gradients
        assert not torch.isnan(positions.grad).any()
        assert not torch.isnan(colors.grad).any()
        assert not torch.isnan(scales.grad).any()

        # Colors should have gradient (most direct influence)
        assert torch.abs(colors.grad).sum() > 0

        # R-8: both compositors (in place into full-image buffers, and layer by layer) must give
        # bit-identical images and gradients, and small splats in a large image must not be
        # written into full-image buffers in place, which makes autograd copy the whole image
        # once per splat in the backward pass.
        from transformation_portal.spatial_ai.reconstruction import gaussian_rasterizer

        for threshold in (0, 10**12):  # force the layered, then the in-place compositor
            monkeypatch.setattr(gaussian_rasterizer, "_IN_PLACE_COMPOSITE_MAX_IMAGE_TO_WINDOW_AREA", threshold)
            for n_splats in (3, 40):
                scene = _overlapping_scene(n_splats, image_size)
                leaves = [tensor.clone().requires_grad_(True) for tensor in (scene[0], scene[1], scene[2], scene[4])]
                rendered = render_gaussians(leaves[0], leaves[1], leaves[2], scene[3], leaves[3], *scene[5:], image_size)
                weights = torch.linspace(-1.0, 2.0, rendered.numel()).view(rendered.shape)
                (rendered * weights).sum().backward()

                reference_leaves = [tensor.clone().requires_grad_(True) for tensor in (scene[0], scene[1], scene[2], scene[4])]
                reference = _render_gaussians_in_place_reference(*reference_leaves, *scene[5:], image_size)
                (reference * weights).sum().backward()
                assert torch.equal(rendered, reference)
                for leaf, reference_leaf in zip(leaves, reference_leaves):
                    assert torch.equal(leaf.grad, reference_leaf.grad)
        monkeypatch.undo()

        large_image = (96, 128)
        copy_slices = []
        for n_splats in (3, 40):
            positions, colors, scales, rotations, opacities, intrinsics, extrinsics = _overlapping_scene(n_splats, large_image)
            rendered = render_gaussians(
                positions.requires_grad_(True), colors, scales * 0.1, rotations, opacities, intrinsics, extrinsics, large_image
            )
            copy_slices.append(_count_grad_nodes(rendered, "CopySlices"))
        assert copy_slices[0] == copy_slices[1], f"in-place full-image writes grow with the splat count: {copy_slices}"


def _overlapping_scene(n_splats, image_size):
    """Deterministic splats in front of the camera with heavily overlapping footprints."""
    generator = torch.Generator().manual_seed(n_splats)
    positions = torch.cat(
        [(torch.rand(n_splats, 2, generator=generator) - 0.5) * 0.8, torch.rand(n_splats, 1, generator=generator) * 3 + 2],
        dim=1,
    )
    colors = torch.rand(n_splats, 3, generator=generator)
    scales = torch.rand(n_splats, 3, generator=generator) * 0.2 + 0.02
    opacities = torch.rand(n_splats, 1, generator=generator) * 0.9 + 0.05
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n_splats, 1)
    intrinsics = torch.tensor([[FX, 0.0, image_size[1] / 2.0], [0.0, FY, image_size[0] / 2.0], [0.0, 0.0, 1.0]])
    return positions, colors, scales, rotations, opacities, intrinsics, torch.eye(4)


def _count_grad_nodes(output, node_name):
    """Count autograd nodes of one type reachable from ``output``."""
    seen, stack, count = set(), [output.grad_fn], 0
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        count += type(node).__name__ == node_name
        stack.extend(next_node for next_node, _ in node.next_functions)
    return count


def _render_gaussians_in_place_reference(positions, colors, scales, opacities, intrinsics, extrinsics, image_size):
    """Sequential front-to-back compositing written into full-image buffers (the original formulation)."""
    from transformation_portal.spatial_ai.reconstruction.gaussian_rasterizer import _compute_splat_bounds_batch

    height, width = image_size
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(positions.shape[0], 1)
    mean_2d, cov_2d, depths, valid = project_gaussians_2d(positions, scales, rotations, intrinsics, extrinsics, image_size)
    order = torch.argsort(depths[valid], stable=True)
    mean_2d, cov_2d = mean_2d[valid][order], cov_2d[valid][order]
    colors, opacities = colors[valid][order], opacities[valid][order]
    cov_2d_inv = torch.inverse(cov_2d)
    rendered = torch.zeros(height, width, 3)
    accumulated_alpha = torch.zeros(height, width)
    for i, bounds in enumerate(_compute_splat_bounds_batch(mean_2d, cov_2d, image_size)):
        if bounds is None:
            continue
        u_min, u_max, v_min, v_max = bounds
        u_coords = torch.arange(u_min, u_max, dtype=mean_2d.dtype)
        v_coords = torch.arange(v_min, v_max, dtype=mean_2d.dtype)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")
        diff = torch.stack([u_grid, v_grid], dim=-1) - mean_2d[i].view(1, 1, 2)
        tmp = torch.einsum("...c,cd->...d", diff, cov_2d_inv[i])
        alpha_i = torch.exp(-0.5 * (tmp * diff).sum(dim=-1)) * opacities[i, 0]
        rendered_patch = rendered[v_min:v_max, u_min:u_max, :]
        alpha_patch = accumulated_alpha[v_min:v_max, u_min:u_max]
        transmittance = 1.0 - alpha_patch
        rendered[v_min:v_max, u_min:u_max, :] = (
            rendered_patch + alpha_i.unsqueeze(-1) * transmittance.unsqueeze(-1) * colors[i]
        )
        accumulated_alpha[v_min:v_max, u_min:u_max] = alpha_patch + alpha_i * transmittance
    return torch.clamp(rendered, 0.0, 1.0)


@pytest.mark.skipif(not torch.cuda.is_available() and not torch.backends.mps.is_available(), reason="No GPU available")
class TestDeviceCompatibility:
    """Test GPU/MPS compatibility."""

    def test_mps_rendering(self):
        """Test rendering on MPS (Apple Silicon)."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        device = "mps"

        positions = torch.tensor([[0.0, 0.0, 5.0]]).to(device)
        colors = torch.tensor([[1.0, 0.0, 0.0]]).to(device)
        scales = torch.tensor([[0.5, 0.5, 0.5]]).to(device)
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).to(device)
        opacities = torch.tensor([[1.0]]).to(device)

        intrinsics = torch.tensor([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]]).to(device)

        extrinsics = torch.eye(4).to(device)
        image_size = IMAGE_SIZE

        rendered = render_gaussians(
            positions, colors, scales, rotations, opacities, intrinsics, extrinsics, image_size, device=device
        )

        assert rendered.device.type == "mps"
        assert rendered.shape == (H, W, 3)
        assert not torch.isnan(rendered).any()
