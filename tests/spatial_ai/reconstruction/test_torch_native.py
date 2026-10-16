"""Explicit native reconstruction contracts; these are not scene acceptance."""

# Optional Torch must be guarded before importing the native backend.
# pylint: disable=wrong-import-position

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.ml, pytest.mark.unit]

from transformation_portal.spatial_ai.reconstruction import (  # noqa: E402
    CameraParams,
    GaussianBackend,
    PLYExporter,
    ReconstructionInput,
)
from transformation_portal.spatial_ai.reconstruction.gaussian_rasterizer import render_gaussians_native  # noqa: E402


def metric_input():
    image = np.full((4, 4, 3), [0.2, 0.5, 0.7], dtype=np.float32)
    intrinsics = np.array([[4, 0, 1.5], [0, 4, 1.5], [0, 0, 1]], dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)
    shifted = extrinsics.copy()
    shifted[0, 3] = 0.1
    cameras = [CameraParams(intrinsics, pose, 4, 4) for pose in (extrinsics, shifted)]
    return ReconstructionInput(
        images=[image.copy(), image.copy()],
        cameras=cameras,
        depth_maps=[np.full((4, 4), 2, dtype=np.float32) for _ in cameras],
        gamma=1.0,
        tier="apex_research",
    )


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_native_composites_near_surface_first_with_finite_gradients(chunk_size):
    positions = torch.tensor([[0, 0, 1], [0, 0, 2]], dtype=torch.float32, requires_grad=True)
    colors = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.float32, requires_grad=True)
    scales = torch.full((2, 3), 0.1, requires_grad=True)
    opacities = torch.full((2, 1), 0.5, requires_grad=True)
    rendered = render_gaussians_native(
        positions,
        colors,
        scales,
        torch.tensor([[1, 0, 0, 0]] * 2, dtype=torch.float32),
        opacities,
        torch.eye(3),
        torch.eye(4),
        (1, 1),
        chunk_size=chunk_size,
    )
    torch.testing.assert_close(rendered[0, 0], torch.tensor([0.25, 0, 0.5]))
    rendered.sum().backward()
    for parameter in (positions, colors, scales, opacities):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    torch.testing.assert_close(opacities.grad[:, 0], torch.tensor([0.5, 0.5]))


def test_native_invisible_scene_has_zero_connected_gradients():
    positions = torch.tensor([[0, 0, -1]], dtype=torch.float32, requires_grad=True)
    rendered = render_gaussians_native(
        positions,
        torch.ones(1, 3),
        torch.ones(1, 3),
        torch.tensor([[1, 0, 0, 0]]),
        torch.ones(1, 1),
        torch.eye(3),
        torch.eye(4),
        (2, 2),
    )
    rendered.sum().backward()
    torch.testing.assert_close(positions.grad, torch.zeros_like(positions))


@pytest.mark.parametrize("iterations", [0, -1, True, 1.5])
def test_native_rejects_nonpositive_or_noninteger_iterations(iterations):
    backend = GaussianBackend(device="cpu", implementation="torch_native")
    with pytest.raises(ValueError, match="positive integer"):
        backend.reconstruct(metric_input(), iterations=iterations)


def test_native_requires_metric_depth_and_preserves_pinned_loader_failure():
    inputs = metric_input()
    inputs.depth_maps = None
    backend = GaussianBackend(device="cpu", implementation="torch_native")
    with pytest.raises(ValueError, match="metric depth"):
        backend.reconstruct(inputs, iterations=1)
    revision = "54c035f7834b564019656c3e3fcc3646292f727d"
    with pytest.raises(ValueError, match="cannot load or attest"):
        GaussianBackend(device="cpu", implementation="torch_native", model_revision=revision)
    legacy = GaussianBackend(device="cpu", model_revision=revision)
    with pytest.raises(NotImplementedError, match="checkpoint integration"):
        legacy.reconstruct(metric_input(), iterations=1)


@pytest.mark.parametrize("recipe", ["isotropic_full_batch_v1", "anisotropic_shuffled_rgb_v1", "anisotropic_shuffled_rgbd_v1"])
def test_native_rejects_mixed_training_geometry_before_initialization(monkeypatch, recipe):
    from transformation_portal.spatial_ai.reconstruction import gaussian_backend

    inputs = metric_input()
    inputs.images[1] = inputs.images[1][:2, :2].copy()
    inputs.depth_maps[1] = inputs.depth_maps[1][:2, :2].copy()
    inputs.cameras[1].width = inputs.cameras[1].height = 2

    def unexpected_initialization(*args, **kwargs):
        pytest.fail("Mixed native training geometry must fail before metric initialization")

    monkeypatch.setattr(gaussian_backend, "initialize_metric_splats", unexpected_initialization)
    backend = GaussianBackend(device="cpu", implementation="torch_native", native_recipe=recipe, optimization_seed=7)
    with pytest.raises(ValueError, match="matching training image spatial dimensions"):
        backend.reconstruct(inputs, iterations=1)


@pytest.mark.parametrize("invalid", ["nan_rgb", "zero_depth", "distortion", "nonrigid_pose", "pbr", "empty_mask"])
def test_native_rejects_unsupported_or_invalid_inputs(invalid):
    inputs = metric_input()
    if invalid == "nan_rgb":
        inputs.images[0][0, 0, 0] = np.nan
    elif invalid == "zero_depth":
        inputs.depth_maps[0][0, 0] = 0
    elif invalid == "distortion":
        inputs.cameras[0].distortion = np.array([0.1, 0, 0, 0], dtype=np.float32)
    elif invalid == "nonrigid_pose":
        inputs.cameras[0].extrinsics[0, 0] = 2
    elif invalid == "pbr":
        # Keep the input contract valid; reject the unsupported native capability.
        inputs.material_maps = [{}] * 2
    else:
        inputs.masks = [np.zeros((4, 4), dtype=bool)] * 2
    backend = GaussianBackend(device="cpu", implementation="torch_native")
    with pytest.raises(ValueError, match="torch_native"):
        backend.reconstruct(inputs, iterations=1)


def test_native_masks_can_exclude_invalid_depth():
    inputs = metric_input()
    inputs.depth_maps[0][0, 0] = 0
    inputs.masks = [np.ones((4, 4), dtype=bool) for _ in range(2)]
    inputs.masks[0][0, 0] = False
    backend = GaussianBackend(device="cpu", implementation="torch_native", optimization_max_gaussians=8)
    scene = backend.reconstruct(inputs, iterations=1)
    assert scene.splats.num_gaussians == 8
    assert scene.splats.metadata["input_depth_support_per_view"] == [15, 16]


@pytest.mark.parametrize("mutation", ["pose_projective_row", "pose_homogeneous_scale", "pose_shape", "intrinsics_shape"])
def test_native_revalidates_mutated_camera_before_initialization(monkeypatch, mutation):
    from transformation_portal.spatial_ai.reconstruction import gaussian_backend

    inputs = metric_input()
    camera = inputs.cameras[0]
    if mutation == "pose_projective_row":
        camera.extrinsics[3, 0] = 0.2
    elif mutation == "pose_homogeneous_scale":
        camera.extrinsics[3, 3] = 2
    elif mutation == "pose_shape":
        camera.extrinsics = np.eye(3, dtype=np.float32)
    else:
        camera.intrinsics = np.eye(4, dtype=np.float32)

    def unexpected_initialization(*args, **kwargs):
        pytest.fail("Mutated invalid cameras must fail before metric unprojection")

    monkeypatch.setattr(gaussian_backend, "initialize_metric_splats", unexpected_initialization)
    backend = GaussianBackend(device="cpu", implementation="torch_native")
    with pytest.raises(ValueError, match="Extrinsics|Intrinsics"):
        backend.reconstruct(inputs, iterations=1)


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_native_gaussian_cap_requires_positive_integer(cap):
    with pytest.raises(ValueError, match="positive integer"):
        GaussianBackend(device="cpu", implementation="torch_native", optimization_max_gaussians=cap)


def test_native_dense_training_budget_fails_before_optimization(monkeypatch):
    from transformation_portal.spatial_ai.reconstruction import torch_native

    monkeypatch.setattr(torch_native, "MAX_NATIVE_TRAINING_GAUSSIAN_PIXELS", 1)
    backend = GaussianBackend(device="cpu", implementation="torch_native", optimization_max_gaussians=8)

    def unexpected_optimization(*args, **kwargs):
        pytest.fail("Over-budget inputs must not initialize optimization tensors")

    monkeypatch.setattr(backend, "_optimize", unexpected_optimization)
    with pytest.raises(ValueError, match="Gaussian-pixel budget"):
        backend.reconstruct(metric_input(), iterations=1)


def test_native_metric_spacing_is_rigid_invariant_and_batches_memory(monkeypatch):
    from transformation_portal.spatial_ai.reconstruction import torch_native

    positions = np.column_stack([np.arange(300), np.zeros(300), np.zeros(300)]).astype(np.float32)
    original_einsum = np.einsum
    rows = []

    def bounded_einsum(subscripts, difference, other):
        rows.append(difference.shape[0])
        assert difference.shape[0] <= 128
        return original_einsum(subscripts, difference, other)

    monkeypatch.setattr(torch_native.np, "einsum", bounded_einsum)
    scales = torch_native.metric_neighbor_scales(positions)
    assert rows == [128, 128, 44]
    assert scales[150, 0] == pytest.approx(np.sqrt((1 + 1 + 4) / 3))
    rotated = positions[:, [1, 0, 2]] + np.array([7, -3, 2], dtype=np.float32)
    np.testing.assert_allclose(torch_native.metric_neighbor_scales(rotated), scales)
    with pytest.raises(ValueError, match="metric spacing requires"):
        torch_native.metric_neighbor_scales(np.zeros((torch_native.MAX_NATIVE_SPACING_POINTS + 1, 3), dtype=np.float32))


def test_native_frees_each_view_graph_before_rendering_next(monkeypatch):
    from transformation_portal.spatial_ai.reconstruction import gaussian_backend, torch_native

    inputs = metric_input()
    backend = GaussianBackend(device="cpu", implementation="torch_native", optimization_seed=42, optimization_max_gaussians=8)
    initial = torch_native.initialize_metric_splats(inputs, True, 8, 42)
    original_render = gaussian_backend.render_gaussians_native
    events = []

    def tracked_render(*args, **kwargs):
        output = original_render(*args, **kwargs)
        if torch.is_grad_enabled():
            events.append("forward")
            output.register_hook(lambda gradient: events.append("backward"))
        return output

    monkeypatch.setattr(gaussian_backend, "render_gaussians_native", tracked_render)
    optimized, _, _, _ = backend._optimize(initial, inputs, 1, True, False)
    assert events == ["forward", "backward", "forward", "backward"]
    # Log scales avoid the one-step scale collapse that a linear 0.005m Adam
    # update can cause for small sensor-surface splats.
    ratio = optimized.scales / initial.scales
    assert np.all(ratio >= np.exp(-backend.DEFAULT_SCALING_LR) - 1e-6)
    assert np.all(ratio <= np.exp(backend.DEFAULT_SCALING_LR) + 1e-6)


def test_native_optimizes_deterministically_and_reports_actual_saved_scene_rmse(tmp_path):
    inputs = metric_input()
    backend = GaussianBackend(device="cpu", implementation="torch_native", optimization_seed=42, optimization_max_gaussians=8)
    np.random.seed(712)
    ambient_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    scene = backend.reconstruct(inputs, iterations=3)
    restored_state = np.random.get_state()
    assert ambient_state[0] == restored_state[0]
    np.testing.assert_array_equal(ambient_state[1], restored_state[1])
    assert ambient_state[2:] == restored_state[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    repeat = backend.reconstruct(inputs, iterations=3)
    np.testing.assert_array_equal(scene.splats.positions, repeat.splats.positions)
    np.testing.assert_array_equal(scene.splats.colors, repeat.splats.colors)
    errors = [
        np.mean((backend.render_view(scene, camera) - image) ** 2) for camera, image in zip(inputs.cameras, inputs.images)
    ]
    assert scene.rmse == pytest.approx(np.sqrt(np.mean(errors)), rel=1e-6)
    assert scene.splats.metadata["loss_history"][-1] < scene.splats.metadata["loss_history"][0]
    assert scene.splats.metadata["rmse_scope"] == "all_training_views_post_update"
    assert not backend._model_loaded
    provenance = scene.metadata["reconstruction_provenance"]
    assert provenance["pretrained_checkpoint"] is None
    assert provenance["initialization"] == "metric_depth_from_scratch"
    assert provenance["optimization_seed"] == 42
    assert len(provenance["views"]) == 2
    assert all(len(value) == 64 for value in provenance["source_sha256"].values())

    output = PLYExporter().export(scene, tmp_path / "native.ply")
    sidecar = json.loads(output.with_suffix(".provenance.json").read_text())
    assert sidecar["output_hash_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert sidecar["implementation"] == "torch_native"
    assert sidecar["reconstruction_provenance"] == provenance
    plyfile = pytest.importorskip("plyfile")
    vertex = plyfile.PlyData.read(output)["vertex"]
    for field, names in [
        ("positions", ("x", "y", "z")),
        ("scales", ("scale_x", "scale_y", "scale_z")),
        ("rotations", ("rot_w", "rot_x", "rot_y", "rot_z")),
        ("opacities", ("opacity",)),
    ]:
        np.testing.assert_array_equal(np.column_stack([vertex[name] for name in names]), getattr(scene.splats, field))
    reloaded_colors = np.column_stack([vertex[name] for name in ("red", "green", "blue")]).astype(np.float32) / 255
    assert np.max(np.abs(reloaded_colors - scene.splats.colors)) <= 1 / 255


def test_multiview_pipeline_forwards_explicit_native_settings(tmp_path, monkeypatch):
    from transformation_portal.core.geometry import CoreCameraParams, MultiViewReconstructionRequest
    from transformation_portal.spatial_ai.orchestration import PipelineConfig, SpatialAIPipeline

    inputs = metric_input()
    cameras = [
        CoreCameraParams(
            fx=4,
            fy=4,
            cx=1.5,
            cy=1.5,
            width=4,
            height=4,
            source="explicit",
            extrinsics=camera.extrinsics,
        )
        for camera in inputs.cameras
    ]
    pipeline = SpatialAIPipeline(
        PipelineConfig(
            tier="apex_research",
            stages=["reconstruction"],
            reconstruction={"implementation": "torch_native", "optimization_max_gaussians": 8, "iterations": 2},
        )
    )
    monkeypatch.setattr(pipeline.resource_manager, "select_device", lambda: "cpu")
    request = MultiViewReconstructionRequest(
        cameras=cameras,
        images=inputs.images,
        depth_maps=inputs.depth_maps,
        tier="apex_research",
        optimization_seed=42,
    )
    result = pipeline.process_multiview(request, tmp_path)
    assert result.scene.metadata["implementation"] == "torch_native"
    assert result.scene.splats.num_gaussians == 8
    assert result.scene.iteration == 2
    assert result.scene.metadata["optimization_seed"] == 42
    assert json.loads(result.sidecar_path.read_text())["implementation"] == "torch_native"
    np.testing.assert_array_equal(result.scene.cameras[1].extrinsics, inputs.cameras[1].extrinsics)


def test_native_multiview_pipeline_rejects_missing_calibrated_poses(tmp_path):
    from transformation_portal.core.geometry import CoreCameraParams, MultiViewReconstructionRequest
    from transformation_portal.spatial_ai.orchestration import PipelineConfig, SpatialAIPipeline
    from transformation_portal.spatial_ai.orchestration.error_handler import PipelineError

    inputs = metric_input()
    cameras = [CoreCameraParams(fx=4, fy=4, cx=1.5, cy=1.5, width=4, height=4, source="explicit")] * 2
    request = MultiViewReconstructionRequest(
        cameras=cameras, images=inputs.images, depth_maps=inputs.depth_maps, tier="apex_research"
    )
    pipeline = SpatialAIPipeline(
        PipelineConfig(
            tier="apex_research", stages=["reconstruction"], reconstruction={"implementation": "torch_native", "iterations": 1}
        )
    )
    with pytest.raises(PipelineError, match="supplied world-to-camera extrinsics"):
        pipeline.process_multiview(request, tmp_path)
