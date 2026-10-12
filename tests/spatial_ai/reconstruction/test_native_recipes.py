"""Explicit anisotropic native contracts; genuine scene acceptance is separate."""

# Optional Torch must be guarded before importing the native backend.
# pylint: disable=wrong-import-position

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.ml, pytest.mark.unit]

from transformation_portal.spatial_ai.reconstruction import (
    CameraParams,
    GaussianBackend,
    GaussianSplat,
    PLYExporter,
    ReconstructionInput,
    Scene3D,
    torch_native,
)
from transformation_portal.spatial_ai.reconstruction.gaussian_rasterizer import (
    render_gaussians,
    render_gaussians_anisotropic_native,
    render_gaussians_native,
)


def inputs():
    image = np.full((4, 4, 3), [0.2, 0.5, 0.7], dtype=np.float32)
    K = np.array([[4, 0, 1.5], [0, 4, 1.5], [0, 0, 1]], dtype=np.float32)
    poses = [np.eye(4, dtype=np.float32) for _ in range(2)]
    poses[1][0, 3] = 0.1
    return ReconstructionInput(
        images=[image.copy() for _ in poses],
        cameras=[CameraParams(K.copy(), E, 4, 4) for E in poses],
        depth_maps=[np.full((4, 4), 2, dtype=np.float32) for _ in poses],
        gamma=1.0,
        tier="apex_research",
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 128])
def test_full_covariance_metric_depth_occlusion_and_order(chunk_size):
    p = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    c = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    s = torch.ones(2, 3) * 0.1
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
    o = torch.ones(2, 1) * 0.5
    args = (p, c, s, q, o, torch.eye(3), torch.eye(4), (1, 1))
    rgb, depth, alpha = render_gaussians_anisotropic_native(*args, chunk_size=chunk_size)
    torch.testing.assert_close(rgb[0, 0], torch.tensor([0.25, 0, 0.5]))
    torch.testing.assert_close(depth[0, 0], torch.tensor(4 / 3))
    torch.testing.assert_close(alpha[0, 0], torch.tensor(0.75))
    reversed_args = tuple(x.flip(0) for x in args[:5]) + args[5:]
    outputs = render_gaussians_anisotropic_native(*reversed_args, chunk_size=chunk_size)
    for result, expected in zip(outputs, (rgb, depth, alpha)):
        torch.testing.assert_close(result, expected)


def test_full_covariance_off_diagonal_camera_rotation_and_gradients():
    p = torch.tensor([[0.0, 0.0, 2.0]], requires_grad=True)
    c = torch.tensor([[0.8, 0.3, 0.1]], requires_grad=True)
    s = torch.tensor([[0.35, 0.1, 0.1]], requires_grad=True)
    q = torch.tensor([[np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)]], dtype=torch.float32, requires_grad=True)
    o = torch.tensor([[0.7]], requires_grad=True)
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    rgb, depth, alpha = render_gaussians_anisotropic_native(p, c, s, q, o, K, torch.eye(4), (9, 9))
    assert rgb[5, 5, 0] > rgb[5, 3, 0]
    (rgb[5, 4].square().sum() + 0.1 * (depth[4, 4] - 2.2).square()).backward()
    for parameter in (p, c, s, q, o):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    E = torch.eye(4)
    E[:2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    base = render_gaussians_anisotropic_native(
        p.detach(), c.detach(), s.detach(), identity, o.detach(), K, torch.eye(4), (9, 9)
    )[0]
    rotated = render_gaussians_anisotropic_native(p.detach(), c.detach(), s.detach(), identity, o.detach(), K, E, (9, 9))[0]
    torch.testing.assert_close(rotated, base.transpose(0, 1), rtol=1e-5, atol=1e-6)


def test_central_isotropic_full_covariance_matches_existing_native_renderer():
    p = torch.tensor([[0.0, 0.0, 2.0]])
    args = (
        p,
        torch.ones(1, 3) * 0.4,
        torch.ones(1, 3) * 0.2,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.ones(1, 1) * 0.7,
        torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]),
        torch.eye(4),
        (9, 9),
    )
    torch.testing.assert_close(render_gaussians_anisotropic_native(*args)[0], render_gaussians_native(*args))


def test_empty_visibility_metric_depth_outputs_keep_finite_connected_gradients():
    params = [
        torch.tensor([[0.0, 0.0, -2.0]], requires_grad=True),
        torch.ones(1, 3, requires_grad=True),
        torch.ones(1, 3, requires_grad=True),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True),
        torch.ones(1, 1, requires_grad=True),
    ]
    outputs = render_gaussians_anisotropic_native(*params, torch.eye(3), torch.eye(4), (2, 2))
    assert all(torch.isfinite(x).all() and not x.any() for x in outputs)
    sum(x.sum() for x in outputs).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)


def test_depth_loss_uses_only_fixed_supplied_mask_with_metre_normalization():
    target = torch.tensor([[2.0, 0.0], [float("nan"), 3.0]])
    mask = torch.tensor([[True, False], [False, True]])
    prediction = torch.tensor([[2.2, 100.0], [100.0, 2.7]], requires_grad=True)
    loss = torch_native.metric_depth_supervision_loss(prediction, target, mask)
    torch.testing.assert_close(loss, torch.tensor(0.065), rtol=0, atol=1e-6)
    loss.backward()
    assert not prediction.grad[~mask].any()
    with pytest.raises(ValueError, match="nonempty"):
        torch_native.metric_depth_supervision_loss(prediction, target, mask & False)
    with pytest.raises(ValueError, match="finite and positive"):
        torch_native.metric_depth_supervision_loss(prediction, target, mask | True)


@pytest.mark.parametrize("seed", [None, True, -1, 1.5])
def test_shuffled_recipes_require_explicit_valid_local_seed(seed):
    with pytest.raises(ValueError, match="optimization_seed"):
        GaussianBackend(
            device="cpu", implementation="torch_native", native_recipe="anisotropic_shuffled_rgb_v1", optimization_seed=seed
        )


def test_invalid_recipe_and_legacy_recipe_mismatch_fail_closed():
    with pytest.raises(ValueError, match="native_recipe"):
        GaussianBackend(device="cpu", implementation="torch_native", native_recipe="unknown")
    with pytest.raises(ValueError, match="torch_native"):
        GaussianBackend(device="cpu", native_recipe="anisotropic_shuffled_rgb_v1", optimization_seed=7)


@pytest.mark.parametrize("recipe", torch_native.NATIVE_RECIPES[1:])
def test_recipe_determinism_saved_rmse_provenance_and_render_identity(recipe, tmp_path):
    data = inputs()
    backend = GaussianBackend(
        device="cpu", implementation="torch_native", native_recipe=recipe, optimization_seed=7, optimization_max_gaussians=8
    )
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    scene = backend.reconstruct(data, iterations=5)
    repeat = backend.reconstruct(data, iterations=5)
    np.testing.assert_array_equal(scene.splats.positions, repeat.splats.positions)
    np.testing.assert_array_equal(scene.splats.colors, repeat.splats.colors)
    restored = np.random.get_state()
    np.testing.assert_array_equal(numpy_state[1], restored[1])
    assert numpy_state[2:] == restored[2:]
    torch.testing.assert_close(torch_state, torch.get_rng_state())
    errors = [np.mean((backend.render_view(scene, camera) - image) ** 2) for image, camera in zip(data.images, data.cameras)]
    assert scene.rmse == pytest.approx(np.sqrt(np.mean(errors)), rel=1e-6)
    details = scene.splats.metadata["native_optimizer"]
    assert details["performed_updates"] == 5
    rng = np.random.default_rng(7)
    expected = np.concatenate([rng.permutation(2) for _ in range(3)])[:5].tolist()
    assert details["camera_schedule"] == expected
    assert scene.metadata["native_recipe"] == recipe
    assert scene.metadata["reconstruction_provenance"]["native_recipe"] == recipe
    for wrong in (GaussianBackend(device="cpu"), GaussianBackend(device="cpu", implementation="torch_native")):
        with pytest.raises(ValueError, match="recipe"):
            wrong.render_view(scene, data.cameras[0])
    output = PLYExporter().export(scene, tmp_path / (recipe + ".ply"))
    sidecar = json.loads(output.with_suffix(".provenance.json").read_text())
    assert sidecar["native_recipe"] == recipe
    assert sidecar["reconstruction_provenance"] == scene.metadata["reconstruction_provenance"]
    assert sidecar["output_hash_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    plyfile = pytest.importorskip("plyfile")
    v = plyfile.PlyData.read(output)["vertex"]
    arrays = {
        name: np.column_stack([v[field] for field in fields])
        for name, fields in {
            "positions": ("x", "y", "z"),
            "scales": ("scale_x", "scale_y", "scale_z"),
            "rotations": ("rot_w", "rot_x", "rot_y", "rot_z"),
            "opacities": ("opacity",),
        }.items()
    }
    for name, array in arrays.items():
        np.testing.assert_array_equal(array, getattr(scene.splats, name))
    colors = np.column_stack([v[field] for field in ("red", "green", "blue")]).astype(np.float32) / 255
    reloaded = Scene3D(
        GaussianSplat(**arrays, colors=colors, metadata={"native_recipe": recipe}),
        cameras=data.cameras,
        rmse=scene.rmse,
        iteration=scene.iteration,
        convergence=scene.convergence,
        metadata={
            "implementation": "torch_native",
            "native_recipe": sidecar["native_recipe"],
            "reconstruction_provenance": sidecar["reconstruction_provenance"],
        },
    )
    assert (
        np.max(np.abs(backend.render_view(reloaded, data.cameras[0]) - backend.render_view(scene, data.cameras[0])))
        <= 1 / 255 + 1e-6
    )


def test_default_recipe_is_identical_to_explicit_isotropic_default():
    data = inputs()
    kwargs = dict(device="cpu", implementation="torch_native", optimization_seed=7, optimization_max_gaussians=8)
    default = GaussianBackend(**kwargs).reconstruct(data, iterations=2)
    explicit = GaussianBackend(**kwargs, native_recipe=torch_native.NATIVE_DEFAULT_RECIPE).reconstruct(data, iterations=2)
    for name in ("positions", "colors", "scales", "rotations", "opacities"):
        np.testing.assert_array_equal(getattr(default.splats, name), getattr(explicit.splats, name))
    assert default.rmse == explicit.rmse


@pytest.mark.parametrize("recipe", torch_native.NATIVE_RECIPES)
def test_public_validator_matches_saved_native_recipe_without_fitting(monkeypatch, recipe):
    from transformation_portal.spatial_ai.reconstruction import GeometricValidator

    data = inputs()
    backend = GaussianBackend(
        device="cpu",
        implementation="torch_native",
        native_recipe=recipe,
        optimization_seed=7,
        optimization_max_gaussians=8,
    )
    scene = backend.reconstruct(data, iterations=2)
    rendered = [backend.render_view(scene, camera) for camera in data.cameras]
    expected = GeometricValidator().compute_rmse(scene, data.images, rendered_images=rendered)

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Public saved-scene validation must not fit or change the model")

    monkeypatch.setattr(GaussianBackend, "reconstruct", unexpected_fit)
    actual = GeometricValidator().compute_rmse(scene, data.images)

    assert actual == pytest.approx(expected, rel=1e-6)
    assert actual == pytest.approx(scene.rmse, rel=1e-6)
    assert scene.metadata["native_recipe"] == recipe
    assert scene.iteration == 2


def test_known_default_native_scene_rejects_legacy_renderer_but_unmarked_legacy_scene_remains_supported():
    data = inputs()
    native = GaussianBackend(device="cpu", implementation="torch_native", optimization_seed=7, optimization_max_gaussians=8)
    scene = native.reconstruct(data, iterations=1)
    legacy = GaussianBackend(device="cpu")
    with pytest.raises(ValueError, match="implementation"):
        legacy.render_view(scene, data.cameras[0])
    assert np.isfinite(native.render_view(scene, data.cameras[0])).all()
    unmarked = Scene3D(
        scene.splats,
        cameras=scene.cameras,
        rmse=scene.rmse,
        iteration=scene.iteration,
        convergence=scene.convergence,
        metadata={},
    )
    assert np.isfinite(legacy.render_view(unmarked, data.cameras[0])).all()
    unmarked.metadata["implementation"] = "legacy"
    with pytest.raises(ValueError, match="implementation"):
        native.render_view(unmarked, data.cameras[0])


def test_rgbd_rejects_lower_objective_checkpoint_with_real_support_loss():
    K = np.eye(3, dtype=np.float32)
    E = np.eye(4, dtype=np.float32)
    data = ReconstructionInput(
        images=[np.zeros((1, 1, 3), dtype=np.float32)] * 2,
        cameras=[CameraParams(K.copy(), E.copy(), 1, 1)] * 2,
        depth_maps=[np.full((1, 1), 2, dtype=np.float32)] * 2,
        gamma=1.0,
        tier="apex_research",
    )
    initial = GaussianSplat(
        positions=np.array([[0, 0, 2], [0, 0, 2]], dtype=np.float32),
        colors=np.ones((2, 3), dtype=np.float32),
        scales=np.ones((2, 3), dtype=np.float32) * 0.1,
        rotations=np.array([[1, 0, 0, 0]] * 2, dtype=np.float32),
        opacities=np.full((2, 1), 0.5, dtype=np.float32),
    )
    result, _, _, _ = torch_native.optimize_shuffled_metric_splats(
        initial, data, 12, "cpu", 7, "anisotropic_shuffled_rgbd_v1", True, 0.02
    )
    details = result.metadata["native_optimizer"]
    assert any(not checkpoint["coverage_guard_passed"] for checkpoint in details["checkpoints"])
    assert details["selected_checkpoint_step"] < 12
    assert all(
        actual >= original - 0.02
        for actual, original in zip(details["selected_coverage_per_view"], details["initial_coverage_per_view"])
    )


def test_shuffled_update_cap_fails_before_metric_initialization(monkeypatch):
    from transformation_portal.spatial_ai.reconstruction import gaussian_backend

    def unexpected(*args, **kwargs):
        pytest.fail("Over-budget update counts must fail before initialization")

    monkeypatch.setattr(gaussian_backend, "initialize_metric_splats", unexpected)
    backend = GaussianBackend(
        device="cpu", implementation="torch_native", native_recipe="anisotropic_shuffled_rgb_v1", optimization_seed=7
    )
    with pytest.raises(ValueError, match="optimizer updates"):
        backend.reconstruct(inputs(), iterations=torch_native.MAX_NATIVE_SHUFFLED_UPDATES + 1)


def test_recorded_recipe_disagreement_fails_before_rendering():
    backend = GaussianBackend(
        device="cpu",
        implementation="torch_native",
        native_recipe="anisotropic_shuffled_rgb_v1",
        optimization_seed=7,
        optimization_max_gaussians=8,
    )
    data = inputs()
    scene = backend.reconstruct(data, iterations=1)
    scene.metadata["native_recipe"] = "anisotropic_shuffled_rgbd_v1"
    with pytest.raises(ValueError, match="recipe"):
        backend.render_view(scene, data.cameras[0])


def test_multiview_pipeline_forwards_recipe_and_preserves_saved_identity(tmp_path, monkeypatch):
    from transformation_portal.core.geometry import CoreCameraParams, MultiViewReconstructionRequest
    from transformation_portal.spatial_ai.orchestration import PipelineConfig, SpatialAIPipeline

    data = inputs()
    cameras = [
        CoreCameraParams(fx=4, fy=4, cx=1.5, cy=1.5, width=4, height=4, source="explicit", extrinsics=camera.extrinsics)
        for camera in data.cameras
    ]
    pipeline = SpatialAIPipeline(
        PipelineConfig(
            tier="apex_research",
            stages=["reconstruction"],
            reconstruction={
                "implementation": "torch_native",
                "native_recipe": "anisotropic_shuffled_rgb_v1",
                "optimization_max_gaussians": 8,
                "iterations": 3,
            },
        )
    )
    monkeypatch.setattr(pipeline.resource_manager, "select_device", lambda: "cpu")
    request = MultiViewReconstructionRequest(
        images=data.images, cameras=cameras, depth_maps=data.depth_maps, tier="apex_research", optimization_seed=7
    )
    result = pipeline.process_multiview(request, tmp_path)
    assert result.scene.metadata["native_recipe"] == "anisotropic_shuffled_rgb_v1"
    assert json.loads(result.sidecar_path.read_text())["native_recipe"] == result.scene.metadata["native_recipe"]
    assert result.scene.splats.metadata["native_optimizer"]["performed_updates"] == 3


def identity_scene(recipe="anisotropic_shuffled_rgb_v1"):
    splats = GaussianSplat(
        positions=np.array([[0, 0, 2]], dtype=np.float32),
        colors=np.ones((1, 3), dtype=np.float32) * 0.5,
        scales=np.ones((1, 3), dtype=np.float32) * 0.1,
        rotations=np.array([[1, 0, 0, 0]], dtype=np.float32),
        opacities=np.full((1, 1), 0.5, dtype=np.float32),
        metadata={"native_recipe": recipe, "native_optimizer": {"native_recipe": recipe}},
    )
    return Scene3D(
        splats,
        cameras=inputs().cameras,
        rmse=0.1,
        iteration=1,
        convergence="max_iterations",
        metadata={
            "implementation": "torch_native",
            "native_recipe": recipe,
            "reconstruction_provenance": {
                "implementation": "torch_native",
                "native_recipe": recipe,
                "optimizer": {"native_recipe": recipe},
            },
        },
    )


@pytest.mark.parametrize("recipe", [torch_native.NATIVE_DEFAULT_RECIPE, "anisotropic_shuffled_rgb_v1"])
@pytest.mark.parametrize(
    "mutation",
    [
        "pose_projective_row",
        "pose_homogeneous_scale",
        "pose_nonrigid",
        "pose_reflection",
        "pose_shape",
        "pose_nan",
        "intrinsics_shape",
        "intrinsics_dtype",
        "intrinsics_nan",
        "intrinsics_focal",
        "intrinsics_skew",
        "intrinsics_bottom_row",
        "distortion",
        "dimensions",
    ],
)
def test_native_novel_render_revalidates_camera_before_tensor_conversion(monkeypatch, recipe, mutation):
    scene = identity_scene(recipe)
    camera = CameraParams(np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32), 4, 4)
    if mutation == "pose_projective_row":
        camera.extrinsics[3, 0] = 0.2
    elif mutation == "pose_homogeneous_scale":
        camera.extrinsics[3, 3] = 2
    elif mutation == "pose_nonrigid":
        camera.extrinsics[0, 0] = 2
    elif mutation == "pose_reflection":
        camera.extrinsics[0, 0] = -1
    elif mutation == "pose_shape":
        camera.extrinsics = np.eye(3, dtype=np.float32)
    elif mutation == "pose_nan":
        camera.extrinsics[0, 3] = np.nan
    elif mutation == "intrinsics_shape":
        camera.intrinsics = np.eye(4, dtype=np.float32)
    elif mutation == "intrinsics_dtype":
        camera.intrinsics = camera.intrinsics.astype(np.float64)
    elif mutation == "intrinsics_nan":
        camera.intrinsics[0, 2] = np.nan
    elif mutation == "intrinsics_focal":
        camera.intrinsics[0, 0] = 0
    elif mutation == "intrinsics_skew":
        camera.intrinsics[0, 1] = 0.1
    elif mutation == "intrinsics_bottom_row":
        camera.intrinsics[2, 0] = 0.1
    elif mutation == "distortion":
        camera.distortion = np.array([0.1, 0, 0, 0], dtype=np.float32)
    else:
        camera.width = 0

    from transformation_portal.spatial_ai.reconstruction import gaussian_backend

    converted = []
    original = gaussian_backend._numpy_array_to_tensor

    def track_conversion(*args, **kwargs):
        converted.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(gaussian_backend, "_numpy_array_to_tensor", track_conversion)
    backend = GaussianBackend(device="cpu", implementation="torch_native", native_recipe=recipe, optimization_seed=7)
    with pytest.raises(ValueError):
        backend.render_view(scene, camera)
    assert not converted


@pytest.mark.parametrize(
    "implementation,recipe", [("legacy", None), *(("torch_native", r) for r in torch_native.NATIVE_RECIPES)]
)
def test_valid_novel_camera_preserves_renderer_exactly(implementation, recipe):
    scene = identity_scene(recipe or torch_native.NATIVE_DEFAULT_RECIPE)
    if implementation == "legacy":
        scene.metadata = {}
        scene.splats.metadata = {}
    K = np.array([[4, 0, 1.5], [0, 4, 1.5], [0, 0, 1]], dtype=np.float32)
    E = np.eye(4, dtype=np.float32)
    E[:2, :2] = [[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]]
    E[:3, 3] = [0.1, -0.1, 0.03]
    distortion = np.zeros(5, dtype=np.float32)
    if implementation == "legacy":
        distortion[0] = 0.1  # Preserve the legacy projection policy, which ignores distortion.
    camera = CameraParams(K, E, 4, 4, distortion=distortion)
    before = (K.copy(), E.copy(), camera.distortion.copy())
    kwargs = {"device": "cpu", "implementation": implementation, "optimization_seed": 7}
    if recipe is not None:
        kwargs["native_recipe"] = recipe
    backend = GaussianBackend(**kwargs)
    actual = backend.render_view(scene, camera)
    renderer = (
        render_gaussians
        if implementation == "legacy"
        else render_gaussians_native if recipe == torch_native.NATIVE_DEFAULT_RECIPE else render_gaussians_anisotropic_native
    )
    args = [
        torch.from_numpy(getattr(scene.splats, name)) for name in ("positions", "colors", "scales", "rotations", "opacities")
    ]
    expected = renderer(*args, torch.from_numpy(K), torch.from_numpy(E), (4, 4), device="cpu")
    if isinstance(expected, tuple):
        expected = expected[0]
    np.testing.assert_array_equal(actual, expected.numpy())
    for current, original in zip((K, E, camera.distortion), before):
        np.testing.assert_array_equal(current, original)


@pytest.mark.parametrize("write_sidecar", [False, True])
@pytest.mark.parametrize(
    "mismatch",
    ["scene", "splats", "provenance", "splat_optimizer", "provenance_optimizer", "unsupported", "legacy_implementation"],
)
def test_export_rejects_disagreeing_native_identity_before_any_output(tmp_path, write_sidecar, mismatch):
    scene = identity_scene()
    other = "anisotropic_shuffled_rgbd_v1"
    if mismatch == "scene":
        scene.metadata["native_recipe"] = other
    elif mismatch == "splats":
        scene.splats.metadata["native_recipe"] = other
    elif mismatch == "provenance":
        scene.metadata["reconstruction_provenance"]["native_recipe"] = other
    elif mismatch == "splat_optimizer":
        scene.splats.metadata["native_optimizer"]["native_recipe"] = other
    elif mismatch == "provenance_optimizer":
        scene.metadata["reconstruction_provenance"]["optimizer"]["native_recipe"] = other
    elif mismatch == "unsupported":
        scene.metadata["native_recipe"] = "future_unknown_recipe"
    else:
        scene.metadata["implementation"] = "legacy"
    output = tmp_path / "uncreated" / "scene.ply"
    with pytest.raises(ValueError, match="recipe|implementation"):
        PLYExporter().export(scene, output, write_sidecar=write_sidecar)
    assert not output.parent.exists()
    assert not output.exists()
    assert not output.with_suffix(".provenance.json").exists()
    backend = GaussianBackend(
        device="cpu", implementation="torch_native", native_recipe="anisotropic_shuffled_rgb_v1", optimization_seed=7
    )
    with pytest.raises(ValueError, match="recipe|implementation"):
        backend.render_view(scene, scene.cameras[0])


@pytest.mark.parametrize("location", ["scene", "splats", "provenance", "splat_optimizer", "provenance_optimizer"])
def test_export_preserves_explicit_recipe_without_implementation_marker(tmp_path, location):
    scene = identity_scene()
    recipe = scene.metadata["native_recipe"]
    scene.metadata = {}
    scene.splats.metadata = {}
    if location == "scene":
        scene.metadata["native_recipe"] = recipe
    elif location == "splats":
        scene.splats.metadata["native_recipe"] = recipe
    elif location == "provenance":
        scene.metadata["reconstruction_provenance"] = {"native_recipe": recipe}
    elif location == "splat_optimizer":
        scene.splats.metadata["native_optimizer"] = {"native_recipe": recipe}
    else:
        scene.metadata["reconstruction_provenance"] = {"optimizer": {"native_recipe": recipe}}
    output = PLYExporter().export(scene, tmp_path / (location + ".ply"))
    sidecar = json.loads(output.with_suffix(".provenance.json").read_text())
    assert sidecar["native_recipe"] == recipe
    assert sidecar["implementation"] == "torch_native"


def test_historical_native_default_without_recipe_descriptors_exports(tmp_path):
    scene = identity_scene()
    scene.splats.metadata = {}
    scene.metadata = {
        "implementation": "torch_native",
        "reconstruction_provenance": {"initialization": "metric_depth_from_scratch"},
    }
    output = PLYExporter().export(scene, tmp_path / "historical.ply")
    sidecar = json.loads(output.with_suffix(".provenance.json").read_text())
    assert sidecar["native_recipe"] == torch_native.NATIVE_DEFAULT_RECIPE
    assert sidecar["reconstruction_provenance"] == scene.metadata["reconstruction_provenance"]


def test_explicit_native_default_without_implementation_cannot_use_legacy_renderer():
    scene = identity_scene()
    scene.metadata = {"native_recipe": torch_native.NATIVE_DEFAULT_RECIPE}
    scene.splats.metadata = {}
    with pytest.raises(ValueError, match="recipe"):
        GaussianBackend(device="cpu").render_view(scene, scene.cameras[0])


@pytest.mark.parametrize("invalid", ["nan_input", "zero_scale", "opacity_outside", "invalid_chunk"])
def test_anisotropic_renderer_fails_closed_on_invalid_inputs(invalid):
    args = [
        torch.tensor([[0.0, 0.0, 2.0]]),
        torch.ones(1, 3),
        torch.ones(1, 3) * 0.1,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.ones(1, 1) * 0.5,
        torch.eye(3),
        torch.eye(4),
        (1, 1),
    ]
    options = {}
    if invalid == "nan_input":
        args[0][0, 0] = float("nan")
    elif invalid == "zero_scale":
        args[2][0, 0] = 0
    elif invalid == "opacity_outside":
        args[4][0, 0] = 1.1
    else:
        options["chunk_size"] = True
    with pytest.raises((RuntimeError, ValueError)):
        render_gaussians_anisotropic_native(*args, **options)
