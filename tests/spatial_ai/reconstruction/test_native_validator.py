"""Recipe-aware public validation with a stand-in backend; no Torch required."""

import builtins
import sys
from types import ModuleType

import numpy as np
import pytest

from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams, GaussianSplat, Scene3D
from transformation_portal.spatial_ai.reconstruction.geometric_validator import GeometricValidator
from transformation_portal.spatial_ai.reconstruction.protocol import NATIVE_DEFAULT_RECIPE, NATIVE_RECIPES

pytestmark = pytest.mark.unit


def scene_for_validation(recipe=None):
    metadata = {}
    splat_metadata = {}
    if recipe is not None:
        metadata = {
            "implementation": "torch_native",
            "native_recipe": recipe,
            "tier": "apex_research",
            "device": "cpu",
            "optimization_seed": 7,
            "reconstruction_provenance": {
                "implementation": "torch_native",
                "native_recipe": recipe,
                "optimization_seed": 7,
                "runtime": {"device": "cpu"},
                "optimizer": {"native_recipe": recipe},
            },
        }
        splat_metadata = {"native_recipe": recipe, "native_optimizer": {"native_recipe": recipe}}
    cameras = [CameraParams(np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32), size, size) for size in (1, 2)]
    splats = GaussianSplat(
        positions=np.array([[0, 0, 2]], dtype=np.float32),
        colors=np.full((1, 3), 0.5, dtype=np.float32),
        scales=np.full((1, 3), 0.1, dtype=np.float32),
        rotations=np.array([[1, 0, 0, 0]], dtype=np.float32),
        opacities=np.full((1, 1), 0.5, dtype=np.float32),
        metadata=splat_metadata,
    )
    return Scene3D(splats, cameras, 0.5, 1, "max_iterations", metadata)


def install_backend_standin(monkeypatch):
    events = {"config": [], "cameras": []}

    class StandInBackend:
        def __init__(self, **kwargs):
            events["config"].append(kwargs)

        def render_view(self, scene, camera):
            events["cameras"].append(camera)
            return np.full((camera.height, camera.width, 3), 0.25 * camera.width, dtype=np.float32)

        def reconstruct(self, *args, **kwargs):
            pytest.fail("Public validation must not perform fitting")

    module = ModuleType("transformation_portal.spatial_ai.reconstruction.gaussian_backend")
    setattr(module, "GaussianBackend", StandInBackend)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return events


def forbid_backend_import(monkeypatch):
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name.endswith("gaussian_backend"):
            pytest.fail("Invalid saved metadata must fail before importing the optional backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)


@pytest.mark.parametrize("recipe", NATIVE_RECIPES)
def test_public_rmse_selects_saved_recipe_and_full_rgb_denominator(monkeypatch, recipe):
    scene = scene_for_validation(recipe)
    events = install_backend_standin(monkeypatch)
    references = [np.zeros((camera.height, camera.width, 3), dtype=np.float32) for camera in scene.cameras]

    actual = GeometricValidator().compute_rmse(scene, references)

    assert actual == pytest.approx(np.sqrt((3 * 0.25**2 + 12 * 0.5**2) / 15))
    assert events["config"] == [
        {
            "tier": "apex_research",
            "device": "cpu",
            "optimization_seed": 7,
            "implementation": "torch_native",
            "native_recipe": recipe,
        }
    ]
    assert all(actual is expected for actual, expected in zip(events["cameras"], scene.cameras))
    assert scene.iteration == 1 and scene.convergence == "max_iterations"


def test_unmarked_legacy_keeps_exact_backend_defaults(monkeypatch):
    scene = scene_for_validation()
    scene.metadata.update(device="unsupported_legacy_note", optimization_seed=False)
    events = install_backend_standin(monkeypatch)
    GeometricValidator()._render_all_views(scene)
    assert events["config"] == [{"tier": "experimental"}]


@pytest.mark.parametrize("seed", ["absent", None])
def test_historical_native_isotropic_without_seed_is_supported(monkeypatch, seed):
    scene = scene_for_validation()
    scene.metadata = {"implementation": "torch_native", "device": "cpu"}
    if seed is None:
        scene.metadata["optimization_seed"] = None
    events = install_backend_standin(monkeypatch)
    GeometricValidator()._render_all_views(scene)
    expected = {
        "tier": "experimental",
        "device": "cpu",
        "implementation": "torch_native",
        "native_recipe": NATIVE_DEFAULT_RECIPE,
    }
    if seed is None:
        expected["optimization_seed"] = None
    assert events["config"] == [expected]


@pytest.mark.parametrize("location", ["scene", "splats", "provenance", "splat_optimizer", "provenance_optimizer"])
def test_recipe_only_saved_descriptors_select_native(monkeypatch, location):
    recipe = NATIVE_RECIPES[1]
    scene = scene_for_validation()
    scene.metadata = {"device": "cpu", "optimization_seed": 7}
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
    events = install_backend_standin(monkeypatch)
    GeometricValidator()._render_all_views(scene)
    assert events["config"][0]["native_recipe"] == recipe
    assert events["config"][0]["implementation"] == "torch_native"


@pytest.mark.parametrize(
    "invalid",
    [
        "recipe_conflict",
        "unknown_recipe",
        "implementation_conflict",
        "malformed_runtime",
        "seed_conflict",
        "device_conflict",
        "tier_conflict",
        "missing_device",
        "invalid_device",
        "missing_seed",
        "bool_seed",
        "float_seed",
        "negative_seed",
    ],
)
def test_invalid_saved_identity_refuses_before_backend_import(monkeypatch, invalid):
    scene = scene_for_validation(NATIVE_RECIPES[1])
    provenance = scene.metadata["reconstruction_provenance"]
    if invalid == "recipe_conflict":
        scene.splats.metadata["native_recipe"] = NATIVE_RECIPES[2]
    elif invalid == "unknown_recipe":
        scene.metadata["native_recipe"] = "unknown"
    elif invalid == "implementation_conflict":
        provenance["implementation"] = "legacy"
    elif invalid == "malformed_runtime":
        provenance["runtime"] = []
    elif invalid == "seed_conflict":
        provenance["optimization_seed"] = 8
    elif invalid == "device_conflict":
        provenance["runtime"]["device"] = "mps"
    elif invalid == "tier_conflict":
        provenance["tier"] = "experimental"
    elif invalid == "missing_device":
        del scene.metadata["device"]
        provenance["runtime"] = {}
    elif invalid == "invalid_device":
        scene.metadata["device"] = provenance["runtime"]["device"] = "unknown"
    elif invalid == "missing_seed":
        del scene.metadata["optimization_seed"]
        del provenance["optimization_seed"]
    else:
        value = {"bool_seed": True, "float_seed": 7.0, "negative_seed": -1}[invalid]
        scene.metadata["optimization_seed"] = provenance["optimization_seed"] = value
    forbid_backend_import(monkeypatch)
    with pytest.raises(ValueError):
        GeometricValidator()._render_all_views(scene)


def test_native_settings_can_come_from_provenance_without_ambient_defaults(monkeypatch):
    scene = scene_for_validation(NATIVE_RECIPES[1])
    for name in ("device", "optimization_seed", "tier", "implementation", "native_recipe"):
        del scene.metadata[name]
    scene.metadata["reconstruction_provenance"]["tier"] = "apex_research_ultra"
    events = install_backend_standin(monkeypatch)
    GeometricValidator()._render_all_views(scene)
    assert events["config"][0]["device"] == "cpu"
    assert events["config"][0]["optimization_seed"] == 7
    assert events["config"][0]["tier"] == "apex_research_ultra"


def test_pre_rendered_rmse_keeps_existing_backend_bypass(monkeypatch):
    scene = scene_for_validation(NATIVE_RECIPES[1])
    scene.metadata["native_recipe"] = "unresolved_descriptor"
    images = [np.zeros((camera.height, camera.width, 3), dtype=np.float32) for camera in scene.cameras]
    forbid_backend_import(monkeypatch)
    assert GeometricValidator().compute_rmse(scene, images, rendered_images=images) == 0.0
