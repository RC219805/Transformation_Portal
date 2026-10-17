"""No-Torch orchestration policy tests, not numerical/native acceptance.

Execute the real backend's scalar admission/delegation/bridge code with narrow
optional-dependency doubles. Every renderer, initializer and optimizer raises.
Existing 32/80 genuine Torch tests keep their markers and numerical assertions.
"""

import builtins
import importlib.machinery
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import transformation_portal.spatial_ai.reconstruction as reconstruction
from transformation_portal.spatial_ai.reconstruction.contracts import LicenseRestrictionError
from transformation_portal.spatial_ai.reconstruction.protocol import NATIVE_DEFAULT_RECIPE, NATIVE_RECIPES

pytestmark = pytest.mark.unit


def forbidden_numerical_call(*_args, **_kwargs):
    pytest.fail("This core policy test must not invoke a renderer, initializer or optimizer")


@contextmanager
def isolated_backend_policy_runtime():
    """Load real control code, restoring exact prior modules/package attributes."""
    prefix = "transformation_portal.spatial_ai.reconstruction"
    names = ("torch", prefix + ".gaussian_backend", prefix + ".gaussian_rasterizer", prefix + ".torch_native")
    missing = object()
    saved_modules = {name: sys.modules.get(name, missing) for name in names}
    attrs = ("gaussian_backend", "gaussian_rasterizer", "torch_native")
    saved_attrs = {name: vars(reconstruction).get(name, missing) for name in attrs}
    torch = ModuleType("torch")
    for name in ("float16", "float32", "float64", "int16", "int32", "int64", "uint8", "bool"):
        setattr(torch, name, "test-dtype-" + name)
    availability = {"cuda": False, "mps": False}
    setattr(torch, "cuda", SimpleNamespace(is_available=lambda: availability["cuda"]))
    setattr(torch, "backends", SimpleNamespace(mps=SimpleNamespace(is_available=lambda: availability["mps"])))
    setattr(torch, "from_numpy", forbidden_numerical_call)
    setattr(torch, "tensor", forbidden_numerical_call)
    rasterizer = ModuleType(prefix + ".gaussian_rasterizer")
    for name in (
        "compute_rgb_loss",
        "render_gaussians",
        "render_gaussians_fast",
        "render_gaussians_native",
        "render_gaussians_anisotropic_native",
    ):
        setattr(rasterizer, name, forbidden_numerical_call)
    native = ModuleType(prefix + ".torch_native")
    constants = {
        "MAX_NATIVE_SHUFFLED_UPDATES": 30_000,
        "NATIVE_DEFAULT_RECIPE": NATIVE_DEFAULT_RECIPE,
        "NATIVE_RECIPES": NATIVE_RECIPES,
        "NATIVE_POSITION_PRIOR_WEIGHT": 0.01,
        "NATIVE_LOG_SCALE_PRIOR_WEIGHT": 0.001,
    }
    for name, value in constants.items():
        setattr(native, name, value)
    for name in (
        "initialize_metric_splats",
        "native_provenance",
        "optimize_shuffled_metric_splats",
        "validate_native_camera",
        "validate_native_input",
    ):
        setattr(native, name, forbidden_numerical_call)
    try:
        sys.modules["torch"] = torch
        sys.modules[prefix + ".gaussian_rasterizer"] = rasterizer
        sys.modules[prefix + ".torch_native"] = native
        module_name = prefix + ".gaussian_backend"
        path = Path(reconstruction.__file__).with_name("gaussian_backend.py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        backend = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = backend
        spec.loader.exec_module(backend)
        yield backend, torch, availability
    finally:
        for name, value in saved_modules.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        for name, value in saved_attrs.items():
            if value is missing:
                vars(reconstruction).pop(name, None)
            else:
                setattr(reconstruction, name, value)


@pytest.fixture
def backend_policy_runtime():
    with isolated_backend_policy_runtime() as runtime:
        yield runtime


def module_graph_snapshot():
    prefix = "transformation_portal.spatial_ai.reconstruction"
    names = ("torch", prefix + ".gaussian_backend", prefix + ".gaussian_rasterizer", prefix + ".torch_native")
    modules = {name: sys.modules[name] for name in names if name in sys.modules}
    attrs = {
        name: vars(reconstruction)[name]
        for name in ("gaussian_backend", "gaussian_rasterizer", "torch_native")
        if name in vars(reconstruction)
    }
    return modules, attrs


def assert_same_module_graph(before):
    after = module_graph_snapshot()
    for old, new in zip(before, after):
        assert old.keys() == new.keys()
        assert all(new[name] is value for name, value in old.items())


def test_dependency_double_graph_restores_after_success():
    before = module_graph_snapshot()
    with isolated_backend_policy_runtime() as (module, torch, _availability):
        assert sys.modules["torch"] is torch
        assert sys.modules["transformation_portal.spatial_ai.reconstruction.gaussian_backend"] is module
    assert_same_module_graph(before)


def test_dependency_double_graph_restores_after_backend_import_failure(monkeypatch):
    before = module_graph_snapshot()
    failure = RuntimeError("deliberate backend module import failure")

    class FailureLoader:
        def create_module(self, _spec):
            return None

        def exec_module(self, _module):
            raise failure

    def spec_for_failure(name, _path):
        return importlib.machinery.ModuleSpec(name, FailureLoader())

    monkeypatch.setattr(importlib.util, "spec_from_file_location", spec_for_failure)
    with pytest.raises(RuntimeError) as caught:
        with isolated_backend_policy_runtime():
            pytest.fail("Import failure must not enter the test body")
    assert caught.value is failure
    assert_same_module_graph(before)


@pytest.mark.parametrize("tier", ["commercial", "unknown"])
def test_license_refusal_precedes_optional_device_detection(backend_policy_runtime, monkeypatch, tier):
    backend, _torch, _availability = backend_policy_runtime
    monkeypatch.setattr(backend.GaussianBackend, "_detect_device", forbidden_numerical_call)
    with pytest.raises(LicenseRestrictionError, match="research tier"):
        backend.GaussianBackend(tier=tier)


@pytest.mark.parametrize("tier", ["apex_research", "apex_research_ultra", "experimental"])
def test_explicit_legacy_defaults_stay_lazy_and_checkpoint_unloaded(backend_policy_runtime, tier):
    module, _torch, _availability = backend_policy_runtime
    backend = module.GaussianBackend(tier=tier, device="cpu")
    assert backend.implementation == "legacy"
    assert backend.native_recipe == NATIVE_DEFAULT_RECIPE
    assert backend.optimization_seed is None
    assert backend.optimization_max_gaussians == 5000
    assert backend.device == "cpu" and backend.tier == tier
    assert backend._model is None and not backend._model_loaded


@pytest.mark.parametrize("implementation", ["unknown", None, True])
def test_unknown_implementation_is_not_silently_selected(backend_policy_runtime, implementation):
    module, _torch, _availability = backend_policy_runtime
    with pytest.raises(ValueError, match="Unsupported Gaussian implementation"):
        module.GaussianBackend(device="cpu", implementation=implementation)


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_native_cap_refuses_coercion_before_kernel_use(backend_policy_runtime, cap):
    module, _torch, _availability = backend_policy_runtime
    with pytest.raises(ValueError, match="positive integer"):
        module.GaussianBackend(device="cpu", implementation="torch_native", optimization_max_gaussians=cap)


@pytest.mark.parametrize("cap", [1, 5000])
def test_native_explicit_cap_and_isotropic_seed_none_are_preserved(backend_policy_runtime, cap):
    module, _torch, _availability = backend_policy_runtime
    backend = module.GaussianBackend(device="cpu", implementation="torch_native", optimization_max_gaussians=cap)
    assert backend.optimization_max_gaussians == cap
    assert backend.optimization_seed is None and backend.native_recipe == NATIVE_DEFAULT_RECIPE
    assert not backend._model_loaded


@pytest.mark.parametrize(
    "settings",
    [
        {"model_repo": "unrelated/model"},
        {"model_revision": "0123456789abcdef0123456789abcdef01234567"},
    ],
)
def test_native_rejects_upstream_checkpoint_claim(backend_policy_runtime, settings):
    module, _torch, _availability = backend_policy_runtime
    with pytest.raises(ValueError, match="cannot load or attest"):
        module.GaussianBackend(device="cpu", implementation="torch_native", **settings)


@pytest.mark.parametrize("recipe", ["unknown", None, True])
def test_unknown_recipe_is_not_an_isotropic_fallback(backend_policy_runtime, recipe):
    module, _torch, _availability = backend_policy_runtime
    with pytest.raises(ValueError, match="Unsupported native_recipe"):
        module.GaussianBackend(device="cpu", implementation="torch_native", native_recipe=recipe)


@pytest.mark.parametrize("recipe", NATIVE_RECIPES[1:])
def test_shuffled_recipe_cannot_implicitly_enable_native(backend_policy_runtime, recipe):
    module, _torch, _availability = backend_policy_runtime
    with pytest.raises(ValueError, match="requires implementation"):
        module.GaussianBackend(device="cpu", native_recipe=recipe, optimization_seed=7)


@pytest.mark.parametrize("seed", [None, True, -1, 1.5])
def test_shuffled_seed_refuses_ambient_and_coerced_values(backend_policy_runtime, seed):
    module, _torch, _availability = backend_policy_runtime
    with pytest.raises(ValueError, match="explicit nonnegative integer"):
        module.GaussianBackend(
            device="cpu", implementation="torch_native", native_recipe=NATIVE_RECIPES[1], optimization_seed=seed
        )


def test_zero_shuffled_seed_is_a_valid_explicit_identity(backend_policy_runtime):
    module, _torch, _availability = backend_policy_runtime
    backend = module.GaussianBackend(
        device="cpu", implementation="torch_native", native_recipe=NATIVE_RECIPES[2], optimization_seed=0
    )
    assert backend.optimization_seed == 0 and backend.native_recipe == NATIVE_RECIPES[2]


@pytest.mark.parametrize("cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_device_selection_preserves_explicit_priority(backend_policy_runtime, cuda, mps, expected):
    module, _torch, availability = backend_policy_runtime
    availability.update(cuda=cuda, mps=mps)
    assert module.GaussianBackend().device == expected


def test_device_detection_import_failure_is_cpu_only(backend_policy_runtime, monkeypatch, caplog):
    module, _torch, _availability = backend_policy_runtime
    original_import = builtins.__import__

    def block_torch(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("deliberate optional dependency absence")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_torch)
    assert module.GaussianBackend().device == "cpu"
    assert "PyTorch not available" in caplog.text


def test_placeholder_model_marks_only_lazy_mock_loading(backend_policy_runtime):
    module, _torch, _availability = backend_policy_runtime
    backend = module.GaussianBackend(device="cpu")
    backend._load_model()
    assert backend._model_loaded and backend._model is None
    sentinel = object()
    backend._model = sentinel
    backend._load_model()
    assert backend._model is sentinel


def test_pinned_checkpoint_fails_before_any_reconstruction_input_use(backend_policy_runtime):
    module, _torch, _availability = backend_policy_runtime

    class UnconsumedInput:
        @property
        def gamma(self):
            pytest.fail("Pinned loader refusal must precede input consumption or fitting")

    backend = module.GaussianBackend(device="cpu", model_revision="0123456789abcdef0123456789abcdef01234567")
    with pytest.raises(NotImplementedError, match="checkpoint integration"):
        backend.reconstruct(UnconsumedInput(), iterations=1)
    assert backend._model is None and not backend._model_loaded


def test_native_admission_failure_is_propagated_without_legacy_loading(backend_policy_runtime, monkeypatch):
    module, _torch, _availability = backend_policy_runtime
    backend = module.GaussianBackend(device="cpu", implementation="torch_native")
    monkeypatch.setattr(backend, "_load_model", forbidden_numerical_call)
    supplied_input, calls = object(), []

    def reject(value, iterations, depth, segmentation, pbr):
        calls.append((value, iterations, depth, segmentation, pbr))
        raise ValueError("native admission sentinel")

    monkeypatch.setattr(module, "validate_native_input", reject)
    with pytest.raises(ValueError, match="native admission sentinel"):
        backend.reconstruct(supplied_input, iterations=3, use_depth_prior=False, use_segmentation=False, use_pbr_textures=True)
    assert calls == [(supplied_input, 3, False, False, True)]
    assert not backend._model_loaded


def test_shuffled_update_bound_precedes_initialization(backend_policy_runtime, monkeypatch):
    module, _torch, _availability = backend_policy_runtime
    backend = module.GaussianBackend(
        device="cpu", implementation="torch_native", native_recipe=NATIVE_RECIPES[1], optimization_seed=7
    )
    calls = []
    monkeypatch.setattr(module, "validate_native_input", lambda *args: calls.append(args))
    value = object()
    with pytest.raises(ValueError, match="optimizer updates"):
        backend.reconstruct(value, iterations=30_001)
    assert calls == [(value, 30_001, True, True, False)]


@pytest.mark.parametrize("selected,recorded", [("legacy", "torch_native"), ("torch_native", "legacy")])
def test_saved_implementation_refuses_before_tensor_conversion(backend_policy_runtime, monkeypatch, selected, recorded):
    module, _torch, _availability = backend_policy_runtime
    backend = module.GaussianBackend(device="cpu", implementation=selected)
    monkeypatch.setattr(module, "_numpy_array_to_tensor", forbidden_numerical_call)
    scene = SimpleNamespace(metadata={"implementation": recorded}, splats=SimpleNamespace(metadata={}))
    with pytest.raises(ValueError, match="implementation does not match"):
        backend.render_view(scene, object())


def test_native_novel_camera_passes_context_and_propagates_failure(backend_policy_runtime, monkeypatch):
    module, _torch, _availability = backend_policy_runtime
    backend = module.GaussianBackend(device="cpu", implementation="torch_native")
    camera, calls = object(), []

    def reject(value, context):
        calls.append((value, context))
        raise ValueError("camera admission sentinel")

    monkeypatch.setattr(module, "validate_native_camera", reject)
    monkeypatch.setattr(module, "_numpy_array_to_tensor", forbidden_numerical_call)
    scene = SimpleNamespace(metadata={"implementation": "torch_native"}, splats=SimpleNamespace(metadata={}))
    with pytest.raises(ValueError, match="camera admission sentinel"):
        backend.render_view(scene, camera)
    assert calls == [(camera, "torch_native novel view")]


@pytest.mark.parametrize("selected", ["legacy", "torch_native"])
def test_saved_shuffled_recipe_never_uses_a_different_renderer(backend_policy_runtime, monkeypatch, selected):
    module, _torch, _availability = backend_policy_runtime
    backend = module.GaussianBackend(device="cpu", implementation=selected)
    monkeypatch.setattr(module, "_numpy_array_to_tensor", forbidden_numerical_call)
    scene = SimpleNamespace(metadata={}, splats=SimpleNamespace(metadata={"native_recipe": NATIVE_RECIPES[1]}))
    with pytest.raises(ValueError, match="recipe does not match"):
        backend.render_view(scene, object())


class BridgeValue:
    """Only the documented bridge methods, no tensor arithmetic or autograd."""

    def __init__(self, value, dtype, error=None):
        self.value, self.dtype, self.error = value, dtype, error
        self.calls = []

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def numpy(self):
        self.calls.append("numpy")
        if self.error is not None:
            raise self.error
        return self.value

    def tolist(self):
        self.calls.append("tolist")
        return self.value.tolist()

    def to(self, device):
        self.calls.append(("to", device))
        return self

    def requires_grad_(self, enabled):
        self.calls.append(("requires_grad", enabled))
        return self


def test_tensor_bridge_direct_path_preserves_array(backend_policy_runtime):
    module, torch, _availability = backend_policy_runtime
    value = np.array([[0.25, -0.5]], dtype=np.float32)
    bridge = BridgeValue(value, torch.float32)
    assert module._tensor_to_numpy_array(bridge) is value
    assert bridge.calls == ["detach", "cpu", "numpy"]


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_tensor_bridge_missing_numpy_uses_exact_dtype(backend_policy_runtime, dtype_name):
    module, torch, _availability = backend_policy_runtime
    value = np.array([[0.25, -0.5]], dtype=getattr(np, dtype_name))
    bridge = BridgeValue(value, getattr(torch, dtype_name), RuntimeError("Numpy is not available"))
    actual = module._tensor_to_numpy_array(bridge)
    np.testing.assert_array_equal(actual, value)
    assert actual.dtype == value.dtype
    assert bridge.calls == ["detach", "cpu", "numpy", "tolist"]


def test_tensor_bridge_unrelated_runtime_error_is_not_masked(backend_policy_runtime):
    module, torch, _availability = backend_policy_runtime
    error = RuntimeError("unexpected conversion failure")
    bridge = BridgeValue(np.zeros((1, 3), dtype=np.float32), torch.float32, error)
    with pytest.raises(RuntimeError) as caught:
        module._tensor_to_numpy_array(bridge)
    assert caught.value is error and "tolist" not in bridge.calls


@pytest.mark.parametrize("requires_grad", [False, True])
def test_numpy_bridge_preserves_device_and_gradient_intent(backend_policy_runtime, monkeypatch, requires_grad):
    module, torch, _availability = backend_policy_runtime
    value = np.array([[0.25]], dtype=np.float32)
    bridge = BridgeValue(value, torch.float32)
    calls = []

    def from_numpy(array):
        calls.append(array)
        return bridge

    monkeypatch.setattr(torch, "from_numpy", from_numpy)
    actual = module._numpy_array_to_tensor(value, "cpu", requires_grad=requires_grad)
    assert actual is bridge and len(calls) == 1 and calls[0] is value
    assert bridge.calls == [("to", "cpu")] + ([("requires_grad", True)] if requires_grad else [])


def test_numpy_bridge_missing_numpy_preserves_float64(backend_policy_runtime, monkeypatch):
    module, torch, _availability = backend_policy_runtime
    value = np.array([[0.25, -0.5]], dtype=np.float64)
    calls = []

    def missing_bridge(_array):
        raise RuntimeError("Numpy is not available")

    def tensor(values, dtype):
        calls.append((values, dtype))
        return BridgeValue(np.asarray(values), dtype)

    monkeypatch.setattr(torch, "from_numpy", missing_bridge)
    monkeypatch.setattr(torch, "tensor", tensor)
    actual = module._numpy_array_to_tensor(value, "mps", requires_grad=True)
    assert calls == [(value.tolist(), torch.float64)]
    assert actual.dtype == torch.float64
    assert actual.calls == [("to", "mps"), ("requires_grad", True)]


def test_numpy_bridge_unrelated_error_does_not_fallback(backend_policy_runtime, monkeypatch):
    module, torch, _availability = backend_policy_runtime
    error = RuntimeError("unsupported bridge input")

    def failure(_array):
        raise error

    monkeypatch.setattr(torch, "from_numpy", failure)
    with pytest.raises(RuntimeError) as caught:
        module._numpy_array_to_tensor(np.zeros((1, 3), dtype=np.float32), "cpu")
    assert caught.value is error
