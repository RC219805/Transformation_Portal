"""Pure saved-recipe/export and adapter guards; no Torch or inference evidence."""

import builtins
import hashlib
import json
import struct
import sys
from types import ModuleType

import numpy as np
import pytest

import transformation_portal.spatial_ai.reconstruction as reconstruction
from transformation_portal.spatial_ai.reconstruction import protocol
from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams, GaussianSplat, Scene3D
from transformation_portal.spatial_ai.reconstruction.export_ply import PLYExportConfig, PLYExporter
from transformation_portal.spatial_ai.reconstruction.geometric_validator import GeometricValidator

pytestmark = pytest.mark.unit


class DescriptorTensor:
    """Only the metadata interface consumed by protocol guards, never a tensor kernel."""

    def __init__(self, shape, *, floating=True, device="cpu", finite=True):
        self.shape = shape
        self.ndim = len(shape)
        self.device = device
        self.dtype = "float32" if floating else "int32"
        self.floating = floating
        self.finite = finite

    def is_floating_point(self):
        return self.floating


class FiniteResult:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self.value


@pytest.fixture
def descriptor_torch(monkeypatch):
    """Isolate import availability and metadata guards without loading Torch."""
    module = ModuleType("torch")
    setattr(module, "Tensor", DescriptorTensor)
    setattr(module, "isfinite", lambda value: FiniteResult(value.finite))
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def _payload():
    return {
        "positions": DescriptorTensor((2, 3)),
        "colors": DescriptorTensor((2, 3)),
        "scales": DescriptorTensor((2, 3)),
        "rotations": DescriptorTensor((2, 4)),
        "opacities": DescriptorTensor((2, 1)),
        "intrinsics": DescriptorTensor((3, 3)),
        "extrinsics": DescriptorTensor((4, 4)),
        "image_size": (3, 4),
    }


def test_backend_payload_metadata_accepts_consistent_contract(descriptor_torch):
    assert protocol._require_torch() is descriptor_torch
    assert protocol.validate_backend_rasterizer_payload(**_payload()) is None


@pytest.mark.parametrize(
    "field,value,error,match",
    [
        ("positions", object(), TypeError, "must be a torch.Tensor"),
        ("positions", DescriptorTensor((6,)), ValueError, "must have rank 2"),
        ("positions", DescriptorTensor((2, 4)), ValueError, "invalid shape"),
        ("colors", DescriptorTensor((2, 3), floating=False), TypeError, "floating-point dtype"),
        ("colors", DescriptorTensor((3, 3)), ValueError, "batch size mismatch"),
        ("opacities", DescriptorTensor((2, 1), device="mps"), ValueError, "all tensors on one device"),
        ("image_size", (3,), TypeError, "2-tuple of ints"),
        ("image_size", (3.0, 4), TypeError, "2-tuple of ints"),
        ("image_size", (0, 4), ValueError, "positive values"),
        ("positions", DescriptorTensor((2, 3), finite=False), ValueError, "non-finite"),
    ],
)
def test_backend_payload_metadata_rejects_contract_drift(descriptor_torch, field, value, error, match):
    payload = _payload()
    payload[field] = value
    with pytest.raises(error, match=match):
        protocol.validate_backend_rasterizer_payload(**payload)


@pytest.mark.parametrize(
    "rendered,image_size,error,match",
    [
        (DescriptorTensor((3, 4, 3)), (3,), TypeError, "2-tuple of ints"),
        (DescriptorTensor((3, 4, 3)), (3.0, 4), TypeError, "2-tuple of ints"),
        (object(), (3, 4), TypeError, "must be a torch.Tensor"),
        (DescriptorTensor((3, 4, 4)), (3, 4), ValueError, "shape mismatch"),
        (DescriptorTensor((3, 4, 3), floating=False), (3, 4), TypeError, "floating-point dtype"),
        (DescriptorTensor((3, 4, 3), finite=False), (3, 4), ValueError, "non-finite"),
    ],
)
def test_rendered_metadata_rejects_contract_drift(descriptor_torch, rendered, image_size, error, match):
    with pytest.raises(error, match=match):
        protocol.validate_rasterizer_output(rendered=rendered, image_size=image_size)


def test_rendered_metadata_accepts_exact_rgb_frame(descriptor_torch):
    assert protocol.validate_rasterizer_output(rendered=DescriptorTensor((3, 4, 3)), image_size=(3, 4)) is None


def test_torch_unavailable_preserves_explicit_dependency_error(monkeypatch):
    original_import = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("invented optional-dependency absence")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    with pytest.raises(RuntimeError, match="Torch is required") as failure:
        protocol._require_torch()
    assert isinstance(failure.value.__cause__, ModuleNotFoundError)


def _scene():
    splats = GaussianSplat(
        positions=np.array([[0.25, -0.5, 2.0]], dtype=np.float32),
        colors=np.array([[1.0, 0.5, 0.0]], dtype=np.float32),
        scales=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([[0.75]], dtype=np.float32),
    )
    cameras = [CameraParams(np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32), 2, 2) for _ in range(2)]
    return Scene3D(splats, cameras, 0.5, 4, "max_iterations", {})


@pytest.fixture
def no_optional_backend(monkeypatch):
    original_import = builtins.__import__

    def no_native(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch.") or name.endswith("gaussian_backend"):
            pytest.fail("Pure metadata/export must not import an optional native backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_native)


@pytest.mark.parametrize("recipe", protocol.NATIVE_RECIPES)
def test_native_sidecar_preserves_exact_saved_identity_without_torch(tmp_path, no_optional_backend, recipe):
    scene = _scene()
    provenance = {"implementation": "torch_native", "native_recipe": recipe, "optimizer": {"native_recipe": recipe}}
    optimizer = {"native_recipe": recipe, "steps": 4}
    scene.metadata.update(implementation="torch_native", native_recipe=recipe, reconstruction_provenance=provenance)
    scene.splats.metadata.update(native_recipe=recipe, native_optimizer=optimizer)
    output = tmp_path / "scene.ply"
    spoof = {"native_recipe": "caller-note", "implementation": "legacy", "output_hash_sha256": "caller-note"}

    PLYExporter(PLYExportConfig(binary=False)).export(scene, output, additional_metadata=spoof)

    data = output.read_bytes()
    assert data.split(b"end_header\n", 1)[1].split() == [
        b"0.250000",
        b"-0.500000",
        b"2.000000",
        b"255",
        b"127",
        b"0",
        b"0.100000",
        b"0.200000",
        b"0.300000",
        b"1.000000",
        b"0.000000",
        b"0.000000",
        b"0.000000",
        b"0.750000",
    ]
    sidecar = json.loads(output.with_suffix(".provenance.json").read_bytes())
    assert sidecar["implementation"] == "torch_native"
    assert sidecar["native_recipe"] == recipe
    assert sidecar["reconstruction_provenance"] == provenance
    assert sidecar["native_optimizer"] == optimizer
    assert sidecar["color_encoding"] == "linear_rgb_quantized_uint8"
    assert sidecar["output_hash_sha256"] == hashlib.sha256(data).hexdigest()
    assert sidecar["request_metadata"] == spoof
    assert scene.iteration == 4 and scene.rmse == 0.5


def test_historical_native_sidecar_keeps_default_recipe_without_fabricating_optimizer(tmp_path, no_optional_backend):
    scene = _scene()
    scene.metadata["implementation"] = "torch_native"
    output = tmp_path / "historical.ply"
    PLYExporter().export(scene, output)
    sidecar = json.loads(output.with_suffix(".provenance.json").read_bytes())
    assert sidecar["native_recipe"] == protocol.NATIVE_DEFAULT_RECIPE
    assert "reconstruction_provenance" not in sidecar
    assert "native_optimizer" not in sidecar


@pytest.mark.parametrize("write_sidecar", [False, True])
@pytest.mark.parametrize("bad_location", ["provenance", "splat_optimizer", "provenance_optimizer", "unsupported_recipe"])
def test_invalid_saved_recipe_refuses_before_output_directory(tmp_path, no_optional_backend, write_sidecar, bad_location):
    scene = _scene()
    scene.metadata["implementation"] = "torch_native"
    if bad_location == "provenance":
        scene.metadata["reconstruction_provenance"] = []
    elif bad_location == "splat_optimizer":
        scene.splats.metadata["native_optimizer"] = []
    elif bad_location == "provenance_optimizer":
        scene.metadata["reconstruction_provenance"] = {"optimizer": []}
    else:
        scene.splats.metadata["native_recipe"] = False
    destination = tmp_path / "must-not-exist" / "scene.ply"
    with pytest.raises(ValueError, match="metadata containers|unsupported"):
        PLYExporter().export(scene, destination, write_sidecar=write_sidecar)
    assert not destination.parent.exists()


def test_binary_export_without_attributes_has_exact_little_endian_rgb_record(tmp_path, no_optional_backend):
    scene = _scene()
    output = tmp_path / "positions-rgb.ply"
    PLYExporter(PLYExportConfig(include_attributes=False)).export(scene, output, write_sidecar=False)
    header, data = output.read_bytes().split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0" in header
    assert b"property float scale_x" not in header
    assert data == struct.pack("<fffBBB", 0.25, -0.5, 2.0, 255, 127, 0)


def test_binary_export_accepts_opacity_vector_and_refuses_zero_width(tmp_path, no_optional_backend):
    scene = _scene()
    scene.splats.opacities = np.array([0.75], dtype=np.float32)
    output = tmp_path / "vector-opacity.ply"
    PLYExporter().export(scene, output, write_sidecar=False)
    data = output.read_bytes().split(b"end_header\n", 1)[1]
    assert len(data) == 47
    assert struct.unpack("<f", data[-4:]) == (0.75,)
    scene.splats.opacities = np.empty((1, 0), dtype=np.float32)
    with pytest.raises(ValueError, match="opacities second dimension"):
        PLYExporter().export(scene, tmp_path / "invalid-opacity.ply", write_sidecar=False)


@pytest.mark.parametrize(
    "name,module_name",
    [
        ("GaussianBackend", "gaussian_backend"),
        ("SceneBuilder", "scene_builder"),
        ("NVDiffRecBackend", "nvdiffrec_backend"),
        ("NVDiffRecConfig", "nvdiffrec_backend"),
        ("NVDiffRecLicenseError", "nvdiffrec_backend"),
        ("NVDiffRecEnvironmentError", "nvdiffrec_backend"),
    ],
)
def test_lazy_public_symbol_routing_uses_declared_module_only(monkeypatch, name, module_name):
    sentinel = object()
    qualified_name = f"transformation_portal.spatial_ai.reconstruction.{module_name}"
    module = ModuleType(qualified_name)
    setattr(module, name, sentinel)
    monkeypatch.setitem(sys.modules, qualified_name, module)
    monkeypatch.setattr(reconstruction, module_name, module, raising=False)
    assert reconstruction.__getattr__(name) is sentinel


def test_lazy_unknown_symbol_fails_closed():
    with pytest.raises(AttributeError, match="has no attribute 'UnknownNativeBackend'"):
        reconstruction.__getattr__("UnknownNativeBackend")


def test_rmse_reference_count_refuses_before_backend(no_optional_backend):
    with pytest.raises(ValueError, match="reference images.*number of cameras"):
        GeometricValidator().compute_rmse(_scene(), [])


def test_public_validation_uses_requested_reference_and_depth_paths(monkeypatch):
    """Real RMSE/depth aggregation over controlled images, not native rendering."""
    scene = _scene()
    validator = GeometricValidator()
    reference = np.zeros((2, 2, 3), dtype=np.float32)
    rendered = np.full((2, 2, 3), 0.25, dtype=np.float32)
    calls = []

    def controlled_render(actual_scene):
        calls.append(actual_scene)
        return [rendered, rendered]

    monkeypatch.setattr(validator, "_render_all_views", controlled_render)
    depth = np.full((2, 2), 2.0, dtype=np.float32)
    result = validator.validate_scene(scene, reference_images=[reference, reference], depth_maps=[depth, depth])
    assert calls == [scene]
    assert result["rmse"] == 0.25
    assert result["rmse_pass"] is False
    assert result["depth_consistency"] == 1.0
    assert result["quality_grade"] == "D"
