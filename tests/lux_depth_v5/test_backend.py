"""Real precision contexts, sky semantics, and bounded versioned worker transport."""

from __future__ import annotations

import hashlib
import io
import json
import random
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from transformation_portal.lux_depth_v4 import backend as previous_backend
from transformation_portal.lux_depth_v5 import backend, worker

pytestmark = pytest.mark.unit


def _plan(precision="fp32"):
    return SimpleNamespace(
        plan_fingerprint_sha256="b" * 64,
        to_payload=lambda: {"configuration": {"depth": {"precision": precision}}},
    )


def _array_bytes(values=None, *, shape=(14, 28), dtype="<f4", header_only=False):
    data = io.BytesIO()
    if header_only:
        np.lib.format.write_array_header_1_0(data, {"shape": shape, "fortran_order": False, "descr": dtype})
    else:
        np.save(data, np.ones(shape, dtype=dtype) if values is None else values, allow_pickle=False)
    return data.getvalue()


def _session(tmp_path, members, *, updates=None):
    session = backend.DA3Session.__new__(backend.DA3Session)
    session.root = tmp_path
    session._counter = 0
    session.runtime = SimpleNamespace(runtime_identity_sha256="a" * 64)
    session.plan = _plan()

    def rpc(request):
        path = Path(request["output"])
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in members:
                archive.writestr(name, payload)
        raw = path.read_bytes()
        return {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "runtime_identity_sha256": "a" * 64,
            "plan_fingerprint_sha256": "b" * 64,
            "precision": "fp32",
            "inference_recipe": backend.INFERENCE_RECIPE,
            "sky_mask_policy": backend.SKY_MASK_POLICY,
            "native_semantics": "da3_metric_uncalibrated",
            "sky_available": any(name == "sky_mask.npy" for name, _ in members),
            "confidence_available": False,
            **(updates or {}),
        }

    session._rpc = rpc
    return session


@pytest.mark.parametrize("sky_available", [True, False])
def test_transport_preserves_bool_sky_or_explicit_unavailability(tmp_path, sky_available):
    sky = np.zeros((14, 28), dtype=bool)
    sky[:, 7:] = True
    members = [("native_depth.npy", _array_bytes())]
    if sky_available:
        members.append(("sky_mask.npy", _array_bytes(sky)))
    arrays, receipt = _session(tmp_path, members).compute(np.zeros((14, 28, 3), np.uint8))
    assert receipt["sky_available"] is sky_available
    assert ("sky_mask" in arrays) is sky_available
    if sky_available:
        np.testing.assert_array_equal(arrays["sky_mask"], sky)
        assert arrays["sky_mask"].dtype == np.bool_


@pytest.mark.parametrize(
    "member, error",
    [
        (("sky_mask.npy", _array_bytes(dtype="float32", header_only=True)), "header differs"),
        (("sky_mask.npy", _array_bytes(shape=(1_000_000, 1_000_000), dtype="|b1", header_only=True)), "header differs"),
        (("sky_mask.npy", _array_bytes(dtype="|O", header_only=True)), "header differs"),
        (("sky_mask.npy", b"x" * 100_000), "expanded array budget"),
        (("confidence.npy", _array_bytes()), "expanded array budget"),
    ],
)
def test_transport_rejects_unbounded_or_untyped_arrays_before_loading(tmp_path, monkeypatch, member, error):
    session = _session(tmp_path, [("native_depth.npy", _array_bytes()), member])
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("Header must be validated before np.load"))
    with pytest.raises(RuntimeError, match=error):
        session.compute(np.zeros((14, 28, 3), np.uint8))


@pytest.mark.parametrize(
    "updates",
    [
        {"precision": "fp16"},
        {"inference_recipe": "unversioned"},
        {"sky_mask_policy": "da3_sky_ge_0_5"},
        {"native_semantics": "metric_distance_m"},
        {"confidence_available": True},
        {"sky_available": True},
        {"sky_available": 0},
    ],
)
def test_transport_rejects_receipt_contract_drift(tmp_path, updates):
    session = _session(tmp_path, [("native_depth.npy", _array_bytes())], updates=updates)
    with pytest.raises(RuntimeError, match="prepared depth contract"):
        session.compute(np.zeros((14, 28, 3), np.uint8))


@pytest.mark.parametrize("key", ["runtime_identity_sha256", "plan_fingerprint_sha256", "sha256"])
def test_transport_still_binds_exact_execution_identity(tmp_path, key):
    session = _session(tmp_path, [("native_depth.npy", _array_bytes())], updates={key: "f" * 64})
    with pytest.raises(RuntimeError, match="identity mismatch"):
        session.compute(np.zeros((14, 28, 3), np.uint8))


def test_transport_reuses_supervision_and_selects_versioned_worker():
    assert backend.DA3Session._rpc is previous_backend.DA3Session._rpc
    assert backend.DA3Session.checkpoint is previous_backend.DA3Session.checkpoint
    assert backend.DA3Session.close is previous_backend.DA3Session.close
    assert backend.DA3Session._WORKER_MODULE == "transformation_portal.lux_depth_v5.worker"
    assert previous_backend.DA3Session._WORKER_MODULE == "transformation_portal.lux_depth_v4.worker"
    assert set(previous_backend.DA3Session._ARRAY_DTYPES) == {"native_depth", "confidence"}


def _model(*, sky=None, bad_depth=False):
    torch = pytest.importorskip("torch")

    class DepthAnything3Net(torch.nn.Module):
        __module__ = "depth_anything_3.model.da3"

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((3, 3), dtype=torch.float32))
            self.observed = None

        def forward(
            self,
            x,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=None,
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy="saddle_balanced",
        ):
            assert (extrinsics, intrinsics, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy) == (
                None,
                None,
                [],
                False,
                False,
                "saddle_balanced",
            )
            computed = torch.mm(x[0, 0, :, :1, :3].reshape(3, 3), self.weight)
            self.observed = (computed.dtype, torch.is_autocast_enabled("cpu"), torch.is_inference_mode_enabled())
            depth = torch.ones((1, 1, *x.shape[-2:]), dtype=torch.float32)
            if bad_depth:
                depth[..., 0, 0] = float("nan")
            result = {"depth": depth}
            if sky is not None:
                result["sky"] = torch.as_tensor(sky)[None, None]
            return result

    class DepthAnything3:
        __module__ = "depth_anything_3.api"

        def __init__(self):
            self.model = DepthAnything3Net()

        def parameters(self):
            return self.model.parameters()

        def inference(self, *_args, **_kwargs):
            pytest.fail("Implicit upstream inference precision must never execute")

        def forward(self, *_args, **_kwargs):
            pytest.fail("Implicit upstream forward precision must never execute")

        def _preprocess_inputs(
            self, image, extrinsics=None, intrinsics=None, process_res=504, process_res_method="upper_bound_resize"
        ):
            proxy = image[0]
            assert process_res == max(proxy.size)
            assert process_res_method == "upper_bound_resize"
            return torch.ones((1, 3, proxy.height, proxy.width), dtype=torch.float32), extrinsics, intrinsics

        def _prepare_model_inputs(self, imgs_cpu, extrinsics, intrinsics):
            return imgs_cpu[None], extrinsics, intrinsics

        def _convert_to_prediction(self, raw_output):
            return SimpleNamespace(depth=raw_output["depth"][0].numpy())

    return DepthAnything3()


@pytest.mark.parametrize("precision, expected_dtype, enabled", [("fp32", "float32", False), ("fp16", "float16", True)])
def test_precision_controls_real_operation_inside_network(precision, expected_dtype, enabled):
    torch = pytest.importorskip("torch")
    model = _model()
    # An ambient context must not silently change explicit fp32 execution.
    with torch.autocast("cpu", dtype=torch.float16):
        depth, sky = worker._predict_native(model, Image.new("RGB", (28, 14)), precision)
    assert model.model.observed == (getattr(torch, expected_dtype), enabled, True)
    assert depth.dtype == np.float32
    assert sky is None


def test_sky_mask_covers_every_upstream_substitution_threshold():
    sky = np.zeros((14, 28), dtype=np.float32)
    sky[0, :4] = [0.299, 0.3, 0.49, 0.5]
    _, mask = worker._predict_native(_model(sky=sky), Image.new("RGB", (28, 14)), "fp32")
    assert mask.dtype == np.bool_
    assert mask[0, :4].tolist() == [False, True, True, True]


@pytest.mark.parametrize("sky", [np.ones((13, 28), np.float32), np.full((14, 28), np.nan, np.float32)])
def test_sky_geometry_and_finiteness_are_not_coerced(sky):
    with pytest.raises(RuntimeError, match="sky output"):
        worker._predict_native(_model(sky=sky), Image.new("RGB", (28, 14)), "fp32")


def test_nonfinite_native_prediction_is_rejected():
    with pytest.raises(RuntimeError, match="native output"):
        worker._predict_native(_model(bad_depth=True), Image.new("RGB", (28, 14)), "fp32")


def test_low_precision_weights_cannot_be_claimed_as_fp32():
    model = _model()
    model.model.half()
    with pytest.raises(RuntimeError, match="float32 model weights"):
        worker._predict_native(model, Image.new("RGB", (28, 14)), "fp32")


def test_pinned_api_signature_drift_fails_before_network():
    model = _model()
    model._prepare_model_inputs = lambda changed: changed
    with pytest.raises(RuntimeError, match="interfaces changed"):
        worker._predict_native(model, Image.new("RGB", (28, 14)), "fp32")


def test_source_revision_change_requires_recipe_readmission(monkeypatch):
    def prepare(instance, _plan):
        instance.evidence = SimpleNamespace(to_mapping=lambda: {"evidence": {"source_revision": "0" * 40}})

    monkeypatch.setattr(worker.GovernedNativeDepthWorker, "__init__", prepare)
    with pytest.raises(RuntimeError, match="source revision"):
        worker.NativeDepthWorker(_plan())


@pytest.mark.parametrize("sky_available", [True, False])
def test_worker_persists_native_sky_and_precision_receipt(tmp_path, monkeypatch, sky_available):
    native = worker.NativeDepthWorker.__new__(worker.NativeDepthWorker)
    native.precision = "fp32"
    native.plan = _plan()
    native.evidence = SimpleNamespace(runtime_identity_sha256="a" * 64)
    native.verify = Mock()
    sky = np.zeros((14, 28), dtype=np.float32)
    sky[:, 7:] = 0.4
    native.engine = SimpleNamespace(_load_model=lambda: None, model=_model(sky=sky if sky_available else None))
    from transformation_portal.depth.backends import da3_worker

    monkeypatch.setattr(da3_worker, "_seed_isolated_inference", lambda _image: None)
    input_path = tmp_path / "proxy.png"
    Image.new("RGB", (28, 14)).save(input_path)
    output_path = tmp_path / "depth.npz"
    receipt = native.infer(input_path, output_path)
    with np.load(output_path, allow_pickle=False) as archive:
        assert set(archive.files) == ({"native_depth", "sky_mask"} if sky_available else {"native_depth"})
        if sky_available:
            stored_sky = np.asarray(archive["sky_mask"])
            assert stored_sky.dtype == np.bool_
            assert stored_sky[:, 7:].all()
    assert receipt["precision"] == "fp32"
    assert receipt["sky_available"] is sky_available
    assert receipt["inference_recipe"] == backend.INFERENCE_RECIPE
    assert native.verify.call_count == 2


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_worker_sampling_is_content_seeded_after_lazy_model_initialization(tmp_path, precision):
    """Actual worker seeding keeps sampled depth independent of model warmth."""
    torch = pytest.importorskip("torch")

    class LazySamplingEngine:
        model = None
        initialization_changed_rng = False

        def _load_model(self):
            if self.model is not None:
                return
            before = torch.random.get_rng_state().clone()
            torch.rand(8192)  # Model construction consumes RNG only on the cold path.
            self.initialization_changed_rng = not torch.equal(before, torch.random.get_rng_state())
            model = _model()
            original_forward = model.model.forward

            def sampling_forward(
                x,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=None,
                infer_gs=False,
                use_ray_pose=False,
                ref_view_strategy="saddle_balanced",
            ):
                output = original_forward(
                    x, extrinsics, intrinsics, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy
                )
                # Exercise the upstream sky recipe's stochastic subsampling,
                # retaining sample values too so seed changes cannot hide behind
                # equal rounded quantiles. No inference or seeding seam is mocked.
                population = torch.linspace(0.01, 199.0, 250_000)
                indices = torch.randint(0, population.numel(), (100_000,))
                sampled = population[indices]
                height, width = x.shape[-2:]
                output["depth"] = sampled[: height * width].reshape(1, 1, height, width).clone()
                output["depth"][..., :2, :] = torch.quantile(sampled, 0.99)
                return output

            model.model.forward = sampling_forward
            self.model = model

    native = worker.NativeDepthWorker.__new__(worker.NativeDepthWorker)
    native.precision = precision
    native.plan = _plan(precision)
    native.evidence = SimpleNamespace(runtime_identity_sha256="a" * 64)
    native.verify = lambda: None
    native.engine = LazySamplingEngine()
    original = Image.new("RGB", (28, 14), (40, 100, 200))
    original.save(tmp_path / "original.png")
    original.save(tmp_path / "same-pixels.png", compress_level=0)
    changed = original.copy()
    changed.putpixel((0, 0), (41, 100, 200))
    changed.save(tmp_path / "changed.png")
    assert (tmp_path / "original.png").read_bytes() != (tmp_path / "same-pixels.png").read_bytes()

    def infer(name, source):
        destination = tmp_path / f"{name}.npz"
        native.infer(tmp_path / source, destination)
        with np.load(destination, allow_pickle=False) as archive:
            return archive["native_depth"].copy()

    # infer() runs in an isolated process in production; preserve the unit test
    # process's Python, NumPy, and CPU Torch RNG state around this real seam.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            cold = infer("cold", "original.png")
            assert native.engine.initialization_changed_rng
            torch.rand(4096)  # Ambient work must not influence the next request.
            warm = infer("warm", "same-pixels.png")
            different = infer("different", "changed.png")
            repeated = infer("repeated", "original.png")
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    np.testing.assert_array_equal(cold, warm)
    np.testing.assert_array_equal(cold, repeated)
    assert not np.array_equal(cold, different)


@pytest.mark.parametrize("line", [b'{"command":"verify"}\n', b'{"command":"prepare","command":"infer"}\n'])
def test_worker_protocol_rejects_invalid_lifecycle_and_duplicate_keys(monkeypatch, capsys, line):
    monkeypatch.setattr(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(line)))
    assert worker.main([]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False


def test_native_shape_mismatch_cannot_be_resampled_silently():
    model = _model()
    model._convert_to_prediction = lambda raw_output: SimpleNamespace(depth=np.ones((1, 13, 28), np.float32))
    with pytest.raises(RuntimeError, match="native output"):
        worker._predict_native(model, Image.new("RGB", (28, 14)), "fp32")


def test_worker_rejects_large_proxy_before_decoding(tmp_path, monkeypatch):
    native = worker.NativeDepthWorker.__new__(worker.NativeDepthWorker)
    native.verify = lambda: None

    class OversizedImage:
        mode = "RGB"
        size = (140_000, 140_000)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def load(self):
            pytest.fail("Unbounded proxy must be rejected before decode")

    monkeypatch.setattr(Image, "open", lambda _path: OversizedImage())
    with pytest.raises(ValueError, match="bounded RGB model proxy"):
        native.infer(tmp_path / "input.png", tmp_path / "output.npz")
