"""Unit tests for the Phase II camera_native_linear ingest contract."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from transformation_portal.spatial_ai.ingest import phase2_camera_native_linear as phase2

pytestmark = [pytest.mark.unit]


class _FakeRaw:
    def __init__(self):
        self.rgb_xyz_matrix = np.eye(3, dtype=np.float32)
        self.camera_whitebalance = [2.0, 1.0, 1.0, 1.0]
        self._rgb16 = np.array(
            [
                [[0, 32768, 65535], [65535, 0, 32768]],
                [[16384, 16384, 16384], [65535, 65535, 65535]],
            ],
            dtype=np.uint16,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False

    def postprocess(self, **kwargs):
        assert kwargs["gamma"] == (1, 1)
        assert kwargs["no_auto_bright"] is True
        assert kwargs["no_auto_scale"] is True
        assert kwargs["output_bps"] == 16
        return self._rgb16


def _install_fake_rawpy(monkeypatch) -> None:
    demosaic = types.SimpleNamespace(name="AHD")
    fake_rawpy = types.SimpleNamespace(
        DemosaicAlgorithm=types.SimpleNamespace(AHD=demosaic),
        ColorSpace=types.SimpleNamespace(raw=object()),
        imread=lambda _path: _FakeRaw(),
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)


def test_apply_3x3_f32_hwc_validates_inputs():
    vec = np.zeros((2, 2, 3), dtype=np.float32)
    mat = np.eye(3, dtype=np.float32)

    out = phase2._apply_3x3_f32_hwc(vec, mat)  # pylint: disable=protected-access
    assert out.dtype == np.float32
    assert out.shape == vec.shape

    with pytest.raises(ValueError, match="vec3 must be float32"):
        phase2._apply_3x3_f32_hwc(vec.astype(np.float64), mat)  # pylint: disable=protected-access
    with pytest.raises(ValueError, match="HWC shape with 3 channels"):
        phase2._apply_3x3_f32_hwc(np.zeros((2, 2), dtype=np.float32), mat)  # pylint: disable=protected-access
    with pytest.raises(ValueError, match="HWC shape with 3 channels"):
        phase2._apply_3x3_f32_hwc(np.zeros((2, 2, 4), dtype=np.float32), mat)  # pylint: disable=protected-access
    with pytest.raises(ValueError, match="mat3x3 must be float32 shape"):
        phase2._apply_3x3_f32_hwc(vec, np.eye(4, dtype=np.float32))  # pylint: disable=protected-access


def test_ingest_phase2_rejects_invalid_wb_mode(monkeypatch):
    monkeypatch.setattr(phase2, "enforce_ftz_daz_disabled", lambda: None)
    _install_fake_rawpy(monkeypatch)

    with pytest.raises(ValueError, match="wb_mode must be one of"):
        phase2.ingest_phase2_xyz_d50_linear_fp32(Path("dummy.CR3"), wb_mode="invalid")


def test_ingest_phase2_raises_when_rawpy_missing(monkeypatch):
    monkeypatch.setattr(phase2, "enforce_ftz_daz_disabled", lambda: None)
    monkeypatch.setitem(sys.modules, "rawpy", None)

    with pytest.raises(RuntimeError, match="rawpy is required"):
        phase2.ingest_phase2_xyz_d50_linear_fp32(Path("dummy.CR3"))


def test_ingest_phase2_valid_decode_and_fingerprint(monkeypatch):
    monkeypatch.setattr(phase2, "enforce_ftz_daz_disabled", lambda: None)
    _install_fake_rawpy(monkeypatch)

    path = "inputs/sample.CR3"
    tensor1, fp1 = phase2.ingest_phase2_xyz_d50_linear_fp32(path, wb_mode="camera", demosaic="AHD")
    tensor2, fp2 = phase2.ingest_phase2_xyz_d50_linear_fp32(path, wb_mode="camera", demosaic="AHD")

    assert tensor1.dtype == np.float32
    assert tensor1.shape == (2, 2, 3)
    assert np.allclose(tensor1, tensor2)
    assert fp1 == fp2
    assert fp1["schema_version"] == "1.0.0"
    assert fp1["contract"] == "camera_native_linear"
    assert fp1["input_path"] == "sample.CR3"
    assert fp1["demosaic"] == "AHD"
    assert fp1["wb_mode"] == "camera"
    assert fp1["dtype"] == "float32"
    assert fp1["order"] == "C"


def test_ingest_phase2_rejects_unknown_demosaic(monkeypatch):
    monkeypatch.setattr(phase2, "enforce_ftz_daz_disabled", lambda: None)
    fake_rawpy = types.SimpleNamespace(
        DemosaicAlgorithm=types.SimpleNamespace(),
        ColorSpace=types.SimpleNamespace(raw=object()),
        imread=lambda _path: _FakeRaw(),
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)

    with pytest.raises(ValueError, match="Unknown demosaic algorithm"):
        phase2.ingest_phase2_xyz_d50_linear_fp32(Path("sample.CR3"), demosaic="NOT_REAL")


def test_native_libraw_matrix_inverts_xyz_to_camera_and_records_effective_transform(monkeypatch):
    raw = _FakeRaw()
    # Genuine DJI FC3170 ColorMatrix2: XYZ-to-camera, with inactive final row.
    native = np.array([[0.8157, -0.2211, -0.0505], [-0.401, 1.2314, 0.1881], [-0.0492, 0.1794, 0.5008]], np.float32)
    raw.rgb_xyz_matrix = np.vstack((native, np.zeros((1, 3), np.float32)))
    raw.num_colors = 3
    _install_fake_rawpy(monkeypatch)
    monkeypatch.setattr(sys.modules["rawpy"], "imread", lambda path: raw)
    monkeypatch.setattr(phase2, "enforce_ftz_daz_disabled", lambda: None)
    tensor, fingerprint = phase2.ingest_phase2_xyz_d50_linear_fp32("genuine-layout.DNG")
    reference_matrix = np.linalg.inv(native.astype(np.float64))
    reference = (raw._rgb16.astype(np.float64) / 65535.0) @ reference_matrix.T
    reference = reference @ phase2.BRADFORD_D65_TO_D50_F32.astype(np.float64).T
    np.testing.assert_allclose(tensor, reference, rtol=2e-6, atol=5e-7)
    np.testing.assert_allclose(fingerprint["rgb_xyz_matrix_f32"], reference_matrix, rtol=2e-6, atol=5e-7)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous


@pytest.mark.parametrize("colors", [None, 4, 2, "3", 3.0, True])
def test_native_matrix_rejects_missing_or_non_three_color_authority(colors):
    raw = types.SimpleNamespace(rgb_xyz_matrix=np.vstack((np.eye(3), np.zeros((1, 3)))), num_colors=colors)
    with pytest.raises(ValueError, match="three colors and an inactive fourth row"):
        phase2._camera_to_xyz_f32(raw)


@pytest.mark.parametrize("fourth", [[0.0, 0.0, 1.0], [0.0, float("nan"), 0.0], [0.0, 0.0, float("inf")]])
def test_native_matrix_rejects_active_or_nonfinite_fourth_row(fourth):
    raw = types.SimpleNamespace(rgb_xyz_matrix=np.vstack((np.eye(3), fourth)), num_colors=3)
    with pytest.raises(ValueError, match="inactive fourth row|finite coefficients"):
        phase2._camera_to_xyz_f32(raw)


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), np.ones((3, 3)), np.diag([1.0, 1.0, 1e-8])])
def test_native_matrix_rejects_singular_or_ill_conditioned_transform(matrix):
    raw = types.SimpleNamespace(rgb_xyz_matrix=np.vstack((matrix, np.zeros((1, 3)))), num_colors=3)
    with pytest.raises(ValueError, match="singular"):
        phase2._camera_to_xyz_f32(raw)


def test_native_dng_cmatrix_restores_daylight_scales_before_xyz_conversion():
    # Genuine DJI FC3170 native coefficients; original DNG ColorMatrix2 is the
    # independently read XYZ-to-camera reference, not the normalized cmatrix.
    color_matrix = np.array(
        [
            [1.84180868, -0.55250984, -0.28929886, 0],
            [-0.23723812, 1.59948575, -0.36224768, 0],
            [-0.02146835, -0.53612238, 1.55759072, 0],
        ],
        np.float32,
    )
    daylight = [2.00315523147583, 0.9478038549423218, 1.4750938415527344, 0]
    raw = types.SimpleNamespace(
        rgb_xyz_matrix=np.zeros((4, 3)), color_matrix=color_matrix, daylight_whitebalance=daylight, num_colors=3
    )
    reference = np.array([[0.8157, -0.2211, -0.0505], [-0.401, 1.2314, 0.1881], [-0.0492, 0.1794, 0.5008]])
    np.testing.assert_allclose(phase2._camera_to_xyz_f32(raw), np.linalg.inv(reference), rtol=2e-6, atol=5e-7)
    # A present, nonzero invalid XYZ matrix must not be bypassed by valid cmatrix.
    raw.rgb_xyz_matrix[0, 0] = 1
    with pytest.raises(ValueError, match="singular or ill-conditioned"):
        phase2._camera_to_xyz_f32(raw)


@pytest.mark.parametrize(
    "field, value",
    [
        ("color_matrix", np.zeros((3, 4))),
        ("color_matrix", np.ones((3, 4))),
        ("color_matrix", np.vstack((np.eye(3), np.zeros((1, 3))))),
        ("daylight_whitebalance", [1, 0, 1, 0]),
        ("daylight_whitebalance", [1, 1, -1, 0]),
        ("daylight_whitebalance", [1, 1, float("nan"), 0]),
        ("daylight_whitebalance", [1, 1, 1, -1]),
        ("daylight_whitebalance", [1, 1, 1]),
    ],
)
def test_native_dng_cmatrix_fallback_rejects_invalid_coefficients_or_scales(field, value):
    raw = types.SimpleNamespace(
        rgb_xyz_matrix=np.zeros((4, 3)),
        color_matrix=np.hstack((np.eye(3), np.zeros((3, 1)))),
        daylight_whitebalance=[1, 1, 1, 0],
        num_colors=3,
    )
    setattr(raw, field, value)
    with pytest.raises(ValueError, match="requires valid inactive"):
        phase2._camera_to_xyz_f32(raw)


def test_ingest_phase2_fails_closed_when_fpstate_enforcement_fails(monkeypatch):
    def _raise() -> None:
        raise RuntimeError("fpstate violation")

    monkeypatch.setattr(phase2, "enforce_ftz_daz_disabled", _raise)
    with pytest.raises(RuntimeError, match="fpstate violation"):
        phase2.ingest_phase2_xyz_d50_linear_fp32(Path("sample.CR3"))


def test_ingest_phase2_uses_dedicated_raw_runtime(monkeypatch):
    captured: dict[str, object] = {}
    enforced = {"called": False}
    fake_tensor = np.asfortranarray(np.full((2, 2, 3), 0.5, dtype=np.float64))
    fake_fingerprint = {
        "schema_version": "1.0.0",
        "contract": "camera_native_linear",
        "input_path": "sample.CR3",
    }

    def fake_run_raw_worker(*, python_executable, command_name, input_path, payload, start):
        captured["python_executable"] = python_executable
        captured["command_name"] = command_name
        captured["input_path"] = input_path
        captured["payload"] = payload
        captured["start"] = start
        return fake_tensor, {"fingerprint": fake_fingerprint}

    monkeypatch.setattr(phase2, "run_raw_worker", fake_run_raw_worker)
    monkeypatch.setattr(phase2, "enforce_ftz_daz_disabled", lambda: enforced.__setitem__("called", True))

    tensor, fingerprint = phase2.ingest_phase2_xyz_d50_linear_fp32(
        Path("sample.CR3"),
        wb_mode="auto",
        demosaic="AHD",
        raw_python_executable="./.venv-raw/bin/python",
    )

    assert enforced["called"] is True
    assert np.array_equal(tensor, fake_tensor.astype(np.float32))
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]
    assert fingerprint == fake_fingerprint
    assert captured["python_executable"] == "./.venv-raw/bin/python"
    assert captured["command_name"] == "phase2_decode"
    assert captured["input_path"] == Path("sample.CR3")
    assert captured["payload"] == {
        "wb_mode": "auto",
        "demosaic": "AHD",
    }


def test_ingest_phase2_subprocess_branch_fails_closed_before_dispatch(monkeypatch):
    def _raise() -> None:
        raise RuntimeError("fpstate violation")

    monkeypatch.setattr(phase2, "enforce_ftz_daz_disabled", _raise)
    monkeypatch.setattr(
        phase2,
        "run_raw_worker",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("subprocess dispatch should not run")),
    )

    with pytest.raises(RuntimeError, match="fpstate violation"):
        phase2.ingest_phase2_xyz_d50_linear_fp32(
            Path("sample.CR3"),
            raw_python_executable="./.venv-raw/bin/python",
        )


def test_ingest_phase2_rejects_invalid_subprocess_tensor_shape(monkeypatch):
    def fake_run_raw_worker(*, python_executable, command_name, input_path, payload, start):
        del python_executable, command_name, input_path, payload, start
        return np.zeros((4, 4), dtype=np.float32), {"fingerprint": {"schema_version": "1.0.0"}}

    monkeypatch.setattr(phase2, "run_raw_worker", fake_run_raw_worker)

    with pytest.raises(RuntimeError, match="invalid Phase II tensor shape"):
        phase2.ingest_phase2_xyz_d50_linear_fp32(
            Path("sample.CR3"),
            raw_python_executable="./.venv-raw/bin/python",
        )
