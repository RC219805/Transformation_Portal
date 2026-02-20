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


def test_ingest_phase2_fails_closed_when_fpstate_enforcement_fails(monkeypatch):
    def _raise() -> None:
        raise RuntimeError("fpstate violation")

    monkeypatch.setattr(phase2, "enforce_ftz_daz_disabled", _raise)
    with pytest.raises(RuntimeError, match="fpstate violation"):
        phase2.ingest_phase2_xyz_d50_linear_fp32(Path("sample.CR3"))
