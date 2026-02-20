"""Unit tests for spatial ingest contract dispatcher."""

from __future__ import annotations

import types

import numpy as np
import pytest

from transformation_portal.spatial_ai.ingest.contracts import IngestOptions, decode_contract

pytestmark = [pytest.mark.unit]


def test_ingest_options_defaults():
    opts = IngestOptions(contract="camera_native_linear")
    assert opts.tensor_role == "xyz_d50_linear_fp32"
    assert opts.wb_mode == "camera"
    assert opts.demosaic == "AHD"


def test_decode_contract_camera_native_linear_forwards_params(monkeypatch):
    captured = {}
    expected = np.ones((2, 2, 3), dtype=np.float32)

    def _fake_ingest(path, *, wb_mode, demosaic):  # noqa: ANN001
        captured["path"] = path
        captured["wb_mode"] = wb_mode
        captured["demosaic"] = demosaic
        return expected, {"contract": "camera_native_linear"}

    monkeypatch.setattr(
        "transformation_portal.spatial_ai.ingest.phase2_camera_native_linear.ingest_phase2_xyz_d50_linear_fp32",
        _fake_ingest,
    )

    opts = IngestOptions(contract="camera_native_linear", wb_mode="auto", demosaic="AHD")
    out = decode_contract("example.CR3", opts)

    assert out.dtype == np.float32
    assert out.shape == (2, 2, 3)
    assert np.allclose(out, expected)
    assert captured["wb_mode"] == "auto"
    assert captured["demosaic"] == "AHD"


def test_decode_contract_legacy_linear_srgb_returns_linear_rgb(monkeypatch):
    expected = np.full((4, 4, 3), 0.25, dtype=np.float32)

    class _FakeDecoder:
        def __init__(self, gamma, strict_ingest):  # noqa: ANN001
            assert gamma == 1.0
            assert strict_ingest is True

        def decode(self, input_path):  # noqa: ANN001
            assert input_path == "legacy_input.tiff"
            return types.SimpleNamespace(linear_rgb=expected)

    monkeypatch.setattr("transformation_portal.spatial_ai.ingest.linear_decoder.LinearDecoder", _FakeDecoder)
    opts = IngestOptions(contract="legacy_linear_srgb")
    out = decode_contract("legacy_input.tiff", opts)

    assert out.dtype == np.float32
    assert out.shape == (4, 4, 3)
    assert np.allclose(out, expected)


def test_decode_contract_raises_on_unknown_contract():
    opts = IngestOptions(contract="unknown_contract")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown ingest contract"):
        decode_contract("ignored", opts)


def test_decode_contract_propagates_phase2_fail_closed(monkeypatch):
    def _raise(*_args, **_kwargs):  # noqa: ANN002,ANN003
        raise RuntimeError("FTZ/DAZ enabled")

    monkeypatch.setattr(
        "transformation_portal.spatial_ai.ingest.phase2_camera_native_linear.ingest_phase2_xyz_d50_linear_fp32",
        _raise,
    )
    opts = IngestOptions(contract="camera_native_linear")
    with pytest.raises(RuntimeError, match="FTZ/DAZ enabled"):
        decode_contract("example.CR3", opts)


def test_decode_contract_camera_native_linear_enforces_tensor_role(monkeypatch):
    monkeypatch.setattr(
        "transformation_portal.spatial_ai.ingest.phase2_camera_native_linear.ingest_phase2_xyz_d50_linear_fp32",
        lambda *args, **kwargs: (np.ones((1, 1, 3), dtype=np.float32), {}),  # noqa: ANN002,ANN003
    )
    opts = IngestOptions(contract="camera_native_linear", tensor_role="linear_srgb")
    with pytest.raises(ValueError, match="requires tensor_role='xyz_d50_linear_fp32'"):
        decode_contract("example.CR3", opts)
