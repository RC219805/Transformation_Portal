"""Phase C1 tests: canonical RAW ingest adapter wiring for lux_depth_v3."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from transformation_portal.lux_depth_v3.inference import DA3InferenceEngine
from transformation_portal.lux_depth_v3.ingest_adapter import RawIngestError, decode_for_lux_depth, probe_raw_dimensions
from transformation_portal.lux_depth_v3.preprocessing import preprocess_image
from transformation_portal.lux_depth_v3.provenance import capture_provenance

pytestmark = pytest.mark.unit


def _raw_cfg(mode: str = "auto") -> SimpleNamespace:
    return SimpleNamespace(raw_ingest_mode=mode, raw_wb_mode="camera", raw_demosaic="AHD")


def test_probe_raw_dimensions_uses_validated_external_worker_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "scene_probe.dng"
    raw_path.write_bytes(b"raw")
    config = _raw_cfg()
    config.raw_python_executable = "/runtime/raw-python"
    captured: dict[str, object] = {}

    def fake_run_raw_worker(**kwargs):
        captured.update(kwargs)
        return np.empty((0,), dtype=np.uint8), {"input_size": [3024, 4032]}

    monkeypatch.delenv("TP_ALLOW_RAW_PREVIEW", raising=False)
    monkeypatch.setattr("transformation_portal.lux_depth_v3.ingest_adapter.run_raw_worker", fake_run_raw_worker)

    assert probe_raw_dimensions(raw_path, config) == (4032, 3024)
    assert captured == {
        "python_executable": "/runtime/raw-python",
        "command_name": "probe",
        "input_path": raw_path,
        "payload": {},
        "start": raw_path,
    }


@pytest.mark.parametrize(
    "input_size",
    [None, [3024], [0, 4032], [True, 4032], [3024.0, 4032]],
)
def test_probe_raw_dimensions_rejects_invalid_external_worker_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    input_size: object,
) -> None:
    raw_path = tmp_path / "scene_invalid_probe.dng"
    raw_path.write_bytes(b"raw")
    config = _raw_cfg()
    config.raw_python_executable = "/runtime/raw-python"

    monkeypatch.delenv("TP_ALLOW_RAW_PREVIEW", raising=False)
    monkeypatch.setattr(
        "transformation_portal.lux_depth_v3.ingest_adapter.run_raw_worker",
        lambda **_kwargs: (np.empty((0,), dtype=np.uint8), {"input_size": input_size}),
    )

    with pytest.raises(RawIngestError, match="RAW dimension probe returned invalid input_size"):
        probe_raw_dimensions(raw_path, config)


def test_preprocess_image_routes_raw_to_canonical_ingest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw_path = tmp_path / "scene_01.dng"
    raw_path.write_bytes(b"phase_c1_fake_raw_payload")

    captured: dict[str, object] = {}

    def fake_decode_contract(input_path, opts):
        captured["path"] = Path(input_path)
        captured["opts"] = opts
        return np.full((32, 32, 3), 0.5, dtype=np.float32)

    monkeypatch.setattr("transformation_portal.spatial_ai.ingest.contracts.decode_contract", fake_decode_contract)

    def fail_if_pil_opened(*_args, **_kwargs):
        raise AssertionError("PIL preview path should not be used for RAW decode")

    monkeypatch.setattr("transformation_portal.lux_depth_v3.preprocessing.Image.open", fail_if_pil_opened)

    result, original_shape = preprocess_image(raw_path)

    assert original_shape == (32, 32)
    assert result.dtype == np.float32
    assert captured["path"] == raw_path
    opts = captured["opts"]
    assert getattr(opts, "no_auto_bright") is True
    assert getattr(opts, "no_auto_scale") is True
    assert getattr(opts, "gamma_mode") == "linear"
    assert getattr(opts, "wb_mode") == "camera"
    assert getattr(opts, "demosaic") == "AHD"


def test_infer_from_path_routes_raw_to_canonical_ingest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw_path = tmp_path / "scene_02.dng"
    raw_path.write_bytes(b"phase_c1_fake_raw_payload")

    monkeypatch.setattr(
        "transformation_portal.lux_depth_v3.ingest_adapter.decode_for_lux_depth",
        lambda _path, _cfg: np.full((16, 16, 3), 0.5, dtype=np.float32),
    )

    def fail_if_pil_opened(*_args, **_kwargs):
        raise AssertionError("PIL path should not be used for RAW in infer_from_path")

    monkeypatch.setattr("transformation_portal.lux_depth_v3.inference.Image.open", fail_if_pil_opened)

    engine = DA3InferenceEngine.__new__(DA3InferenceEngine)
    engine.config = _raw_cfg("auto")

    captured: dict[str, np.ndarray] = {}
    sentinel = object()

    def fake_infer(image_np: np.ndarray):
        captured["image"] = image_np
        return sentinel

    engine.infer = fake_infer  # type: ignore[assignment]

    result = DA3InferenceEngine.infer_from_path(engine, raw_path)

    assert result is sentinel
    assert captured["image"].dtype == np.uint8
    assert captured["image"].shape == (16, 16, 3)
    assert int(captured["image"].mean()) == 127


def test_force_preview_mode_requires_explicit_env_escape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw_path = tmp_path / "scene_03.dng"
    raw_path.write_bytes(b"phase_c1_fake_raw_payload")

    monkeypatch.delenv("TP_ALLOW_RAW_PREVIEW", raising=False)

    with pytest.raises(RawIngestError, match="TP_ALLOW_RAW_PREVIEW=1"):
        _ = decode_for_lux_depth(raw_path, _raw_cfg("force_preview"))


def test_force_preview_wraps_preview_decode_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw_path = tmp_path / "scene_04.dng"
    raw_path.write_bytes(b"phase_c1_fake_raw_payload")
    monkeypatch.setenv("TP_ALLOW_RAW_PREVIEW", "1")

    def fail_preview_decode(*_args, **_kwargs):
        raise OSError("cannot identify image file")

    monkeypatch.setattr("transformation_portal.lux_depth_v3.ingest_adapter.Image.open", fail_preview_decode)

    with pytest.raises(RawIngestError, match="RAW preview decode failed") as exc_info:
        _ = decode_for_lux_depth(raw_path, _raw_cfg("force_preview"))

    assert "cannot identify image file" in str(exc_info.value)


def test_auto_mode_preview_fallback_wraps_preview_decode_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "scene_05.dng"
    raw_path.write_bytes(b"phase_c1_fake_raw_payload")
    monkeypatch.setenv("TP_ALLOW_RAW_PREVIEW", "1")

    def fail_decode_contract(*_args, **_kwargs):
        raise RuntimeError("canonical decode failed")

    def fail_preview_decode(*_args, **_kwargs):
        raise OSError("cannot identify image file")

    monkeypatch.setattr("transformation_portal.spatial_ai.ingest.contracts.decode_contract", fail_decode_contract)
    monkeypatch.setattr("transformation_portal.lux_depth_v3.ingest_adapter.Image.open", fail_preview_decode)

    with pytest.raises(RawIngestError, match="RAW preview decode failed") as exc_info:
        _ = decode_for_lux_depth(raw_path, _raw_cfg("auto"))

    assert "cannot identify image file" in str(exc_info.value)


def test_auto_mode_retries_non_positive_camera_wb_with_rawpy_auto_wb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "scene_06.dng"
    raw_path.write_bytes(b"phase_c1_fake_raw_payload")

    def fail_decode_contract(*_args, **_kwargs):
        raise RuntimeError(
            "RAW metadata: camera_whitebalance has zero or negative gain(s) at channel(s) [3]: [1.9, 1.0, 1.6, 0.0]."
        )

    captured: dict[str, object] = {}

    def fake_load_raw_as_rgb(path, **kwargs):
        captured["path"] = Path(path)
        captured["kwargs"] = kwargs
        return np.full((8, 8, 3), 32768, dtype=np.uint16)

    monkeypatch.setattr("transformation_portal.spatial_ai.ingest.contracts.decode_contract", fail_decode_contract)
    monkeypatch.setattr("transformation_portal.lux_depth_v3.raw_loader.load_raw_as_rgb", fake_load_raw_as_rgb)

    result = decode_for_lux_depth(raw_path, _raw_cfg("auto"))

    assert result.shape == (8, 8, 3)
    assert result.dtype == np.float32
    assert np.allclose(result.mean(), 32768 / 65535.0, atol=1e-6)
    assert captured["path"] == raw_path
    assert captured["kwargs"] == {
        "use_camera_wb": False,
        "half_size": False,
        "output_bps": 16,
        "output_linear": True,
        "python_executable": None,
    }


def test_auto_mode_surfaces_auto_wb_retry_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "scene_07.dng"
    raw_path.write_bytes(b"phase_c1_fake_raw_payload")

    def fail_decode_contract(*_args, **_kwargs):
        raise RuntimeError(
            "RAW metadata: camera_whitebalance has zero or negative gain(s) at channel(s) [3]: [1.9, 1.0, 1.6, 0.0]."
        )

    def fail_load_raw_as_rgb(*_args, **_kwargs):
        raise ValueError("rawpy auto WB failed")

    monkeypatch.setattr("transformation_portal.spatial_ai.ingest.contracts.decode_contract", fail_decode_contract)
    monkeypatch.setattr("transformation_portal.lux_depth_v3.raw_loader.load_raw_as_rgb", fail_load_raw_as_rgb)

    with pytest.raises(RawIngestError, match="Auto-WB retry also failed: rawpy auto WB failed"):
        _ = decode_for_lux_depth(raw_path, _raw_cfg("auto"))


def test_capture_provenance_records_ingest_digest_fields(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), color=(120, 80, 40)).save(image_path)

    provenance = capture_provenance(
        image_path=image_path,
        config_fingerprint="sha256:" + ("f" * 64),
        require_exiftool=False,
        ingest_profile="tp.raw_ingest.deterministic_v1",
        ingest_settings_hash="a" * 64,
    )

    assert provenance.ingest_context.ingest_profile == "tp.raw_ingest.deterministic_v1"
    assert provenance.ingest_context.ingest_settings_hash == "a" * 64
