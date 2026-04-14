"""Unit tests for the isolated RAW ingest worker entrypoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transformation_portal.spatial_ai.ingest import raw_worker

pytestmark = [pytest.mark.unit]


def test_main_resolves_relative_input_path_before_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    input_path = tmp_path / "inputs" / "sample.cr2"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"raw")
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("{}", encoding="utf-8")

    captured: dict[str, Path] = {}

    def fake_run_load_rgb(resolved_input_path: Path, payload: dict[str, object]):
        captured["input_path"] = resolved_input_path
        captured["payload"] = payload
        return np.zeros((1, 1, 3), dtype=np.uint8), {"dtype": "uint8"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(raw_worker, "_run_load_rgb", fake_run_load_rgb)
    monkeypatch.setattr(raw_worker, "_write_outputs", lambda *args, **kwargs: 0)

    exit_code = raw_worker.main(
        [
            "--command",
            "load_rgb",
            "--input-path",
            "inputs/sample.cr2",
            "--payload-json",
            str(payload_path),
            "--output-array",
            "out.npy",
            "--output-json",
            "out.json",
        ]
    )

    assert exit_code == 0
    assert captured["input_path"] == input_path.resolve()
    assert captured["input_path"].is_absolute()
    assert captured["payload"] == {}
