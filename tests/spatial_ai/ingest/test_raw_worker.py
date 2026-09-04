"""Unit tests for the isolated RAW ingest worker entrypoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from transformation_portal.spatial_ai.ingest import raw_worker

pytestmark = [pytest.mark.unit]


def test_probe_reads_visible_dimensions_without_postprocessing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "sample.dng"
    input_path.write_bytes(b"raw")
    output_array = tmp_path / "probe.npy"
    output_json = tmp_path / "probe.json"
    calls: dict[str, object] = {"postprocess": 0}

    class FakeRaw:
        sizes = SimpleNamespace(height=3024, width=4032)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def postprocess(self, **_kwargs):
            calls["postprocess"] = int(calls["postprocess"]) + 1
            raise AssertionError("dimension probe must not demosaic the RAW input")

    def fake_imread(path: str):
        calls["input_path"] = path
        return FakeRaw()

    monkeypatch.setitem(sys.modules, "rawpy", SimpleNamespace(imread=fake_imread))

    exit_code = raw_worker.main(
        [
            "--command",
            "probe",
            "--input-path",
            str(input_path),
            "--output-array",
            str(output_array),
            "--output-json",
            str(output_json),
        ]
    )

    assert exit_code == 0
    assert calls == {"postprocess": 0, "input_path": str(input_path.resolve())}
    assert np.load(output_array, allow_pickle=False).shape == (0,)
    assert json.loads(output_json.read_text(encoding="utf-8")) == {"input_size": [3024, 4032]}


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
