"""Unit coverage for depth tools batch driver behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transformation_portal.depth import tools

pytestmark = pytest.mark.unit


def _write_rgb(path: Path, value: int = 128) -> None:
    rgb = np.full((8, 8, 3), value, dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _write_depth(path: Path) -> None:
    depth = np.linspace(0, 65535, 8 * 8, dtype=np.uint16).reshape(8, 8)
    Image.fromarray(depth, mode="I;16").save(path)


def test_process_single_writes_output_with_default_tiff_format(tmp_path) -> None:
    images = tmp_path / "images"
    depths = tmp_path / "depths"
    output = tmp_path / "out"
    images.mkdir()
    depths.mkdir()
    output.mkdir()
    _write_rgb(images / "villa_enh.png")
    depth_path = depths / "villa_depth16.png"
    _write_depth(depth_path)

    opts = tools.BatchOptions(images_root=str(images), depths_root=str(depths), out_root=str(output), workers=1)

    base, out_path, error = tools._process_single(str(depth_path), opts)

    assert base == "villa"
    assert error is None
    assert out_path is not None
    assert Path(out_path).exists()
    assert Path(out_path).suffix in {".tif", ".png"}


def test_process_single_skips_missing_sources_when_configured(tmp_path) -> None:
    depth_path = tmp_path / "depths" / "missing_depth16.png"
    depth_path.parent.mkdir()
    _write_depth(depth_path)
    opts = tools.BatchOptions(
        images_root=str(tmp_path / "images"),
        depths_root=str(depth_path.parent),
        out_root=str(tmp_path / "out"),
        skip_missing=True,
    )

    base, out_path, error = tools._process_single(str(depth_path), opts)

    assert base == "missing"
    assert out_path is None
    assert error == "No source image found for base missing"


def test_process_single_reports_missing_sources_when_fail_missing(tmp_path) -> None:
    depth_path = tmp_path / "depths" / "missing_depth16.png"
    depth_path.parent.mkdir()
    _write_depth(depth_path)
    opts = tools.BatchOptions(
        images_root=str(tmp_path / "images"),
        depths_root=str(depth_path.parent),
        out_root=str(tmp_path / "out"),
        skip_missing=False,
    )

    base, out_path, error = tools._process_single(str(depth_path), opts)

    assert base == "missing"
    assert out_path is None
    assert error == "No source image found for base missing"


def test_process_batch_tracks_errors_and_progress_with_single_worker(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    depths = tmp_path / "depths"
    depths.mkdir()
    first = depths / "ok_depth16.png"
    second = depths / "bad_depth16.png"
    _write_depth(first)
    _write_depth(second)
    progress: list[tuple[int, int, str]] = []

    def fake_process_single(dp: str, opts: tools.BatchOptions) -> tuple[str, str | None, str | None]:
        if Path(dp).name.startswith("ok"):
            return "ok", str(tmp_path / "out" / "ok.png"), None
        return "bad", None, "failed by fixture"

    monkeypatch.setattr(tools, "_process_single", fake_process_single)
    opts = tools.BatchOptions(
        images_root=str(tmp_path / "images"),
        depths_root=str(depths),
        out_root=str(tmp_path / "out"),
        workers=1,
    )

    error_count = tools.process_batch(opts, progress=lambda done, total, base: progress.append((done, total, base)))

    assert error_count == 1
    assert progress == [(1, 2, "bad"), (2, 2, "ok")]
    assert (tmp_path / "out").is_dir()


def test_process_batch_raises_when_no_depth_maps_exist(tmp_path) -> None:
    opts = tools.BatchOptions(
        images_root=str(tmp_path / "images"),
        depths_root=str(tmp_path / "depths"),
        out_root=str(tmp_path / "out"),
        workers=1,
    )
    Path(opts.depths_root).mkdir()

    with pytest.raises(SystemExit, match="No depth maps found"):
        tools.process_batch(opts)


def test_main_returns_success_for_partial_success_when_one_file_succeeds(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    depths = tmp_path / "depths"
    images = tmp_path / "images"
    output = tmp_path / "out"
    depths.mkdir()
    images.mkdir()
    _write_depth(depths / "ok_depth16.png")
    _write_depth(depths / "bad_depth16.png")
    monkeypatch.setattr(tools, "process_batch", lambda opts, progress=None: 1)

    exit_code = tools.main(
        [
            "haze",
            str(images),
            str(depths),
            str(output),
            "--workers",
            "1",
            "--allow-partial-success",
        ]
    )

    assert exit_code == 0


def test_main_returns_failure_for_strict_partial_errors(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    depths = tmp_path / "depths"
    images = tmp_path / "images"
    output = tmp_path / "out"
    depths.mkdir()
    images.mkdir()
    _write_depth(depths / "bad_depth16.png")
    monkeypatch.setattr(tools, "process_batch", lambda opts, progress=None: 1)

    exit_code = tools.main(["clarity", str(images), str(depths), str(output), "--workers", "1"])

    assert exit_code == 1


def test_main_returns_failure_when_partial_success_all_files_fail(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    depths = tmp_path / "depths"
    images = tmp_path / "images"
    output = tmp_path / "out"
    depths.mkdir()
    images.mkdir()
    _write_depth(depths / "bad_depth16.png")
    monkeypatch.setattr(tools, "process_batch", lambda opts, progress=None: 1)

    exit_code = tools.main(
        [
            "do",
            str(images),
            str(depths),
            str(output),
            "--workers",
            "1",
            "--allow-partial-success",
        ]
    )

    assert exit_code == 1


def test_main_auto_workers_handles_missing_cpu_count(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, tools.BatchOptions] = {}
    depths = tmp_path / "depths"
    images = tmp_path / "images"
    output = tmp_path / "out"
    depths.mkdir()
    images.mkdir()
    _write_depth(depths / "villa_depth16.png")

    def fake_process_batch(opts: tools.BatchOptions, progress=None) -> int:
        captured["opts"] = opts
        return 0

    monkeypatch.setattr(tools.os, "cpu_count", lambda: None)
    monkeypatch.setattr(tools, "process_batch", fake_process_batch)

    exit_code = tools.main(["haze", str(images), str(depths), str(output)])

    assert exit_code == 0
    assert captured["opts"].workers == 1


def test_main_maps_cli_effect_options_to_batch_options(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, tools.BatchOptions] = {}
    depths = tmp_path / "depths"
    images = tmp_path / "images"
    output = tmp_path / "out"
    depths.mkdir()
    images.mkdir()
    _write_depth(depths / "villa_depth16.png")

    def fake_process_batch(opts: tools.BatchOptions, progress=None) -> int:
        captured["opts"] = opts
        return 0

    monkeypatch.setattr(tools, "process_batch", fake_process_batch)

    exit_code = tools.main(
        [
            "do",
            str(images),
            str(depths),
            str(output),
            "--workers",
            "1",
            "--fmt",
            "png",
            "--quality",
            "fast",
            "--focus",
            "40",
            "--aperture",
            "0.3",
            "--clarity",
            "0.1",
            "--falloff",
            "2.0",
        ]
    )

    opts = captured["opts"]
    assert exit_code == 0
    assert opts.mode == "do"
    assert opts.workers == 1
    assert opts.fmt == "png"
    assert opts.quality == "fast"
    assert opts.focus == 40.0
    assert opts.aperture == 0.3
    assert opts.clarity == 0.1
    assert opts.falloff == 2.0


def test_main_preserves_legacy_fallof_alias(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, tools.BatchOptions] = {}
    depths = tmp_path / "depths"
    images = tmp_path / "images"
    output = tmp_path / "out"
    depths.mkdir()
    images.mkdir()
    _write_depth(depths / "villa_depth16.png")

    def fake_process_batch(opts: tools.BatchOptions, progress=None) -> int:
        captured["opts"] = opts
        return 0

    monkeypatch.setattr(tools, "process_batch", fake_process_batch)

    exit_code = tools.main(
        [
            "do",
            str(images),
            str(depths),
            str(output),
            "--workers",
            "1",
            "--fallof",
            "2.5",
        ]
    )

    assert exit_code == 0
    assert captured["opts"].falloff == 2.5
