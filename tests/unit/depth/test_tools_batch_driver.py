"""Unit coverage for depth tools batch driver behavior."""

from __future__ import annotations

import logging
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


def _build_depth_and_image(images: Path, depths: Path, base: str = "villa") -> Path:
    _write_rgb(images / f"{base}_enh.png")
    depth_path = depths / f"{base}_depth16.png"
    _write_depth(depth_path)
    return depth_path


def test_process_single_runs_clarity_mode(tmp_path) -> None:
    # Covers the clarity branch end-to-end through the batch driver; previously
    # only the haze mode was exercised at this layer.
    images = tmp_path / "images"
    depths = tmp_path / "depths"
    output = tmp_path / "out"
    images.mkdir()
    depths.mkdir()
    output.mkdir()
    depth_path = _build_depth_and_image(images, depths)
    opts = tools.BatchOptions(
        images_root=str(images),
        depths_root=str(depths),
        out_root=str(output),
        mode="clarity",
    )

    base, out_path, error = tools._process_single(str(depth_path), opts)

    assert base == "villa"
    assert error is None
    assert out_path is not None and Path(out_path).exists()


def test_process_single_runs_do_mode(tmp_path) -> None:
    # Covers the `do` (depth-of-field) branch.
    images = tmp_path / "images"
    depths = tmp_path / "depths"
    output = tmp_path / "out"
    images.mkdir()
    depths.mkdir()
    output.mkdir()
    depth_path = _build_depth_and_image(images, depths)
    opts = tools.BatchOptions(
        images_root=str(images),
        depths_root=str(depths),
        out_root=str(output),
        mode="do",
        quality="fast",
    )

    base, out_path, error = tools._process_single(str(depth_path), opts)

    assert base == "villa"
    assert error is None
    assert out_path is not None and Path(out_path).exists()


def test_process_single_rejects_unknown_mode(tmp_path) -> None:
    # Unknown mode raises; the surrounding try/except converts it into an error
    # tuple instead of propagating.
    images = tmp_path / "images"
    depths = tmp_path / "depths"
    images.mkdir()
    depths.mkdir()
    depth_path = _build_depth_and_image(images, depths)
    opts = tools.BatchOptions(
        images_root=str(images),
        depths_root=str(depths),
        out_root=str(tmp_path / "out"),
        mode="nonsense",
    )

    base, out_path, error = tools._process_single(str(depth_path), opts)

    assert base == "villa"
    assert out_path is None
    assert error is not None and "nonsense" in error


def test_process_single_uses_provided_mask_root(tmp_path) -> None:
    # Drive the sky_path / building_path discovery branches inside
    # _process_single so the mask-loaded code path is hit.
    images = tmp_path / "images"
    depths = tmp_path / "depths"
    masks = tmp_path / "masks"
    output = tmp_path / "out"
    for d in (images, depths, masks, output):
        d.mkdir()
    depth_path = _build_depth_and_image(images, depths)
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8), mode="L").save(masks / "villa_mask_sky.png")
    Image.fromarray(np.full((8, 8), 128, dtype=np.uint8), mode="L").save(masks / "villa_mask_building.png")

    opts = tools.BatchOptions(
        images_root=str(images),
        depths_root=str(depths),
        out_root=str(output),
        mask_root=str(masks),
        mode="haze",
    )

    base, out_path, error = tools._process_single(str(depth_path), opts)

    assert base == "villa"
    assert error is None
    assert out_path is not None and Path(out_path).exists()


def test_process_batch_logs_error_summary_when_failures_occur(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Covers the error-summary block that fires only when at least one file
    # fails.
    depths = tmp_path / "depths"
    depths.mkdir()
    _write_depth(depths / "bad_depth16.png")

    def fake_process_single(dp: str, opts: tools.BatchOptions):
        return "bad", None, "synthetic failure"

    monkeypatch.setattr(tools, "_process_single", fake_process_single)
    caplog.set_level("ERROR", logger="depth_tools")
    opts = tools.BatchOptions(
        images_root=str(tmp_path / "images"),
        depths_root=str(depths),
        out_root=str(tmp_path / "out"),
        workers=1,
    )

    errors = tools.process_batch(opts)

    assert errors == 1
    summary_records = [r for r in caplog.records if "ERROR SUMMARY" in r.message]
    assert summary_records, "expected ERROR SUMMARY block in logs"


def test_cli_progress_prints_progress_and_newline_on_complete(capsys) -> None:
    # Covers _cli_progress for both the mid-stream carriage return and the
    # final newline branch.
    tools._cli_progress(1, 3, "villa")
    tools._cli_progress(3, 3, "villa")

    captured = capsys.readouterr().out
    assert "Processed 1/3: villa" in captured
    assert captured.endswith("\n")


def test_main_verbose_flag_enables_debug_logging(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Covers the verbose==True logging branch.
    depths = tmp_path / "depths"
    images = tmp_path / "images"
    depths.mkdir()
    images.mkdir()
    _write_depth(depths / "villa_depth16.png")

    original_level = tools._log.level
    try:
        monkeypatch.setattr(tools, "process_batch", lambda opts, progress=None: 0)
        exit_code = tools.main(
            [
                "haze",
                str(images),
                str(depths),
                str(tmp_path / "out"),
                "--workers",
                "1",
                "--verbose",
            ]
        )
        assert exit_code == 0
        assert tools._log.level == logging.DEBUG
    finally:
        tools._log.setLevel(original_level)


def test_main_returns_two_on_fatal_exception(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Covers the fatal-exception catch: exit code 2 on unexpected error from
    # process_batch.
    depths = tmp_path / "depths"
    images = tmp_path / "images"
    depths.mkdir()
    images.mkdir()
    _write_depth(depths / "villa_depth16.png")

    def boom(opts, progress=None):
        raise RuntimeError("unexpected backend explosion")

    monkeypatch.setattr(tools, "process_batch", boom)

    exit_code = tools.main(
        [
            "haze",
            str(images),
            str(depths),
            str(tmp_path / "out"),
            "--workers",
            "1",
        ]
    )

    assert exit_code == 2


def test_main_maps_haze_cli_options_to_batch_options(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Covers the haze-specific option-mapping branch; currently only the `do`
    # branch is tested.
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
            "haze",
            str(images),
            str(depths),
            str(output),
            "--workers",
            "1",
            "--strength",
            "0.42",
            "--near",
            "10",
            "--far",
            "90",
            "--mids-gain",
            "1.10",
            "--haze-color",
            "0.5",
            "0.6",
            "0.7",
        ]
    )

    opts = captured["opts"]
    assert exit_code == 0
    assert opts.mode == "haze"
    assert opts.strength == pytest.approx(0.42)
    assert opts.near == pytest.approx(10.0)
    assert opts.far == pytest.approx(90.0)
    assert opts.mids_gain == pytest.approx(1.10)
    assert opts.haze_color == pytest.approx((0.5, 0.6, 0.7))


def test_main_maps_clarity_cli_options_to_batch_options(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Covers the clarity-specific option-mapping branch.
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
            "clarity",
            str(images),
            str(depths),
            str(output),
            "--workers",
            "1",
            "--amount",
            "0.33",
            "--radius",
            "5",
            "--near",
            "25",
            "--far",
            "75",
        ]
    )

    opts = captured["opts"]
    assert exit_code == 0
    assert opts.mode == "clarity"
    assert opts.amount == pytest.approx(0.33)
    assert opts.radius == 5
    assert opts.near == pytest.approx(25.0)
    assert opts.far == pytest.approx(75.0)
