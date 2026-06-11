"""Offline coverage for ``luxury_video_master_grader`` CLI orchestration.

The existing ``tests/test_luxury_video_master_grader.py`` exercises the pure
helpers (filter graph, command builder, frame-rate assessment, probe summary).
This suite fills the remaining cold-zone region — ``build_config`` and the
``main()`` control flow — without invoking real ffmpeg/ffprobe: ``probe_source``
and ``ensure_tools_available`` are monkeypatched, and ``--dry-run`` exits before
the subprocess call. Covers the early-exit/error return codes (2/3/4/7), the
dry-run success path, the print-filter-graph branch, the real-run success and
``ffmpeg`` failure paths, plus ``ensure_tools_available`` and the
``--list-presets`` action.
"""

from __future__ import annotations

import subprocess

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.processors import luxury_video_master_grader as grader


def _probe() -> dict:
    return {
        "format": {"duration": "12.5"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30000/1001",
            }
        ],
    }


@pytest.fixture
def stub_environment(monkeypatch):
    """Make ffmpeg/ffprobe appear available and probing deterministic."""
    monkeypatch.setattr(grader, "ensure_tools_available", lambda: None)
    monkeypatch.setattr(grader, "probe_source", lambda path: _probe())


@pytest.fixture
def video_paths(tmp_path):
    src = tmp_path / "in.mov"
    src.write_bytes(b"fake video bytes")
    out = tmp_path / "out.mov"
    return src, out


@pytest.fixture
def lut_file(tmp_path):
    """A real (empty) .cube file so build_filter_graph's existence check passes.

    The preset LUTs live under an external assets/ tree that is not committed,
    so reaching the filter-graph stage requires overriding with --custom-lut.
    """
    lut = tmp_path / "look.cube"
    lut.write_text("# minimal cube placeholder\n")
    return lut


# --------------------------------------------------------------------------- #
# build_config
# --------------------------------------------------------------------------- #


def test_build_config_defaults_from_preset() -> None:
    args = grader.parse_arguments(["in.mov", "out.mov"])
    config = grader.build_config(args)
    expected = grader.PRESETS["signature_estate"].to_dict()
    # Default preset config is returned unchanged when no overrides are given.
    assert config["lut_strength"] == expected["lut_strength"]


def test_build_config_applies_overrides() -> None:
    args = grader.parse_arguments(
        [
            "in.mov",
            "out.mov",
            "--lut-strength",
            "0.5",
            "--contrast",
            "1.2",
            "--saturation",
            "1.1",
            "--gamma",
            "0.9",
            "--brightness",
            "0.1",
            "--grain",
            "0.3",
            "--halation-intensity",
            "0.4",
            "--halation-radius",
            "8",
            "--halation-threshold",
            "0.7",
        ]
    )
    config = grader.build_config(args, target_fps="24000/1001")
    assert config["lut_strength"] == 0.5
    assert config["contrast"] == 1.2
    assert config["saturation"] == 1.1
    assert config["gamma"] == 0.9
    assert config["brightness"] == 0.1
    assert config["grain"] == 0.3
    assert config["halation_intensity"] == 0.4
    assert config["halation_radius"] == 8
    assert config["halation_threshold"] == 0.7
    assert config["target_fps"] == "24000/1001"


def test_build_config_applies_choice_and_tint_overrides() -> None:
    args = grader.parse_arguments(
        [
            "in.mov",
            "out.mov",
            "--warmth",
            "0.2",
            "--cool",
            "0.1",
            "--sharpen",
            "of",
            "--denoise",
            "of",
            "--deband",
            "of",
        ]
    )
    config = grader.build_config(args)
    assert config["warmth"] == 0.2
    assert config["cool"] == 0.1
    assert config["sharpen"] == "of"
    assert config["denoise"] == "of"
    assert config["deband"] == "of"


def test_build_config_merges_enabled_tone_map_plan() -> None:
    args = grader.parse_arguments(["in.mov", "out.mov"])
    plan = grader.ToneMapPlan(enabled=True, note="x", config={"tone_map_peak": 600.0}, metadata=(None, None, None))
    config = grader.build_config(args, tone_map_plan=plan)
    assert config["tone_map_peak"] == 600.0


# --------------------------------------------------------------------------- #
# main() — success paths
# --------------------------------------------------------------------------- #


def test_main_dry_run_returns_zero(stub_environment, video_paths, lut_file, capsys) -> None:
    src, out = video_paths
    rc = grader.main([str(src), str(out), "--dry-run", "--custom-lut", str(lut_file)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Dry run requested" in captured.out
    assert "FFmpeg command:" in captured.out


def test_main_print_filter_graph_branch(stub_environment, video_paths, lut_file, capsys) -> None:
    src, out = video_paths
    rc = grader.main([str(src), str(out), "--dry-run", "--print-filter-graph", "--custom-lut", str(lut_file)])
    assert rc == 0
    assert "Filter graph (human-readable):" in capsys.readouterr().out


def test_main_real_run_invokes_ffmpeg(stub_environment, video_paths, lut_file, monkeypatch, capsys) -> None:
    src, out = video_paths
    calls = []
    monkeypatch.setattr(grader.subprocess, "run", lambda cmd, check: calls.append(cmd))
    rc = grader.main([str(src), str(out), "--custom-lut", str(lut_file)])
    assert rc == 0
    assert calls and calls[0][0] == "ffmpeg"
    assert "Master grade created" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# main() — error/exit paths
# --------------------------------------------------------------------------- #


def test_main_missing_input_returns_2(stub_environment, tmp_path, capsys) -> None:
    rc = grader.main([str(tmp_path / "absent.mov"), str(tmp_path / "out.mov"), "--dry-run"])
    assert rc == 2
    assert "Input video not found" in capsys.readouterr().err


def test_main_existing_output_without_overwrite_returns_3(stub_environment, video_paths, capsys) -> None:
    src, out = video_paths
    out.write_bytes(b"already here")
    rc = grader.main([str(src), str(out), "--dry-run"])
    assert rc == 3
    assert "already exists" in capsys.readouterr().err


def test_main_probe_failure_returns_4(monkeypatch, video_paths, capsys) -> None:
    src, out = video_paths
    monkeypatch.setattr(grader, "ensure_tools_available", lambda: None)

    def _raise(path):
        raise RuntimeError("ffprobe blew up")

    monkeypatch.setattr(grader, "probe_source", _raise)
    rc = grader.main([str(src), str(out), "--dry-run"])
    assert rc == 4
    assert "ffprobe error" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value", "needle"),
    [
        ("--contrast", "0", "contrast"),
        ("--saturation", "0", "saturation"),
        ("--gamma", "0", "gamma"),
        ("--brightness", "2", "brightness"),
        ("--lut-strength", "2", "lut_strength"),
        ("--halation-intensity", "-1", "halation_intensity"),
        ("--halation-radius", "-1", "halation_radius"),
        ("--halation-threshold", "2", "halation_threshold"),
    ],
)
def test_main_parameter_validation_matrix(stub_environment, video_paths, lut_file, capsys, flag, value, needle) -> None:
    src, out = video_paths
    rc = grader.main([str(src), str(out), "--dry-run", "--custom-lut", str(lut_file), flag, value])
    assert rc == 7
    assert needle in capsys.readouterr().err


def test_main_filter_graph_failure_returns_5(stub_environment, video_paths, capsys) -> None:
    # No --custom-lut: the default preset points at an uncommitted asset LUT, so
    # build_filter_graph raises FileNotFoundError -> main returns 5.
    src, out = video_paths
    rc = grader.main([str(src), str(out), "--dry-run"])
    assert rc == 5
    assert "Failed to build filter graph" in capsys.readouterr().err


def test_main_frame_rate_error_returns_6(stub_environment, video_paths, lut_file, capsys) -> None:
    src, out = video_paths
    rc = grader.main([str(src), str(out), "--dry-run", "--custom-lut", str(lut_file), "--target-fps", "not-a-rate"])
    assert rc == 6
    assert "Frame-rate configuration error" in capsys.readouterr().err


def test_main_ffmpeg_failure_propagates_returncode(stub_environment, video_paths, lut_file, monkeypatch) -> None:
    src, out = video_paths

    def _fail(cmd, check):
        raise subprocess.CalledProcessError(returncode=42, cmd=cmd)

    monkeypatch.setattr(grader.subprocess, "run", _fail)
    rc = grader.main([str(src), str(out), "--custom-lut", str(lut_file)])
    assert rc == 42


# --------------------------------------------------------------------------- #
# ensure_tools_available + list presets
# --------------------------------------------------------------------------- #


def test_ensure_tools_available_passes_when_present(monkeypatch) -> None:
    monkeypatch.setattr(grader, "shutil_which", lambda binary: f"/usr/bin/{binary}")
    grader.ensure_tools_available()  # must not raise


def test_ensure_tools_available_raises_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(grader, "shutil_which", lambda binary: None)
    with pytest.raises(SystemExit, match="ffmpeg"):
        grader.ensure_tools_available()


def test_list_presets_action_prints_and_exits(capsys) -> None:
    with pytest.raises(SystemExit):
        grader.parse_arguments(["--list-presets"])
    assert "Available presets:" in capsys.readouterr().out


def test_list_presets_returns_all_presets() -> None:
    text = grader.list_presets()
    for key in grader.PRESETS:
        assert key in text
