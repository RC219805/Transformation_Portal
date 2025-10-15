from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

import pytest

import golden_hour_courtyard_workflow as ghc


class _SpyCapabilities:
    def __init__(self) -> None:
        self.invocations = 0

    def assert_luxury_grade(self) -> None:
        self.invocations += 1


def test_process_courtyard_scene_invokes_pipeline(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    namespace = argparse.Namespace()

    def fake_parse_args(argv: list[str]) -> argparse.Namespace:
        captured["argv"] = list(argv)
        namespace.input = Path(argv[0])
        namespace.output = Path(argv[1]) if len(argv) > 1 and not argv[1].startswith("--") else Path("out")
        namespace.preset = "golden_hour_courtyard"
        namespace.recursive = False
        namespace.suffix = "_lux"
        namespace.overwrite = True
        namespace.compression = "tiff_lzw"
        namespace.resize_long_edge = None
        namespace.dry_run = False
        namespace.exposure = 0.08
        namespace.white_balance_temp = 5600.0
        namespace.white_balance_tint = None
        namespace.shadow_lift = 0.24
        namespace.highlight_recovery = 0.18
        namespace.midtone_contrast = 0.10
        namespace.vibrance = 0.28
        namespace.saturation = None
        namespace.clarity = 0.20
        namespace.chroma_denoise = None
        namespace.glow = 0.12
        namespace.log_level = "INFO"
        return namespace

    processed = {"count": None}

    def fake_run_pipeline(args: argparse.Namespace) -> int:
        processed["count"] = 7
        processed["namespace"] = args
        return processed["count"]

    spy = _SpyCapabilities()

    monkeypatch.setattr(ghc.ltiff, "parse_args", fake_parse_args)
    monkeypatch.setattr(ghc.ltiff, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(ghc.ltiff, "ProcessingCapabilities", lambda: spy)

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    result = ghc.process_courtyard_scene(input_dir, output_dir, overwrite=True)

    assert result == 7
    assert spy.invocations == 1
    argv = captured["argv"]
    assert argv[:2] == [str(input_dir), str(output_dir)]
    assert argv[2:4] == ["--preset", "golden_hour_courtyard"]
    assert "--overwrite" in argv
    for flag, value in (
        ("--exposure", "0.08"),
        ("--shadow-lift", "0.24"),
        ("--highlight-recovery", "0.18"),
        ("--vibrance", "0.28"),
        ("--clarity", "0.2"),
        ("--luxury-glow", "0.12"),
        ("--white-balance-temp", "5600"),
        ("--midtone-contrast", "0.1"),
    ):
        assert flag in argv
        assert argv[argv.index(flag) + 1].startswith(value)


def test_process_courtyard_scene_rejects_unknown_override():
    with pytest.raises(ValueError):
        ghc.process_courtyard_scene("in", "out", overrides={"unknown": 1.0})


def test_process_courtyard_scene_allows_override_removal(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    def fake_parse_args(argv: list[str]) -> argparse.Namespace:
        captured["argv"] = list(argv)
        return argparse.Namespace()

    monkeypatch.setattr(ghc.ltiff, "parse_args", fake_parse_args)
    monkeypatch.setattr(ghc.ltiff, "run_pipeline", lambda args: 0)
    monkeypatch.setattr(ghc.ltiff, "ProcessingCapabilities", lambda: _SpyCapabilities())

    ghc.process_courtyard_scene(tmp_path / "input", overrides={"vibrance": None})

    argv = captured["argv"]
    assert "--vibrance" not in argv
