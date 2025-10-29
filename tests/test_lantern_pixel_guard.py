from __future__ import annotations
from PIL import Image

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL.Image")


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (
    ROOT
    / "09_Client_Deliverables"
    / "Lantern_Logo_Implementation_Kit"
    / "lantern_pixel_guard.py"
)


def _write_png(path: Path, pixels: list[list[tuple[int, int, int, int]]]) -> None:
    height = len(pixels)
    width = len(pixels[0]) if height else 0
    image = Image.new("RGBA", (width, height))
    for y, row in enumerate(pixels):
        for x, value in enumerate(row):
            image.putpixel((x, y), value)
    image.save(path)


def test_metrics_and_diff(tmp_path: Path) -> None:
    original = tmp_path / "lantern_logo.png"
    candidate = tmp_path / "lantern_final.png"
    diff = tmp_path / "lantern_diff.png"
    metrics_json = tmp_path / "metrics.json"

    _write_png(
        original,
        [
            [(10, 10, 10, 255), (10, 10, 10, 255)],
            [(10, 10, 10, 255), (40, 40, 40, 255)],
        ],
    )
    _write_png(
        candidate,
        [
            [(10, 10, 10, 255), (10, 10, 10, 255)],
            [(22, 10, 10, 255), (40, 44, 40, 255)],
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(original),
            str(candidate),
            "--diff",
            str(diff),
            "--json",
            str(metrics_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = completed.stdout
    assert "Color count change: 2 -> 3" in stdout
    assert "Maximum channel delta:" in stdout

    diff_image = Image.open(diff)
    assert diff_image.getpixel((0, 0)) == (0, 0, 0, 0)
    assert diff_image.getpixel((0, 1))[0] == 12  # delta on red channel

    metrics = json.loads(metrics_json.read_text())
    assert metrics["original_colors"] == 2
    assert metrics["candidate_colors"] == 3
    assert metrics["changed_pixels"] == 2


def test_threshold_failure(tmp_path: Path) -> None:
    original = tmp_path / "lantern_logo.png"
    candidate = tmp_path / "lantern_final.png"

    _write_png(
        original,
        [[(0, 0, 0, 255), (0, 0, 0, 255)]],
    )
    _write_png(
        candidate,
        [[(30, 0, 0, 255), (0, 0, 0, 255)]],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(original),
            str(candidate),
            "--max-pixel-change",
            "10",
            "--max-color-delta",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "ERROR" in completed.stderr
    assert "--max-pixel-change 10" in completed.stderr
    assert "--max-color-delta 0" in completed.stderr
