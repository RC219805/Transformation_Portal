"""Utilities to compare Lantern logo raster assets.

The brand team frequently exports the vector master to PNG before handing the
files to vendors.  When we iterate on the pixel art rendition we want to be
confident that the new image retains the limited palette and only introduces
the intentional tweaks.  The helpers in this module replicate the quick
ImageMagick/Pillow checks the design team usually runs by hand and wrap them in
an easy to lint, dependency-light command line program.

Example
-------
>>> python lantern_pixel_guard.py lantern_logo.png lantern_final.png \
...     --diff lantern_diff.png --max-pixel-change 12
Color count change: 12 -> 12 (Δ0)
Changed pixels: 24 / 16384 (0.15%)
Maximum channel delta: 9
Mean channel delta: 0.02

If any of the optional guard-rail arguments are exceeded the program exits with
status code ``1`` so it can slot directly into CI pipelines.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageChops


@dataclass
class ImageMetrics:
    """Summary statistics for a raster comparison."""

    original_colors: int
    candidate_colors: int
    changed_pixels: int
    total_pixels: int
    max_delta: int
    mean_delta: float

    @property
    def color_delta(self) -> int:
        """Return the difference in color count."""
        return self.candidate_colors - self.original_colors

    @property
    def changed_ratio(self) -> float:
        """Return the ratio of changed pixels to total pixels."""
        if self.total_pixels == 0:
            return 0.0
        return self.changed_pixels / self.total_pixels

    def to_dict(self) -> dict[str, float]:
        """Return the metrics as a dictionary."""
        return {
            "original_colors": self.original_colors,
            "candidate_colors": self.candidate_colors,
            "color_delta": self.color_delta,
            "changed_pixels": self.changed_pixels,
            "total_pixels": self.total_pixels,
            "changed_ratio": self.changed_ratio,
            "max_delta": self.max_delta,
            "mean_delta": self.mean_delta,
        }


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGBA")


def _count_colors(image: Image.Image) -> int:
    data = np.array(image)
    unique = np.unique(data.reshape(-1, data.shape[-1]), axis=0)
    return int(unique.shape[0])


def _compute_metrics(original: Image.Image, candidate: Image.Image) -> ImageMetrics:
    if original.size != candidate.size:
        raise ValueError(
            "Images must share the same dimensions to compare palette and pixels"
        )

    original_colors = _count_colors(original)
    candidate_colors = _count_colors(candidate)

    original_arr = np.array(original, dtype=np.int16)
    candidate_arr = np.array(candidate, dtype=np.int16)
    deltas = np.abs(original_arr - candidate_arr)

    max_delta = int(deltas.max(initial=0))
    mean_delta = float(deltas.mean())

    changed_mask = np.any(deltas > 0, axis=-1)
    changed_pixels = int(np.count_nonzero(changed_mask))
    total_pixels = int(original_arr.shape[0] * original_arr.shape[1])

    return ImageMetrics(
        original_colors=original_colors,
        candidate_colors=candidate_colors,
        changed_pixels=changed_pixels,
        total_pixels=total_pixels,
        max_delta=max_delta,
        mean_delta=mean_delta,
    )


def _save_diff(original: Image.Image, candidate: Image.Image, destination: Path) -> None:
    diff = ImageChops.difference(original, candidate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    diff.save(destination)


def _validate_thresholds(metrics: ImageMetrics, args: argparse.Namespace) -> Iterable[str]:
    if args.max_pixel_change is not None and metrics.max_delta > args.max_pixel_change:
        yield (
            f"Maximum pixel delta {metrics.max_delta} exceeds"
            f" --max-pixel-change {args.max_pixel_change}"
        )
    if args.max_color_count is not None and metrics.candidate_colors > args.max_color_count:
        yield (
            f"Candidate color count {metrics.candidate_colors} exceeds"
            f" --max-color-count {args.max_color_count}"
        )
    if (
        args.max_color_delta is not None
        and metrics.color_delta > args.max_color_delta
    ):
        yield (
            f"Color count delta {metrics.color_delta} exceeds"
            f" --max-color-delta {args.max_color_delta}"
        )


def _print_metrics(metrics: ImageMetrics) -> None:
    delta_symbol = f"Δ{metrics.color_delta:+d}" if metrics.color_delta else "Δ0"
    print(
        f"Color count change: {metrics.original_colors} ->"
        f" {metrics.candidate_colors} ({delta_symbol})"
    )
    changed_percentage = metrics.changed_ratio * 100
    print(
        "Changed pixels:"
        f" {metrics.changed_pixels} / {metrics.total_pixels}"
        f" ({changed_percentage:.2f}%)"
    )
    print(f"Maximum channel delta: {metrics.max_delta}")
    print(f"Mean channel delta: {metrics.mean_delta:.4f}")


def _export_json(path: Path, metrics: ImageMetrics) -> None:
    payload = metrics.to_dict()
    path.write_text(json.dumps(payload, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path, help="Reference PNG asset")
    parser.add_argument("candidate", type=Path, help="Updated PNG asset")
    parser.add_argument(
        "--diff",
        type=Path,
        default=None,
        help="Optional path to write a channel-difference visualization",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Optional path to export metrics as JSON"
    )
    parser.add_argument(
        "--max-pixel-change",
        type=int,
        default=None,
        help="Fail if any channel delta exceeds this value",
    )
    parser.add_argument(
        "--max-color-count",
        type=int,
        default=None,
        help="Fail if the candidate image contains more unique colors",
    )
    parser.add_argument(
        "--max-color-delta",
        type=int,
        default=None,
        help="Fail if the color-count increase exceeds this delta",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the script."""
    args = parse_args(argv)

    original = _load_image(args.original)
    candidate = _load_image(args.candidate)

    metrics = _compute_metrics(original, candidate)

    if args.diff is not None:
        _save_diff(original, candidate, args.diff)

    if args.json is not None:
        _export_json(args.json, metrics)

    _print_metrics(metrics)

    failures = list(_validate_thresholds(metrics, args))
    if failures:
        for message in failures:
            print(f"ERROR: {message}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())