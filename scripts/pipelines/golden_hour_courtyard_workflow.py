"""Golden Hour Courtyard helper utilities.

This module wraps :mod:`luxury_tiff_batch_processor` with the preset and
parameter overrides that the Material Response review requested for the
Montecito coastal courtyard aerial.  It keeps the command-line surface while
providing a Python API that callers can reuse in notebooks or orchestration
scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import luxury_tiff_batch_processor as ltiff


@dataclass(frozen=True)
class _AdjustmentFlag:
    """Mapping between adjustment attribute names and CLI flags."""

    attribute: str
    flag: str


_ADJUSTMENT_FLAGS: Sequence[_AdjustmentFlag] = (
    _AdjustmentFlag("exposure", "--exposure"),
    _AdjustmentFlag("white_balance_temp", "--white-balance-temp"),
    _AdjustmentFlag("white_balance_tint", "--white-balance-tint"),
    _AdjustmentFlag("shadow_lift", "--shadow-lift"),
    _AdjustmentFlag("highlight_recovery", "--highlight-recovery"),
    _AdjustmentFlag("midtone_contrast", "--midtone-contrast"),
    _AdjustmentFlag("vibrance", "--vibrance"),
    _AdjustmentFlag("saturation", "--saturation"),
    _AdjustmentFlag("clarity", "--clarity"),
    _AdjustmentFlag("chroma_denoise", "--chroma-denoise"),
    _AdjustmentFlag("glow", "--luxury-glow"),
)


_DEFAULT_GOLDEN_HOUR_OVERRIDES: Dict[str, float] = {
    "exposure": 0.08,
    "shadow_lift": 0.24,
    "highlight_recovery": 0.18,
    "vibrance": 0.28,
    "clarity": 0.20,
    "glow": 0.12,
    "white_balance_temp": 5600.0,
    "midtone_contrast": 0.10,
}


def _format_value(value: float) -> str:
    """Render numeric overrides for the CLI while preserving precision."""

    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _merge_overrides(overrides: Mapping[str, float | None] | None) -> Dict[str, float]:
    merged: MutableMapping[str, float] = dict(_DEFAULT_GOLDEN_HOUR_OVERRIDES)
    if overrides is None:
        return dict(merged)

    valid_attributes = {entry.attribute for entry in _ADJUSTMENT_FLAGS}
    for attribute, value in overrides.items():
        if attribute not in valid_attributes:
            raise ValueError(f"Unknown adjustment override '{attribute}'")
        if value is None:
            merged.pop(attribute, None)
        else:
            merged[attribute] = float(value)

    return dict(merged)


def _build_cli_vector(
    input_dir: Path | str,
    output_dir: Path | str | None,
    *,
    recursive: bool,
    overwrite: bool,
    dry_run: bool,
    suffix: str,
    compression: str,
    resize_long_edge: int | None,
    log_level: str,
    overrides: Mapping[str, float | None] | None,
) -> list[str]:
    vector: list[str] = [str(input_dir)]
    if output_dir is not None:
        vector.append(str(output_dir))

    vector.extend(["--preset", "golden_hour_courtyard"])

    if recursive:
        vector.append("--recursive")
    if overwrite:
        vector.append("--overwrite")
    if dry_run:
        vector.append("--dry-run")
    if suffix != "_lux":
        vector.extend(["--suffix", suffix])
    if compression != "tiff_lzw":
        vector.extend(["--compression", compression])
    if resize_long_edge is not None:
        vector.extend(["--resize-long-edge", str(int(resize_long_edge))])
    if log_level.upper() != "INFO":
        vector.extend(["--log-level", log_level.upper()])

    merged_overrides = _merge_overrides(overrides)
    flag_lookup = {entry.attribute: entry.flag for entry in _ADJUSTMENT_FLAGS}
    for attribute, value in merged_overrides.items():
        flag = flag_lookup[attribute]
        vector.extend([flag, _format_value(value)])

    return vector


def process_courtyard_scene(
    input_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    recursive: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    suffix: str = "_lux",
    compression: str = "tiff_lzw",
    resize_long_edge: int | None = None,
    log_level: str = "INFO",
    overrides: Mapping[str, float | None] | None = None,
) -> int:
    """Process a folder of TIFFs using the Golden Hour Courtyard recipe.

    Parameters
    ----------
    input_dir:
        Directory that contains the source TIFF files.
    output_dir:
        Destination directory.  When ``None`` the processor mirrors the CLI
        behaviour and writes to ``<input>_lux`` next to the input folder.
    recursive:
        Mirror the input folder tree recursively when ``True``.
    overwrite:
        Permit overwriting of existing files in the destination.
    dry_run:
        Print the proposed work without writing files when ``True``.
    suffix:
        Filename suffix appended before the extension for processed files.
    compression:
        TIFF compression passed through to Pillow/tifffile.
    resize_long_edge:
        Optional long-edge clamp applied before grading.
    log_level:
        Logging verbosity understood by :mod:`luxury_tiff_batch_processor`.
    overrides:
        Mapping of adjustment attribute names to override values.  ``None``
        values remove the default Golden Hour override so the preset value is
        used instead.

    Returns
    -------
    int
        Number of successfully processed images as reported by
        :func:`luxury_tiff_batch_processor.run_pipeline`.
    """

    cli_vector = _build_cli_vector(
        input_dir,
        output_dir,
        recursive=recursive,
        overwrite=overwrite,
        dry_run=dry_run,
        suffix=suffix,
        compression=compression,
        resize_long_edge=resize_long_edge,
        log_level=log_level,
        overrides=overrides,
    )

    capabilities = ltiff.ProcessingCapabilities()
    capabilities.assert_luxury_grade()

    args = ltiff.parse_args(cli_vector)
    return ltiff.run_pipeline(args)


__all__ = ["process_courtyard_scene"]
