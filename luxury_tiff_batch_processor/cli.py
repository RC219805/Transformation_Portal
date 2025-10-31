"""Command-line interface wiring for the luxury TIFF batch processor."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None

from .adjustments import AdjustmentSettings, LUXURY_PRESETS
from .pipeline import (
    _process_image_worker,
    _wrap_with_progress,
    collect_images,
    ensure_output_path,
    process_single_image,
)
from .profiles import DEFAULT_PROFILE_NAME, PROCESSING_PROFILES

LOGGER = logging.getLogger("luxury_tiff_batch_processor")


def _load_config_data(path: Path) -> Mapping[str, Any]:
    """Load configuration from JSON or YAML file.

    Args:
        path: Path to configuration file (.json, .yaml, or .yml).

    Returns:
        Dictionary mapping configuration keys to values.

    Raises:
        FileNotFoundError: If configuration file doesn't exist.
        RuntimeError: If YAML file requested but pyyaml not installed.
        ValueError: If file content is not a valid mapping.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("YAML configuration files require the optional 'pyyaml' dependency")
            data = yaml.safe_load(path.read_text())  # type: ignore[no-untyped-call]
        else:
            data = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - exact exception varies by backend
        raise ValueError(f"Unable to parse configuration file {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Configuration file {path} must contain a mapping of option names to values")
    return data


def _normalise_config_keys(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Convert configuration keys to CLI-compatible underscore format.

    Args:
        raw: Raw configuration dictionary with potentially hyphenated keys.

    Returns:
        Dictionary with keys normalized to underscore format.

    Raises:
        ValueError: If any key is not a string.
    """
    normalised: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError("Configuration keys must be strings")
        normalised[key.replace("-", "_")] = value
    return normalised


def _build_parser_aliases(parser: argparse.ArgumentParser) -> tuple[dict[str, argparse.Action], dict[str, str]]:
    """Build lookup tables mapping argument names to parser actions.

    Args:
        parser: Configured ArgumentParser instance.

    Returns:
        Tuple of (dest_to_action, alias_to_dest) dictionaries for resolving
        configuration file keys to parser actions.
    """
    dest_to_action: dict[str, argparse.Action] = {}
    alias_to_dest: dict[str, str] = {}
    # Use public methods to get all actions
    actions = list(parser._get_positional_actions()) + list(parser._get_optional_actions())
    for action in actions:
        if action.dest in {argparse.SUPPRESS, "help", "config"}:
            continue
        dest_to_action[action.dest] = action
        alias_to_dest[action.dest.replace("-", "_")] = action.dest
        for option_string in action.option_strings:
            alias = option_string.lstrip("-").replace("-", "_")
            alias_to_dest[alias] = action.dest
    return dest_to_action, alias_to_dest


def _coerce_config_value(
    action: argparse.Action, value: Any, *, source: Path, key: str
) -> Any:  # pragma: no cover - thin wrapper around argparse semantics
    """Convert configuration values so they match argparse expectations."""

    if value is None:
        return None

    if isinstance(action, argparse._StoreTrueAction):  # type: ignore[attr-defined]
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        raise ValueError(
            f"Invalid boolean for '{key}' in {source}: expected true/false value, got {value!r}"
        )

    if isinstance(action, argparse._StoreFalseAction):  # type: ignore[attr-defined]
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        raise ValueError(
            f"Invalid boolean for '{key}' in {source}: expected true/false value, got {value!r}"
        )

    if action.type is not None:
        try:
            converted = action.type(value)
        except Exception as exc:  # pragma: no cover - delegated to argparse
            raise ValueError(f"Invalid value for '{key}' in {source}: {exc}") from exc
    else:
        converted = value

    if action.choices is not None and converted not in action.choices:
        raise ValueError(
            f"Invalid value for '{key}' in {source}: {converted!r} (choose from {sorted(action.choices)})"
        )

    return converted


def default_output_folder(input_folder: Path) -> Path:
    """Return the default output folder for a given input directory."""

    if input_folder.name:
        return input_folder.parent / f"{input_folder.name}_lux"
    return input_folder / "luxury_output"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch enhance TIFF files for ultra-luxury marketing output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional configuration file (JSON by default, YAML when 'pyyaml' is installed)",
    )
    parser.add_argument("input", type=Path, help="Folder that contains source TIFF files")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Folder where processed files will be written. Defaults to '<input>_lux' next to the input folder.",
    )
    parser.add_argument(
        "--preset",
        default="signature",
        choices=sorted(LUXURY_PRESETS.keys()),
        help="Adjustment preset that provides a starting point",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_NAME,
        choices=sorted(PROCESSING_PROFILES.keys()),
        help="Processing profile balancing fidelity and speed",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process folders recursively and mirror the directory tree in the output",
    )
    parser.add_argument(
        "--suffix",
        default="_lux",
        help="Filename suffix appended before the extension for processed files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the destination",
    )
    parser.add_argument(
        "--compression",
        default="tiff_lzw",
        help="TIFF compression to use when saving (as understood by Pillow)",
    )
    parser.add_argument(
        "--resize-long-edge",
        type=int,
        default=None,
        help="Optionally resize the longest image edge to this many pixels while preserving aspect ratio",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview the work without writing any files")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress reporting (useful for minimal or non-interactive environments)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel image processing",
    )

    # Fine control overrides.
    parser.add_argument("--exposure", type=float, default=None, help="Exposure adjustment in stops")
    parser.add_argument(
        "--white-balance-temp",
        type=float,
        default=None,
        dest="white_balance_temp",
        help="Target color temperature in Kelvin",
    )
    parser.add_argument(
        "--white-balance-tint",
        type=float,
        default=None,
        dest="white_balance_tint",
        help="Green-magenta tint compensation (positive skews magenta)",
    )
    parser.add_argument("--shadow-lift", type=float, default=None, help="Shadow recovery strength (0-1)")
    parser.add_argument(
        "--highlight-recovery",
        type=float,
        default=None,
        help="Highlight compression strength (0-1)",
    )
    parser.add_argument(
        "--midtone-contrast", type=float, default=None, dest="midtone_contrast", help="Midtone contrast strength"
    )
    parser.add_argument("--vibrance", type=float, default=None, help="Vibrance strength (0-1)")
    parser.add_argument("--saturation", type=float, default=None, help="Additional saturation multiplier delta")
    parser.add_argument("--clarity", type=float, default=None, help="Local contrast boost strength (0-1)")
    parser.add_argument(
        "--chroma-denoise",
        type=float,
        default=None,
        dest="chroma_denoise",
        help="Chrominance denoising amount (0-1)",
    )
    parser.add_argument(
        "--luxury-glow", type=float, default=None, dest="glow", help="Diffusion glow strength (0-1)"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )

    argv_list = list(argv) if argv is not None else None

    config_probe, _ = parser.parse_known_args(argv_list)
    if config_probe.config is not None:
        try:
            raw_config = _load_config_data(config_probe.config)
            normalised_config = _normalise_config_keys(raw_config)
            dest_to_action, alias_to_dest = _build_parser_aliases(parser)

            converted_defaults: dict[str, Any] = {}
            for key, value in normalised_config.items():
                dest = alias_to_dest.get(key)
                if dest is None:
                    raise ValueError(
                        f"Unknown configuration option '{key}' in {config_probe.config}"
                    )
                action = dest_to_action[dest]
                converted_defaults[dest] = _coerce_config_value(
                    action, value, source=config_probe.config, key=key
                )

            parser.set_defaults(**converted_defaults)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))

    args = parser.parse_args(argv_list)
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    if args.output is None:
        args.output = default_output_folder(args.input)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    return args


def build_adjustments(args: argparse.Namespace) -> AdjustmentSettings:
    """Construct adjustment settings from preset and CLI overrides.

    Args:
        args: Parsed command-line arguments.

    Returns:
        AdjustmentSettings with preset values overridden by CLI arguments.
    """
    base = dataclasses.replace(LUXURY_PRESETS[args.preset])
    for field in dataclasses.fields(base):
        value = getattr(args, field.name, None)
        if value is not None:
            setattr(base, field.name, value)
            base._validate()
    LOGGER.debug("Using adjustments: %s", base)
    return base


def _ensure_non_overlapping(input_root: Path, output_root: Path) -> None:
    def _contains(parent: Path, child: Path) -> bool:
        try:
            child.relative_to(parent)
        except ValueError:
            return False
        return True

    if input_root == output_root:
        raise SystemExit("Output folder must be different from the input folder to avoid self-overwrites.")
    if _contains(input_root, output_root):
        raise SystemExit(
            "Output folder cannot be located inside the input folder; choose a sibling or separate directory."
        )
    if _contains(output_root, input_root):
        raise SystemExit(
            "Input folder cannot be located inside the output folder; choose non-overlapping directories."
        )


def run_pipeline(args: argparse.Namespace) -> int:
    """Run the batch processor with the provided arguments."""

    run_id = uuid.uuid4().hex
    adjustments = build_adjustments(args)
    profile = PROCESSING_PROFILES[args.profile]
    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")
    if not input_root.is_dir():
        raise SystemExit(f"Input folder '{input_root}' does not exist or is not a directory")

    _ensure_non_overlapping(input_root, output_root)

    LOGGER.info(
        "Starting batch run %s for %s using '%s' profile",
        run_id,
        input_root,
        profile.name,
    )
    images = sorted(collect_images(input_root, args.recursive))
    if not images:
        LOGGER.warning("No TIFF images found in %s (run %s)", input_root, run_id)
        return 0

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Found %s image(s) to process", len(images))
    processed = 0

    workers = getattr(args, "workers", 1)
    resize_long_edge = getattr(args, "resize_long_edge", None)
    resize_target = getattr(args, "resize_target", None)
    compression = profile.resolve_compression(args.compression)

    if workers <= 1:
        progress_iterable = _wrap_with_progress(
            images,
            total=len(images),
            description="Processing images",
            enabled=not getattr(args, "no_progress", False),
        )

        for image_path in progress_iterable:
            destination = ensure_output_path(
                input_root,
                output_root,
                image_path,
                args.suffix,
                args.recursive,
                create=not args.dry_run,
            )
            if destination.exists() and not args.overwrite and not args.dry_run:
                LOGGER.warning("Skipping %s (exists, use --overwrite to replace)", destination)
                continue
            if args.dry_run:
                LOGGER.info("Dry run: would process %s -> %s", image_path, destination)
            # pylint: disable=duplicate-code  # Similar call in pipeline.py with same params
            process_single_image(
                image_path,
                destination,
                adjustments,
                compression=compression,
                resize_long_edge=resize_long_edge,
                resize_target=resize_target,
                dry_run=args.dry_run,
                profile=profile,
            )
            # pylint: enable=duplicate-code
            if not args.dry_run:
                processed += 1
    else:
        progress_range = _wrap_with_progress(
            range(len(images)),
            total=len(images),
            description="Processing images",
            enabled=not getattr(args, "no_progress", False),
        )
        progress_iterator = iter(progress_range)

        def advance_progress() -> None:
            try:
                next(progress_iterator)
            except StopIteration:
                pass

        futures = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for image_path in images:
                destination = ensure_output_path(
                    input_root,
                    output_root,
                    image_path,
                    args.suffix,
                    args.recursive,
                    create=not args.dry_run,
                )
                if destination.exists() and not args.overwrite and not args.dry_run:
                    LOGGER.warning("Skipping %s (exists, use --overwrite to replace)", destination)
                    advance_progress()
                    continue
                if args.dry_run:
                    LOGGER.info("Dry run: would process %s -> %s", image_path, destination)
                futures.append(
                    executor.submit(
                        _process_image_worker,
                        image_path,
                        destination,
                        adjustments,
                        compression=compression,
                        resize_long_edge=resize_long_edge,
                        resize_target=resize_target,
                        dry_run=args.dry_run,
                        profile=profile,
                    )
                )

            for future in as_completed(futures):
                try:
                    wrote_output = future.result()
                except Exception:
                    advance_progress()
                    raise
                if wrote_output:
                    processed += 1
                advance_progress()

    LOGGER.info("Finished batch run %s; processed %s image(s)", run_id, processed)
    return processed


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_pipeline(args)


__all__ = [
    "build_adjustments",
    "default_output_folder",
    "main",
    "parse_args",
    "run_pipeline",
]
