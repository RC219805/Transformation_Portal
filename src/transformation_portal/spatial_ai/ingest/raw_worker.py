"""Subprocess worker for isolated RAW ingest operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ...ingest.canonical_json import dump_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RAW ingest operations in an isolated Python environment.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only validate that RAW ingest dependencies import cleanly.",
    )
    parser.add_argument(
        "--command",
        choices=("load_rgb", "linear_decode", "phase2_decode"),
        help="RAW operation to execute.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        help="Input RAW image path.",
    )
    parser.add_argument(
        "--payload-json",
        type=Path,
        help="JSON file with structured command options.",
    )
    parser.add_argument(
        "--output-array",
        type=Path,
        help="Output .npy path for the array payload.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output JSON path for structured metadata.",
    )
    return parser


def _load_payload(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("RAW worker payload must be a JSON object.")
    return loaded


def _check_availability() -> int:
    import rawpy  # noqa: F401

    from ...lux_depth_v3.raw_loader import load_raw_as_rgb
    from .linear_decoder import LinearDecoder
    from .phase2_camera_native_linear import ingest_phase2_xyz_d50_linear_fp32

    _ = (load_raw_as_rgb, LinearDecoder, ingest_phase2_xyz_d50_linear_fp32)
    return 0


def _run_load_rgb(input_path: Path, payload: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
    from ...lux_depth_v3.raw_loader import load_raw_as_rgb

    array = load_raw_as_rgb(
        input_path,
        use_camera_wb=bool(payload.get("use_camera_wb", True)),
        half_size=bool(payload.get("half_size", False)),
        output_bps=int(payload.get("output_bps", 8)),
        output_linear=bool(payload.get("output_linear", False)),
        python_executable=None,
        demosaic=str(payload.get("demosaic", "AHD")),
    )
    return array, {
        "dtype": str(array.dtype),
        "shape": [int(dim) for dim in array.shape],
    }


def _run_linear_decode(input_path: Path, payload: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
    from .linear_decoder import LinearDecoder

    decoder = LinearDecoder(
        gamma=float(payload.get("gamma", 1.0)),
        bit_depth=int(payload.get("bit_depth", 32)),
        strict_ingest=bool(payload.get("strict_ingest", False)),
        raw_python_executable=None,
        demosaic=str(payload.get("demosaic", "AHD")),
    )
    result = decoder.decode(input_path)
    return result.linear_rgb, {
        "input_size": [int(result.input_size[0]), int(result.input_size[1])],
        "input_format": result.input_format,
        "color_space": result.color_space,
        "ingest_fingerprint": result.ingest_fingerprint,
        "dtype": result.dtype,
    }


def _run_phase2_decode(input_path: Path, payload: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
    from .phase2_camera_native_linear import ingest_phase2_xyz_d50_linear_fp32

    tensor, fingerprint = ingest_phase2_xyz_d50_linear_fp32(
        input_path,
        wb_mode=str(payload.get("wb_mode", "camera")),
        demosaic=str(payload.get("demosaic", "AHD")),
        raw_python_executable=None,
    )
    return tensor, {"fingerprint": fingerprint}


def _write_outputs(output_array: Path, output_json: Path, array: np.ndarray, payload: Dict[str, Any]) -> int:
    output_array.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_array, np.asarray(array), allow_pickle=False)
    with output_json.open("w", encoding="utf-8") as handle:
        dump_json(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.check:
        return _check_availability()

    if args.command is None or args.input_path is None or args.output_array is None or args.output_json is None:
        parser.error("--command, --input-path, --output-array, and --output-json are required unless --check is used.")

    payload = _load_payload(args.payload_json)
    input_path = args.input_path.expanduser().resolve()

    if args.command == "load_rgb":
        array, metadata = _run_load_rgb(input_path, payload)
    elif args.command == "linear_decode":
        array, metadata = _run_linear_decode(input_path, payload)
    else:
        array, metadata = _run_phase2_decode(input_path, payload)

    return _write_outputs(
        args.output_array.expanduser(),
        args.output_json.expanduser(),
        array,
        metadata,
    )


if __name__ == "__main__":
    raise SystemExit(main())
