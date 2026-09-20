"""Create source-bound material evidence and inspect evaluation results.

Photographic application belongs to the existing Lux executor. This command
produces its explicit, versioned companion inputs; it never selects a fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path
from typing import Any

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.core.execution_plan_v2 import require_digest
from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.ingest.canonical_json import canonicalize_json, dumps_json
from transformation_portal.lux_depth_v4.companions import _mask_array, _object, _portable_path
from transformation_portal.lux_depth_v4.io import directory_path, snapshot
from transformation_portal.lux_depth_v4.io import write_evidence as write_bytes

from .artifacts import load_evidence, write_evidence
from .contracts import MaterialEvidence, MaterialLimits, RegionEvidence
from .engine import ResponsePolicy


def _read(path: Path, maximum_bytes: int) -> tuple[bytes, dict[str, Any]]:
    path = path.expanduser().absolute()
    root = directory_path(path.parent)
    return snapshot(root, root / path.name, maximum_bytes=maximum_bytes)


def _source(args: argparse.Namespace) -> ImageMaster:
    from transformation_portal.lux_depth_v4.photography import decode_master

    data, _ = _read(args.source, args.max_input_bytes)
    return decode_master(data, source_name=args.source.name, input_color=args.input_color, max_pixels=args.max_pixels)


def _supplied(args: argparse.Namespace, master: ImageMaster) -> MaterialEvidence:
    data, _ = _read(args.regions, min(args.max_input_bytes, 1024 * 1024))
    payload = decode_bounded_json_object(data)
    _object(payload, {"schema", "source_sha256", "shape", "regions"})
    if payload["schema"] != "tp.materials.supplied.v1":
        raise ValueError("Unsupported supplied-region schema")
    if payload["source_sha256"] != master.source_sha256 or payload["shape"] != list(master.shape):
        raise ValueError("Supplied regions do not bind the canonical source bytes and geometry")
    regions = payload["regions"]
    if not isinstance(regions, list) or not 1 <= len(regions) <= MaterialLimits().max_regions:
        raise ValueError("Supplied regions require a bounded nonempty list")
    root = directory_path(args.regions.expanduser().absolute().parent)
    observations = []
    total = len(data)
    decoded_bytes = len(regions) * master.shape[0] * master.shape[1] * 4
    if decoded_bytes > min(args.max_input_bytes, MaterialLimits().max_mask_bytes):
        raise ValueError("Supplied masks exceed the aggregate decoded byte budget")
    for record in regions:
        _object(record, {"region_id", "label", "mask_path", "mask_sha256", "semantic_confidence"})
        relative = _portable_path(record["mask_path"])
        require_digest(record["mask_sha256"])
        mask_bytes, receipt = snapshot(
            root,
            root / relative,
            maximum_bytes=min(args.max_input_bytes - total, master.shape[0] * master.shape[1] * 4 + 4096 + 12),
        )
        total += receipt["size_bytes"]
        if receipt["sha256"] != record["mask_sha256"]:
            raise ValueError("Supplied mask digest differs from its declaration")
        if len(mask_bytes) < 10 or mask_bytes[:6] != b"\x93NUMPY":
            raise ValueError("Supplied mask requires a numeric NPY file")
        version = tuple(mask_bytes[6:8])
        if version == (1, 0):
            header_size = struct.unpack("<H", mask_bytes[8:10])[0]
        elif version == (2, 0) and len(mask_bytes) >= 12:
            header_size = struct.unpack("<I", mask_bytes[8:12])[0]
        else:
            raise ValueError("Supplied masks require NPY format 1 or 2")
        if not 0 < header_size <= 4096:
            raise ValueError("Supplied mask header exceeds its byte budget")
        mask = _mask_array(mask_bytes, args.max_pixels)
        if mask.shape != master.shape:
            raise ValueError("Supplied mask geometry differs from the canonical source")
        observations.append(
            RegionEvidence(
                record["region_id"],
                record["label"],
                mask,
                semantic_confidence=record["semantic_confidence"],
                provenance="supplied",
            )
        )
    return MaterialEvidence(
        master.source_sha256,
        master.shape,
        tuple(observations),
        producer={"kind": "caller_supplied", "source_manifest_sha256": hashlib.sha256(data).hexdigest()},
    )


def _publish(evidence: MaterialEvidence, args: argparse.Namespace) -> dict[str, Any]:
    relative = _portable_path(args.source_relative_path or args.source.name)
    destination = directory_path(args.output_dir, allow_missing=True)
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    write_evidence(evidence, destination / "evidence.json")
    _, receipt = snapshot(destination, destination / "evidence.json", maximum_bytes=1024 * 1024, retain_bytes=False)
    manifest = {
        "schema": "tp.lux.materials_manifest.v1",
        "inputs": [
            {
                "path": relative,
                "source_sha256": evidence.source_sha256,
                "evidence_path": "evidence.json",
                "evidence_sha256": receipt["sha256"],
                "shape": list(evidence.shape),
            }
        ],
    }
    write_bytes(destination, "lux-materials.json", canonicalize_json(manifest))
    return {
        "schema": "tp.materials.preparation.v1",
        "evidence": str(destination / "evidence.json"),
        "materials_manifest": str(destination / "lux-materials.json"),
        "content_sha256": evidence.content_hash(),
        "status": evidence.status,
        "regions": len(evidence.regions),
        "production_acceptance": "pending",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materials V4 evidence tools; photographic edits run through lux-depth-v4.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("policy", help="Print the complete conservative response policy")
    evaluate = commands.add_parser("evaluate", help="Evaluate declared held-out labels; never asserts acceptance")
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    for name in ("import-supplied", "inspect", "infer"):
        command = commands.add_parser(name)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--input-color", choices=("auto", "srgb", "linear_srgb"), default="auto")
        command.add_argument("--max-pixels", type=int, default=100_000_000)
        command.add_argument("--max-input-bytes", type=int, default=1024**3)
        if name == "inspect":
            command.add_argument("--evidence", type=Path, required=True)
        else:
            command.add_argument("--output-dir", type=Path, required=True)
            command.add_argument("--source-relative-path")
        if name == "import-supplied":
            command.add_argument("--regions", type=Path, required=True)
        if name == "infer":
            command.add_argument("--sam2-checkpoint", type=Path, required=True)
            command.add_argument("--clip-checkpoint", type=Path, required=True)
            command.add_argument("--device", choices=("cpu", "mps"), default="cpu")
            command.add_argument("--proxy-longest-side", type=int, default=1024)
            command.add_argument("--max-proposals", type=int, default=64)
            command.add_argument("--classifier-batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        if args.command == "policy":
            result = ResponsePolicy().to_payload()
        elif args.command == "evaluate":
            from .evaluation import evaluate_dataset

            data, _ = _read(args.input, 16 * 1024 * 1024)
            result = evaluate_dataset(decode_bounded_json_object(data))
            if args.output is not None:
                output = args.output.expanduser().absolute()
                write_bytes(directory_path(output.parent), output.name, canonicalize_json(result))
            print(dumps_json(result, sort_keys=True))
            return 0 if result.get("status") == "measured" else 2
        else:
            if not 0 < args.max_input_bytes <= 2 * 1024**3 or not 0 < args.max_pixels <= MaterialLimits().max_pixels:
                raise ValueError("Source budgets must be positive and within the supported bounds")
            master = _source(args)
            if args.command == "inspect":
                result = load_evidence(
                    args.evidence,
                    master.source_sha256,
                    master.shape,
                    limits=MaterialLimits(
                        max_pixels=args.max_pixels,
                        max_bundle_bytes=min(args.max_input_bytes, MaterialLimits().max_bundle_bytes),
                        max_mask_bytes=min(args.max_input_bytes, MaterialLimits().max_mask_bytes),
                    ),
                ).to_payload()
            elif args.command == "import-supplied":
                result = _publish(_supplied(args, master), args)
            else:
                from .inference import InferenceConfig, infer_materials, prepare_inference

                config = InferenceConfig(
                    sam2_checkpoint=args.sam2_checkpoint,
                    clip_checkpoint=args.clip_checkpoint,
                    device=args.device,
                    proxy_longest_side=args.proxy_longest_side,
                    max_proposals=args.max_proposals,
                    classifier_batch_size=args.classifier_batch_size,
                )
                evidence = infer_materials(master, prepare_inference(master, config))
                result = _publish(evidence, args)
        print(dumps_json(result, sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"materials-v4: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
