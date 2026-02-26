"""Deterministic mixed-media ingest batch runner and manifest builder."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .normalize_machine_json import DEFAULT_NORMALIZATION_PROFILE, canonical_json_bytes, normalize_machine_payload
from .provenance import capture_provenance

BATCH_MANIFEST_SCHEMA = "tp.ingest.batch_manifest.v1"
BATCH_MANIFEST_FILENAME = "batch_manifest.normalized.json"
SUPPORTED_BATCH_EXTENSIONS = frozenset(
    {
        ".cr2",
        ".cr3",
        ".nef",
        ".nrw",
        ".arw",
        ".srf",
        ".dng",
        ".raf",
        ".orf",
        ".rw2",
        ".pef",
        ".srw",
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
        ".png",
        ".heic",
        ".heif",
        ".mov",
    }
)

IngestPayloadFactory = Callable[[Path], Mapping[str, Any]]


def discover_batch_inputs(input_dir: Path, *, recursive: bool = True) -> list[Path]:
    """Discover supported inputs in deterministic lexicographic order."""
    if recursive:
        candidates = input_dir.rglob("*")
    else:
        candidates = input_dir.glob("*")
    return sorted(
        (path for path in candidates if path.is_file() and path.suffix.lower() in SUPPORTED_BATCH_EXTENSIONS),
        key=lambda path: path.as_posix(),
    )


def _default_ingest_payload_factory(profile: str) -> IngestPayloadFactory:
    def _capture(path: Path) -> Mapping[str, Any]:
        sidecar = capture_provenance(
            input_path=path,
            cli_args=["--batch-normalization-profile", profile],
            config_dict={"mode": "ingest_batch", "normalization_profile": profile},
            preset="batch",
        )
        return sidecar.model_dump()

    return _capture


def _normalized_relpath(input_relative_path: Path) -> Path:
    filename = f"{input_relative_path.name}.normalized.json"
    return input_relative_path.with_name(filename)


def _batch_root_projection(items: Sequence[Mapping[str, str]], profile: str) -> dict[str, Any]:
    return {
        "normalization_profile": profile,
        "items": [
            {
                "relative_path": item["relative_path"],
                "normalized_json_sha256": item["normalized_json_sha256"],
            }
            for item in items
        ],
    }


def compute_batch_root_sha256(items: Sequence[Mapping[str, str]], *, profile: str) -> str:
    """Compute deterministic pre-Merkle batch root from normalized-item digests."""
    projection = _batch_root_projection(items, profile)
    return hashlib.sha256(canonical_json_bytes(projection)).hexdigest()


def run_ingest_batch(
    *,
    input_dir: Path,
    output_dir: Path,
    profile: str = DEFAULT_NORMALIZATION_PROFILE,
    ingest_payload_factory: IngestPayloadFactory | None = None,
    recursive: bool = True,
    manifest_filename: str = BATCH_MANIFEST_FILENAME,
) -> dict[str, Any]:
    """Run batch ingest normalization and emit deterministic manifest output."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
    if not manifest_filename.endswith(".json"):
        raise ValueError("manifest_filename must end with .json")

    output_dir.mkdir(parents=True, exist_ok=True)
    ingest_fn = ingest_payload_factory or _default_ingest_payload_factory(profile)

    inputs = discover_batch_inputs(input_dir, recursive=recursive)
    items: list[dict[str, str]] = []

    for input_path in inputs:
        relative_path = input_path.relative_to(input_dir)
        raw_payload = ingest_fn(input_path)
        if not isinstance(raw_payload, Mapping):
            raise TypeError(
                f"ingest_payload_factory must return a mapping payload, got {type(raw_payload).__name__} for {input_path}"
            )

        normalized_payload = normalize_machine_payload(dict(raw_payload), profile=profile)
        normalized_bytes = canonical_json_bytes(normalized_payload)
        normalized_sha256 = hashlib.sha256(normalized_bytes).hexdigest()

        normalized_relpath = Path("normalized") / _normalized_relpath(relative_path)
        normalized_output_path = output_dir / normalized_relpath
        normalized_output_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_output_path.write_bytes(normalized_bytes)

        items.append(
            {
                "relative_path": relative_path.as_posix(),
                "normalized_json_relpath": normalized_relpath.as_posix(),
                "normalized_json_sha256": normalized_sha256,
            }
        )

    items = sorted(items, key=lambda item: item["relative_path"])
    batch_root_sha256 = compute_batch_root_sha256(items, profile=profile)
    manifest = {
        "schema": BATCH_MANIFEST_SCHEMA,
        "normalization_profile": profile,
        "item_count": len(items),
        "items": items,
        "batch_root_sha256": batch_root_sha256,
    }

    manifest_path = output_dir / manifest_filename
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest
