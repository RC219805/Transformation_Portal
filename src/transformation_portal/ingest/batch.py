"""Deterministic mixed-media ingest batch runner and manifest builder."""

from __future__ import annotations

import hashlib
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..lux_depth_v3.io_atomic import atomic_write_bytes, durable_unlink, publication_lock
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


def discover_batch_inputs(
    input_dir: Path,
    *,
    recursive: bool = True,
) -> list[Path]:
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
            config_dict={
                "mode": "ingest_batch",
                "normalization_profile": profile,
            },
            preset="batch",
        )
        return sidecar.model_dump()

    return _capture


def _normalized_relpath(input_relative_path: Path) -> Path:
    filename = f"{input_relative_path.name}.normalized.json"
    return input_relative_path.with_name(filename)


def _batch_root_projection(
    items: Sequence[Mapping[str, str]],
    profile: str,
) -> dict[str, Any]:
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


def compute_batch_root_sha256(
    items: Sequence[Mapping[str, str]],
    *,
    profile: str,
) -> str:
    """Compute deterministic pre-Merkle batch root.

    The helper enforces deterministic ordering by sorting on
    ``relative_path`` (with ``normalized_json_sha256`` as a stable
    tiebreaker) so callers do not need to pre-sort inputs.
    """
    ordered_items = sorted(
        items,
        key=lambda item: (
            item["relative_path"],
            item["normalized_json_sha256"],
        ),
    )
    projection = _batch_root_projection(ordered_items, profile)
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
    """Run batch ingest normalization and emit deterministic manifest."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
    if not manifest_filename.endswith(".json"):
        raise ValueError("manifest_filename must end with .json")

    output_dir.mkdir(parents=True, exist_ok=True)
    ingest_fn = ingest_payload_factory or _default_ingest_payload_factory(profile)

    inputs = discover_batch_inputs(input_dir, recursive=recursive)
    items: list[dict[str, str]] = []
    # Keep only paths and digests in memory. Until all inputs have been staged,
    # a failed ingest leaves the prior manifest and its outputs untouched.
    with tempfile.TemporaryDirectory(prefix=".ingest-", dir=output_dir) as staging_dir:
        spool = Path(staging_dir)
        pending_writes: list[tuple[Path, Path]] = []
        for input_path in inputs:
            relative_path = input_path.relative_to(input_dir)
            raw_payload = ingest_fn(input_path)
            if not isinstance(raw_payload, Mapping):
                raise TypeError(
                    "ingest_payload_factory must return"
                    " a mapping payload, got "
                    f"{type(raw_payload).__name__}"
                    f" for {input_path}"
                )

            normalized_payload = normalize_machine_payload(dict(raw_payload), profile=profile)
            normalized_bytes = canonical_json_bytes(normalized_payload)
            normalized_sha256 = hashlib.sha256(normalized_bytes).hexdigest()
            normalized_relpath = Path("normalized") / _normalized_relpath(relative_path)
            staged_path = spool / str(len(items))
            staged_path.write_bytes(normalized_bytes)
            pending_writes.append((staged_path, output_dir / normalized_relpath))

            items.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "normalized_json_relpath": normalized_relpath.as_posix(),
                    "normalized_json_sha256": normalized_sha256,
                }
            )

        items.sort(key=lambda item: item["relative_path"])
        manifest = {
            "schema": BATCH_MANIFEST_SCHEMA,
            "normalization_profile": profile,
            "item_count": len(items),
            "items": items,
            "batch_root_sha256": compute_batch_root_sha256(items, profile=profile),
        }
        staged_manifest = spool / "manifest"
        staged_manifest.write_bytes(canonical_json_bytes(manifest))

        manifest_path = output_dir / manifest_filename
        with publication_lock(manifest_path):
            try:
                manifest_mode = stat.S_IMODE(manifest_path.stat().st_mode)
            except FileNotFoundError:
                manifest_mode = stat.S_IMODE(staged_manifest.stat().st_mode)
            # A failed publication must leave no manifest attesting to a mix
            # of old and new bytes. Publish its replacement only after the data.
            durable_unlink(manifest_path)
            for staged_path, destination in pending_writes:
                staged_mode = stat.S_IMODE(staged_path.stat().st_mode)
                atomic_write_bytes(destination, staged_path.read_bytes(), new_file_mode=staged_mode)
            atomic_write_bytes(manifest_path, staged_manifest.read_bytes(), new_file_mode=manifest_mode)
    return manifest
