"""Fail-closed materialized runtime identity for the isolated DA3 worker.

The helper intentionally performs local-only Hugging Face resolution.  A
missing snapshot or governance lock makes the result non-authorizing; cache
preparation must never turn a cache lookup into an ungoverned download.
"""

from __future__ import annotations

import base64
import binascii
import contextvars
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import stat
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ...core.da3_runtime import find_repo_root
from ...ingest.canonical_json import canonicalize_json, dumps_json

DA3_RUNTIME_IDENTITY_SCHEMA = "tp.da3.runtime-identity.v1"
DA3_RUNTIME_GOVERNANCE_SCHEMA = "tp.da3.runtime-governance.v1"
DA3_WEIGHT_MANIFEST_SCHEMA = "tp.da3.materialized-weights.v1"
DA3_MODEL_MANIFEST_SCHEMA = "tp.da3.materialized-model.v1"
DA3_DEPENDENCY_IDENTITY_SCHEMA = "tp.da3.dependencies.v1"
DA3_SOURCE_IDENTITY_SCHEMA = "tp.da3.source.v1"
DA3_INTERPRETER_DEPENDENCY_SCHEMA = "tp.da3.interpreter-dependencies.v1"
DA3_SOURCE_MODEL_SCHEMA = "tp.da3.source-model.v1"
DA3_PARENT_RUNTIME_SCHEMA = "tp.da3.parent-output-runtime.v1"
DA3_IMPORT_ENVIRONMENT_SCHEMA = "tp.da3.import-environment.v1"

_SHA256_HEX_LENGTH = 64
_MAX_INDEX_BYTES = 16 * 1024 * 1024
_MAX_WEIGHT_MAP_ENTRIES = 250_000
_MAX_WEIGHT_FILES = 1024
_MAX_WEIGHT_BYTES = 64 * 1024 * 1024 * 1024
_MAX_CONFIG_FILES = 256
_MAX_CONFIG_BYTES = 64 * 1024 * 1024
_MAX_MODEL_TREE_ENTRIES = 20_000
_MAX_MODEL_TREE_DIRECTORIES = 4096
_MAX_SOURCE_FILES = 4096
_MAX_SOURCE_BYTES = 128 * 1024 * 1024
_MAX_SOURCE_TREE_ENTRIES = 50_000
_MAX_SOURCE_TREE_DIRECTORIES = 8192
_MAX_DISTRIBUTION_FILES = 100_000
_MAX_DISTRIBUTION_BYTES = 16 * 1024 * 1024 * 1024
_MAX_HANDSHAKE_BYTES = 4 * 1024 * 1024
_MAX_LOCK_BYTES = 4 * 1024 * 1024
_MAX_LOCK_DISTRIBUTIONS = 2048
_MAX_RUNTIME_MARKER_BYTES = 64 * 1024
_MAX_RUNTIME_BASELINE_BYTES = 16 * 1024
_MAX_GOVERNANCE_BYTES = 64 * 1024
_MAX_DISTRIBUTION_RECORD_BYTES = 16 * 1024 * 1024
_MAX_DISTRIBUTION_DIRECT_URL_BYTES = 1024 * 1024
_MAX_DISTRIBUTION_CORE_METADATA_BYTES = 4 * 1024 * 1024
_MAX_INTERPRETER_BYTES = 1024 * 1024 * 1024
_MAX_MODULE_SEARCH_ROOTS = 32
_MAX_IMPORT_SEARCH_PATHS = 256
_MAX_IMPORT_MODULES = 4096
_MAX_IMPORT_ORIGIN_BYTES = 2 * 1024 * 1024 * 1024
_MAX_IMPORT_PATH_BYTES = 4096
_MAX_IMPORT_CONFIGURATION_FILES = 1024
_MAX_IMPORT_CONFIGURATION_BYTES = 16 * 1024 * 1024
_MAX_IMPORT_CONFIGURATION_DIRECTORY_ENTRIES = 50_000
_MAX_VERIFICATION_TOKEN_BYTES = 32 * 1024 * 1024
_MAX_VERIFICATION_ENTRIES = 200_000
_DEFAULT_GOVERNANCE_PATH = "config/da3_runtime_identity_contract.json"
_RUNTIME_AUTHORITY_SCHEMA = "tp.da3.runtime-authority.v1"
_DISTRIBUTION_RECORD_KEYS = {
    "name",
    "version",
    "direct_url_sha256",
    "record_sha256",
    "installed_files_sha256",
}
_BACKEND_IDENTITY_KEYS = {
    "backend_id",
    "model_canonical_key",
    "model_repo_id",
    "model_lock_revision",
    "executed_backend",
    "requested_device",
    "actual_device",
    "materialized_weights_sha256",
    "materialized_model_sha256",
}
_DETAILED_EVIDENCE_KEYS = {
    "model_files",
    "dependencies",
    "interpreter",
    "platform",
    "accelerator",
    "source_files",
    "optional_source_modules",
    "source_revision",
    "import_environment",
}
_FILE_RECORD_KEYS = {"path", "role", "sha256", "size_bytes"}
_TP_SOURCE_MODULES = ("transformation_portal",)
_TP_OPTIONAL_SOURCE_MODULES: tuple[str, ...] = ()
_PARENT_OUTPUT_SOURCE_MODULES = ("transformation_portal",)
_PARENT_OUTPUT_OPTIONAL_SOURCE_MODULES: tuple[str, ...] = ()
_WORKER_THIRD_PARTY_IMPORT_MODULES = (
    "PIL",
    "addict",
    "cryptography",
    "cv2",
    "e3nn",
    "einops",
    "evo",
    "fastapi",
    "huggingface_hub",
    "imageio",
    "moviepy",
    "numpy",
    "omegaconf",
    "open3d",
    "pillow_heif",
    "plyfile",
    "requests",
    "safetensors",
    "torch",
    "torchvision",
    "transformers",
    "trimesh",
    "typer",
    "uvicorn",
)
_WORKER_IMPORT_MODULES = tuple(
    sorted(
        {
            *_TP_SOURCE_MODULES,
            *_TP_OPTIONAL_SOURCE_MODULES,
            *_WORKER_THIRD_PARTY_IMPORT_MODULES,
            "depth_anything_3",
        }
    )
)
_WORKER_EXTERNAL_SOURCE_IMPORT_MODULES = frozenset({*_TP_SOURCE_MODULES, "depth_anything_3"})
_WORKER_RUNTIME_ENVIRONMENT_NAMES = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TP_MODEL_LOCK_MANIFEST",
    "TP_STRICT_MODEL_LOCK",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
)
_VERIFICATION_ENTRIES: contextvars.ContextVar[dict[str, dict[str, Any]] | None] = contextvars.ContextVar(
    "da3_verification_entries",
    default=None,
)
_SOURCE_REVISION_PROBE: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "da3_source_revision_probe",
    default=None,
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
        and value != "0" * _SHA256_HEX_LENGTH
    )


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonicalize_json(dict(payload))).hexdigest()


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _record_verification_stat(path: Path, observed: os.stat_result | None = None) -> None:
    entries = _VERIFICATION_ENTRIES.get()
    if entries is None:
        return
    resolved = path.resolve(strict=True)
    stat_result = observed or resolved.stat()
    key = str(resolved)
    record = {
        "path": key,
        "kind": "directory" if stat.S_ISDIR(stat_result.st_mode) else "file",
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
        "size_bytes": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
        "ctime_ns": int(stat_result.st_ctime_ns),
    }
    previous = entries.get(key)
    if previous is not None and previous != record:
        raise ValueError(f"DA3 identity input changed while preparing verification token: {path}")
    entries[key] = record
    if len(entries) > _MAX_VERIFICATION_ENTRIES:
        raise ValueError("DA3 runtime verification token contains too many entries")


def _hash_regular_file(path: Path, *, maximum_bytes: int | None = None) -> tuple[str, int]:
    """Hash one opened regular file and reject in-flight size changes."""

    digest = hashlib.sha256()
    digest_count = 0
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"DA3 identity input is not a regular file: {path}")
        if maximum_bytes is not None and (maximum_bytes < 0 or before.st_size > maximum_bytes):
            raise ValueError(f"DA3 identity input exceeds its remaining byte budget: {path}")
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            if maximum_bytes is not None and digest_count + len(chunk) > maximum_bytes:
                raise ValueError(f"DA3 identity input exceeds its remaining byte budget: {path}")
            digest.update(chunk)
            digest_count += len(chunk)
        after = os.fstat(handle.fileno())
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"DA3 identity input changed while hashing: {path}")
    _record_verification_stat(path, after)
    return digest.hexdigest(), int(after.st_size)


def _read_bounded_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"DA3 identity input is not a regular file: {path}")
        raw = handle.read(maximum_bytes + 1)
        after = os.fstat(handle.fileno())
    if not raw or len(raw) > maximum_bytes:
        raise ValueError(f"DA3 identity input is empty or oversized: {path}")
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"DA3 identity input changed while reading: {path}")
    _record_verification_stat(path, after)
    return raw


def _validated_distribution_metadata_root(
    distribution: Any,
    *,
    distribution_name: str,
) -> tuple[Path, Path]:
    """Resolve one filesystem-backed dist-info root inside this interpreter."""

    metadata_root_value = getattr(distribution, "_path", None)
    if metadata_root_value is None:
        raise ValueError(f"DA3 dependency has no bounded metadata root: {distribution_name}")
    allowed_root = Path(sys.prefix).resolve()
    try:
        metadata_root = Path(metadata_root_value).resolve(strict=True)
        metadata_root.relative_to(allowed_root)
        observed = metadata_root.stat()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"DA3 dependency metadata escapes its interpreter: {distribution_name}") from exc
    if not stat.S_ISDIR(observed.st_mode):
        raise ValueError(f"DA3 dependency metadata root is not a directory: {distribution_name}")
    _record_verification_stat(metadata_root.parent)
    _record_verification_stat(metadata_root, observed)
    return metadata_root, allowed_root


def _distribution_metadata_identity(
    distribution: Any,
    *,
    distribution_name: str,
) -> tuple[str, str, Path, Path]:
    """Read exact Name/Version without importlib.metadata's unbounded parser."""

    metadata_root, allowed_root = _validated_distribution_metadata_root(
        distribution,
        distribution_name=distribution_name,
    )
    if metadata_root.name.endswith(".dist-info"):
        metadata_path = metadata_root / "METADATA"
    elif metadata_root.name.endswith(".egg-info"):
        metadata_path = metadata_root / "PKG-INFO"
    else:
        raise ValueError(f"DA3 dependency has an unsupported metadata root: {distribution_name}")
    try:
        raw = _read_bounded_regular_file(
            metadata_path,
            maximum_bytes=_MAX_DISTRIBUTION_CORE_METADATA_BYTES,
        )
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"DA3 dependency has invalid bounded core metadata: {distribution_name}") from exc

    values: dict[str, list[str]] = {"name": [], "version": []}
    current_header: str | None = None
    for line in text.splitlines():
        if not line:
            break
        if line[0] in " \t":
            if current_header in values:
                raise ValueError(f"DA3 dependency folds an identity field: {distribution_name}")
            continue
        field_name, separator, field_value = line.partition(":")
        if not separator or re.fullmatch(r"[A-Za-z0-9-]+", field_name) is None:
            raise ValueError(f"DA3 dependency has malformed core metadata: {distribution_name}")
        current_header = field_name.lower()
        if current_header in values:
            value = field_value.strip()
            if not value or any(ord(character) < 33 or ord(character) > 126 for character in value):
                raise ValueError(f"DA3 dependency has invalid identity metadata: {distribution_name}")
            values[current_header].append(value)

    if len(values["name"]) != 1 or len(values["version"]) != 1:
        raise ValueError(f"DA3 dependency has ambiguous Name/Version metadata: {distribution_name}")
    raw_name = values["name"][0]
    raw_version = values["version"][0]
    if (
        len(raw_name) > 512
        or re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", raw_name) is None
        or len(raw_version) > 512
    ):
        raise ValueError(f"DA3 dependency has invalid Name/Version metadata: {distribution_name}")
    return _canonical_distribution_name(raw_name), raw_version, metadata_path, allowed_root


def _file_record(
    path: Path,
    *,
    logical_path: str,
    role: str,
    maximum_bytes: int | None = None,
) -> dict[str, Any]:
    sha256, size_bytes = _hash_regular_file(path, maximum_bytes=maximum_bytes)
    _record_verification_stat(path.parent)
    return {
        "path": logical_path,
        "role": role,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _bounded_matching_files(
    root: Path,
    *,
    suffixes: set[str],
    maximum_matches: int,
    maximum_entries: int,
    maximum_directories: int,
    allow_symlink_files: bool,
) -> tuple[Path, ...]:
    """Walk without following directory symlinks and cap every visited node."""

    try:
        resolved_root = root.resolve(strict=True)
        root_stat = resolved_root.stat()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"DA3 identity tree is unavailable: {root}") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"DA3 identity tree root is not a directory: {root}")
    pending = [resolved_root]
    matches: list[Path] = []
    visited_entries = 0
    visited_directories = 0
    while pending:
        directory = pending.pop()
        visited_directories += 1
        if visited_directories > maximum_directories:
            raise ValueError(f"DA3 identity tree has too many directories: {root}")
        _record_verification_stat(directory)
        try:
            iterator = os.scandir(directory)
        except OSError as exc:
            raise ValueError(f"DA3 identity tree directory is unreadable: {directory}") from exc
        with iterator:
            for entry in iterator:
                visited_entries += 1
                if visited_entries > maximum_entries:
                    raise ValueError(f"DA3 identity tree has too many entries: {root}")
                path = Path(entry.path)
                try:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                        continue
                    is_regular = entry.is_file(follow_symlinks=False)
                    if entry.is_symlink():
                        if not allow_symlink_files:
                            continue
                        target = path.resolve(strict=True)
                        is_regular = stat.S_ISREG(target.stat().st_mode)
                except (OSError, RuntimeError) as exc:
                    raise ValueError(f"DA3 identity tree entry is unstable: {path}") from exc
                if is_regular and path.suffix.lower() in suffixes:
                    matches.append(path)
                    if len(matches) > maximum_matches:
                        raise ValueError(f"DA3 identity tree has too many matching files: {root}")
    return tuple(sorted(matches, key=lambda path: path.relative_to(resolved_root).as_posix()))


def _safe_snapshot_file(snapshot_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe DA3 snapshot path: {relative_path!r}")
    candidate = snapshot_root / relative
    if not candidate.is_file():
        raise ValueError(f"Missing DA3 snapshot artifact: {relative_path}")
    return candidate


def _materialized_model_manifests(snapshot_root: Path) -> tuple[str, str, tuple[dict[str, Any], ...]]:
    """Hash the exact safetensor set and bounded output-affecting config."""

    snapshot_root = snapshot_root.resolve()
    index_path = snapshot_root / "model.safetensors.index.json"
    single_weight = snapshot_root / "model.safetensors"
    weight_records: list[dict[str, Any]] = []
    expected_weight_paths: set[str]

    if index_path.is_file():

        def reject_index_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            payload: dict[str, Any] = {}
            for key, value in pairs:
                if key in payload:
                    raise ValueError("DA3 safetensors index repeats a key")
                payload[key] = value
            return payload

        try:
            index_raw = _read_bounded_regular_file(index_path, maximum_bytes=_MAX_INDEX_BYTES)
            index_payload = json.loads(index_raw.decode("utf-8"), object_pairs_hook=reject_index_duplicates)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("DA3 safetensors index is not valid bounded JSON") from exc
        weight_map = index_payload.get("weight_map") if isinstance(index_payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("DA3 safetensors index has no non-empty weight_map")
        if len(weight_map) > _MAX_WEIGHT_MAP_ENTRIES:
            raise ValueError("DA3 safetensors index has too many weight-map entries")
        if any(not isinstance(value, str) or not value for value in weight_map.values()):
            raise ValueError("DA3 safetensors index has an invalid weight-map path")
        expected_weight_paths = set(weight_map.values())
        if any(not value.endswith(".safetensors") for value in expected_weight_paths):
            raise ValueError("DA3 safetensors index references an unsupported weight format")
        if len(expected_weight_paths) > _MAX_WEIGHT_FILES:
            raise ValueError("DA3 local snapshot has too many weight artifacts")
        weight_records.append(
            _file_record(
                index_path,
                logical_path=index_path.name,
                role="weight_index",
                maximum_bytes=_MAX_INDEX_BYTES,
            )
        )
    elif single_weight.is_file():
        expected_weight_paths = {single_weight.name}
    else:
        raise ValueError("DA3 local snapshot has no governed safetensors weight artifact")

    discovered_files = _bounded_matching_files(
        snapshot_root,
        suffixes={".safetensors", ".json", ".yaml", ".yml"},
        maximum_matches=_MAX_WEIGHT_FILES + _MAX_CONFIG_FILES + 1,
        maximum_entries=_MAX_MODEL_TREE_ENTRIES,
        maximum_directories=_MAX_MODEL_TREE_DIRECTORIES,
        allow_symlink_files=True,
    )
    discovered_weights = {
        path.relative_to(snapshot_root).as_posix() for path in discovered_files if path.suffix.lower() == ".safetensors"
    }
    if discovered_weights != expected_weight_paths:
        raise ValueError("DA3 local snapshot contains an ambiguous safetensors artifact set")
    if len(expected_weight_paths) > _MAX_WEIGHT_FILES:
        raise ValueError("DA3 local snapshot has too many weight artifacts")

    weight_bytes = 0
    for relative_path in sorted(expected_weight_paths):
        record = _file_record(
            _safe_snapshot_file(snapshot_root, relative_path),
            logical_path=relative_path,
            role="weight",
            maximum_bytes=_MAX_WEIGHT_BYTES - weight_bytes,
        )
        weight_bytes += int(record["size_bytes"])
        weight_records.append(record)
    weight_records.sort(key=lambda record: (str(record["path"]), str(record["role"])))

    config_paths = tuple(
        path for path in discovered_files if path != index_path and path.suffix.lower() in {".json", ".yaml", ".yml"}
    )
    if not (snapshot_root / "config.json").is_file():
        raise ValueError("DA3 local snapshot is missing config.json")
    if len(config_paths) > _MAX_CONFIG_FILES:
        raise ValueError("DA3 local snapshot has too many configuration files")

    config_records: list[dict[str, Any]] = []
    config_bytes = 0
    for path in config_paths:
        record = _file_record(
            path,
            logical_path=path.relative_to(snapshot_root).as_posix(),
            role="model_config",
            maximum_bytes=_MAX_CONFIG_BYTES - config_bytes,
        )
        config_bytes += int(record["size_bytes"])
        config_records.append(record)

    weights_payload = {
        "schema": DA3_WEIGHT_MANIFEST_SCHEMA,
        "files": weight_records,
    }
    model_records = sorted(
        [*weight_records, *config_records],
        key=lambda record: (str(record["path"]), str(record["role"])),
    )
    model_payload = {
        "schema": DA3_MODEL_MANIFEST_SCHEMA,
        "files": model_records,
    }
    return _sha256_payload(weights_payload), _sha256_payload(model_payload), tuple(model_records)


def _distribution_record(
    distribution_name: str,
    *,
    distribution: Any | None = None,
    verify_record_hashes: bool = True,
    import_module_names: set[str] | None = None,
) -> dict[str, Any]:
    distribution = distribution or importlib.metadata.distribution(distribution_name)
    canonical_name, version, _metadata_path, allowed_root = _distribution_metadata_identity(
        distribution,
        distribution_name=distribution_name,
    )
    if canonical_name != _canonical_distribution_name(distribution_name):
        raise ValueError(f"DA3 dependency metadata name disagrees with lookup: {distribution_name}")
    metadata_root = _metadata_path.parent

    try:
        record_raw = _read_bounded_regular_file(
            metadata_root / "RECORD",
            maximum_bytes=_MAX_DISTRIBUTION_RECORD_BYTES,
        )
        record = record_raw.decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"DA3 dependency has no bounded UTF-8 wheel RECORD: {distribution_name}") from exc
    direct_url_path = metadata_root / "direct_url.json"
    try:
        direct_url_raw = _read_bounded_regular_file(
            direct_url_path,
            maximum_bytes=_MAX_DISTRIBUTION_DIRECT_URL_BYTES,
        )
    except FileNotFoundError:
        direct_url_raw = b""
    except (OSError, ValueError) as exc:
        raise ValueError(f"DA3 dependency has invalid direct-url metadata: {distribution_name}") from exc

    _record_verification_stat(allowed_root)
    installed_files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    discovered_import_module_names: set[str] = set()
    total_bytes = 0
    row_count = 0
    for row_count, row in enumerate(csv.reader(io.StringIO(record, newline="")), start=1):
        if row_count > _MAX_DISTRIBUTION_FILES:
            raise ValueError(f"DA3 dependency RECORD is empty or unbounded: {distribution_name}")
        if len(row) != 3 or not row[0]:
            raise ValueError(f"DA3 dependency RECORD has an empty path: {distribution_name}")
        logical_path = row[0]
        if logical_path in seen_paths:
            raise ValueError(f"DA3 dependency RECORD repeats a path: {distribution_name}")
        seen_paths.add(logical_path)
        import_module_name = _record_top_level_import_name(logical_path)
        if (
            import_module_name is not None
            and import_module_names is not None
            and import_module_name not in import_module_names
            and import_module_name not in discovered_import_module_names
        ):
            if len(import_module_names) + len(discovered_import_module_names) >= _MAX_IMPORT_MODULES:
                raise ValueError(f"DA3 dependency import inventory is unbounded: {distribution_name}")
            discovered_import_module_names.add(import_module_name)
        candidate = Path(str(distribution.locate_file(logical_path)))
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(allowed_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"DA3 dependency RECORD path escapes its interpreter: {distribution_name}") from exc
        sha256, size_bytes = _hash_regular_file(
            resolved,
            maximum_bytes=_MAX_DISTRIBUTION_BYTES - total_bytes,
        )
        _record_verification_stat(resolved.parent)
        total_bytes += size_bytes
        declared_hash = row[1] if len(row) > 1 else ""
        declared_size = row[2]
        if declared_hash:
            algorithm, separator, encoded = declared_hash.partition("=")
            if algorithm != "sha256" or not separator:
                raise ValueError(f"DA3 dependency RECORD uses an unsupported digest: {distribution_name}")
            padded = encoded + "=" * (-len(encoded) % 4)
            try:
                expected_sha256 = base64.urlsafe_b64decode(padded).hex()
            except (ValueError, TypeError, binascii.Error) as exc:
                raise ValueError(f"DA3 dependency RECORD has an invalid digest: {distribution_name}") from exc
            if verify_record_hashes and expected_sha256 != sha256:
                raise ValueError(f"DA3 dependency file differs from RECORD: {distribution_name}:{logical_path}")
            if not declared_size.isdigit():
                raise ValueError(f"DA3 dependency RECORD has an invalid size: {distribution_name}:{logical_path}")
            if verify_record_hashes and int(declared_size) != size_bytes:
                raise ValueError(f"DA3 dependency size differs from RECORD: {distribution_name}:{logical_path}")
        elif declared_size:
            raise ValueError(f"DA3 dependency RECORD has size without digest: {distribution_name}:{logical_path}")
        installed_files.append(
            {
                "path": logical_path,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    if row_count == 0:
        raise ValueError(f"DA3 dependency RECORD is empty or unbounded: {distribution_name}")
    if import_module_names is not None:
        import_module_names.update(discovered_import_module_names)
    installed_files.sort(key=lambda value: str(value["path"]))
    return {
        "name": canonical_name,
        "version": version,
        "direct_url_sha256": hashlib.sha256(direct_url_raw).hexdigest(),
        "record_sha256": hashlib.sha256(record_raw).hexdigest(),
        "installed_files_sha256": _sha256_payload(
            {
                "schema": "tp.da3.installed-distribution.v1",
                "files": installed_files,
            }
        ),
    }


def _record_top_level_import_name(logical_path: str) -> str | None:
    """Project one importable top-level name from a wheel RECORD path."""

    parts = logical_path.split("/")
    if not parts or any(part in {"", ".", "..", "__pycache__"} for part in parts):
        return None
    leaf = parts[-1]
    if not leaf.endswith((".py", ".pyc", ".so", ".pyd", ".dll", ".dylib")):
        return None
    candidate = parts[0]
    if len(parts) == 1:
        candidate = candidate.split(".", 1)[0]
    return candidate if candidate.isidentifier() else None


def _installed_distribution_index() -> dict[str, tuple[Any, str]]:
    """Return a bounded, duplicate-free installed distribution inventory."""

    try:
        distributions = importlib.metadata.distributions()
    except (OSError, ValueError) as exc:
        raise ValueError("DA3 installed dependency inventory is unavailable") from exc
    installed: dict[str, tuple[Any, str]] = {}
    interpreter_root = Path(sys.prefix).resolve()
    for ordinal, distribution in enumerate(distributions, start=1):
        if ordinal > _MAX_LOCK_DISTRIBUTIONS:
            raise ValueError("DA3 installed dependency inventory is unbounded")
        metadata_root_value = getattr(distribution, "_path", None)
        if metadata_root_value is None:
            raise ValueError("DA3 installed dependency has no filesystem metadata root")
        try:
            Path(metadata_root_value).resolve(strict=True).relative_to(interpreter_root)
        except ValueError:
            # PYTHONPATH/editable source trees can expose project metadata
            # outside the venv. Relevant source bytes and import precedence
            # are bound separately; they are not interpreter distributions.
            continue
        except (OSError, RuntimeError, TypeError) as exc:
            raise ValueError("DA3 installed dependency metadata is unstable") from exc
        canonical_name, observed_version, _metadata_path, _allowed_root = _distribution_metadata_identity(
            distribution,
            distribution_name=f"installed-{ordinal}",
        )
        if canonical_name in installed:
            raise ValueError(f"DA3 installed dependency inventory repeats {canonical_name!r}")
        installed[canonical_name] = (distribution, observed_version)
    return installed


def _validated_dependency_inventory(
    dependency_inventory: object,
) -> tuple[dict[str, Any], ...]:
    """Validate the closed test seam used in place of live RECORD hashing."""

    if not isinstance(dependency_inventory, (list, tuple)):
        raise ValueError("Injected DA3 dependency inventory is not a sequence")
    if len(dependency_inventory) > _MAX_LOCK_DISTRIBUTIONS:
        raise ValueError("Injected DA3 dependency inventory is unbounded")
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw_record in dependency_inventory:
        if not isinstance(raw_record, Mapping) or set(raw_record) != _DISTRIBUTION_RECORD_KEYS:
            raise ValueError("Injected DA3 dependency inventory has an unknown record shape")
        record = dict(raw_record)
        name = record.get("name")
        version = record.get("version")
        if not isinstance(name, str) or not name or name != _canonical_distribution_name(name):
            raise ValueError("Injected DA3 dependency inventory has a noncanonical name")
        if name in names:
            raise ValueError("Injected DA3 dependency inventory repeats a distribution")
        if not isinstance(version, str) or not version:
            raise ValueError("Injected DA3 dependency inventory has no exact version")
        if any(
            not _is_sha256(record.get(field_name))
            for field_name in ("direct_url_sha256", "record_sha256", "installed_files_sha256")
        ):
            raise ValueError("Injected DA3 dependency inventory has an invalid materialized digest")
        names.add(name)
        records.append(record)
    records.sort(key=lambda record: str(record["name"]))
    return tuple(records)


def _dependency_inventory(
    expected_distributions: Mapping[str, str],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], tuple[str, ...]]:
    """Hash the exact installed closure and reject every extra or mismatch."""

    records: list[dict[str, Any]] = []
    reasons: list[str] = []
    import_module_names: set[str] = set()
    try:
        installed = _installed_distribution_index()
    except (OSError, TypeError, ValueError):
        return (), ("dependency_inventory_invalid",), ()

    expected_names = set(expected_distributions)
    installed_names = set(installed)
    for distribution_name in sorted(expected_names - installed_names):
        reasons.append(f"dependency_missing:{distribution_name}")
    for distribution_name in sorted(installed_names - expected_names):
        reasons.append(f"dependency_extra:{distribution_name}")

    for distribution_name in sorted(expected_names & installed_names):
        distribution, observed_version = installed[distribution_name]
        expected_version = expected_distributions[distribution_name]
        if observed_version != expected_version:
            reasons.append(f"dependency_version_mismatch:{distribution_name}:{expected_version}:{observed_version}")
        try:
            records.append(
                _distribution_record(
                    distribution_name,
                    distribution=distribution,
                    import_module_names=import_module_names,
                )
            )
        except (OSError, ValueError):
            reasons.append(f"dependency_materialization_invalid:{distribution_name}")
    records.sort(key=lambda record: str(record["name"]))
    return tuple(records), tuple(reasons), tuple(sorted(import_module_names))


def _dependency_closure_reasons(
    dependency_records: Sequence[Mapping[str, Any]],
    expected_distributions: Mapping[str, str],
) -> tuple[str, ...]:
    observed = {str(record["name"]): str(record["version"]) for record in dependency_records}
    reasons: list[str] = []
    for distribution_name in sorted(set(expected_distributions) - set(observed)):
        reasons.append(f"dependency_missing:{distribution_name}")
    for distribution_name in sorted(set(observed) - set(expected_distributions)):
        reasons.append(f"dependency_extra:{distribution_name}")
    for distribution_name in sorted(set(observed) & set(expected_distributions)):
        expected_version = expected_distributions[distribution_name]
        observed_version = observed[distribution_name]
        if observed_version != expected_version:
            reasons.append(f"dependency_version_mismatch:{distribution_name}:{expected_version}:{observed_version}")
    return tuple(reasons)


def _interpreter_payload() -> tuple[dict[str, Any], tuple[str, ...]]:
    reasons: list[str] = []
    executable = Path(sys.executable).resolve()
    try:
        executable_sha256, executable_size = _hash_regular_file(
            executable,
            maximum_bytes=_MAX_INTERPRETER_BYTES,
        )
    except (OSError, ValueError):
        executable_sha256 = None
        executable_size = None
        reasons.append("interpreter_executable_unhashable")
    return (
        {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
            "soabi": sysconfig.get_config_var("SOABI"),
            "executable_sha256": executable_sha256,
            "executable_size_bytes": executable_size,
        },
        tuple(reasons),
    )


def _hardware_payload() -> dict[str, str]:
    """Return a bounded local hardware identity for CPU/MPS cache authority."""

    payload: dict[str, str] = {}
    if platform.system() == "Darwin":
        for field_name, sysctl_name in (
            ("hardware_model", "hw.model"),
            ("cpu_brand", "machdep.cpu.brand_string"),
        ):
            try:
                result = subprocess.run(
                    ["/usr/sbin/sysctl", "-n", sysctl_name],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=2,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            value = result.stdout.strip()
            if result.returncode == 0 and value and len(value.encode("utf-8")) <= 1024:
                payload[field_name] = value
    else:
        processor = platform.processor().strip()
        machine = platform.machine().strip()
        if processor:
            payload["processor"] = processor[:1024]
        if machine:
            payload["machine"] = machine[:1024]
    return payload


def _platform_payload() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "hardware": _hardware_payload(),
    }


def _accelerator_payload(
    *, requested_device: str, actual_device: str, executed_backend: str
) -> tuple[dict[str, Any], tuple[str, ...]]:
    reasons: list[str] = []
    payload: dict[str, Any] = {
        "requested_device": requested_device,
        "actual_device": actual_device,
        "executed_backend": executed_backend,
    }
    try:
        import torch
    except ImportError:
        reasons.append("dependency_missing:torch")
        return payload, tuple(reasons)

    payload["torch_version"] = str(torch.__version__)
    normalized_actual = actual_device.lower()
    if normalized_actual == "cuda":
        available = bool(torch.cuda.is_available())
        payload.update(
            {
                "available": available,
                "cuda_runtime": getattr(torch.version, "cuda", None),
                "cudnn_version": torch.backends.cudnn.version() if available else None,
                "device_name": torch.cuda.get_device_name(0) if available else None,
                "compute_capability": list(torch.cuda.get_device_capability(0)) if available else None,
            }
        )
        if not available:
            reasons.append("accelerator_unavailable:cuda")
    elif normalized_actual == "mps":
        available = bool(torch.backends.mps.is_available())
        hardware = _hardware_payload()
        payload.update(
            {
                "available": available,
                "built": bool(torch.backends.mps.is_built()),
                "hardware": hardware,
            }
        )
        if not available:
            reasons.append("accelerator_unavailable:mps")
        if not hardware:
            reasons.append("hardware_identity_unavailable")
    elif normalized_actual == "coreml":
        reasons.append("coreml_materialization_not_supported")
    elif normalized_actual == "cpu":
        hardware = _hardware_payload()
        payload["available"] = True
        payload["processor"] = platform.processor()
        payload["hardware"] = hardware
        if not hardware:
            reasons.append("hardware_identity_unavailable")
    else:
        reasons.append(f"accelerator_unknown:{normalized_actual}")

    if requested_device.lower() != normalized_actual:
        reasons.append(f"device_mismatch:{requested_device.lower()}:{normalized_actual}")
    return payload, tuple(reasons)


def _module_source_files(
    module_name: str,
    *,
    maximum_files: int = _MAX_SOURCE_FILES,
) -> dict[str, Path]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ValueError(f"DA3 runtime source module is unavailable: {module_name}")
    if spec.submodule_search_locations:
        roots = [Path(value) for value in spec.submodule_search_locations]
        if not roots or len(roots) > _MAX_MODULE_SEARCH_ROOTS:
            raise ValueError(f"DA3 runtime source module has unbounded search roots: {module_name}")
        records: dict[str, Path] = {}
        suffixes = (
            {".py", ".yaml", ".yml", ".json"} if module_name in {"depth_anything_3", "transformation_portal"} else {".py"}
        )
        for root in roots:
            remaining = maximum_files - len(records)
            if remaining <= 0:
                raise ValueError(f"DA3 runtime source module has too many files: {module_name}")
            for path in _bounded_matching_files(
                root,
                suffixes=suffixes,
                maximum_matches=remaining,
                maximum_entries=_MAX_SOURCE_TREE_ENTRIES,
                maximum_directories=_MAX_SOURCE_TREE_DIRECTORIES,
                allow_symlink_files=True,
            ):
                logical_path = f"{module_name}/{path.relative_to(root.resolve(strict=True)).as_posix()}"
                previous = records.get(logical_path)
                if previous is not None and previous.resolve(strict=True) != path.resolve(strict=True):
                    raise ValueError(f"DA3 runtime source module has ambiguous files: {module_name}")
                records[logical_path] = path
        return records
    if not spec.origin or spec.origin in {"built-in", "frozen"}:
        raise ValueError(f"DA3 runtime source module has no hashable origin: {module_name}")
    if maximum_files < 1:
        raise ValueError(f"DA3 runtime source module has too many files: {module_name}")
    return {module_name: Path(spec.origin)}


def _source_file_mapping() -> tuple[dict[str, Path], tuple[dict[str, Any], ...]]:
    records: dict[str, Path] = {}
    for module_name in (*_TP_SOURCE_MODULES, "depth_anything_3"):
        records.update(_module_source_files(module_name, maximum_files=_MAX_SOURCE_FILES - len(records)))
    records.update(_runtime_configuration_file_mapping())
    optional_records: list[dict[str, Any]] = []
    for module_name in _TP_OPTIONAL_SOURCE_MODULES:
        present = importlib.util.find_spec(module_name) is not None
        optional_records.append({"name": module_name, "present": present})
        if present:
            records.update(_module_source_files(module_name, maximum_files=_MAX_SOURCE_FILES - len(records)))
    return records, tuple(optional_records)


def _runtime_configuration_file_mapping() -> dict[str, Path]:
    """Resolve external policy bytes consumed while revalidating a carried plan."""

    from ...core.security.model_lock import model_lock_manifest_path

    manifest_path = model_lock_manifest_path()
    try:
        resolved = manifest_path.resolve(strict=True)
        observed = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise ValueError("DA3 runtime model-lock policy is unavailable") from exc
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError("DA3 runtime model-lock policy is not a regular file")
    return {"runtime_config/model_lock_manifest.yaml": resolved}


def _optional_source_module_records() -> tuple[dict[str, Any], ...]:
    return tuple(
        {"name": module_name, "present": importlib.util.find_spec(module_name) is not None}
        for module_name in _TP_OPTIONAL_SOURCE_MODULES
    )


def _source_identity(
    source_files: Mapping[str, Path],
    *,
    source_revision: str | None,
    optional_source_modules: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[dict[str, Any], ...]]:
    if len(source_files) > _MAX_SOURCE_FILES:
        raise ValueError("DA3 runtime source has too many files")
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for logical_path, path in sorted(source_files.items()):
        record = _file_record(
            Path(path),
            logical_path=str(logical_path),
            role="runtime_source",
            maximum_bytes=_MAX_SOURCE_BYTES - total_bytes,
        )
        total_bytes += int(record["size_bytes"])
        records.append(record)
    payload = {
        "schema": DA3_SOURCE_IDENTITY_SCHEMA,
        "files": records,
        "optional_source_modules": [dict(value) for value in optional_source_modules],
        "source_revision": source_revision,
    }
    return _sha256_payload(payload), tuple(records)


def _normalized_import_path(raw_path: str) -> dict[str, Any]:
    if len(raw_path.encode("utf-8")) > _MAX_IMPORT_PATH_BYTES:
        raise ValueError("DA3 import search path is oversized")
    candidate = Path(raw_path) if raw_path else Path.cwd()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(os.fspath(candidate)))
    try:
        resolved = absolute.resolve(strict=True)
        observed = resolved.stat()
    except FileNotFoundError:
        existing = absolute.parent
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        resolved_parent = existing.resolve(strict=True)
        _record_verification_stat(resolved_parent)
        return {"path": str(absolute), "kind": "missing"}
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"DA3 import search path is unstable: {raw_path!r}") from exc
    if stat.S_ISDIR(observed.st_mode):
        kind = "directory"
    elif stat.S_ISREG(observed.st_mode):
        kind = "file"
    else:
        raise ValueError(f"DA3 import search path has unsupported type: {raw_path!r}")
    _record_verification_stat(resolved, observed)
    return {"path": str(resolved), "kind": kind}


def _module_resolution_record(
    module_name: str,
    *,
    origin_hashes: dict[str, tuple[str, int]],
    maximum_origin_bytes: int,
) -> tuple[dict[str, Any], int]:
    try:
        spec = importlib.util.find_spec(module_name)
    except (AttributeError, ImportError, ValueError) as exc:
        raise ValueError(f"DA3 import resolution failed for {module_name}") from exc
    if spec is None:
        return (
            {
                "name": module_name,
                "present": False,
                "origin": None,
                "origin_sha256": None,
                "origin_size_bytes": None,
                "search_locations": [],
            },
            0,
        )
    origin: str | None
    origin_sha256: str | None = None
    origin_size_bytes: int | None = None
    consumed_origin_bytes = 0
    raw_origin = spec.origin
    if raw_origin is None or raw_origin in {"built-in", "frozen"}:
        origin = raw_origin
    else:
        try:
            origin_path = Path(raw_origin).resolve(strict=True)
            origin_stat = origin_path.stat()
        except (OSError, RuntimeError, TypeError) as exc:
            raise ValueError(f"DA3 import origin is unstable for {module_name}") from exc
        if not stat.S_ISREG(origin_stat.st_mode):
            raise ValueError(f"DA3 import origin is not a regular file for {module_name}")
        origin = str(origin_path)
        cached_origin = origin_hashes.get(origin)
        if cached_origin is None:
            origin_sha256, origin_size_bytes = _hash_regular_file(
                origin_path,
                maximum_bytes=maximum_origin_bytes,
            )
            origin_hashes[origin] = (origin_sha256, origin_size_bytes)
            consumed_origin_bytes = origin_size_bytes
        else:
            origin_sha256, origin_size_bytes = cached_origin
        _record_verification_stat(origin_path.parent)
    raw_locations: list[str] = []
    for ordinal, raw_location in enumerate(spec.submodule_search_locations or (), start=1):
        if ordinal > _MAX_MODULE_SEARCH_ROOTS:
            raise ValueError(f"DA3 import resolution has too many roots for {module_name}")
        raw_locations.append(raw_location)
    locations: list[str] = []
    for raw_location in raw_locations:
        if not isinstance(raw_location, str) or len(raw_location.encode("utf-8")) > _MAX_IMPORT_PATH_BYTES:
            raise ValueError(f"DA3 import resolution has an invalid root for {module_name}")
        try:
            location = Path(raw_location).resolve(strict=True)
            location_stat = location.stat()
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"DA3 import resolution has an unstable root for {module_name}") from exc
        if not stat.S_ISDIR(location_stat.st_mode):
            raise ValueError(f"DA3 import resolution root is not a directory for {module_name}")
        _record_verification_stat(location, location_stat)
        locations.append(str(location))
    return (
        {
            "name": module_name,
            "present": True,
            "origin": origin,
            "origin_sha256": origin_sha256,
            "origin_size_bytes": origin_size_bytes,
            "search_locations": locations,
        },
        consumed_origin_bytes,
    )


def _import_environment_payload(module_names: Sequence[str]) -> dict[str, Any]:
    """Project ordered import roots and exact module origins without imports."""

    if len(sys.path) > _MAX_IMPORT_SEARCH_PATHS:
        raise ValueError("DA3 import search path inventory is unbounded")
    if len(module_names) > _MAX_IMPORT_MODULES:
        raise ValueError("DA3 import probe has too many modules")
    if any(
        not isinstance(module_name, str)
        or not module_name.isidentifier()
        or len(module_name.encode("utf-8")) > _MAX_IMPORT_PATH_BYTES
        for module_name in module_names
    ):
        raise ValueError("DA3 import probe has an invalid module name")
    if tuple(module_names) != tuple(sorted(set(module_names))):
        raise ValueError("DA3 import probe is not canonical")
    search_paths = [
        {"ordinal": ordinal, **_normalized_import_path(raw_path)}
        for ordinal, raw_path in enumerate(sys.path)
        if isinstance(raw_path, str)
    ]
    if len(search_paths) != len(sys.path):
        raise ValueError("DA3 import search path contains a non-string entry")
    configuration_files: dict[str, dict[str, Any]] = {}
    scanned_entries = 0
    total_bytes = 0
    for search_path in search_paths:
        if search_path["kind"] != "directory":
            continue
        try:
            iterator = os.scandir(str(search_path["path"]))
        except OSError as exc:
            raise ValueError("DA3 import search path cannot be inventoried") from exc
        with iterator:
            for entry in iterator:
                scanned_entries += 1
                if scanned_entries > _MAX_IMPORT_CONFIGURATION_DIRECTORY_ENTRIES:
                    raise ValueError("DA3 import configuration inventory is unbounded")
                if not (entry.name.endswith((".pth", ".egg-link")) or entry.name in {"sitecustomize.py", "usercustomize.py"}):
                    continue
                if len(configuration_files) >= _MAX_IMPORT_CONFIGURATION_FILES:
                    raise ValueError("DA3 import configuration inventory has too many files")
                try:
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        raise ValueError("DA3 import configuration is not a regular file")
                    resolved = Path(entry.path).resolve(strict=True)
                    sha256, size_bytes = _hash_regular_file(
                        resolved,
                        maximum_bytes=_MAX_IMPORT_CONFIGURATION_BYTES - total_bytes,
                    )
                except OSError as exc:
                    raise ValueError("DA3 import configuration is unstable") from exc
                total_bytes += size_bytes
                configuration_files[str(resolved)] = {
                    "path": str(resolved),
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                }
    origin_hashes: dict[str, tuple[str, int]] = {}
    origin_bytes = 0
    modules: list[dict[str, Any]] = []
    for module_name in module_names:
        record, consumed_bytes = _module_resolution_record(
            module_name,
            origin_hashes=origin_hashes,
            maximum_origin_bytes=_MAX_IMPORT_ORIGIN_BYTES - origin_bytes,
        )
        origin_bytes += consumed_bytes
        modules.append(record)
    return {
        "schema": DA3_IMPORT_ENVIRONMENT_SCHEMA,
        "search_paths": search_paths,
        "path_configuration_files": sorted(configuration_files.values(), key=lambda value: str(value["path"])),
        "runtime_environment": {name: os.environ.get(name) for name in _WORKER_RUNTIME_ENVIRONMENT_NAMES},
        "modules": modules,
    }


def _parent_import_environment_payload() -> dict[str, Any]:
    module_names = tuple(
        sorted(
            {
                *_PARENT_OUTPUT_SOURCE_MODULES,
                *_PARENT_OUTPUT_OPTIONAL_SOURCE_MODULES,
                "PIL",
                "cv2",
                "numpy",
                "scipy",
            }
        )
    )
    return _import_environment_payload(module_names)


def _worker_import_environment_payload(additional_module_names: Iterable[str] = ()) -> dict[str, Any]:
    """Capture the worker's ordered import roots and resolved runtime modules."""

    module_name_set = set(_WORKER_IMPORT_MODULES)
    for ordinal, module_name in enumerate(additional_module_names, start=1):
        if ordinal > _MAX_IMPORT_MODULES:
            raise ValueError("DA3 worker import probe input is unbounded")
        if not isinstance(module_name, str):
            raise ValueError("DA3 worker import probe has an invalid module name")
        module_name_set.add(module_name)
        if len(module_name_set) > _MAX_IMPORT_MODULES:
            raise ValueError("DA3 worker import probe has too many modules")
    module_names = tuple(sorted(module_name_set))
    return _import_environment_payload(module_names)


def _worker_import_environment_reasons(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Reject dependency imports that resolve outside the governed interpreter."""

    raw_modules = payload.get("modules")
    if not isinstance(raw_modules, list):
        return ("import_environment_invalid",)
    modules = {
        str(record.get("name")): record
        for record in raw_modules
        if isinstance(record, Mapping) and isinstance(record.get("name"), str)
    }
    try:
        interpreter_root = Path(sys.prefix).resolve(strict=True)
    except (OSError, RuntimeError):
        return ("import_environment_invalid",)
    reasons: list[str] = []
    for module_name, record in sorted(modules.items()):
        if module_name in _WORKER_EXTERNAL_SOURCE_IMPORT_MODULES:
            continue
        if record.get("present") is not True:
            reasons.append(f"dependency_import_missing:{module_name}")
            continue
        raw_origin = record.get("origin")
        raw_locations = record.get("search_locations")
        paths: list[str] = []
        if isinstance(raw_origin, str) and raw_origin not in {"built-in", "frozen"}:
            paths.append(raw_origin)
        if isinstance(raw_locations, list):
            paths.extend(value for value in raw_locations if isinstance(value, str))
        if not paths:
            reasons.append(f"dependency_import_unbound:{module_name}")
            continue
        for raw_path in paths:
            try:
                Path(raw_path).resolve(strict=True).relative_to(interpreter_root)
            except (OSError, RuntimeError, ValueError):
                reasons.append(f"dependency_import_outside_runtime:{module_name}")
                break
    return tuple(reasons)


def _git_source_revision() -> str | None:
    spec = importlib.util.find_spec("depth_anything_3")
    if spec is None:
        return None
    origin = Path(spec.origin).resolve() if spec.origin else None
    if origin is None:
        return None
    for candidate in (origin.parent, *origin.parents):
        if (candidate / ".git").exists():
            try:
                repository_path = candidate.resolve(strict=True)
                result = subprocess.run(
                    ["git", "-C", str(repository_path), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
            except (OSError, RuntimeError, subprocess.SubprocessError):
                return None
            revision = result.stdout.strip().lower()
            if (
                result.returncode != 0
                or len(revision) != 40
                or any(character not in "0123456789abcdef" for character in revision)
            ):
                return None
            probe = _SOURCE_REVISION_PROBE.get()
            if probe is not None:
                probe.update(
                    {
                        "repository_path": str(repository_path),
                        "revision": revision,
                    }
                )
            return revision
    return None


def _verify_source_revision_probe(payload: object) -> bool:
    """Re-resolve the governed DA3 checkout revision for every token check."""

    if not isinstance(payload, Mapping) or set(payload) != {"repository_path", "revision"}:
        return False
    raw_path = payload.get("repository_path")
    revision = payload.get("revision")
    if (
        not isinstance(raw_path, str)
        or not Path(raw_path).is_absolute()
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        return False
    try:
        repository_path = Path(raw_path).resolve(strict=True)
        if str(repository_path) != raw_path or not (repository_path / ".git").exists():
            return False
        result = subprocess.run(
            ["git", "-C", raw_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip().lower() == revision


def _default_governance_path() -> Path | None:
    repo_root = find_repo_root(Path(__file__))
    return None if repo_root is None else repo_root / _DEFAULT_GOVERNANCE_PATH


def _load_governance_contract(path: Path | None) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    if path is None or not path.is_file():
        return None, ("governance_contract_missing",)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("DA3 governance contract repeats a key")
            result[key] = value
        return result

    try:
        raw = _read_bounded_regular_file(path, maximum_bytes=_MAX_GOVERNANCE_BYTES)
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None, ("governance_contract_invalid",)
    expected_keys = {
        "schema",
        "cache_authority_enabled",
        "source_repository",
        "source_revision",
        "dependency_closure_complete",
        "dependency_lock_path",
        "dependency_lock_sha256",
        "governed_additional_distributions",
        "runtime_marker_filename",
        "non_authorizing_reason",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        return None, ("governance_contract_invalid",)
    if payload.get("schema") != DA3_RUNTIME_GOVERNANCE_SCHEMA:
        return None, ("governance_contract_schema_mismatch",)
    if type(payload.get("cache_authority_enabled")) is not bool:
        return None, ("governance_contract_invalid",)
    if type(payload.get("dependency_closure_complete")) is not bool:
        return None, ("governance_contract_invalid",)
    if not isinstance(payload.get("source_repository"), str) or not payload["source_repository"]:
        return None, ("governance_contract_invalid",)
    source_revision = payload.get("source_revision")
    if (
        not isinstance(source_revision, str)
        or len(source_revision) != 40
        or source_revision != source_revision.lower()
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        return None, ("governance_contract_invalid",)
    additional = payload.get("governed_additional_distributions")
    if not isinstance(additional, dict):
        return None, ("governance_contract_invalid",)
    for name, version in additional.items():
        if (
            not isinstance(name, str)
            or not name
            or name != _canonical_distribution_name(name)
            or not isinstance(version, str)
            or not version
        ):
            return None, ("governance_contract_invalid",)
    marker_filename = payload.get("runtime_marker_filename")
    marker_path = Path(marker_filename) if isinstance(marker_filename, str) else None
    if (
        marker_path is None
        or not marker_filename
        or marker_path.is_absolute()
        or len(marker_path.parts) != 1
        or marker_path.as_posix() != marker_filename
    ):
        return None, ("governance_contract_invalid",)
    if payload["cache_authority_enabled"]:
        if payload["dependency_closure_complete"] is not True:
            return None, ("governance_contract_invalid",)
        if not isinstance(payload.get("dependency_lock_path"), str) or not payload["dependency_lock_path"]:
            return None, ("governance_contract_invalid",)
        if not _is_sha256(payload.get("dependency_lock_sha256")):
            return None, ("governance_contract_invalid",)
    elif payload.get("dependency_lock_path") is not None or payload.get("dependency_lock_sha256") is not None:
        return None, ("governance_contract_invalid",)
    reason = payload.get("non_authorizing_reason")
    if (
        not isinstance(reason, str)
        or (not payload["cache_authority_enabled"] and not reason)
        or (payload["cache_authority_enabled"] and reason)
    ):
        return None, ("governance_contract_invalid",)
    return payload, ()


def _parse_exact_dependency_lock(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("DA3 dependency lock is not UTF-8") from exc
    expected: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)", line)
        if match is None:
            raise ValueError("DA3 dependency lock contains a non-exact requirement")
        name = _canonical_distribution_name(match.group(1))
        version = match.group(2)
        if name in expected:
            raise ValueError("DA3 dependency lock repeats a distribution")
        expected[name] = version
        if len(expected) > _MAX_LOCK_DISTRIBUTIONS:
            raise ValueError("DA3 dependency lock contains too many distributions")
    if not expected:
        raise ValueError("DA3 dependency lock contains no exact distributions")
    return expected


def _verified_dependency_lock(
    governance: Mapping[str, Any],
    *,
    governance_contract_path: Path,
) -> tuple[str | None, dict[str, str], tuple[str, ...]]:
    """Hash the checked-in exact lock and bind the contract to its bytes."""

    if governance.get("cache_authority_enabled") is not True:
        return None, {}, ("governance_cache_authority_disabled", "dependency_lock_unavailable")
    raw_path = governance.get("dependency_lock_path")
    if not isinstance(raw_path, str):  # pragma: no cover - loader enforces this
        return None, {}, ("dependency_lock_path_invalid",)
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != raw_path:
        return None, {}, ("dependency_lock_path_invalid",)
    repository_root = find_repo_root(governance_contract_path)
    allowed_root = (repository_root or governance_contract_path.parent).resolve()
    try:
        lock_path = (allowed_root / relative).resolve(strict=True)
        lock_path.relative_to(allowed_root)
        raw = _read_bounded_regular_file(lock_path, maximum_bytes=_MAX_LOCK_BYTES)
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        expected_distributions = _parse_exact_dependency_lock(raw)
    except (OSError, RuntimeError, ValueError):
        return None, {}, ("dependency_lock_materialization_invalid",)
    if actual_sha256 != governance.get("dependency_lock_sha256"):
        return None, {}, ("dependency_lock_mismatch",)
    additional = dict(governance.get("governed_additional_distributions", {}))
    overlap = set(expected_distributions) & set(additional)
    if overlap:
        return None, {}, ("dependency_governance_overlap",)
    return actual_sha256, {**expected_distributions, **additional}, ()


def _verified_runtime_authority_marker(
    governance: Mapping[str, Any],
    *,
    dependency_lock_sha256: str,
    marker_path: Path,
) -> tuple[str | None, tuple[str, ...]]:
    try:
        raw = _read_bounded_regular_file(marker_path, maximum_bytes=_MAX_RUNTIME_MARKER_BYTES)

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            payload: dict[str, Any] = {}
            for key, value in pairs:
                if key in payload:
                    raise ValueError("DA3 runtime authority marker repeats a key")
                payload[key] = value
            return payload

        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None, ("runtime_authority_marker_invalid",)
    expected = {
        "schema": _RUNTIME_AUTHORITY_SCHEMA,
        "cache_authority_enabled": True,
        "profile": "baseline",
        "python_version": "3.11",
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "dependency_lock_sha256": dependency_lock_sha256,
        "source_revision": governance["source_revision"],
    }
    if payload != expected or canonicalize_json(payload) != raw:
        return None, ("runtime_authority_marker_mismatch",)
    return hashlib.sha256(raw).hexdigest(), ()


def da3_cache_governance_identity(path: Path | None = None) -> tuple[str, str] | None:
    """Return parent-verifiable governance and lock digests when authorized."""

    contract_path = path if path is not None else _default_governance_path()
    governance, reasons = _load_governance_contract(contract_path)
    if governance is None or reasons or contract_path is None:
        return None
    dependency_lock_sha256, _expected_distributions, lock_reasons = _verified_dependency_lock(
        governance,
        governance_contract_path=contract_path,
    )
    if dependency_lock_sha256 is None or lock_reasons:
        return None
    return _sha256_payload(governance), dependency_lock_sha256


@dataclass(frozen=True)
class DA3CacheRuntimeGovernanceIdentity:
    """Parent-anchored governance facts for one configured DA3 venv."""

    governance_contract_sha256: str
    dependency_lock_sha256: str
    runtime_authority_sha256: str
    source_revision: str
    runtime_baseline: Mapping[str, Any]


def _runtime_python_baseline(runtime_python: Path) -> dict[str, Any]:
    """Independently probe the configured interpreter without repo/site imports."""

    resolved_python = runtime_python.resolve(strict=True)
    executable_sha256, executable_size = _hash_regular_file(
        resolved_python,
        maximum_bytes=_MAX_INTERPRETER_BYTES,
    )
    probe = (
        "import json,platform,sys;"
        "print(json.dumps({"
        "'implementation':platform.python_implementation(),"
        "'python_version':platform.python_version(),"
        "'releaselevel':sys.version_info.releaselevel,"
        "'system':platform.system(),"
        "'release':platform.release(),"
        "'platform_version':platform.version(),"
        "'machine':platform.machine()"
        "},sort_keys=True,separators=(',',':')))"
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PYTHON") and key.upper() not in {"VIRTUAL_ENV", "__PYVENV_LAUNCHER__"}
    }
    environment.update(
        {
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        }
    )
    try:
        completed = subprocess.run(
            [str(runtime_python), "-I", "-S", "-c", probe],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("DA3 runtime Python baseline probe failed") from exc
    raw = completed.stdout.encode("utf-8")
    if completed.returncode != 0 or not raw or len(raw) > _MAX_RUNTIME_BASELINE_BYTES:
        raise ValueError("DA3 runtime Python baseline probe returned invalid output")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("DA3 runtime Python baseline probe returned invalid JSON") from exc
    expected_keys = {
        "implementation",
        "python_version",
        "releaselevel",
        "system",
        "release",
        "platform_version",
        "machine",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("DA3 runtime Python baseline probe returned an unknown shape")
    if (
        payload["implementation"] != "CPython"
        or not isinstance(payload["python_version"], str)
        or re.fullmatch(r"3\.11\.\d+", payload["python_version"]) is None
        or payload["releaselevel"] != "final"
        or payload["system"] != "Darwin"
        or str(payload["machine"]).lower() not in {"arm64", "aarch64"}
    ):
        raise ValueError("DA3 runtime Python baseline is not governed Darwin arm64 CPython 3.11 final")
    payload["executable_sha256"] = executable_sha256
    payload["executable_size_bytes"] = executable_size
    return payload


def da3_cache_runtime_governance_identity(
    runtime_python: str | Path,
    path: Path | None = None,
) -> DA3CacheRuntimeGovernanceIdentity | None:
    """Verify the exact contract, lock, and marker outside the worker report."""

    contract_path = path if path is not None else _default_governance_path()
    governance, reasons = _load_governance_contract(contract_path)
    if governance is None or reasons or contract_path is None:
        return None
    dependency_lock_sha256, _expected_distributions, lock_reasons = _verified_dependency_lock(
        governance,
        governance_contract_path=contract_path,
    )
    if dependency_lock_sha256 is None or lock_reasons:
        return None
    try:
        python_path = Path(runtime_python).expanduser()
        if not python_path.is_absolute() or not python_path.exists():
            return None
        runtime_baseline = _runtime_python_baseline(python_path)
        marker_path = python_path.parent.parent / str(governance["runtime_marker_filename"])
        runtime_authority_sha256, marker_reasons = _verified_runtime_authority_marker(
            governance,
            dependency_lock_sha256=dependency_lock_sha256,
            marker_path=marker_path,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if runtime_authority_sha256 is None or marker_reasons:
        return None
    return DA3CacheRuntimeGovernanceIdentity(
        governance_contract_sha256=_sha256_payload(governance),
        dependency_lock_sha256=dependency_lock_sha256,
        runtime_authority_sha256=runtime_authority_sha256,
        source_revision=str(governance["source_revision"]),
        runtime_baseline=runtime_baseline,
    )


def da3_cache_governance_enabled(path: Path | None = None) -> bool:
    """Return true only for an enabled contract whose exact lock is present."""

    return da3_cache_governance_identity(path) is not None


def _local_snapshot(repo_id: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
        )
    )


def load_da3_worker_runtime_handshake(
    path: Path,
    *,
    maximum_bytes: int = _MAX_HANDSHAKE_BYTES,
) -> dict[str, Any]:
    """Parse one bounded, duplicate-free, canonical worker handshake."""

    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    try:
        with path.open("rb") as handle:
            raw = handle.read(maximum_bytes + 1)
    except OSError as exc:
        raise ValueError("DA3 worker runtime handshake is unavailable") from exc
    if not raw or len(raw) > maximum_bytes:
        raise ValueError("DA3 worker runtime handshake is empty or oversized")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"DA3 worker runtime handshake repeats key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("DA3 worker runtime handshake is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("DA3 worker runtime handshake must be a JSON object")
    if canonicalize_json(payload) != raw:
        raise ValueError("DA3 worker runtime handshake is not canonical JSON")
    return payload


def _validated_file_records(
    records: object,
    *,
    allowed_roles: set[str],
    maximum_files: int,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(records, list) or len(records) > maximum_files:
        raise ValueError("DA3 worker evidence has an invalid or unbounded file manifest")
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for raw_record in records:
        if not isinstance(raw_record, Mapping) or set(raw_record) != _FILE_RECORD_KEYS:
            raise ValueError("DA3 worker evidence has an unknown file-record shape")
        record = dict(raw_record)
        raw_path = record.get("path")
        role = record.get("role")
        size_bytes = record.get("size_bytes")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("DA3 worker evidence has an invalid logical file path")
        logical_path = Path(raw_path)
        if logical_path.is_absolute() or ".." in logical_path.parts or logical_path.as_posix() != raw_path:
            raise ValueError("DA3 worker evidence has a noncanonical logical file path")
        if not isinstance(role, str) or role not in allowed_roles:
            raise ValueError("DA3 worker evidence has an unsupported file role")
        if not _is_sha256(record.get("sha256")):
            raise ValueError("DA3 worker evidence has an invalid file digest")
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError("DA3 worker evidence has an invalid file size")
        identity = (raw_path, str(role))
        if identity in identities:
            raise ValueError("DA3 worker evidence repeats a file record")
        identities.add(identity)
        normalized.append(record)
    if normalized != sorted(normalized, key=lambda value: (str(value["path"]), str(value["role"]))):
        raise ValueError("DA3 worker evidence file records are not in canonical order")
    return tuple(normalized)


def _validated_import_environment(payload: object) -> dict[str, Any]:
    """Validate the bounded worker import-resolution evidence envelope."""

    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "search_paths",
        "path_configuration_files",
        "runtime_environment",
        "modules",
    }:
        raise ValueError("DA3 worker evidence has an unknown import-environment shape")
    if payload.get("schema") != DA3_IMPORT_ENVIRONMENT_SCHEMA:
        raise ValueError("DA3 worker evidence has an unknown import-environment schema")

    raw_search_paths = payload.get("search_paths")
    if not isinstance(raw_search_paths, list) or len(raw_search_paths) > _MAX_IMPORT_SEARCH_PATHS:
        raise ValueError("DA3 worker evidence has an unbounded import search path")
    search_paths: list[dict[str, Any]] = []
    for ordinal, raw_record in enumerate(raw_search_paths):
        if not isinstance(raw_record, Mapping) or set(raw_record) != {"ordinal", "path", "kind"}:
            raise ValueError("DA3 worker evidence has an invalid import search path")
        raw_path = raw_record.get("path")
        if (
            type(raw_record.get("ordinal")) is not int
            or raw_record.get("ordinal") != ordinal
            or not isinstance(raw_path, str)
            or not Path(raw_path).is_absolute()
            or len(raw_path.encode("utf-8")) > _MAX_IMPORT_PATH_BYTES
            or raw_record.get("kind") not in {"directory", "file", "missing"}
        ):
            raise ValueError("DA3 worker evidence has a noncanonical import search path")
        search_paths.append(dict(raw_record))

    raw_configuration_files = payload.get("path_configuration_files")
    if not isinstance(raw_configuration_files, list) or len(raw_configuration_files) > _MAX_IMPORT_CONFIGURATION_FILES:
        raise ValueError("DA3 worker evidence has an unbounded import configuration inventory")
    configuration_files: list[dict[str, Any]] = []
    previous_path = ""
    total_bytes = 0
    for raw_record in raw_configuration_files:
        if not isinstance(raw_record, Mapping) or set(raw_record) != {"path", "sha256", "size_bytes"}:
            raise ValueError("DA3 worker evidence has an invalid import configuration record")
        raw_path = raw_record.get("path")
        size_bytes = raw_record.get("size_bytes")
        if (
            not isinstance(raw_path, str)
            or not Path(raw_path).is_absolute()
            or raw_path <= previous_path
            or len(raw_path.encode("utf-8")) > _MAX_IMPORT_PATH_BYTES
            or not _is_sha256(raw_record.get("sha256"))
            or type(size_bytes) is not int
            or size_bytes < 0
        ):
            raise ValueError("DA3 worker evidence has a noncanonical import configuration record")
        total_bytes += size_bytes
        if total_bytes > _MAX_IMPORT_CONFIGURATION_BYTES:
            raise ValueError("DA3 worker evidence has oversized import configuration")
        previous_path = raw_path
        configuration_files.append(dict(raw_record))

    raw_runtime_environment = payload.get("runtime_environment")
    if not isinstance(raw_runtime_environment, Mapping) or set(raw_runtime_environment) != set(
        _WORKER_RUNTIME_ENVIRONMENT_NAMES
    ):
        raise ValueError("DA3 worker evidence has an invalid runtime environment")
    runtime_environment = dict(raw_runtime_environment)
    if any(
        value is not None and (not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_IMPORT_PATH_BYTES)
        for value in runtime_environment.values()
    ):
        raise ValueError("DA3 worker evidence has an oversized runtime environment value")

    raw_modules = payload.get("modules")
    if (
        not isinstance(raw_modules, list)
        or len(raw_modules) < len(_WORKER_IMPORT_MODULES)
        or len(raw_modules) > _MAX_IMPORT_MODULES
    ):
        raise ValueError("DA3 worker evidence has an incomplete import module inventory")
    unvalidated_module_names = [record.get("name") if isinstance(record, Mapping) else None for record in raw_modules]
    if any(
        not isinstance(name, str) or not name.isidentifier() or len(name.encode("utf-8")) > _MAX_IMPORT_PATH_BYTES
        for name in unvalidated_module_names
    ):
        raise ValueError("DA3 worker evidence has a noncanonical import module inventory")
    raw_module_names = [str(name) for name in unvalidated_module_names]
    if raw_module_names != sorted(set(raw_module_names)) or not set(_WORKER_IMPORT_MODULES).issubset(raw_module_names):
        raise ValueError("DA3 worker evidence has a noncanonical import module inventory")
    modules: list[dict[str, Any]] = []
    origin_records: dict[str, tuple[str, int]] = {}
    total_origin_bytes = 0
    for expected_name, raw_record in zip(raw_module_names, raw_modules):
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "name",
            "present",
            "origin",
            "origin_sha256",
            "origin_size_bytes",
            "search_locations",
        }:
            raise ValueError("DA3 worker evidence has an invalid import module record")
        present = raw_record.get("present")
        origin = raw_record.get("origin")
        origin_sha256 = raw_record.get("origin_sha256")
        origin_size_bytes = raw_record.get("origin_size_bytes")
        locations = raw_record.get("search_locations")
        if raw_record.get("name") != expected_name or type(present) is not bool or not isinstance(locations, list):
            raise ValueError("DA3 worker evidence has a noncanonical import module record")
        if len(locations) > _MAX_MODULE_SEARCH_ROOTS:
            raise ValueError("DA3 worker evidence has unbounded import module roots")
        if any(
            not isinstance(value, str) or not Path(value).is_absolute() or len(value.encode("utf-8")) > _MAX_IMPORT_PATH_BYTES
            for value in locations
        ):
            raise ValueError("DA3 worker evidence has an invalid import module root")
        if len(locations) != len(set(locations)):
            raise ValueError("DA3 worker evidence repeats an import module root")
        if present:
            if (
                origin is not None
                and origin not in ("built-in", "frozen")
                and (
                    not isinstance(origin, str)
                    or not Path(origin).is_absolute()
                    or len(origin.encode("utf-8")) > _MAX_IMPORT_PATH_BYTES
                )
            ):
                raise ValueError("DA3 worker evidence has an invalid import module origin")
            if origin is None and not locations:
                raise ValueError("DA3 worker evidence has an unbound import module")
            if isinstance(origin, str) and origin not in {"built-in", "frozen"}:
                if not _is_sha256(origin_sha256) or type(origin_size_bytes) is not int or origin_size_bytes < 0:
                    raise ValueError("DA3 worker evidence has an invalid import origin digest")
                previous_origin = origin_records.get(origin)
                current_origin = (str(origin_sha256), origin_size_bytes)
                if previous_origin is not None and previous_origin != current_origin:
                    raise ValueError("DA3 worker evidence contradicts a repeated import origin")
                if previous_origin is None:
                    total_origin_bytes += origin_size_bytes
                    if total_origin_bytes > _MAX_IMPORT_ORIGIN_BYTES:
                        raise ValueError("DA3 worker evidence has oversized import origins")
                    origin_records[origin] = current_origin
            elif origin_sha256 is not None or origin_size_bytes is not None:
                raise ValueError("DA3 worker evidence has a digest for an unhashable import origin")
        elif origin is not None or origin_sha256 is not None or origin_size_bytes is not None or locations:
            raise ValueError("DA3 worker evidence has contradictory absent-module evidence")
        modules.append(dict(raw_record))
    return {
        "schema": DA3_IMPORT_ENVIRONMENT_SCHEMA,
        "search_paths": search_paths,
        "path_configuration_files": configuration_files,
        "runtime_environment": runtime_environment,
        "modules": modules,
    }


def _validated_backend_identity(payload: object, *, cacheable: bool) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _BACKEND_IDENTITY_KEYS:
        raise ValueError("DA3 worker runtime-identity report has an unknown backend shape")
    backend = dict(payload)
    required_strings = (
        "model_canonical_key",
        "model_repo_id",
        "executed_backend",
        "requested_device",
        "actual_device",
    )
    if backend.get("backend_id") != "da3" or any(
        not isinstance(backend.get(field_name), str) or not backend[field_name] for field_name in required_strings
    ):
        raise ValueError("DA3 worker runtime-identity report has invalid backend identity")
    revision = backend.get("model_lock_revision")
    if revision is not None and (
        not isinstance(revision, str)
        or len(revision) != 40
        or revision != revision.lower()
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("DA3 worker runtime-identity report has an invalid model revision")
    if cacheable and (
        revision is None
        or not _is_sha256(backend.get("materialized_weights_sha256"))
        or not _is_sha256(backend.get("materialized_model_sha256"))
    ):
        raise ValueError("DA3 worker marked incomplete backend identity as cacheable")
    return backend


def _validate_cacheable_runtime_semantics(
    backend: Mapping[str, Any],
    detailed: Mapping[str, Any],
    *,
    model_files: Sequence[Mapping[str, Any]],
    source_files: Sequence[Mapping[str, Any]],
    dependencies: Sequence[Mapping[str, Any]],
    source_revision: str | None,
) -> None:
    """Reject self-consistent reports that describe a non-governed runtime."""

    requested_device = str(backend["requested_device"])
    actual_device = str(backend["actual_device"])
    executed_backend = str(backend["executed_backend"])
    expected_backends = {"cpu": "pytorch_cpu", "mps": "pytorch_mps"}
    if (
        requested_device not in expected_backends
        or actual_device != requested_device
        or executed_backend != expected_backends[requested_device]
    ):
        raise ValueError("DA3 worker cacheable evidence has inconsistent device/backend semantics")

    interpreter = detailed["interpreter"]
    version = interpreter.get("version")
    if (
        interpreter.get("implementation") != "CPython"
        or not isinstance(version, str)
        or re.fullmatch(r"3\.11(?:\.\d+)?", version) is None
        or not _is_sha256(interpreter.get("runtime_authority_sha256"))
        or not _is_sha256(interpreter.get("executable_sha256"))
        or type(interpreter.get("executable_size_bytes")) is not int
        or interpreter["executable_size_bytes"] <= 0
    ):
        raise ValueError("DA3 worker cacheable evidence is not a governed CPython 3.11 runtime")

    runtime_platform = detailed["platform"]
    hardware = runtime_platform.get("hardware")
    if (
        runtime_platform.get("system") != "Darwin"
        or str(runtime_platform.get("machine", "")).lower() not in {"arm64", "aarch64"}
        or not isinstance(hardware, Mapping)
        or not hardware
    ):
        raise ValueError("DA3 worker cacheable evidence is not a governed Darwin arm64 runtime")

    accelerator = detailed["accelerator"]
    if (
        accelerator.get("requested_device") != requested_device
        or accelerator.get("actual_device") != actual_device
        or accelerator.get("executed_backend") != executed_backend
        or accelerator.get("available") is not True
        or accelerator.get("hardware") != hardware
    ):
        raise ValueError("DA3 worker cacheable evidence has inconsistent accelerator semantics")
    if source_revision is None:
        raise ValueError("DA3 worker cacheable evidence has no governed source revision")
    if not dependencies or not source_files:
        raise ValueError("DA3 worker cacheable evidence has an empty runtime materialization")
    if not any(record.get("role") == "weight" for record in model_files) or not any(
        record.get("role") == "model_config" and record.get("path") == "config.json" for record in model_files
    ):
        raise ValueError("DA3 worker cacheable evidence has an incomplete model materialization")


@dataclass(frozen=True)
class DA3RuntimeIdentityEvidence:
    """Closed worker report; only complete reports authorize a cache identity."""

    payload: Mapping[str, Any]

    @property
    def cacheable(self) -> bool:
        return self.payload.get("cacheable") is True

    @property
    def runtime_identity_sha256(self) -> str | None:
        value = self.payload.get("runtime_identity_sha256")
        return value if _is_sha256(value) else None

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(dumps_json(dict(self.payload), allow_nan=False))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DA3RuntimeIdentityEvidence":
        expected_keys = {
            "schema",
            "cacheable",
            "incomplete_reasons",
            "backend_identity",
            "dependency_lock_sha256",
            "dependency_identity_sha256",
            "interpreter_identity_sha256",
            "platform_identity_sha256",
            "accelerator_identity_sha256",
            "source_identity_sha256",
            "import_environment_sha256",
            "governance_contract_sha256",
            "runtime_identity_sha256",
            "evidence",
        }
        if set(payload) != expected_keys or payload.get("schema") != DA3_RUNTIME_IDENTITY_SCHEMA:
            raise ValueError("DA3 worker runtime-identity report has an unknown shape")
        cacheable = payload.get("cacheable") is True
        reasons = payload.get("incomplete_reasons")
        if not isinstance(reasons, list) or any(not isinstance(value, str) or not value for value in reasons):
            raise ValueError("DA3 worker runtime-identity report has invalid reason codes")
        if reasons != sorted(set(reasons)):
            raise ValueError("DA3 worker runtime-identity reason codes are not canonical")
        backend = _validated_backend_identity(payload.get("backend_identity"), cacheable=cacheable)
        detailed = payload.get("evidence")
        if not isinstance(detailed, Mapping) or set(detailed) != _DETAILED_EVIDENCE_KEYS:
            raise ValueError("DA3 worker runtime-identity report has unknown detailed evidence")
        model_files = _validated_file_records(
            detailed.get("model_files"),
            allowed_roles={"weight", "weight_index", "model_config"},
            maximum_files=_MAX_CONFIG_FILES + _MAX_WEIGHT_FILES + 1,
        )
        source_files = _validated_file_records(
            detailed.get("source_files"),
            allowed_roles={"runtime_source"},
            maximum_files=_MAX_SOURCE_FILES,
        )
        optional_source_modules = detailed.get("optional_source_modules")
        if not isinstance(optional_source_modules, list):
            raise ValueError("DA3 worker runtime-identity report has invalid optional source evidence")
        expected_optional_names = list(_TP_OPTIONAL_SOURCE_MODULES)
        if [
            value.get("name") if isinstance(value, Mapping) else None for value in optional_source_modules
        ] != expected_optional_names:
            raise ValueError("DA3 worker runtime-identity report has noncanonical optional source evidence")
        if any(
            not isinstance(value, Mapping) or set(value) != {"name", "present"} or type(value.get("present")) is not bool
            for value in optional_source_modules
        ):
            raise ValueError("DA3 worker runtime-identity report has malformed optional source evidence")
        dependencies = _validated_dependency_inventory(detailed.get("dependencies"))
        import_environment = _validated_import_environment(detailed.get("import_environment"))
        for field_name in ("interpreter", "platform", "accelerator"):
            value = detailed.get(field_name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"DA3 worker runtime-identity report has invalid {field_name} evidence")
        source_revision = detailed.get("source_revision")
        if source_revision is not None and (
            not isinstance(source_revision, str)
            or len(source_revision) != 40
            or source_revision != source_revision.lower()
            or any(character not in "0123456789abcdef" for character in source_revision)
        ):
            raise ValueError("DA3 worker runtime-identity report has an invalid source revision")
        digest_fields = (
            "dependency_lock_sha256",
            "dependency_identity_sha256",
            "interpreter_identity_sha256",
            "platform_identity_sha256",
            "accelerator_identity_sha256",
            "source_identity_sha256",
            "import_environment_sha256",
            "governance_contract_sha256",
        )
        if cacheable and (reasons or any(not _is_sha256(payload.get(field)) for field in digest_fields)):
            raise ValueError("DA3 worker marked incomplete runtime evidence as cacheable")
        runtime_digest = payload.get("runtime_identity_sha256")
        if cacheable != _is_sha256(runtime_digest):
            raise ValueError("DA3 worker runtime identity has inconsistent cache authority")
        if cacheable:
            _validate_cacheable_runtime_semantics(
                backend,
                detailed,
                model_files=model_files,
                source_files=source_files,
                dependencies=dependencies,
                source_revision=source_revision,
            )
            weight_files = tuple(record for record in model_files if record["role"] in {"weight", "weight_index"})
            expected_digests = {
                "materialized_weights_sha256": _sha256_payload({"schema": DA3_WEIGHT_MANIFEST_SCHEMA, "files": weight_files}),
                "materialized_model_sha256": _sha256_payload({"schema": DA3_MODEL_MANIFEST_SCHEMA, "files": model_files}),
                "dependency_identity_sha256": _sha256_payload(
                    {"schema": DA3_DEPENDENCY_IDENTITY_SCHEMA, "distributions": dependencies}
                ),
                "interpreter_identity_sha256": _sha256_payload(dict(detailed["interpreter"])),
                "platform_identity_sha256": _sha256_payload(dict(detailed["platform"])),
                "accelerator_identity_sha256": _sha256_payload(dict(detailed["accelerator"])),
                "source_identity_sha256": _sha256_payload(
                    {
                        "schema": DA3_SOURCE_IDENTITY_SCHEMA,
                        "files": source_files,
                        "optional_source_modules": [dict(value) for value in optional_source_modules],
                        "source_revision": source_revision,
                    }
                ),
                "import_environment_sha256": _sha256_payload(import_environment),
            }
            observed_digests = {
                "materialized_weights_sha256": backend["materialized_weights_sha256"],
                "materialized_model_sha256": backend["materialized_model_sha256"],
                **{field_name: payload[field_name] for field_name in expected_digests if field_name in payload},
            }
            if observed_digests != expected_digests:
                raise ValueError("DA3 worker runtime-identity evidence does not match its digest projections")
            identity_payload = {
                "schema": payload["schema"],
                "backend_identity": backend,
                **{field_name: payload[field_name] for field_name in digest_fields},
            }
            if runtime_digest != _sha256_payload(identity_payload):
                raise ValueError("DA3 worker runtime identity does not match its canonical evidence")
        return cls(payload=dict(payload))


def build_prepared_cache_runtime_evidence(
    evidence: DA3RuntimeIdentityEvidence,
    *,
    plan: Any,
    candidate_authority: Any,
) -> Any | None:
    """Adapt a complete worker report to the core depth-cache hand-off.

    Imports are intentionally local: parser-only worker operations remain
    lightweight, and the DA3 helper does not own the core identity schema.
    """

    if not evidence.cacheable:
        return None

    from ...core.execution_identity_v3 import BackendRuntimeIdentity, ExecutionIdentityV3
    from ...lux_depth_v3.depth_cache_runtime import PreparedDepthCacheRuntimeEvidence

    payload = evidence.to_mapping()
    backend = payload["backend_identity"]
    model_contract = candidate_authority.model_contract
    resolved_model = candidate_authority.resolved_model_contract
    if model_contract is None or resolved_model is None or model_contract.backend_id != "da3":
        raise ValueError("DA3 runtime evidence requires an exact carried model contract")

    if (
        backend.get("backend_id") != "da3"
        or backend.get("model_canonical_key") != resolved_model.canonical_key
        or backend.get("model_lock_revision") != resolved_model.revision
    ):
        raise ValueError("DA3 runtime evidence disagrees with the carried model contract")

    # The core hand-off has one interpreter and one source digest.  Bind the
    # worker's live installed-dependency materialization into the interpreter
    # digest, and the model config/governance materialization into source, so
    # those facts cannot drift while the exposed field set stays closed.
    interpreter_identity_sha256 = _sha256_payload(
        {
            "schema": DA3_INTERPRETER_DEPENDENCY_SCHEMA,
            "interpreter_identity_sha256": payload["interpreter_identity_sha256"],
            "dependency_identity_sha256": payload["dependency_identity_sha256"],
        }
    )
    source_identity_sha256 = _sha256_payload(
        {
            "schema": DA3_SOURCE_MODEL_SCHEMA,
            "source_identity_sha256": payload["source_identity_sha256"],
            "materialized_model_sha256": backend["materialized_model_sha256"],
            "governance_contract_sha256": payload["governance_contract_sha256"],
            "import_environment_sha256": payload["import_environment_sha256"],
        }
    )

    depth_node = next((node for node in plan.nodes if node.node_id == "lux.depth"), None)
    if depth_node is None or not plan.inputs:
        raise ValueError("DA3 runtime evidence requires the canonical Lux depth node and an input")
    seed = ExecutionIdentityV3.from_plan(
        plan,
        stage_node_id=depth_node.node_id,
        candidate_id=candidate_authority.candidate_id,
        input_id=plan.inputs[0].input_id,
        model_backend_id=model_contract.backend_id,
    )
    backend_identity = BackendRuntimeIdentity.from_seed(
        seed,
        materialized_weights_sha256=backend["materialized_weights_sha256"],
        dependency_lock_sha256=payload["dependency_lock_sha256"],
        interpreter_identity_sha256=interpreter_identity_sha256,
        platform_identity_sha256=payload["platform_identity_sha256"],
        accelerator_identity_sha256=payload["accelerator_identity_sha256"],
        source_identity_sha256=source_identity_sha256,
    )
    return PreparedDepthCacheRuntimeEvidence.create(
        backend_runtime_identities=(backend_identity,),
        dependency_lock_sha256=payload["dependency_lock_sha256"],
        interpreter_identity_sha256=interpreter_identity_sha256,
        platform_identity_sha256=payload["platform_identity_sha256"],
        accelerator_identity_sha256=payload["accelerator_identity_sha256"],
        source_identity_sha256=source_identity_sha256,
    )


def _prepare_da3_runtime_identity(
    *,
    model_canonical_key: str,
    model_repo_id: str,
    model_lock_revision: str | None,
    requested_device: str,
    actual_device: str,
    executed_backend: str,
    snapshot_path: Path | None = None,
    governance_contract_path: Path | None = None,
    runtime_authority_path: Path | None = None,
    dependency_inventory: Sequence[Mapping[str, Any]] | None = None,
    source_files: Mapping[str, Path] | None = None,
    interpreter_payload: Mapping[str, Any] | None = None,
    platform_payload: Mapping[str, Any] | None = None,
    accelerator_payload: Mapping[str, Any] | None = None,
    import_environment_payload: Mapping[str, Any] | None = None,
    actual_source_revision: str | None = None,
) -> DA3RuntimeIdentityEvidence:
    """Prepare local-only DA3 runtime evidence without loading model tensors."""

    reasons: list[str] = []
    contract_path = governance_contract_path if governance_contract_path is not None else _default_governance_path()
    governance, governance_reasons = _load_governance_contract(contract_path)
    reasons.extend(governance_reasons)
    governance_sha256 = None
    dependency_lock_sha256 = None
    expected_distributions: dict[str, str] = {}
    runtime_authority_sha256 = None
    expected_source_revision = None
    if governance is not None:
        governance_sha256 = _sha256_payload(governance)
        expected_source_revision = str(governance["source_revision"]).lower()
        if contract_path is None:  # pragma: no cover - governance cannot load without a path
            reasons.append("governance_contract_missing")
        else:
            dependency_lock_sha256, expected_distributions, dependency_lock_reasons = _verified_dependency_lock(
                governance,
                governance_contract_path=contract_path,
            )
            reasons.extend(dependency_lock_reasons)
            if dependency_lock_sha256 is not None:
                marker_path = runtime_authority_path or Path(sys.prefix) / str(governance["runtime_marker_filename"])
                runtime_authority_sha256, marker_reasons = _verified_runtime_authority_marker(
                    governance,
                    dependency_lock_sha256=dependency_lock_sha256,
                    marker_path=marker_path,
                )
                reasons.extend(marker_reasons)

    if not model_lock_revision or len(model_lock_revision) != 40:
        reasons.append("model_lock_revision_missing")
    normalized_revision = str(model_lock_revision or "").lower()

    resolved_snapshot = snapshot_path
    if resolved_snapshot is None and normalized_revision:
        try:
            resolved_snapshot = _local_snapshot(model_repo_id, normalized_revision)
        except (ImportError, OSError, ValueError):
            reasons.append("model_snapshot_unavailable_local_only")

    weights_sha256 = None
    model_sha256 = None
    model_records: tuple[dict[str, Any], ...] = ()
    if resolved_snapshot is not None:
        try:
            if resolved_snapshot.resolve().name.lower() != normalized_revision:
                reasons.append("model_snapshot_revision_mismatch")
            weights_sha256, model_sha256, model_records = _materialized_model_manifests(resolved_snapshot)
        except (OSError, ValueError):
            reasons.append("model_snapshot_materialization_invalid")

    dependency_import_modules: tuple[str, ...] = ()
    if dependency_inventory is None:
        dependency_records, dependency_reasons, dependency_import_modules = _dependency_inventory(expected_distributions)
        reasons.extend(dependency_reasons)
    else:
        try:
            dependency_records = _validated_dependency_inventory(dependency_inventory)
            reasons.extend(_dependency_closure_reasons(dependency_records, expected_distributions))
        except (TypeError, ValueError):
            dependency_records = ()
            reasons.append("dependency_identity_invalid")
    dependency_payload = {
        "schema": DA3_DEPENDENCY_IDENTITY_SCHEMA,
        "distributions": dependency_records,
    }
    dependency_identity_sha256 = _sha256_payload(dependency_payload) if dependency_records else None
    if dependency_identity_sha256 is None:
        reasons.append("dependency_identity_unavailable")

    worker_import_modules = tuple(sorted({*_WORKER_IMPORT_MODULES, *dependency_import_modules}))
    if len(worker_import_modules) > _MAX_IMPORT_MODULES:
        # The dependency inventory is already non-authorizing when its import
        # projection exceeds the bound.  Keep the fallback evidence itself
        # bounded and parseable so callers can safely continue without cache
        # authority instead of turning an oversized probe into worker failure.
        reasons.append("dependency_import_inventory_unbounded")
        dependency_import_modules = ()
        worker_import_modules = _WORKER_IMPORT_MODULES
    unavailable_import_environment = {
        "schema": DA3_IMPORT_ENVIRONMENT_SCHEMA,
        "search_paths": [],
        "path_configuration_files": [],
        "runtime_environment": {name: None for name in _WORKER_RUNTIME_ENVIRONMENT_NAMES},
        "modules": [
            {
                "name": module_name,
                "present": False,
                "origin": None,
                "origin_sha256": None,
                "origin_size_bytes": None,
                "search_locations": [],
            }
            for module_name in worker_import_modules
        ],
    }
    try:
        normalized_import_environment = (
            _worker_import_environment_payload(dependency_import_modules)
            if import_environment_payload is None
            else _validated_import_environment(import_environment_payload)
        )
        if dependency_inventory is None:
            reasons.extend(_worker_import_environment_reasons(normalized_import_environment))
        import_environment_sha256 = _sha256_payload(normalized_import_environment)
    except (ImportError, OSError, TypeError, ValueError):
        normalized_import_environment = unavailable_import_environment
        import_environment_sha256 = None
        reasons.append("import_environment_unavailable")

    if interpreter_payload is None:
        normalized_interpreter, interpreter_reasons = _interpreter_payload()
        reasons.extend(interpreter_reasons)
    else:
        normalized_interpreter = dict(interpreter_payload)
    if runtime_authority_sha256 is not None:
        normalized_interpreter["runtime_authority_sha256"] = runtime_authority_sha256
    version = str(normalized_interpreter.get("version", ""))
    if tuple(version.split(".")[:2]) != ("3", "11"):
        reasons.append("runtime_python_unsupported")
    interpreter_identity_sha256 = _sha256_payload(normalized_interpreter) if normalized_interpreter else None

    normalized_platform = dict(platform_payload) if platform_payload is not None else _platform_payload()
    system_name = str(normalized_platform.get("system", ""))
    machine_name = str(normalized_platform.get("machine", "")).lower()
    if system_name != "Darwin" or machine_name not in {"arm64", "aarch64"}:
        reasons.append("runtime_platform_unsupported")
    if not isinstance(normalized_platform.get("hardware"), Mapping) or not normalized_platform["hardware"]:
        reasons.append("hardware_identity_unavailable")
    platform_identity_sha256 = _sha256_payload(normalized_platform) if normalized_platform else None

    if accelerator_payload is None:
        normalized_accelerator, accelerator_reasons = _accelerator_payload(
            requested_device=requested_device,
            actual_device=actual_device,
            executed_backend=executed_backend,
        )
        reasons.extend(accelerator_reasons)
    else:
        normalized_accelerator = dict(accelerator_payload)
        if requested_device.lower() != actual_device.lower():
            reasons.append(f"device_mismatch:{requested_device.lower()}:{actual_device.lower()}")
        if actual_device.lower() in {"cpu", "mps"} and (
            not isinstance(normalized_accelerator.get("hardware"), Mapping) or not normalized_accelerator["hardware"]
        ):
            reasons.append("hardware_identity_unavailable")
    accelerator_identity_sha256 = _sha256_payload(normalized_accelerator) if normalized_accelerator else None

    observed_source_revision = (actual_source_revision or _git_source_revision() or "").lower()
    source_records: tuple[dict[str, Any], ...] = ()
    source_identity_sha256 = None
    resolved_source_files: Mapping[str, Path]
    try:
        if source_files is None:
            resolved_source_files, optional_source_modules = _source_file_mapping()
        else:
            resolved_source_files = source_files
            optional_source_modules = _optional_source_module_records()
        source_identity_sha256, source_records = _source_identity(
            resolved_source_files,
            source_revision=observed_source_revision or None,
            optional_source_modules=optional_source_modules,
        )
    except (ImportError, OSError, ValueError):
        optional_source_modules = ()
        reasons.append("source_identity_unavailable")

    if expected_source_revision and observed_source_revision != expected_source_revision:
        reasons.append("source_revision_mismatch")

    backend_identity = {
        "backend_id": "da3",
        "model_canonical_key": model_canonical_key,
        "model_repo_id": model_repo_id,
        "model_lock_revision": normalized_revision or None,
        "executed_backend": executed_backend,
        "requested_device": requested_device,
        "actual_device": actual_device,
        "materialized_weights_sha256": weights_sha256,
        "materialized_model_sha256": model_sha256,
    }
    required_digests = (
        weights_sha256,
        model_sha256,
        dependency_lock_sha256,
        dependency_identity_sha256,
        interpreter_identity_sha256,
        platform_identity_sha256,
        accelerator_identity_sha256,
        source_identity_sha256,
        import_environment_sha256,
        governance_sha256,
        runtime_authority_sha256,
    )
    if any(not _is_sha256(value) for value in required_digests):
        reasons.append("required_identity_digest_missing")

    unique_reasons = sorted(set(reasons))
    identity_payload = {
        "schema": DA3_RUNTIME_IDENTITY_SCHEMA,
        "backend_identity": backend_identity,
        "dependency_lock_sha256": dependency_lock_sha256,
        "dependency_identity_sha256": dependency_identity_sha256,
        "interpreter_identity_sha256": interpreter_identity_sha256,
        "platform_identity_sha256": platform_identity_sha256,
        "accelerator_identity_sha256": accelerator_identity_sha256,
        "source_identity_sha256": source_identity_sha256,
        "import_environment_sha256": import_environment_sha256,
        "governance_contract_sha256": governance_sha256,
    }
    cacheable = not unique_reasons
    runtime_identity_sha256 = _sha256_payload(identity_payload) if cacheable else None
    report = {
        **identity_payload,
        "cacheable": cacheable,
        "incomplete_reasons": unique_reasons,
        "runtime_identity_sha256": runtime_identity_sha256,
        "evidence": {
            "model_files": list(model_records),
            "dependencies": list(dependency_records),
            "interpreter": normalized_interpreter,
            "platform": normalized_platform,
            "accelerator": normalized_accelerator,
            "source_files": list(source_records),
            "optional_source_modules": list(optional_source_modules),
            "source_revision": observed_source_revision or None,
            "import_environment": normalized_import_environment,
        },
    }
    return DA3RuntimeIdentityEvidence.from_mapping(report)


def prepare_da3_runtime_identity(**kwargs: Any) -> DA3RuntimeIdentityEvidence:
    """Prepare DA3 identity without retaining the private stat-token surface."""

    return _prepare_da3_runtime_identity(**kwargs)


def prepare_da3_runtime_identity_with_verification_token(
    **kwargs: Any,
) -> tuple[DA3RuntimeIdentityEvidence, dict[str, Any] | None]:
    """Fully hash once and return an authenticated mutation-sensitive stat token."""

    entries: dict[str, dict[str, Any]] = {}
    source_revision_probe: dict[str, str] = {}
    entries_context_token = _VERIFICATION_ENTRIES.set(entries)
    revision_context_token = _SOURCE_REVISION_PROBE.set(source_revision_probe)
    try:
        evidence = _prepare_da3_runtime_identity(**kwargs)
    finally:
        _SOURCE_REVISION_PROBE.reset(revision_context_token)
        _VERIFICATION_ENTRIES.reset(entries_context_token)
    if not evidence.cacheable or evidence.runtime_identity_sha256 is None:
        return evidence, None
    ordered_entries = sorted(entries.values(), key=lambda value: str(value["path"]))
    evidence_payload = evidence.to_mapping()
    import_environment = evidence_payload["evidence"]["import_environment"]
    payload = {
        "schema": "tp.da3.runtime-verification-token.v1",
        "worker_runtime_identity_sha256": evidence.runtime_identity_sha256,
        "worker_import_environment_sha256": evidence_payload["import_environment_sha256"],
        "worker_import_environment": import_environment,
        "prepared_runtime": None,
        "source_revision_probe": source_revision_probe or None,
        "entries": ordered_entries,
    }
    if len(canonicalize_json(payload)) > _MAX_VERIFICATION_TOKEN_BYTES:
        return evidence, None
    return evidence, payload


def runtime_verification_token_sha256(payload: Mapping[str, Any]) -> str:
    return _sha256_payload(payload)


def verify_runtime_verification_token(
    payload: Mapping[str, Any],
    *,
    expected_token_sha256: str,
    expected_worker_runtime_identity_sha256: str,
    expected_prepared_runtime_identity_sha256: str | None = None,
    expected_requested_device: str | None = None,
    expected_actual_device: str | None = None,
    expected_executed_backend: str | None = None,
    revalidate_worker_import_environment: bool = False,
) -> bool:
    """Authenticate and cheaply re-stat every fully hashed runtime input."""

    try:
        payload_keys = set(payload)
        observed_token_sha256 = runtime_verification_token_sha256(payload)
    except (TypeError, ValueError):
        return False
    if payload_keys != {
        "schema",
        "worker_runtime_identity_sha256",
        "worker_import_environment_sha256",
        "worker_import_environment",
        "prepared_runtime",
        "source_revision_probe",
        "entries",
    }:
        return False
    if payload.get("schema") != "tp.da3.runtime-verification-token.v1":
        return False
    if payload.get("worker_runtime_identity_sha256") != expected_worker_runtime_identity_sha256:
        return False
    if not _is_sha256(expected_token_sha256) or observed_token_sha256 != expected_token_sha256:
        return False
    try:
        import_environment = _validated_import_environment(payload.get("worker_import_environment"))
    except (TypeError, ValueError):
        return False
    import_environment_sha256 = payload.get("worker_import_environment_sha256")
    if not _is_sha256(import_environment_sha256) or _sha256_payload(import_environment) != import_environment_sha256:
        return False
    if revalidate_worker_import_environment:
        try:
            module_names = tuple(str(record["name"]) for record in import_environment["modules"])
            if _sha256_payload(_worker_import_environment_payload(module_names)) != import_environment_sha256:
                return False
        except (ImportError, OSError, TypeError, ValueError):
            return False

    expected_binding_values = (
        expected_prepared_runtime_identity_sha256,
        expected_requested_device,
        expected_actual_device,
        expected_executed_backend,
    )
    prepared_runtime = payload.get("prepared_runtime")
    if prepared_runtime is None:
        if any(value is not None for value in expected_binding_values):
            return False
    elif not isinstance(prepared_runtime, Mapping) or set(prepared_runtime) != {
        "schema",
        "runtime_identity_sha256",
        "requested_device",
        "actual_device",
        "executed_backend",
    }:
        return False
    else:
        binding = dict(prepared_runtime)
        if (
            any(value is None for value in expected_binding_values)
            or binding.get("schema") != "tp.da3.prepared-runtime-binding.v1"
            or not _is_sha256(binding.get("runtime_identity_sha256"))
            or any(
                not isinstance(binding.get(field_name), str) or not binding[field_name]
                for field_name in ("requested_device", "actual_device", "executed_backend")
            )
        ):
            return False
        expected_binding = {
            "runtime_identity_sha256": expected_prepared_runtime_identity_sha256,
            "requested_device": expected_requested_device,
            "actual_device": expected_actual_device,
            "executed_backend": expected_executed_backend,
        }
        if any(value is not None and binding[field_name] != value for field_name, value in expected_binding.items()):
            return False
    if not _verify_source_revision_probe(payload.get("source_revision_probe")):
        return False
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries or len(entries) > _MAX_VERIFICATION_ENTRIES:
        return False
    previous_path = ""
    expected_keys = {"path", "kind", "device", "inode", "size_bytes", "mtime_ns", "ctime_ns"}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != expected_keys:
            return False
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute() or raw_path <= previous_path:
            return False
        previous_path = raw_path
        kind = entry.get("kind")
        if kind not in {"file", "directory"}:
            return False
        if any(type(entry.get(field)) is not int or int(entry[field]) < 0 for field in expected_keys - {"path", "kind"}):
            return False
        try:
            resolved = Path(raw_path).resolve(strict=True)
            if str(resolved) != raw_path:
                return False
            observed = resolved.stat()
        except (OSError, RuntimeError):
            return False
        observed_kind = "directory" if stat.S_ISDIR(observed.st_mode) else "file"
        observed_projection = {
            "path": raw_path,
            "kind": observed_kind,
            "device": int(observed.st_dev),
            "inode": int(observed.st_ino),
            "size_bytes": int(observed.st_size),
            "mtime_ns": int(observed.st_mtime_ns),
            "ctime_ns": int(observed.st_ctime_ns),
        }
        if dict(entry) != observed_projection:
            return False
    return True


@dataclass(frozen=True)
class ParentOutputRuntimeIdentityEvidence:
    """Closed parent-process inputs that can affect cached DA3 depth bytes."""

    interpreter_identity_sha256: str
    dependency_identity_sha256: str
    source_identity_sha256: str
    platform_identity_sha256: str
    accelerator_identity_sha256: str
    import_environment_sha256: str
    platform_payload: Mapping[str, Any]
    accelerator_payload: Mapping[str, Any]
    verification_entries: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        digests = (
            self.interpreter_identity_sha256,
            self.dependency_identity_sha256,
            self.source_identity_sha256,
            self.platform_identity_sha256,
            self.accelerator_identity_sha256,
            self.import_environment_sha256,
        )
        if any(not _is_sha256(value) for value in digests):
            raise ValueError("DA3 parent output runtime identity is incomplete")
        if set(self.platform_payload) != {"system", "release", "version", "machine", "hardware"} or set(
            self.accelerator_payload
        ) != {"execution_domain", "actual_device", "available", "hardware"}:
            raise ValueError("DA3 parent output platform evidence has an unknown shape")
        hardware = self.platform_payload.get("hardware")
        if (
            any(
                not isinstance(self.platform_payload.get(field_name), str) or not self.platform_payload[field_name]
                for field_name in ("system", "release", "version", "machine")
            )
            or not isinstance(hardware, Mapping)
            or not hardware
            or self.accelerator_payload.get("execution_domain") != "parent_output"
            or self.accelerator_payload.get("actual_device") != "cpu"
            or self.accelerator_payload.get("available") is not True
            or _sha256_payload(self.platform_payload) != self.platform_identity_sha256
            or _sha256_payload(self.accelerator_payload) != self.accelerator_identity_sha256
            or self.accelerator_payload.get("hardware") != hardware
        ):
            raise ValueError("DA3 parent output platform evidence is inconsistent")


def prepare_parent_output_runtime_identity() -> ParentOutputRuntimeIdentityEvidence:
    """Hash parent code, imports, packages, platform, and output hardware."""

    entries: dict[str, dict[str, Any]] = {}
    context_token = _VERIFICATION_ENTRIES.set(entries)
    try:
        installed = _installed_distribution_index()
        import_environment = _parent_import_environment_payload()
        module_presence = {record["name"]: record["present"] for record in import_environment["modules"]}

        required = {"numpy", "pillow"}
        optional_modules = [
            {"name": "scipy", "present": module_presence["scipy"]},
            {"name": "cv2", "present": module_presence["cv2"]},
        ]
        if module_presence["scipy"]:
            required.add("scipy")
        if module_presence["cv2"]:
            opencv_names = {name for name in installed if name.startswith("opencv-")}
            if not opencv_names:
                raise ValueError("Parent cv2 runtime has no governed installed distribution")
            required.update(opencv_names)
        missing = sorted(required - set(installed))
        if missing or not module_presence["numpy"] or not module_presence["PIL"]:
            raise ValueError(f"Parent output runtime is missing dependencies: {', '.join(missing) or 'module origin'}")

        interpreter, interpreter_reasons = _interpreter_payload()
        if interpreter_reasons:
            raise ValueError("Parent output interpreter is not fully materialized")
        dependency_records = tuple(
            sorted(
                (
                    _distribution_record(
                        name,
                        distribution=installed[name][0],
                        verify_record_hashes=False,
                    )
                    for name in required
                ),
                key=lambda value: str(value["name"]),
            )
        )

        source_files: dict[str, Path] = {}
        for module_name in _PARENT_OUTPUT_SOURCE_MODULES:
            source_files.update(_module_source_files(module_name, maximum_files=_MAX_SOURCE_FILES - len(source_files)))
        source_files.update(_runtime_configuration_file_mapping())
        optional_source_modules: list[dict[str, Any]] = []
        for module_name in _PARENT_OUTPUT_OPTIONAL_SOURCE_MODULES:
            present = module_presence[module_name]
            optional_source_modules.append({"name": module_name, "present": present})
            if present:
                source_files.update(_module_source_files(module_name, maximum_files=_MAX_SOURCE_FILES - len(source_files)))
        base_source_digest, _source_records = _source_identity(
            source_files,
            source_revision=None,
            optional_source_modules=optional_source_modules,
        )

        hardware = _hardware_payload()
        if not hardware:
            raise ValueError("Parent output runtime has no stable hardware identity")
        platform_payload = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "hardware": hardware,
        }
        accelerator_payload = {
            "execution_domain": "parent_output",
            "actual_device": "cpu",
            "available": True,
            "hardware": hardware,
        }
        import_environment_sha256 = _sha256_payload(import_environment)
        dependency_identity_sha256 = _sha256_payload(
            {
                "schema": "tp.da3.parent-output-dependencies.v1",
                "distributions": dependency_records,
                "optional_modules": optional_modules,
                "import_environment_sha256": import_environment_sha256,
            }
        )
        source_identity_sha256 = _sha256_payload(
            {
                "schema": "tp.da3.parent-output-source.v1",
                "source_identity_sha256": base_source_digest,
                "import_environment_sha256": import_environment_sha256,
            }
        )
    finally:
        _VERIFICATION_ENTRIES.reset(context_token)

    return ParentOutputRuntimeIdentityEvidence(
        interpreter_identity_sha256=_sha256_payload(interpreter),
        dependency_identity_sha256=dependency_identity_sha256,
        source_identity_sha256=source_identity_sha256,
        platform_identity_sha256=_sha256_payload(platform_payload),
        accelerator_identity_sha256=_sha256_payload(accelerator_payload),
        import_environment_sha256=import_environment_sha256,
        platform_payload=platform_payload,
        accelerator_payload=accelerator_payload,
        verification_entries=tuple(sorted(entries.values(), key=lambda value: str(value["path"]))),
    )


def verify_parent_output_runtime_identity(evidence: ParentOutputRuntimeIdentityEvidence) -> bool:
    """Re-resolve parent imports plus live platform and hardware identity."""

    try:
        current_import_environment = _sha256_payload(_parent_import_environment_payload())
        hardware = _hardware_payload()
        if not hardware:
            return False
        platform_payload = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "hardware": hardware,
        }
        accelerator_payload = {
            "execution_domain": "parent_output",
            "actual_device": "cpu",
            "available": True,
            "hardware": hardware,
        }
    except (OSError, TypeError, ValueError):
        return False
    return (
        current_import_environment == evidence.import_environment_sha256
        and _sha256_payload(platform_payload) == evidence.platform_identity_sha256
        and _sha256_payload(accelerator_payload) == evidence.accelerator_identity_sha256
    )


def merge_runtime_verification_entries(
    payload: Mapping[str, Any],
    additional_entries: Sequence[Mapping[str, Any]],
    *,
    prepared_runtime_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge freshly observed parent stats into a worker-authenticated token."""

    if set(payload) != {
        "schema",
        "worker_runtime_identity_sha256",
        "worker_import_environment_sha256",
        "worker_import_environment",
        "prepared_runtime",
        "source_revision_probe",
        "entries",
    }:
        raise ValueError("DA3 runtime verification token has an unknown shape")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("DA3 runtime verification token has no entry list")
    merged: dict[str, dict[str, Any]] = {}
    for raw_entry in (*raw_entries, *additional_entries):
        if not isinstance(raw_entry, Mapping) or not isinstance(raw_entry.get("path"), str):
            raise ValueError("DA3 runtime verification token contains an invalid entry")
        entry = dict(raw_entry)
        previous = merged.get(entry["path"])
        if previous is not None and previous != entry:
            raise ValueError("DA3 runtime verification token has conflicting observations")
        merged[entry["path"]] = entry
    if len(merged) > _MAX_VERIFICATION_ENTRIES:
        raise ValueError("DA3 runtime verification token contains too many entries")
    existing_binding = payload.get("prepared_runtime")
    if existing_binding is not None and prepared_runtime_binding is not None and existing_binding != prepared_runtime_binding:
        raise ValueError("DA3 runtime verification token has conflicting prepared-runtime authority")
    selected_binding = existing_binding if prepared_runtime_binding is None else dict(prepared_runtime_binding)
    result = {
        "schema": payload["schema"],
        "worker_runtime_identity_sha256": payload["worker_runtime_identity_sha256"],
        "worker_import_environment_sha256": payload["worker_import_environment_sha256"],
        "worker_import_environment": payload["worker_import_environment"],
        "prepared_runtime": selected_binding,
        "source_revision_probe": payload["source_revision_probe"],
        "entries": sorted(merged.values(), key=lambda value: str(value["path"])),
    }
    if len(canonicalize_json(result)) > _MAX_VERIFICATION_TOKEN_BYTES:
        raise ValueError("DA3 runtime verification token is oversized")
    return result


def bind_parent_output_dependency_identity(
    prepared: Any,
    *,
    parent_runtime_identity: ParentOutputRuntimeIdentityEvidence,
) -> Any:
    """Fold parent postprocessing dependencies into the closed core hand-off."""

    from ...core.execution_identity_v3 import BackendRuntimeIdentity
    from ...lux_depth_v3.depth_cache_runtime import PreparedDepthCacheRuntimeEvidence

    if len(prepared.backend_runtime_identities) != 1:
        raise ValueError("DA3 parent output dependency identity is incomplete")
    original = prepared.backend_runtime_identities[0]
    interpreter_identity_sha256 = _sha256_payload(
        {
            "schema": DA3_PARENT_RUNTIME_SCHEMA,
            "worker_interpreter_identity_sha256": original.interpreter_identity_sha256,
            "parent_interpreter_identity_sha256": parent_runtime_identity.interpreter_identity_sha256,
            "parent_dependency_identity_sha256": parent_runtime_identity.dependency_identity_sha256,
        }
    )
    source_identity_sha256 = _sha256_payload(
        {
            "schema": DA3_PARENT_RUNTIME_SCHEMA,
            "worker_source_identity_sha256": original.source_identity_sha256,
            "parent_source_identity_sha256": parent_runtime_identity.source_identity_sha256,
        }
    )
    platform_identity_sha256 = _sha256_payload(
        {
            "schema": DA3_PARENT_RUNTIME_SCHEMA,
            "worker_platform_identity_sha256": original.platform_identity_sha256,
            "parent_platform_identity_sha256": parent_runtime_identity.platform_identity_sha256,
        }
    )
    accelerator_identity_sha256 = _sha256_payload(
        {
            "schema": DA3_PARENT_RUNTIME_SCHEMA,
            "worker_accelerator_identity_sha256": original.accelerator_identity_sha256,
            "parent_accelerator_identity_sha256": parent_runtime_identity.accelerator_identity_sha256,
        }
    )
    identity_payload = original.to_payload()
    identity_payload["interpreter_identity_sha256"] = interpreter_identity_sha256
    identity_payload["source_identity_sha256"] = source_identity_sha256
    identity_payload["platform_identity_sha256"] = platform_identity_sha256
    identity_payload["accelerator_identity_sha256"] = accelerator_identity_sha256
    identity = BackendRuntimeIdentity.from_payload(identity_payload)
    return PreparedDepthCacheRuntimeEvidence.create(
        backend_runtime_identities=(identity,),
        dependency_lock_sha256=identity.dependency_lock_sha256,
        interpreter_identity_sha256=identity.interpreter_identity_sha256,
        platform_identity_sha256=platform_identity_sha256,
        accelerator_identity_sha256=accelerator_identity_sha256,
        source_identity_sha256=source_identity_sha256,
    )


__all__ = [
    "DA3_RUNTIME_IDENTITY_SCHEMA",
    "DA3CacheRuntimeGovernanceIdentity",
    "DA3RuntimeIdentityEvidence",
    "ParentOutputRuntimeIdentityEvidence",
    "build_prepared_cache_runtime_evidence",
    "da3_cache_governance_enabled",
    "da3_cache_governance_identity",
    "da3_cache_runtime_governance_identity",
    "load_da3_worker_runtime_handshake",
    "prepare_da3_runtime_identity",
    "prepare_parent_output_runtime_identity",
    "verify_parent_output_runtime_identity",
]
