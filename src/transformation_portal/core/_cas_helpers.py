"""Shared CAS serialization and cache write helpers.

This private module centralizes helper routines that are shared by the
execution wrapper and DAG executor. Public compatibility aliases remain in
``execution_wrapper`` for existing callers and tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tempfile
from pathlib import Path
from typing import Any

from transformation_portal.determinism.jcs import dumpb as jcs_dumpb


class CASObjectMissingError(Exception):
    """Raised when a referenced CAS object is missing during cache load."""

    def __init__(self, sha256: str):
        self.sha256 = sha256
        super().__init__(f"CAS object missing: {sha256}")


def sanitize_cas_id_for_filename(cas_id: str) -> str:
    """Extract the hex digest from a CAS ID for safe filename usage."""
    if cas_id.startswith("sha256:"):
        return cas_id[7:]
    return cas_id


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON data to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path_str = tempfile.mkstemp(
        suffix=".tmp",
        prefix=".cache_write_",
        dir=path.parent,
    )
    tmp_path = Path(tmp_path_str)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_path, path)

        if platform.system() != "Windows" and hasattr(os, "O_DIRECTORY"):
            try:
                dir_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def compute_numpy_array_id(arr: Any) -> str:
    """Compute a deterministic identity for a NumPy array."""
    array_manifest = {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }
    return hashlib.sha256(jcs_dumpb(array_manifest)).hexdigest()


def sanitize_key_for_filename(key: str) -> str:
    """Sanitize an artifact key for safe filename usage."""
    result = re.sub(r'[/\\:*?"<>|]', "_", key)
    result = result.replace("..", "_")
    result = result.strip(". \t\n\r")
    return result if result else "_key_"


def make_serializable(
    outputs: dict[str, Any],
    artifact_store: Any,
    base_path: Path,
    cas_id: str,
) -> dict[str, Any]:
    """Convert outputs to JSON-serializable format recursively."""
    import numpy as np

    safe_cas_id = sanitize_cas_id_for_filename(cas_id)
    result = {}
    for key, value in outputs.items():
        safe_key = sanitize_key_for_filename(key)
        if isinstance(value, np.ndarray):
            array_path = base_path / f"{safe_cas_id}_{safe_key}.npy"
            array_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(array_path, value)

            try:
                cas_obj = artifact_store.add_file(array_path)
                result[key] = {
                    "__numpy__": True,
                    "sha256": cas_obj.sha256,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            finally:
                if array_path.exists():
                    array_path.unlink()
        elif isinstance(value, dict):
            result[key] = make_serializable(value, artifact_store, base_path, f"{cas_id}_{safe_key}")
        elif isinstance(value, (list, tuple)):
            result[key] = serialize_list_recursive(value, artifact_store, base_path, f"{cas_id}_{safe_key}")
        else:
            result[key] = value

    return result


def serialize_list_recursive(
    items: list | tuple,
    artifact_store: Any,
    base_path: Path,
    cas_id: str,
) -> list:
    """Serialize a list/tuple, handling nested arrays and dictionaries."""
    import numpy as np

    result = []
    for index, item in enumerate(items):
        if isinstance(item, np.ndarray):
            serialized = make_serializable({"item": item}, artifact_store, base_path, f"{cas_id}_{index}")
            result.append(serialized["item"])
        elif isinstance(item, dict):
            result.append(make_serializable(item, artifact_store, base_path, f"{cas_id}_{index}"))
        elif isinstance(item, (list, tuple)):
            result.append(serialize_list_recursive(item, artifact_store, base_path, f"{cas_id}_{index}"))
        else:
            result.append(item)
    return result


def load_serializable(
    data: dict[str, Any],
    artifact_store: Any,
) -> dict[str, Any]:
    """Reconstruct outputs from serialized format recursively."""
    import numpy as np

    result = {}
    for key, value in data.items():
        if isinstance(value, dict) and value.get("__numpy__"):
            sha256 = value["sha256"]
            cas_obj = artifact_store.get_object(sha256)
            if cas_obj:
                result[key] = np.load(cas_obj.path)
            else:
                raise CASObjectMissingError(sha256)
        elif isinstance(value, dict):
            result[key] = load_serializable(value, artifact_store)
        elif isinstance(value, list):
            result[key] = load_list_recursive(value, artifact_store)
        else:
            result[key] = value

    return result


def load_list_recursive(
    items: list,
    artifact_store: Any,
) -> list:
    """Reconstruct a list, handling nested arrays and dictionaries."""
    import numpy as np

    result = []
    for item in items:
        if isinstance(item, dict) and item.get("__numpy__"):
            sha256 = item["sha256"]
            cas_obj = artifact_store.get_object(sha256)
            if cas_obj:
                result.append(np.load(cas_obj.path))
            else:
                raise CASObjectMissingError(sha256)
        elif isinstance(item, dict):
            result.append(load_serializable(item, artifact_store))
        elif isinstance(item, list):
            result.append(load_list_recursive(item, artifact_store))
        else:
            result.append(item)
    return result
