"""Shared CAS serialization and cache write helpers.

This private module centralizes helper routines that are shared by the
execution wrapper and DAG executor. Public compatibility aliases remain in
``execution_wrapper`` for existing callers and tests.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import re
import tempfile
from pathlib import Path
from typing import Any

from transformation_portal.determinism.jcs import dumpb as jcs_dumpb
from transformation_portal.ingest.canonical_json import dump_json
from transformation_portal.storage.cas_store import CASError


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
            dump_json(data, handle, indent=2, sort_keys=True)
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


def canonical_input_value(value: Any) -> Any:
    """Encode supported stage inputs with unambiguous type and byte identity.

    Paths and arbitrary objects require a caller-owned materialization step;
    their string representation or a duck-typed ``sha256`` is not authority.
    """
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot authorize cached execution")
        return ["ndarray", compute_numpy_array_id(value)]
    if value is None:
        return ["null"]
    if type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Non-finite stage inputs are not cacheable")
        return ["float", value.hex()]
    if type(value) is bytes:
        return ["bytes", hashlib.sha256(value).hexdigest()]
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TypeError("Stage input mappings require string keys")
        return ["dict", {key: canonical_input_value(item) for key, item in value.items()}]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [canonical_input_value(item) for item in value]]
    raise TypeError(f"Unsupported cache input type: {type(value).__name__}; materialize it explicitly")


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
            try:
                np.save(array_path, value, allow_pickle=False)
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


class CASArtifactInvalidError(CASObjectMissingError):
    """A cache reference cannot safely reconstruct verified output bytes."""

    def __init__(self, reason: str, sha256: str = ""):
        self.sha256 = sha256
        Exception.__init__(self, reason)


def _load_numpy(value: dict[str, Any], artifact_store: Any) -> Any:
    try:
        return _load_verified_numpy(value, artifact_store)
    except (CASError, ValueError, TypeError, KeyError, OSError, EOFError, OverflowError) as exc:
        digest = value.get("sha256", "")
        raise CASArtifactInvalidError(
            "Cached array failed integrity or descriptor validation", digest if isinstance(digest, str) else ""
        ) from exc


def _load_verified_numpy(value: dict[str, Any], artifact_store: Any) -> Any:
    """Validate a closed array descriptor and its exact verified NPY bytes."""
    import numpy as np

    if set(value) != {"__numpy__", "sha256", "shape", "dtype"} or value["__numpy__"] is not True:
        raise ValueError("Invalid CAS array descriptor")
    shape = value["shape"]
    if not isinstance(shape, list) or len(shape) > 32 or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid CAS array shape")
    if not isinstance(value["dtype"], str) or len(value["dtype"]) > 512:
        raise ValueError("Invalid CAS array dtype")
    if not isinstance(value["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None:
        raise ValueError("Invalid CAS array digest")
    dtype = np.dtype(value["dtype"])
    if dtype.hasobject or math.prod(shape) * dtype.itemsize > 1024**3:
        raise ValueError("CAS array exceeds the supported allocation contract")
    try:
        with artifact_store.open_verified(value["sha256"]) as snapshot:
            version = np.lib.format.read_magic(snapshot)
            if version == (1, 0):
                stored_shape, _fortran, stored_dtype = np.lib.format.read_array_header_1_0(snapshot)
            elif version == (2, 0):
                stored_shape, _fortran, stored_dtype = np.lib.format.read_array_header_2_0(snapshot)
            else:
                raise ValueError("Unsupported cached NPY format")
            if list(stored_shape) != shape or stored_dtype != dtype or stored_dtype.hasobject:
                raise ValueError("CAS array header does not match its manifest")
            snapshot.seek(0)
            return np.load(snapshot, allow_pickle=False)
    except FileNotFoundError as exc:
        raise CASObjectMissingError(value["sha256"]) from exc


def load_serializable(
    data: dict[str, Any],
    artifact_store: Any,
) -> dict[str, Any]:
    """Reconstruct outputs from serialized format recursively."""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict) and "__numpy__" in value:
            result[key] = _load_numpy(value, artifact_store)
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
    result = []
    for item in items:
        if isinstance(item, dict) and "__numpy__" in item:
            result.append(_load_numpy(item, artifact_store))
        elif isinstance(item, dict):
            result.append(load_serializable(item, artifact_store))
        elif isinstance(item, list):
            result.append(load_list_recursive(item, artifact_store))
        else:
            result.append(item)
    return result
