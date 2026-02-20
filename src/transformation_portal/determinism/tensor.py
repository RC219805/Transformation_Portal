from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class TensorMeta:
    tensor_role: str
    dtype: str
    shape: tuple[int, int, int]
    order: str
    artifact_id: str

    @property
    def tensor_hash_hex(self) -> str:
        return self.artifact_id.split("sha256:", 1)[1]


def canonicalize_tensor_f32_le_c(tensor: np.ndarray) -> np.ndarray:
    """Return tensor as little-endian float32, C-contiguous (HWC)."""
    arr = np.asarray(tensor, dtype="<f4", order="C")
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D HWC tensor, got ndim={arr.ndim}")
    if arr.shape[2] != 3:
        raise ValueError(f"Expected 3 channels, got shape={arr.shape}")
    return arr


def compute_artifact_id(tensor_role: str, tensor: np.ndarray) -> str:
    role = tensor_role.strip().lower()
    if role != tensor_role:
        # Enforce ASCII lowercase contract (SPEC-DH-001 §8.2)
        if tensor_role != role:
            raise ValueError("tensor_role must be ASCII lowercase")
    arr = canonicalize_tensor_f32_le_c(tensor)
    h, w, c = arr.shape
    header = (f"tensor_role={role}\n" f"dtype=float32\n" f"order=C\n" f"shape={h},{w},{c}\n").encode("ascii")
    preimage = header + arr.tobytes(order="C")
    digest = hashlib.sha256(preimage).hexdigest()
    return f"sha256:{digest}"


def write_tensor_bin(path: Path, tensor: np.ndarray) -> None:
    arr = canonicalize_tensor_f32_le_c(tensor)
    path.write_bytes(arr.tobytes(order="C"))


def write_tensor_npy(path: Path, tensor: np.ndarray) -> None:
    # `.npy` is a convenience encoding only; hashing is based on `.bin` bytes.
    import numpy as np

    arr = np.asarray(tensor, dtype=np.float32, order="C")
    np.save(path, arr, allow_pickle=False)


def load_tensor_bin(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    data = path.read_bytes()
    expected = int(shape[0]) * int(shape[1]) * int(shape[2]) * 4
    if len(data) != expected:
        raise ValueError(f"Invalid .bin size: expected {expected} bytes, got {len(data)} bytes")
    arr = np.frombuffer(data, dtype="<f4").reshape(shape, order="C")
    return arr
