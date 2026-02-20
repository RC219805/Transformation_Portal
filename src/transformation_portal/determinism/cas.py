from __future__ import annotations

import errno
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .jcs import dumpb, sha256_hex_of_canonical_json


def _fsync_file(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class CasPaths:
    cas_root: Path

    @property
    def sha256_root(self) -> Path:
        return self.cas_root / "sha256"

    def artifact_dir(self, tensor_hash_hex: str) -> Path:
        return self.sha256_root / tensor_hash_hex

    def runs_dir(self, tensor_hash_hex: str) -> Path:
        return self.artifact_dir(tensor_hash_hex) / "runs"


def atomic_commit_dir(staging_dir: Path, final_dir: Path) -> None:
    """Atomically rename staging_dir -> final_dir (same filesystem)."""
    try:
        os.replace(staging_dir, final_dir)
    except OSError as e:
        if e.errno in (errno.EEXIST, errno.ENOTEMPTY):
            # Target already exists. Treat as no-op.
            raise FileExistsError(str(final_dir)) from e
        raise


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stage_dir(cas_root: Path, prefix: str) -> Path:
    ensure_dir(cas_root)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(cas_root)))


def write_json(path: Path, obj) -> str:
    """Write canonical JSON (JCS) and return sha256 hex of canonical bytes."""
    b = dumpb(obj)
    path.write_bytes(b)
    _fsync_file(path)
    return sha256_hex_of_canonical_json(obj)


def build_artifact_manifest(root_dir: Path) -> Dict[str, str]:
    """Return mapping of relative path -> sha256 hex for files under root_dir."""
    manifest: Dict[str, str] = {}
    for p in sorted(root_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(root_dir))
            manifest[rel] = sha256_file(p)
    return manifest
