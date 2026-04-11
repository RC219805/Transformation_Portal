"""Optional Sigstore helpers for signing attestation sidecars."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_COSIGN_TIMEOUT_SECONDS = 30


def ensure_cosign_available() -> str:
    """Return the resolved cosign executable path or raise a clear error."""
    cosign = shutil.which("cosign")
    if not cosign:
        raise ValueError("cosign executable not found in PATH")
    return cosign


def cosign_sign_blob(
    *,
    blob_path: Path,
    bundle_path: Path,
    key_path: Path | None = None,
    tlog_upload: bool = True,
) -> None:
    """Sign an attestation blob and write a Sigstore bundle sidecar."""
    cosign = ensure_cosign_available()
    cmd = [cosign, "sign-blob", "--bundle", str(bundle_path), str(blob_path)]
    if key_path is not None:
        cmd[2:2] = ["--key", str(key_path)]
    if not tlog_upload:
        cmd[2:2] = ["--tlog-upload=false"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=_COSIGN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"cosign sign-blob timed out after {_COSIGN_TIMEOUT_SECONDS} seconds") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise ValueError(f"cosign sign-blob failed: {stderr}")


def cosign_verify_blob(
    *,
    blob_path: Path,
    bundle_path: Path,
    key_path: Path | None = None,
) -> None:
    """Verify a Sigstore bundle against an attestation blob."""
    cosign = ensure_cosign_available()
    cmd = [cosign, "verify-blob", "--bundle", str(bundle_path), str(blob_path)]
    if key_path is not None:
        cmd[2:2] = ["--key", str(key_path)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=_COSIGN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"cosign verify-blob timed out after {_COSIGN_TIMEOUT_SECONDS} seconds") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise ValueError(f"cosign verify-blob failed: {stderr}")
