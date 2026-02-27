"""Subprocess-backed GPG helpers for detached attestation signing/verification."""

from __future__ import annotations

import subprocess

_GPG_TIMEOUT_SECONDS = 5


def gpg_clearsign_bytes(payload: bytes, *, key_id: str | None = None) -> str:
    """Produce an ASCII-armored cleartext signature over the provided bytes."""
    cmd = ["gpg", "--clearsign", "--armor", "--batch", "--yes", "--no-tty"]
    if key_id:
        cmd.extend(["--local-user", key_id])

    try:
        proc = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            check=False,
            timeout=_GPG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"gpg signing timed out after {_GPG_TIMEOUT_SECONDS} seconds") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise ValueError(f"gpg signing failed: {stderr.strip()}")
    return proc.stdout.decode("utf-8", errors="replace")


def gpg_verify_clearsign(signature_text: str) -> None:
    """Verify an ASCII-armored cleartext signature."""
    try:
        proc = subprocess.run(
            ["gpg", "--verify", "--batch", "--no-tty"],
            input=signature_text.encode("utf-8"),
            capture_output=True,
            check=False,
            timeout=_GPG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"gpg verify timed out after {_GPG_TIMEOUT_SECONDS} seconds") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise ValueError(f"gpg verify failed: {stderr.strip()}")
