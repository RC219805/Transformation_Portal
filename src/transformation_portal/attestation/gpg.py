"""Subprocess-backed GPG helpers for detached attestation signing/verification."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

_GPG_TIMEOUT_SECONDS = 5
_GPG_STATUS_PREFIX = "[GNUPG:] "
_OPENPGP_FINGERPRINT_LENGTHS = frozenset({40, 64})
_HEX_CHARS = frozenset("0123456789ABCDEF")

# Signature verdicts on GnuPG's machine-readable ``--status-fd`` channel.
# GnuPG exits 0 and still reports VALIDSIG for a cryptographically correct
# signature made by a revoked or expired key, so the verdict keyword decides
# acceptance, never the exit code or the human-readable stderr text.
# KEYEXPIRED is deliberately not listed: GnuPG also emits it for expired
# subkeys that did not make the signature, and EXPKEYSIG is the verdict for
# the key that did.
_GPG_REJECTED_STATUS = {
    "BADSIG": "the signature does not match the signed bytes",
    "ERRSIG": "the signature could not be checked",
    "EXPSIG": "the signature has expired",
    "EXPKEYSIG": "the signing key has expired",
    "REVKEYSIG": "the signing key has been revoked",
    "KEYREVOKED": "the signing key has been revoked",
}
_GPG_ACCEPTED_SIGNATURE_STATUS = frozenset({"NEWSIG", "GOODSIG", "VALIDSIG"})


def _normalize_fingerprint(value: str, *, field: str) -> str:
    normalized = value.strip().upper()
    if len(normalized) not in _OPENPGP_FINGERPRINT_LENGTHS:
        raise ValueError(f"{field} must be a 40- or 64-character OpenPGP fingerprint")
    if any(char not in _HEX_CHARS for char in normalized):
        raise ValueError(f"{field} must be hexadecimal")
    return normalized


def _gpg_status_lines(status_output: bytes) -> list[list[str]]:
    lines: list[list[str]] = []
    for raw_line in status_output.decode("utf-8", errors="replace").splitlines():
        if not raw_line.startswith(_GPG_STATUS_PREFIX):
            continue
        fields = raw_line[len(_GPG_STATUS_PREFIX) :].split()
        if fields:
            lines.append(fields)
    return lines


def _verified_primary_fingerprint(proc: subprocess.CompletedProcess[bytes], *, label: str, action: str) -> str:
    """Return the primary-key fingerprint of the one good signature gpg reported.

    Fails closed on a rejected or unrecognised signature verdict, on a failing
    gpg exit code, and unless exactly one GOODSIG and one VALIDSIG are present.
    """
    status_lines = _gpg_status_lines(proc.stdout)
    tags = [fields[0] for fields in status_lines]
    for tag in tags:
        reason = _GPG_REJECTED_STATUS.get(tag)
        if reason is not None:
            raise ValueError(f"gpg {label} rejected: {reason} (status {tag})")
    unrecognised = sorted({tag for tag in tags if tag.endswith("SIG") and tag not in _GPG_ACCEPTED_SIGNATURE_STATUS})
    if unrecognised:
        raise ValueError(f"gpg {label} rejected: unrecognised signature status {', '.join(unrecognised)}")
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise ValueError(f"gpg {action} failed: {stderr.strip()}")

    valid_signatures = [fields[1:] for fields in status_lines if fields[0] == "VALIDSIG"]
    if len(valid_signatures) != 1:
        raise ValueError(f"gpg {label} must contain exactly one VALIDSIG record; found {len(valid_signatures)}")
    good_signature_count = tags.count("GOODSIG")
    if good_signature_count != 1:
        raise ValueError(f"gpg {label} must contain exactly one GOODSIG record; found {good_signature_count}")
    valid_signature = valid_signatures[0]
    if len(valid_signature) < 10:
        raise ValueError("gpg VALIDSIG record does not report a primary-key fingerprint")
    _normalize_fingerprint(valid_signature[0], field="reported signing fingerprint")
    return _normalize_fingerprint(valid_signature[9], field="reported primary fingerprint")


def _resolve_primary_fingerprint(key_id: str) -> str:
    if not isinstance(key_id, str) or not key_id:
        raise ValueError("recorded key_id must be a non-empty string")
    try:
        proc = subprocess.run(
            [
                "gpg",
                "--batch",
                "--no-tty",
                "--with-colons",
                "--fixed-list-mode",
                "--with-fingerprint",
                "--list-keys",
                "--",
                key_id,
            ],
            capture_output=True,
            check=False,
            timeout=_GPG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"gpg key resolution timed out after {_GPG_TIMEOUT_SECONDS} seconds") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise ValueError(f"recorded key_id {key_id!r} could not be resolved by gpg: {stderr.strip()}")

    primary_fingerprints: set[str] = set()
    awaiting_primary_fingerprint = False
    for raw_line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        fields = raw_line.split(":")
        record_type = fields[0] if fields else ""
        if record_type == "pub":
            awaiting_primary_fingerprint = True
            continue
        if record_type in {"sub", "ssb"}:
            awaiting_primary_fingerprint = False
            continue
        if record_type == "fpr" and awaiting_primary_fingerprint:
            if len(fields) <= 9:
                raise ValueError("gpg primary-key fingerprint record is malformed")
            primary_fingerprints.add(_normalize_fingerprint(fields[9], field="resolved primary fingerprint"))
            awaiting_primary_fingerprint = False

    if len(primary_fingerprints) != 1:
        raise ValueError(
            f"recorded key_id {key_id!r} must resolve to exactly one primary OpenPGP fingerprint; "
            f"found {len(primary_fingerprints)}"
        )
    return next(iter(primary_fingerprints))


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


def gpg_verify_clearsign(
    signature_text: str,
    *,
    expected_payload: bytes,
    key_id: str,
) -> None:
    """Verify one clearsign over exact bytes by the recorded primary key."""
    if not isinstance(signature_text, str) or not signature_text:
        raise ValueError("clearsign signature must be a non-empty string")
    if not isinstance(expected_payload, bytes):
        raise ValueError("expected clearsign payload must be bytes")
    with tempfile.TemporaryDirectory(prefix="tp-gpg-clearsign-") as tmp_dir:
        cleartext_path = Path(tmp_dir) / "cleartext.bin"
        try:
            proc = subprocess.run(
                [
                    "gpg",
                    "--batch",
                    "--no-tty",
                    "--status-fd",
                    "1",
                    "--output",
                    str(cleartext_path),
                    "--verify",
                ],
                input=signature_text.encode("utf-8"),
                capture_output=True,
                check=False,
                timeout=_GPG_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"gpg verify timed out after {_GPG_TIMEOUT_SECONDS} seconds") from exc
        reported_primary_fingerprint = _verified_primary_fingerprint(proc, label="clearsign", action="verify")
        try:
            verified_cleartext = cleartext_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"gpg verify did not produce readable cleartext: {exc}") from exc

    # GnuPG cleartext verification extracts one framework line ending, including
    # when the signed input did not end with one.  Our canonical JSON
    # preimages never carry a trailing LF, so remove exactly that one framing
    # byte rather than applying lossy text normalization.
    if verified_cleartext != expected_payload + b"\n":
        raise ValueError("gpg clearsign payload does not match the expected canonical preimage bytes")

    resolved_primary_fingerprint = _resolve_primary_fingerprint(key_id)
    if reported_primary_fingerprint != resolved_primary_fingerprint:
        raise ValueError(
            "gpg clearsign primary fingerprint does not match recorded key_id: "
            f"reported {reported_primary_fingerprint}, resolved {resolved_primary_fingerprint}"
        )


def gpg_detached_sign_bytes(payload: bytes, *, key_id: str | None = None, armor: bool = True) -> bytes:
    """Produce a detached signature over payload bytes.

    The default armored output is stable for DSSE envelope storage because it
    can be base64-encoded directly into the ``sig`` field and verified later.
    """
    cmd = ["gpg", "--detach-sign", "--batch", "--yes", "--no-tty", "--output", "-"]
    if armor:
        cmd.append("--armor")
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
        raise ValueError(f"gpg detached signing timed out after {_GPG_TIMEOUT_SECONDS} seconds") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise ValueError(f"gpg detached signing failed: {stderr.strip()}")
    return proc.stdout


def gpg_verify_detached_signature_bytes(signature_bytes: bytes, payload: bytes, *, key_id: str) -> None:
    """Verify one detached signature over exact bytes by the recorded primary key."""
    if not isinstance(signature_bytes, bytes) or not signature_bytes:
        raise ValueError("detached signature must be non-empty bytes")
    if not isinstance(payload, bytes):
        raise ValueError("detached signature payload must be bytes")
    with tempfile.TemporaryDirectory(prefix="tp-gpg-verify-") as tmp_dir:
        signature_path = Path(tmp_dir) / "signature.asc"
        payload_path = Path(tmp_dir) / "payload.bin"
        signature_path.write_bytes(signature_bytes)
        payload_path.write_bytes(payload)
        try:
            proc = subprocess.run(
                [
                    "gpg",
                    "--batch",
                    "--no-tty",
                    "--status-fd",
                    "1",
                    "--verify",
                    str(signature_path),
                    str(payload_path),
                ],
                capture_output=True,
                check=False,
                timeout=_GPG_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"gpg detached verify timed out after {_GPG_TIMEOUT_SECONDS} seconds") from exc

    reported_primary_fingerprint = _verified_primary_fingerprint(
        proc,
        label="detached signature",
        action="detached verify",
    )
    resolved_primary_fingerprint = _resolve_primary_fingerprint(key_id)
    if reported_primary_fingerprint != resolved_primary_fingerprint:
        raise ValueError(
            "gpg detached signature primary fingerprint does not match recorded key_id: "
            f"reported {reported_primary_fingerprint}, resolved {resolved_primary_fingerprint}"
        )
