#!/usr/bin/env python3
"""Deterministic GPG subprocess fixture for attestation CLI tests."""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

_DEFAULT_FINGERPRINT = "A" * 40


def _option_value(args: list[str], option: str) -> str:
    try:
        return args[args.index(option) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"missing {option} value") from exc


def _fingerprint_record(fingerprint: str) -> str:
    return ":".join(["fpr", *("" for _ in range(8)), fingerprint, ""])


def _embedded_payload(signature: bytes) -> bytes:
    marker = b"X-TP-Fake-Payload: "
    for line in signature.splitlines():
        if line.startswith(marker):
            return base64.b64decode(line[len(marker) :], validate=True)
    raise SystemExit("fake clearsign has no embedded payload")


def main() -> int:
    args = sys.argv[1:]
    stdin = sys.stdin.buffer.read()
    primary = os.environ.get("TP_FAKE_GPG_PRIMARY_FINGERPRINT", _DEFAULT_FINGERPRINT)
    signing = os.environ.get("TP_FAKE_GPG_SIGNING_FINGERPRINT", primary)
    resolved = os.environ.get("TP_FAKE_GPG_RESOLVED_FINGERPRINT", primary)

    if "--list-keys" in args:
        if "--with-fingerprint" not in args or "--with-colons" not in args:
            print("fake key resolution requires machine-readable fingerprints", file=sys.stderr)
            return 2
        resolve_mode = os.environ.get("TP_FAKE_GPG_RESOLVE_MODE", "valid")
        if resolve_mode == "error":
            print("fake key not found", file=sys.stderr)
            return 2
        if resolve_mode == "missing":
            return 0
        print("pub::::::::::")
        print(_fingerprint_record(resolved))
        if resolve_mode == "ambiguous":
            print("pub::::::::::")
            print(_fingerprint_record("B" * 40))
        return 0

    if "--clearsign" in args:
        encoded = base64.b64encode(stdin).decode("ascii")
        sys.stdout.write(
            "-----BEGIN PGP SIGNED MESSAGE-----\n"
            "Hash: SHA256\n"
            f"X-TP-Fake-Payload: {encoded}\n\n"
            "payload\n"
            "-----BEGIN PGP SIGNATURE-----\n"
            "fake-clearsign\n"
            "-----END PGP SIGNATURE-----\n"
        )
        return 0

    if "--detach-sign" in args:
        sys.stdout.write("-----BEGIN PGP SIGNATURE-----\n" "fake-detached\n" "-----END PGP SIGNATURE-----\n")
        return 0

    if "--verify" in args and "--output" in args:
        if "--status-fd" not in args:
            print("fake clearsign verification requires status records", file=sys.stderr)
            return 2
        payload_override = os.environ.get("TP_FAKE_GPG_VERIFIED_PAYLOAD_B64")
        payload = (
            base64.b64decode(payload_override, validate=True) if payload_override is not None else _embedded_payload(stdin)
        )
        Path(_option_value(args, "--output")).write_bytes(payload + b"\n")

        status_mode = os.environ.get("TP_FAKE_GPG_STATUS_MODE", "valid")
        status = f"[GNUPG:] VALIDSIG {signing} 2026-08-28 1787940000 0 4 0 22 8 01 {primary}\n"
        if status_mode == "valid":
            sys.stdout.write(status)
        elif status_mode == "ambiguous":
            sys.stdout.write(status + status)
        elif status_mode == "malformed":
            sys.stdout.write(f"[GNUPG:] VALIDSIG {signing}\n")
        elif status_mode != "missing":
            raise SystemExit(f"unsupported TP_FAKE_GPG_STATUS_MODE: {status_mode}")
        return 0

    if "--verify" in args:
        return 0

    print(f"unsupported fake gpg invocation: {' '.join(args)}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
