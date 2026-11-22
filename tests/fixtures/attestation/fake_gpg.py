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


def _verify_status(*, signing: str, primary: str, sig_class: str) -> tuple[str, int]:
    """Return the ``--status-fd`` text and exit code of one verification.

    Each mode replays the record sequence and exit code GnuPG 2.4 produces for
    that verdict.  A revoked or expired signing key still exits 0 and still
    reports VALIDSIG; only the verdict keyword differs from a good signature.
    """
    key_id = signing[-16:]
    uid = "Fake Signer <fake@example.test>"
    opening = f"[GNUPG:] NEWSIG\n[GNUPG:] KEY_CONSIDERED {primary} 0\n"
    sig_id = "[GNUPG:] SIG_ID ZmFrZS1zaWduYXR1cmUtaWQ 2026-08-28 1787940000\n"
    valid = f"[GNUPG:] VALIDSIG {signing} 2026-08-28 1787940000 0 4 0 22 8 {sig_class} {primary}\n"
    good = opening + sig_id + f"[GNUPG:] GOODSIG {key_id} {uid}\n" + valid + "[GNUPG:] TRUST_UNDEFINED 0 pgp\n"
    failure = "[GNUPG:] FAILURE gpg-exit 33554433\n"
    sequences = {
        "valid": (good, 0),
        "ambiguous": (good + good, 0),
        "missing": ("", 0),
        "malformed": (opening + sig_id + f"[GNUPG:] GOODSIG {key_id} {uid}\n[GNUPG:] VALIDSIG {signing}\n", 0),
        # An expired subkey that did not make the signature: still a good signature.
        "expired_unused_subkey": ("[GNUPG:] KEYEXPIRED 1787930000\n" + good, 0),
        "revoked_key": (
            opening + sig_id + f"[GNUPG:] REVKEYSIG {key_id} {uid}\n" + valid + "[GNUPG:] KEYREVOKED\n"
            f"[GNUPG:] KEY_CONSIDERED {primary} 0\n[GNUPG:] TRUST_ULTIMATE 0 pgp\n",
            0,
        ),
        "expired_key": (
            opening + "[GNUPG:] KEYEXPIRED 1787930000\n" + sig_id + f"[GNUPG:] EXPKEYSIG {key_id} {uid}\n" + valid,
            0,
        ),
        "expired_signature": (
            opening + sig_id + f"[GNUPG:] EXPSIG {key_id} {uid}\n" + valid + "[GNUPG:] TRUST_ULTIMATE 0 pgp\n" + failure,
            1,
        ),
        "bad_signature": (opening + f"[GNUPG:] BADSIG {key_id} {uid}\n" + failure, 1),
        "unknown_key": (
            f"[GNUPG:] NEWSIG\n[GNUPG:] ERRSIG {key_id} 22 8 {sig_class} 1787940000 9 {signing}\n"
            f"[GNUPG:] NO_PUBKEY {key_id}\n" + failure,
            2,
        ),
        # A verdict keyword this repository does not know must never be accepted.
        "unrecognised_verdict": (opening + sig_id + f"[GNUPG:] FUTUREKEYSIG {key_id} {uid}\n" + valid, 0),
        "validsig_without_verdict": (opening + sig_id + valid, 0),
    }
    status_mode = os.environ.get("TP_FAKE_GPG_STATUS_MODE", "valid")
    if status_mode not in sequences:
        raise SystemExit(f"unsupported TP_FAKE_GPG_STATUS_MODE: {status_mode}")
    status, exit_code = sequences[status_mode]
    # Lets a test assert that acceptance never rests on the exit code alone.
    exit_override = os.environ.get("TP_FAKE_GPG_VERIFY_EXIT_CODE")
    return status, (int(exit_override) if exit_override is not None else exit_code)


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

        status, exit_code = _verify_status(signing=signing, primary=primary, sig_class="01")
        sys.stdout.write(status)
        return exit_code

    if "--verify" in args:
        if "--status-fd" not in args:
            print("fake detached verification requires status records", file=sys.stderr)
            return 2
        status, exit_code = _verify_status(signing=signing, primary=primary, sig_class="00")
        sys.stdout.write(status)
        return exit_code

    print(f"unsupported fake gpg invocation: {' '.join(args)}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
