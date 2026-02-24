#!/usr/bin/env python3
"""
Phase 3.1 Detached Merkle Signing
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key

EXIT_SIGN_FAILURE = 4
EXPECTED_ROOTS_FILENAME = "merkle_roots.json"


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        required=True,
        help="Path to merkle_roots.json to sign",
    )
    parser.add_argument(
        "--private-key",
        required=True,
        help="Path to Ed25519 private key in PEM format",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for detached signature JSON",
    )
    args = parser.parse_args()

    roots_path = Path(args.roots)
    key_path = Path(args.private_key)
    out_path = Path(args.out)

    try:
        if roots_path.name != EXPECTED_ROOTS_FILENAME:
            raise ValueError(f"--roots must reference {EXPECTED_ROOTS_FILENAME}")

        artifact_bytes = roots_path.read_bytes()
        artifact_digest = hashlib.sha256(artifact_bytes).hexdigest()

        private_key = load_pem_private_key(key_path.read_bytes(), password=None)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError("Private key must be Ed25519")

        signature = private_key.sign(artifact_bytes)
        envelope = {
            "signature_algorithm": "ed25519",
            "signed_artifact": roots_path.name,
            "signed_artifact_sha256": artifact_digest,
            "signature_base64": base64.b64encode(signature).decode("ascii"),
        }

        serialized = json.dumps(
            envelope,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        ).encode("utf-8")
        atomic_write(out_path, serialized + b"\n")

        print(f"Signature written to {out_path}")
        return 0
    except (OSError, TypeError, ValueError, UnsupportedAlgorithm) as exc:
        print(f"Signing failed: {exc}")
        return EXIT_SIGN_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
