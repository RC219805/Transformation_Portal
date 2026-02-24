#!/usr/bin/env python3
"""
Phase 3.1 Detached Merkle Signature Verification
"""

import argparse
import base64
import binascii
import hashlib
import json
import re
from pathlib import Path

EXIT_INVALID_SIG = 5
EXIT_ARTIFACT_MISMATCH = 6
EXIT_MALFORMED = 7
EXPECTED_ROOTS_FILENAME = "merkle_roots.json"
ENVELOPE_VERSION = "1"
ARTIFACT_DIGEST_ALGORITHM = "sha256"
CRYPTOGRAPHY_INSTALL_HINT = (
    "Install dependencies with `pip install -r requirements/tools-archive.txt` "
    "or `pip install transformation-portal[archive-signing]`."
)
SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")


def _load_crypto_primitives() -> tuple[object, object, object, object] | None:
    try:
        from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
    except ImportError:
        return None
    return InvalidSignature, UnsupportedAlgorithm, Ed25519PublicKey, load_pem_public_key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        required=True,
        help="Path to merkle_roots.json to verify",
    )
    parser.add_argument(
        "--signature",
        required=True,
        help="Path to detached signature JSON file",
    )
    parser.add_argument(
        "--public-key",
        required=True,
        help="Path to Ed25519 public key in PEM format",
    )
    args = parser.parse_args()

    roots_path = Path(args.roots)
    sig_path = Path(args.signature)
    pub_path = Path(args.public_key)

    if roots_path.name != EXPECTED_ROOTS_FILENAME:
        print(f"Artifact mismatch: --roots must reference {EXPECTED_ROOTS_FILENAME}")
        return EXIT_ARTIFACT_MISMATCH

    try:
        envelope = json.loads(sig_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"Malformed signature file: {exc}")
        return EXIT_MALFORMED

    required_fields = {
        "envelope_version",
        "signature_algorithm",
        "artifact_digest_algorithm",
        "signed_artifact",
        "signed_artifact_sha256",
        "signature_base64",
    }
    if not isinstance(envelope, dict) or not required_fields.issubset(envelope):
        print("Malformed signature file")
        return EXIT_MALFORMED

    crypto_primitives = _load_crypto_primitives()
    if crypto_primitives is None:
        print(
            "Verification failed: missing optional dependency 'cryptography'.",
            CRYPTOGRAPHY_INSTALL_HINT,
        )
        return EXIT_INVALID_SIG
    InvalidSignature, UnsupportedAlgorithm, Ed25519PublicKey, load_pem_public_key = crypto_primitives

    try:
        if envelope["envelope_version"] != ENVELOPE_VERSION:
            print(f"Malformed signature file: unsupported envelope_version {envelope['envelope_version']!r}")
            return EXIT_MALFORMED

        if envelope["signature_algorithm"] != "ed25519":
            print("Unsupported signature algorithm")
            return EXIT_INVALID_SIG

        if envelope["artifact_digest_algorithm"] != ARTIFACT_DIGEST_ALGORITHM:
            print(f"Unsupported artifact digest algorithm: {envelope['artifact_digest_algorithm']!r}")
            return EXIT_INVALID_SIG

        if envelope["signed_artifact"] != EXPECTED_ROOTS_FILENAME:
            print("Signed artifact name mismatch")
            return EXIT_ARTIFACT_MISMATCH

        digest_hex = envelope["signed_artifact_sha256"]
        if not isinstance(digest_hex, str) or SHA256_HEX_RE.fullmatch(digest_hex) is None:
            print("Malformed signature file: signed_artifact_sha256 must be 64 lowercase hex characters")
            return EXIT_MALFORMED

        artifact_bytes = roots_path.read_bytes()
        computed_digest = hashlib.sha256(artifact_bytes).hexdigest()
        if computed_digest != digest_hex:
            print("Artifact digest mismatch")
            return EXIT_ARTIFACT_MISMATCH

        signature = base64.b64decode(envelope["signature_base64"], validate=True)

        public_key = load_pem_public_key(pub_path.read_bytes())
        if not isinstance(public_key, Ed25519PublicKey):
            print("Public key must be Ed25519")
            return EXIT_INVALID_SIG

        public_key.verify(signature, artifact_bytes)

        print("Signature valid")
        return 0
    except (KeyError, TypeError, binascii.Error) as exc:
        print(f"Malformed signature file: {exc}")
        return EXIT_MALFORMED
    except OSError as exc:
        print(f"Verification failed: {exc}")
        return EXIT_ARTIFACT_MISMATCH
    except (ValueError, InvalidSignature, UnsupportedAlgorithm) as exc:
        print(f"Verification failed: {exc}")
        return EXIT_INVALID_SIG


if __name__ == "__main__":
    raise SystemExit(main())
