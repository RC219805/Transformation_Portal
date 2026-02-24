"""Tests for Phase 3.1 detached Merkle signing/verification CLIs."""

from __future__ import annotations

import base64
import functools
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGN_TOOL = PROJECT_ROOT / "tools" / "sign_merkle_roots.py"
VERIFY_TOOL = PROJECT_ROOT / "tools" / "verify_merkle_signature.py"
HAS_CRYPTOGRAPHY = importlib.util.find_spec("cryptography") is not None
pytestmark = [
    pytest.mark.regression,
    pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography is required for Merkle signing CLI tests"),
]


@functools.lru_cache(maxsize=1)
def _load_cryptography_primitives() -> tuple[object, object, object, object, object, object]:
    pytest.importorskip("cryptography", reason="cryptography is required for Merkle signing CLI tests")
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

    return rsa, Ed25519PrivateKey, Encoding, NoEncryption, PrivateFormat, PublicFormat


def _run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_roots(path: Path) -> None:
    path.write_text(
        "{\n" '  "hash_algorithm": "sha256",\n' '  "tree_method_version": "v1",\n' '  "global_root": "6f32b71a"\n' "}\n",
        encoding="utf-8",
    )


def _load_envelope(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_keypair(tmp_path: Path, stem: str) -> tuple[Path, Path]:
    _, Ed25519PrivateKey, Encoding, NoEncryption, PrivateFormat, PublicFormat = _load_cryptography_primitives()
    private_key = Ed25519PrivateKey.generate()

    private_key_path = tmp_path / f"{stem}_private.pem"
    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption(),
        )
    )

    public_key_path = tmp_path / f"{stem}_public.pem"
    public_key_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo,
        )
    )

    return private_key_path, public_key_path


def _write_rsa_public_key(tmp_path: Path, stem: str) -> Path:
    rsa, _, Encoding, _, _, PublicFormat = _load_cryptography_primitives()
    rsa_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rsa_public_key_path = tmp_path / f"{stem}_rsa_public.pem"
    rsa_public_key_path.write_bytes(
        rsa_private_key.public_key().public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return rsa_public_key_path


def _write_rsa_private_key(tmp_path: Path, stem: str) -> Path:
    rsa, _, Encoding, NoEncryption, PrivateFormat, _ = _load_cryptography_primitives()
    rsa_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rsa_private_key_path = tmp_path / f"{stem}_rsa_private.pem"
    rsa_private_key_path.write_bytes(
        rsa_private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption(),
        )
    )
    return rsa_private_key_path


def _sign(roots_path: Path, private_key_path: Path, signature_path: Path) -> subprocess.CompletedProcess[str]:
    return _run_cli(
        [
            sys.executable,
            str(SIGN_TOOL),
            "--roots",
            str(roots_path),
            "--private-key",
            str(private_key_path),
            "--out",
            str(signature_path),
        ]
    )


def _verify(roots_path: Path, signature_path: Path, public_key_path: Path) -> subprocess.CompletedProcess[str]:
    return _run_cli(
        [
            sys.executable,
            str(VERIFY_TOOL),
            "--roots",
            str(roots_path),
            "--signature",
            str(signature_path),
            "--public-key",
            str(public_key_path),
        ]
    )


def _verify_without_site_packages(
    roots_path: Path,
    signature_path: Path,
    public_key_path: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_cli(
        [
            sys.executable,
            "-S",
            str(VERIFY_TOOL),
            "--roots",
            str(roots_path),
            "--signature",
            str(signature_path),
            "--public-key",
            str(public_key_path),
        ]
    )


def test_sign_and_verify_roundtrip_success(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)

    private_key_path, public_key_path = _write_keypair(tmp_path, "primary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr

    envelope_text = signature_path.read_text(encoding="utf-8")
    envelope = _load_envelope(signature_path)
    assert envelope["envelope_version"] == "1"
    assert envelope["signature_algorithm"] == "ed25519"
    assert envelope["artifact_digest_algorithm"] == "sha256"
    assert envelope["signed_artifact"] == "merkle_roots.json"
    assert envelope["signed_artifact_sha256"] == hashlib.sha256(roots_path.read_bytes()).hexdigest()

    expected_text = json.dumps(envelope, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
    assert envelope_text == expected_text

    verify_result = _verify(roots_path, signature_path, public_key_path)
    assert verify_result.returncode == 0, verify_result.stderr
    assert "Signature valid" in verify_result.stdout


def test_sign_is_deterministic_for_same_artifact_and_key(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_a_path = tmp_path / "merkle_roots.a.sig.json"
    signature_b_path = tmp_path / "merkle_roots.b.sig.json"
    _write_roots(roots_path)

    private_key_path, _ = _write_keypair(tmp_path, "primary")

    first_result = _sign(roots_path, private_key_path, signature_a_path)
    second_result = _sign(roots_path, private_key_path, signature_b_path)
    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr

    envelope_a = _load_envelope(signature_a_path)
    envelope_b = _load_envelope(signature_b_path)
    assert envelope_a["signature_base64"] == envelope_b["signature_base64"]


def test_sign_fails_with_missing_roots_file(tmp_path: Path) -> None:
    missing_roots_path = tmp_path / "missing_merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    private_key_path, _ = _write_keypair(tmp_path, "primary")

    sign_result = _sign(missing_roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 4
    assert "Signing failed" in sign_result.stdout


def test_sign_fails_when_roots_filename_is_not_merkle_roots_json(tmp_path: Path) -> None:
    roots_path = tmp_path / "other_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)
    private_key_path, _ = _write_keypair(tmp_path, "primary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 4
    assert "--roots must reference merkle_roots.json" in sign_result.stdout


def test_sign_fails_with_missing_private_key_file(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    missing_private_key_path = tmp_path / "missing_private_key.pem"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)

    sign_result = _sign(roots_path, missing_private_key_path, signature_path)
    assert sign_result.returncode == 4
    assert "Signing failed" in sign_result.stdout


def test_sign_fails_with_non_ed25519_private_key(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)
    rsa_private_key_path = _write_rsa_private_key(tmp_path, "primary")

    sign_result = _sign(roots_path, rsa_private_key_path, signature_path)
    assert sign_result.returncode == 4
    assert "Private key must be Ed25519" in sign_result.stdout


def test_sign_creates_output_parent_directory(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "nested" / "signatures" / "merkle_roots.sig.json"
    _write_roots(roots_path)
    private_key_path, _ = _write_keypair(tmp_path, "primary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr
    assert signature_path.exists()


def test_verify_fails_when_artifact_is_tampered(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)
    private_key_path, public_key_path = _write_keypair(tmp_path, "primary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr

    roots_path.write_bytes(roots_path.read_bytes() + b" ")

    verify_result = _verify(roots_path, signature_path, public_key_path)
    assert verify_result.returncode == 6
    assert "Artifact digest mismatch" in verify_result.stdout


def test_verify_fails_with_missing_roots_file(tmp_path: Path) -> None:
    roots_path = tmp_path / "missing" / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(tmp_path / "merkle_roots.json")
    private_key_path, public_key_path = _write_keypair(tmp_path, "primary")

    sign_result = _sign(tmp_path / "merkle_roots.json", private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr

    verify_result = _verify(roots_path, signature_path, public_key_path)
    assert verify_result.returncode == 6
    assert "Verification failed" in verify_result.stdout


def test_verify_fails_with_missing_public_key_file(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    missing_public_key_path = tmp_path / "missing_public_key.pem"
    _write_roots(roots_path)
    private_key_path, _ = _write_keypair(tmp_path, "primary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr

    verify_result = _verify(roots_path, signature_path, missing_public_key_path)
    assert verify_result.returncode == 6
    assert "Verification failed" in verify_result.stdout


def test_verify_fails_when_signature_is_tampered(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)
    private_key_path, public_key_path = _write_keypair(tmp_path, "primary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr

    envelope = _load_envelope(signature_path)
    raw_signature = bytearray(base64.b64decode(envelope["signature_base64"], validate=True))
    raw_signature[0] ^= 0x01
    envelope["signature_base64"] = base64.b64encode(bytes(raw_signature)).decode("ascii")
    signature_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_result = _verify(roots_path, signature_path, public_key_path)
    assert verify_result.returncode == 5
    assert "Verification failed" in verify_result.stdout


def test_verify_fails_with_wrong_public_key(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)

    private_key_path, _ = _write_keypair(tmp_path, "primary")
    _, wrong_public_key_path = _write_keypair(tmp_path, "secondary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr

    verify_result = _verify(roots_path, signature_path, wrong_public_key_path)
    assert verify_result.returncode == 5
    assert "Verification failed" in verify_result.stdout


def test_verify_fails_with_unsupported_algorithm_field(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)
    private_key_path, public_key_path = _write_keypair(tmp_path, "primary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr

    envelope = _load_envelope(signature_path)
    envelope["signature_algorithm"] = "ed448"
    signature_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_result = _verify(roots_path, signature_path, public_key_path)
    assert verify_result.returncode == 5
    assert "Unsupported signature algorithm" in verify_result.stdout


def test_verify_fails_with_unsupported_envelope_version(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)
    private_key_path, public_key_path = _write_keypair(tmp_path, "primary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr

    envelope = _load_envelope(signature_path)
    envelope["envelope_version"] = "2"
    signature_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_result = _verify(roots_path, signature_path, public_key_path)
    assert verify_result.returncode == 7
    assert "unsupported envelope_version" in verify_result.stdout


def test_verify_fails_with_unsupported_digest_algorithm(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)
    private_key_path, public_key_path = _write_keypair(tmp_path, "primary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr

    envelope = _load_envelope(signature_path)
    envelope["artifact_digest_algorithm"] = "sha512"
    signature_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_result = _verify(roots_path, signature_path, public_key_path)
    assert verify_result.returncode == 5
    assert "Unsupported artifact digest algorithm" in verify_result.stdout


def test_verify_fails_with_non_ed25519_public_key(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)
    private_key_path, _ = _write_keypair(tmp_path, "primary")
    rsa_public_key_path = _write_rsa_public_key(tmp_path, "secondary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr

    verify_result = _verify(roots_path, signature_path, rsa_public_key_path)
    assert verify_result.returncode == 5
    assert "Public key must be Ed25519" in verify_result.stdout


def test_verify_fails_with_malformed_envelope(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)
    _, public_key_path = _write_keypair(tmp_path, "primary")

    signature_path.write_text('{"signature_algorithm": "ed25519"}\n', encoding="utf-8")

    verify_result = _verify(roots_path, signature_path, public_key_path)
    assert verify_result.returncode == 7
    assert "Malformed signature file" in verify_result.stdout


def test_verify_fails_with_invalid_digest_format(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)
    private_key_path, public_key_path = _write_keypair(tmp_path, "primary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr

    envelope = _load_envelope(signature_path)
    envelope["signed_artifact_sha256"] = "A" * 64
    signature_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_result = _verify(roots_path, signature_path, public_key_path)
    assert verify_result.returncode == 7
    assert "signed_artifact_sha256" in verify_result.stdout


def test_verify_fails_when_roots_filename_is_not_merkle_roots_json(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    alternate_roots_path = tmp_path / "other_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)
    private_key_path, public_key_path = _write_keypair(tmp_path, "primary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr

    alternate_roots_path.write_bytes(roots_path.read_bytes())
    verify_result = _verify(alternate_roots_path, signature_path, public_key_path)
    assert verify_result.returncode == 6
    assert "--roots must reference merkle_roots.json" in verify_result.stdout


def test_verify_fails_when_artifact_name_binding_mismatches(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)
    private_key_path, public_key_path = _write_keypair(tmp_path, "primary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr

    envelope = _load_envelope(signature_path)
    envelope["signed_artifact"] = "other_roots.json"
    signature_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_result = _verify(roots_path, signature_path, public_key_path)
    assert verify_result.returncode == 6
    assert "Signed artifact name mismatch" in verify_result.stdout


def test_verify_fails_with_missing_cryptography_dependency(tmp_path: Path) -> None:
    roots_path = tmp_path / "merkle_roots.json"
    signature_path = tmp_path / "merkle_roots.sig.json"
    _write_roots(roots_path)
    private_key_path, public_key_path = _write_keypair(tmp_path, "primary")

    sign_result = _sign(roots_path, private_key_path, signature_path)
    assert sign_result.returncode == 0, sign_result.stderr

    verify_result = _verify_without_site_packages(roots_path, signature_path, public_key_path)
    assert verify_result.returncode == 5
    assert "missing optional dependency 'cryptography'" in verify_result.stdout
    assert "requirements/tools-archive.txt" in verify_result.stdout
