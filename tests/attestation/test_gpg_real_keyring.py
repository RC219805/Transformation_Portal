"""Real-GnuPG regressions for attestation signature verdicts and signer binding.

GnuPG exits 0 and reports VALIDSIG for a signature made by a revoked or an
expired key.  These tests build throw-away keys in isolated GNUPGHOME
directories (no network, no pre-existing keyring) and prove that the
verification helpers read the verdict and bind the signer.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from transformation_portal.attestation import gpg as gpg_module
from transformation_portal.attestation.dsse import (
    DSSE_IN_TOTO_JSON_PAYLOAD_TYPE,
    build_dsse_envelope,
    decode_dsse_payload,
    decode_dsse_signature_bytes,
    pre_auth_encode,
)
from transformation_portal.attestation.gpg import (
    gpg_clearsign_bytes,
    gpg_detached_sign_bytes,
    gpg_verify_clearsign,
    gpg_verify_detached_signature_bytes,
)

pytestmark = [pytest.mark.security, pytest.mark.integration]

_SETUP_TIMEOUT_SECONDS = 120
_STATEMENT = b'{"_type":"https://in-toto.io/Statement/v1","predicate":{},"subject":[]}'
_PAE = pre_auth_encode(DSSE_IN_TOTO_JSON_PAYLOAD_TYPE, _STATEMENT)
_PREIMAGE = b'{"schema":"tp.run_card.attestation.detached.v1.preimage","subject":{}}'


@dataclass(frozen=True)
class _Keyring:
    signer_home: Path  # secret keys; the only home that needs gpg-agent
    verifier_home: Path  # the same public keys, nothing revoked, real clock
    revoked_home: Path  # public keys plus the revocation certificate of ``revoked``
    future_home: Path  # public keys, clock three days ahead of the real one
    empty_home: Path  # no keys at all
    release: str
    other: str
    revoked: str
    short_lived: str  # primary key that expires after one day
    rotating: str  # non-expiring primary key ...
    rotating_subkey: str  # ... with a signing subkey that expires after one day


def _run_gpg(home: Path, *args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ)
    env["GNUPGHOME"] = str(home)
    return subprocess.run(
        ["gpg", "--batch", "--no-tty", *args],
        input=stdin,
        capture_output=True,
        check=False,
        timeout=_SETUP_TIMEOUT_SECONDS,
        env=env,
    )


def _require_gpg(home: Path, *args: str, stdin: bytes | None = None) -> bytes:
    """Run a keyring setup step; skip the module when this GnuPG cannot do it."""
    try:
        proc = _run_gpg(home, *args, stdin=stdin)
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"gpg cannot build a throw-away keyring here: {exc}")
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        pytest.skip(f"gpg cannot build a throw-away keyring here: {detail}")
    return proc.stdout


def _fingerprints(home: Path, selector: str) -> list[str]:
    listing = _require_gpg(home, "--with-colons", "--list-keys", "--", selector).decode("utf-8", errors="replace")
    return re.findall(r"^fpr:+([0-9A-F]{40}):", listing, flags=re.MULTILINE)


def _generate_key(home: Path, uid: str, *, expire: str = "0") -> str:
    _require_gpg(
        home,
        "--pinentry-mode",
        "loopback",
        "--passphrase",
        "",
        "--quick-generate-key",
        uid,
        "ed25519",
        "sign",
        expire,
    )
    return _fingerprints(home, uid)[0]


@pytest.fixture(scope="module")
def keyring() -> Iterator[_Keyring]:
    if shutil.which("gpg") is None:
        pytest.skip("gpg is not installed")
    # gpg-agent listens on Unix sockets inside GNUPGHOME and sun_path caps the
    # whole socket path near 100 bytes, which pytest's nested tmp_path exceeds.
    root = Path(tempfile.mkdtemp(prefix="tpgpg-", dir="/tmp" if Path("/tmp").is_dir() else None))
    homes = {name: root / name for name in ("signer", "verifier", "revoked", "future", "empty")}
    try:
        for home in homes.values():
            home.mkdir(mode=0o700)
        signer = homes["signer"]
        release = _generate_key(signer, "Release Signer <release@example.test>")
        other = _generate_key(signer, "Other Keyring Key <other@example.test>")
        revoked = _generate_key(signer, "Compromised Key <compromised@example.test>")
        short_lived = _generate_key(signer, "Short Lived <short-lived@example.test>", expire="1d")
        rotating = _generate_key(signer, "Rotating Subkeys <rotating@example.test>")
        _require_gpg(
            signer,
            "--pinentry-mode",
            "loopback",
            "--passphrase",
            "",
            "--quick-add-key",
            rotating,
            "ed25519",
            "sign",
            "1d",
        )
        rotating_fingerprints = _fingerprints(signer, rotating)
        if len(rotating_fingerprints) != 2:
            pytest.skip("gpg did not report the expected signing subkey")

        public_keys = _require_gpg(signer, "--export", "--armor")
        for name in ("verifier", "revoked", "future"):
            _require_gpg(homes[name], "--import", stdin=public_keys)

        revocation_path = signer / "openpgp-revocs.d" / f"{revoked}.rev"
        if not revocation_path.is_file():
            pytest.skip("this GnuPG does not pre-generate revocation certificates")
        # GnuPG guards the stored certificate against accidental import with a leading colon.
        revocation = revocation_path.read_text(encoding="utf-8").replace(":-----BEGIN", "-----BEGIN")
        _require_gpg(homes["revoked"], "--import", stdin=revocation.encode("utf-8"))

        (homes["future"] / "gpg.conf").write_text(
            f"faked-system-time {int(time.time()) + 3 * 86400}\n",
            encoding="utf-8",
        )
        yield _Keyring(
            signer_home=signer,
            verifier_home=homes["verifier"],
            revoked_home=homes["revoked"],
            future_home=homes["future"],
            empty_home=homes["empty"],
            release=release,
            other=other,
            revoked=revoked,
            short_lived=short_lived,
            rotating=rotating,
            rotating_subkey=rotating_fingerprints[1],
        )
    finally:
        for home in homes.values():
            subprocess.run(
                ["gpgconf", "--homedir", str(home), "--kill", "all"],
                capture_output=True,
                check=False,
                timeout=_SETUP_TIMEOUT_SECONDS,
            )
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(autouse=True)
def _generous_gpg_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    # The behaviour under test is the verdict, not latency on a shared CI host.
    monkeypatch.setattr(gpg_module, "_GPG_TIMEOUT_SECONDS", 60)


def _sign_detached(monkeypatch: pytest.MonkeyPatch, keyring: _Keyring, key_id: str, payload: bytes = _PAE) -> bytes:
    monkeypatch.setenv("GNUPGHOME", str(keyring.signer_home))
    return gpg_detached_sign_bytes(payload, key_id=key_id)


def _sign_clear(monkeypatch: pytest.MonkeyPatch, keyring: _Keyring, key_id: str) -> str:
    monkeypatch.setenv("GNUPGHOME", str(keyring.signer_home))
    return gpg_clearsign_bytes(_PREIMAGE, key_id=key_id)


def test_honest_signatures_verify_against_the_recorded_key(keyring: _Keyring, monkeypatch: pytest.MonkeyPatch) -> None:
    detached = _sign_detached(monkeypatch, keyring, keyring.release)
    clearsign = _sign_clear(monkeypatch, keyring, keyring.release)

    monkeypatch.setenv("GNUPGHOME", str(keyring.verifier_home))

    assert gpg_verify_detached_signature_bytes(detached, _PAE, key_id=keyring.release) is None
    assert gpg_verify_clearsign(clearsign, expected_payload=_PREIMAGE, key_id=keyring.release) is None


def test_dsse_envelope_signed_by_another_keyring_key_is_rejected(keyring: _Keyring, monkeypatch: pytest.MonkeyPatch) -> None:
    """The envelope names the release key, but a different key in the keyring signed it."""
    signature = _sign_detached(monkeypatch, keyring, keyring.other)
    envelope = build_dsse_envelope(
        payload=_STATEMENT,
        payload_type=DSSE_IN_TOTO_JSON_PAYLOAD_TYPE,
        key_id=keyring.release,
        signature_bytes=signature,
    )

    monkeypatch.setenv("GNUPGHOME", str(keyring.verifier_home))

    with pytest.raises(ValueError, match="primary fingerprint does not match recorded key_id"):
        gpg_verify_detached_signature_bytes(
            decode_dsse_signature_bytes(envelope),
            pre_auth_encode(DSSE_IN_TOTO_JSON_PAYLOAD_TYPE, decode_dsse_payload(envelope)),
            key_id=str(envelope["signatures"][0]["keyid"]),
        )
    # The same signature is fine once the envelope names the key that made it.
    assert gpg_verify_detached_signature_bytes(signature, _PAE, key_id=keyring.other) is None


def test_signatures_by_a_revoked_key_are_rejected(keyring: _Keyring, monkeypatch: pytest.MonkeyPatch) -> None:
    detached = _sign_detached(monkeypatch, keyring, keyring.revoked)
    clearsign = _sign_clear(monkeypatch, keyring, keyring.revoked)

    # Control: the verifier that never imported the revocation accepts both.
    monkeypatch.setenv("GNUPGHOME", str(keyring.verifier_home))
    assert gpg_verify_detached_signature_bytes(detached, _PAE, key_id=keyring.revoked) is None
    assert gpg_verify_clearsign(clearsign, expected_payload=_PREIMAGE, key_id=keyring.revoked) is None

    monkeypatch.setenv("GNUPGHOME", str(keyring.revoked_home))
    with pytest.raises(ValueError, match="the signing key has been revoked"):
        gpg_verify_detached_signature_bytes(detached, _PAE, key_id=keyring.revoked)
    with pytest.raises(ValueError, match="the signing key has been revoked"):
        gpg_verify_clearsign(clearsign, expected_payload=_PREIMAGE, key_id=keyring.revoked)


def test_signatures_by_an_expired_key_are_rejected(keyring: _Keyring, monkeypatch: pytest.MonkeyPatch) -> None:
    detached = _sign_detached(monkeypatch, keyring, keyring.short_lived)
    clearsign = _sign_clear(monkeypatch, keyring, keyring.short_lived)

    # Control: before the key expires both signatures are accepted.
    monkeypatch.setenv("GNUPGHOME", str(keyring.verifier_home))
    assert gpg_verify_detached_signature_bytes(detached, _PAE, key_id=keyring.short_lived) is None
    assert gpg_verify_clearsign(clearsign, expected_payload=_PREIMAGE, key_id=keyring.short_lived) is None

    monkeypatch.setenv("GNUPGHOME", str(keyring.future_home))
    with pytest.raises(ValueError, match="the signing key has expired"):
        gpg_verify_detached_signature_bytes(detached, _PAE, key_id=keyring.short_lived)
    with pytest.raises(ValueError, match="the signing key has expired"):
        gpg_verify_clearsign(clearsign, expected_payload=_PREIMAGE, key_id=keyring.short_lived)


def test_expired_subkey_only_rejects_signatures_it_made(keyring: _Keyring, monkeypatch: pytest.MonkeyPatch) -> None:
    """GnuPG reports KEYEXPIRED for any expired subkey; only EXPKEYSIG names the signing key."""
    by_primary = _sign_detached(monkeypatch, keyring, f"{keyring.rotating}!")
    by_subkey = _sign_detached(monkeypatch, keyring, f"{keyring.rotating_subkey}!")

    monkeypatch.setenv("GNUPGHOME", str(keyring.verifier_home))
    # A signing subkey binds to the recorded primary key while it is valid.
    assert gpg_verify_detached_signature_bytes(by_subkey, _PAE, key_id=keyring.rotating) is None

    monkeypatch.setenv("GNUPGHOME", str(keyring.future_home))
    assert gpg_verify_detached_signature_bytes(by_primary, _PAE, key_id=keyring.rotating) is None
    with pytest.raises(ValueError, match="the signing key has expired"):
        gpg_verify_detached_signature_bytes(by_subkey, _PAE, key_id=keyring.rotating)


def test_expired_signature_is_rejected(keyring: _Keyring, monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _run_gpg(
        keyring.signer_home,
        "--default-sig-expire",
        "1d",
        "--detach-sign",
        "--armor",
        "--local-user",
        keyring.release,
        "--output",
        "-",
        stdin=_PAE,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")

    monkeypatch.setenv("GNUPGHOME", str(keyring.verifier_home))
    assert gpg_verify_detached_signature_bytes(proc.stdout, _PAE, key_id=keyring.release) is None

    monkeypatch.setenv("GNUPGHOME", str(keyring.future_home))
    with pytest.raises(ValueError, match="the signature has expired"):
        gpg_verify_detached_signature_bytes(proc.stdout, _PAE, key_id=keyring.release)


def test_tampered_payload_and_unknown_signer_are_rejected(keyring: _Keyring, monkeypatch: pytest.MonkeyPatch) -> None:
    detached = _sign_detached(monkeypatch, keyring, keyring.release)

    monkeypatch.setenv("GNUPGHOME", str(keyring.verifier_home))
    with pytest.raises(ValueError, match="the signature does not match the signed bytes"):
        gpg_verify_detached_signature_bytes(detached, _PAE + b"x", key_id=keyring.release)

    monkeypatch.setenv("GNUPGHOME", str(keyring.empty_home))
    with pytest.raises(ValueError, match="the signature could not be checked"):
        gpg_verify_detached_signature_bytes(detached, _PAE, key_id=keyring.release)


def test_two_good_signatures_in_one_blob_are_rejected(keyring: _Keyring, monkeypatch: pytest.MonkeyPatch) -> None:
    """One recorded keyid cannot vouch for a blob that carries two signers."""
    by_release = _sign_detached(monkeypatch, keyring, keyring.release)
    by_other = _sign_detached(monkeypatch, keyring, keyring.other)

    monkeypatch.setenv("GNUPGHOME", str(keyring.verifier_home))
    with pytest.raises(ValueError, match="exactly one VALIDSIG record; found 2"):
        gpg_verify_detached_signature_bytes(by_release + by_other, _PAE, key_id=keyring.release)
