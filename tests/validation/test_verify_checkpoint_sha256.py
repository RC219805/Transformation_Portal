"""Regression tests for scripts/validation/verify_checkpoint_sha256.sh."""

from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validation" / "verify_checkpoint_sha256.sh"

# Digest the script registers for depth_pro.pt.
DEPTH_PRO_SHA256 = "3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce"


def _write_fake_shasum(bin_dir: Path, digest: str) -> None:
    """Stand-in for ``shasum -a 256 <file>`` so a registered digest can be produced without the real weights."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / "shasum"
    fake.write_text(f"#!/usr/bin/env bash\nprintf '%s  %s\\n' {shlex.quote(digest)} \"$3\"\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_verifier(workdir: Path, *args: str, fake_shasum_digest: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if fake_shasum_digest is not None:
        _write_fake_shasum(workdir / "fake-bin", fake_shasum_digest)
        env["PATH"] = f"{workdir / 'fake-bin'}{os.pathsep}{env['PATH']}"
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_checkpoint(workdir: Path, name: str, content: bytes = b"weights") -> Path:
    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    path = checkpoint_dir / name
    path.write_bytes(content)
    return path


def test_tampered_registered_checkpoint_fails_with_real_shasum(tmp_path: Path) -> None:
    if shutil.which("shasum") is None:
        pytest.skip("shasum is not installed")
    _write_checkpoint(tmp_path, "depth_pro.pt", b"tampered-weights")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "MISMATCH" in result.stdout
    assert "Mismatched:        1" in result.stdout
    assert "Verified:          0" in result.stdout
    assert "1 checkpoint(s) do not match their registered SHA256" in result.stdout


def test_registered_checkpoint_with_matching_digest_passes(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, "depth_pro.pt")

    result = _run_verifier(tmp_path, fake_shasum_digest=DEPTH_PRO_SHA256)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "VERIFIED" in result.stdout
    assert "Verified:          1" in result.stdout
    assert "Mismatched:        0" in result.stdout


def test_mismatch_fails_even_when_other_checkpoints_are_unregistered(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, "depth_pro.pt")
    _write_checkpoint(tmp_path, "custom_model.pth")

    result = _run_verifier(tmp_path, fake_shasum_digest="0" * 64)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Total checkpoints: 2" in result.stdout
    assert "Mismatched:        1" in result.stdout
    assert "Unknown:           1" in result.stdout


def test_unregistered_checkpoint_warns_by_default_and_fails_in_strict_mode(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, "custom_model.pth")

    default_result = _run_verifier(tmp_path, fake_shasum_digest="0" * 64)
    strict_result = _run_verifier(tmp_path, "--strict", fake_shasum_digest="0" * 64)

    assert default_result.returncode == 0, default_result.stdout + default_result.stderr
    assert "UNKNOWN" in default_result.stdout
    assert "Action Required" in default_result.stdout
    assert strict_result.returncode == 1, strict_result.stdout + strict_result.stderr
    assert "--strict: 1 checkpoint(s) have no registered SHA256" in strict_result.stdout


def test_empty_checkpoint_directory_warns_by_default_and_fails_in_strict_mode(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()

    default_result = _run_verifier(tmp_path)
    strict_result = _run_verifier(tmp_path, "--strict")

    assert default_result.returncode == 0, default_result.stdout + default_result.stderr
    assert "No checkpoints found" in default_result.stdout
    assert strict_result.returncode == 1, strict_result.stdout + strict_result.stderr
    assert "--strict: nothing was verified" in strict_result.stdout


def test_missing_checkpoint_directory_fails(tmp_path: Path) -> None:
    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert "Checkpoint directory not found" in result.stdout


def test_unknown_option_is_a_usage_error(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()

    result = _run_verifier(tmp_path, "--not-an-option")

    assert result.returncode == 2
    assert "Unknown option: --not-an-option" in result.stderr


def test_script_avoids_bash4_only_constructs() -> None:
    """macOS system bash is 3.2: no associative arrays, mapfile/readarray or case-modifying expansions."""
    code_lines = [line for line in SCRIPT_PATH.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")]
    code = "\n".join(code_lines)

    assert "declare -A" not in code
    assert "mapfile" not in code
    assert "readarray" not in code
    assert ",,}" not in code
    assert "^^}" not in code
