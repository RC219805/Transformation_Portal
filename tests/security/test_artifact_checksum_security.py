"""Security tests for artifact checksum enforcement in download paths."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from transformation_portal.spatial_ai.segmentation.sam2_backend import _compute_file_sha256, _validate_sha256_hex


def _load_script_module(module_name: str, relative_path: str):
    """Load a script module by repository-relative path."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_install_models_verify_checksum_fails_without_expected_digest(tmp_path: Path):
    """Missing/invalid expected checksums must fail closed."""
    install_models = _load_script_module("install_models_script", "scripts/install_models.py")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact-data")

    assert install_models.verify_checksum(artifact, None) is False
    assert install_models.verify_checksum(artifact, "not-a-sha256") is False


def test_download_depth_models_requires_valid_checksum(monkeypatch, tmp_path: Path):
    """Download helper must reject invalid checksums before any network call."""
    module = _load_script_module("download_depth_models_script", "scripts/download_depth_models.py")
    output_path = tmp_path / "weights.bin"
    called = {"value": False}

    def fake_urlretrieve(url, output, reporthook=None):  # noqa: ANN001
        del url, output, reporthook
        called["value"] = True

    monkeypatch.setattr(module.urllib.request, "urlretrieve", fake_urlretrieve)

    ok = module.download_file("https://example.com/weights.bin", output_path, "bad-digest", "weights")
    assert ok is False
    assert called["value"] is False


def test_download_depth_models_verifies_downloaded_artifact(monkeypatch, tmp_path: Path):
    """Downloaded artifacts should only be promoted on checksum match."""
    module = _load_script_module("download_depth_models_script_ok", "scripts/download_depth_models.py")
    output_path = tmp_path / "weights.bin"
    payload = b"trusted-weights"
    expected = hashlib.sha256(payload).hexdigest()

    def fake_urlretrieve(url, output, reporthook=None):  # noqa: ANN001
        del url, reporthook
        Path(output).write_bytes(payload)
        return str(output), None

    monkeypatch.setattr(module.urllib.request, "urlretrieve", fake_urlretrieve)

    ok = module.download_file("https://example.com/weights.bin", output_path, expected, "weights")
    assert ok is True
    assert output_path.read_bytes() == payload


def test_sam2_checksum_helpers_validate_and_hash(tmp_path: Path):
    """SAM2 download helpers should enforce SHA-256 format and hashing behavior."""
    test_file = tmp_path / "checkpoint.pt"
    payload = b"sam2-checkpoint"
    test_file.write_bytes(payload)

    assert _compute_file_sha256(test_file) == hashlib.sha256(payload).hexdigest()
    assert _validate_sha256_hex("A" * 64) == "a" * 64
    with pytest.raises(RuntimeError):
        _validate_sha256_hex("invalid")


def test_download_sam2_script_validates_sha256_format():
    """Standalone SAM2 checkpoint script should reject invalid checksum overrides."""
    module = _load_script_module("download_sam2_checkpoint_script", "scripts/download_sam2_checkpoint.py")

    assert module.validate_sha256_hex("B" * 64) == "b" * 64
    with pytest.raises(ValueError):
        module.validate_sha256_hex("bad")


pytestmark = [
    pytest.mark.unit,
    pytest.mark.regression,
    pytest.mark.security,
]
