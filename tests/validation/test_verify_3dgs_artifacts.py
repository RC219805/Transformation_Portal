from __future__ import annotations

import importlib.util
from pathlib import Path
from textwrap import dedent

import pytest

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validation" / "verify_3dgs_artifacts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_3dgs_artifacts", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFY_3DGS = _load_module()

VALID_MANIFEST = """
artifact_attestation:
  gaussian_splatting:
    backend: inria_graphdeco
    source_type: direct_checkpoint
    source_url: "https://example.com/gaussian_splatting_base.pt"
    source_commit_or_tag: "v1.0.0"
    artifacts:
      - filename: "gaussian_splatting_base.pt"
        sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        filesize_bytes: 1
    verification:
      method: sha256_only
"""


def _write_manifest(tmp_path: Path, content: str) -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(dedent(content), encoding="utf-8")
    return manifest_path


@pytest.mark.parametrize("content", ["", "- just\n- a\n- list\n"])
def test_load_manifest_rejects_non_mapping_root(tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str) -> None:
    manifest_path = _write_manifest(tmp_path, content)

    assert VERIFY_3DGS.load_manifest(manifest_path) is None

    captured = capsys.readouterr()
    assert "Manifest root must be a mapping/dict" in captured.out


def test_main_fails_when_check_files_checkpoint_dir_is_missing(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, VALID_MANIFEST)
    checkpoint_dir = tmp_path / "missing-checkpoints"

    exit_code = VERIFY_3DGS.main(
        [
            "--manifest",
            str(manifest_path),
            "--check-files",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--quiet",
        ]
    )

    assert exit_code == 2


def test_main_fails_when_expected_checkpoint_file_is_missing(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, VALID_MANIFEST)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    exit_code = VERIFY_3DGS.main(
        [
            "--manifest",
            str(manifest_path),
            "--check-files",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--quiet",
        ]
    )

    assert exit_code == 2


def test_source_only_attestation_with_pinned_commit_produces_no_errors_or_warnings() -> None:
    """source_type: git_release + empty artifacts + verification.method:
    source_commit is the canonical shape for upstreams that distribute source
    rather than weights (e.g., graphdeco-inria/gaussian-splatting). With the
    commit pinned, the entry must produce no errors and no warnings."""
    entry = {
        "backend": "inria_graphdeco",
        "source_type": "git_release",
        "source_url": "https://github.com/graphdeco-inria/gaussian-splatting",
        "source_commit_or_tag": "a" * 40,  # pinned 40-hex commit
        "artifacts": [],
        "verification": {"method": "source_commit", "required": True},
    }
    errors, warnings = VERIFY_3DGS.verify_attestation_entry("gaussian_splatting", entry)
    assert errors == []
    assert warnings == []


def test_source_only_attestation_with_pending_commit_warns_but_does_not_error() -> None:
    """Same shape as above but with a pending commit — should warn (pending
    attestation), not error."""
    entry = {
        "backend": "inria_graphdeco",
        "source_type": "git_release",
        "source_url": "https://github.com/graphdeco-inria/gaussian-splatting",
        "source_commit_or_tag": "PENDING_VERIFICATION",
        "artifacts": [],
        "verification": {"method": "source_commit"},
    }
    errors, warnings = VERIFY_3DGS.verify_attestation_entry("gaussian_splatting", entry)
    assert errors == []
    assert any("source_commit_or_tag is pending" in w for w in warnings)


def test_direct_checkpoint_with_empty_artifacts_still_warns() -> None:
    """Empty artifacts on a non-source-only (direct_checkpoint) attestation
    must continue to warn — the previous behaviour for binary-bearing upstreams
    is unchanged."""
    entry = {
        "backend": "inria_graphdeco",
        "source_type": "direct_checkpoint",
        "source_url": "https://example.com/weights.pt",
        "source_commit_or_tag": "v1.0.0",
        "artifacts": [],
        "verification": {"method": "sha256+source_commit"},
    }
    errors, warnings = VERIFY_3DGS.verify_attestation_entry("gaussian_splatting", entry)
    assert errors == []
    assert any("'artifacts' list is empty" in w for w in warnings)


def test_unknown_verification_method_is_rejected() -> None:
    """Methods outside the valid set must still error — the relaxation only
    adds `source_commit`, it doesn't open the gate to arbitrary strings."""
    entry = {
        "backend": "inria_graphdeco",
        "source_type": "git_release",
        "source_url": "https://github.com/example/repo",
        "source_commit_or_tag": "a" * 40,
        "artifacts": [],
        "verification": {"method": "trust_me_bro"},
    }
    errors, _ = VERIFY_3DGS.verify_attestation_entry("gaussian_splatting", entry)
    assert any("verification.method must be one of" in e and "trust_me_bro" in e for e in errors)
