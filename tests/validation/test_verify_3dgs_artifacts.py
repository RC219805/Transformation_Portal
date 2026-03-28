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
