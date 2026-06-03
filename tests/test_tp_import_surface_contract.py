"""Contract tests for the public ``tp`` import surface."""

from __future__ import annotations

import importlib
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TP_ROOT = SRC_ROOT / "tp"
SNAKE_CASE_MODULE_RE = re.compile(r"(?:__init__|[a-z][a-z0-9]*(?:_[a-z0-9]+)*)")
GENERATED_SUFFIXES = {".pyc", ".pyo"}

PUBLIC_TP_MODULES = (
    "tp",
    "tp.crypto",
    "tp.crypto.ct_merkle",
    "tp.crypto.merkle",
    "tp.merkle",
    "tp.phase4",
    "tp.phase4.canonicalize_capture_metadata",
    "tp.phase4.exceptions",
    "tp.phase4.hash_capture_metadata",
    "tp.phase4.provenance_capture",
    "tp.phase4.schema_validation",
    "tp.phase4.types",
    "tp.phase4.validation_helpers",
    "tp.phase4.verify_phase4_chain",
)


def _tracked_tp_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "src/tp"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def test_tp_remains_separate_public_import_surface() -> None:
    assert TP_ROOT.is_dir()
    assert (SRC_ROOT / "transformation_portal").is_dir()

    tp_module = importlib.import_module("tp")
    portal_module = importlib.import_module("transformation_portal")

    assert Path(tp_module.__file__).resolve() == TP_ROOT / "__init__.py"
    assert Path(portal_module.__file__).resolve().parent == SRC_ROOT / "transformation_portal"


def test_public_tp_modules_remain_importable() -> None:
    for module_name in PUBLIC_TP_MODULES:
        assert importlib.import_module(module_name)


def test_tp_merkle_reexport_remains_backward_compatible() -> None:
    from tp.crypto.merkle import merkle_root_sha256
    from tp.merkle import merkle_root_sha256 as reexported_merkle_root_sha256

    assert reexported_merkle_root_sha256 is merkle_root_sha256


def test_tracked_tp_files_are_source_only_and_consistently_named() -> None:
    tracked_files = _tracked_tp_files()

    assert tracked_files
    assert tracked_files == sorted(tracked_files)

    generated_files = [path for path in tracked_files if "__pycache__" in path.parts or path.suffix in GENERATED_SUFFIXES]
    assert generated_files == []

    non_python_files = [path for path in tracked_files if path.suffix != ".py"]
    assert non_python_files == []

    invalid_module_names = [path for path in tracked_files if not SNAKE_CASE_MODULE_RE.fullmatch(path.stem)]
    assert invalid_module_names == []

    package_dirs = {part for path in tracked_files for part in path.relative_to("src/tp").parts[:-1]}
    invalid_package_dirs = [dirname for dirname in sorted(package_dirs) if not SNAKE_CASE_MODULE_RE.fullmatch(dirname)]
    assert invalid_package_dirs == []
