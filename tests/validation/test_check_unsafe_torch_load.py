from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "scripts" / "validation" / "check_unsafe_torch_load.py"
SPEC = importlib.util.spec_from_file_location("check_unsafe_torch_load", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def test_find_python_files_skips_repo_local_runtime_envs(tmp_path: Path) -> None:
    unsafe_runtime_file = tmp_path / ".runtime" / "Depth-Anything-3" / "loader.py"
    unsafe_runtime_file.parent.mkdir(parents=True)
    unsafe_runtime_file.write_text("import torch\ntorch.load('model.pt')\n", encoding="utf-8")
    unsafe_venv_file = tmp_path / ".venv-da3" / "site-packages" / "loader.py"
    unsafe_venv_file.parent.mkdir(parents=True)
    unsafe_venv_file.write_text("import torch\ntorch.load('model.pt')\n", encoding="utf-8")
    repo_file = tmp_path / "src" / "loader.py"
    repo_file.parent.mkdir()
    repo_file.write_text("import torch\ntorch.load('model.pt', weights_only=True)\n", encoding="utf-8")

    found = {path.relative_to(tmp_path) for path in checker.find_python_files(tmp_path)}

    assert found == {Path("src/loader.py")}


def test_platform_security_profile_test_is_allowed() -> None:
    path = PROJECT_ROOT / "tests" / "security" / "test_platform_security_profile.py"

    assert checker.is_allowed_file(path, PROJECT_ROOT)
