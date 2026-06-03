from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_repo_script(repo_root: Path) -> Path:
    source = REPO_ROOT / "scripts" / "setup" / "pre-commit-check.sh"
    destination = repo_root / "scripts" / "setup" / "pre-commit-check.sh"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return destination


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def _init_repo(repo_root: Path) -> None:
    result = _run(["git", "init", "-q"], repo_root)
    assert result.returncode == 0, result.stdout + result.stderr


def _write(path: Path, content: str = "test\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_missing_root_allowlist_behaves_like_zero_legacy_entries(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "README.md")
    _write(repo_root / "legacy.md")
    add_result = _run(["git", "add", "README.md", "legacy.md", "scripts/setup/pre-commit-check.sh"], repo_root)
    assert add_result.returncode == 0, add_result.stdout + add_result.stderr

    result = _run(
        [
            "bash",
            "scripts/setup/pre-commit-check.sh",
            "--all",
            "--legacy-allowlist",
            "scripts/governance/does-not-exist.txt",
        ],
        repo_root,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "legacy.md" in result.stdout
    assert "grandfathered" not in result.stdout


def test_cloudflare_workers_build_root_shim_files_are_allowed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "package.json", '{"private": true}\n')
    _write(repo_root / "package-lock.json", '{"lockfileVersion": 3}\n')
    _write(repo_root / "wrangler.jsonc", '{"name": "transformationportal"}\n')
    add_result = _run(
        [
            "git",
            "add",
            "package.json",
            "package-lock.json",
            "wrangler.jsonc",
            "scripts/setup/pre-commit-check.sh",
        ],
        repo_root,
    )
    assert add_result.returncode == 0, add_result.stdout + add_result.stderr

    result = _run(["bash", "scripts/setup/pre-commit-check.sh", "--staged"], repo_root)

    assert result.returncode == 0, result.stdout + result.stderr


def test_known_root_requirement_shims_are_allowed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    for path in ("requirements.txt", "requirements-ci.txt", "requirements-dev.txt", "requirements-lint.txt"):
        _write(repo_root / path, "# governed root requirement shim\n")

    add_result = _run(
        [
            "git",
            "add",
            "requirements.txt",
            "requirements-ci.txt",
            "requirements-dev.txt",
            "requirements-lint.txt",
            "scripts/setup/pre-commit-check.sh",
        ],
        repo_root,
    )
    assert add_result.returncode == 0, add_result.stdout + add_result.stderr

    result = _run(["bash", "scripts/setup/pre-commit-check.sh", "--staged"], repo_root)

    assert result.returncode == 0, result.stdout + result.stderr


def test_governed_root_dotfile_configs_are_allowed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    governed_dotfiles = [
        ".dockerignore",
        ".env.example",
        ".gitattributes",
        ".git-blame-ignore-revs",
        ".gitignore",
        ".gitleaks.toml",
        ".pre-commit-config.yaml",
        ".pylintrc",
    ]
    for path in governed_dotfiles:
        _write(repo_root / path)

    add_result = _run(["git", "add", *governed_dotfiles, "scripts/setup/pre-commit-check.sh"], repo_root)
    assert add_result.returncode == 0, add_result.stdout + add_result.stderr

    result = _run(["bash", "scripts/setup/pre-commit-check.sh", "--staged"], repo_root)

    assert result.returncode == 0, result.stdout + result.stderr


def test_broad_root_allowlist_patterns_do_not_allow_new_clutter(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    unexpected_root_files = [
        ".envrc",
        ".gitlab-ci.yml",
        "requirements-local.txt",
    ]
    for path in unexpected_root_files:
        _write(repo_root / path)

    add_result = _run(["git", "add", *unexpected_root_files, "scripts/setup/pre-commit-check.sh"], repo_root)
    assert add_result.returncode == 0, add_result.stdout + add_result.stderr

    result = _run(["bash", "scripts/setup/pre-commit-check.sh", "--staged"], repo_root)

    assert result.returncode == 1, result.stdout + result.stderr
    for path in unexpected_root_files:
        assert path in result.stdout
    assert "an approved project subdirectory" in result.stdout


def test_retired_root_config_files_are_rejected(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    retired_root_configs = [
        "setup.py",
        "setup.cfg",
        "requirements-test.txt",
        "Pipfile",
        "Pipfile.lock",
        "poetry.lock",
        "pytest.ini",
        "tox.ini",
        ".coveragerc",
        ".flake8",
        "docker-compose.yaml",
        "PKG-INFO",
        "MANIFEST.in",
        "__init__.py",
    ]
    for path in retired_root_configs:
        _write(repo_root / path)

    add_result = _run(["git", "add", *retired_root_configs, "scripts/setup/pre-commit-check.sh"], repo_root)
    assert add_result.returncode == 0, add_result.stdout + add_result.stderr

    result = _run(["bash", "scripts/setup/pre-commit-check.sh", "--staged"], repo_root)

    assert result.returncode == 1, result.stdout + result.stderr
    for path in retired_root_configs:
        assert path in result.stdout


def test_allowed_top_level_directories_are_accepted(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "assets" / "textures" / "board_materials" / "plaster.png")
    _write(repo_root / "public" / "portal-assets" / "portal.js")
    _write(repo_root / "src" / "transformation_portal" / "__init__.py")
    add_result = _run(
        [
            "git",
            "add",
            "assets/textures/board_materials/plaster.png",
            "public/portal-assets/portal.js",
            "src/transformation_portal/__init__.py",
            "scripts/setup/pre-commit-check.sh",
        ],
        repo_root,
    )
    assert add_result.returncode == 0, add_result.stdout + add_result.stderr

    result = _run(["bash", "scripts/setup/pre-commit-check.sh", "--all"], repo_root)

    assert result.returncode == 0, result.stdout + result.stderr


def test_retired_root_directories_are_rejected(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    retired_paths = [
        "dashboard/.gitkeep",
        "data/luts/location/_location_notes.md",
        "linear_ingest_demo/manifest.json",
        "projects/.gitkeep",
        "test_sky_fix/sky_fix_comparison.jpg",
        "textures/board_materials/plaster.png",
    ]
    for path in retired_paths:
        _write(repo_root / path)

    add_result = _run(["git", "add", *retired_paths, "scripts/setup/pre-commit-check.sh"], repo_root)
    assert add_result.returncode == 0, add_result.stdout + add_result.stderr

    result = _run(["bash", "scripts/setup/pre-commit-check.sh", "--staged"], repo_root)

    assert result.returncode == 1, result.stdout + result.stderr
    for path in retired_paths:
        assert path in result.stdout
    assert "assets/luts/" in result.stdout
    assert "assets/textures/" in result.stdout
    assert "output/examples/linear_ingest_demo/" in result.stdout
    assert "output/materials_v3/" in result.stdout


def test_productivity_root_bundle_paths_are_rejected(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo(repo_root)
    _copy_repo_script(repo_root)

    _write(repo_root / "productivity" / "README.md", "# old bundle\n")
    _write(repo_root / "productivity" / "scripts" / "ci_monitor.py", "print('placeholder')\n")
    add_result = _run(
        [
            "git",
            "add",
            "productivity/README.md",
            "productivity/scripts/ci_monitor.py",
            "scripts/setup/pre-commit-check.sh",
        ],
        repo_root,
    )
    assert add_result.returncode == 0, add_result.stdout + add_result.stderr

    result = _run(["bash", "scripts/setup/pre-commit-check.sh", "--staged"], repo_root)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "productivity/README.md" in result.stdout
    assert "productivity/scripts/ci_monitor.py" in result.stdout
    assert "docs/historical/ or another approved docs archive" in result.stdout
    assert "archive/scripts/ or scripts/" in result.stdout
