"""Contract tests for the repository Pylint configuration."""

from __future__ import annotations

import configparser
import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYLINTRC_PATH = PROJECT_ROOT / ".pylintrc"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"


def _load_pylintrc() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    read_files = config.read(PYLINTRC_PATH, encoding="utf-8")
    assert read_files == [str(PYLINTRC_PATH)]
    return config


def _config_list(value: str) -> set[str]:
    items: set[str] = set()
    for part in value.replace("\n", ",").split(","):
        item = part.split("#", 1)[0].strip()
        if item:
            items.add(item)
    return items


def test_pylintrc_uses_single_process_for_direct_pylint_runs() -> None:
    config = _load_pylintrc()

    assert config["MASTER"].getint("jobs") == 1


def test_pylintrc_path_exclusions_use_ignore_paths() -> None:
    config = _load_pylintrc()
    ignored_names = _config_list(config["MASTER"]["ignore"])
    ignored_paths = _config_list(config["MASTER"]["ignore-paths"])

    assert all("/" not in ignored_name for ignored_name in ignored_names)
    assert ignored_paths == {
        "^tools/deprecated(?:/.*)?$",
        "^src/transformation_portal(?:/.*)?$",
        "^src/luxury_tiff_batch_processor(?:/.*)?$",
        "^scripts(?:/.*)?$",
    }

    expected_matches = {
        "^tools/deprecated(?:/.*)?$": ("tools/deprecated", "tools/deprecated/legacy.py"),
        "^src/transformation_portal(?:/.*)?$": (
            "src/transformation_portal",
            "src/transformation_portal/__init__.py",
        ),
        "^src/luxury_tiff_batch_processor(?:/.*)?$": (
            "src/luxury_tiff_batch_processor",
            "src/luxury_tiff_batch_processor/__init__.py",
        ),
        "^scripts(?:/.*)?$": ("scripts", "scripts/lint_runner.sh"),
    }
    for pattern, excluded_targets in expected_matches.items():
        compiled = re.compile(pattern)
        for excluded_target in excluded_targets:
            assert compiled.match(excluded_target)


def test_pylintrc_keeps_unused_import_signal_enabled() -> None:
    config = _load_pylintrc()
    disabled_messages = _config_list(config["MESSAGES CONTROL"]["disable"])

    assert "W0611" not in disabled_messages
    assert "unused-import" not in disabled_messages


def test_pyproject_does_not_define_shadow_pylint_config() -> None:
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))

    assert "pylint" not in pyproject.get("tool", {})
