from __future__ import annotations

import importlib
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_transformation_portal_module() -> None:
    yield
    import transformation_portal

    importlib.reload(transformation_portal)


def test_runtime_version_prefers_installed_package_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal

    monkeypatch.setattr(metadata, "version", lambda package_name: "9.8.7")

    module = importlib.reload(transformation_portal)

    assert module.__version__ == "9.8.7"


def test_runtime_version_falls_back_to_pyproject_when_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal

    expected_version = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]

    def _raise_package_not_found(_package_name: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", _raise_package_not_found)

    module = importlib.reload(transformation_portal)

    assert module.__version__ == expected_version


def test_cli_version_command_uses_runtime_version(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformation_portal
    from transformation_portal.__main__ import app as main_app

    monkeypatch.setattr(transformation_portal, "__version__", "0.1.0")

    result = CliRunner().invoke(main_app, ["version"])

    assert result.exit_code == 0
    assert "Transformation Portal v0.1.0" in result.output
