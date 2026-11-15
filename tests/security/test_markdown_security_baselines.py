"""Preserve the patched Markdown parser across governed runtime closures."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

pytestmark = [pytest.mark.unit, pytest.mark.security]

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "markdown-it-py"
MINIMUM = Version("4.3.0")


def _markdown_requirements(lines: list[str]) -> list[Requirement]:
    requirements = [
        Requirement(line.split("#", 1)[0].strip())
        for line in lines
        if line.strip() and not line.lstrip().startswith(("#", "-"))
    ]
    return [requirement for requirement in requirements if canonicalize_name(requirement.name) == PACKAGE]


@pytest.mark.parametrize(
    "relative_path",
    ["requirements/base.in", "requirements/security.in", "requirements/da3-runtime-darwin-arm64.in", "pyproject.toml"],
)
def test_runtime_and_scanner_declare_the_patched_markdown_floor(relative_path: str) -> None:
    path = REPO_ROOT / relative_path
    if relative_path == "pyproject.toml":
        lines = tomllib.loads(path.read_text(encoding="utf-8"))["project"]["dependencies"]
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
    declarations = _markdown_requirements(lines)
    assert len(declarations) == 1, f"{relative_path} must declare the Markdown security floor exactly once"
    requirement = declarations[0]
    assert requirement.url is None and requirement.marker is None and not requirement.extras
    assert MINIMUM in requirement.specifier
    assert Version("4.3.1") in requirement.specifier
    assert Version("5.0.0") not in requirement.specifier
    for rejected in ("4.0.0", "4.1.0", "4.2.0", "4.3.0rc1"):
        assert Version(rejected) not in requirement.specifier


@pytest.mark.parametrize(
    ("filename", "required"),
    [
        ("all.txt", True),
        ("base.txt", True),
        ("ci.txt", True),
        ("security.txt", True),
        ("ml-core-darwin-arm64.txt", True),
        ("da3-runtime-darwin-arm64.txt", True),
        ("dev.txt", False),
        ("tools-archive.txt", False),
    ],
)
def test_governed_markdown_pins_never_reintroduce_the_vulnerable_parser(filename: str, required: bool) -> None:
    declarations = _markdown_requirements((REPO_ROOT / "requirements" / filename).read_text(encoding="utf-8").splitlines())
    if required:
        assert len(declarations) == 1, f"{filename} must pin the Markdown parser"
    assert len(declarations) <= 1
    for requirement in declarations:
        assert requirement.url is None and requirement.marker is None and not requirement.extras
        pins = list(requirement.specifier)
        assert len(pins) == 1 and pins[0].operator == "=="
        version = Version(pins[0].version)
        assert not version.is_prerelease and not version.is_devrelease and version.local is None
        assert MINIMUM <= version < Version("5.0.0"), f"{filename} must exclude the vulnerable Markdown parser"
