"""Keep native lock inputs and closures above reviewed security floors."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

pytestmark = [pytest.mark.unit, pytest.mark.security]
REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements"


def test_da3_bootstrap_setuptools_matches_patched_runtime_authority() -> None:
    root = REQUIREMENTS.parent
    contract = json.loads((root / "config/da3_runtime_identity_contract.json").read_text())
    installer = (root / "scripts/setup/install_da3_runtime.sh").read_text()
    declaration = re.search(r'^SETUPTOOLS_VERSION="([^"]+)"$', installer, re.MULTILINE)
    assert declaration is not None
    version = declaration[1]
    assert Version(version) >= Version("84.1.0")  # GHSA-xmj3-9gf7-h52w
    assert contract["governed_additional_distributions"]["setuptools"] == version


@pytest.mark.parametrize(
    "filename,package,minimum",
    [
        ("ml-core-darwin-arm64.in", "fsspec", "2026.6.0"),
        ("ml-core-darwin-arm64.in", "filelock", "4.2.1"),
        ("da3-runtime-darwin-arm64.in", "werkzeug", "3.1.9"),
        ("da3-runtime-darwin-arm64.in", "certifi", "2026.10.19"),
        ("da3-runtime-darwin-arm64.in", "filelock", "4.2.1"),
    ],
)
def test_native_inputs_exclude_vulnerable_versions(filename: str, package: str, minimum: str) -> None:
    declarations = [
        Requirement(line.split("#", 1)[0].strip())
        for line in (REQUIREMENTS / filename).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-"))
    ]
    requirement = next(item for item in declarations if canonicalize_name(item.name) == package)
    assert Version(minimum) in requirement.specifier
    assert any(item.operator == ">=" and Version(item.version) >= Version(minimum) for item in requirement.specifier)


@pytest.mark.parametrize(
    "filename,package,minimum",
    [
        ("ml-core-darwin-arm64.txt", "fsspec", "2026.6.0"),
        ("ml-core-darwin-arm64.txt", "filelock", "4.2.1"),
        ("ml-core-darwin-arm64.txt", "requests", "2.34.2"),
        ("ml-core-darwin-arm64.txt", "pillow", "13.0.0"),
        ("ml-core-darwin-arm64.txt", "certifi", "2026.10.19"),
        ("da3-runtime-darwin-arm64.txt", "werkzeug", "3.1.9"),
        ("da3-runtime-darwin-arm64.txt", "requests", "2.34.2"),
        ("da3-runtime-darwin-arm64.txt", "pillow", "13.0.0"),
        ("da3-runtime-darwin-arm64.txt", "certifi", "2026.10.19"),
        ("da3-runtime-darwin-arm64.txt", "filelock", "4.2.1"),
    ],
)
def test_native_locks_install_security_baselines(filename: str, package: str, minimum: str) -> None:
    pins = [
        Requirement(line.strip())
        for line in (REQUIREMENTS / filename).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-"))
    ]
    requirement = next(item for item in pins if canonicalize_name(item.name) == package)
    assert len(requirement.specifier) == 1
    pin = next(iter(requirement.specifier))
    assert pin.operator == "=="
    assert Version(pin.version) >= Version(minimum)
