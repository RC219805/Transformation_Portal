"""Prevent Node manifests and resolved closures from reintroducing known vulnerabilities."""

import json
from pathlib import Path

import pytest
from packaging.version import Version

pytestmark = [pytest.mark.unit, pytest.mark.security]

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("relative_directory", "package", "minimum_version", "manifest_section"),
    (
        (".", "undici", "7.29.1", "overrides"),
        ("cloudflare/transformationportal-worker", "undici", "7.29.1", "overrides"),
        ("web/secure-landing", "next", "16.3.6", "dependencies"),
        ("web/secure-landing", "fast-uri", "3.1.8", None),
    ),
)
def test_node_manifest_and_closure_meet_security_floor(
    relative_directory: str,
    package: str,
    minimum_version: str,
    manifest_section: str | None,
) -> None:
    directory = REPO_ROOT / relative_directory
    manifest = json.loads((directory / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((directory / "package-lock.json").read_text(encoding="utf-8"))
    minimum = Version(minimum_version)

    if manifest_section is not None:
        assert Version(manifest[manifest_section][package]) >= minimum
    resolved = {path: entry["version"] for path, entry in lock["packages"].items() if path.endswith(f"node_modules/{package}")}
    assert resolved, f"{relative_directory} must resolve {package} in its committed lock"
    for path, version in resolved.items():
        assert Version(version) >= minimum, f"{relative_directory}/{path} resolves vulnerable {package} {version}"
