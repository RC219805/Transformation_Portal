"""Reject known vulnerable npm resolutions across supported install surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from packaging.version import Version

pytestmark = [pytest.mark.unit, pytest.mark.security]

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("lock_path", "package_name", "minimum_version"),
    (
        # GHSA-wq5f-xc86-pv6w: bundled librsvg vulnerability.
        ("package-lock.json", "sharp", "0.35.5"),
        ("cloudflare/transformationportal-worker/package-lock.json", "sharp", "0.35.5"),
        ("web/secure-landing/package-lock.json", "sharp", "0.35.5"),
        # Next.js 16.3.8 fixes the October 2026 cache and SSRF advisories.
        ("web/secure-landing/package-lock.json", "next", "16.3.8"),
        # GHSA-68fv-2mgg-jv7q: indexed source-map denial of service.
        ("web/secure-landing/package-lock.json", "source-map-js", "1.2.2"),
    ),
)
def test_supported_npm_locks_reject_vulnerable_resolutions(
    lock_path: str,
    package_name: str,
    minimum_version: str,
) -> None:
    lock = json.loads((REPO_ROOT / lock_path).read_text(encoding="utf-8"))
    suffix = f"node_modules/{package_name}"
    resolutions = [
        (location, package["version"])
        for location, package in lock["packages"].items()
        if location == suffix or location.endswith(f"/{suffix}")
    ]

    assert resolutions, f"{package_name} must remain covered in {lock_path}"
    for location, version in resolutions:
        assert Version(version) >= Version(
            minimum_version
        ), f"{lock_path}: {location} resolves vulnerable {version}; minimum is {minimum_version}"


def test_frontdoor_braces_uses_the_reviewed_local_security_fork() -> None:
    frontdoor = REPO_ROOT / "web/secure-landing"
    manifest = json.loads((frontdoor / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((frontdoor / "package-lock.json").read_text(encoding="utf-8"))
    assert manifest["devDependencies"]["braces"] == "file:./vendor/braces"
    assert manifest["overrides"]["braces"] == "$braces"
    fork = lock["packages"]["vendor/braces"]
    assert fork["name"] == "@transformation-portal/braces"
    assert fork["version"] == "3.0.3-tp.1"
    resolutions = [
        package
        for location, package in lock["packages"].items()
        if location == "node_modules/braces" or location.endswith("/node_modules/braces")
    ]
    assert resolutions, "The Stylelint dependency chain must resolve the bounded fork"
    assert all(package.get("link") is True and package["resolved"] == "vendor/braces" for package in resolutions)
