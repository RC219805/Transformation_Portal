"""Prevent Node manifests and resolved closures from reintroducing known vulnerabilities."""

import json
from pathlib import Path

import pytest
from packaging.version import Version

pytestmark = [pytest.mark.unit, pytest.mark.security]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _assert_node_security_floor(
    directory: Path,
    package: str,
    minimum_version: str,
    manifest_section: str | None,
    *,
    required_resolution: bool,
) -> None:
    manifest = json.loads((directory / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((directory / "package-lock.json").read_text(encoding="utf-8"))
    minimum = Version(minimum_version)

    if manifest_section is not None:
        assert Version(manifest[manifest_section][package]) >= minimum
    resolved = {path: entry["version"] for path, entry in lock["packages"].items() if path.endswith(f"node_modules/{package}")}
    declared = any(
        package in manifest.get(section, {})
        for section in ("dependencies", "devDependencies", "optionalDependencies", "overrides")
    )
    if required_resolution or declared:
        assert resolved, f"{directory} must resolve {package} in its committed lock"
    for path, version in resolved.items():
        assert Version(version) >= minimum, f"{directory}/{path} resolves vulnerable {package} {version}"


@pytest.mark.parametrize(
    ("relative_directory", "package", "minimum_version", "manifest_section", "required_resolution"),
    (
        (".", "undici", "7.30.1", "overrides", True),
        ("cloudflare/transformationportal-worker", "undici", "7.30.1", "overrides", True),
        ("web/secure-landing", "next", "16.3.6", "dependencies", True),
        # Stylelint 18 removed the validator chain that previously installed fast-uri.
        ("web/secure-landing", "fast-uri", "3.1.8", None, False),
    ),
)
def test_node_manifest_and_closure_meet_security_floor(
    relative_directory: str,
    package: str,
    minimum_version: str,
    manifest_section: str | None,
    required_resolution: bool,
) -> None:
    _assert_node_security_floor(
        REPO_ROOT / relative_directory,
        package,
        minimum_version,
        manifest_section,
        required_resolution=required_resolution,
    )


@pytest.mark.parametrize(
    ("resolved_version", "declared", "required_resolution", "error"),
    (
        (None, False, False, None),
        ("3.1.8", False, False, None),
        ("3.1.7", False, False, "resolves vulnerable fast-uri"),
        (None, True, False, "must resolve fast-uri"),
        (None, False, True, "must resolve fast-uri"),
    ),
)
def test_removed_transitives_still_enforce_installed_and_declared_security_floors(
    tmp_path: Path,
    resolved_version: str | None,
    declared: bool,
    required_resolution: bool,
    error: str | None,
) -> None:
    manifest = {"devDependencies": {"fast-uri": "3.1.8"}} if declared else {}
    packages = {"node_modules/fast-uri": {"version": resolved_version}} if resolved_version else {}
    (tmp_path / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "package-lock.json").write_text(json.dumps({"packages": packages}), encoding="utf-8")

    if error:
        with pytest.raises(AssertionError, match=error):
            _assert_node_security_floor(tmp_path, "fast-uri", "3.1.8", None, required_resolution=required_resolution)
    else:
        _assert_node_security_floor(tmp_path, "fast-uri", "3.1.8", None, required_resolution=required_resolution)
