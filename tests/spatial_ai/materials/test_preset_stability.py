"""Tests for preset stability enforcement (Phase 5F).

Ensures stable presets remain immutable without explicit version promotion.
This guards against unintended configuration drift.
"""

import hashlib
from pathlib import Path

import pytest



pytestmark = pytest.mark.unit

def test_stable_preset_immutable():
    """Verify stable preset hasn't changed without version bump.

    This test enforces that config/presets/material_pbr.yaml (stable v5.0.0)
    remains unchanged. If this test fails:

    1. If change is intentional:
       - Bump version in preset (e.g., v5.0.0 → v5.1.0)
       - Update EXPECTED_HASH_V5_0_0 below
       - Document change in CHANGELOG.md
       - Update migration guide

    2. If change is unintentional:
       - Revert changes to material_pbr.yaml
       - Changes to stable presets must go through governance

    Rationale: "Stable" must be enforceable, not semantic.
    """
    EXPECTED_HASH_V5_0_0 = "149082a43e215c0c4040d079fe9c5bff909e8bdcb5f833ee0fec2153a56cabf2"

    preset_path = Path("config/presets/material_pbr.yaml")
    assert preset_path.exists(), "Stable preset missing"

    content = preset_path.read_bytes()
    actual_hash = hashlib.sha256(content).hexdigest()

    assert actual_hash == EXPECTED_HASH_V5_0_0, (
        f"Stable preset (material_pbr.yaml) has been modified!\n"
        f"  Expected: {EXPECTED_HASH_V5_0_0}\n"
        f"  Actual:   {actual_hash}\n"
        f"\n"
        f"If this change is intentional:\n"
        f"  1. Bump version in preset YAML (version: 5.1.0)\n"
        f"  2. Update EXPECTED_HASH_V5_0_0 in this test\n"
        f"  3. Document in CHANGELOG.md and mention this in PR description\n"
        f"  4. Update docs/guides/MATERIAL_PBR_MIGRATION.md\n"
        f"\n"
        f"Stable presets require explicit promotion governance."
    )


def test_stable_preset_version_declared():
    """Verify stable preset declares correct version."""
    import yaml

    preset_path = Path("config/presets/material_pbr.yaml")
    with preset_path.open() as f:
        preset = yaml.safe_load(f)

    assert "version" in preset, "Stable preset must declare version"
    assert preset["version"] == "5.0.0", (
        f"Version mismatch: expected 5.0.0, got {preset['version']}\n" f"Update test after version bump"
    )

    assert "tier" in preset, "Stable preset must declare tier"
    assert preset["tier"] == "stable", f"Tier mismatch: expected 'stable', got {preset['tier']}"


def test_stable_preset_backend_locked():
    """Verify stable preset uses heuristic backend (CPU-only, always available)."""
    import yaml

    preset_path = Path("config/presets/material_pbr.yaml")
    with preset_path.open() as f:
        preset = yaml.safe_load(f)

    assert "backend" in preset, "Missing backend section"
    assert "type" in preset["backend"], "Missing backend type specification"

    backend_type = preset["backend"]["type"]
    assert backend_type == "heuristic", (
        f"Stable preset must use 'heuristic' backend (CPU-only), got '{backend_type}'\n"
        f"GPU backends belong in canary/experimental presets"
    )

    # Verify device is CPU
    assert preset["backend"]["device"] == "cpu", f"Stable preset must use CPU device, got {preset['backend']['device']}"


def test_canary_preset_allows_optional_backends():
    """Verify canary preset supports optional GPU backends with fallback."""
    import yaml

    preset_path = Path("config/presets/material_pbr_canary.yaml")
    if not preset_path.exists():
        pytest.skip("Canary preset not present")

    with preset_path.open() as f:
        preset = yaml.safe_load(f)

    assert preset["tier"] == "canary", "Canary preset must be marked as tier=canary"

    # Canary can use any backend (pbr_fusion, heuristic, etc.)
    backend_type = preset["backend"].get("type")
    assert backend_type in [
        "pbr_fusion",
        "heuristic",
        "nvdiffrec",
        "material_gan",
    ], f"Unknown backend in canary preset: {backend_type}"


def test_preset_hierarchy_documented():
    """Verify stable preset documents promotion/rollback paths."""
    import yaml

    preset_path = Path("config/presets/material_pbr.yaml")
    with preset_path.open() as f:
        content = f.read()

    # Check for documentation keywords
    assert "stable" in content.lower(), "Preset should document tier"
    assert "v5.0.0" in content or "5.0.0" in content, "Preset should document version"

    # Verify YAML structure loads correctly
    preset = yaml.safe_load(content)
    assert "name" in preset, "Preset must have name"
    assert "version" in preset, "Preset must have version"
    assert "tier" in preset, "Preset must have tier"


def test_material_pbr_docs_exist():
    """Verify preset-referenced Material PBR docs exist in the repository."""
    required_docs = [
        Path("docs/guides/MATERIAL_PBR_GUIDE.md"),
        Path("docs/guides/MATERIAL_PBR_MIGRATION.md"),
    ]

    for doc_path in required_docs:
        assert doc_path.exists(), f"Missing required Material PBR documentation: {doc_path}"


def test_material_pbr_release_documented_in_changelog():
    """Verify the stable/canary preset release is documented in CHANGELOG."""
    changelog_path = Path("CHANGELOG.md")
    assert changelog_path.exists(), "CHANGELOG.md missing"

    changelog = changelog_path.read_text(encoding="utf-8")
    assert "material_pbr.yaml" in changelog, "Stable PBR preset entry missing from CHANGELOG.md"
    assert "material_pbr_canary.yaml" in changelog, "Canary PBR preset entry missing from CHANGELOG.md"
