"""Maintained Python entrypoints must consume native prepared authority."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAINTAINED_SURFACES = (
    "examples/pbr_preset_example.py",
    "examples/process_750_picacho_pbr.py",
    "scripts/validation/validate_efficientsam_production.py",
    "scripts/ci/apex/matrix_runner.py",
    "scripts/analysis/benchmark_phase2.py",
    "src/transformation_portal/lux_depth_v3/pbr_presets.py",
    "src/transformation_portal/lux_depth_v3/README.md",
    "docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md",
    "docs/guides/PBR_PROCESSOR_QUICKSTART.md",
    "docs/reference/PBR_PRESETS_QUICK_REFERENCE.md",
)

_PBR_DOCUMENTATION_SURFACES = {
    "docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md",
    "docs/guides/PBR_PROCESSOR_QUICKSTART.md",
    "docs/reference/PBR_PRESETS_QUICK_REFERENCE.md",
}


@pytest.mark.parametrize("relative_path", _MAINTAINED_SURFACES)
def test_maintained_python_surfaces_use_prepared_execution(relative_path: str) -> None:
    source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert "EnhanceOrchestrator(" not in source
    assert "prepare_lux_execution" in source
    assert "EnhanceOrchestrator.from_prepared" in source
    if relative_path in _PBR_DOCUMENTATION_SURFACES:
        assert "ImageInput" in source
