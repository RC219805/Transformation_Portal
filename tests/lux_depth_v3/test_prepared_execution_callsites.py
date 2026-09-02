"""Maintained Python entrypoints must consume native prepared authority."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution

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

_MAINTAINED_PYTHON_ENTRYPOINTS = {
    "examples/pbr_preset_example.py",
    "examples/process_750_picacho_pbr.py",
    "scripts/validation/validate_efficientsam_production.py",
    "scripts/ci/apex/matrix_runner.py",
    "scripts/analysis/benchmark_phase2.py",
}


@pytest.mark.parametrize("relative_path", _MAINTAINED_SURFACES)
def test_maintained_python_surfaces_use_prepared_execution(relative_path: str) -> None:
    source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    if relative_path in _MAINTAINED_PYTHON_ENTRYPOINTS:
        tree = ast.parse(source, filename=relative_path)
        direct_constructors = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "EnhanceOrchestrator"
        ]
        prepare_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "prepare_lux_execution"
        ]
        prepared_constructors = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_prepared"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "EnhanceOrchestrator"
        ]

        assert not direct_constructors
        assert prepare_calls
        assert prepared_constructors
        for call in prepare_calls:
            input_expression = (
                call.args[2]
                if len(call.args) >= 3
                else next(
                    (keyword.value for keyword in call.keywords if keyword.arg == "input_files"),
                    None,
                )
            )
            assert input_expression is not None
            assert any(
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "absolute"
                for node in ast.walk(input_expression)
            )
        assert any(isinstance(node, ast.Attribute) and node.attr == "input_files" for node in ast.walk(tree))
    else:
        assert "EnhanceOrchestrator(" not in source
        assert "prepare_lux_execution" in source
        assert "EnhanceOrchestrator.from_prepared" in source
    if relative_path in _PBR_DOCUMENTATION_SURFACES:
        assert "ImageInput" in source


def test_lexically_anchored_input_prepares_under_relative_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A discovered child must not be joined to its relative root twice."""

    monkeypatch.chdir(tmp_path)
    relative_root = Path("inputs")
    relative_root.mkdir()
    relative_image = relative_root / "scene.jpg"
    relative_image.write_bytes(b"not-decoded-during-preparation")

    prepared = prepare_lux_execution(
        EnhanceConfig(
            depth_backend="synthetic",
            allow_synthetic_fallback=True,
            enable_v2=False,
        ),
        relative_root,
        [relative_image.absolute()],
    )

    assert prepared.input_root == relative_root.resolve()
    assert prepared.input_files == (relative_image.resolve(),)
