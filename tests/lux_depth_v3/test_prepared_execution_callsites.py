"""Maintained Python entrypoints must consume native prepared authority."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import examples.pbr_preset_example as preset_example
import examples.process_750_picacho_pbr as picacho_example
from examples.pbr_preset_example import _authoritative_output_paths as preset_output_paths
from examples.process_750_picacho_pbr import _authoritative_output_paths as picacho_output_paths
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
from transformation_portal.lux_depth_v3.manifest import CombinedManifest

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

_PBR_EXAMPLES = {
    "examples/pbr_preset_example.py",
    "examples/process_750_picacho_pbr.py",
}

_PBR_ORCHESTRATOR_OUTPUT_DOCS = {
    "examples/README.md",
    "docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md",
    "docs/reference/PBR_PRESETS_QUICK_REFERENCE.md",
}

_PREPARED_OUTPUT_INVENTORY_DOCS = {
    "examples/README.md",
    "docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md",
    "docs/reference/PBR_PRESETS_QUICK_REFERENCE.md",
    "scripts/TEST_V2_INTEGRATION_README.md",
    "src/transformation_portal/lux_depth_v3/README.md",
}

_PREPARED_INPUT_ROOT_DOCS = {
    "docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md",
    "docs/guides/PBR_PROCESSOR_QUICKSTART.md",
    "docs/reference/PBR_PRESETS_QUICK_REFERENCE.md",
    "src/transformation_portal/lux_depth_v3/README.md",
}

_ORCHESTRATOR_PATH_DOC_EXPECTATIONS = (
    (
        "docs/guides/PBR_PROCESSOR_QUICKSTART.md",
        ('result["depth_float_path"]', "caller-managed depth file"),
        ("ls output/*_depth.npy",),
    ),
    (
        "docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md",
        ("orchestrator_results", 'result.get("depth_float_path")'),
        ('Path("output/batch_depths/")',),
    ),
    (
        "src/transformation_portal/lux_depth_v3/README.md",
        (
            "depth/<input-key>_depth.npy",
            "pbr/<input-key>_normal.png",
            "manifests/<input-key>_combined.json",
        ),
        ("│   ├── normal/", "- `*_depth.png` -"),
    ),
    (
        "docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md",
        (
            "depth/<input-key>_depth.npy",
            "pbr/<input-key>_normal.png",
            "manifests/<input-key>_combined.json",
        ),
        ("- `*_depth.png`:", "- `*_combined.json`:"),
    ),
    (
        "docs/architecture/APEX_WORKFLOW_DESIGN.md",
        ('manifest_path = Path(result["manifest"])',),
        ("image_manifest.json",),
    ),
    (
        "scripts/TEST_V2_INTEGRATION_README.md",
        (
            "<input-key>_depth.png",
            "<input-key>_normal.png",
            "<input-key>_combined.json",
        ),
        ("_pbr.png", "_manifest.json"),
    ),
)

_STANDALONE_PBRPROCESSOR_DOCS = (
    "docs/guides/PBR_PROCESSOR_QUICKSTART.md",
    "docs/guides/PBR_ENHANCE_CONFIG_GUIDE.md",
)


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
        method_calls = {
            node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "enhance_batch" in method_calls
        assert "enhance_image" not in method_calls
        assert "enhance_batch_parallel" not in method_calls
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
        assert ".enhance_batch(" in source
        assert ".enhance_image(" not in source
        assert ".enhance_batch_parallel(" not in source
        assert "ImageInput" not in source
    if relative_path in _PBR_EXAMPLES:
        assert "CombinedManifest.load" in source
        assert 'pbr_assets.get(f"{label}_path")' in source
        assert 'output_root / f"{stem}_depth.png"' not in source
        assert 'output_dir / f"{base_name}_manifest.json"' not in source


def _authoritative_example_result(tmp_path: Path) -> tuple[dict[str, str], dict[str, Path]]:
    expected = {
        "depth": tmp_path / "depth" / "planned_depth.png",
        "depth_float": tmp_path / "depth" / "planned_depth.npy",
        "manifest": tmp_path / "manifests" / "planned_combined.json",
        "normal": tmp_path / "pbr" / "planned_normal.png",
        "roughness": tmp_path / "pbr" / "planned_roughness.png",
        "ao": tmp_path / "pbr" / "planned_ao.png",
    }
    for path in expected.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path != expected["manifest"]:
            path.write_bytes(b"artifact")
    CombinedManifest(
        pbr_assets={
            "normal_path": str(expected["normal"]),
            "roughness_path": str(expected["roughness"]),
            "ao_path": str(expected["ao"]),
        }
    ).save(expected["manifest"])
    result = {
        "depth_path": str(expected["depth"]),
        "depth_float_path": str(expected["depth_float"]),
        "manifest": str(expected["manifest"]),
    }
    return result, expected


def test_pbr_preset_example_reports_authoritative_result_and_manifest_paths(tmp_path: Path) -> None:
    result, expected = _authoritative_example_result(tmp_path)

    assert preset_output_paths(result, require_float_depth=True) == expected


def test_picacho_example_reports_authoritative_result_and_manifest_paths(tmp_path: Path) -> None:
    result, expected = _authoritative_example_result(tmp_path)

    assert picacho_output_paths(result) == expected


def _stub_example_execution(monkeypatch: pytest.MonkeyPatch, module, input_path: Path, missing_path: Path) -> None:
    prepared = SimpleNamespace(input_root=input_path.parent, input_files=(input_path,))

    class StubOrchestrator:
        @classmethod
        def from_prepared(cls, _prepared, _output_root):
            return cls()

        def enhance_batch(self, _input_root, *, input_files):
            assert input_files == [input_path]
            return [{"status": "ok"}]

    monkeypatch.setattr(module, "prepare_lux_execution", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(module, "EnhanceOrchestrator", StubOrchestrator)
    monkeypatch.setattr(module, "_authoritative_output_paths", lambda *_args, **_kwargs: {"depth": missing_path})


def test_pbr_preset_example_returns_failure_for_missing_evidence_bound_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    input_path = input_root / "scene.png"
    input_path.write_bytes(b"fixture")
    config = SimpleNamespace(
        pbr_normal_strength=1.0,
        pbr_normal_blur_radius=0,
        pbr_roughness_strength=1.0,
        pbr_roughness_blur_radius=0,
        pbr_ao_strength=1.0,
        pbr_ao_blur_radius=0,
        pbr_ao_bias=0.5,
        save_float_depth=True,
        model_key="synthetic",
    )
    monkeypatch.setattr(preset_example, "get_preset", lambda _preset: config)
    _stub_example_execution(monkeypatch, preset_example, input_path, tmp_path / "missing.png")
    monkeypatch.setattr(
        preset_example.sys,
        "argv",
        ["pbr_preset_example.py", "--input", str(input_root)],
    )

    assert preset_example.main() == 1


def test_picacho_example_returns_failure_for_missing_evidence_bound_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "scene.png"
    input_path.write_bytes(b"fixture")
    monkeypatch.setattr(picacho_example, "print_header", lambda: None)
    monkeypatch.setattr(picacho_example, "analyze_source_file", lambda _path: {})
    monkeypatch.setattr(picacho_example, "print_source_analysis", lambda _info: None)
    monkeypatch.setattr(picacho_example, "get_preset", lambda _preset: object())
    monkeypatch.setattr(picacho_example, "print_preset_config", lambda _config, _preset: None)
    monkeypatch.setattr(picacho_example, "print_outputs", lambda _output: None)
    _stub_example_execution(monkeypatch, picacho_example, input_path, tmp_path / "missing.png")

    assert picacho_example.process_image(input_path, tmp_path / "output") == 1


@pytest.mark.parametrize("relative_path", sorted(_PBR_ORCHESTRATOR_OUTPUT_DOCS))
def test_pbr_orchestrator_docs_do_not_reconstruct_legacy_flat_paths(relative_path: str) -> None:
    source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert "_depth_float.npy" not in source
    assert "_manifest.json" not in source
    assert "<input-key>_depth.npy" in source
    assert "<input-key>_combined.json" in source


@pytest.mark.parametrize(
    ("relative_path", "required_fragments", "forbidden_fragments"),
    _ORCHESTRATOR_PATH_DOC_EXPECTATIONS,
)
def test_maintained_orchestrator_docs_use_evidence_bound_paths(
    relative_path: str,
    required_fragments: tuple[str, ...],
    forbidden_fragments: tuple[str, ...],
) -> None:
    source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    for fragment in required_fragments:
        assert fragment in source, f"{relative_path} is missing current path guidance {fragment!r}"
    for fragment in forbidden_fragments:
        assert fragment not in source, f"{relative_path} retains stale orchestrator path {fragment!r}"


@pytest.mark.parametrize("relative_path", _STANDALONE_PBRPROCESSOR_DOCS)
def test_standalone_pbrprocessor_docs_keep_caller_owned_flat_outputs(relative_path: str) -> None:
    source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert 'depth_path=Path("output/scene1_depth.npy")' in source
    assert 'output_dir=Path("output/pbr/")' in source
    assert "scene1_normal.png" in source


@pytest.mark.parametrize("relative_path", sorted(_PREPARED_OUTPUT_INVENTORY_DOCS))
def test_prepared_output_inventories_include_completion_records(relative_path: str) -> None:
    source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    for fragment in (
        "<input-key>_depth_metadata.json",
        "batch_<batch-id>.json",
        "execution_evidence_<batch-id>.json",
        "run_card_<batch-id>.json",
        "run_card_<batch-id>.self.json",
    ):
        assert fragment in source, f"{relative_path} omits required output inventory entry {fragment!r}"


def test_source_readme_uses_current_prepared_and_pbrprocessor_apis() -> None:
    source = (_REPO_ROOT / "src/transformation_portal/lux_depth_v3/README.md").read_text(encoding="utf-8")

    assert "prepared.input_root" in source
    assert "input_files=list(prepared.input_files)" in source
    assert 'get_preset("premium").to_pbr_config()' in source
    assert "PBRProcessor.from_cached_depth(" in source
    assert "processor.process_image(" not in source


@pytest.mark.parametrize("relative_path", sorted(_PREPARED_INPUT_ROOT_DOCS))
def test_documented_prepared_input_roots_are_absolute_before_discovery(relative_path: str) -> None:
    source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assignments = [line.strip() for line in source.splitlines() if line.strip().startswith("input_root = Path(")]

    assert assignments, f"{relative_path} has no documented input-root assignment"
    for assignment in assignments:
        assert assignment.endswith(".resolve()"), f"{relative_path} leaves a prepared input root relative: {assignment}"


def test_pbrprocessor_constructor_examples_use_path_output_directories() -> None:
    for relative_path in (
        "examples/README.md",
        "docs/guides/PBR_PROCESSOR_QUICKSTART.md",
        "src/transformation_portal/lux_depth_v3/pbr_processor.py",
    ):
        source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert 'output_dir="output/' not in source


@pytest.mark.parametrize(
    ("relative_path", "marker"),
    (
        ("examples/README.md", 'output_dir=Path("output/custom_pbr/")'),
        ("docs/guides/PBR_PROCESSOR_QUICKSTART.md", 'output_dir=Path("output/custom/")'),
    ),
)
def test_pbrprocessor_path_examples_import_path_in_the_same_fence(relative_path: str, marker: str) -> None:
    source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
    fences = [segment.split("```", 1)[0] for segment in source.split("```python")[1:]]
    matching = [fence for fence in fences if marker in fence]

    assert len(matching) == 1
    assert "from pathlib import Path" in matching[0]


def test_pbrprocessor_pipeline_docs_do_not_mutate_shared_presets() -> None:
    source = (_REPO_ROOT / "docs/guides/PBR_PROCESSOR_QUICKSTART.md").read_text(encoding="utf-8")

    assert 'replace(get_preset("premium"), generate_pbr=False)' in source
    assert "config.generate_pbr = False" not in source


def test_package_readme_workflow_output_sections_are_complete() -> None:
    source = (_REPO_ROOT / "src/transformation_portal/lux_depth_v3/README.md").read_text(encoding="utf-8")
    pbr_outputs = source.split("### PBR-Only (No Enhancement)", 1)[1].split("### Client Deliverable (APEX)", 1)[0]
    client_outputs = source.split("### Client Deliverable (APEX)", 1)[1].split("### Run Card Trust Layers", 1)[0]

    required = (
        "<input-key>_depth_metadata.json",
        "batch_<batch-id>.json",
        "execution_evidence_<batch-id>.json",
        "run_card_<batch-id>.json",
        "run_card_<batch-id>.self.json",
    )
    for section in (pbr_outputs, client_outputs):
        for fragment in required:
            assert fragment in section


def test_examples_readme_distinguishes_result_and_manifest_output_paths() -> None:
    source = (_REPO_ROOT / "examples/README.md").read_text(encoding="utf-8")
    normalized = " ".join(source.split())

    assert "Per-image depth and manifest paths are returned by `enhance_batch`" in normalized
    assert "PBR paths are recorded in each combined manifest" in normalized
    assert "The exact evidence-bound paths are returned by `enhance_batch`" not in normalized


def test_cli_inventory_uses_real_run_card_names() -> None:
    source = (_REPO_ROOT / "docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md").read_text(encoding="utf-8")

    assert "run_card_<batch-id>.json" in source
    assert "run_card_<batch-id>.self.json" in source
    assert "*_run_card.json" not in source


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
