"""Core-lane coverage for the ComfyUI workflow builder.

``workflow_builder.py`` is pure-Python (dict/JSON construction, a fluent
builder, an enum reverse-lookup, and a save/load round-trip), but it lived
behind ``comfyui/__init__``'s eager ``custom_nodes`` import, which pulls
``torch`` — so it was only reachable in the ML lane and sat at 0% core-lane
coverage. After the PEP 562 lazy-import seam in ``comfyui/__init__`` it is
importable torch-free; these tests exercise it deterministically (no ML
runtimes, no network).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from transformation_portal.comfyui.workflow_builder import (
    Node,
    NodeConnection,
    NodeType,
    Workflow,
    WorkflowBuilder,
)

# --------------------------------------------------------------------------- #
# Import seam contract
# --------------------------------------------------------------------------- #


def test_workflow_builder_importable_without_torch_backend() -> None:
    """Importing the builder must not drag in the torch-bound custom_nodes module."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "from transformation_portal.comfyui.workflow_builder import WorkflowBuilder; "
                "import sys; "
                "raise SystemExit('transformation_portal.comfyui.custom_nodes' in sys.modules)"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_package_dir_lists_public_exports() -> None:
    import transformation_portal.comfyui as comfyui

    assert "WorkflowBuilder" in dir(comfyui)
    # Lazy access of the pure symbol resolves to the same class.
    assert comfyui.WorkflowBuilder is WorkflowBuilder


def test_package_unknown_attribute_raises() -> None:
    import transformation_portal.comfyui as comfyui

    with pytest.raises(AttributeError):
        _ = comfyui.NoSuchSymbol


# --------------------------------------------------------------------------- #
# NodeConnection slot mapping
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "source_output, expected_slot",
    [
        ("IMAGE", 0),
        ("MASK", 1),
        ("REPORT", 2),
        ("UNKNOWN_NAME", 0),  # falls through to default 0
        (3, 3),  # explicit int slot is preserved
    ],
)
def test_node_connection_slot_mapping(source_output, expected_slot) -> None:
    conn = NodeConnection(
        source_node_id="a",
        source_output=source_output,
        target_node_id="b",
        target_input="image",
    )
    assert conn.to_comfyui_format() == ["a", expected_slot]


# --------------------------------------------------------------------------- #
# Node / Workflow serialization
# --------------------------------------------------------------------------- #


def test_node_to_comfyui_merges_parameters_and_inputs() -> None:
    node = Node(
        node_id="n1",
        node_type=NodeType.FLUX_ENHANCEMENT,
        parameters={"strength": 0.5},
        inputs={"image": ["n0", 0]},
    )
    out = node.to_comfyui_format()
    assert out["class_type"] == "FluxEnhancementNode"
    assert out["inputs"] == {"strength": 0.5, "image": ["n0", 0]}
    assert out["_meta"]["title"] == "FluxEnhancementNode"


def test_workflow_to_comfyui_wires_connections_into_target_inputs() -> None:
    wf = Workflow()
    wf.nodes["in"] = Node("in", NodeType.INPUT, parameters={"image": "a.jpg"})
    wf.nodes["flux"] = Node("flux", NodeType.FLUX_ENHANCEMENT, parameters={"strength": 0.4})
    wf.connections.append(NodeConnection("in", "IMAGE", "flux", "image"))

    formatted = wf.to_comfyui_format()
    assert formatted["flux"]["inputs"]["image"] == ["in", 0]


def test_workflow_to_comfyui_ignores_connection_to_missing_target() -> None:
    wf = Workflow()
    wf.nodes["in"] = Node("in", NodeType.INPUT)
    wf.connections.append(NodeConnection("in", "IMAGE", "ghost", "image"))

    # Should not raise even though "ghost" was never added.
    formatted = wf.to_comfyui_format()
    assert "ghost" not in formatted


def test_workflow_save_creates_parent_dirs_and_writes_json(tmp_path: Path) -> None:
    wf = Workflow()
    wf.nodes["in"] = Node("in", NodeType.INPUT, parameters={"image": "a.jpg"})
    wf.metadata = {"name": "demo"}

    out = tmp_path / "nested" / "wf.json"
    wf.save(out)

    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["last_node_id"] == 1
    assert data["metadata"]["name"] == "demo"
    assert data["nodes"]["in"]["class_type"] == "LoadImage"


def test_workflow_save_accepts_str_path(tmp_path: Path) -> None:
    wf = Workflow()
    wf.nodes["in"] = Node("in", NodeType.INPUT)
    target = tmp_path / "wf.json"
    wf.save(str(target))
    assert target.is_file()


def test_workflow_load_roundtrip(tmp_path: Path) -> None:
    wf = Workflow()
    wf.nodes["in"] = Node("in", NodeType.INPUT, parameters={"image": "a.jpg"})
    wf.metadata = {"name": "demo", "version": "2.0"}
    out = tmp_path / "wf.json"
    wf.save(out)

    loaded = Workflow.load(out)
    assert loaded.metadata == {"name": "demo", "version": "2.0"}
    assert loaded.nodes["in"].node_type is NodeType.INPUT
    assert loaded.nodes["in"].parameters == {"image": "a.jpg"}


def test_workflow_load_skips_unknown_node_types(tmp_path: Path) -> None:
    payload = {
        "metadata": {"name": "mixed"},
        "nodes": {
            "good": {"class_type": "LoadImage", "inputs": {}},
            "bad": {"class_type": "TotallyUnknownNode", "inputs": {}},
        },
    }
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(payload))

    loaded = Workflow.load(path)
    assert "good" in loaded.nodes
    assert "bad" not in loaded.nodes  # unknown type is logged and skipped


# --------------------------------------------------------------------------- #
# Fluent builder
# --------------------------------------------------------------------------- #


def test_builder_metadata_defaults() -> None:
    wf = WorkflowBuilder(name="My Pipeline").build()
    assert wf.metadata["name"] == "My Pipeline"
    assert wf.metadata["version"] == "2.0"


def test_builder_add_input_without_node_id_creates_unconnected_node() -> None:
    wf = WorkflowBuilder().add_input("a.jpg").build()
    # One node, no inbound connection (it is the chain head).
    assert len(wf.nodes) == 1
    assert wf.connections == []
    only = next(iter(wf.nodes.values()))
    assert only.node_type is NodeType.INPUT
    assert only.parameters == {"image": "a.jpg"}


def test_builder_add_input_with_explicit_node_id_sets_chain_head() -> None:
    # When an explicit upstream node_id is supplied, no INPUT node is created;
    # the next stage connects to that id instead.
    wf = WorkflowBuilder().add_input("a.jpg", node_id="external_0").add_flux_enhancement().build()
    assert len(wf.nodes) == 1  # only the flux node
    assert wf.connections[0].source_node_id == "external_0"


def test_builder_linear_chain_connects_sequential_stages() -> None:
    wf = WorkflowBuilder().add_input("a.jpg").add_flux_enhancement(strength=0.6).add_output("out.jpg").build()
    assert len(wf.nodes) == 3
    # Two connections for a 3-node linear chain.
    assert len(wf.connections) == 2
    pairs = {(c.source_node_id.split("_")[0], c.target_node_id.split("_")[0]) for c in wf.connections}
    assert ("loadimage", "fluxenhancementnode") in pairs


def test_builder_flux_enhancement_includes_prompt_only_when_provided() -> None:
    with_prompt = WorkflowBuilder().add_input("a.jpg").add_flux_enhancement(prompt="luxury villa").build()
    flux = [n for n in with_prompt.nodes.values() if n.node_type is NodeType.FLUX_ENHANCEMENT][0]
    assert flux.parameters["prompt"] == "luxury villa"

    without = WorkflowBuilder().add_input("a.jpg").add_flux_enhancement().build()
    flux2 = [n for n in without.nodes.values() if n.node_type is NodeType.FLUX_ENHANCEMENT][0]
    assert "prompt" not in flux2.parameters


def test_builder_skygan_optional_physics_params() -> None:
    wf = WorkflowBuilder().add_input("a.jpg").add_skygan_sky(sun_azimuth=120.0, sun_elevation=35.0, turbidity=2.5).build()
    sky = [n for n in wf.nodes.values() if n.node_type is NodeType.SKYGAN_SKY][0]
    assert sky.parameters["sun_azimuth"] == 120.0
    assert sky.parameters["sun_elevation"] == 35.0
    assert sky.parameters["turbidity"] == 2.5


def test_builder_skygan_omits_unset_physics_params() -> None:
    wf = WorkflowBuilder().add_input("a.jpg").add_skygan_sky().build()
    sky = [n for n in wf.nodes.values() if n.node_type is NodeType.SKYGAN_SKY][0]
    for key in ("sun_azimuth", "sun_elevation", "turbidity"):
        assert key not in sky.parameters
    assert sky.parameters["location"] == "montecito"


def test_builder_scene_analysis_and_quality_validation_are_sidecars() -> None:
    builder = WorkflowBuilder().add_input("a.jpg")
    builder.add_scene_analysis(detailed=False)
    assert builder._last_output == "IMAGE"

    builder.add_quality_validation(pass_threshold=8.0, warning_threshold=6.0)
    assert builder._last_output == "IMAGE"

    qv = [n for n in builder.workflow.nodes.values() if n.node_type is NodeType.QUALITY_VALIDATION][0]
    assert qv.parameters == {
        "pass_threshold": 8.0,
        "warning_threshold": 6.0,
        "check_realism": True,
        "check_structural_accuracy": True,
        "check_material_consistency": False,
    }


def test_builder_full_chain_matches_docstring_example(tmp_path: Path) -> None:
    workflow = (
        WorkflowBuilder()
        .add_input("image.jpg")
        .add_scene_analysis()
        .add_flux_enhancement(strength=0.45)
        .add_skygan_sky(location="montecito", time_of_day="golden_hour", auto_correct=True)
        .add_quality_validation(pass_threshold=7.0)
        .add_output("enhanced.jpg")
        .build()
    )
    assert len(workflow.nodes) == 6
    # Round-trips through ComfyUI JSON without error.
    formatted = workflow.to_comfyui_format()
    assert len(formatted) == 6


def test_builder_repr_reports_node_count() -> None:
    builder = WorkflowBuilder().add_input("a.jpg")
    assert repr(builder) == "WorkflowBuilder(nodes=1)"
