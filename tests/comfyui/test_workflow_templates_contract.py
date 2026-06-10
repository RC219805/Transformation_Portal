"""Pure ComfyUI workflow construction contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from transformation_portal.comfyui.workflow_builder import NodeType, WorkflowBuilder
from transformation_portal.comfyui.workflow_templates import WorkflowTemplates

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_comfyui_package_imports_without_site_packages():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import transformation_portal.comfyui as c; print(c.WorkflowBuilder)",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "factory",
    [
        lambda: WorkflowTemplates.full_luxury_estate_pipeline("input.jpg", "output.jpg"),
        lambda: WorkflowTemplates.quick_iterative_enhancement("input.jpg", "output.jpg"),
        lambda: WorkflowTemplates.material_specific_enhancement("input.jpg", "output.jpg"),
        lambda: WorkflowTemplates.location_specific_atmospheric("input.jpg", "output.jpg"),
        lambda: WorkflowTemplates.coastal_property_golden_hour("input.jpg", "output.jpg"),
    ],
)
def test_single_workflow_templates_build(factory):
    workflow = factory()

    assert workflow.nodes
    assert workflow.metadata["name"]
    assert workflow.to_comfyui_format()


def test_multi_variant_generation_builds_expected_count():
    workflows = WorkflowTemplates.multi_variant_generation(
        input_path="input.jpg",
        output_dir="variants",
        num_variants=4,
    )

    assert len(workflows) == 4
    assert all(workflow.nodes for workflow in workflows)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_variants": 0}, "num_variants must be >= 1"),
        ({"emotional_targets": []}, "emotional_targets must not be empty"),
        ({"flux_strengths": []}, "flux_strengths must not be empty"),
    ],
)
def test_multi_variant_generation_rejects_invalid_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        WorkflowTemplates.multi_variant_generation(
            input_path="input.jpg",
            output_dir="variants",
            **kwargs,
        )


def test_save_all_templates_writes_expected_files(tmp_path):
    WorkflowTemplates.save_all_templates(str(tmp_path))

    expected = {
        "full_luxury_estate_pipeline.json",
        "quick_iterative_enhancement.json",
        "material_specific_enhancement.json",
        "location_specific_atmospheric.json",
        "coastal_property_golden_hour.json",
        "multi_variant_1.json",
        "multi_variant_2.json",
        "multi_variant_3.json",
    }

    assert expected <= {path.name for path in tmp_path.iterdir()}


def test_scene_analysis_is_sidecar_not_image_chain_head():
    workflow = WorkflowBuilder().add_input("input.jpg").add_scene_analysis(detailed=True).add_flux_enhancement().build()

    flux_node_id = next(node_id for node_id, node in workflow.nodes.items() if node.node_type == NodeType.FLUX_ENHANCEMENT)

    flux_input_connection = next(conn for conn in workflow.connections if conn.target_node_id == flux_node_id)

    source_node = workflow.nodes[flux_input_connection.source_node_id]
    assert source_node.node_type == NodeType.INPUT
    assert flux_input_connection.source_output == "IMAGE"


def test_quality_validation_is_sidecar_not_output_image_source():
    workflow = (
        WorkflowBuilder()
        .add_input("input.jpg")
        .add_flux_enhancement()
        .add_quality_validation(pass_threshold=7.0)
        .add_output("output.jpg")
        .build()
    )

    output_node_id = next(node_id for node_id, node in workflow.nodes.items() if node.node_type == NodeType.OUTPUT)

    output_input_connection = next(conn for conn in workflow.connections if conn.target_node_id == output_node_id)

    source_node = workflow.nodes[output_input_connection.source_node_id]
    assert source_node.node_type == NodeType.FLUX_ENHANCEMENT
    assert output_input_connection.source_output == "IMAGE"


def test_executor_fails_fast_for_unsupported_nodes():
    from transformation_portal.comfyui.executor import WorkflowExecutor

    workflow = WorkflowTemplates.quick_iterative_enhancement("input.jpg", "output.jpg")
    executor = WorkflowExecutor(cache_models=False)

    result = executor.execute(workflow)

    assert result["success"] is False
    assert "No executor implementation" in result["errors"][0]
