"""Tests for execution-graph NVDiffRec node truthfulness semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from transformation_portal.execution_graph.nodes.base import NodeResult
from transformation_portal.execution_graph.nodes.nvdiffrec_node import NVDiffRecNode
from transformation_portal.stage_graph.stage import StageStatus

pytestmark = pytest.mark.unit


def test_node_result_error_defaults_to_failed() -> None:
    result = NodeResult(error="boom")

    assert result.status == StageStatus.FAILED
    assert result.success is False


def test_unwired_nvdiffrec_node_returns_unavailable(tmp_path: Path) -> None:
    node = NVDiffRecNode()

    result = node.run(
        image_paths=[tmp_path / "view1.png", tmp_path / "view2.png"],
        output_dir=tmp_path / "out",
    )

    assert result.status == StageStatus.UNAVAILABLE
    assert result.success is False
    assert result.outputs == {}
    assert result.metadata["backend_available"] is False
    assert result.metadata["capability"]["availability_state"] == "backend_unwired"
    assert result.metadata["capability"]["stub_mode"] is True
