"""Tests for ReplayEngine against DAGs produced by ExecutionEngine (T-09)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from transformation_portal.runtime.engine import EngineConfig, ExecutionEngine
from transformation_portal.runtime.process_executor import ProcessResult, ProcessTask
from transformation_portal.runtime.replay import NodeRegistry, ReplayEngine, ReplayError

pytestmark = pytest.mark.unit


class EchoNode:
    """Deterministic node whose output depends on its inputs and kwargs."""

    def __init__(self, salt: str = "") -> None:
        self.salt = salt

    def run(self, *, sandbox, text: str = "default") -> Dict[str, Any]:
        return {"result_sha": hashlib.sha256((self.salt + text).encode("utf-8")).hexdigest()}


def _fake_run(task: ProcessTask, timeout=None) -> ProcessResult:
    """Run the node in-process, honouring the task's inputs and kwargs."""
    node = task.node_cls(**task.node_kwargs)
    return ProcessResult(outputs=node.run(sandbox=None, **task.inputs), manifest={})


@pytest.fixture
def engine(tmp_path: Path) -> ExecutionEngine:
    config = EngineConfig(workspace_root=tmp_path / "ws", cas_root=tmp_path / "cas", enable_provenance=True)
    with patch("transformation_portal.runtime.engine.ProcessExecutor") as mock_exec_cls:
        executor = MagicMock()
        executor.run.side_effect = _fake_run
        mock_exec_cls.return_value = executor
        eng = ExecutionEngine(config)
    eng.gpu_pool = None
    return eng


def test_replay_resolves_custom_registry_name_and_reuses_inputs(engine: ExecutionEngine) -> None:
    registry = NodeRegistry()
    registry.register("echo", EchoNode)  # registered under a name that differs from __name__

    node_hash, outputs = engine.run_node(EchoNode, inputs={"text": "hello"}, node_id="echo_a", node_kwargs={"salt": "s"})
    result = ReplayEngine(engine.dag, engine, registry).replay(node_hash)

    assert result.outputs == outputs
    assert result.outputs_match is True
    replay_call = engine.executor.run.call_args_list[-1]
    task = replay_call.args[0]
    assert task.inputs == {"text": "hello"}
    assert task.node_kwargs == {"salt": "s"}


def test_replay_detects_different_outputs(engine: ExecutionEngine) -> None:
    registry = NodeRegistry()
    registry.register("echo", EchoNode)
    node_hash, _ = engine.run_node(EchoNode, inputs={"text": "hello"}, node_id="echo_a")

    engine.dag.nodes[node_hash].outputs["result_sha"] = "0" * 64
    assert ReplayEngine(engine.dag, engine, registry).replay(node_hash).outputs_match is False


def test_replay_fails_closed_without_recorded_inputs(engine: ExecutionEngine) -> None:
    registry = NodeRegistry()
    registry.register("echo", EchoNode)
    node_hash, _ = engine.run_node(EchoNode, inputs={"text": "x"}, node_id="a")
    calls_before = engine.executor.run.call_count
    # Legacy nodes (no recorded inputs) and nodes whose inputs were not JSON-plain.
    for legacy in (True, False):
        meta = engine.dag.nodes[node_hash].metadata
        if legacy:
            meta.pop("original_inputs", None)
        else:
            meta["original_inputs"] = None
        with pytest.raises(ReplayError, match="no recorded inputs"):
            ReplayEngine(engine.dag, engine, registry).replay(node_hash)
    assert engine.executor.run.call_count == calls_before


def test_replay_unknown_class_raises(engine: ExecutionEngine) -> None:
    node_hash, _ = engine.run_node(EchoNode, inputs={"text": "x"}, node_id="a")
    with pytest.raises(ReplayError, match="Unknown node type"):
        ReplayEngine(engine.dag, engine, NodeRegistry()).replay(node_hash)
