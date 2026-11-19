"""Tests for ClusterAutoscheduler heartbeat status, wire payloads and scale checks."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

from transformation_portal.runtime.autoscheduler import (
    ClusterAutoscheduler,
    SchedulerError,
    WorkerState,
    WorkerStatus,
    task_to_wire,
)
from transformation_portal.runtime.process_executor import ProcessTask

pytestmark = [pytest.mark.unit, pytest.mark.regression]


class _Node:
    """Stand-in DAG node class (never instantiated)."""


def _task(**sandbox: Any) -> ProcessTask:
    config: Dict[str, Any] = {"node_id": "n1", "workspace_root": Path("/tmp/ws")}
    config.update(sandbox)
    return ProcessTask(node_cls=_Node, node_kwargs={"scale": 2}, inputs={"image": "a" * 64}, sandbox_config=config)


@pytest.mark.parametrize("status", [WorkerStatus.DRAINING, WorkerStatus.BUSY])
def test_heartbeat_does_not_undrain_worker(status: WorkerStatus) -> None:
    scheduler = ClusterAutoscheduler()
    scheduler.register_worker("w1", WorkerState(host="127.0.0.1", port=1))
    scheduler.workers["w1"].status = status

    scheduler.heartbeat("w1")

    assert scheduler.workers["w1"].status is status
    with pytest.raises(SchedulerError):
        scheduler.select_worker()


def test_heartbeat_brings_offline_worker_back_online() -> None:
    scheduler = ClusterAutoscheduler()
    scheduler.register_worker("w1", WorkerState(host="127.0.0.1", port=1, last_heartbeat=0.0))
    scheduler._check_heartbeats()
    assert scheduler.workers["w1"].status is WorkerStatus.OFFLINE

    scheduler.heartbeat("w1")

    assert scheduler.workers["w1"].status is WorkerStatus.ONLINE
    assert scheduler.select_worker().worker_id == "w1"


def test_task_to_wire_is_json_native() -> None:
    wire = task_to_wire(_task(extra=(Path("/a"), 1.5)))

    assert wire == {
        "node_cls": f"{_Node.__module__}:_Node",
        "node_kwargs": {"scale": 2},
        "inputs": {"image": "a" * 64},
        "sandbox_config": {"node_id": "n1", "workspace_root": "/tmp/ws", "extra": ["/a", 1.5]},
    }
    assert json.loads(json.dumps(wire)) == wire


def test_task_to_wire_rejects_unserializable_fields() -> None:
    with pytest.raises(SchedulerError, match="sandbox_config.handle"):
        task_to_wire(_task(handle=object()))
    with pytest.raises(SchedulerError, match="finite"):
        task_to_wire(_task(weight=float("nan")))

    def make_local() -> type:
        class Local:
            pass

        return Local

    local_task = ProcessTask(node_cls=make_local(), node_kwargs={}, inputs={}, sandbox_config={})
    with pytest.raises(SchedulerError, match="importable"):
        task_to_wire(local_task)


def test_dispatch_with_default_send_fn_delivers_wire_payload() -> None:
    received: List[bytes] = []
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(10)

    def serve() -> None:
        conn, _ = server.accept()
        with conn:
            conn.settimeout(10)
            chunks = []
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            received.append(b"".join(chunks))
            conn.sendall(b'{"outputs": {"depth": "' + b"d" * 64 + b'"}}')

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        scheduler = ClusterAutoscheduler()
        scheduler.register_worker("w1", WorkerState(host="127.0.0.1", port=server.getsockname()[1]))

        outputs = scheduler.dispatch(_task())
    finally:
        thread.join(timeout=10)
        server.close()

    assert outputs == {"depth": "d" * 64}
    assert json.loads(received[0].decode("utf-8")) == task_to_wire(_task())
    assert scheduler.workers["w1"].load == 0


def test_dispatch_fails_closed_before_send_for_unserializable_task() -> None:
    sent: List[Dict[str, Any]] = []
    scheduler = ClusterAutoscheduler()
    scheduler.register_worker("w1", WorkerState(host="h", port=1))

    with pytest.raises(SchedulerError, match="not wire-serializable"):
        scheduler.dispatch(_task(handle=object()), send_fn=lambda h, p, d: sent.append(d) or {"outputs": {}})

    assert sent == []
    assert scheduler.workers["w1"].load == 0
    assert scheduler.get_stats()["dispatch_count"] == 0


def _scheduler_with_loads(*loads: int) -> ClusterAutoscheduler:
    scheduler = ClusterAutoscheduler()
    for index, load in enumerate(loads):
        state = WorkerState(host="h", port=index)
        state.load = load
        scheduler.register_worker(f"w{index}", state)
    return scheduler


def _fired(scheduler: ClusterAutoscheduler) -> List[str]:
    calls: List[str] = []
    scheduler.set_scale_callbacks(scale_up=lambda: calls.append("up"), scale_down=lambda: calls.append("down"))
    scheduler.scale_check()
    return calls


def test_scale_check_scales_up_when_every_worker_is_loaded() -> None:
    assert _fired(_scheduler_with_loads(50, 50, 50)) == ["up"]


def test_scale_check_scales_up_with_no_workers() -> None:
    assert _fired(_scheduler_with_loads()) == ["up"]


def test_scale_check_does_not_scale_up_with_an_idle_worker() -> None:
    assert _fired(_scheduler_with_loads(3, 0, 2)) == []


def test_scale_check_scales_down_when_most_workers_idle() -> None:
    assert _fired(_scheduler_with_loads(0, 0, 1)) == ["down"]
