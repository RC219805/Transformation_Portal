"""Tests for GPUSessionPool result routing and task serialization (CPU-only workers).

The worker functions live at module level so the spawn start method can
import them in the child processes.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import pytest

from transformation_portal.runtime.gpu_session_pool import GPUSessionPool, SessionPoolError

pytestmark = [pytest.mark.unit, pytest.mark.regression]

_RESULT_TIMEOUT = 30.0
_INIT_TIMEOUT = 60.0


def _cpu_init(device_id: int) -> Dict[str, Any]:
    return {"device": device_id}


def _pinned_init(device_id: int) -> Dict[str, Any]:
    """Fail unless the worker sees its pinned GPU as the visible ordinal 0."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if device_id != 0 or not visible.isdigit():
        raise RuntimeError(f"init_fn got device_id={device_id} with CUDA_VISIBLE_DEVICES={visible!r}")
    return {"device": device_id, "visible": visible}


def _echo(state: Dict[str, Any], item: str) -> str:
    if item == "boom":
        raise ValueError("bad item")
    return f"result-for-{item}"


def _unpicklable_result(state: Dict[str, Any], item: Any) -> Any:
    return lambda: item


def _visible_device(state: Dict[str, Any], _item: Any) -> str:
    return str(state["visible"])


def test_failed_map_does_not_leak_results_into_next_map() -> None:
    with GPUSessionPool(num_workers=1, init_fn=_cpu_init, init_timeout=_INIT_TIMEOUT) as pool:
        with pytest.raises(SessionPoolError, match="bad item"):
            pool.map(_echo, ["boom", "OLD-b", "OLD-c"], timeout=_RESULT_TIMEOUT)

        got = pool.map(_echo, ["new-a", "new-b", "new-c"], timeout=_RESULT_TIMEOUT)

    assert got == ["result-for-new-a", "result-for-new-b", "result-for-new-c"]


def test_map_keeps_unrelated_results_for_get() -> None:
    with GPUSessionPool(num_workers=1, init_fn=_cpu_init, init_timeout=_INIT_TIMEOUT) as pool:
        pool.submit(_echo, "single")

        got = pool.map(_echo, ["a", "b"], timeout=_RESULT_TIMEOUT)
        single = pool.get(timeout=_RESULT_TIMEOUT)

    assert got == ["result-for-a", "result-for-b"]
    assert single == "result-for-single"


def test_unpicklable_task_fails_at_submit() -> None:
    with GPUSessionPool(num_workers=1, init_fn=_cpu_init, init_timeout=_INIT_TIMEOUT) as pool:
        with pytest.raises(SessionPoolError, match="cannot be sent to a worker"):
            pool.submit(lambda state, x: x, 1)

        # The pool stays usable and no phantom result is pending.
        pool.submit(_echo, "after")
        assert pool.get(timeout=_RESULT_TIMEOUT) == "result-for-after"


def test_unpicklable_result_is_reported_as_task_error() -> None:
    with GPUSessionPool(num_workers=1, init_fn=_cpu_init, init_timeout=_INIT_TIMEOUT) as pool:
        task_id = pool.submit(_unpicklable_result, 1)
        with pytest.raises(SessionPoolError, match=f"Task {task_id} failed: result could not be pickled"):
            pool.get(timeout=_RESULT_TIMEOUT)


def test_get_raises_when_all_workers_have_exited() -> None:
    with GPUSessionPool(num_workers=1, init_fn=_cpu_init, init_timeout=_INIT_TIMEOUT) as pool:
        for proc in pool.workers:
            proc.terminate()
            proc.join(timeout=_RESULT_TIMEOUT)
        with pytest.raises(SessionPoolError, match="workers have exited"):
            pool.get(timeout=5.0)


def test_init_fn_receives_visible_ordinal_zero() -> None:
    with GPUSessionPool(num_workers=2, init_fn=_pinned_init, init_timeout=_INIT_TIMEOUT) as pool:
        visible = pool.map(_visible_device, [0, 1, 2, 3], timeout=_RESULT_TIMEOUT)

    assert set(visible) <= {"0", "1"}
