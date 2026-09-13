"""Owned subprocess-group termination and bounded cleanup contracts."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

import pytest

import app as orchestrator_app

pytestmark = pytest.mark.unit


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX process groups and fork")
@pytest.mark.parametrize("inherit_stdout", [True, False])
@pytest.mark.parametrize("leader_exit", ["during_cancel", "before_cancel"])
def test_terminate_process_kills_resistant_descendant_after_leader_exit(inherit_stdout: bool, leader_exit: str) -> None:
    async def scenario() -> None:
        witness_read, witness_write = os.pipe()
        os.set_blocking(witness_read, False)
        script = """
import os, signal, sys, time
witness = int(sys.argv[1])
ready_read, ready_write = os.pipe()
child = os.fork()
if child == 0:
    os.close(ready_read)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if sys.argv[2] == 'False':
        os.close(1)
    os.write(ready_write, b'1')
    os.close(ready_write)
    time.sleep(30)
else:
    os.close(witness)
    os.close(ready_write)
    os.read(ready_read, 1)
    os.close(ready_read)
    print(child, flush=True)
    if sys.argv[3] == 'during_cancel':
        time.sleep(30)
"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-c",
            script,
            str(witness_write),
            str(inherit_stdout),
            leader_exit,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(witness_write,),
        )
        os.close(witness_write)
        # The successful spawn requested a new session; pin that owned group
        # while the handle belongs to this live execution, as _run_job does.
        setattr(proc, "_tp_owned_process_group_id", proc.pid)
        assert proc.stdout is not None
        try:
            assert int(await asyncio.wait_for(proc.stdout.readline(), 2)) > 0
            if leader_exit == "before_cancel":
                for _ in range(200):
                    if proc.returncode is not None:
                        break
                    await asyncio.sleep(0.01)
                assert proc.returncode == 0
            await asyncio.wait_for(orchestrator_app._terminate_process(proc, grace_seconds=0.05), 1)
            # This pipe has only the descendant's writer. EOF proves it exited
            # even on systems where an orphan zombie briefly retains its PID.
            for _ in range(100):
                try:
                    assert os.read(witness_read, 1) == b""
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.01)
            else:
                pytest.fail("The owned SIGTERM-resistant descendant remains alive")
            assert proc.returncode is not None
        finally:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            await asyncio.wait_for(proc.stdout.read(), 2)
            await asyncio.wait_for(proc.wait(), 2)
            os.close(witness_read)

    asyncio.run(scenario())


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX process groups and fork")
def test_terminate_process_bounds_wait_when_detached_descendant_keeps_stdout() -> None:
    async def scenario() -> None:
        script = """
import os, signal, time
ready_read, ready_write = os.pipe()
child = os.fork()
if child == 0:
    os.close(ready_read)
    os.setsid()
    os.write(ready_write, b'1')
    os.close(ready_write)
    time.sleep(30)
else:
    os.close(ready_write)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.read(ready_read, 1)
    os.close(ready_read)
    print(child, flush=True)
    time.sleep(30)
"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        assert proc.stdout is not None
        child_pid: int | None = None
        try:
            child_pid = int(await asyncio.wait_for(proc.stdout.readline(), 2))
            await asyncio.wait_for(orchestrator_app._terminate_process(proc, grace_seconds=0.05), 1)
            assert proc.returncode is not None
            # Detached groups are outside the parent's verified group authority.
            os.kill(child_pid, 0)
        finally:
            if child_pid is not None:
                with suppress(ProcessLookupError):
                    os.kill(child_pid, signal.SIGKILL)
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            await asyncio.wait_for(proc.stdout.read(), 2)
            await asyncio.wait_for(proc.wait(), 2)

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["windows", "different_group", "different_session", "forged_pin"])
def test_terminate_process_preserves_direct_child_fallback(monkeypatch: pytest.MonkeyPatch, boundary: str) -> None:
    class DirectChild:
        pid = 123
        returncode: int | None = None
        terminated = False

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        async def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    def unexpected_group_signal(*_args: Any) -> None:
        pytest.fail("A non-owned process group must never be signalled")

    monkeypatch.setattr(
        orchestrator_app,
        "os",
        SimpleNamespace(
            name="nt" if boundary == "windows" else "posix",
            getpgid=lambda _pid: 456 if boundary == "different_group" else 123,
            getsid=lambda _pid: 456 if boundary == "different_session" else 123,
            killpg=unexpected_group_signal,
        ),
    )
    proc = DirectChild()
    if boundary == "forged_pin":
        setattr(proc, "_tp_owned_process_group_id", 456)
    asyncio.run(orchestrator_app._terminate_process(proc, grace_seconds=0.05))
    assert proc.terminated
