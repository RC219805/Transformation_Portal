"""V5 consumes exact admitted intent and independently verifies private outputs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tests.lux_depth_v5 import test_pipeline as controlled_pipeline
from transformation_portal.core.execution_plan import ExecutionPlanError
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v5 import pipeline
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
from transformation_portal.orchestrator.execution_dispatch import (
    command_from_dispatch_plan,
    execute_dispatch_plan,
    validate_dispatch_plan,
)
from transformation_portal.orchestrator.photography_adapter import (
    PhotographyBindings,
    consume_photography_dispatch,
    prepare_photography_dispatch,
    verify_photography_dispatch_result,
)

pytestmark = pytest.mark.unit
request_case = controlled_pipeline.request_case


@pytest.fixture
def admitted(request_case, monkeypatch):
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", sys.executable)
    monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
    return prepare_photography_dispatch(request_case, publisher=GenerationPublisher(artifact_store=None, record_store=None))


def test_admission_separates_exact_logical_plan_from_closed_physical_bindings(admitted, request_case):
    plan = validate_dispatch_plan(admitted.plan_bytes, admitted.bindings_bytes)
    bindings = PhotographyBindings(admitted.bindings_bytes).to_payload()
    assert plan.schema == "tp.execution.plan.v4"
    assert plan.canonical_bytes == admitted.plan_bytes
    assert plan.plan_fingerprint_sha256 != hashlib.sha256(admitted.plan_bytes).hexdigest()
    assert bindings["input_root"] == str(request_case.input_dir)
    assert bindings["runtime_python"] == sys.executable
    assert "output_root" not in bindings
    assert str(request_case.input_dir).encode() not in admitted.plan_bytes
    assert not request_case.output_dir.exists()


@pytest.mark.parametrize("field,value", [("argv", ["id"]), ("module", "os"), ("output_root", "/tmp/arbitrary")])
def test_bindings_reject_executable_and_output_authority(admitted, field, value):
    payload = json.loads(admitted.bindings_bytes)
    payload[field] = value
    with pytest.raises(ExecutionPlanError, match="closed"):
        validate_dispatch_plan(admitted.plan_bytes, canonicalize_json(payload))


@pytest.mark.parametrize("path", ["relative", "/tmp/../escape", "/tmp//double", "/tmp/./dot", "/tmp/bad\\name", ""])
def test_bindings_reject_noncanonical_paths(admitted, path):
    payload = json.loads(admitted.bindings_bytes)
    payload["input_root"] = path
    with pytest.raises(ExecutionPlanError, match="canonical absolute"):
        PhotographyBindings(canonicalize_json(payload))


def test_bindings_parser_is_pure_and_does_not_require_local_paths(admitted):
    payload = json.loads(admitted.bindings_bytes)
    payload["input_root"] = "/not/mounted/on/admission/host"
    validate_dispatch_plan(admitted.plan_bytes, canonicalize_json(payload))


def test_optional_namespaces_cannot_be_added_outside_logical_plan(admitted):
    payload = json.loads(admitted.bindings_bytes)
    payload["companion_root"] = "/tmp/companions"
    with pytest.raises(ExecutionPlanError, match="companion bindings"):
        validate_dispatch_plan(admitted.plan_bytes, canonicalize_json(payload))


def test_exact_canonical_carriers_and_mandatory_bindings(admitted):
    with pytest.raises(ExecutionPlanError, match="immutable photography bindings"):
        validate_dispatch_plan(admitted.plan_bytes)
    with pytest.raises(ExecutionPlanError, match="canonical"):
        validate_dispatch_plan(admitted.plan_bytes + b"\n", admitted.bindings_bytes)
    with pytest.raises(ExecutionPlanError, match="closed"):
        validate_dispatch_plan(admitted.plan_bytes, admitted.bindings_bytes + b"\n")


def test_worker_hydrates_without_preparation_or_rediscovery(admitted, tmp_path, monkeypatch):
    from transformation_portal.lux_depth_v5 import lifecycle

    monkeypatch.setattr(lifecycle, "prepare", lambda *_a, **_kw: pytest.fail("Worker re-prepared the admitted plan"))
    prepared = consume_photography_dispatch(admitted.plan_bytes, admitted.bindings_bytes, execution_root=tmp_path / "attempt")
    assert prepared.canonical_plan_bytes == admitted.plan_bytes
    assert prepared.output_root == tmp_path / "attempt"


def test_changed_server_interpreter_policy_cannot_execute(admitted, tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", "/tmp/other-python")
    with pytest.raises(ExecutionPlanError, match="current server policy"):
        consume_photography_dispatch(admitted.plan_bytes, admitted.bindings_bytes, execution_root=tmp_path / "attempt")
    assert not (tmp_path / "attempt").exists()


def test_untrusted_runtime_request_cannot_be_admitted(request_case, monkeypatch):
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", sys.executable)
    with pytest.raises(ExecutionPlanError, match="selected by the server"):
        prepare_photography_dispatch(
            replace(request_case, runtime_python="/tmp/attacker-python"),
            publisher=GenerationPublisher(artifact_store=None, record_store=None),
        )


def test_worker_enforces_exact_publication_limits_before_backend(admitted, tmp_path, monkeypatch):
    prepared = consume_photography_dispatch(admitted.plan_bytes, admitted.bindings_bytes, execution_root=tmp_path / "attempt")
    monkeypatch.setattr(pipeline._ExecutionProfile, "session_type", lambda *_a, **_kw: pytest.fail("Backend initialized"))
    limits = GenerationPublisher(artifact_store=None, record_store=None).limits
    with pytest.raises(ValueError, match="limits changed"):
        pipeline.run(prepared, publication_limits=replace(limits, max_files=limits.max_files + 1))
    assert not prepared.output_root.exists()


def test_dispatch_runs_real_graph_privately_and_exports_verifiable_inventory(admitted, tmp_path, monkeypatch):
    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(tmp_path / "private"))
    output = tmp_path / "attempt"
    assert execute_dispatch_plan(admitted.plan_bytes, execution_bindings=admitted.bindings_bytes, output_root=output) == 0
    result = verify_photography_dispatch_result(admitted.plan_bytes, output_root=output)
    assert result.input_count == 1
    assert result.depth_cache_misses == 1
    assert "input-0000/delivery.tif" in result.artifact_paths
    assert (output / "execution-plan.json").read_bytes() == admitted.plan_bytes
    assert not list((tmp_path / "private").iterdir())
    (output / "input-0000/delivery.tif").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        verify_photography_dispatch_result(admitted.plan_bytes, output_root=output)


def test_fixed_subprocess_rejects_mutated_bindings_before_execution(admitted, tmp_path):
    plan_path = tmp_path / "plan.json"
    bindings_path = tmp_path / "bindings.json"
    output = tmp_path / "attempt"
    plan_path.write_bytes(admitted.plan_bytes)
    bindings_path.write_bytes(admitted.bindings_bytes + b"\n")
    command = command_from_dispatch_plan(
        admitted.plan_bytes,
        output_root=output,
        plan_path=plan_path,
        execution_bindings=admitted.bindings_bytes,
        bindings_path=bindings_path,
    )
    assert command[:3] == [sys.executable, "-m", "transformation_portal.orchestrator.execution_dispatch"]
    result = subprocess.run(
        command,
        capture_output=True,
        timeout=20,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert result.returncode != 0
    assert b"bindings carrier digest does not match" in result.stderr
    assert not output.exists()


def test_inherited_native_cleanup_never_signals_the_parent_group(monkeypatch):
    import signal
    from types import SimpleNamespace

    from transformation_portal.lux_depth_v4 import backend

    signals = []
    child = SimpleNamespace(poll=lambda: None, returncode=None, send_signal=signals.append)
    monkeypatch.setattr(backend.os, "killpg", lambda *_args: pytest.fail("Native cleanup signaled the managed parent"))
    backend._signal_process_group(child, signal.SIGTERM, own_process_group=False)
    assert signals == [signal.SIGTERM]


def test_inherited_native_worker_requires_an_isolated_managed_consumer(monkeypatch):
    from transformation_portal.lux_depth_v4 import backend

    monkeypatch.setattr(backend.os, "getsid", lambda _pid: os.getpid() + 1)
    with pytest.raises(RuntimeError, match="orchestrator-owned isolated session"):
        backend.validate_process_group_ownership(False)
    with pytest.raises(ValueError, match="explicit boolean"):
        backend.validate_process_group_ownership(1)


def test_managed_da3_session_inherits_group_and_reaps_direct_child(admitted, monkeypatch):
    from transformation_portal.lux_depth_v4 import backend
    from transformation_portal.lux_depth_v5.backend import DA3Session

    actual_popen = subprocess.Popen
    children = []
    monkeypatch.setattr(backend.os, "getsid", lambda _pid: os.getpid())
    monkeypatch.setattr(backend.os, "getpgrp", os.getpid)

    def launch(_command, **kwargs):
        assert kwargs["start_new_session"] is False
        child = actual_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    def fail_handshake(*_args):
        raise RuntimeError("Controlled handshake failure")

    monkeypatch.setattr(backend.subprocess, "Popen", launch)
    monkeypatch.setattr(DA3Session, "_rpc", fail_handshake)
    plan = validate_dispatch_plan(admitted.plan_bytes, admitted.bindings_bytes)
    with pytest.raises(RuntimeError, match="Controlled handshake failure"):
        DA3Session(sys.executable, plan, own_process_group=False)
    assert len(children) == 1
    assert children[0].poll() is not None


def test_managed_raw_child_inherits_the_outer_group(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace

    from transformation_portal.lux_depth_v4 import raw

    outer_group = os.getpgrp()
    monkeypatch.setattr(raw.os, "getsid", lambda _pid: os.getpid())
    monkeypatch.setattr(raw.os, "getpgrp", os.getpid)
    monkeypatch.setattr(
        raw,
        "require_process_supervisor",
        lambda: SimpleNamespace(
            Process=lambda _pid: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0), children=lambda **_kw: [])
        ),
    )
    raw._run_command(
        [sys.executable, "-c", "import os; from pathlib import Path; Path('group.txt').write_text(str(os.getpgrp()))"],
        root=tmp_path,
        resources={"memory_mib": 1024, "max_output_bytes": 1024},
        cancellation=None,
        deadline=time.monotonic() + 5,
        log_name="worker.log",
        own_process_group=False,
    )
    assert (tmp_path / "group.txt").read_text() == str(outer_group)
