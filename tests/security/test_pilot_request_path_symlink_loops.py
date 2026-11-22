"""Pilot request paths that run through a symlink loop are rejected at admission on every Python.

Up to Python 3.12 a non-strict ``Path.resolve()`` raised for a symlink loop; Python 3.13 returns a
best-effort path instead, so admission has to detect the loop itself.

These cases live here, not in ``tests/orchestrator/test_managed_photography_api.py``, because that
module's SHA-256 is pinned by the gitleaks workflow contract and the documentation catalog.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock

import pytest

import app

pytestmark = pytest.mark.security


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "1")
    monkeypatch.setenv("TP_ORCHESTRATOR_STATE_BACKEND", "postgres")
    monkeypatch.setenv("TP_ORCHESTRATOR_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", sys.executable)
    monkeypatch.setattr(app, "ALLOWED_INPUT_ROOTS", [tmp_path])
    monkeypatch.setattr(app, "ALLOWED_OUTPUT_ROOTS", [tmp_path])
    (tmp_path / "in").mkdir()
    return {"pipeline": "lux-depth-v5", "args": {"input_dir": str(tmp_path / "in"), "output_dir": str(tmp_path / "out")}}


@pytest.mark.parametrize("scope", ["tenant", "shared"])
@pytest.mark.asyncio
async def test_tenant_request_path_through_a_symlink_loop_is_rejected(configured, monkeypatch, tmp_path, scope):
    monkeypatch.setattr(app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(app, "PILOT_TENANT_WORKSPACE_ROOT", tmp_path / "tenants")
    monkeypatch.setattr(app, "PILOT_TENANT_CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(app, "_PILOT_TENANT_MANAGER", None)
    tenant = app._pilot_tenant_manager().create_tenant("tenant_a")
    if scope == "tenant":
        parent = tmp_path / "tenants" / "tenant_a"
        field = "input_dir"
    else:
        # Shared runtime roots skip the tenant guard, so admission itself
        # must reject a selector it cannot resolve.
        parent = tmp_path / "sam2"
        monkeypatch.setattr(app, "MANAGED_SAM2_TRUSTED_ROOTS", [parent])
        field = "sam2_checkpoint_path"
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "loop").symlink_to("loop")
    configured["args"] = {field: str(parent / "loop" / "selected")}
    error = AsyncMock(return_value="rejected")
    monkeypatch.setattr(app, "_pilot_tenant_error", error)
    assert await app._pilot_enforce_request_paths(tenant, configured, action="job_create", request=None) == "rejected"
    assert error.call_args.kwargs["field"] == field
    assert error.call_args.kwargs["reason"] == "tenant_path_outside_workspace"
