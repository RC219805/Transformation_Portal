"""Workers enforce current filesystem policy against frozen dispatch plans."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import app
from transformation_portal.core.security.tenant import TenantError
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
from transformation_portal.orchestrator.dispatch import DispatchLocator
from transformation_portal.orchestrator.execution_dispatch import prepare_dispatch_plan
from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost

pytestmark = pytest.mark.unit


@pytest.fixture
def paths(tmp_path, monkeypatch):
    owned = tmp_path / "tenants" / "tenant_a"
    inputs = owned / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "scene.png").write_bytes(b"planning does not decode images")
    for name in ("ALLOWED_INPUT_ROOTS", "ALLOWED_OUTPUT_ROOTS", "ALLOWED_PATH_ROOTS", "FASTVLM_RUNTIME_ALLOWED_ROOTS"):
        monkeypatch.setattr(app, name, [tmp_path])
    monkeypatch.setattr(app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(app, "PILOT_ALLOWED_TENANTS", {"tenant_a"})
    monkeypatch.setattr(app, "PILOT_ALLOWED_PIPELINES", {"lux-depth-v3"})
    monkeypatch.setattr(app, "PILOT_TENANT_WORKSPACE_ROOT", tmp_path / "tenants")
    monkeypatch.setattr(app, "PILOT_TENANT_CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(app, "_PILOT_TENANT_MANAGER", None)
    return inputs, owned / "attempt"


def _frozen(inputs: Path, **overrides):
    config = EnhanceConfig(depth_backend="synthetic", allow_synthetic_fallback=True, enable_v2=False, **overrides)
    data = prepare_lux_execution(config, inputs, [inputs / "scene.png"]).canonical_plan_bytes
    locator = DispatchLocator("job_policy", "attempt_policy", "dispatch_policy", hashlib.sha256(data).hexdigest(), "tenant_a")
    return locator, data


@pytest.mark.parametrize("root_kind", ["ALLOWED_INPUT_ROOTS", "ALLOWED_OUTPUT_ROOTS"])
def test_worker_root_revocation_rejects_frozen_plan_before_output_creation(paths, monkeypatch, tmp_path, root_kind):
    inputs, output = paths
    locator, data = _frozen(inputs)
    app._revalidate_dispatch_paths(locator, data, output)
    monkeypatch.setattr(app, root_kind, [tmp_path / "new_policy"])
    with pytest.raises(ValueError, match="outside allowed roots"):
        app._revalidate_dispatch_paths(locator, data, output)
    assert not output.exists()


def test_worker_camera_sidecar_rechecks_current_input_policy(paths, monkeypatch):
    inputs, output = paths
    sidecar = inputs.parent / "cameras.json"
    sidecar.write_text("{}")
    locator, data = _frozen(
        inputs,
        enable_reconstruction=True,
        cameras_sidecar_path=str(sidecar),
        non_commercial_ok=True,
        accept_research_tools_license=True,
    )
    app._revalidate_dispatch_paths(locator, data, output)
    monkeypatch.setattr(app, "ALLOWED_INPUT_ROOTS", [inputs])
    with pytest.raises(ValueError, match="outside allowed roots"):
        app._revalidate_dispatch_paths(locator, data, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "field", ["fastvlm_model_path", "fastvlm_review_model_path", "fastvlm_python_executable", "fastvlm_mlx_vlm_dir"]
)
def test_worker_fastvlm_shared_root_revocation_rechecks_carried_paths(paths, monkeypatch, tmp_path, field):
    inputs, output = paths
    shared = tmp_path / "old_shared_runtime"
    monkeypatch.setattr(app, "default_fastvlm_runtime_root", lambda: shared)
    locator, data = _frozen(inputs, vlm_captioning_enabled=True, **{field: str(shared / "artifact")})
    app._revalidate_dispatch_paths(locator, data, output)
    monkeypatch.setattr(app, "default_fastvlm_runtime_root", lambda: tmp_path / "new_shared_runtime")
    with pytest.raises(TenantError):
        app._revalidate_dispatch_paths(locator, data, output)
    assert not output.exists()


def test_worker_sam2_shared_root_revocation_rechecks_carried_checkpoint(paths, monkeypatch, tmp_path):
    inputs, output = paths
    shared = tmp_path / "old_shared_models"
    monkeypatch.setattr(app, "MANAGED_SAM2_TRUSTED_ROOTS", [shared])
    locator, data = _frozen(inputs, enable_materials_v3=True, sam2_checkpoint_path=str(shared / "model.pt"))
    app._revalidate_dispatch_paths(locator, data, output)
    monkeypatch.setattr(app, "MANAGED_SAM2_TRUSTED_ROOTS", [tmp_path / "new_shared_models"])
    with pytest.raises(TenantError):
        app._revalidate_dispatch_paths(locator, data, output)
    assert not output.exists()


def test_worker_changed_interpreter_selection_does_not_alias_shared_binary(paths, monkeypatch, tmp_path):
    inputs, output = paths
    old_python = tmp_path / "old_venv_python"
    new_python = tmp_path / "new_venv_python"
    old_python.symlink_to(Path(app.sys.executable).resolve())
    new_python.symlink_to(Path(app.sys.executable).resolve())
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", str(old_python))
    locator, data = _frozen(inputs)
    app._revalidate_dispatch_paths(locator, data, output)
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", str(new_python))
    assert old_python.resolve() == new_python.resolve()
    with pytest.raises(DispatchAuthorityLost, match="worker runtime selection changed"):
        app._revalidate_dispatch_paths(locator, data, output)
    assert not output.exists()


def test_worker_tenant_membership_revocation_rejects_frozen_plan(paths, monkeypatch):
    inputs, output = paths
    locator, data = _frozen(inputs)
    app._revalidate_dispatch_paths(locator, data, output)
    monkeypatch.setattr(app, "PILOT_ALLOWED_TENANTS", {"tenant_b"})
    with pytest.raises(DispatchAuthorityLost):
        app._revalidate_dispatch_paths(locator, data, output)
    assert not output.exists()


def _frozen_rights(paths, monkeypatch, *, policy=None, manifest=None):
    inputs, output = paths
    monkeypatch.setattr(app, "PILOT_ALLOWED_PIPELINES", {"archive-gate-a"})
    governed = app.REPO_ROOT / "policy" / "archive"
    # Keep the governed policy's global input authorization independent of
    # tenant ownership. Revoking either policy must still reject the plan.
    for name in ("ALLOWED_INPUT_ROOTS", "ALLOWED_PATH_ROOTS"):
        monkeypatch.setattr(app, name, [*getattr(app, name), governed])
    if manifest is None:
        manifest = inputs / "manifest.jsonl"
        manifest.write_text('{"fixture":true}\n')
    args = {
        "archive_command": "rights-apply",
        "input_dir": str(inputs),
        "output_dir": str(output),
        "manifest_jsonl": str(manifest),
    }
    if policy is not None:
        args["policy_yaml"] = str(policy)
    request = {"pipeline": "archive-gate-a", "args": args}
    command = app._archive_gate_argv("archive-gate-a", args, str(inputs), str(output))
    data = prepare_dispatch_plan(request, trusted_argv=command)
    locator = DispatchLocator("job_rights", "attempt_rights", "dispatch_rights", hashlib.sha256(data).hexdigest(), "tenant_a")
    return locator, data


@pytest.mark.parametrize("policy_kind", ["default", "explicit_governed", "tenant_owned"])
def test_worker_rights_policy_preserves_governed_default_and_tenant_overrides(paths, monkeypatch, policy_kind):
    inputs, output = paths
    policy = None
    if policy_kind == "explicit_governed":
        policy = app.REPO_ROOT / "policy" / "archive" / "rights_flags.yml"
    elif policy_kind == "tenant_owned":
        policy = inputs / "custom_policy.yml"
        policy.write_text("version: 1\n")
    locator, data = _frozen_rights(paths, monkeypatch, policy=policy)
    app._revalidate_dispatch_paths(locator, data, output)
    assert not output.exists()


def test_worker_governed_policy_still_requires_current_input_root_authorization(paths, monkeypatch):
    inputs, output = paths
    locator, data = _frozen_rights(paths, monkeypatch)
    # Output authorization cannot substitute for revoked input authorization.
    governed = app.ARCHIVE_RIGHTS_POLICY_ROOT
    monkeypatch.setattr(app, "ALLOWED_INPUT_ROOTS", [inputs.parent])
    monkeypatch.setattr(app, "ALLOWED_OUTPUT_ROOTS", [*app.ALLOWED_OUTPUT_ROOTS, governed])
    monkeypatch.setattr(app, "ALLOWED_PATH_ROOTS", list(dict.fromkeys([*app.ALLOWED_INPUT_ROOTS, *app.ALLOWED_OUTPUT_ROOTS])))
    with pytest.raises(ValueError, match="outside allowed roots"):
        app._revalidate_dispatch_paths(locator, data, output)
    assert not output.exists()


def test_worker_governed_policy_exemption_does_not_authorize_archive_data(paths, monkeypatch):
    _, output = paths
    governed_file = app.REPO_ROOT / "policy" / "archive" / "rights_flags.yml"
    monkeypatch.setattr(app, "ALLOWED_OUTPUT_ROOTS", [*app.ALLOWED_OUTPUT_ROOTS, governed_file.parent])
    locator, data = _frozen_rights(paths, monkeypatch, manifest=governed_file)
    with pytest.raises(TenantError):
        app._revalidate_dispatch_paths(locator, data, output)
    assert not output.exists()


def test_worker_rights_policy_rejects_another_tenants_file(paths, monkeypatch):
    inputs, output = paths
    policy = inputs.parents[1] / "tenant_b" / "policy.yml"
    policy.parent.mkdir()
    policy.write_text("version: 1\n")
    locator, data = _frozen_rights(paths, monkeypatch, policy=policy)
    with pytest.raises(TenantError):
        app._revalidate_dispatch_paths(locator, data, output)
    assert not output.exists()
