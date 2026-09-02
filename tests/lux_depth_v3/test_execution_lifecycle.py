"""Native execution-complete Lux lifecycle contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from transformation_portal.core.execution_plan import (
    STRUCTURAL_LEGACY,
    CanonicalExecutionPlan,
    ExecutionPlanError,
    load_execution_plan_schema,
    with_execution_plan_fingerprint,
)
from transformation_portal.depth.backends.registry import DepthBackendRegistry
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.config_resolver import ConfigResolver
from transformation_portal.lux_depth_v3.execution_lifecycle import (
    authorize_prepared_input,
    backend_candidate_authority,
    consume_lux_execution_plan,
    prepare_lux_execution,
    runtime_config_from_execution_plan,
    validate_prepared_lux_execution,
)
from transformation_portal.lux_depth_v3.execution_plan_adapter import LuxExecutionPlanAuthorityError

pytestmark = pytest.mark.unit


def _input_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "inputs"
    root.mkdir()
    image = root / "scene.jpg"
    image.write_bytes(b"not-decoded-by-plan-preparation")
    return root, image


def _synthetic_config(**overrides: object) -> EnhanceConfig:
    values: dict[str, object] = {
        "depth_backend": "synthetic",
        "allow_synthetic_fallback": True,
        "enable_v2": False,
    }
    values.update(overrides)
    return EnhanceConfig(**values)


def test_prepare_is_deterministic_and_does_not_mutate_source(tmp_path: Path) -> None:
    root, image = _input_tree(tmp_path)
    source = EnhanceConfig(
        model_key="da3-metric",
        enable_v2=False,
        mask_feather_sigma_overrides={"glass": 2.5},
    )
    original = copy.deepcopy(source)

    first = prepare_lux_execution(source, root, [image])
    second = prepare_lux_execution(source, root, [image])

    assert first.canonical_plan_bytes == second.canonical_plan_bytes
    assert first.plan.plan_fingerprint_sha256 == second.plan.plan_fingerprint_sha256
    assert source == original
    assert first.runtime_config is not source
    assert first.runtime_config.execution_plan_authority is first.plan
    assert first.plan.configuration_completeness == "execution_complete"


def test_direct_python_and_cli_plan_bytes_are_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image
    from typer.testing import CliRunner

    from transformation_portal.lux_depth_v3.__main__ import app

    root = tmp_path / "inputs"
    root.mkdir()
    image = root / "sample.png"
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(image)
    output = tmp_path / "output"
    monkeypatch.delenv("TP_ALLOW_SYNTHETIC_FALLBACK", raising=False)

    root_logger = logging.getLogger()
    saved_handlers = root_logger.handlers[:]
    saved_level = root_logger.level
    try:
        result = CliRunner().invoke(
            app,
            [
                "--input-dir",
                str(root),
                "--output-dir",
                str(output),
                "--model-key",
                "da3-metric",
                "--plan",
            ],
        )
    finally:
        root_logger.handlers[:] = saved_handlers
        root_logger.setLevel(saved_level)

    assert result.exit_code == 0, result.output
    cli_bytes = next(line.strip().encode("utf-8") for line in result.output.splitlines() if line.strip().startswith("{"))
    direct = prepare_lux_execution(
        EnhanceConfig(model_key="da3-metric", preset_requested="premium"),
        root,
        [image],
    )

    assert cli_bytes == direct.canonical_plan_bytes
    assert not output.exists()


def test_prepared_inputs_are_real_absolute_and_exactly_authorized(tmp_path: Path) -> None:
    root, image = _input_tree(tmp_path)

    prepared = prepare_lux_execution(_synthetic_config(), root, [image])

    assert prepared.input_root == root.resolve()
    assert prepared.input_root.is_absolute()
    assert prepared.input_files == (image.resolve(),)
    assert all(path.is_absolute() and not path.is_symlink() for path in prepared.input_files)
    assert authorize_prepared_input(prepared, image) == image.resolve()


@pytest.mark.security
@pytest.mark.parametrize("boundary", ["preprocess", "enhance_image"])
def test_prepared_orchestrator_uses_authorized_path_after_alias_retarget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    from transformation_portal.lux_depth_v3.input_manager import ImageInput
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    root, image = _input_tree(tmp_path)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    alias = root / "alias.jpg"
    alias.symlink_to(image)
    prepared = prepare_lux_execution(_synthetic_config(), root, [image])

    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator._prepared_execution = prepared
    orchestrator.config = prepared.runtime_config
    orchestrator.depth_dir = tmp_path / "depth"
    orchestrator.manifests_dir = tmp_path / "manifests"
    orchestrator.logs_dir = tmp_path / "logs"
    orchestrator._backend_metadata = object()
    observed: dict[str, object] = {}

    def authorize_then_retarget(path: Path) -> Path:
        resolved = authorize_prepared_input(prepared, path)
        alias.unlink()
        alias.symlink_to(outside)
        return resolved

    monkeypatch.setattr(orchestrator, "_require_prepared_input", authorize_then_retarget)

    if boundary == "preprocess":

        def should_skip_depth(
            _depth_path: Path,
            _manifest_path: Path,
            authorized_input: ImageInput,
        ) -> bool:
            observed["path"] = authorized_input.path
            observed["payload"] = authorized_input.path.read_bytes()
            return False

        monkeypatch.setattr(orchestrator, "should_skip_depth", should_skip_depth)
        result = orchestrator._preprocess_single(ImageInput(alias), root)
        assert result["image_input"].path == image.resolve()
    else:

        class AccessObserved(RuntimeError):
            pass

        def observe_depth_input(**kwargs: object) -> None:
            authorized_input = kwargs["image_input"]
            assert isinstance(authorized_input, ImageInput)
            observed["path"] = authorized_input.path
            observed["payload"] = authorized_input.path.read_bytes()
            raise AccessObserved

        monkeypatch.setattr(orchestrator, "_compute_depth_stage", observe_depth_input)
        with pytest.raises(AccessObserved):
            orchestrator.enhance_image(
                ImageInput(alias),
                input_root=root,
                _precomputed_paths={
                    "output_key": Path("scene"),
                    "depth_path": orchestrator.depth_dir / "scene_depth.png",
                    "manifest_path": orchestrator.manifests_dir / "scene_combined.json",
                    "should_skip": False,
                },
            )

    assert alias.resolve() == outside.resolve()
    assert observed == {
        "path": image.resolve(),
        "payload": b"not-decoded-by-plan-preparation",
    }


@pytest.mark.security
def test_prepared_carrier_rejects_forged_bytes_runtime_and_input_membership(tmp_path: Path) -> None:
    root, image = _input_tree(tmp_path)
    unplanned = root / "unplanned.jpg"
    unplanned.write_bytes(b"unplanned")
    prepared = prepare_lux_execution(_synthetic_config(), root, [image])

    with pytest.raises(LuxExecutionPlanAuthorityError, match="exact canonical bytes"):
        validate_prepared_lux_execution(replace(prepared, canonical_plan_bytes=b"{}"))

    forged_runtime = runtime_config_from_execution_plan(prepared.plan)
    forged_runtime.depth_device = "cuda"
    with pytest.raises(LuxExecutionPlanAuthorityError, match="typed plan projection"):
        validate_prepared_lux_execution(replace(prepared, runtime_config=forged_runtime))

    with pytest.raises(LuxExecutionPlanAuthorityError, match="plan-derived filesystem binding"):
        validate_prepared_lux_execution(replace(prepared, input_files=(unplanned.resolve(),)))


@pytest.mark.security
def test_orchestrator_rejects_forged_prepared_bytes_before_output_initialization(tmp_path: Path) -> None:
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    root, image = _input_tree(tmp_path)
    output_root = tmp_path / "output"
    prepared = prepare_lux_execution(_synthetic_config(), root, [image])

    with pytest.raises(LuxExecutionPlanAuthorityError, match="exact canonical bytes"):
        EnhanceOrchestrator.from_prepared(
            replace(prepared, canonical_plan_bytes=b"{}"),
            output_root,
        )

    assert not output_root.exists()


@pytest.mark.security
def test_consume_rejects_root_mismatch_and_noncanonical_bytes(tmp_path: Path) -> None:
    root, image = _input_tree(tmp_path)
    other_root = tmp_path / "other"
    other_root.mkdir()
    prepared = prepare_lux_execution(_synthetic_config(), root, [image])

    with pytest.raises(LuxExecutionPlanAuthorityError, match="not the authorized root"):
        consume_lux_execution_plan(prepared.canonical_plan_bytes, authorized_input_root=other_root)

    pretty = json.dumps(prepared.plan.to_payload(), indent=2)
    with pytest.raises(ExecutionPlanError, match="exact canonical"):
        consume_lux_execution_plan(pretty, authorized_input_root=root)


@pytest.mark.security
def test_consume_rejects_fingerprint_drift_and_structural_legacy(tmp_path: Path) -> None:
    root, image = _input_tree(tmp_path)
    prepared = prepare_lux_execution(_synthetic_config(), root, [image])

    drifted = prepared.canonical_plan_bytes.replace(b'"quality_tier":"standard"', b'"quality_tier":"premium"')
    with pytest.raises(ExecutionPlanError, match="fingerprint"):
        consume_lux_execution_plan(drifted, authorized_input_root=root)

    legacy = prepared.plan.to_payload()
    legacy["configuration_completeness"] = STRUCTURAL_LEGACY
    for node in legacy["nodes"]:
        node["configuration"]["configuration_completeness"] = STRUCTURAL_LEGACY
    legacy = with_execution_plan_fingerprint(legacy)
    legacy_bytes = CanonicalExecutionPlan.from_payload(legacy).to_canonical_json().encode("utf-8")
    with pytest.raises(LuxExecutionPlanAuthorityError, match="parse-only"):
        consume_lux_execution_plan(legacy_bytes, authorized_input_root=root)


@pytest.mark.security
def test_traversal_and_symlink_swap_fail_closed(tmp_path: Path) -> None:
    root, image = _input_tree(tmp_path)
    prepared = prepare_lux_execution(_synthetic_config(), root, [image])

    traversal = prepared.plan.to_payload()
    traversal["input_selection"]["files"][0]["path"] = "../escape.jpg"
    traversal = with_execution_plan_fingerprint(traversal)
    traversal_bytes = json.dumps(traversal, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with pytest.raises(ExecutionPlanError, match="contained"):
        consume_lux_execution_plan(traversal_bytes, authorized_input_root=root)

    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    image.unlink()
    image.symlink_to(outside)
    with pytest.raises(LuxExecutionPlanAuthorityError, match="escapes|symlink"):
        consume_lux_execution_plan(prepared.canonical_plan_bytes, authorized_input_root=root)


@pytest.mark.security
def test_missing_and_unplanned_inputs_reject_at_access_time(tmp_path: Path) -> None:
    root, image = _input_tree(tmp_path)
    unplanned = root / "unplanned.jpg"
    unplanned.write_bytes(b"unplanned")
    prepared = prepare_lux_execution(_synthetic_config(), root, [image])

    with pytest.raises(LuxExecutionPlanAuthorityError, match="not present"):
        authorize_prepared_input(prepared, unplanned)

    image.unlink()
    with pytest.raises(LuxExecutionPlanAuthorityError, match="cannot be resolved"):
        authorize_prepared_input(prepared, image)
    with pytest.raises(LuxExecutionPlanAuthorityError, match="cannot be resolved"):
        consume_lux_execution_plan(prepared.canonical_plan_bytes, authorized_input_root=root)


def test_model_resolution_occurs_once_across_prepare_project_and_consume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.config_resolver as config_resolver_module
    import transformation_portal.lux_depth_v3.execution_lifecycle as lifecycle

    root, image = _input_tree(tmp_path)
    actual_resolver = lifecycle.resolve_model_contract
    calls = 0

    def counted_resolver(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return actual_resolver(*args, **kwargs)

    monkeypatch.setattr(lifecycle, "resolve_model_contract", counted_resolver)
    monkeypatch.setattr(config_resolver_module, "resolve_model_contract", counted_resolver)
    prepared = lifecycle.prepare_lux_execution(
        EnhanceConfig(model_key="da3-metric", enable_v2=False),
        root,
        [image],
    )
    runtime_config_from_execution_plan(prepared.plan)
    ConfigResolver().resolve(prepared.runtime_config)
    consume_lux_execution_plan(prepared.canonical_plan_bytes, authorized_input_root=root)

    assert calls == 1


def test_prepare_never_constructs_backend_or_creates_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, image = _input_tree(tmp_path)
    before = sorted(path.relative_to(root) for path in root.rglob("*"))

    def forbidden_backend(*args: object, **kwargs: object) -> object:
        raise AssertionError("plan preparation must not instantiate a backend")

    def forbidden_mkdir(*args: object, **kwargs: object) -> None:
        raise AssertionError("plan preparation must not create directories")

    monkeypatch.setattr(DepthBackendRegistry, "get_backend", forbidden_backend)
    monkeypatch.setattr(Path, "mkdir", forbidden_mkdir)

    prepare_lux_execution(_synthetic_config(), root, [image])

    assert sorted(path.relative_to(root) for path in root.rglob("*")) == before


@pytest.mark.parametrize(
    "config,expected_chain",
    (
        (_synthetic_config(), ("synthetic",)),
        (EnhanceConfig(depth_backend="da2", enable_v2=False), ("da2",)),
        (EnhanceConfig(model_key="da3-metric", enable_v2=False), ("da3", "da2")),
        (
            EnhanceConfig(
                depth_backend="depth_pro",
                non_commercial_ok=True,
                accept_apple_depth_pro_research_license=True,
                enable_v2=False,
            ),
            ("depth_pro",),
        ),
        (
            EnhanceConfig(
                depth_backend="ensemble",
                non_commercial_ok=True,
                accept_apple_depth_pro_research_license=True,
                accept_research_tools_license=True,
                model_key="da3-metric",
                enable_v2=False,
            ),
            ("ensemble",),
        ),
    ),
)
def test_current_backend_shapes_are_closed_and_projectable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: EnhanceConfig,
    expected_chain: tuple[str, ...],
) -> None:
    root, image = _input_tree(tmp_path)
    # The repository-wide test harness enables synthetic fallback globally;
    # this matrix verifies the explicit production candidate shapes.
    monkeypatch.delenv("TP_ALLOW_SYNTHETIC_FALLBACK", raising=False)
    if expected_chain != ("synthetic",):
        config.allow_synthetic_fallback = False

    prepared = prepare_lux_execution(config, root, [image])

    assert prepared.plan.candidate_fallback_chain == expected_chain
    for candidate_id in expected_chain:
        authority = backend_candidate_authority(prepared.plan, candidate_id)
        assert authority.candidate_id == candidate_id


def test_da2_cuda_request_freezes_safe_cpu_candidate_device(tmp_path: Path) -> None:
    root, image = _input_tree(tmp_path)
    prepared = prepare_lux_execution(
        EnhanceConfig(depth_backend="da2", depth_device="cuda", enable_v2=False),
        root,
        [image],
    )

    authority = backend_candidate_authority(prepared.plan, "da2")

    assert authority.device == "cpu"
    assert (
        runtime_config_from_execution_plan(
            prepared.plan,
            candidate_authority=authority,
        ).depth_device
        == "cpu"
    )


def test_cuda_da3_fallback_constructs_da2_from_cpu_candidate_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.pipeline_coordinator as coordinator

    root, image = _input_tree(tmp_path)
    monkeypatch.delenv("TP_ALLOW_SYNTHETIC_FALLBACK", raising=False)
    prepared = prepare_lux_execution(
        EnhanceConfig(model_key="da3-metric", depth_device="cuda", enable_v2=False),
        root,
        [image],
    )

    class Backend:
        def __init__(self, name: str, *, available: bool) -> None:
            self.name = name
            self._available = available

        def ensure_available(self) -> None:
            if not self._available:
                raise ImportError("DA3 runtime unavailable")

    class Registry:
        @staticmethod
        def get_backend(
            backend_id: str,
            config: EnhanceConfig,
            *,
            candidate_authority: object,
            canonical_plan_bytes: bytes,
        ) -> Backend:
            assert canonical_plan_bytes == prepared.canonical_plan_bytes
            if backend_id == "da2":
                assert config.depth_device == "cpu"
                assert candidate_authority == backend_candidate_authority(prepared.plan, "da2")
            return Backend(backend_id, available=backend_id == "da2")

    selection = coordinator.select_backend("da3", prepared.runtime_config, Registry())

    assert selection.resolved_backend == "da2"
    assert selection.device == "cpu"


def test_ensemble_constituent_authority_is_exact(tmp_path: Path) -> None:
    root, image = _input_tree(tmp_path)
    prepared = prepare_lux_execution(
        EnhanceConfig(
            depth_backend="ensemble",
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
            accept_research_tools_license=True,
            model_key="da3-metric",
            enable_v2=False,
        ),
        root,
        [image],
    )

    aggregate = backend_candidate_authority(prepared.plan, "ensemble")
    da3 = backend_candidate_authority(prepared.plan, "ensemble", model_backend_id="da3")

    assert aggregate.backend_id == "ensemble"
    assert aggregate.model_contract is None
    assert da3.candidate_id == "ensemble"
    assert da3.backend_id == "da3"
    assert da3.constituent_backend_id == "da3"
    assert da3.resolved_model_contract is not None
    assert da3.resolved_model_contract.canonical_key == "da3_metric"


def test_carried_candidate_chain_does_not_add_environment_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformation_portal.lux_depth_v3.pipeline_coordinator import resolve_runtime_backend_chain

    root, image = _input_tree(tmp_path)
    prepared = prepare_lux_execution(
        EnhanceConfig(depth_backend="da2", enable_v2=False),
        root,
        [image],
    )
    monkeypatch.setenv("TP_ALLOW_SYNTHETIC_FALLBACK", "1")

    assert resolve_runtime_backend_chain("da2", prepared.runtime_config) == ["da2"]


def test_carried_backend_selection_never_reapplies_runtime_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.config_resolver as config_resolver_module
    import transformation_portal.lux_depth_v3.pipeline_coordinator as coordinator

    root, image = _input_tree(tmp_path)
    prepared = prepare_lux_execution(EnhanceConfig(depth_backend="da2", enable_v2=False), root, [image])

    def forbidden_environment_resolution(*args: object, **kwargs: object) -> None:
        raise AssertionError("carried execution must not re-read runtime environment")

    monkeypatch.setattr(coordinator, "apply_effective_da3_runtime_config", forbidden_environment_resolution)
    monkeypatch.setattr(coordinator, "apply_effective_depth_pro_runtime_config", forbidden_environment_resolution)
    monkeypatch.setattr(config_resolver_module, "apply_effective_da3_runtime_config", forbidden_environment_resolution)
    monkeypatch.setattr(config_resolver_module, "apply_effective_raw_runtime_config", forbidden_environment_resolution)
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", "/forged/runtime/python")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON", "/forged/depth-pro/python")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_RAW_PYTHON", "/forged/raw/python")

    class Backend:
        name = "da2"

        @staticmethod
        def ensure_available() -> None:
            return None

    class Registry:
        @staticmethod
        def get_backend(
            backend_id: str,
            config: EnhanceConfig,
            *,
            candidate_authority: object,
            canonical_plan_bytes: bytes,
        ) -> Backend:
            assert backend_id == "da2"
            assert candidate_authority == backend_candidate_authority(prepared.plan, "da2")
            assert canonical_plan_bytes == prepared.canonical_plan_bytes
            backend = Backend()
            backend._candidate_authority = candidate_authority  # type: ignore[attr-defined]
            backend._canonical_plan_bytes = canonical_plan_bytes  # type: ignore[attr-defined]
            return backend

    selection = coordinator.select_backend("da2", prepared.runtime_config, Registry())
    ConfigResolver().resolve(runtime_config_from_execution_plan(prepared.plan))

    assert selection.resolved_backend == "da2"


@pytest.mark.parametrize(
    "selector_source",
    ["explicit_relative", "environment_relative", "path_name", "repo_local"],
)
def test_prepare_freezes_runtime_interpreters_against_cwd_and_path_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector_source: str,
) -> None:
    import transformation_portal.lux_depth_v3.config_resolver as config_resolver_module

    preparation_cwd = tmp_path / "preparation"
    runtime_dir = preparation_cwd / "runtime"
    runtime_dir.mkdir(parents=True)
    runtime_specs = {
        "da3_python_executable": (
            "TRANSFORMATION_PORTAL_DA3_PYTHON",
            "da3-python",
            "_repo_local_da3_python_path",
        ),
        "depth_pro_python_executable": (
            "TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON",
            "depth-pro-python",
            "_repo_local_depth_pro_python_path",
        ),
        "raw_python_executable": (
            "TRANSFORMATION_PORTAL_RAW_PYTHON",
            "raw-python",
            "_repo_local_raw_python_path",
        ),
    }
    expected: dict[str, str] = {}
    for field_name, (environment_name, filename, _repo_resolver_name) in runtime_specs.items():
        executable = runtime_dir / filename
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
        expected[field_name] = str(executable.absolute())
        monkeypatch.delenv(environment_name, raising=False)

    config = _synthetic_config()
    for field_name, (environment_name, filename, repo_resolver_name) in runtime_specs.items():
        executable = runtime_dir / filename
        monkeypatch.setattr(config_resolver_module, repo_resolver_name, lambda: None)
        if selector_source == "explicit_relative":
            setattr(config, field_name, str(Path("runtime") / filename))
        elif selector_source == "environment_relative":
            monkeypatch.setenv(environment_name, str(Path("runtime") / filename))
        elif selector_source == "path_name":
            setattr(config, field_name, filename)
        else:
            monkeypatch.setattr(
                config_resolver_module,
                repo_resolver_name,
                lambda executable=executable: executable,
            )
    if selector_source == "path_name":
        monkeypatch.setenv("PATH", str(runtime_dir))

    root, image = _input_tree(tmp_path)
    monkeypatch.chdir(preparation_cwd)
    prepared = prepare_lux_execution(config, root, [image])

    worker_cwd = tmp_path / "worker"
    worker_cwd.mkdir()
    forged_path = worker_cwd / "forged-bin"
    forged_path.mkdir()
    monkeypatch.chdir(worker_cwd)
    monkeypatch.setenv("PATH", str(forged_path))
    for environment_name, _filename, _repo_resolver_name in runtime_specs.values():
        monkeypatch.setenv(environment_name, "forged-python")

    runtime = runtime_config_from_execution_plan(prepared.plan)
    assert runtime.da3_python_executable == expected["da3_python_executable"]
    assert runtime.depth_pro_python_executable == expected["depth_pro_python_executable"]
    assert runtime.raw_python_executable == expected["raw_python_executable"]
    assert config_resolver_module.resolve_effective_da3_python_executable(runtime) == expected["da3_python_executable"]
    assert (
        config_resolver_module.resolve_effective_depth_pro_python_executable(runtime)
        == expected["depth_pro_python_executable"]
    )
    assert config_resolver_module.resolve_effective_raw_python_executable(runtime) == expected["raw_python_executable"]

    preprocess = next(
        node.configuration for node in prepared.plan.nodes if node.stage_registry_id.value == "tp.stage.lux.preprocess.v1"
    )
    depth = next(node.configuration for node in prepared.plan.nodes if node.stage_registry_id.value == "tp.stage.lux.depth.v1")
    assert preprocess["raw_python_executable"] == expected["raw_python_executable"]
    assert depth["da3_python_executable"] == expected["da3_python_executable"]
    assert depth["depth_pro_python_executable"] == expected["depth_pro_python_executable"]


def test_prepare_rejects_unresolved_path_name_runtime_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.config_resolver as config_resolver_module

    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))
    monkeypatch.setattr(config_resolver_module, "_repo_local_da3_python_path", lambda: None)
    root, image = _input_tree(tmp_path)

    with pytest.raises(ExecutionPlanError, match="not found on preparation PATH"):
        prepare_lux_execution(
            _synthetic_config(da3_python_executable="missing-da3-python"),
            root,
            [image],
        )


@pytest.mark.parametrize("checkpoint_source", ["explicit", "environment", "default"])
def test_prepare_freezes_depth_pro_checkpoint_and_worker_consumes_exact_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_source: str,
) -> None:
    import io
    from types import SimpleNamespace

    from transformation_portal.depth.backends import depth_pro_worker
    from transformation_portal.depth.backends.depth_pro import DepthProBackend

    preparation_cwd = tmp_path / "preparation"
    preparation_cwd.mkdir()
    monkeypatch.delenv("TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT", raising=False)
    config = EnhanceConfig(
        depth_backend="depth_pro",
        non_commercial_ok=True,
        accept_apple_depth_pro_research_license=True,
        enable_v2=False,
    )
    if checkpoint_source == "explicit":
        target = preparation_cwd / "checkpoint-target.pt"
        target.write_bytes(b"checkpoint")
        checkpoint = preparation_cwd / "models" / "depth-pro-link.pt"
        checkpoint.parent.mkdir()
        checkpoint.symlink_to(target)
        config.depth_pro_checkpoint_path = "models/depth-pro-link.pt"
    elif checkpoint_source == "environment":
        checkpoint = preparation_cwd / "environment" / "depth-pro.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint")
        monkeypatch.setenv("TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT", "environment/depth-pro.pt")
    else:
        checkpoint = preparation_cwd / DepthProBackend.DEFAULT_CHECKPOINT

    root, image = _input_tree(tmp_path)
    monkeypatch.chdir(preparation_cwd)
    prepared = prepare_lux_execution(config, root, [image])
    expected = str(checkpoint.absolute())
    authority = backend_candidate_authority(prepared.plan, "depth_pro")

    assert authority.model_contract is not None
    assert authority.model_contract.artifact_path == expected
    assert prepared.runtime_config.depth_pro_checkpoint_path == expected
    if checkpoint_source == "explicit":
        assert Path(expected).is_symlink()
        assert expected != str(Path(expected).resolve())

    worker_cwd = tmp_path / "worker"
    worker_cwd.mkdir()
    monkeypatch.chdir(worker_cwd)
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT", "forged-checkpoint.pt")
    projected = runtime_config_from_execution_plan(prepared.plan, candidate_authority=authority)
    assert projected.depth_pro_checkpoint_path == expected

    monkeypatch.setattr(
        depth_pro_worker.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(prepared.canonical_plan_bytes)),
    )
    worker_authority = depth_pro_worker._consume_canonical_worker_authority(
        candidate_id="depth_pro",
        model_backend_id=None,
    )
    assert worker_authority.checkpoint == Path(expected)


def test_frozen_none_runtime_choices_ignore_post_plan_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.config_resolver as config_resolver_module
    from transformation_portal.depth.backends.da3 import DA3Backend
    from transformation_portal.depth.backends.depth_pro import DepthProBackend
    from transformation_portal.lux_depth_v3.ingest_adapter import build_raw_ingest_options, raw_ingest_summary

    for name in (
        "TRANSFORMATION_PORTAL_DA3_PYTHON",
        "TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON",
        "TRANSFORMATION_PORTAL_RAW_PYTHON",
        "TP_ALLOW_RAW_PREVIEW",
        "TP_ALLOW_SYNTHETIC_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config_resolver_module, "_repo_local_da3_python_path", lambda: None)
    monkeypatch.setattr(config_resolver_module, "_repo_local_depth_pro_python_path", lambda: None)
    monkeypatch.setattr(config_resolver_module, "_repo_local_raw_python_path", lambda: None)

    root, image = _input_tree(tmp_path)
    da3_prepared = prepare_lux_execution(
        EnhanceConfig(model_key="da3-metric", enable_v2=False),
        root,
        [image],
    )
    depth_pro_prepared = prepare_lux_execution(
        EnhanceConfig(
            depth_backend="depth_pro",
            non_commercial_ok=True,
            accept_apple_depth_pro_research_license=True,
            enable_v2=False,
        ),
        root,
        [image],
    )

    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", "/forged/da3/python")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON", "/forged/depth-pro/python")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_RAW_PYTHON", "/forged/raw/python")
    monkeypatch.setenv("TP_ALLOW_RAW_PREVIEW", "1")

    da3_config = runtime_config_from_execution_plan(
        da3_prepared.plan,
        candidate_authority=backend_candidate_authority(da3_prepared.plan, "da3"),
    )
    depth_pro_config = runtime_config_from_execution_plan(
        depth_pro_prepared.plan,
        candidate_authority=backend_candidate_authority(depth_pro_prepared.plan, "depth_pro"),
    )
    assert config_resolver_module.resolve_effective_da3_python_executable(da3_config) is None
    assert config_resolver_module.resolve_effective_depth_pro_python_executable(depth_pro_config) is None
    assert config_resolver_module.resolve_effective_raw_python_executable(da3_config) is None
    assert build_raw_ingest_options(da3_config).raw_python_executable is None
    assert raw_ingest_summary(da3_config)["preview_escape_enabled"] is False

    da3_backend = DA3Backend(
        da3_config,
        candidate_authority=backend_candidate_authority(da3_prepared.plan, "da3"),
        canonical_plan_bytes=da3_prepared.canonical_plan_bytes,
    )
    depth_pro_backend = DepthProBackend(
        depth_pro_config,
        candidate_authority=backend_candidate_authority(depth_pro_prepared.plan, "depth_pro"),
        canonical_plan_bytes=depth_pro_prepared.canonical_plan_bytes,
    )
    assert da3_backend._python_executable is None
    assert depth_pro_backend._python_executable is None


def test_prepared_constructor_rejects_unrelated_config_before_output_initialization(tmp_path: Path) -> None:
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    root, image = _input_tree(tmp_path)
    prepared = prepare_lux_execution(_synthetic_config(), root, [image])
    output_root = tmp_path / "output"

    with pytest.raises(LuxExecutionPlanAuthorityError, match="does not match the authoritative plan projection"):
        EnhanceOrchestrator(
            EnhanceConfig(depth_backend="synthetic", depth_device="cuda", enable_v2=False),
            output_root,
            _prepared_execution=prepared,
        )

    assert not output_root.exists()


@pytest.mark.parametrize("carrier_state", ["paired", "authority_only", "bytes_only"])
def test_direct_constructor_rejects_execution_authority_without_prepared_input_binding(
    tmp_path: Path,
    carrier_state: str,
) -> None:
    from transformation_portal.lux_depth_v3.input_manager import ImageInput
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    root, image = _input_tree(tmp_path)
    prepared = prepare_lux_execution(_synthetic_config(), root, [image])
    carried_config = runtime_config_from_execution_plan(prepared.plan)
    if carrier_state == "authority_only":
        carried_config.execution_plan_canonical_bytes = None
    elif carrier_state == "bytes_only":
        carried_config.execution_plan_authority = None
    output_root = tmp_path / f"output-{carrier_state}"
    unauthorized_input = tmp_path / "outside.jpg"
    unauthorized_input.write_bytes(b"must-not-enter-prepared-execution")

    with pytest.raises(LuxExecutionPlanAuthorityError, match="must be constructed with from_prepared"):
        orchestrator = EnhanceOrchestrator(carried_config, output_root)
        orchestrator.enhance_image(ImageInput(unauthorized_input))  # pragma: no cover - constructor must reject first

    assert not output_root.exists()


def test_from_prepared_constructor_never_applies_ambient_runtime_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformation_portal.lux_depth_v3.orchestrator as orchestrator_module

    root, image = _input_tree(tmp_path)
    prepared = prepare_lux_execution(_synthetic_config(), root, [image])

    def forbidden_runtime_resolution(*args: object, **kwargs: object) -> None:
        raise AssertionError("prepared construction must not re-read ambient runtime defaults")

    monkeypatch.setattr(orchestrator_module, "apply_effective_da3_runtime_config", forbidden_runtime_resolution)
    monkeypatch.setattr(orchestrator_module, "apply_effective_raw_runtime_config", forbidden_runtime_resolution)

    orchestrator = orchestrator_module.EnhanceOrchestrator.from_prepared(prepared, tmp_path / "output")

    assert orchestrator.execution_plan is prepared.plan
    assert orchestrator.config.execution_plan_canonical_bytes == prepared.canonical_plan_bytes


def test_depthcrafter_execution_complete_fails_closed(tmp_path: Path) -> None:
    root, image = _input_tree(tmp_path)

    with pytest.raises(LuxExecutionPlanAuthorityError, match="pinned executable identity"):
        prepare_lux_execution(
            EnhanceConfig(depth_backend="depthcrafter", non_commercial_ok=True, enable_v2=False),
            root,
            [image],
        )


def test_fastvlm_and_reconstruction_environment_is_frozen_into_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    root, image = _input_tree(tmp_path)
    cameras_sidecar = tmp_path / "cameras.json"
    cameras_sidecar.write_text('{"schema":"test"}\n', encoding="utf-8")
    frozen_paths = {
        "TP_FASTVLM_MODEL": tmp_path / "frozen-default-model",
        "TP_FASTVLM_REVIEW_MODEL": tmp_path / "frozen-review-model",
        "TP_FASTVLM_PYTHON": tmp_path / "frozen-python",
        "TP_FASTVLM_MLX_VLM_DIR": tmp_path / "frozen-mlx-vlm",
    }
    for environment_name, path in frozen_paths.items():
        monkeypatch.setenv(environment_name, str(path))
    monkeypatch.setenv("TP_FASTVLM_MAX_TOKENS", "321")
    monkeypatch.setenv("TP_FASTVLM_TEMPERATURE", "0.25")
    monkeypatch.setenv("TP_RECONSTRUCTION_RISK_THRESHOLD", "0.42")

    prepared = prepare_lux_execution(
        _synthetic_config(
            vlm_captioning_enabled=True,
            vlm_captioning_model="review",
            enable_reconstruction=True,
            grouping_mode="parent_dir",
            cameras_sidecar_path=str(cameras_sidecar),
            reconstruction_iterations=77,
            reconstruction_tier="test_research",
            emit_scene_debug_bundle=True,
            reconstruction_risk_threshold=0.9,
            non_commercial_ok=True,
            accept_research_tools_license=True,
        ),
        root,
        [image],
    )

    for environment_name in frozen_paths:
        monkeypatch.setenv(environment_name, str(tmp_path / f"mutated-{environment_name.lower()}"))
    monkeypatch.setenv("TP_FASTVLM_MAX_TOKENS", "999")
    monkeypatch.setenv("TP_FASTVLM_TEMPERATURE", "0.99")
    monkeypatch.setenv("TP_RECONSTRUCTION_RISK_THRESHOLD", "0.01")

    output_node = next(node for node in prepared.plan.nodes if node.stage_registry_id.value == "tp.stage.lux.output.v1")
    captioning = output_node.configuration["captioning"]
    assert captioning["model_path"] == str(frozen_paths["TP_FASTVLM_MODEL"])
    assert captioning["review_model_path"] == str(frozen_paths["TP_FASTVLM_REVIEW_MODEL"])
    assert captioning["python_executable"] == str(frozen_paths["TP_FASTVLM_PYTHON"])
    assert captioning["mlx_vlm_dir"] == str(frozen_paths["TP_FASTVLM_MLX_VLM_DIR"])
    assert captioning["max_tokens"] == 321
    assert captioning["temperature"] == 0.25

    reconstruction_node = next(
        node for node in prepared.plan.nodes if node.stage_registry_id.value == "tp.stage.lux.reconstruction.v1"
    )
    expected_reconstruction = {
        "schema": "tp.stage.config.lux.reconstruction.v1",
        "configuration_completeness": "execution_complete",
        "grouping_mode": "parent_dir",
        "cameras_sidecar_path": str(cameras_sidecar),
        "cameras_sidecar_sha256": hashlib.sha256(cameras_sidecar.read_bytes()).hexdigest(),
        "iterations": 77,
        "tier": "test_research",
        "emit_scene_debug_bundle": True,
        "risk_threshold": 0.42,
    }
    assert dict(reconstruction_node.configuration) == expected_reconstruction

    runtime = runtime_config_from_execution_plan(prepared.plan)
    assert runtime.fastvlm_model_path == str(frozen_paths["TP_FASTVLM_MODEL"])
    assert runtime.fastvlm_review_model_path == str(frozen_paths["TP_FASTVLM_REVIEW_MODEL"])
    assert runtime.fastvlm_python_executable == str(frozen_paths["TP_FASTVLM_PYTHON"])
    assert runtime.fastvlm_mlx_vlm_dir == str(frozen_paths["TP_FASTVLM_MLX_VLM_DIR"])
    assert runtime.fastvlm_max_tokens == 321
    assert runtime.fastvlm_temperature == 0.25
    assert runtime.grouping_mode == "parent_dir"
    assert runtime.cameras_sidecar_path == str(cameras_sidecar)
    assert runtime.cameras_sidecar_sha256 == expected_reconstruction["cameras_sidecar_sha256"]
    assert runtime.reconstruction_iterations == 77
    assert runtime.reconstruction_tier == "test_research"
    assert runtime.emit_scene_debug_bundle is True
    assert runtime.reconstruction_risk_threshold == 0.42

    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator.config = runtime
    default_path, default_role, _ = orchestrator._resolve_vlm_captioning_model_path("default")
    review_path, review_role, _ = orchestrator._resolve_vlm_captioning_model_path("review")
    fastvlm_runtime = orchestrator._fastvlm_runtime_config(review_path)
    assert (default_path, default_role) == (frozen_paths["TP_FASTVLM_MODEL"], "default")
    assert (review_path, review_role) == (frozen_paths["TP_FASTVLM_REVIEW_MODEL"], "review")
    assert fastvlm_runtime.python_path == frozen_paths["TP_FASTVLM_PYTHON"]
    assert fastvlm_runtime.mlx_vlm_dir == frozen_paths["TP_FASTVLM_MLX_VLM_DIR"]
    assert fastvlm_runtime.max_tokens == 321
    assert fastvlm_runtime.temperature == 0.25
    assert orchestrator._effective_reconstruction_risk_threshold() == 0.42


def test_fastvlm_frozen_none_ignores_post_plan_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
    from transformation_portal.vlm_captioning import default_fastvlm_runtime_root, resolve_fastvlm_model_path

    environment_names = (
        "TP_FASTVLM_MODEL",
        "TP_FASTVLM_REVIEW_MODEL",
        "TP_FASTVLM_PYTHON",
        "TP_FASTVLM_MLX_VLM_DIR",
        "TP_FASTVLM_MAX_TOKENS",
        "TP_FASTVLM_TEMPERATURE",
    )
    for environment_name in environment_names:
        monkeypatch.delenv(environment_name, raising=False)
    root, image = _input_tree(tmp_path)
    prepared = prepare_lux_execution(
        _synthetic_config(vlm_captioning_enabled=True),
        root,
        [image],
    )
    for environment_name in environment_names:
        monkeypatch.setenv(environment_name, str(tmp_path / f"forged-{environment_name.lower()}"))

    output_node = next(node for node in prepared.plan.nodes if node.stage_registry_id.value == "tp.stage.lux.output.v1")
    captioning = output_node.configuration["captioning"]
    assert captioning["model_path"] is None
    assert captioning["review_model_path"] is None
    assert captioning["python_executable"] is None
    assert captioning["mlx_vlm_dir"] is None
    assert captioning["max_tokens"] == 120
    assert captioning["temperature"] == 0.0

    runtime = runtime_config_from_execution_plan(prepared.plan)
    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator.config = runtime
    model_path, model_role, _ = orchestrator._resolve_vlm_captioning_model_path("default")
    fastvlm_runtime = orchestrator._fastvlm_runtime_config(model_path)
    runtime_root = default_fastvlm_runtime_root()
    assert model_path == resolve_fastvlm_model_path("default")
    assert model_role == "default"
    assert fastvlm_runtime.python_path == runtime_root / ".venv-fastvlm/bin/python"
    assert fastvlm_runtime.mlx_vlm_dir == runtime_root / "mlx-vlm"
    assert fastvlm_runtime.max_tokens == 120
    assert fastvlm_runtime.temperature == 0.0


def test_legacy_fastvlm_explicit_config_precedes_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    monkeypatch.setenv("TP_FASTVLM_PYTHON", str(tmp_path / "env-python"))
    monkeypatch.setenv("TP_FASTVLM_MLX_VLM_DIR", str(tmp_path / "env-mlx"))
    monkeypatch.setenv("TP_FASTVLM_MAX_TOKENS", "999")
    monkeypatch.setenv("TP_FASTVLM_TEMPERATURE", "0.99")
    orchestrator = object.__new__(EnhanceOrchestrator)
    orchestrator.config = EnhanceConfig(
        fastvlm_python_executable=str(tmp_path / "configured-python"),
        fastvlm_mlx_vlm_dir=str(tmp_path / "configured-mlx"),
        fastvlm_max_tokens=234,
        fastvlm_temperature=0.2,
    )

    runtime = orchestrator._fastvlm_runtime_config(tmp_path / "model")

    assert runtime.python_path == tmp_path / "configured-python"
    assert runtime.mlx_vlm_dir == tmp_path / "configured-mlx"
    assert runtime.max_tokens == 234
    assert runtime.temperature == 0.2


def test_captioning_schema_extensions_are_optional_for_a1_plan_compatibility(
    tmp_path: Path,
) -> None:
    schema = load_execution_plan_schema()
    captioning_schema = schema["$defs"]["captioningConfiguration"]
    for field_name in ("model_path", "review_model_path", "max_tokens", "temperature"):
        assert field_name in captioning_schema["properties"]
        assert field_name not in captioning_schema["required"]

    root, image = _input_tree(tmp_path)
    prepared = prepare_lux_execution(_synthetic_config(vlm_captioning_enabled=True), root, [image])
    a1_payload = prepared.plan.to_payload()
    output_node = next(node for node in a1_payload["nodes"] if node["stage_registry_id"] == "tp.stage.lux.output.v1")
    for field_name in ("model_path", "review_model_path", "max_tokens", "temperature"):
        output_node["configuration"]["captioning"].pop(field_name)
    a1_payload = with_execution_plan_fingerprint(a1_payload)
    a1_plan = CanonicalExecutionPlan.from_payload(a1_payload)

    consumed = consume_lux_execution_plan(a1_plan.to_canonical_json().encode("utf-8"), authorized_input_root=root)
    runtime = runtime_config_from_execution_plan(consumed.plan)

    assert runtime.fastvlm_model_path is None
    assert runtime.fastvlm_review_model_path is None
    assert runtime.fastvlm_max_tokens == 120
    assert runtime.fastvlm_temperature == 0.0
