"""Fail-closed materialization tests for isolated DA3 cache identity."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from transformation_portal.depth.backends import da3_runtime_identity as identity_module
from transformation_portal.depth.backends.da3_runtime_identity import (
    ParentOutputRuntimeIdentityEvidence,
    bind_parent_output_dependency_identity,
    build_prepared_cache_runtime_evidence,
    load_da3_worker_runtime_handshake,
    prepare_da3_runtime_identity,
    prepare_da3_runtime_identity_with_verification_token,
    runtime_verification_token_sha256,
    verify_parent_output_runtime_identity,
    verify_runtime_verification_token,
)
from transformation_portal.ingest.canonical_json import canonicalize_json

pytestmark = pytest.mark.unit

_MODEL_REVISION = "4" * 40
_SOURCE_REVISION = "9" * 40


def _write_governance(path: Path, *, enabled: bool = True) -> Path:
    lock_path = path.with_name("da3-exact.lock")
    if enabled:
        lock_path.write_bytes(b"torch==2.13.0\ntransformers==5.5.0\n")
        lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    else:
        lock_sha256 = None
    path.write_text(
        json.dumps(
            {
                "schema": "tp.da3.runtime-governance.v1",
                "cache_authority_enabled": enabled,
                "source_repository": "https://example.invalid/Depth-Anything-3",
                "source_revision": _SOURCE_REVISION,
                "dependency_closure_complete": enabled,
                "dependency_lock_path": lock_path.name if enabled else None,
                "dependency_lock_sha256": lock_sha256,
                "governed_additional_distributions": {},
                "runtime_marker_filename": ".tp-da3-runtime-authority.json",
                "non_authorizing_reason": "" if enabled else "fixture disabled",
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_runtime_authority(path: Path, governance_path: Path, *, profile: str = "baseline") -> Path:
    governance = json.loads(governance_path.read_text(encoding="utf-8"))
    payload = {
        "schema": "tp.da3.runtime-authority.v1",
        "cache_authority_enabled": profile == "baseline",
        "profile": profile,
        "python_version": "3.11",
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "dependency_lock_sha256": governance["dependency_lock_sha256"],
        "source_revision": governance["source_revision"],
    }
    path.write_bytes(canonicalize_json(payload))
    return path


def _write_snapshot(root: Path, *, revision: str = _MODEL_REVISION) -> Path:
    snapshot = root / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights-v1")
    (snapshot / "config.json").write_text('{"model_type":"da3"}', encoding="utf-8")
    return snapshot


def _inputs(tmp_path: Path) -> dict[str, Any]:
    source = tmp_path / "runtime.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    governance_path = _write_governance(tmp_path / "governance.json")
    return {
        "model_canonical_key": "da3_metric",
        "model_repo_id": "depth-anything/DA3METRIC-LARGE",
        "model_lock_revision": _MODEL_REVISION,
        "requested_device": "cpu",
        "actual_device": "cpu",
        "executed_backend": "pytorch_cpu",
        "governance_contract_path": governance_path,
        "runtime_authority_path": _write_runtime_authority(tmp_path / ".tp-da3-runtime-authority.json", governance_path),
        "dependency_inventory": (
            {
                "name": "torch",
                "version": "2.13.0",
                "direct_url_sha256": "b" * 64,
                "record_sha256": "c" * 64,
                "installed_files_sha256": "1" * 64,
            },
            {
                "name": "transformers",
                "version": "5.5.0",
                "direct_url_sha256": "d" * 64,
                "record_sha256": "e" * 64,
                "installed_files_sha256": "2" * 64,
            },
        ),
        "source_files": {"runtime.py": source},
        "interpreter_payload": {
            "implementation": "CPython",
            "version": "3.11.15",
            "cache_tag": "cpython-311",
            "soabi": "cpython-311-darwin",
            "executable_sha256": "f" * 64,
            "executable_size_bytes": 1024,
        },
        "platform_payload": {
            "system": "Darwin",
            "release": "fixture",
            "version": "fixture",
            "machine": "arm64",
            "hardware": {"hardware_model": "Mac14,2", "cpu_brand": "Apple M2"},
        },
        "accelerator_payload": {
            "requested_device": "cpu",
            "actual_device": "cpu",
            "executed_backend": "pytorch_cpu",
            "available": True,
            "hardware": {"hardware_model": "Mac14,2", "cpu_brand": "Apple M2"},
        },
        "actual_source_revision": _SOURCE_REVISION,
    }


def _verification_token(
    *,
    entries: list[dict[str, Any]],
    source_revision_probe: dict[str, str] | None = None,
) -> dict[str, Any]:
    import_environment = identity_module._worker_import_environment_payload()
    return {
        "schema": "tp.da3.runtime-verification-token.v1",
        "worker_runtime_identity_sha256": "a" * 64,
        "worker_import_environment_sha256": identity_module._sha256_payload(import_environment),
        "worker_import_environment": import_environment,
        "prepared_runtime": None,
        "source_revision_probe": source_revision_probe,
        "entries": entries,
    }


def test_weight_mutation_changes_materialized_and_runtime_identity(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)

    first = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)
    (snapshot / "model.safetensors").write_bytes(b"weights-v2")
    second = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert first.cacheable is True
    assert second.cacheable is True
    assert (
        first.payload["backend_identity"]["materialized_weights_sha256"]
        != second.payload["backend_identity"]["materialized_weights_sha256"]
    )
    assert first.runtime_identity_sha256 != second.runtime_identity_sha256


def test_model_config_mutation_changes_canonical_model_manifest(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)

    first = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)
    (snapshot / "config.json").write_text('{"model_type":"mutated"}', encoding="utf-8")
    second = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert (
        first.payload["backend_identity"]["materialized_model_sha256"]
        != second.payload["backend_identity"]["materialized_model_sha256"]
    )
    assert first.runtime_identity_sha256 != second.runtime_identity_sha256


def test_snapshot_revision_mismatch_is_non_authorizing(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path, revision="5" * 40)

    evidence = prepare_da3_runtime_identity(snapshot_path=snapshot, **_inputs(tmp_path))

    assert evidence.cacheable is False
    assert evidence.runtime_identity_sha256 is None
    assert "model_snapshot_revision_mismatch" in evidence.payload["incomplete_reasons"]


def test_actual_device_mismatch_is_non_authorizing(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    arguments["actual_device"] = "mps"

    evidence = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert evidence.cacheable is False
    assert evidence.runtime_identity_sha256 is None
    assert "device_mismatch:cpu:mps" in evidence.payload["incomplete_reasons"]


def test_dependency_materialization_changes_runtime_identity(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    first = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)
    changed = dict(arguments)
    changed_inventory = [dict(record) for record in arguments["dependency_inventory"]]
    changed_inventory[0]["installed_files_sha256"] = "3" * 64
    changed["dependency_inventory"] = tuple(changed_inventory)

    second = prepare_da3_runtime_identity(snapshot_path=snapshot, **changed)

    assert first.payload["dependency_identity_sha256"] != second.payload["dependency_identity_sha256"]
    assert first.runtime_identity_sha256 != second.runtime_identity_sha256


@pytest.mark.parametrize(
    ("mutator", "reason_prefix"),
    (
        (lambda records: records[:-1], "dependency_missing:transformers"),
        (
            lambda records: [*records, {**records[0], "name": "unexpected-package"}],
            "dependency_extra:unexpected-package",
        ),
        (
            lambda records: [{**records[0], "version": "2.13.1"}, records[1]],
            "dependency_version_mismatch:torch:2.13.0:2.13.1",
        ),
    ),
)
def test_dependency_closure_mismatch_is_non_authorizing(
    tmp_path: Path,
    mutator: Any,
    reason_prefix: str,
) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    arguments["dependency_inventory"] = tuple(mutator([dict(value) for value in arguments["dependency_inventory"]]))

    evidence = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert evidence.cacheable is False
    assert reason_prefix in evidence.payload["incomplete_reasons"]


def test_complete_evidence_adapts_to_exact_core_runtime_contract(tmp_path: Path) -> None:
    from transformation_portal.lux_depth_v3.config import EnhanceConfig
    from transformation_portal.lux_depth_v3.depth_cache_runtime import PreparedDepthCacheRuntimeEvidence
    from transformation_portal.lux_depth_v3.execution_lifecycle import (
        backend_candidate_authority,
        prepare_lux_execution,
    )

    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(b"fixture")
    prepared = prepare_lux_execution(EnhanceConfig(depth_backend="da3"), tmp_path, [input_path])
    authority = backend_candidate_authority(prepared.plan, "da3")
    resolved = authority.resolved_model_contract
    assert resolved is not None and resolved.revision is not None
    snapshot = _write_snapshot(tmp_path / "snapshot", revision=resolved.revision)
    arguments = _inputs(tmp_path)
    arguments.update(
        {
            "model_canonical_key": resolved.canonical_key,
            "model_repo_id": resolved.spec.repo_id,
            "model_lock_revision": resolved.revision,
        }
    )
    evidence = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    core_evidence = build_prepared_cache_runtime_evidence(
        evidence,
        plan=prepared.plan,
        candidate_authority=authority,
    )

    assert isinstance(core_evidence, PreparedDepthCacheRuntimeEvidence)
    assert len(core_evidence.backend_runtime_identities) == 1
    assert core_evidence.backend_runtime_identities[0].model_canonical_key == resolved.canonical_key
    assert core_evidence.dependency_lock_sha256 == evidence.payload["dependency_lock_sha256"]


def test_missing_dependency_governance_never_fabricates_cache_authority(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    arguments["governance_contract_path"] = tmp_path / "missing.json"

    evidence = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert evidence.cacheable is False
    assert evidence.runtime_identity_sha256 is None
    assert "governance_contract_missing" in evidence.payload["incomplete_reasons"]
    assert "required_identity_digest_missing" in evidence.payload["incomplete_reasons"]


def test_forged_dependency_lock_digest_is_non_authorizing(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    governance_path = arguments["governance_contract_path"]
    governance = json.loads(governance_path.read_text(encoding="utf-8"))
    governance["dependency_lock_sha256"] = "f" * 64
    governance_path.write_text(json.dumps(governance), encoding="utf-8")

    evidence = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert evidence.cacheable is False
    assert evidence.runtime_identity_sha256 is None
    assert "dependency_lock_mismatch" in evidence.payload["incomplete_reasons"]
    assert identity_module.da3_cache_governance_enabled(governance_path) is False


def test_dependency_lock_path_escape_is_non_authorizing(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    governance_path = arguments["governance_contract_path"]
    governance = json.loads(governance_path.read_text(encoding="utf-8"))
    governance["dependency_lock_path"] = "../escaped.lock"
    governance_path.write_text(json.dumps(governance), encoding="utf-8")

    evidence = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert evidence.cacheable is False
    assert evidence.runtime_identity_sha256 is None
    assert "dependency_lock_path_invalid" in evidence.payload["incomplete_reasons"]


def test_incomplete_dependency_closure_cannot_enable_cache_authority(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    governance_path = arguments["governance_contract_path"]
    governance = json.loads(governance_path.read_text(encoding="utf-8"))
    governance["dependency_closure_complete"] = False
    governance_path.write_text(json.dumps(governance), encoding="utf-8")

    evidence = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert evidence.cacheable is False
    assert "governance_contract_invalid" in evidence.payload["incomplete_reasons"]


def test_injected_dependency_inventory_requires_materialized_file_digest(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    arguments["dependency_inventory"] = (
        {
            "name": "torch",
            "version": "2.13.0",
            "direct_url_sha256": "b" * 64,
            "record_sha256": "c" * 64,
        },
    )

    evidence = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert evidence.cacheable is False
    assert evidence.runtime_identity_sha256 is None
    assert "dependency_identity_invalid" in evidence.payload["incomplete_reasons"]


def test_optional_profile_marker_is_non_authorizing(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    arguments["runtime_authority_path"] = _write_runtime_authority(
        tmp_path / "optional-authority.json",
        arguments["governance_contract_path"],
        profile="colmap",
    )

    evidence = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert evidence.cacheable is False
    assert "runtime_authority_marker_mismatch" in evidence.payload["incomplete_reasons"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("interpreter_payload", {"implementation": "CPython", "version": "3.12.9"}, "runtime_python_unsupported"),
        (
            "platform_payload",
            {"system": "Linux", "release": "fixture", "machine": "x86_64", "hardware": {"machine": "x86_64"}},
            "runtime_platform_unsupported",
        ),
    ),
)
def test_non_authoritative_runtime_baseline_is_non_cacheable(
    tmp_path: Path,
    field: str,
    value: dict[str, Any],
    reason: str,
) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    arguments[field] = value

    evidence = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert evidence.cacheable is False
    assert reason in evidence.payload["incomplete_reasons"]


def test_hardware_change_changes_runtime_identity(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    first = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)
    changed = dict(arguments)
    changed["platform_payload"] = {
        **arguments["platform_payload"],
        "hardware": {"hardware_model": "Mac15,3", "cpu_brand": "Apple M3"},
    }
    changed["accelerator_payload"] = {
        **arguments["accelerator_payload"],
        "hardware": {"hardware_model": "Mac15,3", "cpu_brand": "Apple M3"},
    }

    second = prepare_da3_runtime_identity(snapshot_path=snapshot, **changed)

    assert first.cacheable is True and second.cacheable is True
    assert first.runtime_identity_sha256 != second.runtime_identity_sha256


def test_postprocessing_source_mutation_changes_runtime_identity(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    source = tmp_path / "postprocessing.py"
    source.write_text("DEPTH_SCALE = 1\n", encoding="utf-8")
    arguments["source_files"] = {
        "transformation_portal/lux_depth_v3/postprocessing.py": source,
    }
    first = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)
    source.write_text("DEPTH_SCALE = 2\n", encoding="utf-8")
    second = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert first.cacheable is True and second.cacheable is True
    assert first.payload["source_identity_sha256"] != second.payload["source_identity_sha256"]
    assert first.runtime_identity_sha256 != second.runtime_identity_sha256


def test_da3_yaml_resource_mutation_changes_runtime_identity(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    resource = tmp_path / "model.yaml"
    resource.write_text("encoder: large\n", encoding="utf-8")
    arguments["source_files"] = {"depth_anything_3/configs/model.yaml": resource}
    first = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)
    resource.write_text("encoder: giant\n", encoding="utf-8")
    second = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert first.cacheable is True and second.cacheable is True
    assert first.payload["source_identity_sha256"] != second.payload["source_identity_sha256"]


def test_source_revision_tampering_breaks_digest_projection(tmp_path: Path) -> None:
    evidence = prepare_da3_runtime_identity(snapshot_path=_write_snapshot(tmp_path), **_inputs(tmp_path))
    payload = evidence.to_mapping()
    payload["evidence"]["source_revision"] = "8" * 40

    with pytest.raises(ValueError, match="digest projections"):
        identity_module.DA3RuntimeIdentityEvidence.from_mapping(payload)


@pytest.mark.parametrize(
    "forgery",
    ("python", "platform", "source_revision", "device"),
)
def test_self_consistent_forged_cacheable_semantics_are_rejected(tmp_path: Path, forgery: str) -> None:
    payload = prepare_da3_runtime_identity(snapshot_path=_write_snapshot(tmp_path), **_inputs(tmp_path)).to_mapping()
    if forgery == "python":
        payload["evidence"]["interpreter"]["version"] = "3.12.0"
    elif forgery == "platform":
        payload["evidence"]["platform"]["system"] = "Linux"
    elif forgery == "source_revision":
        payload["evidence"]["source_revision"] = None
    else:
        payload["backend_identity"]["actual_device"] = "mps"
        payload["backend_identity"]["executed_backend"] = "pytorch_mps"
        payload["evidence"]["accelerator"]["actual_device"] = "mps"
        payload["evidence"]["accelerator"]["executed_backend"] = "pytorch_mps"

    payload["interpreter_identity_sha256"] = identity_module._sha256_payload(payload["evidence"]["interpreter"])
    payload["platform_identity_sha256"] = identity_module._sha256_payload(payload["evidence"]["platform"])
    payload["accelerator_identity_sha256"] = identity_module._sha256_payload(payload["evidence"]["accelerator"])
    payload["source_identity_sha256"] = identity_module._sha256_payload(
        {
            "schema": identity_module.DA3_SOURCE_IDENTITY_SCHEMA,
            "files": payload["evidence"]["source_files"],
            "optional_source_modules": payload["evidence"]["optional_source_modules"],
            "source_revision": payload["evidence"]["source_revision"],
        }
    )
    payload["runtime_identity_sha256"] = identity_module._sha256_payload(
        {
            "schema": payload["schema"],
            "backend_identity": payload["backend_identity"],
            **{
                field: payload[field]
                for field in (
                    "dependency_lock_sha256",
                    "dependency_identity_sha256",
                    "interpreter_identity_sha256",
                    "platform_identity_sha256",
                    "accelerator_identity_sha256",
                    "source_identity_sha256",
                    "governance_contract_sha256",
                )
            },
        }
    )

    with pytest.raises(ValueError, match="cacheable evidence|governed source"):
        identity_module.DA3RuntimeIdentityEvidence.from_mapping(payload)


def test_exact_lock_parser_rejects_ranges_duplicates_and_markers() -> None:
    assert identity_module._parse_exact_dependency_lock(b"torch==2.13.0\n") == {"torch": "2.13.0"}
    for raw in (
        b"torch>=2.13\n",
        b"torch==2.13.0\ntorch==2.13.0\n",
        b"torch==2.13.0; python_version == '3.11'\n",
    ):
        with pytest.raises(ValueError):
            identity_module._parse_exact_dependency_lock(raw)


def test_checked_in_governance_contract_rejects_old_ambient_runtime_without_marker(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    arguments.pop("governance_contract_path")
    arguments.pop("runtime_authority_path")

    evidence = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert evidence.cacheable is False
    assert "runtime_authority_marker_invalid" in evidence.payload["incomplete_reasons"]


def test_installed_dependency_mutation_is_rejected_against_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_file = tmp_path / "lib" / "python3.11" / "site-packages" / "fake_dep" / "runtime.py"
    record_file = tmp_path / "lib" / "python3.11" / "site-packages" / "fake_dep-1.0.dist-info" / "RECORD"
    package_file.parent.mkdir(parents=True)
    record_file.parent.mkdir(parents=True)
    package_file.write_bytes(b"trusted dependency bytes")
    (record_file.parent / "METADATA").write_text(
        "Name: fake-dep\nVersion: 1.0\n\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(package_file.read_bytes()).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    package_relative = package_file.relative_to(tmp_path).as_posix()
    record_relative = record_file.relative_to(tmp_path).as_posix()
    record_text = f"{package_relative},sha256={encoded},{package_file.stat().st_size}\n" f"{record_relative},,\n"
    record_file.write_text(record_text, encoding="utf-8")

    class FakeDistribution:
        metadata = {"Name": "fake-dep"}
        version = "1.0"
        _path = record_file.parent

        @staticmethod
        def locate_file(relative: str) -> Path:
            return tmp_path / relative

    monkeypatch.setattr(identity_module.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(identity_module.importlib.metadata, "distribution", lambda _name: FakeDistribution())

    identity_module._distribution_record("fake-dep")
    package_file.write_bytes(b"mutated dependency bytes")

    with pytest.raises(ValueError, match="differs from RECORD"):
        identity_module._distribution_record("fake-dep")


def test_distribution_metadata_reads_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root = tmp_path / "lib" / "python3.11" / "site-packages" / "fake_dep-1.0.dist-info"
    metadata_root.mkdir(parents=True)
    (metadata_root / "METADATA").write_text(
        "Name: fake-dep\nVersion: 1.0\n\n",
        encoding="utf-8",
    )
    (metadata_root / "RECORD").write_bytes(b"x" * 9)

    class FakeDistribution:
        metadata = {"Name": "fake-dep"}
        version = "1.0"
        _path = metadata_root

    monkeypatch.setattr(identity_module.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(identity_module, "_MAX_DISTRIBUTION_RECORD_BYTES", 8)
    with pytest.raises(ValueError, match="bounded UTF-8 wheel RECORD"):
        identity_module._distribution_record("fake-dep", distribution=FakeDistribution())

    (metadata_root / "RECORD").write_text("fake_dep-1.0.dist-info/RECORD,,\n", encoding="utf-8")
    (metadata_root / "direct_url.json").write_bytes(b"x" * 9)
    monkeypatch.setattr(identity_module, "_MAX_DISTRIBUTION_RECORD_BYTES", 1024)
    monkeypatch.setattr(identity_module, "_MAX_DISTRIBUTION_DIRECT_URL_BYTES", 8)
    with pytest.raises(ValueError, match="invalid direct-url metadata"):
        identity_module._distribution_record("fake-dep", distribution=FakeDistribution())


def test_distribution_record_rejects_at_first_row_beyond_cardinality_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root = tmp_path / "lib" / "python3.11" / "site-packages" / "fake_dep-1.0.dist-info"
    metadata_root.mkdir(parents=True)
    (metadata_root / "METADATA").write_text("Name: fake-dep\nVersion: 1.0\n\n", encoding="utf-8")
    record_path = metadata_root / "RECORD"
    record_relative = record_path.relative_to(tmp_path).as_posix()
    record_path.write_text(f"{record_relative},,\nsecond.py,,\nthird.py,,\n", encoding="utf-8")

    class FakeDistribution:
        _path = metadata_root

        @staticmethod
        def locate_file(relative: str) -> Path:
            return tmp_path / relative

    monkeypatch.setattr(identity_module.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(identity_module, "_MAX_DISTRIBUTION_FILES", 1)

    with pytest.raises(ValueError, match="empty or unbounded"):
        identity_module._distribution_record("fake-dep", distribution=FakeDistribution())


def test_distribution_import_roots_reject_transitive_shadow_and_bind_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    site_packages = runtime_root / "lib" / "python3.11" / "site-packages"
    metadata_root = site_packages / "fake_dep-1.0.dist-info"
    package_root = site_packages / "tokenizers"
    metadata_root.mkdir(parents=True)
    package_root.mkdir()
    (metadata_root / "METADATA").write_text("Name: fake-dep\nVersion: 1.0\n\n", encoding="utf-8")
    (package_root / "__init__.py").write_text("VALUE = 'governed'\n", encoding="utf-8")
    pycache_root = site_packages / "__pycache__"
    pycache_root.mkdir()
    (pycache_root / "standalone.cpython-311.pyc").write_bytes(b"cached")
    record_path = metadata_root / "RECORD"
    record_path.write_text(
        "tokenizers/__init__.py,,\n"
        "__pycache__/standalone.cpython-311.pyc,,\n"
        f"{record_path.relative_to(site_packages).as_posix()},,\n",
        encoding="utf-8",
    )

    class FakeDistribution:
        _path = metadata_root

        @staticmethod
        def locate_file(relative: str) -> Path:
            return site_packages / relative

    monkeypatch.setattr(identity_module.sys, "prefix", str(runtime_root))
    import_module_names: set[str] = set()
    identity_module._distribution_record(
        "fake-dep",
        distribution=FakeDistribution(),
        import_module_names=import_module_names,
    )
    assert import_module_names == {"tokenizers"}

    attacker_root = tmp_path / "attacker"
    attacker_root.mkdir()
    shadow = attacker_root / "tokenizers.py"
    shadow.write_text("VALUE = 'shadow-1'\n", encoding="utf-8")
    monkeypatch.delitem(identity_module.sys.modules, "tokenizers", raising=False)
    monkeypatch.setattr(identity_module.sys, "path", [str(attacker_root), str(site_packages), *identity_module.sys.path])

    first = identity_module._worker_import_environment_payload(import_module_names)
    tokenizers_record = next(record for record in first["modules"] if record["name"] == "tokenizers")
    assert tokenizers_record["origin"] == str(shadow.resolve())
    assert tokenizers_record["origin_sha256"] == hashlib.sha256(shadow.read_bytes()).hexdigest()
    assert "dependency_import_outside_runtime:tokenizers" in identity_module._worker_import_environment_reasons(first)

    stable_file = tmp_path / "stable-token-input"
    stable_file.write_text("stable\n", encoding="utf-8")
    stable_stat = stable_file.stat()
    token = _verification_token(
        entries=[
            {
                "path": str(stable_file.resolve()),
                "kind": "file",
                "device": stable_stat.st_dev,
                "inode": stable_stat.st_ino,
                "size_bytes": stable_stat.st_size,
                "mtime_ns": stable_stat.st_mtime_ns,
                "ctime_ns": stable_stat.st_ctime_ns,
            }
        ]
    )
    token["worker_import_environment"] = first
    token["worker_import_environment_sha256"] = identity_module._sha256_payload(first)
    round_tripped_token = json.loads(canonicalize_json(token))
    token_sha256 = runtime_verification_token_sha256(round_tripped_token)
    monkeypatch.setattr(identity_module, "_verify_source_revision_probe", lambda _payload: True)
    assert verify_runtime_verification_token(
        round_tripped_token,
        expected_token_sha256=token_sha256,
        expected_worker_runtime_identity_sha256="a" * 64,
        revalidate_worker_import_environment=True,
    )

    shadow.write_text("VALUE = 'shadow-2'\n", encoding="utf-8")
    second = identity_module._worker_import_environment_payload(import_module_names)

    assert tokenizers_record["origin_sha256"] != next(
        record["origin_sha256"] for record in second["modules"] if record["name"] == "tokenizers"
    )
    assert identity_module._sha256_payload(first) != identity_module._sha256_payload(second)
    assert not verify_runtime_verification_token(
        round_tripped_token,
        expected_token_sha256=token_sha256,
        expected_worker_runtime_identity_sha256="a" * 64,
        revalidate_worker_import_environment=True,
    )


def test_import_module_probe_is_bounded_before_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(identity_module, "_MAX_IMPORT_MODULES", 1)
    monkeypatch.setattr(
        identity_module,
        "_module_resolution_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("module resolution ran before bound check")),
    )

    with pytest.raises(ValueError, match="too many modules"):
        identity_module._import_environment_payload(("first_module", "second_module"))

    worker_limit = len(identity_module._WORKER_IMPORT_MODULES) + 1
    monkeypatch.setattr(identity_module, "_MAX_IMPORT_MODULES", worker_limit)
    with pytest.raises(ValueError, match="input is unbounded"):
        identity_module._worker_import_environment_payload("repeated_probe" for _ in range(worker_limit + 1))


def test_worker_import_module_probe_order_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import_root = tmp_path / "imports"
    import_root.mkdir()
    (import_root / "alpha_probe.py").write_text("VALUE = 1\n", encoding="utf-8")
    (import_root / "zeta_probe.py").write_text("VALUE = 2\n", encoding="utf-8")
    monkeypatch.setattr(identity_module.sys, "path", [str(import_root), *identity_module.sys.path])

    first = identity_module._worker_import_environment_payload(("zeta_probe", "alpha_probe"))
    second = identity_module._worker_import_environment_payload(("alpha_probe", "zeta_probe"))

    assert first == second
    assert [record["name"] for record in first["modules"]] == sorted(record["name"] for record in first["modules"])


def test_dependency_import_inventory_overflow_fails_closed_with_bounded_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _inputs(tmp_path)
    dependency_records = arguments.pop("dependency_inventory")
    monkeypatch.setattr(identity_module, "_MAX_IMPORT_MODULES", len(identity_module._WORKER_IMPORT_MODULES) + 1)
    monkeypatch.setattr(
        identity_module,
        "_dependency_inventory",
        lambda _expected: (dependency_records, (), ("overflow_alpha", "overflow_zeta")),
    )

    evidence = prepare_da3_runtime_identity(snapshot_path=_write_snapshot(tmp_path), **arguments)

    assert evidence.cacheable is False
    assert "dependency_import_inventory_unbounded" in evidence.payload["incomplete_reasons"]
    assert [record["name"] for record in evidence.payload["evidence"]["import_environment"]["modules"]] == list(
        identity_module._WORKER_IMPORT_MODULES
    )


def test_distribution_core_metadata_rejects_oversize_and_duplicate_identity_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root = tmp_path / "lib" / "python3.11" / "site-packages" / "fake_dep-1.0.dist-info"
    metadata_root.mkdir(parents=True)

    class FakeDistribution:
        _path = metadata_root

    monkeypatch.setattr(identity_module.sys, "prefix", str(tmp_path))
    (metadata_root / "METADATA").write_text("Name: fake-dep\nVersion: 1.0\n\n", encoding="utf-8")
    monkeypatch.setattr(identity_module, "_MAX_DISTRIBUTION_CORE_METADATA_BYTES", 8)
    with pytest.raises(ValueError, match="bounded core metadata"):
        identity_module._distribution_metadata_identity(FakeDistribution(), distribution_name="fake-dep")

    monkeypatch.setattr(identity_module, "_MAX_DISTRIBUTION_CORE_METADATA_BYTES", 1024)
    (metadata_root / "METADATA").write_text(
        "Name: fake-dep\nName: other-dep\nVersion: 1.0\n\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ambiguous Name/Version"):
        identity_module._distribution_metadata_identity(FakeDistribution(), distribution_name="fake-dep")


def test_model_discovery_and_hashing_enforce_limits_before_unbounded_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _write_snapshot(tmp_path)
    monkeypatch.setattr(identity_module, "_MAX_MODEL_TREE_ENTRIES", 1)
    with pytest.raises(ValueError, match="too many entries"):
        identity_module._materialized_model_manifests(snapshot)

    monkeypatch.setattr(identity_module, "_MAX_MODEL_TREE_ENTRIES", 20_000)
    monkeypatch.setattr(identity_module, "_MAX_WEIGHT_BYTES", 1)
    with pytest.raises(ValueError, match="remaining byte budget"):
        identity_module._materialized_model_manifests(snapshot)


def test_weight_map_cardinality_is_rejected_before_file_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / _MODEL_REVISION
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "missing-a.safetensors", "b": "missing-b.safetensors"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(identity_module, "_MAX_WEIGHT_MAP_ENTRIES", 1)

    with pytest.raises(ValueError, match="too many weight-map entries"):
        identity_module._materialized_model_manifests(snapshot)


def test_new_distribution_directory_invalidates_verification_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "lib" / "python3.11" / "site-packages"
    metadata_root = site_packages / "fake_dep-1.0.dist-info"
    metadata_root.mkdir(parents=True)
    (metadata_root / "METADATA").write_text("Name: fake-dep\nVersion: 1.0\n\n", encoding="utf-8")

    class FakeDistribution:
        _path = metadata_root

    monkeypatch.setattr(identity_module.sys, "prefix", str(tmp_path))
    entries: dict[str, dict[str, Any]] = {}
    context_token = identity_module._VERIFICATION_ENTRIES.set(entries)
    try:
        identity_module._distribution_metadata_identity(FakeDistribution(), distribution_name="fake-dep")
    finally:
        identity_module._VERIFICATION_ENTRIES.reset(context_token)
    token = _verification_token(entries=sorted(entries.values(), key=lambda value: value["path"]))
    monkeypatch.setattr(identity_module, "_verify_source_revision_probe", lambda _payload: True)
    digest = runtime_verification_token_sha256(token)
    assert verify_runtime_verification_token(
        token,
        expected_token_sha256=digest,
        expected_worker_runtime_identity_sha256="a" * 64,
    )

    (site_packages / "new_dep-1.0.dist-info").mkdir()
    assert not verify_runtime_verification_token(
        token,
        expected_token_sha256=digest,
        expected_worker_runtime_identity_sha256="a" * 64,
    )


def test_parent_runtime_governance_is_anchored_to_exact_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    governance_path = _write_governance(tmp_path / "governance.json")
    python_path = tmp_path / "runtime" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_bytes(b"python")
    marker_path = _write_runtime_authority(
        python_path.parent.parent / ".tp-da3-runtime-authority.json",
        governance_path,
    )
    monkeypatch.setattr(
        identity_module,
        "_runtime_python_baseline",
        lambda _path: {
            "implementation": "CPython",
            "python_version": "3.11.15",
            "releaselevel": "final",
            "system": "Darwin",
            "release": "fixture",
            "platform_version": "fixture",
            "machine": "arm64",
            "executable_sha256": "f" * 64,
            "executable_size_bytes": 1024,
        },
    )

    identity = identity_module.da3_cache_runtime_governance_identity(python_path, governance_path)

    assert identity is not None
    assert identity.source_revision == _SOURCE_REVISION
    assert identity.runtime_authority_sha256 == hashlib.sha256(marker_path.read_bytes()).hexdigest()

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["source_revision"] = "8" * 40
    marker_path.write_bytes(canonicalize_json(marker))
    assert identity_module.da3_cache_runtime_governance_identity(python_path, governance_path) is None


def test_same_size_inflight_file_mutation_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_bytes(b"same-size")
    real_fstat = identity_module.os.fstat
    calls = 0

    def changing_fstat(file_descriptor: int):
        nonlocal calls
        calls += 1
        observed = real_fstat(file_descriptor)
        if calls != 2:
            return observed
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_size=observed.st_size,
            st_mtime_ns=observed.st_mtime_ns + 1,
            st_ctime_ns=observed.st_ctime_ns + 1,
        )

    monkeypatch.setattr(identity_module.os, "fstat", changing_fstat)
    with pytest.raises(ValueError, match="changed while hashing"):
        identity_module._hash_regular_file(runtime_file)


def test_governance_contract_rejects_duplicate_keys(tmp_path: Path) -> None:
    governance = tmp_path / "governance.json"
    governance.write_bytes(b'{"schema":"first","schema":"second"}')

    payload, reasons = identity_module._load_governance_contract(governance)

    assert payload is None
    assert reasons == ("governance_contract_invalid",)


def test_worker_handshake_rejects_duplicate_and_noncanonical_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"schema":"first","schema":"second"}')
    with pytest.raises(ValueError, match="repeats key"):
        load_da3_worker_runtime_handshake(duplicate)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(b'{ "schema": "value" }')
    with pytest.raises(ValueError, match="not canonical JSON"):
        load_da3_worker_runtime_handshake(noncanonical)


def test_worker_handshake_enforces_bounded_canonical_payload(tmp_path: Path) -> None:
    handshake = tmp_path / "handshake.json"
    payload = {"schema": "tp.da3.worker-runtime-handshake.v1", "value": "bounded"}
    handshake.write_bytes(canonicalize_json(payload))

    assert load_da3_worker_runtime_handshake(handshake) == payload
    with pytest.raises(ValueError, match="oversized"):
        load_da3_worker_runtime_handshake(handshake, maximum_bytes=8)


def test_stat_token_reuses_unchanged_evidence_and_detects_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``actual_source_revision`` is the closed fixture seam; production tokens
    # carry a live checkout probe which is covered separately below.
    monkeypatch.setattr(identity_module, "_verify_source_revision_probe", lambda _payload: True)
    arguments = _inputs(tmp_path)
    evidence, token = prepare_da3_runtime_identity_with_verification_token(
        snapshot_path=_write_snapshot(tmp_path),
        **arguments,
    )
    assert evidence.cacheable is True and token is not None
    token_sha256 = runtime_verification_token_sha256(token)
    assert verify_runtime_verification_token(
        token,
        expected_token_sha256=token_sha256,
        expected_worker_runtime_identity_sha256=evidence.runtime_identity_sha256,
    )

    arguments["source_files"]["runtime.py"].write_text("VALUE = 2\n", encoding="utf-8")

    assert not verify_runtime_verification_token(
        token,
        expected_token_sha256=token_sha256,
        expected_worker_runtime_identity_sha256=evidence.runtime_identity_sha256,
    )
    reprepared, _new_token = prepare_da3_runtime_identity_with_verification_token(
        snapshot_path=tmp_path / _MODEL_REVISION,
        **arguments,
    )
    assert reprepared.runtime_identity_sha256 != evidence.runtime_identity_sha256


def test_worker_import_environment_is_bound_into_runtime_identity(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    arguments = _inputs(tmp_path)
    first_environment = identity_module._worker_import_environment_payload()
    arguments["import_environment_payload"] = first_environment
    first = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    alternate_root = tmp_path / "alternate-import-root"
    alternate_root.mkdir()
    second_environment = json.loads(json.dumps(first_environment))
    second_environment["search_paths"][0]["path"] = str(alternate_root.resolve())
    arguments["import_environment_payload"] = second_environment
    second = prepare_da3_runtime_identity(snapshot_path=snapshot, **arguments)

    assert first.cacheable is True
    assert second.cacheable is True
    assert first.payload["import_environment_sha256"] != second.payload["import_environment_sha256"]
    assert first.runtime_identity_sha256 != second.runtime_identity_sha256


def test_worker_dependency_import_shadow_is_non_authorizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attacker_root = tmp_path / "attacker"
    attacker_root.mkdir()
    (attacker_root / "addict.py").write_text("VALUE = 'shadow'\n", encoding="utf-8")
    monkeypatch.delitem(identity_module.sys.modules, "addict", raising=False)
    monkeypatch.setattr(identity_module.sys, "path", [str(attacker_root), *identity_module.sys.path])

    payload = identity_module._worker_import_environment_payload()
    addict = next(record for record in payload["modules"] if record["name"] == "addict")

    assert addict["origin"] == str((attacker_root / "addict.py").resolve())
    assert "dependency_import_outside_runtime:addict" in identity_module._worker_import_environment_reasons(payload)


def test_import_environment_hashes_executable_pth_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_root = tmp_path / "site-packages"
    import_root.mkdir()
    path_configuration = import_root / "runtime-path.pth"
    path_configuration.write_text("first-root\n", encoding="utf-8")
    monkeypatch.setattr(identity_module.sys, "path", [str(import_root)])

    first = identity_module._import_environment_payload(())
    path_configuration.write_text("other-root\n", encoding="utf-8")
    second = identity_module._import_environment_payload(())

    assert first["path_configuration_files"][0]["path"] == str(path_configuration.resolve())
    assert first["path_configuration_files"][0]["sha256"] != second["path_configuration_files"][0]["sha256"]
    assert identity_module._sha256_payload(first) != identity_module._sha256_payload(second)


def test_worker_source_inventory_covers_selection_and_plan_helpers() -> None:
    source_files = identity_module._module_source_files("transformation_portal")
    assert {
        "transformation_portal/__init__.py",
        "transformation_portal/core/execution_plan.py",
        "transformation_portal/core/ml_dependency_health.py",
        "transformation_portal/core/security/model_lock.py",
        "transformation_portal/depth/backends/protocol.py",
        "transformation_portal/depth/backends/registry.py",
        "transformation_portal/lux_depth_v3/_backend_contract.py",
        "transformation_portal/lux_depth_v3/execution_lifecycle.py",
        "transformation_portal/lux_depth_v3/execution_plan_adapter.py",
        "transformation_portal/lux_depth_v3/model_registry.py",
        "transformation_portal/lux_depth_v3/resolved_invocation.py",
    }.issubset(source_files)
    assert identity_module._runtime_configuration_file_mapping() == {
        "runtime_config/model_lock_manifest.yaml": (Path("config/model_lock_manifest.yaml").resolve(strict=True))
    }


def test_package_source_identity_changes_for_transitive_helper_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "transformation_portal"
    helper = package_root / "lux_depth_v3" / "stage_graph" / "registry.py"
    helper.parent.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    fake_spec = SimpleNamespace(submodule_search_locations=[str(package_root)], origin=str(package_root / "__init__.py"))
    real_find_spec = identity_module.importlib.util.find_spec
    monkeypatch.setattr(
        identity_module.importlib.util,
        "find_spec",
        lambda name: fake_spec if name == "transformation_portal" else real_find_spec(name),
    )
    source_files = identity_module._module_source_files("transformation_portal")
    first, _records = identity_module._source_identity(
        source_files,
        source_revision=None,
        optional_source_modules=(),
    )

    helper.write_text("VALUE = 2\n", encoding="utf-8")
    second, _records = identity_module._source_identity(
        source_files,
        source_revision=None,
        optional_source_modules=(),
    )

    assert first != second


def test_worker_import_environment_is_revalidated_by_worker_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identity_module, "_verify_source_revision_probe", lambda _payload: True)
    evidence, token = prepare_da3_runtime_identity_with_verification_token(
        snapshot_path=_write_snapshot(tmp_path),
        **_inputs(tmp_path),
    )
    assert evidence.cacheable is True and token is not None
    digest = runtime_verification_token_sha256(token)
    assert verify_runtime_verification_token(
        token,
        expected_token_sha256=digest,
        expected_worker_runtime_identity_sha256=str(evidence.runtime_identity_sha256),
        revalidate_worker_import_environment=True,
    )

    monkeypatch.setenv("HF_HOME", str(tmp_path / "changed-hf-home"))

    assert not verify_runtime_verification_token(
        token,
        expected_token_sha256=digest,
        expected_worker_runtime_identity_sha256=str(evidence.runtime_identity_sha256),
        revalidate_worker_import_environment=True,
    )


def test_final_runtime_and_device_binding_is_authenticated_by_worker_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identity_module, "_verify_source_revision_probe", lambda _payload: True)
    evidence, token = prepare_da3_runtime_identity_with_verification_token(
        snapshot_path=_write_snapshot(tmp_path),
        **_inputs(tmp_path),
    )
    assert evidence.cacheable is True and token is not None
    binding = {
        "schema": "tp.da3.prepared-runtime-binding.v1",
        "runtime_identity_sha256": "7" * 64,
        "requested_device": "cpu",
        "actual_device": "cpu",
        "executed_backend": "pytorch_cpu",
    }
    finalized = identity_module.merge_runtime_verification_entries(
        token,
        (),
        prepared_runtime_binding=binding,
    )
    digest = runtime_verification_token_sha256(finalized)
    expected = {
        "expected_token_sha256": digest,
        "expected_worker_runtime_identity_sha256": str(evidence.runtime_identity_sha256),
        "expected_prepared_runtime_identity_sha256": "7" * 64,
        "expected_requested_device": "cpu",
        "expected_actual_device": "cpu",
        "expected_executed_backend": "pytorch_cpu",
    }

    assert verify_runtime_verification_token(finalized, **expected)
    assert not verify_runtime_verification_token(
        finalized,
        **{**expected, "expected_prepared_runtime_identity_sha256": "8" * 64},
    )
    assert not verify_runtime_verification_token(
        finalized,
        **{**expected, "expected_requested_device": "mps"},
    )


def test_stat_token_revalidates_live_source_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "Depth-Anything-3"
    (repository / ".git").mkdir(parents=True)
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("VALUE = 1\n", encoding="utf-8")
    observed = runtime_file.stat()
    token = _verification_token(
        source_revision_probe={
            "repository_path": str(repository.resolve()),
            "revision": _SOURCE_REVISION,
        },
        entries=[
            {
                "path": str(runtime_file.resolve()),
                "kind": "file",
                "device": observed.st_dev,
                "inode": observed.st_ino,
                "size_bytes": observed.st_size,
                "mtime_ns": observed.st_mtime_ns,
                "ctime_ns": observed.st_ctime_ns,
            }
        ],
    )

    class GitResult:
        returncode = 0
        stdout = f"{_SOURCE_REVISION}\n"

    monkeypatch.setattr(identity_module.subprocess, "run", lambda *_args, **_kwargs: GitResult())
    digest = runtime_verification_token_sha256(token)
    assert verify_runtime_verification_token(
        token,
        expected_token_sha256=digest,
        expected_worker_runtime_identity_sha256="a" * 64,
    )

    GitResult.stdout = f"{'8' * 40}\n"
    assert not verify_runtime_verification_token(
        token,
        expected_token_sha256=digest,
        expected_worker_runtime_identity_sha256="a" * 64,
    )


def test_parent_dependency_drift_changes_prepared_identity(tmp_path: Path) -> None:
    from transformation_portal.lux_depth_v3.config import EnhanceConfig
    from transformation_portal.lux_depth_v3.execution_lifecycle import backend_candidate_authority, prepare_lux_execution

    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(b"fixture")
    execution = prepare_lux_execution(EnhanceConfig(depth_backend="da3"), tmp_path, [input_path])
    authority = backend_candidate_authority(execution.plan, "da3")
    resolved = authority.resolved_model_contract
    assert resolved is not None and resolved.revision is not None
    arguments = _inputs(tmp_path)
    arguments.update(
        {
            "model_canonical_key": resolved.canonical_key,
            "model_repo_id": resolved.spec.repo_id,
            "model_lock_revision": resolved.revision,
        }
    )
    worker = build_prepared_cache_runtime_evidence(
        prepare_da3_runtime_identity(
            snapshot_path=_write_snapshot(tmp_path / "snapshot", revision=resolved.revision),
            **arguments,
        ),
        plan=execution.plan,
        candidate_authority=authority,
    )
    hardware = {"hardware_model": "Mac14,2", "cpu_brand": "Apple M2"}
    platform_payload = {
        "system": "Darwin",
        "release": "fixture",
        "version": "fixture",
        "machine": "arm64",
        "hardware": hardware,
    }
    accelerator_payload = {
        "execution_domain": "parent_output",
        "actual_device": "cpu",
        "available": True,
        "hardware": hardware,
    }
    parent = ParentOutputRuntimeIdentityEvidence(
        interpreter_identity_sha256="a" * 64,
        dependency_identity_sha256="b" * 64,
        source_identity_sha256="c" * 64,
        platform_identity_sha256=identity_module._sha256_payload(platform_payload),
        accelerator_identity_sha256=identity_module._sha256_payload(accelerator_payload),
        import_environment_sha256="f" * 64,
        platform_payload=platform_payload,
        accelerator_payload=accelerator_payload,
        verification_entries=(),
    )
    first = bind_parent_output_dependency_identity(worker, parent_runtime_identity=parent)

    for field_name, digest in (
        ("interpreter_identity_sha256", "1" * 64),
        ("dependency_identity_sha256", "2" * 64),
        ("source_identity_sha256", "3" * 64),
    ):
        changed = bind_parent_output_dependency_identity(
            worker,
            parent_runtime_identity=replace(parent, **{field_name: digest}),
        )
        assert first.runtime_identity_sha256 != changed.runtime_identity_sha256

    changed_platform = {**platform_payload, "release": "changed"}
    changed = bind_parent_output_dependency_identity(
        worker,
        parent_runtime_identity=replace(
            parent,
            platform_identity_sha256=identity_module._sha256_payload(changed_platform),
            platform_payload=changed_platform,
        ),
    )
    assert first.runtime_identity_sha256 != changed.runtime_identity_sha256

    changed_hardware = {**hardware, "cpu_brand": "Apple M3"}
    changed_platform = {**platform_payload, "hardware": changed_hardware}
    changed_accelerator = {**accelerator_payload, "hardware": changed_hardware}
    changed = bind_parent_output_dependency_identity(
        worker,
        parent_runtime_identity=replace(
            parent,
            platform_identity_sha256=identity_module._sha256_payload(changed_platform),
            accelerator_identity_sha256=identity_module._sha256_payload(changed_accelerator),
            platform_payload=changed_platform,
            accelerator_payload=changed_accelerator,
        ),
    )
    assert first.runtime_identity_sha256 != changed.runtime_identity_sha256


def test_parent_import_precedence_drift_invalidates_runtime_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hardware = {"hardware_model": "Mac14,2", "cpu_brand": "Apple M2"}
    monkeypatch.setattr(identity_module, "_hardware_payload", lambda: hardware)
    monkeypatch.setattr(identity_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(identity_module.platform, "release", lambda: "fixture")
    monkeypatch.setattr(identity_module.platform, "version", lambda: "fixture")
    monkeypatch.setattr(identity_module.platform, "machine", lambda: "arm64")
    platform_payload = {
        "system": "Darwin",
        "release": "fixture",
        "version": "fixture",
        "machine": "arm64",
        "hardware": hardware,
    }
    accelerator_payload = {
        "execution_domain": "parent_output",
        "actual_device": "cpu",
        "available": True,
        "hardware": hardware,
    }
    import_environment_sha256 = identity_module._sha256_payload(identity_module._parent_import_environment_payload())
    parent = ParentOutputRuntimeIdentityEvidence(
        interpreter_identity_sha256="a" * 64,
        dependency_identity_sha256="b" * 64,
        source_identity_sha256="c" * 64,
        platform_identity_sha256=identity_module._sha256_payload(platform_payload),
        accelerator_identity_sha256=identity_module._sha256_payload(accelerator_payload),
        import_environment_sha256=import_environment_sha256,
        platform_payload=platform_payload,
        accelerator_payload=accelerator_payload,
        verification_entries=(),
    )
    assert verify_parent_output_runtime_identity(parent)

    attacker_root = tmp_path / "attacker"
    attacker_root.mkdir()
    monkeypatch.setattr(identity_module.sys, "path", [str(attacker_root), *identity_module.sys.path])

    assert not verify_parent_output_runtime_identity(parent)


def test_parent_hardware_drift_invalidates_runtime_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    original_hardware = {"hardware_model": "Mac14,2", "cpu_brand": "Apple M2"}
    monkeypatch.setattr(identity_module, "_hardware_payload", lambda: original_hardware)
    monkeypatch.setattr(identity_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(identity_module.platform, "release", lambda: "fixture")
    monkeypatch.setattr(identity_module.platform, "version", lambda: "fixture")
    monkeypatch.setattr(identity_module.platform, "machine", lambda: "arm64")
    import_environment_sha256 = identity_module._sha256_payload(identity_module._parent_import_environment_payload())
    platform_payload = {
        "system": "Darwin",
        "release": "fixture",
        "version": "fixture",
        "machine": "arm64",
        "hardware": original_hardware,
    }
    accelerator_payload = {
        "execution_domain": "parent_output",
        "actual_device": "cpu",
        "available": True,
        "hardware": original_hardware,
    }
    parent = ParentOutputRuntimeIdentityEvidence(
        interpreter_identity_sha256="a" * 64,
        dependency_identity_sha256="b" * 64,
        source_identity_sha256="c" * 64,
        platform_identity_sha256=identity_module._sha256_payload(platform_payload),
        accelerator_identity_sha256=identity_module._sha256_payload(accelerator_payload),
        import_environment_sha256=import_environment_sha256,
        platform_payload=platform_payload,
        accelerator_payload=accelerator_payload,
        verification_entries=(),
    )
    assert verify_parent_output_runtime_identity(parent)

    monkeypatch.setattr(
        identity_module,
        "_hardware_payload",
        lambda: {"hardware_model": "Mac14,2", "cpu_brand": "Apple M3"},
    )
    assert not verify_parent_output_runtime_identity(parent)
