"""Regression contracts for the shared executor's cache and compiler boundary."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from transformation_portal.core.cas_dag_executor import AuthoritativeStageIdentity, CASDAGConfig, CASDAGExecutor
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.stage_graph.graph import StageGraph
from transformation_portal.stage_graph.stage import Stage, StageContext, StageResult, StageStatus
from transformation_portal.storage.cas_store import ArtifactStore, CASError

pytestmark = pytest.mark.unit


class InputStage(Stage):
    def __init__(self):
        super().__init__("input_stage", "1.0.0")
        self.calls = 0

    def get_cache_key(self, context):
        raise AssertionError("CAS execution must never consult the legacy cache key")

    def compute(self, context):
        self.calls += 1
        return StageResult(self.name, self.version, StageStatus.COMPLETED, artifacts={"result": context.artifacts["image"]})


@pytest.fixture
def setup(tmp_path):
    store = ArtifactStore(tmp_path / "cas")
    executor = CASDAGExecutor(store, tmp_path / "cache", CASDAGConfig(code_paths=[], lockfile_path="requirements/base.txt"))
    stage = InputStage()
    graph = StageGraph("authority")
    graph.add_stage(stage)
    with patch("transformation_portal.core.execution_identity.get_env_fingerprint", return_value="sha256:stable-env"):
        yield executor, store, stage, graph


def test_root_image_content_is_part_of_cache_identity(setup):
    executor, _store, stage, graph = setup
    first = executor.execute(graph, StageContext(artifacts={"image": np.zeros((2, 2, 3), dtype=np.uint8)}))
    second = executor.execute(graph, StageContext(artifacts={"image": np.ones((2, 2, 3), dtype=np.uint8)}))
    assert first.success and second.success
    assert stage.calls == 2
    assert first.identities[stage.name].cas_id != second.identities[stage.name].cas_id
    assert np.all(second.stage_results[stage.name].artifacts["result"] == 1)


@pytest.mark.parametrize("left,right", [(1, 2), ([1], [2]), (1, True), ([1], (1,)), ({"a": 1}, {"b": 1})])
def test_scalar_container_and_key_identity_is_unambiguous(setup, left, right):
    executor, _store, stage, _graph = setup
    first = executor._compute_stage_identity(stage, StageContext(artifacts={"image": left}), {})
    second = executor._compute_stage_identity(stage, StageContext(artifacts={"image": right}), {})
    assert first.cas_id != second.cas_id


@pytest.mark.parametrize("value", [Path("/tmp/input.png"), object(), np.array([object()], dtype=object)])
def test_unmaterialized_or_unsupported_inputs_fail_before_compute(setup, value):
    executor, _store, stage, graph = setup
    result = executor.execute(graph, StageContext(artifacts={"image": value}))
    assert not result.success
    assert stage.calls == 0


def test_same_context_reuse_is_not_polluted_by_prior_outputs(setup):
    executor, _store, stage, graph = setup
    context = StageContext(artifacts={"image": np.ones((2, 2), dtype=np.float32)})
    first = executor.execute(graph, context)
    second = executor.execute(graph, context)
    assert first.success and second.success
    assert second.cache_hits == 1
    assert stage.calls == 1
    assert set(context.artifacts) == {"image"}


def _cached_array(setup):
    executor, store, stage, graph = setup
    context = StageContext(artifacts={"image": np.ones((2, 2), dtype=np.float32)})
    result = executor.execute(graph, context)
    assert result.success
    identity = result.identities[stage.name]
    path = executor._cache_path(identity.cas_id)
    manifest = json.loads(path.read_text())
    obj = store.get_object(manifest["artifacts"]["result"]["sha256"])
    return executor, store, stage, graph, context, path, manifest, obj


def test_tampered_cas_bytes_cannot_be_loaded_even_when_verify_flag_is_false(setup):
    executor, _store, stage, graph, context, _path, _manifest, obj = _cached_array(setup)
    executor.config.verify_on_load = False
    with obj.path.open("wb") as stream:
        np.save(stream, np.full((2, 2), 999, dtype=np.float32))
    result = executor.execute(graph, context)
    assert result.success and result.cache_misses == 1
    assert stage.calls == 2
    assert np.all(result.stage_results[stage.name].artifacts["result"] == 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cas_id", "sha256:" + "f" * 64),
        ("stage_name", "wrong"),
        ("stage_version", "99"),
        ("identity", {}),
        ("unexpected", True),
        ("result_metadata", {"tampered": True}),
    ],
)
def test_tampered_manifest_is_a_cache_miss(setup, field, value):
    executor, _store, stage, graph, context, path, manifest, _obj = _cached_array(setup)
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    result = executor.execute(graph, context)
    assert result.success and result.cache_misses == 1
    assert stage.calls == 2


def test_verified_snapshot_survives_source_replacement(setup):
    _executor, store, _stage, _graph, _context, _path, _manifest, obj = _cached_array(setup)
    with store.open_verified(obj.sha256) as snapshot:
        expected = snapshot.read()
        obj.path.write_bytes(b"replaced after verification")
        snapshot.seek(0)
        assert snapshot.read() == expected


def test_verified_cas_rejects_leaf_symlink(setup, tmp_path):
    _executor, store, _stage, _graph, _context, _path, _manifest, obj = _cached_array(setup)
    outside = tmp_path / "outside.npy"
    outside.write_bytes(obj.path.read_bytes())
    obj.path.unlink()
    obj.path.symlink_to(outside)
    with pytest.raises(CASError):
        with store.open_verified(obj.sha256):
            pytest.fail("symlink was opened")


def test_cas_bypasses_secondary_cache_and_policy_can_disable_cache(setup, tmp_path):
    executor, _store, stage, graph = setup
    context = StageContext(artifacts={"image": 42}, cache_enabled=True, cache_dir=tmp_path / "legacy")
    with (
        patch.object(stage, "_load_from_cache", side_effect=AssertionError("secondary read")),
        patch.object(stage, "_save_to_cache", side_effect=AssertionError("secondary write")),
    ):
        for _ in range(2):
            assert executor.execute(graph, context, cache_policy=lambda _stage, _context: False).success
    assert stage.calls == 2
    assert not context.cache_dir.exists()
    assert not (executor.cache_dir / "dag_cache").exists()


def test_checkpoint_cancellation_after_compute_prevents_cache_publication(setup):
    executor, _store, stage, graph = setup

    def checkpoint(_name, phase, _context):
        if phase == "after_compute":
            raise RuntimeError("lease lost")

    result = executor.execute(graph, StageContext(artifacts={"image": 42}), checkpoint=checkpoint)
    assert not result.success and result.error == "lease lost"
    assert stage.calls == 1
    assert not (executor.cache_dir / "dag_cache").exists()


def test_cache_hit_checks_cancellation_before_propagation(setup):
    executor, _store, stage, graph = setup
    context = StageContext(artifacts={"image": 42})
    assert executor.execute(graph, context).success
    phases = []

    def checkpoint(_name, phase, _context):
        phases.append(phase)
        if phase == "before_propagate":
            raise RuntimeError("canceled cache hit")

    result = executor.execute(graph, context, checkpoint=checkpoint)
    assert not result.success and result.error == "canceled cache hit"
    assert "after_cache" in phases and "before_compute" not in phases
    assert stage.calls == 1


def test_authoritative_provider_binds_exact_identity_payload(setup):
    executor, _store, stage, graph = setup
    context = StageContext(artifacts={"image": 42})
    payload = {"schema": "tp.execution.identity.v4", "inputs": {"image": "a" * 64}}

    def provider(stage, _context, _upstream):
        return AuthoritativeStageIdentity(stage.name, stage.version, "a" * 64, canonicalize_json(payload))

    first = executor.execute(graph, context, identity_provider=provider)
    assert first.success
    assert executor.execute(graph, context, identity_provider=provider).cache_hits == 1
    payload["inputs"]["image"] = "b" * 64
    # Even a trusted-provider bug that reuses a key cannot cross-read identities.
    assert executor.execute(graph, context, identity_provider=provider).cache_misses == 1
    assert stage.calls == 2


def test_direct_stage_legacy_cache_contract_remains_available(tmp_path):
    class LegacyStage(InputStage):
        def get_cache_key(self, context):
            return "legacy-fixed-key"

    stage = LegacyStage()
    context = StageContext(artifacts={"image": 42}, cache_enabled=True, cache_dir=tmp_path / "legacy")
    assert stage.execute(context).is_success()
    result = stage.execute(context)
    assert result.cache_hit and stage.calls == 1


def test_array_descriptor_must_match_verified_npy_header(setup):
    executor, _store, stage, graph, context, path, manifest, _obj = _cached_array(setup)
    manifest["artifacts"]["result"]["shape"] = [4]
    import hashlib

    from transformation_portal.determinism.jcs import dumpb

    manifest["artifacts_sha256"] = hashlib.sha256(dumpb(manifest["artifacts"])).hexdigest()
    manifest["metadata"]["artifact_id"] = manifest["artifacts_sha256"]
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = hashlib.sha256(dumpb(manifest)).hexdigest()
    path.write_text(json.dumps(manifest))
    result = executor.execute(graph, context)
    assert result.success and result.cache_misses == 1
    assert stage.calls == 2


def test_verified_cas_rejects_symlinked_shard(setup, tmp_path):
    _executor, store, _stage, _graph, _context, _path, _manifest, obj = _cached_array(setup)
    shard = obj.path.parent
    moved = tmp_path / "moved-shard"
    shard.rename(moved)
    shard.symlink_to(moved, target_is_directory=True)
    with pytest.raises(CASError):
        with store.open_verified(obj.sha256):
            pytest.fail("symlinked shard was opened")
