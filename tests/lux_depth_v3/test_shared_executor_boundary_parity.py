"""Test-only V3 boundary parity; this does not authorize executor migration.

The production ``_compute_depth_stage`` still interleaves snapshot preprocessing,
runtime authority, fallback, semantic gates, resize, Materials, output writing,
and manifest state. Extracting its complete semantics is a separate prerequisite
for the ADR-051 V3 activation gate. These tests deliberately exercise only the
existing preprocessing, identity-v3/cache, and quantized-writer boundaries. The
inference function is a deterministic fixture, never real-model evidence. No V3
resize or fallback implementation is copied into this harness.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from tests.lux_depth_v3.test_depth_cache_runtime import _prepared_da2_evidence
from transformation_portal.core.cas_dag_executor import AuthoritativeStageIdentity, CASDAGConfig, CASDAGExecutor
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3.depth_cache import DEPTH_CACHE_SCHEMA, DepthCache
from transformation_portal.lux_depth_v3.depth_writer import atomic_write_depth_u16_png_with_stats
from transformation_portal.lux_depth_v3.execution_lifecycle import validate_prepared_lux_execution
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
from transformation_portal.lux_depth_v3.preprocessing import preprocess_image_snapshot
from transformation_portal.stage_graph.graph import StageGraph
from transformation_portal.stage_graph.stage import Stage, StageContext, StageResult, StageStatus
from transformation_portal.storage.cas_store import ArtifactStore

pytestmark = pytest.mark.unit


class _Boundary(Stage):
    def __init__(self, name, dependencies, operation):
        super().__init__(name, "1.0.0")
        self.dependencies = dependencies
        self.operation = operation

    def get_dependencies(self):
        return self.dependencies

    def get_cache_key(self, context):
        raise AssertionError("The legacy Stage cache cannot authorize this harness")

    def compute(self, context):
        return StageResult(self.name, self.version, StageStatus.COMPLETED, artifacts=self.operation(context))


class _FixtureBackend:
    """Explicit synthetic fixture with the real V3 runtime-evidence handshake."""

    def __init__(self, prepared, evidence):
        self.prepared = prepared
        self.evidence = evidence
        self.live = True
        self.calls = 0

    def prepare_cache_runtime_identity(self, **kwargs):
        assert kwargs == {
            "execution_plan": self.prepared.plan,
            "candidate_id": "da2",
            "canonical_plan_bytes": self.prepared.canonical_plan_bytes,
        }
        return self.evidence

    def verify_prepared_cache_runtime_identity(self, *, runtime_identity_sha256):
        return self.live and runtime_identity_sha256 == self.evidence.runtime_identity_sha256

    @staticmethod
    def fixture_depth(image):
        return np.ascontiguousarray(image[:, :, 0], dtype=np.float32)

    def infer(self, image):
        self.calls += 1
        return SimpleNamespace(
            depth=self.fixture_depth(image),
            metadata={"runtime_identity_sha256": self.evidence.runtime_identity_sha256},
        )


class _ParityHarness:
    """Tests scheduling of unchanged primitives; never a production adapter."""

    def __init__(self, tmp_path, *, shape=(31, 45), orientation=1):
        self.prepared, self.image, evidence = _prepared_da2_evidence(tmp_path)
        pixels = np.arange(np.prod(shape) * 3, dtype=np.uint8).reshape(*shape, 3)
        exif = Image.Exif()
        exif[274] = orientation
        Image.fromarray(pixels).save(self.image, exif=exif)
        self.input_sha256 = hashlib.sha256(self.image.read_bytes()).hexdigest()
        self.backend = _FixtureBackend(self.prepared, evidence)
        self.orchestrator = object.__new__(EnhanceOrchestrator)
        self.orchestrator._prepared_execution = self.prepared
        self.authority = self._authority()
        self.cache = DepthCache(tmp_path / "v3-cache")
        self.executor = CASDAGExecutor(
            ArtifactStore(tmp_path / "cas"),
            tmp_path / "shared",
            CASDAGConfig(enable_caching=False, parallel=False, code_paths=[], lockfile_path="requirements/base.txt"),
        )
        self.output = tmp_path / "shared.png"
        self.graph = StageGraph("v3-test-only-boundary-parity")
        self.graph.add_stage(_Boundary("preprocess", [], self._preprocess))
        self.graph.add_stage(_Boundary("depth", ["preprocess"], self._depth))
        self.graph.add_stage(_Boundary("output", ["depth"], self._output))

    def _authority(self):
        authority = self.orchestrator._prepare_depth_cache_authority(
            backend=self.backend,
            backend_id="da2",
            image_path=self.image,
            input_content_sha256=self.input_sha256,
        )
        if authority is None:
            raise RuntimeError("Fixture runtime authority was revoked")
        return authority

    def _preprocess(self, _context):
        image, shape, digest = preprocess_image_snapshot(self.image, raw_config=self.prepared.runtime_config)
        if digest != self.input_sha256:
            raise RuntimeError("Fixture input bytes changed")
        return {"image": image, "original_shape": shape}

    def _depth(self, context):
        authority = self._authority()
        assert authority.identity.to_canonical_bytes() == self.authority.identity.to_canonical_bytes()
        depth = self.cache.get(authority.identity)
        if depth is None:
            result = self.backend.infer(context.artifacts["image"])
            self.orchestrator._verify_depth_cache_runtime_echo(result, authority, backend_id="da2")
            self._authority()
            depth = result.depth
            assert self.cache.store(authority.identity, depth)
        self._authority()
        return {"depth": depth}

    def _output(self, context):
        _, _, stats = atomic_write_depth_u16_png_with_stats(
            self.output,
            context.artifacts["depth"],
            method=self.prepared.runtime_config.depth_quantization,
        )
        return {"stats": stats}

    def _identity(self, stage, _context, _upstream):
        if stage.name == "depth":
            identity = self.authority.identity
            return AuthoritativeStageIdentity(
                stage.name, stage.version, identity.cache_key(DEPTH_CACHE_SCHEMA), identity.to_canonical_bytes()
            )
        # Non-depth nodes deliberately have no reusable authorizing cache.
        payload = canonicalize_json({"schema": "tp.test.v3-boundary-parity.v1", "stage": stage.name})
        return AuthoritativeStageIdentity(stage.name, stage.version, hashlib.sha256(payload).hexdigest(), payload)

    def _checkpoint(self, _stage, _phase, _context):
        validate_prepared_lux_execution(self.prepared)
        if hashlib.sha256(self.image.read_bytes()).hexdigest() != self.input_sha256:
            raise RuntimeError("Fixture input bytes changed")
        self._authority()

    def run(self, checkpoint=None):
        return self.executor.execute(
            self.graph,
            StageContext(),
            identity_provider=self._identity,
            checkpoint=checkpoint or self._checkpoint,
            cache_policy=lambda _stage, _context: False,
        )


@pytest.mark.parametrize("shape,orientation", [((28, 42), 1), ((31, 45), 1), ((31, 45), 6)])
def test_v3_primitive_bytes_and_identity_remain_equal_through_shared_scheduling(tmp_path, shape, orientation):
    harness = _ParityHarness(tmp_path, shape=shape, orientation=orientation)
    direct_image, direct_shape, digest = preprocess_image_snapshot(harness.image, raw_config=harness.prepared.runtime_config)
    direct_depth = harness.backend.fixture_depth(direct_image)
    expected = tmp_path / "direct.png"
    _, _, direct_stats = atomic_write_depth_u16_png_with_stats(expected, direct_depth)
    plan_bytes = harness.prepared.canonical_plan_bytes
    identity_bytes = harness.authority.identity.to_canonical_bytes()

    with (
        patch.object(Stage, "_load_from_cache", side_effect=AssertionError("legacy cache read")),
        patch.object(Stage, "_save_to_cache", side_effect=AssertionError("legacy cache write")),
    ):
        first = harness.run()
        second = harness.run()

    assert first.success and second.success
    assert first.execution_order == ["preprocess", "depth", "output"]
    np.testing.assert_array_equal(first.stage_results["preprocess"].artifacts["image"], direct_image)
    assert first.stage_results["preprocess"].artifacts["original_shape"] == direct_shape
    np.testing.assert_array_equal(first.stage_results["depth"].artifacts["depth"], direct_depth)
    assert first.stage_results["output"].artifacts["stats"] == direct_stats
    assert harness.output.read_bytes() == expected.read_bytes()
    assert harness.prepared.canonical_plan_bytes == plan_bytes
    assert digest == harness.authority.identity.input_content_sha256
    assert first.identities["depth"].canonical_identity_bytes == identity_bytes
    assert first.identities["depth"].cache_key == harness.authority.identity.cache_key(DEPTH_CACHE_SCHEMA)
    assert harness.backend.calls == 1  # Reuse came only from the existing V3 DepthCache.
    assert harness.cache._entry_path(first.identities["depth"].cache_key).is_file()
    assert not (harness.executor.cache_dir / "dag_cache").exists()
    assert first.cache_hits == second.cache_hits == 0


@pytest.mark.parametrize("mutation", ["input", "runtime", "canonical-plan"])
def test_changed_authority_stops_before_first_stage_or_output(tmp_path, mutation):
    harness = _ParityHarness(tmp_path)
    if mutation == "input":
        Image.new("RGB", (31, 45), color="black").save(harness.image)
    elif mutation == "runtime":
        harness.backend.live = False
    else:
        harness.prepared = replace(harness.prepared, canonical_plan_bytes=b"{}")
    result = harness.run()
    assert not result.success
    assert not result.stage_results
    assert harness.backend.calls == 0
    assert not harness.output.exists()
    assert harness.cache.stats()["entry_count"] == 0


def test_runtime_revocation_during_fixture_inference_prevents_cache_and_output(tmp_path, monkeypatch):
    harness = _ParityHarness(tmp_path)
    original_infer = harness.backend.infer

    def revoke(image):
        result = original_infer(image)
        harness.backend.live = False
        return result

    monkeypatch.setattr(harness.backend, "infer", revoke)
    result = harness.run()
    assert not result.success
    assert not harness.output.exists()
    assert harness.cache.stats()["entry_count"] == 0


@pytest.mark.parametrize("failure", ["backend-error", "wrong-runtime-echo"])
def test_failed_fixture_inference_cannot_become_success_or_publish_depth(tmp_path, monkeypatch, failure):
    harness = _ParityHarness(tmp_path)

    def fail(image):
        if failure == "backend-error":
            raise RuntimeError("Fixture backend failed")
        return SimpleNamespace(depth=harness.backend.fixture_depth(image), metadata={"runtime_identity_sha256": "f" * 64})

    monkeypatch.setattr(harness.backend, "infer", fail)
    result = harness.run()
    assert not result.success
    assert result.stage_results["depth"].status is StageStatus.FAILED
    assert not harness.output.exists()
    assert harness.cache.stats()["entry_count"] == 0


def test_cancel_before_output_preserves_unpublished_state(tmp_path):
    harness = _ParityHarness(tmp_path)

    def checkpoint(stage, phase, context):
        harness._checkpoint(stage, phase, context)
        if stage == "output" and phase == "before_compute":
            raise RuntimeError("Fixture lease revoked")

    result = harness.run(checkpoint)
    assert not result.success
    assert result.error == "Fixture lease revoked"
    assert not harness.output.exists()
    assert "output" not in result.stage_results
