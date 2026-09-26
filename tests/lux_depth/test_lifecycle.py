"""Unified dispatch preserves native execution and independent replay authority.

Only neural workers and runtime probes are controlled fixtures. Pixel decoding,
processing, product serialization, native evidence, and verification are real.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.lux_depth_v4 import test_pipeline as v4_pipeline
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth import lifecycle as unified
from transformation_portal.lux_depth_v4 import evidence as shared_evidence
from transformation_portal.lux_depth_v4.lifecycle import LuxDepthV4Request
from transformation_portal.lux_depth_v5 import evidence as v5_evidence
from transformation_portal.lux_depth_v6.depth_maps import DepthMapRecipe
from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request
from transformation_portal.lux_depth_v6.plan import LuxDepthV6Request
from transformation_portal.lux_depth_v6.source import SourceLimits
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher

pytestmark = pytest.mark.unit
v4_prepared = v4_pipeline.prepared
_ROUTES = ("v4", "inference", "photography", "finishing", "depth-pro")
_SOURCE_ROUTES = ("finishing", "finishing-no-depth", "depth-pro")
_LARGE_SOURCE_LIMITS = SourceLimits(max_input_bytes=1024**4, max_pixels=200_000_000, memory_mib=262144)


def _prepare_route(request, route, *, source_limits=None):
    if route == "v4":
        original = request.getfixturevalue("v4_prepared")
        monkeypatch = request.getfixturevalue("monkeypatch")
        initialize = v4_pipeline.SessionFixture.__init__

        def initialize_bound_runtime(self, python, plan, *, cancellation):
            initialize(self, python, plan, cancellation=cancellation)
            payload = plan.to_payload()
            backend = {
                "model_canonical_key": payload["model"]["canonical_key"],
                "model_repo_id": payload["model"]["repo_id"],
                "model_lock_revision": payload["model"]["revision"],
                "actual_device": payload["device"],
            }
            self.runtime.to_mapping = lambda: {"backend_identity": copy.deepcopy(backend)}

        monkeypatch.setattr(v4_pipeline.SessionFixture, "__init__", initialize_bound_runtime)
        monkeypatch.setattr(
            shared_evidence.DA3RuntimeIdentityEvidence,
            "from_mapping",
            lambda value: SimpleNamespace(cacheable=True, to_mapping=lambda: copy.deepcopy(value)),
        )
        value = LuxDepthV4Request(
            original.input_root,
            original.output_root,
            input_color="srgb",
            strength=0,
            target_size=56,
            cache_dir=original.cache_root,
        )
        return unified.prepare(value), None
    if route == "depth-pro":
        value = request.getfixturevalue("native_case")
        if source_limits is not None:
            value = replace(value, source_limits=source_limits)
        return unified.prepare(value), value.input_dir
    value = request.getfixturevalue("request_case")
    if route == "photography":
        return unified.prepare(ManagedLuxDepthV6Request(value)), None
    if route in {"finishing", "finishing-no-depth"}:
        source = unified.run(unified.prepare(value))
        finishing = LuxDepthV6Request(
            source.output_root,
            source.output_root.with_name("finished"),
            source_limits=source_limits or SourceLimits(),
            depth_maps=DepthMapRecipe() if route == "finishing" else None,
        )
        return unified.prepare(finishing), source.output_root
    return unified.prepare(value), None


def test_operator_import_is_lazy_and_has_no_numeric_or_neural_runtime():
    script = """
import sys
import transformation_portal.lux_depth as lux
from transformation_portal.lux_depth import prepare, run, verify, result_summary
from transformation_portal.lux_depth.__main__ import main
assert callable(prepare) and callable(run) and callable(verify) and callable(result_summary)
assert 'PhotographyRequest' in dir(lux)
try:
    main(['verify', '--help'])
except SystemExit as exc:
    assert exc.code == 0
assert not {'numpy', 'scipy', 'PIL', 'torch', 'transformers', 'depth_pro'} & set(sys.modules)
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    completed = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("route", _ROUTES)
def test_native_roundtrip_keeps_exact_plan_bytes_and_evidence(request, route):
    prepared, source_root = _prepare_route(request, route)
    plan_bytes = prepared.canonical_plan_bytes
    assert not prepared.output_root.exists()
    result = unified.run(prepared)
    expected = hashlib.sha256(plan_bytes).hexdigest()
    verified = unified.verify(result.output_root, source_root=source_root, expected_plan_sha256=expected)
    assert verified.plan_sha256 == expected
    assert verified.output_root == result.output_root
    assert prepared.canonical_plan_bytes == plan_bytes
    evidence_path = result.output_root / ("execution-evidence.json" if source_root is None else "evidence.json")
    assert verified.canonical_bytes == evidence_path.read_bytes()
    summary = unified.result_summary(result)
    assert summary["plan_sha256"] == expected
    assert summary["input_count"] == 1
    assert summary["production_acceptance"] == "not_established"


@pytest.mark.parametrize("route", _ROUTES)
def test_cancellation_prevents_output_for_every_engine(request, route):
    prepared, _ = _prepare_route(request, route)
    with pytest.raises(RuntimeError, match="cancelled"):
        unified.run(prepared, cancellation=lambda: True)
    assert not prepared.output_root.exists()


@pytest.mark.parametrize("kind", ["request", "prepared"])
@pytest.mark.parametrize("forgery", ["subclass", "name_spoof"])
def test_exact_carrier_identity_prevents_forged_dispatch(request_case, kind, forgery):
    original = request_case if kind == "request" else unified.prepare(request_case)
    original_type = type(original)
    if forgery == "subclass":
        impostor_type = type(original_type.__name__, (original_type,), {"__module__": original_type.__module__})
        impostor = impostor_type(**vars(original))
    else:
        impostor = type(original_type.__name__, (), {"__module__": original_type.__module__})()
    with pytest.raises(TypeError, match="exact supported"):
        (unified.prepare if kind == "request" else unified.run)(impostor)
    assert not request_case.output_dir.exists()


def test_aliases_are_exact_native_types():
    from transformation_portal.lux_depth import DepthProRequest, FinishingRequest, InferenceRequest, PhotographyRequest
    from transformation_portal.lux_depth_v5.lifecycle import LuxDepthV5Request
    from transformation_portal.lux_depth_v6.depth_pro import NativeDepthProRequest

    assert InferenceRequest is LuxDepthV5Request
    assert PhotographyRequest is ManagedLuxDepthV6Request
    assert FinishingRequest is LuxDepthV6Request
    assert DepthProRequest is NativeDepthProRequest


def test_wrong_byte_digest_and_unsigned_fingerprint_are_not_interchangeable(request_case):
    prepared = unified.prepare(request_case)
    result = unified.run(prepared)
    byte_digest = hashlib.sha256(prepared.canonical_plan_bytes).hexdigest()
    assert byte_digest != prepared.plan.plan_fingerprint_sha256
    for invalid in ("f" * 64, prepared.plan.plan_fingerprint_sha256):
        with pytest.raises(ValueError, match="expected exact bytes"):
            unified.verify(result.output_root, expected_plan_sha256=invalid)
    assert unified.verify(result.output_root, expected_plan_sha256=byte_digest).plan_sha256 == byte_digest


@pytest.mark.parametrize("route", ["photography", "inference"])
def test_self_contained_evidence_rejects_external_source_override(request, route, tmp_path):
    prepared, _ = _prepare_route(request, route)
    result = unified.run(prepared)
    with pytest.raises(ValueError, match="omit source_root"):
        unified.verify(result.output_root, source_root=tmp_path)


@pytest.mark.parametrize("route", ["finishing", "depth-pro"])
def test_source_dependent_evidence_requires_source_root(request, route):
    prepared, _ = _prepare_route(request, route)
    result = unified.run(prepared)
    with pytest.raises(ValueError, match="require source_root"):
        unified.verify(result.output_root)


def _forbid_source_payload_replay(monkeypatch):
    from transformation_portal.lux_depth_v6 import depth_pro, evidence, source

    def forbidden(*args, **kwargs):
        pytest.fail("Source limits must reject before reading images, loading arrays, or replaying products")

    monkeypatch.setattr(source, "_inspect_image", forbidden)
    monkeypatch.setattr(source, "verify_execution_evidence_v3", forbidden)
    monkeypatch.setattr(evidence, "verify_artifacts", forbidden)
    monkeypatch.setattr(depth_pro, "_verify_artifacts", forbidden)


@pytest.mark.parametrize("route", _SOURCE_ROUTES)
@pytest.mark.parametrize("field", ["max_input_bytes", "max_pixels", "memory_mib"])
def test_caller_source_limits_reject_real_outputs_before_payload_reads(request, route, field, monkeypatch):
    prepared, source_root = _prepare_route(request, route, source_limits=_LARGE_SOURCE_LIMITS)
    result = unified.run(prepared)
    _forbid_source_payload_replay(monkeypatch)
    with pytest.raises(ValueError, match="bounded regular|byte budget|pixel|memory admission"):
        unified.verify(result.output_root, source_root=source_root, source_limits=SourceLimits(**{field: 1}))
    assert prepared.canonical_plan_bytes == (result.output_root / "plan.json").read_bytes()


@pytest.mark.parametrize("route", _SOURCE_ROUTES)
def test_recorded_limits_cannot_raise_default_verification_memory(request, route, monkeypatch):
    prepared, source_root = _prepare_route(request, route, source_limits=_LARGE_SOURCE_LIMITS)
    result = unified.run(prepared)
    plan = prepared.plan.to_payload()
    image = plan["inputs"][0] if route == "depth-pro" else plan["source"]["images"][0]
    # Valid bounded geometry under the large recorded limits, but too large for
    # default caller memory. Rehashing completion cannot authorize allocation.
    image["shape"] = [8192, 8192]
    raw = canonicalize_json(plan)
    (result.output_root / "plan.json").write_bytes(raw)
    completion = result.output_root / "evidence.json"
    evidence = json.loads(completion.read_bytes())
    evidence["plan_sha256"] = hashlib.sha256(raw).hexdigest()
    completion.write_bytes(canonicalize_json(evidence))
    _forbid_source_payload_replay(monkeypatch)
    with pytest.raises(ValueError, match="memory admission"):
        unified.verify(result.output_root, source_root=source_root)


@pytest.mark.parametrize("route", _SOURCE_ROUTES)
def test_explicit_source_limits_preserve_real_replay_and_exact_plan(request, route):
    prepared, source_root = _prepare_route(request, route)
    result = unified.run(prepared)
    verified = unified.verify(result.output_root, source_root=source_root, source_limits=_LARGE_SOURCE_LIMITS)
    assert verified.plan_sha256 == hashlib.sha256(prepared.canonical_plan_bytes).hexdigest()
    assert verified.canonical_bytes == (result.output_root / "evidence.json").read_bytes()


@pytest.mark.parametrize(
    "schema", ["tp.execution.plan.v2", "tp.execution.plan.v3", "tp.execution.plan.v4", "tp.execution.plan.v5"]
)
def test_self_contained_schemas_reject_unsupported_limits_before_native_replay(tmp_path, schema):
    (tmp_path / "execution-plan.json").write_bytes(canonicalize_json({"schema": schema}))
    with pytest.raises(ValueError, match="does not support source_limits"):
        unified.verify(tmp_path, source_limits=SourceLimits())


@pytest.mark.parametrize("limits", [False, {"max_pixels": 1}, SimpleNamespace(**SourceLimits().to_payload())])
def test_untyped_source_limits_fail_before_opening_output(tmp_path, limits):
    with pytest.raises(TypeError, match="exact SourceLimits"):
        unified.verify(tmp_path / "nonexistent", source_limits=limits)


@pytest.mark.parametrize("mutation", ["changed", "ambiguous"])
def test_final_plan_rebind_rejects_mutation_after_native_verifier(request_case, monkeypatch, mutation):
    result = unified.run(unified.prepare(request_case))
    native = v5_evidence.verify_execution_evidence_v3

    def mutate_after_verified(*args, **kwargs):
        verified = native(*args, **kwargs)
        plan = result.output_root / "execution-plan.json"
        if mutation == "changed":
            plan.write_bytes(plan.read_bytes() + b" ")
        else:
            (result.output_root / "plan.json").write_bytes(plan.read_bytes())
        return verified

    monkeypatch.setattr(v5_evidence, "verify_execution_evidence_v3", mutate_after_verified)
    with pytest.raises(ValueError, match="changed during verification|exactly one native plan"):
        unified.verify(result.output_root)


@pytest.mark.parametrize("kind", ["ambiguous", "symlink", "missing", "oversized"])
def test_plan_selection_rejects_unsafe_or_unbounded_paths(tmp_path, kind):
    output = tmp_path / "output"
    output.mkdir()
    plan = output / "plan.json"
    if kind == "ambiguous":
        plan.write_bytes(b"{}")
        (output / "execution-plan.json").write_bytes(b"{}")
    elif kind == "symlink":
        external = tmp_path / "external.json"
        external.write_bytes(b"{}")
        plan.symlink_to(external)
    elif kind == "oversized":
        with plan.open("wb") as stream:
            stream.truncate(16 * 1024**2 + 1)
    with pytest.raises((ValueError, RuntimeError)):
        unified.verify(output)


@pytest.mark.parametrize("payload", [b'{"schema":"unknown"}', b'{"schema":null}', b'{"schema":3}', b"[]"])
def test_unknown_or_malformed_schema_cannot_choose_an_engine(tmp_path, payload):
    (tmp_path / "plan.json").write_bytes(payload)
    with pytest.raises((ValueError, TypeError)):
        unified.verify(tmp_path)


def test_cancelled_verification_stops_before_opening_output(tmp_path):
    with pytest.raises(RuntimeError, match="cancelled"):
        unified.verify(tmp_path / "nonexistent", cancellation=lambda: True)


def test_cancellation_after_native_replay_cannot_report_success(request_case, monkeypatch):
    result = unified.run(unified.prepare(request_case))
    native = v5_evidence.verify_execution_evidence_v3
    cancelled = False

    def cancel_after_verified(*args, **kwargs):
        nonlocal cancelled
        verified = native(*args, **kwargs)
        cancelled = True
        return verified

    monkeypatch.setattr(v5_evidence, "verify_execution_evidence_v3", cancel_after_verified)
    with pytest.raises(RuntimeError, match="cancelled"):
        unified.verify(result.output_root, cancellation=lambda: cancelled)


@pytest.mark.parametrize("route", ["v4", "finishing", "depth-pro"])
@pytest.mark.parametrize("option", ["publication_limits", "managed_process_group"])
def test_standalone_routes_reject_managed_execution_options(request, route, option):
    prepared, _ = _prepare_route(request, route)
    publisher = GenerationPublisher(artifact_store=None, record_store=None)
    value = publisher.limits if option == "publication_limits" else True
    with pytest.raises(ValueError, match="managed execution options"):
        unified.run(prepared, **{option: value})
    assert not prepared.output_root.exists()


@pytest.mark.parametrize("route", ["v4", "finishing", "depth-pro"])
def test_standalone_requests_reject_publisher_before_admission(request, route, tmp_path):
    if route == "depth-pro":
        value = request.getfixturevalue("native_case")
    elif route == "v4":
        value = LuxDepthV4Request(tmp_path / "missing", tmp_path / "output")
    else:
        value = LuxDepthV6Request(tmp_path / "missing", tmp_path / "output")
    publisher = GenerationPublisher(artifact_store=None, record_store=None)
    with pytest.raises(ValueError, match="managed publication"):
        unified.prepare(value, publisher=publisher)
    assert not value.output_dir.exists()


def test_mutated_input_cannot_bypass_native_prepared_authority(request_case):
    prepared = unified.prepare(request_case)
    (request_case.input_dir / "ramp.tif").write_bytes(b"changed after admission")
    with pytest.raises((ValueError, RuntimeError), match="changed"):
        unified.run(prepared)
    assert not prepared.output_root.exists()


def test_rehashed_master_cannot_bypass_semantic_verification(request_case):
    import json

    import numpy as np

    result = unified.run(unified.prepare(request_case))
    target = result.output_root / "input-0000/master.npy"
    pixels = np.load(target, allow_pickle=False)
    pixels.flat[0] += 0.1
    np.save(target, pixels, allow_pickle=False)
    completion = result.output_root / "execution-evidence.json"
    evidence = json.loads(completion.read_bytes())
    record = next(row for row in evidence["artifacts"] if row["path"] == "input-0000/master.npy")
    record.update(size_bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    completion.write_bytes(canonicalize_json(evidence))
    with pytest.raises(ValueError):
        unified.verify(result.output_root)


def test_managed_process_ownership_requires_an_exact_boolean(request_case):
    prepared = unified.prepare(request_case)
    with pytest.raises(TypeError, match="exact boolean"):
        unified.run(prepared, managed_process_group=1)
    assert not prepared.output_root.exists()


@pytest.mark.parametrize("field", ["non_commercial_ok", "accept_license"])
def test_depth_pro_research_acknowledgements_remain_required(native_case, field):
    with pytest.raises(ValueError, match="acknowledgements"):
        unified.prepare(replace(native_case, **{field: False}))
    assert not native_case.output_dir.exists()
