"""V5 stage identities validate their plan carrier once, and only from validated bytes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from transformation_portal.core import execution_identity_v5
from transformation_portal.core.execution_identity_v5 import materialize_stage_identity
from transformation_portal.core.execution_plan import ExecutionPlanError
from transformation_portal.core.execution_plan_v4 import ExecutionPlanV4
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v5 import LuxDepthV5Request, prepare

pytestmark = pytest.mark.unit


@pytest.fixture
def plan(tmp_path) -> ExecutionPlanV4:
    source = tmp_path / "inputs"
    source.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        Image.new("RGB", (42, 28), (45, 78, 91)).save(source / name)
    request = LuxDepthV5Request(source, tmp_path / "output", input_color="srgb", cache_dir=tmp_path / "cache")
    return prepare(request).plan


@pytest.fixture
def plan_validations(monkeypatch):
    """Record every full plan validation done on behalf of identity materialization."""

    def forget_validated_plans():
        cache = getattr(execution_identity_v5, "_validated_plan_payload", None)
        if cache is not None:
            cache.cache_clear()

    validated = []
    parse = execution_identity_v5.parse_plan_v4

    def recording_parse(data):
        validated.append(data)
        return parse(data)

    forget_validated_plans()
    monkeypatch.setattr(execution_identity_v5, "parse_plan_v4", recording_parse)
    yield validated
    forget_validated_plans()


def _identity(plan, node, input_id="input-0000"):
    return materialize_stage_identity(
        plan,
        SimpleNamespace(name=node["id"], version=node["stage"]),
        input_id,
        {name: "e" * 64 for name in node["inputs"]},
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )


def test_identities_for_one_plan_validate_its_carrier_once(plan, plan_validations):
    payload = plan.to_payload()

    identities = [_identity(plan, node, item["id"]) for item in payload["inputs"] for node in payload["nodes"]]

    assert len(identities) == 12
    assert plan_validations == [plan.canonical_bytes]
    by_stage = {identity.stage_name: identity.to_dict() for identity in identities[:4]}
    assert by_stage["output"]["plan_fingerprint_sha256"] == plan.plan_fingerprint_sha256
    assert by_stage["depth"]["model"] == payload["model"]


def test_memoized_plan_validation_still_rejects_forged_carriers(plan, plan_validations):
    node = plan.to_payload()["nodes"][1]
    original = _identity(plan, node)

    tampered = plan.to_payload()
    tampered["device"] = "mps" if tampered["device"] == "cpu" else "cpu"
    forged = object.__new__(ExecutionPlanV4)
    object.__setattr__(forged, "canonical_bytes", canonicalize_json(tampered))
    with pytest.raises(ExecutionPlanError, match="fingerprint mismatch"):
        _identity(forged, node)

    textual = object.__new__(ExecutionPlanV4)
    object.__setattr__(textual, "canonical_bytes", plan.canonical_bytes.decode("utf-8"))
    with pytest.raises(ExecutionPlanError, match="must be bytes"):
        _identity(textual, node)

    assert _identity(plan, node) == original
    assert len(plan_validations) == 2
