"""Native Materials plan authority is closed, ordered and version separated."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from PIL import Image

from tests.core.test_execution_plan import _node
from transformation_portal.core.execution_plan import (
    CanonicalExecutionPlan,
    ExecutionPlanError,
    load_execution_plan_schema,
    parse_execution_plan_json,
    with_execution_plan_fingerprint,
)
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.config_resolver import build_run_card_config_fingerprint
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
from transformation_portal.lux_depth_v3.geometry import NORMALIZED_GEOMETRY_RECIPE
from transformation_portal.lux_depth_v3.native_materials import (
    NATIVE_MATERIALS_LIMITS,
    native_materials_binding_hash,
)
from transformation_portal.materials_v4.engine import ResponsePolicy
from transformation_portal.materials_v4.operations import operation_contract_hash
from transformation_portal.stage_graph.registry import StageRegistryIdentifier

pytestmark = pytest.mark.unit


def _binding(files):
    binding = {
        "schema": "tp.lux.native_materials.v1",
        "recipe": "native_sdr_v1",
        "input_color": "srgb",
        "evidence_root": "/private/metadata-only-fixture",
        "manifest": {"path": "materials.json", "sha256": "b" * 64, "size_bytes": 64},
        "policy": ResponsePolicy().to_payload(),
        "operations_sha256": operation_contract_hash(),
        "limits": dict(NATIVE_MATERIALS_LIMITS),
        "sources": [
            {
                "path": item["path"],
                "sha256": hashlib.sha256(item["path"].encode()).hexdigest(),
                "master_sha256": "c" * 64,
                "size_bytes": 64,
                "shape": [31, 45],
                "source_bit_depth": 16,
                "alpha": False,
                "mask_bytes": 0,
                "evidence": None,
            }
            for item in files
        ],
    }
    return _rehash(binding)


def _rehash(binding):
    binding["binding_sha256"] = native_materials_binding_hash(binding)
    return binding


def _v2(payload):
    return next(node["configuration"] for node in payload["nodes"] if node["id"] == "lux.v2")


def _parse(payload):
    return CanonicalExecutionPlan.from_payload(with_execution_plan_fingerprint(payload))


@pytest.fixture
def native_plan_payload(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    files = [inputs / "a.jpg", inputs / "b.jpg"]
    for path in files:
        Image.new("RGB", (45, 31), (70, 90, 110)).save(path)
    prepared = prepare_lux_execution(
        EnhanceConfig(model_key="da3-metric", geometry_recipe=NORMALIZED_GEOMETRY_RECIPE, v2_input_color="srgb"), inputs, files
    )
    payload = prepared.plan.to_payload()
    v2 = _v2(payload)
    v2["native_materials"] = _binding(payload["input_selection"]["files"])
    v2["processing_version"] = "1.4.0"
    return payload


def test_native_plan_static_validation_never_discovers_or_decodes_files(native_plan_payload, monkeypatch):
    from transformation_portal.lux_depth_v3 import native_materials

    def forbidden(*_args, **_kwargs):
        pytest.fail("Static canonical validation attempted image/evidence I/O")

    for name in ("snapshot", "decode_master", "load_bound_evidence", "revalidate_native_materials"):
        monkeypatch.setattr(native_materials, name, forbidden)
    plan = _parse(native_plan_payload)
    recovered = parse_execution_plan_json(plan.to_canonical_json())
    assert recovered.to_payload() == plan.to_payload()
    assert _v2(recovered.to_payload())["processing_version"] == "1.4.0"


@pytest.mark.parametrize("field", ["schema", "recipe", "policy", "limits", "operations_sha256", "binding_sha256"])
def test_native_plan_rejects_changed_static_binding_even_after_plan_refingerprinting(native_plan_payload, field):
    binding = _v2(native_plan_payload)["native_materials"]
    if field == "policy":
        binding[field]["allow_supplied_confidence"] = not binding[field]["allow_supplied_confidence"]
    elif field == "limits":
        binding[field]["max_working_bytes"] *= 2
    else:
        binding[field] = "untrusted"
    with pytest.raises(ExecutionPlanError):
        _parse(native_plan_payload)


@pytest.mark.parametrize("where", ["binding", "source", "policy", "manifest", "limits"])
def test_native_plan_schema_rejects_unknown_fields(native_plan_payload, where):
    binding = _v2(native_plan_payload)["native_materials"]
    target = binding if where == "binding" else binding["sources"][0] if where == "source" else binding[where]
    target["invented_authority"] = True
    _rehash(binding)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(native_plan_payload, load_execution_plan_schema())
    with pytest.raises(ExecutionPlanError):
        _parse(native_plan_payload)


@pytest.mark.parametrize("change", ["reorder", "missing", "extra", "path"])
def test_native_sources_exactly_match_ordered_plan_selection(native_plan_payload, change):
    binding = _v2(native_plan_payload)["native_materials"]
    if change == "reorder":
        binding["sources"].reverse()
    elif change == "missing":
        binding["sources"].pop()
    elif change == "extra":
        binding["sources"].append({**binding["sources"][0], "path": "unselected.jpg"})
    else:
        binding["sources"][0]["path"] = "unselected.jpg"
    _rehash(binding)
    with pytest.raises(ExecutionPlanError, match="ordered input selection"):
        _parse(native_plan_payload)


@pytest.mark.parametrize(
    "field,value",
    [("input_color", "linear_srgb"), ("materials_color_handoff", "validated_srgb"), ("materials_mask_handoff", "required")],
)
def test_native_v2_has_one_explicit_color_and_mask_authority(native_plan_payload, field, value):
    _v2(native_plan_payload)[field] = value
    with pytest.raises(ExecutionPlanError):
        _parse(native_plan_payload)


@pytest.mark.parametrize("version", ["1.2.0", "1.3.0", None])
def test_native_binding_requires_its_exact_processing_version(native_plan_payload, version):
    v2 = _v2(native_plan_payload)
    if version is None:
        v2.pop("processing_version")
    else:
        v2["processing_version"] = version
    with pytest.raises(ExecutionPlanError):
        _parse(native_plan_payload)


def test_processing_version_cannot_authorize_native_operations_without_binding(native_plan_payload):
    _v2(native_plan_payload).pop("native_materials")
    with pytest.raises(ExecutionPlanError):
        _parse(native_plan_payload)


def test_native_binding_cannot_use_legacy_geometry(native_plan_payload):
    for node in native_plan_payload["nodes"]:
        node["configuration"].pop("geometry_recipe", None)
        node["configuration"].pop("model_max_long_edge", None)
    with pytest.raises(ExecutionPlanError):
        _parse(native_plan_payload)


def test_native_binding_cannot_promote_structural_legacy_configuration(native_plan_payload):
    native_plan_payload["configuration_completeness"] = "structural_legacy"
    for node in native_plan_payload["nodes"]:
        node["configuration"]["configuration_completeness"] = "structural_legacy"
    with pytest.raises(ExecutionPlanError):
        _parse(native_plan_payload)


@pytest.mark.parametrize(
    "stage",
    [StageRegistryIdentifier.LUX_MATERIALS_V3, StageRegistryIdentifier.LUX_PBR, StageRegistryIdentifier.LUX_RECONSTRUCTION],
)
def test_native_plan_rejects_unmapped_or_competing_stages(native_plan_payload, stage):
    configuration = {"configuration_completeness": "execution_complete"}
    if stage is StageRegistryIdentifier.LUX_MATERIALS_V3:
        # A fully valid legacy Materials node is unnecessary: the source of
        # authority is already conflicting before its optional settings matter.
        configuration.update(
            schema="tp.stage.config.lux.materials_v3.v2",
            geometry_recipe=NORMALIZED_GEOMETRY_RECIPE,
            processing_version="1.3.0",
        )
    elif stage is StageRegistryIdentifier.LUX_PBR:
        configuration["schema"] = "tp.stage.config.lux.pbr.v1"
    else:
        configuration["schema"] = "tp.stage.config.lux.reconstruction.v1"
    # Use structural configuration temporarily so the schema admits a minimal
    # representation; semantic completeness must still reject the whole plan.
    configuration["configuration_completeness"] = "structural_legacy"
    native_plan_payload["nodes"].insert(
        -1, _node("lux.conflicting", stage, configuration=configuration, outputs=[], optional=True)
    )
    with pytest.raises(ExecutionPlanError):
        _parse(native_plan_payload)


def test_native_plan_rejects_raw_even_with_consistent_refingerprinted_binding(native_plan_payload):
    native_plan_payload["input_selection"]["files"][0]["path"] = "a.dng"
    binding = _v2(native_plan_payload)["native_materials"]
    binding["sources"][0]["path"] = "a.dng"
    _rehash(binding)
    with pytest.raises(ExecutionPlanError, match="supported SDR raster"):
        _parse(native_plan_payload)


@pytest.mark.parametrize("schema_name", ["run_card.v1.schema.json", "run_card.v2.schema.json"])
def test_run_card_native_identity_and_processing_version_are_mutually_required(schema_name):
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "src/transformation_portal/schemas/run_card" / schema_name).read_bytes())["properties"][
        "config_fingerprint"
    ]
    backend = SimpleNamespace(requested_backend="da3", resolved_backend="da3", device="cpu")
    fingerprint = build_run_card_config_fingerprint(
        EnhanceConfig(geometry_recipe=NORMALIZED_GEOMETRY_RECIPE), backend_metadata=backend
    )
    fingerprint.update(native_materials_sha256="c" * 64, v2_processing_version="1.4.0")
    jsonschema.validate(fingerprint, schema)
    for field in ("geometry_recipe", "native_materials_sha256", "v2_processing_version"):
        changed = copy.deepcopy(fingerprint)
        changed.pop(field)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(changed, schema)
    fingerprint["v2_processing_version"] = "1.3.0"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(fingerprint, schema)
