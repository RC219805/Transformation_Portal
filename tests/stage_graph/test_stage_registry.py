"""Static allowlist contract for canonical execution-plan stages."""

from __future__ import annotations

from dataclasses import fields

import pytest

from transformation_portal.core.execution_plan import load_execution_plan_schema
from transformation_portal.stage_graph.registry import (
    OUTPUT_DEFINITIONS,
    STAGE_DEFINITIONS,
    StageDefinition,
    StageRegistryIdentifier,
    UnknownStageRegistryIdentifier,
    get_output_definition,
    get_stage_definition,
    stage_registry_identifiers,
)

pytestmark = pytest.mark.unit


def test_registry_is_exactly_typed_and_schema_allowlisted() -> None:
    schema = load_execution_plan_schema()
    schema_identifiers = tuple(schema["$defs"]["stageRegistryId"]["enum"])

    assert stage_registry_identifiers() == schema_identifiers
    assert tuple(STAGE_DEFINITIONS) == tuple(StageRegistryIdentifier)
    assert all(definition.identifier is identifier for identifier, definition in STAGE_DEFINITIONS.items())


def test_registry_definitions_are_semantic_metadata_not_factories() -> None:
    field_names = {field.name for field in fields(StageDefinition)}
    assert field_names == {
        "identifier",
        "configuration_schema",
        "allowed_output_kinds",
        "resources",
    }
    forbidden_tokens = {"class", "callable", "factory", "module", "command", "argv", "executable"}
    assert field_names.isdisjoint(forbidden_tokens)
    for definition in STAGE_DEFINITIONS.values():
        assert not any(callable(value) for value in vars(definition).values())


def test_registry_configuration_and_output_kinds_match_closed_schema() -> None:
    schema = load_execution_plan_schema()
    config_refs = schema["$defs"]["stageConfiguration"]["oneOf"]
    configuration_schemas = {
        schema["$defs"][reference["$ref"].removeprefix("#/$defs/")]["properties"]["schema"]["const"]
        for reference in config_refs
    }
    output_kinds = set(schema["$defs"]["allArtifactKind"]["enum"])
    output_scopes = set(schema["$defs"]["outputDeclaration"]["properties"]["scope"]["enum"])
    output_cardinalities = set(schema["$defs"]["outputDeclaration"]["properties"]["cardinality"]["enum"])

    assert set(OUTPUT_DEFINITIONS) == output_kinds
    assert {definition.scope.value for definition in OUTPUT_DEFINITIONS.values()} <= output_scopes
    assert {definition.cardinality.value for definition in OUTPUT_DEFINITIONS.values()} <= output_cardinalities

    for definition in STAGE_DEFINITIONS.values():
        assert definition.configuration_schema in configuration_schemas
        assert set(definition.allowed_output_kinds) <= output_kinds

    batch_manifest = get_output_definition("batch_manifest_json")
    assert batch_manifest.scope.value == "per_run"
    assert batch_manifest.cardinality.value == "one"
    run_card = get_output_definition("run_card")
    assert run_card.scope.value == "per_run"
    assert run_card.cardinality.value == "one"
    reconstruction = get_output_definition("reconstruction_bundle")
    assert reconstruction.scope.value == "per_run"
    assert reconstruction.cardinality.value == "many"
    materials_masks = get_output_definition("materials_v3_masks")
    assert materials_masks.scope.value == "per_input"
    assert materials_masks.cardinality.value == "one"
    assert "bit_depth_16_intermediates" not in OUTPUT_DEFINITIONS


def test_registry_has_no_alias_or_dynamic_registration_path() -> None:
    with pytest.raises(UnknownStageRegistryIdentifier, match="Unknown"):
        get_stage_definition("depth")
    with pytest.raises(UnknownStageRegistryIdentifier, match="Unknown"):
        get_stage_definition("some.module:Stage")
    with pytest.raises(TypeError):
        STAGE_DEFINITIONS[StageRegistryIdentifier.LUX_DEPTH] = object()  # type: ignore[index]


def test_lazy_stage_graph_root_preserves_existing_public_exports() -> None:
    import transformation_portal.stage_graph as stage_graph
    from transformation_portal.stage_graph.graph import GraphBuilder

    assert stage_graph.GraphBuilder is GraphBuilder
    assert stage_graph.StageRegistryIdentifier is StageRegistryIdentifier
    assert "GraphBuilder" in dir(stage_graph)
