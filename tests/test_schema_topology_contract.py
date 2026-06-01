"""Contract tests for repo schema/profile placement and naming."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from transformation_portal.ingest.evidence import (
    DEFAULT_PROJECTION_PROFILE,
    DEFAULT_PROJECTION_PROFILE_PATH,
    MACHINE_SCHEMA_VERSION,
)

pytestmark = [
    pytest.mark.unit,
]

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"

PHASE4_SCHEMA_CONTRACT_FIELDS = {
    "metadata.schema.json": ("metadata_contract_version", "tp.meta.capture.v1"),
    "metadata_manifest.schema.json": ("metadata_manifest_contract_version", "tp.meta.capture_manifest.v1"),
    "provenance_manifest.schema.json": ("provenance_contract_version", "tp.meta.provenance.v1"),
    "provenance_merkle.schema.json": ("provenance_merkle_contract_version", "tp.meta.provenance_merkle.v1"),
    "verification_report.schema.json": ("verification_contract_version", "tp.meta.verification_report.v1"),
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path} must contain a JSON object"
    return payload


def test_phase4_schemas_remain_authoritative_root_contracts() -> None:
    phase4_root = REPO_ROOT / "schemas" / "phase4"
    schema_paths = sorted(phase4_root.glob("*.schema.json"))

    assert [path.name for path in schema_paths] == sorted(PHASE4_SCHEMA_CONTRACT_FIELDS)

    for path in schema_paths:
        payload = _load_json(path)
        jsonschema.Draft202012Validator.check_schema(payload)

        assert payload["$schema"] == DRAFT_2020_12
        assert payload["$id"].endswith(path.relative_to(REPO_ROOT).as_posix())

        version_field, contract_version = PHASE4_SCHEMA_CONTRACT_FIELDS[path.name]
        assert payload["properties"][version_field]["const"] == contract_version


def test_docs_schema_tree_contains_published_json_schema_contracts() -> None:
    docs_schema_paths = sorted((REPO_ROOT / "docs" / "schemas").rglob("*.schema.json"))

    assert docs_schema_paths
    for path in docs_schema_paths:
        payload = _load_json(path)
        jsonschema.Draft202012Validator.check_schema(payload)

        assert payload["$schema"] == DRAFT_2020_12
        assert "$id" in payload
        assert "title" in payload


def test_runtime_projection_profiles_are_not_json_schema_contracts() -> None:
    profile_root = REPO_ROOT / "schemas" / "profiles"
    profile_paths = sorted(profile_root.glob("*.json"))

    assert [path.name for path in profile_paths] == [f"{DEFAULT_PROJECTION_PROFILE}.json"]
    assert DEFAULT_PROJECTION_PROFILE_PATH == profile_paths[0]

    for path in profile_paths:
        assert not path.name.endswith(".schema.json")
        payload = _load_json(path)

        assert "$schema" not in payload
        assert payload["schema"] == path.stem == DEFAULT_PROJECTION_PROFILE
        assert payload["source_schema"] == MACHINE_SCHEMA_VERSION
        assert all(isinstance(pointer, str) and pointer.startswith("/") for pointer in payload["drop_paths"])
