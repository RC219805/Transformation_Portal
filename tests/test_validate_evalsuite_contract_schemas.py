"""Tests for EvalSuite contract schema validator tooling."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jsonschema
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "validate_evalsuite_contract_schemas.py"
SPEC = importlib.util.spec_from_file_location("validate_evalsuite_contract_schemas", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_json_reports_parse_errors(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    with pytest.raises(validator.InputParseError, match="Failed to load JSON file"):
        validator.load_json(broken)


def test_parse_lockfile_rejects_malformed_lines(tmp_path: Path) -> None:
    lockfile = tmp_path / "SCHEMA_LOCKS.sha256"
    lockfile.write_text("abc  one  two\n", encoding="utf-8")

    with pytest.raises(validator.InputInvariantError, match="Invalid lockfile line"):
        validator.parse_lockfile(lockfile)


def test_discover_schemas_scans_only_schema_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    schema_root = repo_root / "docs" / "schemas" / "evalsuite"
    valid_schema = schema_root / "a.v0" / "a.schema.json"
    ignored_json = schema_root / "b.v0" / "b.json"
    _write_json(valid_schema, {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"})
    _write_json(ignored_json, {"note": "ignored"})

    monkeypatch.setattr(validator, "REPO_ROOT", repo_root)
    monkeypatch.setattr(validator, "SCHEMA_ROOT", schema_root)

    discovered = validator.discover_schemas()
    assert discovered == [valid_schema]


def test_validate_schemas_are_valid_rejects_invalid_schema(tmp_path: Path) -> None:
    schema_path = tmp_path / "invalid.schema.json"
    _write_json(
        schema_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"bad": 1},
        },
    )

    with pytest.raises(jsonschema.SchemaError):
        validator.validate_schemas_are_valid([schema_path])


def test_validate_examples_rejects_missing_fixture(tmp_path: Path) -> None:
    schema_path = tmp_path / "simple.schema.json"
    _write_json(schema_path, {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"})

    with pytest.raises(validator.InputInvariantError, match="Missing example fixture"):
        validator.validate_examples([(tmp_path / "missing.example.json", schema_path)])


def test_validate_examples_enforces_datetime_format(tmp_path: Path) -> None:
    schema_path = tmp_path / "resolution.schema.json"
    _write_json(
        schema_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["created_at_utc"],
            "properties": {"created_at_utc": {"type": "string", "format": "date-time"}},
        },
    )
    example_path = tmp_path / "bad.example.json"
    _write_json(example_path, {"created_at_utc": "not-a-datetime"})

    with pytest.raises(jsonschema.ValidationError):
        validator.validate_examples([(example_path, schema_path)])


def test_validate_lockfile_rejects_extra_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    schema_path = repo_root / "docs" / "schemas" / "evalsuite" / "taxonomy.v0" / "taxonomy.schema.json"
    _write_json(schema_path, {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"})

    rel = schema_path.relative_to(repo_root).as_posix()
    digest = validator.sha256_file(schema_path)
    lockfile = repo_root / "docs" / "contracts" / "SCHEMA_LOCKS.sha256"
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    lockfile.write_text(
        "\n".join(
            [
                "# generated",
                f"{digest}  {rel}",
                "0000000000000000000000000000000000000000000000000000000000000000  docs/schemas/evalsuite/extra.v0/extra.schema.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(validator, "REPO_ROOT", repo_root)

    with pytest.raises(validator.InputInvariantError, match="Unexpected extra entries"):
        validator.validate_lockfile([schema_path], lockfile)


def test_validate_lockfile_rejects_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    schema_path = repo_root / "docs" / "schemas" / "evalsuite" / "taxonomy.v0" / "taxonomy.schema.json"
    _write_json(schema_path, {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"})

    rel = schema_path.relative_to(repo_root).as_posix()
    lockfile = repo_root / "docs" / "contracts" / "SCHEMA_LOCKS.sha256"
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    lockfile.write_text(f"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff  {rel}\n", encoding="utf-8")

    monkeypatch.setattr(validator, "REPO_ROOT", repo_root)

    with pytest.raises(validator.InputInvariantError, match="Schema drift detected"):
        validator.validate_lockfile([schema_path], lockfile)


def test_main_returns_expected_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validator, "discover_schemas", lambda: [])
    monkeypatch.setattr(validator, "validate_schemas_are_valid", lambda _schemas: None)
    monkeypatch.setattr(validator, "validate_examples", lambda _examples: None)
    monkeypatch.setattr(validator, "validate_lockfile", lambda _schemas, _lockfile: None)
    assert validator.main() == validator.EXIT_SUCCESS

    def _raise_parse() -> list[Path]:
        raise validator.InputParseError("parse error")

    monkeypatch.setattr(validator, "discover_schemas", _raise_parse)
    assert validator.main() == validator.EXIT_INPUT_PARSE_ERROR

    def _raise_invariant() -> list[Path]:
        raise validator.InputInvariantError("invariant error")

    monkeypatch.setattr(validator, "discover_schemas", _raise_invariant)
    assert validator.main() == validator.EXIT_INPUT_INVARIANT_FAILURE

    monkeypatch.setattr(validator, "discover_schemas", lambda: [])

    def _raise_validation(_examples: list[tuple[Path, Path]]) -> None:
        raise jsonschema.ValidationError("validation error")

    monkeypatch.setattr(validator, "validate_examples", _raise_validation)
    assert validator.main() == validator.EXIT_SCHEMA_VALIDATION_FAILURE
