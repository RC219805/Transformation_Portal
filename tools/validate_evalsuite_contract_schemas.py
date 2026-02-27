#!/usr/bin/env python3
"""
EvalSuite contract schema validation + drift lock enforcement.

Validates:
  1) Schemas are valid Draft 2020-12 JSON Schemas
  2) Example fixtures validate against their schemas
  3) docs/contracts/SCHEMA_LOCKS.sha256 matches schema file digests
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jsonschema

EXIT_SUCCESS = 0
EXIT_INPUT_PARSE_ERROR = 2
EXIT_INPUT_INVARIANT_FAILURE = 3
EXIT_SCHEMA_VALIDATION_FAILURE = 4


class ContractValidationError(Exception):
    """Base class for predictable contract validation failures."""

    exit_code = EXIT_INPUT_INVARIANT_FAILURE


class InputParseError(ContractValidationError):
    """Raised when JSON/text inputs cannot be parsed."""

    exit_code = EXIT_INPUT_PARSE_ERROR


class InputInvariantError(ContractValidationError):
    """Raised when required files or lockfile invariants are violated."""

    exit_code = EXIT_INPUT_INVARIANT_FAILURE


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = REPO_ROOT / "docs" / "schemas" / "evalsuite"
FORMAT_CHECKER = jsonschema.FormatChecker()

LOCKFILE = REPO_ROOT / "docs" / "contracts" / "SCHEMA_LOCKS.sha256"

EXAMPLES: List[Tuple[Path, Path]] = [
    (
        REPO_ROOT / "docs" / "contracts" / "examples" / "taxonomy_v0.example.json",
        REPO_ROOT / "docs" / "schemas" / "evalsuite" / "taxonomy.v0" / "taxonomy.schema.json",
    ),
    (
        REPO_ROOT / "docs" / "contracts" / "examples" / "evalsuite_v0.example.json",
        REPO_ROOT / "docs" / "schemas" / "evalsuite" / "evalsuite.v0" / "evalsuite.schema.json",
    ),
    (
        REPO_ROOT / "docs" / "contracts" / "examples" / "evalsuite_resolution_v0.example.json",
        REPO_ROOT / "docs" / "schemas" / "evalsuite" / "evalsuite.resolution.v0" / "resolution.schema.json",
    ),
]


@FORMAT_CHECKER.checks("date-time")
def is_rfc3339_datetime(instance: object) -> bool:
    """Strict enough RFC3339 gate for contract fields like created_at_utc."""
    if not isinstance(instance, str):
        return True
    if "T" not in instance:
        return False
    normalized = instance.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputParseError(f"Failed to load JSON file {path}: {exc}") from exc


def parse_lockfile(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise InputInvariantError(f"LOCKFILE missing: {path}")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise InputParseError(f"Failed to read lockfile {path}: {exc}") from exc

    locks: Dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise InputInvariantError(f"Invalid lockfile line: {line!r}")
        digest, rel = parts
        locks[rel] = digest
    return locks


def discover_schemas() -> List[Path]:
    if not SCHEMA_ROOT.exists():
        raise InputInvariantError(f"Schema root not found: {SCHEMA_ROOT}")

    schemas = sorted(
        [p for p in SCHEMA_ROOT.rglob("*.schema.json") if p.is_file()],
        key=lambda p: p.relative_to(REPO_ROOT).as_posix(),
    )
    if not schemas:
        raise InputInvariantError(f"No schema files found under: {SCHEMA_ROOT}")
    return schemas


def validate_schemas_are_valid(schemas: List[Path]) -> None:
    for schema_path in schemas:
        if not schema_path.exists():
            raise InputInvariantError(f"Missing schema file: {schema_path}")
        schema_obj = load_json(schema_path)
        # Ensures each schema document itself is valid Draft 2020-12 JSON Schema.
        jsonschema.Draft202012Validator.check_schema(schema_obj)


def validate_examples(examples: List[Tuple[Path, Path]]) -> None:
    for example_path, schema_path in examples:
        if not example_path.exists():
            raise InputInvariantError(f"Missing example fixture: {example_path}")
        if not schema_path.exists():
            raise InputInvariantError(f"Missing schema for example: {schema_path}")

        schema_obj = load_json(schema_path)
        instance = load_json(example_path)

        jsonschema.Draft202012Validator(
            schema_obj,
            format_checker=FORMAT_CHECKER,
        ).validate(instance)


def validate_lockfile(schemas: List[Path], lockfile: Path) -> None:
    locks = parse_lockfile(lockfile)
    expected_relpaths = {s.relative_to(REPO_ROOT).as_posix() for s in schemas}
    lock_relpaths = set(locks.keys())

    missing = sorted(expected_relpaths - lock_relpaths)
    extra = sorted(lock_relpaths - expected_relpaths)
    if missing or extra:
        lines = ["Lockfile entries do not exactly match discovered schemas."]
        if missing:
            lines.append("Missing entries:")
            lines.extend(f"  - {rel}" for rel in missing)
        if extra:
            lines.append("Unexpected extra entries:")
            lines.extend(f"  - {rel}" for rel in extra)
        lines.append("")
        lines.append("If intentional, update lockfile with:")
        lines.append("  python tools/update_schema_locks.py")
        raise InputInvariantError("\n".join(lines))

    for schema_path in schemas:
        rel = schema_path.relative_to(REPO_ROOT).as_posix()
        got = sha256_file(schema_path)
        exp = locks[rel]
        if got != exp:
            raise InputInvariantError(
                "Schema drift detected.\n"
                f"  file:     {rel}\n"
                f"  expected: {exp}\n"
                f"  got:      {got}\n\n"
                f"If intentional, update lockfile with:\n"
                f"  python tools/update_schema_locks.py\n"
            )


def main() -> int:
    try:
        schemas = discover_schemas()
        validate_schemas_are_valid(schemas)
        validate_examples(EXAMPLES)
        validate_lockfile(schemas, LOCKFILE)
    except (jsonschema.ValidationError, jsonschema.SchemaError) as exc:
        print("❌ JSON Schema validation failed", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return EXIT_SCHEMA_VALIDATION_FAILURE
    except ContractValidationError as exc:
        print("❌ EvalSuite contract validation failed", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return exc.exit_code

    print("✅ OK: evalsuite schemas valid, examples valid, lockfile matches.")
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
