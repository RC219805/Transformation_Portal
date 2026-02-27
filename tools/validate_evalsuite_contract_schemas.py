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
from pathlib import Path
from typing import Dict, List, Tuple

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_ROOT = REPO_ROOT / "docs" / "schemas" / "evalsuite"
LOCK_SCOPE_GLOB = "docs/schemas/evalsuite/**/*.json"

LOCKFILE = REPO_ROOT / "docs" / "contracts" / "SCHEMA_LOCKS.sha256"

FORMAT_CHECKER = jsonschema.FormatChecker()

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_schema_files(schema_root: Path) -> List[Path]:
    """
    Discover evalsuite contract schemas under docs/schemas/evalsuite/**.

    NOTE: This intentionally matches the lockfile scope (LOCK_SCOPE_GLOB):
      docs/schemas/evalsuite/**/*.json
    """
    if not schema_root.exists():
        raise SystemExit(f"Schema root not found: {schema_root}")

    schema_files = [p for p in schema_root.rglob("*.json") if p.is_file()]
    if not schema_files:
        raise SystemExit(f"No schema JSON files found under: {schema_root}")

    # Canonical ordering by repo-relative path
    return sorted(schema_files, key=lambda p: p.relative_to(REPO_ROOT).as_posix())


def parse_lockfile(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise SystemExit(f"LOCKFILE missing: {path}")

    locks: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise SystemExit(f"Invalid lockfile line: {line!r}")
        digest, rel = parts
        if rel in locks:
            raise SystemExit(f"Duplicate lockfile entry for: {rel}")
        locks[rel] = digest
    return locks


def validate_schemas_are_valid(schemas: List[Path]) -> None:
    for s in schemas:
        if not s.exists():
            raise SystemExit(f"Missing schema file: {s}")
        schema_obj = load_json(s)
        # Validates that this JSON is a valid Draft 2020-12 schema
        jsonschema.Draft202012Validator.check_schema(schema_obj)


def validate_examples(examples: List[Tuple[Path, Path]]) -> None:
    for example_path, schema_path in examples:
        if not example_path.exists():
            raise SystemExit(f"Missing example fixture: {example_path}")
        if not schema_path.exists():
            raise SystemExit(f"Missing schema for example: {schema_path}")

        schema_obj = load_json(schema_path)
        instance = load_json(example_path)

        jsonschema.Draft202012Validator(
            schema_obj,
            format_checker=FORMAT_CHECKER,
        ).validate(instance)


def validate_lockfile(schemas: List[Path], lockfile: Path) -> None:
    locks = parse_lockfile(lockfile)

    expected_rels = {s.relative_to(REPO_ROOT).as_posix() for s in schemas}
    lock_rels = set(locks.keys())

    missing = expected_rels - lock_rels
    if missing:
        missing_sorted = "\n".join(f"  - {p}" for p in sorted(missing))
        raise SystemExit(
            "Lockfile missing entries for the following schema file(s):\n"
            f"{missing_sorted}\n\n"
            "If intentional, update lockfile with:\n"
            "  python tools/update_schema_locks.py\n"
        )

    extra = lock_rels - expected_rels
    if extra:
        extra_sorted = "\n".join(f"  - {p}" for p in sorted(extra))
        raise SystemExit(
            "Lockfile contains extra entries (removed/renamed schema files or scope mismatch).\n"
            f"Expected scope: {LOCK_SCOPE_GLOB}\n\n"
            "Extra entries:\n"
            f"{extra_sorted}\n\n"
            "If intentional, regenerate lockfile with:\n"
            "  python tools/update_schema_locks.py\n"
        )

    for s in schemas:
        rel = s.relative_to(REPO_ROOT).as_posix()
        got = sha256_file(s)
        exp = locks[rel]
        if got != exp:
            raise SystemExit(
                "Schema drift detected.\n"
                f"  file:     {rel}\n"
                f"  expected: {exp}\n"
                f"  got:      {got}\n\n"
                f"If intentional, update lockfile with:\n"
                f"  python tools/update_schema_locks.py\n"
            )


def main() -> int:
    try:
        schemas = discover_schema_files(SCHEMA_ROOT)
        validate_schemas_are_valid(schemas)
        validate_examples(EXAMPLES)
        validate_lockfile(schemas, LOCKFILE)
    except jsonschema.ValidationError as e:
        print("❌ JSON Schema validation failed", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 2

    print("✅ OK: evalsuite schemas valid, examples valid, lockfile matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
