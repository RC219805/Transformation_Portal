"""Closed archive-operation configuration for core execution-plan carriers.

These definitions describe data, never executable names or arbitrary CLI fields.
The existing archive governance runner remains the only archive implementation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from transformation_portal.ingest.canonical_json import canonicalize_json

ARCHIVE_CONFIGURATION_SCHEMA = "tp.stage.config.archive.operation.v1"


@dataclass(frozen=True)
class ArchiveOperation:
    """Exact static parameter roles for an existing archive operation."""

    pipeline: str
    files: tuple[str, ...] = ()
    directories: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    integers: tuple[str, ...] = ()
    strings: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()

    @property
    def parameters(self) -> frozenset[str]:
        return frozenset(self.files + self.directories + self.outputs + self.integers + self.strings + self.flags)


ARCHIVE_OPERATIONS: Mapping[str, ArchiveOperation] = MappingProxyType(
    {
        "fixity-scan": ArchiveOperation(
            "archive-gate-a",
            ("archive_index",),
            ("archive_root",),
            ("out_dir",),
            ("workers",),
            flags=("strict", "strict_identity", "validate_schemas"),
            optional=("validate_schemas",),
        ),
        "fixity-verify": ArchiveOperation(
            "archive-gate-a",
            ("hash_manifest",),
            ("archive_root",),
            ("report_path",),
            ("verify_sample", "workers"),
        ),
        "manifest-build": ArchiveOperation(
            "archive-gate-a",
            ("archive_index", "hash_manifest", "rights_jsonl"),
            ("archive_root",),
            ("out_jsonl", "out_summary"),
            strings=("collection_id", "owner"),
            optional=("rights_jsonl",),
        ),
        "rights-apply": ArchiveOperation(
            "archive-gate-a",
            ("manifest_jsonl", "policy_yaml"),
            outputs=("out_jsonl", "out_summary"),
        ),
        "bag-build": ArchiveOperation(
            "archive-gate-b",
            ("manifest_jsonl",),
            ("archive_root",),
            ("bag_dir", "report_json"),
            strings=("source_organization",),
            flags=("validate_with_bagit_python",),
            optional=("validate_with_bagit_python",),
        ),
        "bag-validate": ArchiveOperation(
            "archive-gate-b",
            directories=("bag_dir",),
            outputs=("report_json",),
            flags=("validate_with_bagit_python",),
            optional=("validate_with_bagit_python",),
        ),
        "dedup-plan": ArchiveOperation(
            "archive-gate-b",
            ("manifest_jsonl",),
            outputs=("out_ledger", "out_summary"),
            strings=("approver",),
        ),
        "mets-export": ArchiveOperation(
            "archive-gate-c",
            ("manifest_jsonl",),
            outputs=("out_xml", "out_summary"),
            strings=("href_prefix",),
        ),
        "prov-export": ArchiveOperation(
            "archive-gate-c",
            ("manifest_jsonl",),
            outputs=("out_prov_jsonld", "out_summary"),
            strings=("datetime_field",),
        ),
        "stac-export": ArchiveOperation(
            "archive-gate-c",
            ("manifest_jsonl",),
            outputs=("out_prov_jsonld", "out_stac_catalog", "out_stac_items_dir", "out_summary"),
            strings=("datetime_field",),
            flags=("require_stac",),
            optional=("require_stac",),
        ),
    }
)


def validate_archive_configuration(configuration: Mapping[str, Any]) -> None:
    """Reject incomplete, cross-pipeline, executable or escaping data."""

    expected = {"schema", "configuration_completeness", "pipeline", "operation", "parameters", "input_fingerprints"}
    if set(configuration) != expected:
        raise ValueError("Archive configuration must contain exactly the closed operation fields")
    if (
        configuration["schema"] != ARCHIVE_CONFIGURATION_SCHEMA
        or configuration["configuration_completeness"] != "execution_complete"
    ):
        raise ValueError("Archive operations require execution-complete configuration")
    operation = ARCHIVE_OPERATIONS.get(configuration["operation"])
    if operation is None or configuration["pipeline"] != operation.pipeline:
        raise ValueError("Archive operation does not belong to its pipeline")
    parameters = configuration["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError("Archive parameters must be an object")
    if not operation.parameters.issuperset(parameters) or not operation.parameters.difference(operation.optional).issubset(
        parameters
    ):
        raise ValueError("Archive parameters do not match the closed operation template")
    for name, value in parameters.items():
        if name in operation.flags:
            if type(value) is not bool:
                raise ValueError(f"Archive flag {name} must be a boolean")
        elif name in operation.integers:
            minimum = 0 if name == "verify_sample" else 1
            if type(value) is not int or not minimum <= value <= 1_000_000:
                raise ValueError(f"Archive integer {name} is outside its bound")
        else:
            if not isinstance(value, str) or not value or len(value) > 4096 or any(ord(c) < 32 for c in value):
                raise ValueError(f"Archive parameter {name} must be a bounded printable string")
            if name in operation.outputs:
                path = PurePosixPath(value)
                if "\\" in value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
                    raise ValueError(f"Archive output {name} must be a contained relative path")
            elif name in operation.files + operation.directories:
                path = PurePosixPath(value)
                if not path.is_absolute() or ".." in path.parts or "\\" in value or path.as_posix() != value:
                    raise ValueError(f"Archive input {name} must be an absolute canonical path")
    if configuration["operation"] == "fixity-scan" and (
        parameters["strict"] is not True or parameters["strict_identity"] is not True
    ):
        raise ValueError("Archive fixity scans require strict identity validation")
    fingerprints = configuration["input_fingerprints"]
    expected_files = set(operation.files).intersection(parameters)
    if not isinstance(fingerprints, Mapping) or set(fingerprints) != expected_files:
        raise ValueError("Archive input fingerprints must cover every file input exactly")
    for digest in fingerprints.values():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Archive input fingerprint must be SHA-256")


def archive_configuration_fingerprint(configuration: Mapping[str, Any]) -> str:
    """Bind the complete semantic configuration to its core-plan identity."""

    validate_archive_configuration(configuration)
    return hashlib.sha256(canonicalize_json(dict(configuration))).hexdigest()
