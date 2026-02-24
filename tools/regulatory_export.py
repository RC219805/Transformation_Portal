#!/usr/bin/env python3
"""
Phase 3.5 Regulatory Export Mode: generate Article 53-aligned disclosure artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from bundle_root_common import (
    EXPECTED_HASH_MANIFEST_FILENAME,
    EXPECTED_HASH_SUMMARY_FILENAME,
    EXPECTED_MANIFEST_FILENAME,
    HEX64_RE,
    compute_bundle_root_sha256,
    validate_manifest_structure,
)

EXIT_EXPORT_BUILD_FAILURE = 31
EXIT_EXPORT_WRITE_FAILURE = 32
EXPORT_MODE_VERSION = "1"
RISK_METADATA_SCHEMA_VERSION = "1"
SOURCE_TAXONOMY_SCHEMA_VERSION = "1"
DEFAULT_COMPLIANCE_PROFILE_ID = "EU-AI-ACT-ART53-GPAI-V1"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SIGNALS_SUPPORTED = {
    "robots_txt",
    "http_headers",
    "meta_tags",
    "rights_registry",
    "platform_api",
}
SOURCE_CATEGORIES = {
    "public_dataset",
    "licensed_private_dataset",
    "web_scraped",
    "user_provided",
    "synthetic",
    "other",
}
PROVENANCE_TYPES = {"first_party", "third_party", "mixed"}
RISK_CONTROL_STATUSES = {"implemented", "planned", "not_applicable"}
HASH_MANIFEST_COLUMNS = [
    "origin_drive",
    "partition",
    "relpath",
    "filesize_bytes",
    "sha256",
    "hash_status",
    "error",
]


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def sha256_hexdigest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _require_string(payload: Mapping[str, object], field: str, context: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{field} must be a non-empty string")
    return value.strip()


def _require_bool(payload: Mapping[str, object], field: str, context: str) -> bool:
    value = payload.get(field)
    if type(value) is not bool:
        raise ValueError(f"{context}.{field} must be a boolean")
    return value


def _require_non_negative_int(payload: Mapping[str, object], field: str, context: str) -> int:
    value = payload.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"{context}.{field} must be a non-negative integer")
    return value


def _require_object(payload: Mapping[str, object], field: str, context: str) -> dict[str, object]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{field} must be an object")
    return value


def _validate_exact_fields(payload: Mapping[str, object], *, allowed: set[str], required: set[str], context: str) -> None:
    keys = set(payload)
    missing = sorted(required - keys)
    if missing:
        raise ValueError(f"{context} missing required field(s): {', '.join(missing)}")
    unexpected = sorted(keys - allowed)
    if unexpected:
        raise ValueError(f"{context} has unexpected field(s): {', '.join(unexpected)}")


def _load_json_object(path: Path, *, context: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a JSON object")
    return payload


def _load_evidence_manifest(path: Path, *, strict: bool) -> dict[str, object]:
    if path.name != EXPECTED_MANIFEST_FILENAME:
        raise ValueError(f"--bundle-manifest must reference {EXPECTED_MANIFEST_FILENAME}")
    payload = _load_json_object(path, context="bundle-manifest")
    validate_manifest_structure(payload, strict=strict)
    if "bundle_root_sha256" not in payload:
        raise ValueError("bundle-manifest must include bundle_root fields; run tools/compute_bundle_root.py --write")
    computed_root = compute_bundle_root_sha256(payload)
    if computed_root != payload["bundle_root_sha256"]:
        raise ValueError("bundle-manifest bundle_root_sha256 does not match canonical projection")
    return payload


def _validate_risk_controls(payload: dict[str, object]) -> list[dict[str, object]]:
    controls = payload.get("risk_controls")
    if controls is None:
        return []
    if not isinstance(controls, list):
        raise ValueError("risk-metadata.risk_controls must be an array")
    normalized: list[dict[str, object]] = []
    for idx, control in enumerate(controls):
        context = f"risk-metadata.risk_controls[{idx}]"
        if not isinstance(control, dict):
            raise ValueError(f"{context} must be an object")
        _validate_exact_fields(
            control,
            allowed={"control_id", "description", "status"},
            required={"control_id", "description", "status"},
            context=context,
        )
        status = _require_string(control, "status", context)
        if status not in RISK_CONTROL_STATUSES:
            allowed_values = ", ".join(sorted(RISK_CONTROL_STATUSES))
            raise ValueError(f"{context}.status must be one of: {allowed_values}")
        normalized.append(
            {
                "control_id": _require_string(control, "control_id", context),
                "description": _require_string(control, "description", context),
                "status": status,
            }
        )
    return normalized


def validate_risk_metadata(payload: dict[str, object]) -> dict[str, object]:
    _validate_exact_fields(
        payload,
        allowed={
            "schema_version",
            "risk_profile_id",
            "regulatory_regime",
            "content_rights",
            "copyright_compliance",
            "risk_controls",
        },
        required={
            "schema_version",
            "risk_profile_id",
            "regulatory_regime",
            "content_rights",
            "copyright_compliance",
        },
        context="risk-metadata",
    )
    if payload["schema_version"] != RISK_METADATA_SCHEMA_VERSION:
        raise ValueError(f"risk-metadata.schema_version must be {RISK_METADATA_SCHEMA_VERSION!r}")

    content_rights = _require_object(payload, "content_rights", "risk-metadata")
    _validate_exact_fields(
        content_rights,
        allowed={"policy_id", "policy_version", "policy_url", "notes"},
        required={"policy_id", "policy_version"},
        context="risk-metadata.content_rights",
    )
    normalized_content_rights = {
        "policy_id": _require_string(content_rights, "policy_id", "risk-metadata.content_rights"),
        "policy_version": _require_string(content_rights, "policy_version", "risk-metadata.content_rights"),
    }
    if "policy_url" in content_rights:
        normalized_content_rights["policy_url"] = _require_string(content_rights, "policy_url", "risk-metadata.content_rights")
    if "notes" in content_rights:
        normalized_content_rights["notes"] = _require_string(content_rights, "notes", "risk-metadata.content_rights")

    copyright_compliance = _require_object(payload, "copyright_compliance", "risk-metadata")
    _validate_exact_fields(
        copyright_compliance,
        allowed={
            "tdm_opt_out_detection",
            "signals_supported",
            "removal_process_documented",
            "removal_deltas_affect_root",
        },
        required={
            "tdm_opt_out_detection",
            "signals_supported",
            "removal_process_documented",
            "removal_deltas_affect_root",
        },
        context="risk-metadata.copyright_compliance",
    )

    signals_supported = copyright_compliance.get("signals_supported")
    if not isinstance(signals_supported, list) or not signals_supported:
        raise ValueError("risk-metadata.copyright_compliance.signals_supported must be a non-empty array")
    normalized_signals: list[str] = []
    for idx, signal in enumerate(signals_supported):
        if not isinstance(signal, str) or not signal.strip():
            raise ValueError(f"risk-metadata.copyright_compliance.signals_supported[{idx}] must be a non-empty string")
        normalized_signal = signal.strip()
        if normalized_signal not in SIGNALS_SUPPORTED:
            allowed_values = ", ".join(sorted(SIGNALS_SUPPORTED))
            raise ValueError(
                "risk-metadata.copyright_compliance.signals_supported "
                f"contains unsupported value {normalized_signal!r}; expected one of: {allowed_values}"
            )
        normalized_signals.append(normalized_signal)

    controls = _validate_risk_controls(payload)
    return {
        "schema_version": RISK_METADATA_SCHEMA_VERSION,
        "risk_profile_id": _require_string(payload, "risk_profile_id", "risk-metadata"),
        "regulatory_regime": _require_string(payload, "regulatory_regime", "risk-metadata"),
        "content_rights": normalized_content_rights,
        "copyright_compliance": {
            "tdm_opt_out_detection": _require_bool(
                copyright_compliance, "tdm_opt_out_detection", "risk-metadata.copyright_compliance"
            ),
            "signals_supported": sorted(set(normalized_signals)),
            "removal_process_documented": _require_bool(
                copyright_compliance, "removal_process_documented", "risk-metadata.copyright_compliance"
            ),
            "removal_deltas_affect_root": _require_bool(
                copyright_compliance, "removal_deltas_affect_root", "risk-metadata.copyright_compliance"
            ),
        },
        "risk_controls": controls,
    }


def _validate_collection_period(payload: dict[str, object], context: str) -> dict[str, str]:
    _validate_exact_fields(
        payload,
        allowed={"start", "end"},
        required={"start", "end"},
        context=context,
    )
    start = _require_string(payload, "start", context)
    end = _require_string(payload, "end", context)
    if DATE_RE.fullmatch(start) is None:
        raise ValueError(f"{context}.start must match YYYY-MM-DD")
    if DATE_RE.fullmatch(end) is None:
        raise ValueError(f"{context}.end must match YYYY-MM-DD")
    if start > end:
        raise ValueError(f"{context}.start must be <= {context}.end")
    return {"start": start, "end": end}


def _validate_string_list(payload: dict[str, object], field: str, context: str) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise ValueError(f"{context}.{field} must be an array")
    normalized: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{context}.{field}[{idx}] must be a non-empty string")
        normalized.append(item.strip())
    return normalized


def validate_source_taxonomy(payload: dict[str, object]) -> dict[str, object]:
    _validate_exact_fields(
        payload,
        allowed={"schema_version", "catalog_id", "sources"},
        required={"schema_version", "sources"},
        context="source-taxonomy",
    )
    if payload["schema_version"] != SOURCE_TAXONOMY_SCHEMA_VERSION:
        raise ValueError(f"source-taxonomy.schema_version must be {SOURCE_TAXONOMY_SCHEMA_VERSION!r}")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source-taxonomy.sources must be a non-empty array")

    normalized_sources: list[dict[str, object]] = []
    for idx, source in enumerate(sources):
        context = f"source-taxonomy.sources[{idx}]"
        if not isinstance(source, dict):
            raise ValueError(f"{context} must be an object")
        _validate_exact_fields(
            source,
            allowed={
                "source_id",
                "category",
                "provenance_type",
                "license_class",
                "synthetic",
                "tdm_compliance_note",
                "crawler",
                "collection_period",
                "top_domains",
                "geographic_coverage",
                "record_count_estimate",
                "byte_count_estimate",
                "processing_disclosure",
            },
            required={
                "source_id",
                "category",
                "provenance_type",
                "license_class",
                "synthetic",
                "tdm_compliance_note",
            },
            context=context,
        )

        category = _require_string(source, "category", context)
        if category not in SOURCE_CATEGORIES:
            allowed_values = ", ".join(sorted(SOURCE_CATEGORIES))
            raise ValueError(f"{context}.category must be one of: {allowed_values}")

        provenance_type = _require_string(source, "provenance_type", context)
        if provenance_type not in PROVENANCE_TYPES:
            allowed_values = ", ".join(sorted(PROVENANCE_TYPES))
            raise ValueError(f"{context}.provenance_type must be one of: {allowed_values}")

        synthetic = _require_bool(source, "synthetic", context)
        if category == "synthetic" and not synthetic:
            raise ValueError(f"{context}.synthetic must be true when category is 'synthetic'")

        normalized_source: dict[str, object] = {
            "source_id": _require_string(source, "source_id", context),
            "category": category,
            "provenance_type": provenance_type,
            "license_class": _require_string(source, "license_class", context),
            "synthetic": synthetic,
            "tdm_compliance_note": _require_string(source, "tdm_compliance_note", context),
        }

        if "collection_period" in source:
            collection_period_value = source["collection_period"]
            if not isinstance(collection_period_value, dict):
                raise ValueError(f"{context}.collection_period must be an object")
            normalized_source["collection_period"] = _validate_collection_period(
                collection_period_value, f"{context}.collection_period"
            )

        if "crawler" in source:
            normalized_source["crawler"] = _require_string(source, "crawler", context)
        if "top_domains" in source:
            normalized_source["top_domains"] = sorted(set(_validate_string_list(source, "top_domains", context)))
        if "geographic_coverage" in source:
            normalized_source["geographic_coverage"] = sorted(
                set(_validate_string_list(source, "geographic_coverage", context))
            )
        if "record_count_estimate" in source:
            normalized_source["record_count_estimate"] = _require_non_negative_int(source, "record_count_estimate", context)
        if "byte_count_estimate" in source:
            normalized_source["byte_count_estimate"] = _require_non_negative_int(source, "byte_count_estimate", context)
        if "processing_disclosure" in source:
            normalized_source["processing_disclosure"] = _require_string(source, "processing_disclosure", context)

        if category == "web_scraped":
            missing = []
            for required_field in ("crawler", "collection_period", "top_domains"):
                if required_field not in normalized_source:
                    missing.append(required_field)
            if missing:
                raise ValueError(f"{context} missing required field(s) for web_scraped category: {', '.join(missing)}")

        normalized_sources.append(normalized_source)

    normalized_sources = sorted(
        normalized_sources,
        key=lambda entry: (str(entry["source_id"]), str(entry["category"]), str(entry["license_class"])),
    )
    normalized_payload: dict[str, object] = {"schema_version": SOURCE_TAXONOMY_SCHEMA_VERSION, "sources": normalized_sources}
    if "catalog_id" in payload:
        normalized_payload["catalog_id"] = _require_string(payload, "catalog_id", "source-taxonomy")
    return normalized_payload


def _counter_to_list(counter: Counter[str], key_name: str, value_name: str, *, limit: int) -> list[dict[str, object]]:
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    limited = ordered[:limit]
    return [{key_name: key, value_name: value} for key, value in limited]


def _open_hash_manifest_reader(path: Path) -> csv.DictReader:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        handle = path.open("r", encoding="utf-8", newline="")

    header_line: str | None = None
    for line in handle:
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        header_line = line
        break

    if header_line is None:
        handle.close()
        raise ValueError(f"hash manifest has no header row: {path}")

    def _lines_with_header(first: str, remainder: Iterable[str]) -> Iterable[str]:
        yield first
        yield from remainder

    reader = csv.DictReader(_lines_with_header(header_line, handle))
    if reader.fieldnames is None:
        handle.close()
        raise ValueError(f"hash manifest has no header row: {path}")
    missing_columns = [col for col in HASH_MANIFEST_COLUMNS if col not in reader.fieldnames]
    if missing_columns:
        handle.close()
        missing = ", ".join(missing_columns)
        raise ValueError(f"hash manifest is missing required columns: {missing}")

    # Keep backing file handle alive for reader lifecycle.
    reader._tp_handle = handle  # type: ignore[attr-defined]
    return reader


def _close_hash_manifest_reader(reader: csv.DictReader) -> None:
    handle = getattr(reader, "_tp_handle", None)
    if handle is not None:
        handle.close()


def _file_extension(relpath: str) -> str:
    basename = Path(relpath).name
    if "." not in basename:
        return "[none]"
    ext = basename.rsplit(".", 1)[1].strip().lower()
    return f".{ext}" if ext else "[none]"


def summarize_hash_manifest(hash_manifest_path: Path, *, top_n: int) -> dict[str, object]:
    status_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    origin_drive_counts: Counter[str] = Counter()
    rows_total = 0
    total_bytes_hashed = 0

    reader = _open_hash_manifest_reader(hash_manifest_path)
    try:
        for row_index, row in enumerate(reader, start=1):
            origin_drive = str(row.get("origin_drive") or "").strip() or "[unknown]"
            relpath = str(row.get("relpath") or "").strip()
            hash_status = str(row.get("hash_status") or "").strip() or "unknown"
            raw_size = str(row.get("filesize_bytes") or "0").strip()
            try:
                filesize = int(raw_size)
            except ValueError as exc:
                raise ValueError(f"hash manifest row {row_index} has invalid filesize_bytes: {raw_size!r}") from exc

            rows_total += 1
            origin_drive_counts[origin_drive] += 1
            extension_counts[_file_extension(relpath)] += 1
            status_counts[hash_status] += 1
            if hash_status == "ok":
                total_bytes_hashed += max(filesize, 0)
    finally:
        _close_hash_manifest_reader(reader)

    return {
        "rows_total": rows_total,
        "total_bytes_hashed": total_bytes_hashed,
        "hash_status_counts": _counter_to_list(status_counts, "status", "row_count", limit=max(top_n, len(status_counts))),
        "top_origin_drives": _counter_to_list(origin_drive_counts, "origin_drive", "row_count", limit=top_n),
        "top_file_extensions": _counter_to_list(extension_counts, "extension", "row_count", limit=top_n),
    }


def summarize_source_taxonomy(source_taxonomy: Mapping[str, object], *, top_n: int) -> dict[str, object]:
    sources = source_taxonomy["sources"]
    assert isinstance(sources, list)

    category_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    crawler_names: set[str] = set()
    collection_starts: list[str] = []
    collection_ends: list[str] = []
    web_scraped_sources = 0

    for source in sources:
        assert isinstance(source, dict)
        category = str(source["category"])
        category_counts[category] += 1

        if category != "web_scraped":
            continue
        web_scraped_sources += 1

        crawler_value = source.get("crawler")
        if isinstance(crawler_value, str) and crawler_value.strip():
            crawler_names.add(crawler_value.strip())

        collection_period = source.get("collection_period")
        if isinstance(collection_period, dict):
            start = collection_period.get("start")
            end = collection_period.get("end")
            if isinstance(start, str):
                collection_starts.append(start)
            if isinstance(end, str):
                collection_ends.append(end)

        top_domains = source.get("top_domains")
        if isinstance(top_domains, list):
            for domain in top_domains:
                if isinstance(domain, str) and domain.strip():
                    domain_counts[domain.strip().lower()] += 1

    web_scraped_summary: dict[str, object] = {
        "source_count": web_scraped_sources,
        "crawler_names": sorted(crawler_names),
        "top_domains": _counter_to_list(domain_counts, "domain", "source_count", limit=top_n),
    }
    if collection_starts and collection_ends:
        web_scraped_summary["collection_period"] = {
            "start": min(collection_starts),
            "end": max(collection_ends),
        }

    return {
        "source_count_total": len(sources),
        "source_category_counts": _counter_to_list(
            category_counts, "category", "source_count", limit=max(top_n, len(category_counts))
        ),
        "web_scraped": web_scraped_summary,
    }


def _format_table(rows: list[tuple[str, str]], headers: tuple[str, str]) -> str:
    lines = [f"| {headers[0]} | {headers[1]} |", "| --- | ---: |"]
    for left, right in rows:
        lines.append(f"| {left} | {right} |")
    return "\n".join(lines)


def _as_bool_word(value: bool) -> str:
    return "true" if value else "false"


def render_markdown(export_payload: Mapping[str, object], *, top_n: int) -> str:
    bundle_binding = export_payload["bundle_binding"]
    assert isinstance(bundle_binding, dict)
    phase_versions = bundle_binding["phase_versions"]
    assert isinstance(phase_versions, dict)

    artifact_digests = export_payload["artifact_digests"]
    assert isinstance(artifact_digests, dict)

    training_summary = export_payload["training_data_summary"]
    assert isinstance(training_summary, dict)

    source_summary = training_summary["source_taxonomy_summary"]
    assert isinstance(source_summary, dict)

    web_scraped = source_summary["web_scraped"]
    assert isinstance(web_scraped, dict)

    hash_manifest_summary = training_summary["hash_manifest_summary"]
    assert isinstance(hash_manifest_summary, dict)

    copyright_compliance = export_payload["copyright_compliance"]
    assert isinstance(copyright_compliance, dict)

    commands = export_payload["verification_commands"]
    assert isinstance(commands, list)

    lines: list[str] = []
    lines.append("# Regulatory Export Summary")
    lines.append("")
    lines.append("## Compliance Profile")
    lines.append(f"- Export mode version: `{EXPORT_MODE_VERSION}`")
    lines.append(f"- Compliance profile: `{export_payload['compliance_profile_id']}`")
    lines.append(f"- Bundle root sha256: `{bundle_binding['bundle_root_sha256']}`")
    lines.append(
        f"- Bundle root contract: `{bundle_binding['bundle_root_algorithm']}` / "
        f"preimage `{bundle_binding['bundle_root_preimage_version']}`"
    )
    lines.append(
        "- Phase versions: "
        f"phase3={phase_versions['phase3_version']}, "
        f"phase3.1={phase_versions['phase3_1_version']}, "
        f"phase3.2={phase_versions['phase3_2_version']}"
    )
    lines.append("")
    lines.append("## Integrity Bindings")
    lines.append(f"- Evidence bundle manifest sha256: `{bundle_binding['bundle_manifest_sha256']}`")
    lines.append(f"- Risk metadata sha256: `{artifact_digests['risk_metadata_sha256']}`")
    lines.append(f"- Source taxonomy sha256: `{artifact_digests['source_taxonomy_sha256']}`")
    lines.append(f"- Hash manifest sha256: `{artifact_digests['hash_manifest_sha256']}`")
    lines.append(f"- Hash summary sha256: `{artifact_digests['hash_summary_sha256']}`")
    lines.append("")
    lines.append("## Training Data Summary")
    lines.append(f"- Rows total: `{int(hash_manifest_summary['rows_total'])}`")
    lines.append(f"- Bytes hashed (`status=ok`): `{int(hash_manifest_summary['total_bytes_hashed'])}`")
    lines.append(f"- Top-N cutoff: `{top_n}`")
    lines.append("")
    lines.append("### Source Categories")
    category_rows: list[tuple[str, str]] = []
    for item in source_summary["source_category_counts"]:
        assert isinstance(item, dict)
        category_rows.append((str(item["category"]), str(int(item["source_count"]))))
    lines.append(_format_table(category_rows, ("Category", "Source Count")))
    lines.append("")
    lines.append("### Top Origin Drives")
    origin_rows: list[tuple[str, str]] = []
    for item in hash_manifest_summary["top_origin_drives"]:
        assert isinstance(item, dict)
        origin_rows.append((str(item["origin_drive"]), str(int(item["row_count"]))))
    lines.append(_format_table(origin_rows, ("Origin Drive", "Row Count")))
    lines.append("")
    lines.append("### Top File Extensions")
    extension_rows: list[tuple[str, str]] = []
    for item in hash_manifest_summary["top_file_extensions"]:
        assert isinstance(item, dict)
        extension_rows.append((str(item["extension"]), str(int(item["row_count"]))))
    lines.append(_format_table(extension_rows, ("Extension", "Row Count")))
    lines.append("")
    lines.append("### Web-Scraped Signals")
    lines.append(f"- Web-scraped source count: `{int(web_scraped['source_count'])}`")
    collection_period = web_scraped.get("collection_period")
    if isinstance(collection_period, dict):
        lines.append(f"- Collection period: `{collection_period['start']}` to `{collection_period['end']}`")
    else:
        lines.append("- Collection period: `not_declared`")
    crawler_names = web_scraped.get("crawler_names")
    if isinstance(crawler_names, list) and crawler_names:
        lines.append(f"- Crawler names: `{', '.join(str(name) for name in crawler_names)}`")
    else:
        lines.append("- Crawler names: `not_declared`")
    lines.append("")
    lines.append("### Top Domains (Web-Scraped)")
    domain_rows: list[tuple[str, str]] = []
    for item in web_scraped["top_domains"]:
        assert isinstance(item, dict)
        domain_rows.append((str(item["domain"]), str(int(item["source_count"]))))
    lines.append(_format_table(domain_rows, ("Domain", "Source Count")))
    lines.append("")
    lines.append("## Copyright Compliance")
    lines.append(
        f"- TDM opt-out detection performed: " f"`{_as_bool_word(bool(copyright_compliance['tdm_opt_out_detection']))}`"
    )
    signals = copyright_compliance["signals_supported"]
    assert isinstance(signals, list)
    lines.append(f"- Signals supported: `{', '.join(str(signal) for signal in signals)}`")
    lines.append(
        f"- Removal process documented: " f"`{_as_bool_word(bool(copyright_compliance['removal_process_documented']))}`"
    )
    lines.append(
        f"- Removal deltas affect current root: "
        f"`{_as_bool_word(bool(copyright_compliance['removal_deltas_affect_root']))}`"
    )
    lines.append("")
    lines.append("## Verification Commands")
    for command in commands:
        lines.append(f"1. `{command}`")
    lines.append("")
    lines.append("## Confidentiality Statement")
    lines.append(str(export_payload["confidentiality_statement"]))
    lines.append("")
    lines.append("## Article 78 Note")
    lines.append(str(export_payload["article_78_reference"]))
    lines.append("")
    return "\n".join(lines)


def build_export_payload(
    *,
    manifest: Mapping[str, object],
    manifest_path: Path,
    hash_manifest_path: Path,
    hash_summary_path: Path,
    risk_metadata: Mapping[str, object],
    risk_metadata_path: Path,
    source_taxonomy: Mapping[str, object],
    source_taxonomy_path: Path,
    compliance_profile_id: str,
    top_n: int,
) -> dict[str, object]:
    hash_manifest_summary = summarize_hash_manifest(hash_manifest_path, top_n=top_n)
    source_taxonomy_summary = summarize_source_taxonomy(source_taxonomy, top_n=top_n)

    hash_summary_payload = _load_json_object(hash_summary_path, context="hash-summary")
    rows_total = hash_summary_payload.get("rows_total")
    if type(rows_total) is int and rows_total != hash_manifest_summary["rows_total"]:
        raise ValueError("hash-summary.rows_total does not match rows parsed from hash-manifest")

    total_bytes_hashed = hash_summary_payload.get("total_bytes_hashed")
    if type(total_bytes_hashed) is int and total_bytes_hashed != hash_manifest_summary["total_bytes_hashed"]:
        raise ValueError("hash-summary.total_bytes_hashed does not match rows parsed from hash-manifest")

    return {
        "export_mode_version": EXPORT_MODE_VERSION,
        "compliance_profile_id": compliance_profile_id,
        "bundle_binding": {
            "bundle_manifest_sha256": sha256_hexdigest(manifest_path),
            "bundle_root_algorithm": manifest["bundle_root_algorithm"],
            "bundle_root_preimage_version": manifest["bundle_root_preimage_version"],
            "bundle_root_sha256": manifest["bundle_root_sha256"],
            "phase_versions": {
                "phase3_version": manifest["phase3_version"],
                "phase3_1_version": manifest["phase3_1_version"],
                "phase3_2_version": manifest["phase3_2_version"],
            },
        },
        "artifact_digests": {
            "risk_metadata_sha256": sha256_hexdigest(risk_metadata_path),
            "source_taxonomy_sha256": sha256_hexdigest(source_taxonomy_path),
            "hash_manifest_sha256": sha256_hexdigest(hash_manifest_path),
            "hash_summary_sha256": sha256_hexdigest(hash_summary_path),
        },
        "risk_metadata_profile_id": risk_metadata["risk_profile_id"],
        "regulatory_regime": risk_metadata["regulatory_regime"],
        "content_rights": risk_metadata["content_rights"],
        "copyright_compliance": risk_metadata["copyright_compliance"],
        "risk_controls": risk_metadata["risk_controls"],
        "training_data_summary": {
            "hash_manifest_summary": hash_manifest_summary,
            "source_taxonomy_summary": source_taxonomy_summary,
        },
        "verification_commands": [
            f"python tools/verify_evidence_bundle_manifest.py --bundle-manifest {manifest_path} --bundle-dir {manifest_path.parent}",
            f"python tools/regulatory_export.py --bundle-manifest {manifest_path} "
            f"--risk-metadata {risk_metadata_path} --source-taxonomy {source_taxonomy_path} "
            f"--out-json <path>/regulatory_export.json --out-markdown <path>/regulatory_export.md",
        ],
        "verification_expected_exit_codes": {
            "verify_evidence_bundle_manifest": 0,
            "regulatory_export": 0,
        },
        "confidentiality_statement": (
            "This public summary intentionally omits file-level and proprietary training corpus details. "
            "Integrity remains cryptographically verifiable via bundle_root_sha256 and referenced digest bindings."
        ),
        "article_78_reference": (
            "Confidential technical details may be disclosed to competent authorities under Article 78 controls."
        ),
    }


def _resolve_artifact_path(
    *,
    provided_path: str | None,
    bundle_dir: Path,
    expected_name: str,
    arg_name: str,
) -> Path:
    if provided_path is not None:
        path = Path(provided_path)
    else:
        path = bundle_dir / expected_name
    if path.name != expected_name:
        raise ValueError(f"--{arg_name} must reference {expected_name}")
    return path


def _ensure_digest_matches(path: Path, *, expected_sha256: str, field_name: str) -> None:
    if HEX64_RE.fullmatch(expected_sha256) is None:
        raise ValueError(f"{field_name} is not a valid lowercase sha256 digest")
    actual = sha256_hexdigest(path)
    if actual != expected_sha256:
        raise ValueError(f"{field_name} does not match digest of {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", required=True, help="Path to evidence_bundle_manifest.json")
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Bundle directory root (defaults to bundle-manifest parent directory)",
    )
    parser.add_argument(
        "--hash-manifest",
        default=None,
        help="Optional override path for hash_manifest.csv.gz (default: bundle-manifest declared path)",
    )
    parser.add_argument(
        "--hash-summary",
        default=None,
        help="Optional override path for hash_summary.json (default: bundle-manifest declared path)",
    )
    parser.add_argument("--risk-metadata", required=True, help="Path to risk metadata JSON")
    parser.add_argument("--source-taxonomy", required=True, help="Path to source taxonomy JSON")
    parser.add_argument("--out-json", required=True, help="Path to write regulatory export JSON")
    parser.add_argument(
        "--out-markdown",
        default=None,
        help="Optional path to write deterministic regulatory export markdown",
    )
    parser.add_argument(
        "--compliance-profile-id",
        default=DEFAULT_COMPLIANCE_PROFILE_ID,
        help=f"Compliance profile identifier (default: {DEFAULT_COMPLIANCE_PROFILE_ID})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N row limits used in summary tables (default: 10)",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strictly validate evidence bundle manifest shape (default: true)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.top_n <= 0:
        print("Regulatory export failed: --top-n must be positive")
        return EXIT_EXPORT_BUILD_FAILURE

    try:
        manifest_path = Path(args.bundle_manifest)
        bundle_dir = Path(args.bundle_dir) if args.bundle_dir is not None else manifest_path.parent
        manifest = _load_evidence_manifest(manifest_path, strict=args.strict)

        hash_manifest_path = _resolve_artifact_path(
            provided_path=args.hash_manifest,
            bundle_dir=bundle_dir,
            expected_name=EXPECTED_HASH_MANIFEST_FILENAME,
            arg_name="hash-manifest",
        )
        hash_summary_path = _resolve_artifact_path(
            provided_path=args.hash_summary,
            bundle_dir=bundle_dir,
            expected_name=EXPECTED_HASH_SUMMARY_FILENAME,
            arg_name="hash-summary",
        )

        # Enforce that these artifacts still match the digests bound in bundle_root projection.
        _ensure_digest_matches(
            hash_manifest_path,
            expected_sha256=str(manifest["hash_manifest_sha256"]),
            field_name="hash_manifest_sha256",
        )
        _ensure_digest_matches(
            hash_summary_path,
            expected_sha256=str(manifest["hash_summary_sha256"]),
            field_name="hash_summary_sha256",
        )

        risk_metadata_path = Path(args.risk_metadata)
        source_taxonomy_path = Path(args.source_taxonomy)
        risk_metadata = validate_risk_metadata(_load_json_object(risk_metadata_path, context="risk-metadata"))
        source_taxonomy = validate_source_taxonomy(_load_json_object(source_taxonomy_path, context="source-taxonomy"))

        export_payload = build_export_payload(
            manifest=manifest,
            manifest_path=manifest_path,
            hash_manifest_path=hash_manifest_path,
            hash_summary_path=hash_summary_path,
            risk_metadata=risk_metadata,
            risk_metadata_path=risk_metadata_path,
            source_taxonomy=source_taxonomy,
            source_taxonomy_path=source_taxonomy_path,
            compliance_profile_id=args.compliance_profile_id.strip(),
            top_n=args.top_n,
        )

        json_bytes = (
            json.dumps(
                export_payload,
                indent=2,
                sort_keys=True,
                separators=(",", ": "),
            ).encode("utf-8")
            + b"\n"
        )
        atomic_write(Path(args.out_json), json_bytes)

        if args.out_markdown is not None:
            markdown = render_markdown(export_payload, top_n=args.top_n)
            atomic_write(Path(args.out_markdown), markdown.encode("utf-8") + b"\n")

        print(f"Regulatory export written to {args.out_json}")
        if args.out_markdown is not None:
            print(f"Regulatory markdown written to {args.out_markdown}")
        return 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"Regulatory export failed: {exc}")
        return EXIT_EXPORT_BUILD_FAILURE
    except Exception as exc:
        print(f"Regulatory export write failed: {exc}")
        return EXIT_EXPORT_WRITE_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
