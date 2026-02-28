#!/usr/bin/env python3
"""PREMIS v3 event emission helpers for archive governance workflows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from archive_governance_common import deterministic_json_dumps

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_VALIDATION_ERROR = 3

RFC3339_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")


def _iso_now_utc() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def build_premis_event(
    *,
    event_type: str,
    event_detail: str,
    event_outcome: str,
    agent_id: str,
    object_ids: list[str],
    event_datetime: str | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Build a PREMIS event payload."""
    if event_outcome not in {"success", "failure"}:
        raise ValueError("event_outcome must be 'success' or 'failure'")

    object_entries = [
        {
            "linkingObjectIdentifierType": "path",
            "linkingObjectIdentifierValue": object_id,
        }
        for object_id in sorted(set(object_ids))
        if object_id
    ]

    payload: dict[str, Any] = {
        "premis_version": "3.0",
        "event": {
            "eventIdentifier": {
                "eventIdentifierType": "uuid",
                "eventIdentifierValue": event_id or str(uuid4()),
            },
            "eventType": event_type,
            "eventDateTime": event_datetime or _iso_now_utc(),
            "eventDetail": event_detail,
            "eventOutcomeInformation": {
                "eventOutcome": event_outcome,
            },
            "linkingAgentIdentifier": [
                {
                    "linkingAgentIdentifierType": "software",
                    "linkingAgentIdentifierValue": agent_id,
                }
            ],
            "linkingObjectIdentifier": object_entries,
        },
    }
    return payload


def append_event(path: Path, payload: dict[str, Any]) -> None:
    """Append one deterministic JSONL PREMIS event record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = deterministic_json_dumps(payload, pretty=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)


def _expect_object(value: Any, *, path: str, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number}: {path} must be an object")
    return value


def _expect_string(value: Any, *, path: str, line_number: int, min_length: int = 0) -> str:
    if not isinstance(value, str):
        raise ValueError(f"line {line_number}: {path} must be a string")
    if len(value) < min_length:
        raise ValueError(f"line {line_number}: {path} must be length >= {min_length}")
    return value


def _expect_array(value: Any, *, path: str, line_number: int) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"line {line_number}: {path} must be an array")
    return value


def _expect_exact_keys(obj: dict[str, Any], *, required: set[str], path: str, line_number: int) -> None:
    missing = sorted(required.difference(obj.keys()))
    if missing:
        raise ValueError(f"line {line_number}: missing {path}.{', '.join(missing)}")
    extras = sorted(set(obj.keys()).difference(required))
    if extras:
        raise ValueError(f"line {line_number}: unexpected {path}.{', '.join(extras)}")


def _validate_rfc3339_timestamp(value: str, *, path: str, line_number: int) -> None:
    if RFC3339_TIMESTAMP_RE.fullmatch(value) is None:
        raise ValueError(f"line {line_number}: {path} must be RFC3339 date-time")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"line {line_number}: {path} must be RFC3339 date-time") from exc


def _validate_event(payload: dict[str, Any], *, line_number: int) -> None:
    _expect_exact_keys(payload, required={"premis_version", "event"}, path="payload", line_number=line_number)

    if payload.get("premis_version") != "3.0":
        raise ValueError(f"line {line_number}: premis_version must be '3.0'")

    event = _expect_object(payload.get("event"), path="event", line_number=line_number)
    _expect_exact_keys(
        event,
        required={
            "eventIdentifier",
            "eventType",
            "eventDateTime",
            "eventDetail",
            "eventOutcomeInformation",
            "linkingAgentIdentifier",
            "linkingObjectIdentifier",
        },
        path="event",
        line_number=line_number,
    )

    event_identifier = _expect_object(event.get("eventIdentifier"), path="event.eventIdentifier", line_number=line_number)
    _expect_exact_keys(
        event_identifier,
        required={"eventIdentifierType", "eventIdentifierValue"},
        path="event.eventIdentifier",
        line_number=line_number,
    )
    event_identifier_type = _expect_string(
        event_identifier.get("eventIdentifierType"),
        path="event.eventIdentifier.eventIdentifierType",
        line_number=line_number,
    )
    event_identifier_value = _expect_string(
        event_identifier.get("eventIdentifierValue"),
        path="event.eventIdentifier.eventIdentifierValue",
        line_number=line_number,
        min_length=1,
    )
    if event_identifier_type == "uuid":
        try:
            UUID(event_identifier_value)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: event.eventIdentifier.eventIdentifierValue must be a valid UUID") from exc

    _expect_string(event.get("eventType"), path="event.eventType", line_number=line_number, min_length=1)
    event_datetime = _expect_string(event.get("eventDateTime"), path="event.eventDateTime", line_number=line_number)
    _validate_rfc3339_timestamp(event_datetime, path="event.eventDateTime", line_number=line_number)
    _expect_string(event.get("eventDetail"), path="event.eventDetail", line_number=line_number)

    event_outcome_information = _expect_object(
        event.get("eventOutcomeInformation"),
        path="event.eventOutcomeInformation",
        line_number=line_number,
    )
    _expect_exact_keys(
        event_outcome_information,
        required={"eventOutcome"},
        path="event.eventOutcomeInformation",
        line_number=line_number,
    )
    event_outcome = _expect_string(
        event_outcome_information.get("eventOutcome"),
        path="event.eventOutcomeInformation.eventOutcome",
        line_number=line_number,
    )
    if event_outcome not in {"success", "failure"}:
        raise ValueError(f"line {line_number}: event.eventOutcomeInformation.eventOutcome must be one of: success, failure")

    linking_agents = _expect_array(
        event.get("linkingAgentIdentifier"), path="event.linkingAgentIdentifier", line_number=line_number
    )
    for index, agent in enumerate(linking_agents):
        agent_path = f"event.linkingAgentIdentifier[{index}]"
        agent_obj = _expect_object(agent, path=agent_path, line_number=line_number)
        _expect_exact_keys(
            agent_obj,
            required={"linkingAgentIdentifierType", "linkingAgentIdentifierValue"},
            path=agent_path,
            line_number=line_number,
        )
        _expect_string(
            agent_obj.get("linkingAgentIdentifierType"),
            path=f"{agent_path}.linkingAgentIdentifierType",
            line_number=line_number,
        )
        _expect_string(
            agent_obj.get("linkingAgentIdentifierValue"),
            path=f"{agent_path}.linkingAgentIdentifierValue",
            line_number=line_number,
        )

    linking_objects = _expect_array(
        event.get("linkingObjectIdentifier"),
        path="event.linkingObjectIdentifier",
        line_number=line_number,
    )
    for index, obj in enumerate(linking_objects):
        obj_path = f"event.linkingObjectIdentifier[{index}]"
        obj_record = _expect_object(obj, path=obj_path, line_number=line_number)
        _expect_exact_keys(
            obj_record,
            required={"linkingObjectIdentifierType", "linkingObjectIdentifierValue"},
            path=obj_path,
            line_number=line_number,
        )
        _expect_string(
            obj_record.get("linkingObjectIdentifierType"),
            path=f"{obj_path}.linkingObjectIdentifierType",
            line_number=line_number,
        )
        _expect_string(
            obj_record.get("linkingObjectIdentifierValue"),
            path=f"{obj_path}.linkingObjectIdentifierValue",
            line_number=line_number,
        )


def _validate_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number}: payload must be an object")
            _validate_event(payload, line_number=line_number)
            count += 1
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_emit = subparsers.add_parser("emit", help="Append one PREMIS event record")
    parser_emit.add_argument("--out-jsonl", required=True, help="Output PREMIS JSONL path")
    parser_emit.add_argument("--event-type", required=True, help="PREMIS event type")
    parser_emit.add_argument("--event-detail", required=True, help="Human-readable event detail")
    parser_emit.add_argument("--event-outcome", required=True, choices=["success", "failure"], help="Event outcome")
    parser_emit.add_argument("--agent-id", default="tp.archive.governance.v1", help="Software agent identifier")
    parser_emit.add_argument(
        "--object-id",
        action="append",
        default=[],
        help="Linked object identifier value (repeatable)",
    )
    parser_emit.add_argument("--event-datetime", default=None, help="Optional RFC3339 UTC timestamp")
    parser_emit.add_argument("--event-id", default=None, help="Optional UUID override")

    parser_validate = subparsers.add_parser("validate", help="Validate PREMIS JSONL records")
    parser_validate.add_argument("--input-jsonl", required=True, help="Input PREMIS JSONL path")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "emit":
        try:
            event = build_premis_event(
                event_type=args.event_type,
                event_detail=args.event_detail,
                event_outcome=args.event_outcome,
                agent_id=args.agent_id,
                object_ids=list(args.object_id),
                event_datetime=args.event_datetime,
                event_id=args.event_id,
            )
            append_event(Path(args.out_jsonl), event)
        except ValueError as exc:
            print(f"Input error: {exc}", file=sys.stderr)
            return EXIT_INPUT_ERROR
        print(f"Appended PREMIS event to {args.out_jsonl}")
        return EXIT_SUCCESS

    try:
        count = _validate_jsonl(Path(args.input_jsonl))
    except ValueError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return EXIT_VALIDATION_ERROR

    print(f"Validated {count} PREMIS event records")
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
