#!/usr/bin/env python3
"""Summarize portal RUM JSONL records by route, view, and cohort."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

RUM_SCHEMA = "tp.orchestrator.portal_rum.v1"


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty list")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (percentile / 100.0) * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    fraction = index - lower
    return lower_value + (upper_value - lower_value) * fraction


def _read_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"portal rum summary: skipped invalid json line {line_number}: {exc.msg}",
                    file=sys.stderr,
                )
                continue
            if payload.get("schema") != RUM_SCHEMA:
                continue
            records.append(payload)
    return records


def _group_records(records: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, str, str], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("auth_mode") or ""),
            str(record.get("route") or ""),
            str(record.get("view") or ""),
            str(record.get("cohort_bucket") or ""),
        )
        groups[key].append(record)
    return groups


def _values_for(records: Iterable[Dict[str, Any]], *, event_type: str, metric: str = "") -> List[float]:
    values: List[float] = []
    for record in records:
        if str(record.get("event_type") or "") != event_type:
            continue
        if metric and str(record.get("metric") or "") != metric:
            continue
        value = record.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def build_summary(records: Iterable[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    groups = _group_records(records)
    for key in sorted(groups):
        auth_mode, route, view, cohort_bucket = key
        group_records = groups[key]
        line_parts = [
            f"auth_mode={auth_mode or 'unknown'}",
            f"route={route or 'unknown'}",
            f"view={view or 'unknown'}",
            f"cohort={cohort_bucket or 'unknown'}",
            f"samples={len(group_records)}",
        ]

        metric_specs = (
            ("lcp_p75_ms", "core_web_vital", "lcp"),
            ("inp_p75_ms", "core_web_vital", "inp"),
            ("cls_p75", "core_web_vital", "cls"),
            ("bootstrap_ready_p75_ms", "bootstrap_ready", ""),
            ("first_view_interactive_p75_ms", "first_view_interactive", ""),
            ("portal_shell_rendered_p75_ms", "portal_shell_rendered", ""),
            ("queue_submit_p75_ms", "queue_request", "submit"),
            ("queue_cancel_p75_ms", "queue_request", "cancel"),
        )
        for label, event_type, metric in metric_specs:
            values = _values_for(group_records, event_type=event_type, metric=metric)
            if values:
                line_parts.append(f"{label}={_percentile(values, 75):.2f}")

        sse_reconnect_count = len(_values_for(group_records, event_type="sse_reconnect"))
        line_parts.append(f"sse_reconnect_count={sse_reconnect_count}")
        lines.append(" ".join(line_parts))

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize portal RUM JSONL logs.")
    parser.add_argument("--input", required=True, help="Path to the portal RUM JSONL log file.")
    args = parser.parse_args()

    records = _read_records(Path(args.input))
    if not records:
        print("portal rum summary: no records")
        return 0

    for line in build_summary(records):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
