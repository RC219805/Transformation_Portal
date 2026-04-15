#!/usr/bin/env python3
"""Summarize repo-owned portal modernization evidence from JSONL telemetry sinks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

RUM_SCHEMA = "tp.orchestrator.portal_rum.v1"
EVENT_SCHEMA = "tp.orchestrator.portal_event.v1"
PASS = "pass"
FAIL = "fail"
INSUFFICIENT = "insufficient_data"


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (percentile / 100.0) * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def _read_records(path: Path, schema: str) -> List[Dict[str, Any]]:
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
                    f"portal modernization evidence: skipped invalid json line {line_number} in {path}: {exc.msg}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(payload, dict):
                print(
                    f"portal modernization evidence: skipped non-object json line {line_number} in {path}",
                    file=sys.stderr,
                )
                continue
            if payload.get("schema") != schema:
                continue
            records.append(payload)
    return records


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


def _metric_result(
    name: str,
    value: Optional[float],
    *,
    target_lte: Optional[float] = None,
    target_lt: Optional[float] = None,
    target_gte: Optional[float] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"name": name, "value": value}
    if target_lte is not None:
        result["target_lte"] = target_lte
    if target_lt is not None:
        result["target_lt"] = target_lt
    if target_gte is not None:
        result["target_gte"] = target_gte
    if value is None:
        result["status"] = INSUFFICIENT
        return result
    if target_lte is not None:
        result["status"] = PASS if value <= target_lte else FAIL
        return result
    if target_lt is not None:
        result["status"] = PASS if value < target_lt else FAIL
        return result
    if target_gte is not None:
        result["status"] = PASS if value >= target_gte else FAIL
        return result
    result["status"] = PASS
    return result


def _visibility_result(name: str, value: Optional[float]) -> Dict[str, Any]:
    return _metric_result(name, value)


def _overall_status(results: Iterable[Dict[str, Any]]) -> str:
    statuses = [str(result.get("status") or INSUFFICIENT) for result in results]
    if not statuses:
        return INSUFFICIENT
    if any(status == FAIL for status in statuses):
        return FAIL
    if any(status == INSUFFICIENT for status in statuses):
        return INSUFFICIENT
    return PASS


def build_report(
    rum_records: List[Dict[str, Any]],
    event_records: List[Dict[str, Any]],
    *,
    operator_hours: Optional[float],
) -> Dict[str, Any]:
    lcp_p75 = _percentile(_values_for(rum_records, event_type="core_web_vital", metric="lcp"), 75)
    inp_p75 = _percentile(_values_for(rum_records, event_type="core_web_vital", metric="inp"), 75)
    cls_p75 = _percentile(_values_for(rum_records, event_type="core_web_vital", metric="cls"), 75)
    bootstrap_ready_p75 = _percentile(_values_for(rum_records, event_type="bootstrap_ready"), 75)
    first_view_interactive_p75 = _percentile(_values_for(rum_records, event_type="first_view_interactive"), 75)
    portal_shell_rendered_p75 = _percentile(_values_for(rum_records, event_type="portal_shell_rendered"), 75)
    queue_submit_p75 = _percentile(_values_for(rum_records, event_type="queue_request", metric="submit"), 75)
    queue_cancel_p75 = _percentile(_values_for(rum_records, event_type="queue_request", metric="cancel"), 75)
    sse_reconnect_count = len(_values_for(rum_records, event_type="sse_reconnect"))

    metric_results = {
        "lcp_p75_ms": _metric_result("lcp_p75_ms", lcp_p75, target_lte=2500.0),
        "inp_p75_ms": _metric_result("inp_p75_ms", inp_p75, target_lte=200.0),
        "cls_p75": _metric_result("cls_p75", cls_p75, target_lte=0.1),
        "bootstrap_ready_p75_ms": _metric_result("bootstrap_ready_p75_ms", bootstrap_ready_p75),
        "first_view_interactive_p75_ms": _metric_result("first_view_interactive_p75_ms", first_view_interactive_p75),
        "portal_shell_rendered_p75_ms": _metric_result("portal_shell_rendered_p75_ms", portal_shell_rendered_p75),
        "queue_submit_p75_ms": _metric_result("queue_submit_p75_ms", queue_submit_p75, target_lte=150.0),
        "queue_cancel_p75_ms": _metric_result("queue_cancel_p75_ms", queue_cancel_p75, target_lte=150.0),
        "sse_reconnect_count": _metric_result("sse_reconnect_count", float(sse_reconnect_count)),
    }
    if operator_hours is None or operator_hours <= 0:
        metric_results["sse_reconnect_rate_per_operator_hour"] = _metric_result(
            "sse_reconnect_rate_per_operator_hour",
            None,
            target_lt=1.0,
        )
    else:
        metric_results["sse_reconnect_rate_per_operator_hour"] = _metric_result(
            "sse_reconnect_rate_per_operator_hour",
            sse_reconnect_count / operator_hours,
            target_lt=1.0,
        )

    viewer_open_count = 0
    viewer_fallback_count = 0
    fallback_reasons: Counter[str] = Counter()
    for record in event_records:
        if str(record.get("surface") or "") != "artifact_review":
            continue
        event_type = str(record.get("event_type") or "")
        metadata = record.get("metadata") or {}
        if event_type == "artifact_viewer_opened":
            viewer_open_count += 1
        elif event_type == "artifact_viewer_fallback":
            viewer_fallback_count += 1
            fallback_reason = str(metadata.get("fallback_reason") or "").strip().lower()
            if fallback_reason:
                fallback_reasons[fallback_reason] += 1
    viewer_success_rate = None
    if viewer_open_count > 0:
        viewer_success_rate = max(viewer_open_count - viewer_fallback_count, 0) / viewer_open_count * 100.0

    m1_inputs = [
        _visibility_result("lcp_p75_visible", lcp_p75),
        _visibility_result("inp_p75_visible", inp_p75),
        _visibility_result("cls_p75_visible", cls_p75),
        _visibility_result("bootstrap_ready_p75_visible", bootstrap_ready_p75),
        _visibility_result("first_view_interactive_p75_visible", first_view_interactive_p75),
        _visibility_result("portal_shell_rendered_p75_visible", portal_shell_rendered_p75),
        _visibility_result("sse_reconnect_count_visible", float(sse_reconnect_count)),
    ]
    if queue_submit_p75 is not None:
        m1_inputs.append(_visibility_result("queue_submit_p75_visible", queue_submit_p75))
    elif queue_cancel_p75 is not None:
        m1_inputs.append(_visibility_result("queue_cancel_p75_visible", queue_cancel_p75))
    else:
        m1_inputs.append(_visibility_result("queue_interaction_visibility", None))

    m1_status = _overall_status(m1_inputs)

    m4_inputs = [
        metric_results["lcp_p75_ms"],
        metric_results["inp_p75_ms"],
        metric_results["cls_p75"],
    ]
    if queue_submit_p75 is not None:
        m4_inputs.append(metric_results["queue_submit_p75_ms"])
    if queue_cancel_p75 is not None:
        m4_inputs.append(metric_results["queue_cancel_p75_ms"])
    if len(m4_inputs) == 3:
        m4_inputs.append(_metric_result("queue_interaction_thresholds", None))

    m5_result = _metric_result("viewer_success_rate_pct", viewer_success_rate, target_gte=95.0)
    if viewer_open_count == 0:
        m5_result["status"] = INSUFFICIENT

    return {
        "inputs": {
            "rum_samples": len(rum_records),
            "event_samples": len(event_records),
            "operator_hours": operator_hours,
        },
        "metrics": metric_results,
        "milestones": {
            "m1_measurement_foundation": {
                "status": m1_status,
                "rum_visibility_confirmed": m1_status != INSUFFICIENT,
            },
            "m4_performance": {
                "status": _overall_status(m4_inputs),
            },
            "m5_artifact_review": {
                "status": m5_result["status"],
                "viewer_open_count": viewer_open_count,
                "viewer_fallback_count": viewer_fallback_count,
                "viewer_success_rate_pct": viewer_success_rate,
                "fallback_reasons": dict(sorted(fallback_reasons.items())),
                "target_gte": 95.0,
            },
        },
    }


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _text_report(report: Dict[str, Any]) -> List[str]:
    lines = [
        (
            "inputs "
            f"rum_samples={report['inputs']['rum_samples']} "
            f"event_samples={report['inputs']['event_samples']} "
            f"operator_hours={_format_metric_value(report['inputs']['operator_hours'])}"
        )
    ]
    lines.append("m1_measurement_foundation " f"status={report['milestones']['m1_measurement_foundation']['status']}")
    for metric_name in (
        "lcp_p75_ms",
        "inp_p75_ms",
        "cls_p75",
        "bootstrap_ready_p75_ms",
        "first_view_interactive_p75_ms",
        "portal_shell_rendered_p75_ms",
        "queue_submit_p75_ms",
        "queue_cancel_p75_ms",
        "sse_reconnect_count",
        "sse_reconnect_rate_per_operator_hour",
    ):
        metric = report["metrics"][metric_name]
        lines.append(f"metric name={metric_name} status={metric['status']} value={_format_metric_value(metric['value'])}")
    lines.append(f"m4_performance status={report['milestones']['m4_performance']['status']}")
    m5 = report["milestones"]["m5_artifact_review"]
    lines.append(
        "m5_artifact_review "
        f"status={m5['status']} "
        f"viewer_open_count={m5['viewer_open_count']} "
        f"viewer_fallback_count={m5['viewer_fallback_count']} "
        f"viewer_success_rate_pct={_format_metric_value(m5['viewer_success_rate_pct'])}"
    )
    for reason, count in sorted(m5["fallback_reasons"].items()):
        lines.append(f"m5_fallback_reason reason={reason} count={count}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize repo-owned portal modernization evidence.")
    parser.add_argument("--rum-log", required=True, help="Path to the portal RUM JSONL log file.")
    parser.add_argument("--event-log", help="Optional path to the portal event JSONL log file.")
    parser.add_argument(
        "--operator-hours",
        type=float,
        help="Optional operator-hour denominator for SSE reconnect rate evaluation.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    args = parser.parse_args()

    rum_records = _read_records(Path(args.rum_log), RUM_SCHEMA)
    event_records = _read_records(Path(args.event_log), EVENT_SCHEMA) if args.event_log else []
    report = build_report(rum_records, event_records, operator_hours=args.operator_hours)

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    for line in _text_report(report):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
