"""Tests for Phase 4B canonicalization config fingerprint governance."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "tools" / "capture_metadata_config.json"
TOOL_PATH = PROJECT_ROOT / "tools" / "capture_metadata_fingerprint.py"
GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "phase4" / "config_fingerprint.txt"

pytestmark = [pytest.mark.regression, pytest.mark.golden]


def _canonical_fingerprint(config_payload: dict[str, object]) -> str:
    canonical = json.dumps(
        config_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _run_tool(config_path: Path = CONFIG_PATH) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL_PATH), "--config", str(config_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_capture_metadata_config_fingerprint_is_reproducible() -> None:
    first = _run_tool()
    second = _run_tool()
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout.strip() == second.stdout.strip()
    assert len(first.stdout.strip()) == 64


def test_capture_metadata_config_fingerprint_matches_golden() -> None:
    result = _run_tool()
    assert result.returncode == 0, result.stderr
    expected = GOLDEN_PATH.read_text(encoding="utf-8").strip()
    assert result.stdout.strip() == expected


def test_capture_metadata_config_fingerprint_matches_canonical_algorithm() -> None:
    config_payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = _canonical_fingerprint(config_payload)
    result = _run_tool()
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_capture_metadata_config_structure_is_complete() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["metadata_contract_version"] == "tp.meta.capture.v1"
    assert isinstance(payload["tag_whitelist"], list) and payload["tag_whitelist"]
    assert payload["datetime_precedence"] == ["GPSDateStamp+GPSTimeStamp", "DateTimeOriginal+OffsetTimeOriginal"]
    assert payload["rounding_rules"]["rounding_mode"] == "bankers"
    assert len(payload["warning_codes"]) >= 4


def test_capture_metadata_config_fingerprint_ignores_key_order_and_whitespace(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    compact_path = tmp_path / "config_compact.json"
    compact_path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")

    reversed_top_level = {key: payload[key] for key in reversed(list(payload.keys()))}
    pretty_path = tmp_path / "config_pretty_reordered.json"
    pretty_path.write_text(json.dumps(reversed_top_level, indent=4, ensure_ascii=False), encoding="utf-8")

    compact_result = _run_tool(compact_path)
    pretty_result = _run_tool(pretty_path)

    assert compact_result.returncode == 0, compact_result.stderr
    assert pretty_result.returncode == 0, pretty_result.stderr
    assert compact_result.stdout.strip() == pretty_result.stdout.strip()
