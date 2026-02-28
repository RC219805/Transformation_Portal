"""Regression tests for deterministic archive governance hardening."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

MANIFEST_TOOL_PATH = PROJECT_ROOT / "tools" / "build_archive_manifest_v2.py"
RIGHTS_TOOL_PATH = PROJECT_ROOT / "tools" / "apply_rights_policy.py"
GOVERNANCE_TOOL_PATH = PROJECT_ROOT / "tools" / "archive_governance.py"
COMMON_TOOL_PATH = PROJECT_ROOT / "tools" / "archive_governance_common.py"
PREMIS_TOOL_PATH = PROJECT_ROOT / "tools" / "premis_events.py"


def _load_tool_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MANIFEST_TOOL = _load_tool_module(MANIFEST_TOOL_PATH, "tests_build_archive_manifest_v2_hardening")
RIGHTS_TOOL = _load_tool_module(RIGHTS_TOOL_PATH, "tests_apply_rights_policy_hardening")
GOVERNANCE_TOOL = _load_tool_module(GOVERNANCE_TOOL_PATH, "tests_archive_governance_hardening")
COMMON_TOOL = _load_tool_module(COMMON_TOOL_PATH, "tests_archive_governance_common_hardening")
PREMIS_TOOL = _load_tool_module(PREMIS_TOOL_PATH, "tests_premis_events_hardening")


def test_build_entry_normalizes_relpath_for_rights_and_provenance(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive_root"
    target_file = archive_root / "DriveA" / "Part1" / "alpha.txt"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("alpha", encoding="utf-8")

    entry = MANIFEST_TOOL._build_entry(
        archive_root=archive_root,
        hash_row={
            "origin_drive": "DriveA",
            "partition": "Part1",
            "relpath": "DriveA\\Part1\\.\\alpha.txt",
            "filesize_bytes": "5",
            "sha256": "abc123",
            "hash_status": "ok",
        },
        rights_map={"DriveA/Part1/alpha.txt": {"rights_flags": ["restricted"], "owner": "Legal"}},
        collection_id="COLL-1",
        default_owner="UNSPECIFIED",
    )

    assert entry["relpath"] == "DriveA/Part1/alpha.txt"
    assert entry["rights_flags"] == ["restricted"]
    assert entry["owner"] == "Legal"
    assert entry["mime"] == "text/plain"
    expected_seed = "\0".join(["DriveA", "Part1", "DriveA/Part1/alpha.txt"])
    assert entry["provenance_id"] == hashlib.sha256(expected_seed.encode("utf-8")).hexdigest()


def test_load_rights_normalizes_relpath_keys(tmp_path: Path) -> None:
    rights_path = tmp_path / "rights.jsonl"
    rights_path.write_text(
        '{"relpath":"DriveA\\\\Part1\\\\.\\\\alpha.txt","rights_flags":["restricted"],"owner":"Legal"}\n',
        encoding="utf-8",
    )

    rights = MANIFEST_TOOL._load_rights(rights_path)
    assert "DriveA/Part1/alpha.txt" in rights
    assert rights["DriveA/Part1/alpha.txt"]["rights_flags"] == ["restricted"]


def test_deterministic_mime_map_uses_repo_owned_rules() -> None:
    assert MANIFEST_TOOL._deterministic_mime("archive.tar.gz") == "application/gzip"
    assert MANIFEST_TOOL._deterministic_mime("capture.CR3") == "image/x-canon-cr3"
    assert MANIFEST_TOOL._deterministic_mime("unknown.custom") == "application/octet-stream"


def test_manifest_archive_index_validation_streams_rows(tmp_path: Path) -> None:
    archive_index = tmp_path / "archive_index_normalized.csv"
    archive_index.write_text(
        "origin_drive,partition,relpath\nDriveA,Part1,DriveA/Part1/alpha.txt\nDriveB,Part2,DriveB/Part2/beta.txt\n",
        encoding="utf-8",
    )
    assert MANIFEST_TOOL._consume_csv_rows(archive_index) == 2


def test_manifest_archive_index_validation_rejects_missing_required_columns(tmp_path: Path) -> None:
    archive_index = tmp_path / "archive_index_normalized.csv"
    archive_index.write_text(
        "origin_drive,partition\nDriveA,Part1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        MANIFEST_TOOL._consume_csv_rows(
            archive_index,
            required_columns={"origin_drive", "partition", "relpath"},
        )


def test_load_hash_rows_rejects_missing_required_columns_without_data_rows(tmp_path: Path) -> None:
    hash_manifest = tmp_path / "hash_manifest.csv"
    hash_manifest.write_text("origin_drive,partition,relpath\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        MANIFEST_TOOL._load_hash_rows(hash_manifest)


def test_manifest_build_streaming_does_not_publish_partial_output_on_build_error(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive_root"
    (archive_root / "DriveA" / "Part1").mkdir(parents=True, exist_ok=True)
    (archive_root / "DriveA" / "Part1" / "alpha.txt").write_text("alpha", encoding="utf-8")

    archive_index = tmp_path / "archive_index_normalized.csv"
    archive_index.write_text(
        "origin_drive,partition,relpath\n"
        "DriveA,Part1,DriveA/Part1/alpha.txt\n"
        "DriveA,Part1,DriveA/Part1/beta.txt\n",
        encoding="utf-8",
    )

    hash_manifest = tmp_path / "hash_manifest.csv"
    hash_manifest.write_text(
        "origin_drive,partition,relpath,filesize_bytes,sha256,hash_status,error\n"
        "DriveA,Part1,DriveA/Part1/alpha.txt,5,abc,ok,\n"
        "DriveA,Part1,DriveA/Part1/beta.txt,not-an-int,def,error,boom\n",
        encoding="utf-8",
    )

    out_jsonl = tmp_path / "archive_manifest_v2.jsonl"
    out_summary = tmp_path / "archive_manifest_v2.summary.json"
    exit_code = MANIFEST_TOOL.main(
        [
            "--archive-index",
            str(archive_index),
            "--hash-manifest",
            str(hash_manifest),
            "--archive-root",
            str(archive_root),
            "--out-jsonl",
            str(out_jsonl),
            "--out-summary",
            str(out_summary),
        ]
    )

    assert exit_code == MANIFEST_TOOL.EXIT_BUILD_ERROR
    assert not out_jsonl.exists()
    assert not out_summary.exists()


def test_rule_matches_normalizes_slashes_for_path_glob() -> None:
    entry = {"relpath": "DriveA\\Part1\\alpha.txt", "extension": ".txt"}
    rule = {"id": "r1", "flags": ["restricted"], "path_glob": "DriveA/Part1/*.txt"}
    assert RIGHTS_TOOL._rule_matches(entry, rule) is True


def test_rule_matches_keeps_case_sensitive_glob_semantics(monkeypatch) -> None:
    monkeypatch.setattr(RIGHTS_TOOL.fnmatch.os.path, "normcase", lambda value: value.lower())
    entry = {"relpath": "DriveA/Part1/ALPHA.TXT", "extension": ".txt"}
    rule = {"id": "r1", "flags": ["restricted"], "path_glob": "DriveA/Part1/*.txt"}
    assert RIGHTS_TOOL._rule_matches(entry, rule) is False


def test_load_policy_compiles_relpath_regex_once(tmp_path: Path) -> None:
    policy_yaml = tmp_path / "rights_flags.yml"
    policy_yaml.write_text(
        "version: 1\n"
        "default_flags: [unspecified]\n"
        "default_owner: UNSPECIFIED\n"
        "rules:\n"
        "  - id: regex-rule\n"
        "    flags: [restricted]\n"
        "    relpath_regex: '^DriveA/.+\\.txt$'\n",
        encoding="utf-8",
    )
    policy = RIGHTS_TOOL._load_policy(policy_yaml)
    rule = policy["rules"][0]

    assert "_relpath_regex_compiled" in rule
    assert RIGHTS_TOOL._rule_matches({"relpath": "DriveA/Part1/alpha.txt", "extension": ".txt"}, rule)


def test_load_policy_precomputes_extension_match_set(tmp_path: Path) -> None:
    policy_yaml = tmp_path / "rights_flags.yml"
    policy_yaml.write_text(
        "version: 1\n"
        "default_flags: [unspecified]\n"
        "default_owner: UNSPECIFIED\n"
        "rules:\n"
        "  - id: ext-rule\n"
        "    flags: [restricted]\n"
        "    extension_in: ['.TXT', '.csv']\n",
        encoding="utf-8",
    )
    policy = RIGHTS_TOOL._load_policy(policy_yaml)
    rule = policy["rules"][0]

    assert rule["_extension_in_set"] == {".txt", ".csv"}
    assert RIGHTS_TOOL._rule_matches({"relpath": "DriveA/Part1/alpha.txt", "extension": ".txt"}, rule)
    assert RIGHTS_TOOL._rule_matches({"relpath": "DriveA/Part1/alpha.jpg", "extension": ".jpg"}, rule) is False


def test_run_wrapped_tool_normalizes_negative_returncode(monkeypatch) -> None:
    def _fake_run_tool(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=command, returncode=-9, stdout="", stderr="killed")

    captured: dict[str, object] = {}

    def _fake_emit_result(*, args, command_name: str, exit_code: int, data, error=None) -> int:
        captured["command_name"] = command_name
        captured["exit_code"] = exit_code
        captured["error"] = error
        return exit_code

    monkeypatch.setattr(GOVERNANCE_TOOL, "_run_tool", _fake_run_tool)
    monkeypatch.setattr(GOVERNANCE_TOOL, "_emit_result", _fake_emit_result)
    monkeypatch.setattr(GOVERNANCE_TOOL, "_record_premis", lambda **_: None)

    args = SimpleNamespace(
        json=True,
        json_pretty=False,
        json_output=None,
        json_canonical_profile="canonical_v1",
        premis_log=None,
        premis_agent_id="tp.archive.governance.v1",
    )
    exit_code = GOVERNANCE_TOOL._run_wrapped_tool(
        args=args,
        command_name="manifest-build",
        script_name="build_archive_manifest_v2.py",
        tool_args=["--help"],
        premis_event_type=None,
        premis_event_detail="",
        premis_object_ids=[],
    )

    assert exit_code == GOVERNANCE_TOOL.EXIT_OTHER_FAILURE
    assert captured["exit_code"] == GOVERNANCE_TOOL.EXIT_OTHER_FAILURE
    assert "signal 9" in str(captured["error"]["message"])
    assert captured["error"]["exit_code"]["value"] == GOVERNANCE_TOOL.EXIT_OTHER_FAILURE


def test_jcs_profile_ignores_pretty_flag() -> None:
    payload = {"z": "last", "a": "first", "n": 1}
    compact = COMMON_TOOL.deterministic_json_dumps(
        payload,
        pretty=False,
        canonical_profile=COMMON_TOOL.CANONICAL_PROFILE_JCS,
    )
    pretty = COMMON_TOOL.deterministic_json_dumps(
        payload,
        pretty=True,
        canonical_profile=COMMON_TOOL.CANONICAL_PROFILE_JCS,
    )
    assert pretty == compact


def test_apply_rights_policy_streaming_does_not_publish_partial_output_on_invalid_input(tmp_path: Path) -> None:
    manifest_jsonl = tmp_path / "manifest.jsonl"
    manifest_jsonl.write_text(
        '{"relpath":"DriveA/Part1/alpha.txt","extension":".txt","owner":"owner-a"}\n'
        '{"relpath":"DriveA/Part1/beta.txt","extension":".txt","owner":"owner-b"\n',
        encoding="utf-8",
    )
    policy_yaml = tmp_path / "rights_flags.yml"
    policy_yaml.write_text(
        "version: 1\n" + "default_flags: [unspecified]\n" + "default_owner: UNSPECIFIED\n" + "rules: []\n",
        encoding="utf-8",
    )
    out_jsonl = tmp_path / "asset_rights.jsonl"
    out_summary = tmp_path / "asset_rights.summary.json"

    exit_code = RIGHTS_TOOL.main(
        [
            "--manifest-jsonl",
            str(manifest_jsonl),
            "--policy-yaml",
            str(policy_yaml),
            "--out-jsonl",
            str(out_jsonl),
            "--out-summary",
            str(out_summary),
        ]
    )
    assert exit_code == RIGHTS_TOOL.EXIT_INPUT_ERROR
    assert not out_jsonl.exists()
    assert not out_summary.exists()


def test_premis_validate_rejects_invalid_datetime_and_outcome() -> None:
    payload = PREMIS_TOOL.build_premis_event(
        event_type="validation",
        event_detail="event",
        event_outcome="success",
        agent_id="tp.archive.tests",
        object_ids=["/tmp/object"],
        event_datetime="2026-02-28T01:00:00Z",
        event_id="2b9d322b-0785-44a5-a59d-7da8af4f8a07",
    )
    PREMIS_TOOL._validate_event(payload, line_number=1)

    invalid_datetime = dict(payload)
    invalid_datetime["event"] = dict(payload["event"])
    invalid_datetime["event"]["eventDateTime"] = "2026-02-28 01:00:00"
    with pytest.raises(ValueError, match="RFC3339"):
        PREMIS_TOOL._validate_event(invalid_datetime, line_number=2)

    invalid_outcome = dict(payload)
    invalid_outcome["event"] = dict(payload["event"])
    invalid_outcome["event"]["eventOutcomeInformation"] = {"eventOutcome": "maybe"}
    with pytest.raises(ValueError, match="eventOutcome"):
        PREMIS_TOOL._validate_event(invalid_outcome, line_number=3)


def test_premis_validate_rejects_non_finite_json_constants(tmp_path: Path) -> None:
    invalid_jsonl = tmp_path / "premis_invalid.jsonl"
    invalid_jsonl.write_text(
        '{"premis_version":"3.0","event":{"eventIdentifier":{"eventIdentifierType":"uuid","eventIdentifierValue":"x"},"eventType":"fixity","eventDateTime":"2026-02-28T01:00:00Z","eventDetail":NaN,"eventOutcomeInformation":{"eventOutcome":"success"},"linkingAgentIdentifier":[{"linkingAgentIdentifierType":"software","linkingAgentIdentifierValue":"tp.archive.tests"}],"linkingObjectIdentifier":[]}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        PREMIS_TOOL._validate_jsonl(invalid_jsonl)


def test_run_wrapped_tool_reports_missing_script_with_typed_error(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_emit_result(*, args, command_name: str, exit_code: int, data, error=None) -> int:
        captured["command_name"] = command_name
        captured["exit_code"] = exit_code
        captured["data"] = data
        captured["error"] = error
        return exit_code

    monkeypatch.setattr(GOVERNANCE_TOOL, "_emit_result", _fake_emit_result)
    monkeypatch.setattr(GOVERNANCE_TOOL, "_record_premis", lambda **_: None)

    args = SimpleNamespace(
        json=True,
        json_pretty=False,
        json_output=None,
        json_canonical_profile="canonical_v1",
        premis_log=None,
        premis_agent_id="tp.archive.governance.v1",
    )
    exit_code = GOVERNANCE_TOOL._run_wrapped_tool(
        args=args,
        command_name="bag-build",
        script_name="archive_bagit.py",
        tool_args=["build"],
        premis_event_type=None,
        premis_event_detail="",
        premis_object_ids=[],
    )

    assert exit_code == GOVERNANCE_TOOL.EXIT_OTHER_FAILURE
    assert captured["exit_code"] == GOVERNANCE_TOOL.EXIT_OTHER_FAILURE
    assert captured["error"]["type"] == "ToolUnavailableError"
    assert "missing tool script" in str(captured["error"]["message"])
    assert "missing_tool" in captured["data"]


def test_fixity_scan_reports_missing_script_with_typed_error(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_emit_result(*, args, command_name: str, exit_code: int, data, error=None) -> int:
        captured["command_name"] = command_name
        captured["exit_code"] = exit_code
        captured["data"] = data
        captured["error"] = error
        return exit_code

    monkeypatch.setattr(GOVERNANCE_TOOL, "_emit_result", _fake_emit_result)
    monkeypatch.setattr(GOVERNANCE_TOOL, "_record_premis", lambda **_: None)
    monkeypatch.setattr(GOVERNANCE_TOOL, "PROJECT_ROOT", tmp_path)

    args = SimpleNamespace(
        json=True,
        json_pretty=False,
        json_output=None,
        json_canonical_profile="canonical_v1",
        premis_log=None,
        premis_agent_id="tp.archive.governance.v1",
        archive_index="index.csv.gz",
        archive_root="/archive",
        out_dir="archive_reports/fixity",
        workers=4,
        strict=False,
        strict_identity=False,
        validate_schemas=True,
    )
    exit_code = GOVERNANCE_TOOL._handle_fixity_scan(args)

    assert exit_code == GOVERNANCE_TOOL.EXIT_OTHER_FAILURE
    assert captured["command_name"] == "fixity-scan"
    assert captured["error"]["type"] == "ToolUnavailableError"
    assert "missing tool script" in str(captured["error"]["message"])
    assert "missing_tool" in captured["data"]


def test_fixity_verify_reports_missing_script_with_typed_error(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_emit_result(*, args, command_name: str, exit_code: int, data, error=None) -> int:
        captured["command_name"] = command_name
        captured["exit_code"] = exit_code
        captured["data"] = data
        captured["error"] = error
        return exit_code

    monkeypatch.setattr(GOVERNANCE_TOOL, "_emit_result", _fake_emit_result)
    monkeypatch.setattr(GOVERNANCE_TOOL, "_record_premis", lambda **_: None)
    monkeypatch.setattr(GOVERNANCE_TOOL, "PROJECT_ROOT", tmp_path)

    args = SimpleNamespace(
        json=True,
        json_pretty=False,
        json_output=None,
        json_canonical_profile="canonical_v1",
        premis_log=None,
        premis_agent_id="tp.archive.governance.v1",
        hash_manifest="archive_reports/fixity/hash_manifest.csv.gz",
        archive_root="/archive",
        report_path=None,
        workers=4,
        verify_sample=0,
    )
    exit_code = GOVERNANCE_TOOL._handle_fixity_verify(args)

    assert exit_code == GOVERNANCE_TOOL.EXIT_OTHER_FAILURE
    assert captured["command_name"] == "fixity-verify"
    assert captured["error"]["type"] == "ToolUnavailableError"
    assert "missing tool script" in str(captured["error"]["message"])
    assert "missing_tool" in captured["data"]


def test_sealed_eval_reports_missing_harness_with_typed_error(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_emit_result(*, args, command_name: str, exit_code: int, data, error=None) -> int:
        captured["command_name"] = command_name
        captured["exit_code"] = exit_code
        captured["data"] = data
        captured["error"] = error
        return exit_code

    monkeypatch.setattr(GOVERNANCE_TOOL, "_emit_result", _fake_emit_result)
    monkeypatch.setattr(GOVERNANCE_TOOL, "_record_premis", lambda **_: None)

    args = SimpleNamespace(
        json=True,
        json_pretty=False,
        json_output=None,
        json_canonical_profile="canonical_v1",
        premis_log=None,
        premis_agent_id="tp.archive.governance.v1",
        archive_index="index.csv.gz",
        archive_root="/archive",
        out_root="archive_reports/sealed_eval",
        subset_root=None,
        eval_command=None,
        validate_schemas=True,
        allow_writable_subset=False,
    )
    exit_code = GOVERNANCE_TOOL._handle_sealed_eval_run(args)

    assert exit_code == GOVERNANCE_TOOL.EXIT_OTHER_FAILURE
    assert captured["command_name"] == "sealed-eval-run"
    assert captured["error"]["type"] == "ToolUnavailableError"
    assert "missing tool script" in str(captured["error"]["message"])
    assert "missing_tool" in captured["data"]
