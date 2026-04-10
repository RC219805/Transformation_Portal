"""Tests for the skill progression map collector."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from transformation_portal.dev import skill_progression_map as module

pytestmark = pytest.mark.unit


def test_resolve_since_falls_back_to_trailing_seven_days_when_memory_missing(tmp_path: Path) -> None:
    fixed_now = datetime(2026, 4, 10, 18, 0, 0, tzinfo=UTC)
    since, metadata = module.resolve_since(tmp_path / "missing-memory.md", now=fixed_now)

    assert since == fixed_now - timedelta(days=7)
    assert metadata["loaded"] is False
    assert metadata["timestamp_found"] is False
    assert metadata["last_run"] is None


def test_resolve_since_falls_back_when_memory_cannot_be_decoded(tmp_path: Path) -> None:
    fixed_now = datetime(2026, 4, 10, 18, 0, 0, tzinfo=UTC)
    memory_path = tmp_path / "memory.md"
    memory_path.write_bytes(b"\xff\xfe\x00")

    since, metadata = module.resolve_since(memory_path, now=fixed_now)

    assert since == fixed_now - timedelta(days=7)
    assert metadata["loaded"] is False
    assert metadata["status"] == "read_error"
    assert "UnicodeDecodeError" in str(metadata["notes"])


def test_parse_memory_timestamps_handles_utc_suffix_correctly() -> None:
    """Timestamp headings with 'UTC' suffix must be interpreted as UTC, not local time."""

    memory_text = """\
# Skill Progression Run

## 2026-04-10 14:30:00 UTC

Some notes from the run.

## 2026-04-09 10:00:00

Earlier local run.
"""
    timestamps = module.parse_memory_timestamps(memory_text)

    assert len(timestamps) == 2

    # Find timestamp by naive value (date + time) to isolate which is which
    ts_apr10 = next(
        (ts for ts in timestamps if ts.year == 2026 and ts.month == 4 and ts.day == 10),
        None,
    )
    ts_apr9 = next(
        (ts for ts in timestamps if ts.year == 2026 and ts.month == 4 and ts.day == 9),
        None,
    )

    assert ts_apr10 is not None, "Expected 2026-04-10 timestamp"
    assert ts_apr9 is not None, "Expected 2026-04-09 timestamp"

    # The UTC-suffixed timestamp must have exactly UTC timezone
    assert ts_apr10.tzinfo == UTC, f"Expected UTC for '2026-04-10 14:30:00 UTC', got {ts_apr10.tzinfo}"
    assert ts_apr10.hour == 14 and ts_apr10.minute == 30 and ts_apr10.second == 0

    # The unsuffixed timestamp must have LOCAL_TZ (which may be UTC in CI)
    assert ts_apr9.tzinfo == module.LOCAL_TZ, f"Expected LOCAL_TZ for unsuffixed timestamp, got {ts_apr9.tzinfo}"
    assert ts_apr9.hour == 10 and ts_apr9.minute == 0 and ts_apr9.second == 0


def test_parse_memory_timestamps_handles_common_us_timezone_suffixes() -> None:
    """Known timezone suffixes should use deterministic fixed offsets."""

    memory_text = """\
## 2026-04-10 09:00:00 PST

## 2026-04-10 10:00:00 PDT
    """
    timestamps = module.parse_memory_timestamps(memory_text)

    assert len(timestamps) == 2

    pst_timestamp = next((ts for ts in timestamps if ts.hour == 9), None)
    pdt_timestamp = next((ts for ts in timestamps if ts.hour == 10), None)

    assert pst_timestamp is not None
    assert pdt_timestamp is not None
    assert pst_timestamp.tzinfo == module.MEMORY_HEADING_TZINFOS["PST"]
    assert pdt_timestamp.tzinfo == module.MEMORY_HEADING_TZINFOS["PDT"]


def test_parse_memory_timestamps_treats_unknown_timezone_suffix_as_local() -> None:
    """Unknown timezone suffixes should still fall back to local time."""

    memory_text = """\
## 2026-04-10 09:00:00 FOO
"""
    timestamps = module.parse_memory_timestamps(memory_text)

    assert len(timestamps) == 1
    assert timestamps[0].tzinfo == module.LOCAL_TZ
    assert timestamps[0].hour == 9


def test_normalize_review_threads_tags_timeout_issue_for_raw_runtime() -> None:
    pr_summary = {
        "number": 1408,
        "title": "feat(raw): isolate RAW ingest runtime",
        "url": "https://github.com/RC219805/Transformation_Portal/pull/1408",
        "updated_at": "2026-04-10T17:22:23Z",
    }
    threads_payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "nodes": [
                            {
                                "isResolved": True,
                                "isOutdated": False,
                                "comments": {
                                    "nodes": [
                                        {
                                            "author": {"login": "copilot-pull-request-reviewer"},
                                            "body": (
                                                "Both the RAW runtime readiness check and the RAW worker "
                                                "invocation call subprocess.run without a timeout."
                                            ),
                                            "path": "src/transformation_portal/core/raw_runtime.py",
                                            "line": 209,
                                            "originalLine": 171,
                                            "createdAt": "2026-04-10T16:23:49Z",
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                }
            }
        }
    }

    records = module._normalize_review_threads(
        pr_summary=pr_summary,
        threads_payload=threads_payload,
        author_login="RC219805",
        now=datetime(2026, 4, 10, 18, 0, 0, tzinfo=UTC),
    )

    assert len(records) == 1
    assert records[0]["issue_class_tag"] == "timeout/runtime guard"
    assert records[0]["subsystem_tag"] == "raw-runtime"
    assert records[0]["theme_id"] == "ml_runtime_isolation_and_subprocess_contract_design"


def test_classify_issue_class_does_not_treat_block_as_lock() -> None:
    issue_class = module._classify_issue_class(
        "A stuck worker here would block CLI preflight and ingest indefinitely without a timeout.",
        "src/transformation_portal/core/raw_runtime.py",
    )

    assert issue_class == "timeout/runtime guard"


def test_safe_json_loads_returns_none_and_records_note_on_invalid_json() -> None:
    notes: list[str] = []

    result = module._safe_json_loads("not-json", notes=notes, context="test payload")

    assert result is None
    assert notes
    assert "test payload was not valid JSON" in notes[0]


def test_collect_gh_prs_fails_closed_on_invalid_pr_list_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(command: tuple[str, ...], *, cwd: Path, timeout: int) -> module.CommandResult:
        if command[:3] == ("gh", "auth", "status"):
            return module.CommandResult(
                command=tuple(command),
                returncode=0,
                stdout="github.com\n  ✓ Logged in to github.com account RC219805 (keyring)\n",
                stderr="",
            )
        if command[:3] == ("gh", "pr", "list"):
            return module.CommandResult(tuple(command), 0, "not-json", "")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(module, "_run_command", fake_run)

    report = module._collect_gh_prs(
        repo="RC219805/Transformation_Portal",
        author="RC219805",
        since=datetime(2026, 4, 3, 0, 0, 0, tzinfo=UTC),
        limit=5,
        repo_root=tmp_path,
        now=datetime(2026, 4, 10, 18, 0, 0, tzinfo=UTC),
    )

    assert report["success"] is False
    assert report["source_status"]["gh_cli"]["pr_list"] == "failed"
    assert any("not valid JSON" in note for note in report["source_status"]["gh_cli"]["notes"])


def test_collect_gh_prs_marks_degraded_when_review_threads_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], *, cwd: Path, timeout: int) -> module.CommandResult:
        commands.append(tuple(command))
        if command[:3] == ("gh", "auth", "status"):
            return module.CommandResult(
                command=tuple(command),
                returncode=0,
                stdout="github.com\n  ✓ Logged in to github.com account RC219805 (keyring)\n",
                stderr="",
            )
        if command[:3] == ("gh", "pr", "list"):
            payload = [
                {
                    "number": 1408,
                    "title": "feat(raw): isolate RAW ingest runtime",
                    "state": "MERGED",
                    "url": "https://github.com/RC219805/Transformation_Portal/pull/1408",
                    "updatedAt": "2026-04-10T17:22:23Z",
                    "mergedAt": "2026-04-10T17:22:23Z",
                    "isDraft": False,
                }
            ]
            return module.CommandResult(tuple(command), 0, json.dumps(payload), "")
        if command[:3] == ("gh", "pr", "view"):
            payload = {
                "number": 1408,
                "title": "feat(raw): isolate RAW ingest runtime",
                "state": "MERGED",
                "url": "https://github.com/RC219805/Transformation_Portal/pull/1408",
                "updatedAt": "2026-04-10T17:22:23Z",
                "mergedAt": "2026-04-10T17:22:23Z",
                "files": [
                    {
                        "path": "src/transformation_portal/core/raw_runtime.py",
                        "changeType": "MODIFIED",
                    }
                ],
                "reviews": [],
            }
            return module.CommandResult(tuple(command), 0, json.dumps(payload), "")
        if command[:3] == ("gh", "api", "graphql"):
            return module.CommandResult(tuple(command), 1, "", "GraphQL unavailable")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(module, "_run_command", fake_run)

    report = module._collect_gh_prs(
        repo="RC219805/Transformation_Portal",
        author="RC219805",
        since=datetime(2026, 4, 3, 0, 0, 0, tzinfo=UTC),
        limit=5,
        repo_root=tmp_path,
        now=datetime(2026, 4, 10, 18, 0, 0, tzinfo=UTC),
    )

    assert report["success"] is True
    assert report["source_status"]["degraded"] is True
    assert report["source_status"]["gh_cli"]["review_threads"] == "failed"
    assert report["source_status"]["evidence_quality"] == "medium"
    assert report["evidence_records"]
    assert report["evidence_records"][0]["source"] == "changed_file"
    assert any(command[:3] == ("gh", "api", "graphql") for command in commands)


def test_collect_gh_prs_marks_degraded_when_detail_fetch_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """PR detail (gh pr view) failure must degrade review_threads and evidence_quality."""

    def fake_run(command: tuple[str, ...], *, cwd: Path, timeout: int) -> module.CommandResult:
        if command[:3] == ("gh", "auth", "status"):
            return module.CommandResult(
                command=tuple(command),
                returncode=0,
                stdout="github.com\n  ✓ Logged in to github.com account RC219805 (keyring)\n",
                stderr="",
            )
        if command[:3] == ("gh", "pr", "list"):
            payload = [
                {
                    "number": 1409,
                    "title": "feat(portal): fingerprint assets",
                    "state": "MERGED",
                    "url": "https://github.com/RC219805/Transformation_Portal/pull/1409",
                    "updatedAt": "2026-04-10T17:22:23Z",
                    "mergedAt": "2026-04-10T17:22:23Z",
                    "isDraft": False,
                }
            ]
            return module.CommandResult(tuple(command), 0, json.dumps(payload), "")
        if command[:3] == ("gh", "pr", "view"):
            # Simulate a failed detail fetch
            return module.CommandResult(tuple(command), 1, "", "gh: PR not found")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(module, "_run_command", fake_run)

    report = module._collect_gh_prs(
        repo="RC219805/Transformation_Portal",
        author="RC219805",
        since=datetime(2026, 4, 3, 0, 0, 0, tzinfo=UTC),
        limit=5,
        repo_root=tmp_path,
        now=datetime(2026, 4, 10, 18, 0, 0, tzinfo=UTC),
    )

    assert report["success"] is True
    assert report["source_status"]["degraded"] is True
    # detail failure must count as a thread failure → not "ok"
    assert report["source_status"]["gh_cli"]["review_threads"] != "ok"
    # evidence_quality must not be "high" when all detail fetches failed
    assert report["source_status"]["evidence_quality"] != "high"
    assert any("Failed to inspect PR" in note for note in report["source_status"]["gh_cli"]["notes"])


def test_collect_local_git_fallback_matches_git_config_author_when_login_differs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], *, cwd: Path, timeout: int) -> module.CommandResult:
        commands.append(tuple(command))
        if command[:4] == ("git", "config", "--get", "user.name"):
            return module.CommandResult(tuple(command), 0, "Richard Cheetham\n", "")
        if command[:4] == ("git", "config", "--get", "user.email"):
            return module.CommandResult(tuple(command), 0, "richard@example.com\n", "")
        if command[:2] == ("git", "log"):
            assert "--author" not in command
            stdout = "\n".join(
                (
                    "abc12345\t2026-04-10T18:00:00+00:00\tRichard Cheetham\trichard@example.com\tfeat: matching commit",
                    "def67890\t2026-04-10T17:00:00+00:00\tSomeone Else\tother@example.com\tfeat: other commit",
                )
            )
            return module.CommandResult(tuple(command), 0, stdout, "")
        if command[:3] == ("git", "show", "--name-only"):
            return module.CommandResult(tuple(command), 0, "src/transformation_portal/dev/skill_progression_map.py\n", "")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(module, "_run_command", fake_run)

    report = module._collect_local_git_fallback(
        author="RC219805",
        since=datetime(2026, 4, 3, 0, 0, 0, tzinfo=UTC),
        limit=5,
        repo_root=tmp_path,
        now=datetime(2026, 4, 10, 18, 0, 0, tzinfo=UTC),
    )

    assert report["success"] is True
    assert len(report["fallback_commits"]) == 1
    assert report["fallback_commits"][0]["sha"] == "abc12345"


def test_rank_themes_prioritizes_recent_review_threads_over_changed_file_volume() -> None:
    now = datetime(2026, 4, 10, 18, 0, 0, tzinfo=UTC)
    recent_pr = {
        "number": 1408,
        "title": "feat(raw): isolate RAW ingest runtime",
        "url": "https://github.com/RC219805/Transformation_Portal/pull/1408",
        "updated_at": "2026-04-10T17:22:23Z",
    }
    older_pr = {
        "number": 1396,
        "title": "feat(portal): close out Phase 2B hierarchy and context polish",
        "url": "https://github.com/RC219805/Transformation_Portal/pull/1396",
        "updated_at": "2026-04-01T17:22:23Z",
    }

    evidence = [
        module._make_evidence_record(
            evidence_id="review-1",
            source="review_thread",
            pr_summary=recent_pr,
            path="src/transformation_portal/core/raw_runtime.py",
            line=100,
            status="resolved",
            summary="RAW worker invocation calls subprocess.run without a timeout.",
            issue_class="timeout/runtime guard",
            subsystem="raw-runtime",
            base_weight=module.REVIEW_SOURCE_WEIGHT,
            review_comment_count=2,
            now=now,
        ),
        module._make_evidence_record(
            evidence_id="review-2",
            source="review_thread",
            pr_summary=recent_pr,
            path="src/transformation_portal/core/raw_runtime.py",
            line=209,
            status="resolved",
            summary="Normalize input_path to an absolute path before dispatching to the worker.",
            issue_class="path normalization",
            subsystem="raw-runtime",
            base_weight=module.REVIEW_SOURCE_WEIGHT,
            review_comment_count=1,
            now=now,
        ),
    ]

    for index in range(6):
        evidence.append(
            module._make_evidence_record(
                evidence_id=f"file-{index}",
                source="changed_file",
                pr_summary=older_pr,
                path="portal.html",
                line=None,
                status="modified",
                summary="modified file in recurring portal surface.",
                issue_class="contract parity",
                subsystem="portal",
                base_weight=module.CHANGED_FILE_WEIGHT,
                now=now,
            )
        )

    ranked = module.rank_themes(evidence)

    assert ranked[0]["theme_id"] == "ml_runtime_isolation_and_subprocess_contract_design"
    assert ranked[0]["score"] > ranked[1]["score"]


def test_build_report_marks_memory_not_read_when_since_explicit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.md"
    memory_path.write_text("## 2026-04-10 10:00:00 UTC\n", encoding="utf-8")

    def fake_collect_gh_prs(**_: object) -> dict[str, object]:
        return {
            "success": True,
            "inspected_prs": [],
            "evidence_records": [],
            "source_status": {
                "connector": "not-run-by-helper",
                "gh_cli": {"auth": "ok", "pr_list": "ok", "review_threads": "no-prs", "notes": []},
                "local_git": {"used": False, "status": "not-used", "notes": []},
                "memory": {},
                "degraded": False,
                "evidence_quality": "low",
            },
        }

    monkeypatch.setattr(module, "_collect_gh_prs", fake_collect_gh_prs)

    report = module.build_skill_progression_report(
        repo="RC219805/Transformation_Portal",
        author="RC219805",
        since=datetime(2026, 4, 10, 18, 0, 0, tzinfo=UTC),
        limit=1,
        repo_root=tmp_path,
        memory_path=memory_path,
        now=datetime(2026, 4, 10, 18, 0, 0, tzinfo=UTC),
    )

    assert report["source_status"]["memory"]["exists"] is True
    assert report["source_status"]["memory"]["loaded"] is False
    assert report["source_status"]["memory"]["status"] == "not-read"


def test_main_json_output_contains_ranked_themes_and_source_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_report(**_: object) -> dict[str, object]:
        return {
            "report_version": "skill-progression-map.v1",
            "repo": "RC219805/Transformation_Portal",
            "author": "RC219805",
            "window": {
                "since": "2026-04-03T14:00:29Z",
                "until": "2026-04-10T19:00:00Z",
                "limit": 10,
            },
            "source_status": {
                "connector": "not-run-by-helper",
                "gh_cli": {"auth": "ok", "pr_list": "ok", "review_threads": "partial", "notes": []},
                "local_git": {"used": False, "status": "not-used", "notes": []},
                "memory": {"path": "/tmp/memory.md", "loaded": False, "timestamp_found": False, "last_run": None},
                "degraded": True,
                "evidence_quality": "medium",
            },
            "inspected_prs": [],
            "fallback_commits": [],
            "evidence_records": [],
            "ranked_themes": [
                {
                    "theme_id": "deterministic_validation_system_design",
                    "label": "Deterministic validation-system design",
                    "score": 6.5,
                    "evidence_count": 2,
                    "distinct_pr_count": 1,
                    "review_thread_count": 1,
                    "issue_class_tags": ["deterministic validation/preflight"],
                    "subsystem_tags": ["validation"],
                    "top_evidence": [],
                }
            ],
            "top_skills": [
                {
                    "theme_id": "deterministic_validation_system_design",
                    "label": "Deterministic validation-system design",
                    "score": 6.5,
                    "evidence_count": 2,
                    "distinct_pr_count": 1,
                    "review_thread_count": 1,
                    "issue_class_tags": ["deterministic validation/preflight"],
                    "subsystem_tags": ["validation"],
                    "top_evidence": [],
                }
            ],
        }

    monkeypatch.setattr(module, "build_skill_progression_report", fake_report)

    exit_code = module.main(["--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["top_skills"][0]["label"] == "Deterministic validation-system design"
    assert payload["source_status"]["gh_cli"]["review_threads"] == "partial"
