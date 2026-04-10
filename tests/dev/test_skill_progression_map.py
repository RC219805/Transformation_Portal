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
