from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "summary.yml"
DIAGNOSTIC_MARKER = "<!-- ai-summarizer-diagnostic -->"


def _workflow_text() -> str:
    return SUMMARY_WORKFLOW_PATH.read_text(encoding="utf-8")


def _load_workflow() -> dict:
    return yaml.load(_workflow_text(), Loader=yaml.BaseLoader)


def _post_comment_step() -> dict:
    workflow = _load_workflow()
    steps = workflow["jobs"]["summarize"]["steps"]
    return next(step for step in steps if step.get("name") == "Post successful summary comment only")


def _run_post_comment_step(tmp_path: Path, response_body: str) -> tuple[subprocess.CompletedProcess[str], str]:
    response_file = tmp_path / "issue_summary.txt"
    response_file.write_text(response_body, encoding="utf-8")

    calls_file = tmp_path / "gh-calls.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh_path = bin_dir / "gh"
    gh_script = "\n".join(
        [
            "#!/usr/bin/env bash",
            'printf "%s\\n" "$*" >> "$GH_CALLS_FILE"',
            "exit 0",
            "",
        ]
    )
    gh_path.write_text(
        gh_script,
        encoding="utf-8",
    )
    gh_path.chmod(gh_path.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env.update(
        {
            "GH_CALLS_FILE": str(calls_file),
            "GH_TOKEN": "test-token",
            "ISSUE_NUMBER": "1947",
            "MARKER": DIAGNOSTIC_MARKER,
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "RESPONSE_FILE": str(response_file),
        }
    )

    result = subprocess.run(
        ["bash", "-c", _post_comment_step()["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = calls_file.read_text(encoding="utf-8") if calls_file.exists() else ""
    return result, calls


def test_summary_workflow_remains_advisory_and_timeout_bounded() -> None:
    workflow = _load_workflow()
    job = workflow["jobs"]["summarize"]
    run_step = next(step for step in job["steps"] if step.get("name") == "Run summarizer (safe Option B)")

    assert job["continue-on-error"] == "true"
    assert job["timeout-minutes"] == "10"
    assert run_step["timeout-minutes"] == "4"


def test_summary_workflow_suppresses_pr_1947_rate_limit_diagnostic_comment(tmp_path: Path) -> None:
    diagnostic_body = (
        "AI summarizer skipped after bounded retries due to OpenAI rate limiting.\n"
        "This is non-blocking and intentionally does not fail CI.\n"
        f"{DIAGNOSTIC_MARKER}\n"
    )

    result, gh_calls = _run_post_comment_step(tmp_path, diagnostic_body)

    assert result.returncode == 0, result.stderr
    assert gh_calls == ""
    assert "Diagnostic fallback response is log-only; skipping PR comment post." in result.stdout
    assert "Posting comment to issue/PR #1947" not in result.stdout


def test_summary_workflow_still_posts_successful_ai_summary(tmp_path: Path) -> None:
    result, gh_calls = _run_post_comment_step(tmp_path, "This PR updates workflow guidance.\n")

    assert result.returncode == 0, result.stderr
    assert "Posting comment to issue/PR #1947" in result.stdout
    assert gh_calls.startswith("issue comment 1947 --body-file ")
