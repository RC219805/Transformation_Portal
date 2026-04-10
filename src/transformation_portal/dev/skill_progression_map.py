"""Collect and rank recent PR/review evidence for skill progression mapping.

This module is designed for automation use. It prefers GitHub PR/review
evidence via the local ``gh`` CLI and falls back to local git history only when
GitHub evidence cannot be collected.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from transformation_portal.ingest.canonical_json import dumps_json

AUTOMATION_ID = "skill-progression-map"
DEFAULT_LIMIT = 10
FALLBACK_WINDOW_DAYS = 7
GITHUB_COMMAND_TIMEOUT_SECONDS = 30
GRAPHQL_TIMEOUT_SECONDS = 45
LOCAL_GIT_TIMEOUT_SECONDS = 15
TOP_EVIDENCE_PER_THEME = 2
MAX_GH_PR_FETCH = 50
MAX_LOCAL_COMMITS = 50

LOCAL_TZ = datetime.now().astimezone().tzinfo or UTC
GITHUB_NOISE_AUTHORS = frozenset({"github-actions", "chatgpt-codex-connector"})

REVIEW_SOURCE_WEIGHT = 3.0
REVIEW_SUMMARY_WEIGHT = 2.0
CHANGED_FILE_WEIGHT = 0.6
LOCAL_COMMIT_WEIGHT = 0.5


@dataclass(frozen=True)
class CommandResult:
    """Structured subprocess result for deterministic command handling."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str | None = None


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: int,
) -> CommandResult:
    """Run a command and capture structured output without raising."""

    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=tuple(command),
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            timed_out=True,
            error=f"Timed out after {timeout}s",
        )
    except OSError as exc:
        return CommandResult(
            command=tuple(command),
            returncode=127,
            stdout="",
            stderr="",
            error=str(exc),
        )


def _codex_home() -> Path:
    """Return Codex home, falling back to ~/.codex when env is unset."""

    configured = os.environ.get("CODEX_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".codex").resolve()


def default_memory_path(automation_id: str = AUTOMATION_ID) -> Path:
    """Return the default memory path for the automation."""

    return _codex_home() / "automations" / automation_id / "memory.md"


def _parse_iso8601(timestamp: str) -> datetime | None:
    value = timestamp.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.astimezone(UTC)


def parse_memory_timestamps(memory_text: str) -> list[datetime]:
    """Extract timestamps from automation memory."""

    timestamps: list[datetime] = []

    for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\b", memory_text):
        parsed = _parse_iso8601(match.group(0))
        if parsed is not None:
            timestamps.append(parsed)

    for match in re.finditer(
        r"^##\s+(?P<stamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})(?:\s+[A-Z]{2,4})?$",
        memory_text,
        flags=re.MULTILINE,
    ):
        try:
            parsed = datetime.strptime(match.group("stamp"), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        timestamps.append(parsed.replace(tzinfo=LOCAL_TZ))

    return sorted(timestamps)


def resolve_since(
    memory_path: Path | None = None,
    *,
    now: datetime | None = None,
) -> tuple[datetime, dict[str, Any]]:
    """Resolve the default analysis window start from memory or fallback."""

    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    resolved_memory_path = memory_path or default_memory_path()
    memory_meta: dict[str, Any] = {
        "path": str(resolved_memory_path),
        "loaded": False,
        "timestamp_found": False,
        "last_run": None,
    }

    if resolved_memory_path.exists():
        content = resolved_memory_path.read_text(encoding="utf-8")
        memory_meta["loaded"] = True
        parsed_timestamps = parse_memory_timestamps(content)
        if parsed_timestamps:
            latest = parsed_timestamps[-1].astimezone(UTC)
            memory_meta["timestamp_found"] = True
            memory_meta["last_run"] = latest.isoformat().replace("+00:00", "Z")
            return latest, memory_meta

    fallback = current_time - timedelta(days=FALLBACK_WINDOW_DAYS)
    return fallback, memory_meta


def _normalize_slug(remote_url: str) -> str | None:
    match = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+)(?:\.git)?$", remote_url.strip())
    if not match:
        return None
    return f"{match.group('owner')}/{match.group('repo')}"


def resolve_repo_slug(repo_root: Path) -> str:
    """Resolve owner/repo from the current git checkout."""

    for command in (
        ("git", "remote", "get-url", "origin"),
        ("git", "config", "--get", "remote.origin.url"),
    ):
        result = _run_command(command, cwd=repo_root, timeout=5)
        if result.returncode != 0 or not result.stdout.strip():
            continue
        slug = _normalize_slug(result.stdout.strip())
        if slug:
            return slug
    raise RuntimeError("Could not resolve GitHub repository from git remote.origin.url")


def parse_github_login(auth_status_output: str) -> str | None:
    """Extract the active GitHub login from `gh auth status` output."""

    match = re.search(r"Logged in to github\.com account (?P<login>[A-Za-z0-9-]+)", auth_status_output)
    if match:
        return match.group("login")
    return None


def resolve_github_login(repo_root: Path) -> str:
    """Resolve the active GitHub login via the local gh CLI."""

    result = _run_command(("gh", "auth", "status"), cwd=repo_root, timeout=10)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.error or "gh auth status failed"
        raise RuntimeError(detail)
    login = parse_github_login(result.stdout)
    if not login:
        raise RuntimeError("Could not parse active GitHub login from `gh auth status` output")
    return login


def _safe_json_loads(text: str) -> Any:
    payload = text.strip()
    if not payload:
        return None
    return json.loads(payload)


def _normalize_datetime_string(value: str | None) -> str | None:
    if not value:
        return None
    parsed = _parse_iso8601(value)
    if parsed is None:
        return None
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _summary_from_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", re.sub(r"```.*?```", " ", text, flags=re.DOTALL)).strip()
    if not collapsed:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", collapsed, maxsplit=1)[0].strip()
    if len(sentence) > 220:
        return sentence[:217].rstrip() + "..."
    return sentence


def _classify_subsystem(path: str | None) -> str:
    if not path:
        return "repository"

    normalized = path.replace("\\", "/")
    if normalized.startswith("web/secure-landing/"):
        return "frontdoor"
    if normalized in {"app.py", "portal.html"} or normalized.startswith("public/portal-assets/"):
        return "portal"
    if normalized.startswith("scripts/validation/") or normalized.startswith("tests/validation/"):
        return "validation"
    if any(token in normalized for token in ("raw_runtime", "raw_worker", "raw_loader")):
        return "raw-runtime"
    if "artifact_store" in normalized or "orchestration/graph" in normalized:
        return "artifact-store"
    if "/ingest/" in normalized or normalized.startswith("tests/spatial_ai/ingest/"):
        return "ingest"
    if "/depth/backends/" in normalized or normalized.startswith("tests/test_da2_backend.py"):
        return "depth-runtime"
    if normalized.startswith(".github/workflows/") or normalized.startswith("docs/ci/"):
        return "github-workflows"
    if normalized.startswith("src/transformation_portal/"):
        parts = Path(normalized).parts
        if len(parts) >= 3:
            return parts[2].replace("_", "-")
    if normalized.startswith("docs/"):
        return "documentation"
    if normalized.startswith("tests/"):
        return "tests"
    return Path(normalized).parts[0]


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if re.fullmatch(r"[a-z0-9_]+", pattern):
            if re.search(rf"\b{re.escape(pattern)}\b", text):
                return True
            continue
        if pattern in text:
            return True
    return False


def _classify_issue_class(text: str, path: str | None) -> str:
    normalized_path = (path or "").replace("\\", "/").lower()
    normalized_text = f"{normalized_path} {text.lower()}"

    if _matches_any(
        normalized_text,
        (
            "atomic",
            "concurrent",
            "multiprocess",
            "partial write",
            "partial writes",
            "race",
            "lock",
            "fsync",
            "rename",
            "transactional visibility",
            "reader never sees partial",
        ),
    ):
        return "atomicity/concurrency"

    if _matches_any(
        normalized_text,
        (
            "fail closed",
            "fail-closed",
            "reject",
            "required",
            "missing_access",
            "missing_backend_api_key",
            "auth",
            "healthz",
            "access_config",
            "session_store",
            "session_scaling",
            "deployment gate",
            "protected deployment",
        ),
    ):
        return "fail-closed behavior"

    if _matches_any(
        normalized_text,
        (
            "timeout",
            "timed out",
            "hang",
            "stuck",
            "watchdog",
            "indefinite",
            "subprocess.run without a timeout",
        ),
    ):
        return "timeout/runtime guard"

    if _matches_any(
        normalized_text,
        (
            "relative path",
            "absolute path",
            "resolved path",
            "cwd differs",
            "cwd",
            "normalize input_path",
            "resolve it in the worker",
            "path normalization",
        ),
    ):
        return "path normalization"

    if _matches_any(
        normalized_text,
        (
            "lazy import",
            "torch-free",
            "optional dependency",
            "optional dependencies",
            "subprocess runtime",
            "isolated runtime",
            "openmp",
            "rawpy",
            "transformers",
            "import boundary",
            "eager torch import",
        ),
    ):
        return "optional-dependency/runtime isolation"

    if _matches_any(
        normalized_text,
        (
            "preflight",
            "validation",
            "deterministic",
            "smoke",
            "environment",
            "node version",
            "readiness",
            "preview",
            "ci gate",
            "gate",
            "check_local_environment",
        ),
    ):
        return "deterministic validation/preflight"

    if _matches_any(
        normalized_text,
        (
            "contract",
            "payload",
            "envelope",
            "selector",
            "dom marker",
            "route shape",
            "review status",
            "compare state",
            "surface",
            "schema",
        ),
    ):
        return "contract parity"

    if normalized_path.startswith("scripts/validation/") or normalized_path.startswith("tests/validation/"):
        return "deterministic validation/preflight"
    if normalized_path.startswith("web/secure-landing/") and "healthz" in normalized_path:
        return "fail-closed behavior"
    if "artifact_store" in normalized_path:
        return "atomicity/concurrency"
    if any(token in normalized_path for token in ("raw_runtime", "raw_worker", "depth/backends", "raw_loader")):
        return "optional-dependency/runtime isolation"
    if normalized_path in {"app.py", "portal.html"} or normalized_path.startswith("public/portal-assets/"):
        return "contract parity"

    return "general review pressure"


def _theme_catalog(theme_id: str, issue_class: str, subsystem: str) -> tuple[str, str]:
    if issue_class == "atomicity/concurrency":
        return theme_id, "Concurrency-safe storage and atomic commit semantics"

    if subsystem == "frontdoor" and issue_class in {
        "fail-closed behavior",
        "deterministic validation/preflight",
    }:
        return theme_id, "Fail-closed frontdoor operations"

    if issue_class == "contract parity" and subsystem in {"portal", "frontdoor"}:
        return theme_id, "Contract-driven portal/frontdoor state modeling"

    if issue_class == "deterministic validation/preflight":
        return theme_id, "Deterministic validation-system design"

    if issue_class in {
        "optional-dependency/runtime isolation",
        "timeout/runtime guard",
        "path normalization",
    } and subsystem in {"raw-runtime", "ingest", "depth-runtime"}:
        return theme_id, "ML/runtime isolation and subprocess contract design"

    subsystem_label = subsystem.replace("-", " ").title()
    issue_label = issue_class.replace("/", " / ").title()
    return theme_id, f"{issue_label} in {subsystem_label}"


def _theme_id(issue_class: str, subsystem: str) -> str:
    if issue_class == "atomicity/concurrency":
        return "concurrency_safe_storage"
    if subsystem == "frontdoor" and issue_class in {
        "fail-closed behavior",
        "deterministic validation/preflight",
    }:
        return "fail_closed_frontdoor_operations"
    if issue_class == "contract parity" and subsystem in {"portal", "frontdoor"}:
        return "contract_driven_portal_frontdoor_state_modeling"
    if issue_class == "deterministic validation/preflight":
        return "deterministic_validation_system_design"
    if issue_class in {
        "optional-dependency/runtime isolation",
        "timeout/runtime guard",
        "path normalization",
    } and subsystem in {"raw-runtime", "ingest", "depth-runtime"}:
        return "ml_runtime_isolation_and_subprocess_contract_design"
    normalized = re.sub(r"[^a-z0-9]+", "_", f"{issue_class}_{subsystem}".lower()).strip("_")
    return normalized or "general_review_pressure"


def _recency_multiplier(updated_at: str | None, *, now: datetime) -> float:
    normalized = _normalize_datetime_string(updated_at)
    if normalized is None:
        return 1.0
    parsed = _parse_iso8601(normalized)
    if parsed is None:
        return 1.0
    age = max((now - parsed.astimezone(UTC)).days, 0)
    if age <= 1:
        return 1.5
    if age <= 3:
        return 1.3
    if age <= 7:
        return 1.1
    if age <= 14:
        return 1.0
    return 0.8


def _make_pr_summary(pr_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": pr_payload["number"],
        "title": pr_payload["title"],
        "state": pr_payload.get("state", "UNKNOWN"),
        "url": pr_payload.get("url"),
        "updated_at": _normalize_datetime_string(pr_payload.get("updatedAt")),
        "merged_at": _normalize_datetime_string(pr_payload.get("mergedAt")),
        "review_thread_count": 0,
        "changed_file_count": 0,
    }


def _make_evidence_record(
    *,
    evidence_id: str,
    source: str,
    pr_summary: dict[str, Any] | None,
    path: str | None,
    line: int | None,
    status: str,
    summary: str,
    issue_class: str,
    subsystem: str,
    base_weight: float,
    review_comment_count: int = 1,
    now: datetime,
) -> dict[str, Any]:
    updated_at = pr_summary.get("updated_at") if pr_summary else None
    recency = _recency_multiplier(updated_at, now=now)
    intensity = 1.0 + max(review_comment_count - 1, 0) * 0.2
    score = round(base_weight * recency * intensity, 3)
    theme_id = _theme_id(issue_class, subsystem)
    _, theme_label = _theme_catalog(theme_id, issue_class, subsystem)
    return {
        "id": evidence_id,
        "theme_id": theme_id,
        "theme_label": theme_label,
        "source": source,
        "pr_number": pr_summary.get("number") if pr_summary else None,
        "pr_title": pr_summary.get("title") if pr_summary else None,
        "pr_url": pr_summary.get("url") if pr_summary else None,
        "updated_at": updated_at,
        "path": path,
        "line": line,
        "thread_status": status,
        "summary": summary,
        "subsystem_tag": subsystem,
        "issue_class_tag": issue_class,
        "review_comment_count": review_comment_count,
        "score": score,
    }


def _normalize_review_threads(
    *,
    pr_summary: dict[str, Any],
    threads_payload: dict[str, Any],
    author_login: str,
    now: datetime,
) -> list[dict[str, Any]]:
    nodes = (
        threads_payload.get("data", {}).get("repository", {}).get("pullRequest", {}).get("reviewThreads", {}).get("nodes", [])
    )
    evidence_records: list[dict[str, Any]] = []
    for index, thread in enumerate(nodes, start=1):
        comments = thread.get("comments", {}).get("nodes", [])
        actionable = [
            comment
            for comment in comments
            if comment.get("author", {}).get("login") not in GITHUB_NOISE_AUTHORS
            and comment.get("author", {}).get("login") != author_login
        ]
        if not actionable:
            continue

        primary = actionable[0]
        path = primary.get("path")
        summary_text = " ".join(comment.get("body", "") for comment in actionable[:2])
        summary = _summary_from_text(summary_text)
        subsystem = _classify_subsystem(path)
        issue_class = _classify_issue_class(summary_text, path)
        line = primary.get("line") or primary.get("originalLine")
        thread_status = "resolved" if thread.get("isResolved") else "open"
        if thread.get("isOutdated"):
            thread_status = "outdated" if thread_status != "open" else "open-outdated"
        evidence_records.append(
            _make_evidence_record(
                evidence_id=f"pr-{pr_summary['number']}-thread-{index}",
                source="review_thread",
                pr_summary=pr_summary,
                path=path,
                line=line,
                status=thread_status,
                summary=summary,
                issue_class=issue_class,
                subsystem=subsystem,
                base_weight=REVIEW_SOURCE_WEIGHT,
                review_comment_count=len(actionable),
                now=now,
            )
        )

    return evidence_records


def _normalize_review_summaries(
    *,
    pr_summary: dict[str, Any],
    reviews: list[dict[str, Any]],
    author_login: str,
    now: datetime,
) -> list[dict[str, Any]]:
    evidence_records: list[dict[str, Any]] = []
    review_index = 0
    for review in reviews:
        body = str(review.get("body") or "").strip()
        reviewer = review.get("author", {}).get("login")
        if not body or reviewer in GITHUB_NOISE_AUTHORS or reviewer == author_login:
            continue
        issue_class = _classify_issue_class(body, None)
        if issue_class == "general review pressure":
            continue
        review_index += 1
        evidence_records.append(
            _make_evidence_record(
                evidence_id=f"pr-{pr_summary['number']}-review-{review_index}",
                source="review_summary",
                pr_summary=pr_summary,
                path=None,
                line=None,
                status=str(review.get("state") or "commented").lower(),
                summary=_summary_from_text(body),
                issue_class=issue_class,
                subsystem="repository",
                base_weight=REVIEW_SUMMARY_WEIGHT,
                now=now,
            )
        )
    return evidence_records


def _normalize_changed_files(
    *,
    pr_summary: dict[str, Any],
    files: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    seen_keys: set[tuple[str, str]] = set()
    evidence_records: list[dict[str, Any]] = []
    for index, file_info in enumerate(files, start=1):
        path = file_info.get("path")
        subsystem = _classify_subsystem(path)
        issue_class = _classify_issue_class("", path)
        if issue_class == "general review pressure":
            continue
        dedupe_key = (subsystem, issue_class)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        change_type = str(file_info.get("changeType") or "MODIFIED").lower()
        summary = f"{change_type} file in recurring {subsystem} surface."
        evidence_records.append(
            _make_evidence_record(
                evidence_id=f"pr-{pr_summary['number']}-file-{index}",
                source="changed_file",
                pr_summary=pr_summary,
                path=path,
                line=None,
                status=change_type,
                summary=summary,
                issue_class=issue_class,
                subsystem=subsystem,
                base_weight=CHANGED_FILE_WEIGHT,
                now=now,
            )
        )
    return evidence_records


def _collect_gh_prs(
    *,
    repo: str,
    author: str,
    since: datetime,
    limit: int,
    repo_root: Path,
    now: datetime,
) -> dict[str, Any]:
    source_status = {
        "connector": "not-run-by-helper",
        "gh_cli": {
            "auth": "unknown",
            "pr_list": "unknown",
            "review_threads": "unknown",
            "notes": [],
        },
        "local_git": {"used": False, "status": "not-used", "notes": []},
        "memory": {},
        "degraded": False,
        "evidence_quality": "unknown",
    }

    auth_result = _run_command(("gh", "auth", "status"), cwd=repo_root, timeout=10)
    if auth_result.returncode != 0:
        source_status["gh_cli"]["auth"] = "failed"
        source_status["gh_cli"]["notes"].append(auth_result.stderr.strip() or auth_result.error or "gh auth status failed")
        source_status["degraded"] = True
        source_status["evidence_quality"] = "low"
        return {
            "success": False,
            "inspected_prs": [],
            "evidence_records": [],
            "source_status": source_status,
        }
    source_status["gh_cli"]["auth"] = "ok"

    fetch_count = min(max(limit * 4, limit), MAX_GH_PR_FETCH)
    list_command = (
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--author",
        author,
        "--state",
        "all",
        "--limit",
        str(fetch_count),
        "--json",
        "number,title,state,url,isDraft,updatedAt,mergedAt",
    )
    list_result = _run_command(list_command, cwd=repo_root, timeout=GITHUB_COMMAND_TIMEOUT_SECONDS)
    if list_result.returncode != 0:
        source_status["gh_cli"]["pr_list"] = "failed"
        source_status["gh_cli"]["notes"].append(list_result.stderr.strip() or list_result.error or "gh pr list failed")
        source_status["degraded"] = True
        source_status["evidence_quality"] = "low"
        return {
            "success": False,
            "inspected_prs": [],
            "evidence_records": [],
            "source_status": source_status,
        }
    source_status["gh_cli"]["pr_list"] = "ok"

    raw_prs = _safe_json_loads(list_result.stdout) or []
    since_utc = since.astimezone(UTC)
    filtered_prs = []
    for pr in raw_prs:
        updated_at = _parse_iso8601(str(pr.get("updatedAt") or ""))
        if updated_at is None or updated_at.astimezone(UTC) < since_utc:
            continue
        filtered_prs.append(pr)
    filtered_prs.sort(key=lambda item: str(item.get("updatedAt") or ""), reverse=True)
    filtered_prs = filtered_prs[:limit]

    inspected_prs: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    thread_failures = 0
    thread_successes = 0

    for pr in filtered_prs:
        number = int(pr["number"])
        summary = _make_pr_summary(pr)
        inspected_prs.append(summary)

        detail_command = (
            "gh",
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "number,title,state,url,updatedAt,mergedAt,files,reviews",
        )
        detail_result = _run_command(detail_command, cwd=repo_root, timeout=GITHUB_COMMAND_TIMEOUT_SECONDS)
        if detail_result.returncode != 0:
            source_status["gh_cli"]["notes"].append(
                f"Failed to inspect PR #{number}: {detail_result.stderr.strip() or detail_result.error or 'unknown error'}"
            )
            source_status["degraded"] = True
            thread_failures += 1
            continue

        detail_payload = _safe_json_loads(detail_result.stdout) or {}
        summary["changed_file_count"] = len(detail_payload.get("files", []))

        threads_payload: dict[str, Any] | None = None
        graphql_query = (
            "query($owner:String!, $repo:String!, $number:Int!){ "
            "repository(owner:$owner, name:$repo){ "
            "pullRequest(number:$number){ "
            "number title reviewThreads(first:50){ "
            "nodes { "
            "isResolved isOutdated "
            "comments(first:20){ nodes { author { login } body path line originalLine createdAt } } "
            "} } } } }"
        )
        owner, repo_name = repo.split("/", maxsplit=1)
        graphql_command = (
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={graphql_query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"repo={repo_name}",
            "-F",
            f"number={number}",
        )
        thread_result = _run_command(graphql_command, cwd=repo_root, timeout=GRAPHQL_TIMEOUT_SECONDS)
        if thread_result.returncode == 0:
            threads_payload = _safe_json_loads(thread_result.stdout) or {}
            thread_records = _normalize_review_threads(
                pr_summary=summary,
                threads_payload=threads_payload,
                author_login=author,
                now=now,
            )
            evidence_records.extend(thread_records)
            summary["review_thread_count"] = len(thread_records)
            thread_successes += 1
        else:
            thread_failures += 1
            source_status["degraded"] = True
            source_status["gh_cli"]["notes"].append(
                f"Failed to fetch review threads for PR #{number}: "
                f"{thread_result.stderr.strip() or thread_result.error or 'unknown error'}"
            )

        if summary["review_thread_count"] == 0:
            evidence_records.extend(
                _normalize_review_summaries(
                    pr_summary=summary,
                    reviews=detail_payload.get("reviews", []),
                    author_login=author,
                    now=now,
                )
            )

        evidence_records.extend(
            _normalize_changed_files(
                pr_summary=summary,
                files=detail_payload.get("files", []),
                now=now,
            )
        )

    if filtered_prs:
        if thread_failures == 0:
            source_status["gh_cli"]["review_threads"] = "ok"
            source_status["evidence_quality"] = "high"
        elif thread_successes > 0:
            source_status["gh_cli"]["review_threads"] = "partial"
            source_status["evidence_quality"] = "medium"
        else:
            source_status["gh_cli"]["review_threads"] = "failed"
            source_status["evidence_quality"] = "medium" if evidence_records else "low"
    else:
        source_status["gh_cli"]["review_threads"] = "no-prs"
        source_status["evidence_quality"] = "low"

    return {
        "success": True,
        "inspected_prs": inspected_prs,
        "evidence_records": evidence_records,
        "source_status": source_status,
    }


def _collect_local_git_fallback(
    *,
    author: str,
    since: datetime,
    limit: int,
    repo_root: Path,
    now: datetime,
) -> dict[str, Any]:
    source_status = {
        "connector": "not-run-by-helper",
        "gh_cli": {
            "auth": "unavailable",
            "pr_list": "unavailable",
            "review_threads": "unavailable",
            "notes": ["Using degraded local git fallback."],
        },
        "local_git": {"used": True, "status": "unknown", "notes": []},
        "memory": {},
        "degraded": True,
        "evidence_quality": "low",
    }

    log_command = (
        "git",
        "log",
        "--since",
        since.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "--author",
        author,
        "--pretty=format:%H%x09%aI%x09%s",
        f"--max-count={min(max(limit * 5, limit), MAX_LOCAL_COMMITS)}",
    )
    log_result = _run_command(log_command, cwd=repo_root, timeout=LOCAL_GIT_TIMEOUT_SECONDS)
    if log_result.returncode != 0:
        source_status["local_git"]["status"] = "failed"
        source_status["local_git"]["notes"].append(log_result.stderr.strip() or log_result.error or "git log failed")
        return {
            "success": False,
            "fallback_commits": [],
            "evidence_records": [],
            "source_status": source_status,
        }

    fallback_commits: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []

    for index, line in enumerate(filter(None, log_result.stdout.splitlines()), start=1):
        if index > limit:
            break
        try:
            commit_sha, authored_at, subject = line.split("\t", maxsplit=2)
        except ValueError:
            continue
        commit_payload = {
            "sha": commit_sha,
            "subject": subject,
            "updated_at": _normalize_datetime_string(authored_at),
        }
        fallback_commits.append(commit_payload)

        show_result = _run_command(
            ("git", "show", "--name-only", "--format=", commit_sha),
            cwd=repo_root,
            timeout=LOCAL_GIT_TIMEOUT_SECONDS,
        )
        if show_result.returncode != 0:
            source_status["local_git"]["notes"].append(
                f"Failed to inspect commit {commit_sha[:8]}: {show_result.stderr.strip() or show_result.error or 'unknown error'}"
            )
            continue

        seen_keys: set[tuple[str, str]] = set()
        for file_index, path in enumerate(filter(None, (entry.strip() for entry in show_result.stdout.splitlines())), start=1):
            subsystem = _classify_subsystem(path)
            issue_class = _classify_issue_class(subject, path)
            if issue_class == "general review pressure":
                continue
            dedupe_key = (subsystem, issue_class)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            pr_summary = {
                "number": None,
                "title": subject,
                "url": None,
                "updated_at": commit_payload["updated_at"],
            }
            evidence_records.append(
                _make_evidence_record(
                    evidence_id=f"commit-{commit_sha[:8]}-file-{file_index}",
                    source="local_commit",
                    pr_summary=pr_summary,
                    path=path,
                    line=None,
                    status="local-commit",
                    summary=_summary_from_text(subject),
                    issue_class=issue_class,
                    subsystem=subsystem,
                    base_weight=LOCAL_COMMIT_WEIGHT,
                    now=now,
                )
            )

    source_status["local_git"]["status"] = "ok" if fallback_commits else "no-data"
    return {
        "success": bool(fallback_commits),
        "fallback_commits": fallback_commits,
        "evidence_records": evidence_records,
        "source_status": source_status,
    }


def rank_themes(evidence_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank synthesized skill themes from normalized evidence."""

    grouped: dict[str, dict[str, Any]] = {}
    for record in evidence_records:
        theme_id = record["theme_id"]
        grouped.setdefault(
            theme_id,
            {
                "theme_id": theme_id,
                "label": record["theme_label"],
                "score": 0.0,
                "evidence": [],
                "pr_numbers": set(),
                "issue_classes": set(),
                "subsystems": set(),
                "review_thread_count": 0,
            },
        )
        bucket = grouped[theme_id]
        bucket["score"] += float(record["score"])
        bucket["evidence"].append(record)
        if record.get("pr_number") is not None:
            bucket["pr_numbers"].add(record["pr_number"])
        bucket["issue_classes"].add(record["issue_class_tag"])
        bucket["subsystems"].add(record["subsystem_tag"])
        if record["source"] == "review_thread":
            bucket["review_thread_count"] += 1

    ranked: list[dict[str, Any]] = []
    for bucket in grouped.values():
        distinct_prs = len(bucket["pr_numbers"])
        review_thread_bonus = max(bucket["review_thread_count"] - 1, 0) * 1.0
        recurrence_bonus = max(distinct_prs - 1, 0) * 1.5
        final_score = round(bucket["score"] + recurrence_bonus + review_thread_bonus, 3)
        ordered_evidence = sorted(
            bucket["evidence"],
            key=lambda item: (float(item["score"]), item.get("updated_at") or ""),
            reverse=True,
        )
        ranked.append(
            {
                "theme_id": bucket["theme_id"],
                "label": bucket["label"],
                "score": final_score,
                "evidence_count": len(bucket["evidence"]),
                "distinct_pr_count": distinct_prs,
                "review_thread_count": bucket["review_thread_count"],
                "issue_class_tags": sorted(bucket["issue_classes"]),
                "subsystem_tags": sorted(bucket["subsystems"]),
                "top_evidence": ordered_evidence[:TOP_EVIDENCE_PER_THEME],
            }
        )

    ranked.sort(key=lambda item: (item["score"], item["review_thread_count"], item["distinct_pr_count"]), reverse=True)
    return ranked


def build_skill_progression_report(
    *,
    repo: str | None = None,
    author: str | None = None,
    since: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
    repo_root: Path | None = None,
    memory_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a deterministic skill-progression report."""

    if limit <= 0:
        raise ValueError("limit must be positive")

    resolved_repo_root = (repo_root or Path.cwd()).resolve()
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    resolved_repo = repo or resolve_repo_slug(resolved_repo_root)
    resolved_author = author or resolve_github_login(resolved_repo_root)
    resolved_since, memory_meta = (
        resolve_since(memory_path, now=current_time)
        if since is None
        else (
            since.astimezone(UTC),
            {
                "path": str(memory_path or default_memory_path()),
                "loaded": memory_path.exists() if memory_path else default_memory_path().exists(),
                "timestamp_found": False,
                "last_run": None,
            },
        )
    )

    gh_report = _collect_gh_prs(
        repo=resolved_repo,
        author=resolved_author,
        since=resolved_since,
        limit=limit,
        repo_root=resolved_repo_root,
        now=current_time,
    )
    gh_report["source_status"]["memory"] = memory_meta

    fallback_commits: list[dict[str, Any]] = []
    evidence_records = gh_report["evidence_records"]
    source_status = gh_report["source_status"]
    inspected_prs = gh_report["inspected_prs"]

    if not gh_report["success"]:
        fallback_report = _collect_local_git_fallback(
            author=resolved_author,
            since=resolved_since,
            limit=limit,
            repo_root=resolved_repo_root,
            now=current_time,
        )
        fallback_report["source_status"]["memory"] = memory_meta
        source_status = fallback_report["source_status"]
        evidence_records = fallback_report["evidence_records"]
        fallback_commits = fallback_report["fallback_commits"]
        inspected_prs = []

    ranked_themes = rank_themes(evidence_records)
    top_skills = ranked_themes[:5]

    return {
        "report_version": "skill-progression-map.v1",
        "repo": resolved_repo,
        "author": resolved_author,
        "window": {
            "since": resolved_since.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "until": current_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "limit": limit,
        },
        "source_status": source_status,
        "inspected_prs": inspected_prs,
        "fallback_commits": fallback_commits,
        "evidence_records": evidence_records,
        "ranked_themes": ranked_themes,
        "top_skills": top_skills,
    }


def render_text_report(report: dict[str, Any]) -> str:
    """Render a compact human-readable report."""

    lines = [
        f"Skill progression map for {report['author']} in {report['repo']}",
        f"Window: {report['window']['since']} -> {report['window']['until']}",
        f"Evidence quality: {report['source_status']['evidence_quality']}",
    ]
    if not report["top_skills"]:
        lines.append("No ranked themes found.")
        return "\n".join(lines)

    for index, theme in enumerate(report["top_skills"], start=1):
        lines.append(f"{index}. {theme['label']} (score {theme['score']:.2f})")
        for evidence in theme["top_evidence"]:
            anchor = f"PR #{evidence['pr_number']}" if evidence.get("pr_number") else evidence["source"]
            if evidence.get("path"):
                anchor = f"{anchor} {evidence['path']}"
            lines.append(f"   - {anchor}: {evidence['summary']}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="GitHub repository in owner/name format.")
    parser.add_argument("--author", help="GitHub login to analyze.")
    parser.add_argument(
        "--since",
        help="Analysis window start in ISO-8601 format (defaults to automation memory, then trailing 7 days).",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help=f"Maximum recent PRs to inspect (default: {DEFAULT_LIMIT})."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by automation and manual dry runs."""

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    since = None
    if args.since:
        since = _parse_iso8601(args.since)
        if since is None:
            parser.error("--since must be a valid ISO-8601 timestamp")

    try:
        report = build_skill_progression_report(
            repo=args.repo,
            author=args.author,
            since=since,
            limit=args.limit,
        )
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(dumps_json(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    else:
        print(render_text_report(report))

    if report["top_skills"]:
        return 0
    if report["source_status"]["evidence_quality"] == "low":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
