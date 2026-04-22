"""Pin the local gitleaks pre-commit hook against CI parity.

PR #1508 was needed because gitleaks in CI flagged secret-shaped test fixtures
that no local check caught. This contract test keeps ``.pre-commit-config.yaml``
and ``.github/workflows/ci-quality-firewall.yml`` in agreement so the same
verdict is produced locally and in CI.

The test intentionally reads the YAML configs directly (no subprocess) so it
is cheap and runs in Layer 1.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
CI_FIREWALL_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-quality-firewall.yml"

GITLEAKS_REPO_URL = "https://github.com/gitleaks/gitleaks"
REQUIRED_ARGS = ("--config=.gitleaks.toml", "--no-git", "--source=.")


def _load_pre_commit_config() -> dict:
    assert PRE_COMMIT_CONFIG.exists(), f"missing pre-commit config at {PRE_COMMIT_CONFIG}"
    return yaml.safe_load(PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))


def _find_gitleaks_repo(config: dict) -> dict:
    repos = config.get("repos") or []
    matches = [entry for entry in repos if str(entry.get("repo", "")).rstrip("/") == GITLEAKS_REPO_URL]
    assert matches, (
        f"No repo entry for {GITLEAKS_REPO_URL} in {PRE_COMMIT_CONFIG}; "
        "the local gitleaks hook is missing and secret-shaped fixtures will "
        "only be caught in CI (see PR #1508)."
    )
    assert len(matches) == 1, f"Expected exactly one gitleaks repo entry, found {len(matches)}"
    return matches[0]


def _find_gitleaks_hook(repo_entry: dict) -> dict:
    hooks = repo_entry.get("hooks") or []
    matches = [hook for hook in hooks if hook.get("id") == "gitleaks"]
    assert matches, "No hook with id=gitleaks inside the gitleaks repo entry"
    assert len(matches) == 1, f"Expected exactly one gitleaks hook, found {len(matches)}"
    return matches[0]


def _ci_gitleaks_pin() -> str:
    assert CI_FIREWALL_WORKFLOW.exists(), f"missing CI workflow at {CI_FIREWALL_WORKFLOW}"
    body = CI_FIREWALL_WORKFLOW.read_text(encoding="utf-8")
    # Search for the pinned archive name, e.g. gitleaks_8.21.2_linux_x64.tar.gz.
    import re

    match = re.search(r"gitleaks_(\d+\.\d+\.\d+)_linux_x64\.tar\.gz", body)
    assert match, "could not locate gitleaks archive pin in ci-quality-firewall.yml"
    return f"v{match.group(1)}"


def test_gitleaks_hook_registered() -> None:
    config = _load_pre_commit_config()
    repo_entry = _find_gitleaks_repo(config)
    hook = _find_gitleaks_hook(repo_entry)
    # Sanity: the hook must not pass filenames (it scans the full tree), and
    # must be wired into the commit + push lifecycle so `git push` cannot
    # bypass it.
    assert hook.get("pass_filenames") is False, "gitleaks hook must use pass_filenames: false"
    stages = set(hook.get("stages") or [])
    assert "pre-commit" in stages, "gitleaks hook must run at pre-commit stage"
    assert "pre-push" in stages, "gitleaks hook must also run at pre-push stage"


def test_gitleaks_hook_rev_matches_ci_pin() -> None:
    config = _load_pre_commit_config()
    repo_entry = _find_gitleaks_repo(config)
    rev = str(repo_entry.get("rev") or "")
    expected = _ci_gitleaks_pin()
    assert rev == expected, (
        f"gitleaks hook rev ({rev!r}) must match the CI archive pin ({expected!r}) "
        "so local and CI produce the same verdict."
    )


def test_gitleaks_hook_args_match_ci_invocation() -> None:
    config = _load_pre_commit_config()
    repo_entry = _find_gitleaks_repo(config)
    hook = _find_gitleaks_hook(repo_entry)
    # `entry` may carry the binary name; the actionable flags live in `args`.
    args = hook.get("args") or []
    assert args and args[0] == "detect", (
        "gitleaks hook must invoke the `detect` subcommand (not the default "
        "`protect --staged`) so a full-tree scan matches CI."
    )
    joined = " ".join(str(arg) for arg in args)
    for required in REQUIRED_ARGS:
        assert required in joined, f"gitleaks hook args missing required flag: {required}"
    assert "--exit-code=1" in joined, "gitleaks hook must enforce exit-code=1 like CI"
