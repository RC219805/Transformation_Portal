"""Pin the local gitleaks pre-commit hook against CI parity.

PR #1508 was needed because gitleaks in CI flagged secret-shaped test fixtures
that no local check caught. This contract test keeps ``.pre-commit-config.yaml``
and ``.github/workflows/ci-quality-firewall.yml`` in agreement so the same
verdict is produced locally and in CI.

The test intentionally reads the YAML configs directly (no subprocess) so it
is cheap and runs in Layer 1.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pytest
import yaml

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
CI_FIREWALL_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-quality-firewall.yml"

GITLEAKS_REPO_URL = "https://github.com/gitleaks/gitleaks"

# CI in ci-quality-firewall.yml invokes:
#     gitleaks detect --config .gitleaks.toml --source . --verbose --no-git --exit-code 1
# The required (flag, value) pairs below are the tokenized form that the local
# hook must replicate exactly; the pre-commit framework splits multi-word entries
# on whitespace, so `--config=.gitleaks.toml` would not produce argv-identical
# behavior.
REQUIRED_FLAG_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("--config", ".gitleaks.toml"),
    ("--source", "."),
    ("--exit-code", "1"),
)
REQUIRED_SINGLE_FLAGS: Tuple[str, ...] = ("--no-git",)


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


def _hook_args(hook: dict) -> List[str]:
    raw = hook.get("args") or []
    return [str(entry) for entry in raw]


def _ci_gitleaks_pin() -> str:
    assert CI_FIREWALL_WORKFLOW.exists(), f"missing CI workflow at {CI_FIREWALL_WORKFLOW}"
    body = CI_FIREWALL_WORKFLOW.read_text(encoding="utf-8")
    # Search for the pinned archive name, e.g. gitleaks_8.21.2_linux_x64.tar.gz.
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


def test_default_install_hook_types_installs_pre_push() -> None:
    # Declaring `pre-push` in the hook's `stages` is necessary but not
    # sufficient: `pre-commit install -f` (invoked by `make install-hooks`)
    # only installs the hook types listed under `default_install_hook_types`.
    # Without this, `git commit --no-verify && git push` would bypass the
    # gitleaks check entirely.
    config = _load_pre_commit_config()
    defaults = config.get("default_install_hook_types") or []
    assert isinstance(defaults, list), "default_install_hook_types must be a list"
    installed_types = {str(entry) for entry in defaults}
    assert "pre-commit" in installed_types, "default_install_hook_types must include pre-commit"
    assert "pre-push" in installed_types, (
        "default_install_hook_types must include pre-push so `pre-commit install` "
        "actually wires up the push-time gitleaks gate."
    )


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
    args = _hook_args(hook)
    # The hook must invoke the `detect` subcommand (full-tree scan) rather
    # than the upstream default of `protect --staged`.
    assert args and args[0] == "detect", (
        "gitleaks hook must invoke the `detect` subcommand (not the default "
        "`protect --staged`) so a full-tree scan matches CI."
    )
    # CI tokenizes flags and values as separate argv entries. Assert each
    # required pair appears as adjacent tokens so the local hook is
    # argv-identical to CI (no `--foo=bar` drift).
    for flag, value in REQUIRED_FLAG_PAIRS:
        assert flag in args, f"gitleaks hook args missing required flag: {flag}"
        flag_index = args.index(flag)
        assert flag_index + 1 < len(args) and args[flag_index + 1] == value, (
            f"gitleaks hook must pass `{flag} {value}` as two separate tokens " f"(CI parity); got args={args!r}"
        )
    for flag in REQUIRED_SINGLE_FLAGS:
        assert flag in args, f"gitleaks hook args missing required flag: {flag}"
    # Reject the merged `=` forms explicitly so a future editor cannot
    # silently regress parity.
    for arg in args:
        assert "=" not in arg or not arg.startswith("--"), (
            f"gitleaks hook arg {arg!r} uses merged `--flag=value` form; " "CI tokenizes flags and values separately."
        )
