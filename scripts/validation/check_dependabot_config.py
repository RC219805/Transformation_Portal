#!/usr/bin/env python3
"""Validate Dependabot configuration against repository governance policy."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPENDABOT_PATH = REPO_ROOT / ".github" / "dependabot.yml"

REQUIRED_UPDATES = {
    ("pip", "/"),
    ("github-actions", "/"),
    ("npm", "/"),
    ("npm", "/cloudflare/transformationportal-worker"),
    ("npm", "/web/secure-landing"),
}
REQUIRED_NPM_GROUPS = {
    ("npm", "/web/secure-landing"): "frontdoor-node",
}
REQUIRED_SCHEDULES = {
    ("pip", "/"): {"interval": "weekly", "day": "tuesday", "time": "10:00", "timezone": "Etc/UTC"},
    ("github-actions", "/"): {"interval": "weekly", "day": "tuesday", "time": "10:15", "timezone": "Etc/UTC"},
    ("npm", "/"): {"interval": "weekly", "day": "tuesday", "time": "10:30", "timezone": "Etc/UTC"},
    ("npm", "/web/secure-landing"): {"interval": "weekly", "day": "tuesday", "time": "10:45", "timezone": "Etc/UTC"},
    ("npm", "/cloudflare/transformationportal-worker"): {
        "interval": "weekly",
        "day": "tuesday",
        "time": "10:30",
        "timezone": "Etc/UTC",
    },
}
REQUIRED_TARGET_BRANCH = "main"
REQUIRED_LABELS = {"automated", "dependencies"}
REQUIRED_OPEN_PR_LIMIT = 5
REQUIRED_PIP_EXCLUDE_PATHS: set[str] = set()
REQUIRED_NPM_GROUP_PATTERNS = {"*", "@*/*"}
REQUIRED_NPM_GROUP_UPDATE_TYPES = {"minor", "patch"}
REQUIRED_MLX_GROUP_PATTERNS = {"mlx", "mlx-metal"}
REQUIRED_WRANGLER_GROUP_PATTERNS = {"wrangler", "@cloudflare/workers-types"}
ROOT_WORKER_NPM_PAIRS = {
    ("npm", "/"),
    ("npm", "/cloudflare/transformationportal-worker"),
}
FRONTDOOR_NPM_PAIR = ("npm", "/web/secure-landing")
CODEQL_ACTION_PATTERN = "github/codeql-action/*"
REQUIRED_PIP_IGNORES = {
    "redis": {"version-update:semver-major"},
    "transformers": {"version-update:semver-minor"},
}


def _load_config(text: str) -> dict[str, Any]:
    if yaml is None:
        raise ValueError("PyYAML not installed (pip install PyYAML)")
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("dependabot config must be a YAML mapping")
    return loaded


def validate_dependabot_config(text: str) -> list[str]:
    """Return Dependabot contract violations."""
    errors: list[str] = []

    try:
        config = _load_config(text)
    except ValueError as exc:
        return [str(exc)]

    version = config.get("version")
    if version != 2:
        errors.append("dependabot config must set version: 2")

    updates = config.get("updates")
    if not isinstance(updates, list):
        return errors + ["dependabot config must define an updates list"]

    seen_pairs: set[tuple[str, str]] = set()
    for index, entry in enumerate(updates):
        if not isinstance(entry, dict):
            errors.append(f"updates[{index}] must be a mapping")
            continue

        ecosystem = entry.get("package-ecosystem")
        if not isinstance(ecosystem, str) or not ecosystem:
            errors.append(f"updates[{index}] package-ecosystem must be a non-empty string")
            continue

        directory = entry.get("directory")
        directories = entry.get("directories")
        if directory is not None and directories is not None:
            errors.append(f"updates[{index}] must define only one of directory or directories")
            continue
        if directory is not None:
            if not isinstance(directory, str) or not directory:
                errors.append(f"updates[{index}] directory must be a non-empty string")
                continue
            directory_values = [directory]
        else:
            if (
                not isinstance(directories, list)
                or not directories
                or any(not isinstance(value, str) or not value for value in directories)
            ):
                errors.append(f"updates[{index}] directories must be a non-empty list of strings")
                continue
            directory_values = list(dict.fromkeys(directories))
            if len(directory_values) != len(directories):
                errors.append(f"updates[{index}] directories must not contain duplicates")

        entry_pairs = {(ecosystem, value) for value in directory_values}
        for pair in sorted(entry_pairs):
            if pair in seen_pairs:
                errors.append(f"dependabot config contains duplicate update target {pair!r}")
                continue
            seen_pairs.add(pair)
            if pair not in REQUIRED_UPDATES:
                errors.append(
                    "dependabot config contains unsupported update target "
                    f"{pair!r}; expected only {sorted(REQUIRED_UPDATES)!r}"
                )

        target = next(iter(entry_pairs)) if len(entry_pairs) == 1 else tuple(sorted(entry_pairs))
        if len(entry_pairs) > 1 and entry_pairs != ROOT_WORKER_NPM_PAIRS:
            errors.append(
                "dependabot multi-directory target must be the governed root/Worker npm pair "
                f"{sorted(ROOT_WORKER_NPM_PAIRS)!r}"
            )

        if FRONTDOOR_NPM_PAIR in entry_pairs:
            if "target-branch" in entry:
                errors.append("dependabot frontdoor update must omit target-branch so security-update grouping applies")
        elif entry.get("target-branch") != REQUIRED_TARGET_BRANCH:
            errors.append(f"dependabot update {target!r} must target branch {REQUIRED_TARGET_BRANCH!r}")

        if entry.get("open-pull-requests-limit") != REQUIRED_OPEN_PR_LIMIT:
            errors.append(f"dependabot update {target!r} must set open-pull-requests-limit to {REQUIRED_OPEN_PR_LIMIT}")

        labels = entry.get("labels")
        if not isinstance(labels, list):
            errors.append(f"dependabot update {target!r} must define labels as a list")
        else:
            normalized_labels = {value for value in labels if isinstance(value, str) and value}
            if normalized_labels != REQUIRED_LABELS:
                errors.append(f"dependabot update {target!r} must use labels {sorted(REQUIRED_LABELS)!r}")

        schedule = entry.get("schedule")
        if not isinstance(schedule, dict):
            errors.append(f"dependabot update {target!r} must define schedule as a mapping")
        else:
            for pair in sorted(entry_pairs):
                required_schedule = REQUIRED_SCHEDULES.get(pair)
                if required_schedule is None:
                    continue
                for key, expected_value in required_schedule.items():
                    if schedule.get(key) != expected_value:
                        errors.append(f"dependabot update {pair!r} must set schedule {key!r} to {expected_value!r}")

        groups = entry.get("groups")
        if ("pip", "/") in entry_pairs:
            exclude_paths = entry.get("exclude-paths")
            if REQUIRED_PIP_EXCLUDE_PATHS and not isinstance(exclude_paths, list):
                errors.append("dependabot update ('pip', '/') must define exclude-paths as a list")
            elif isinstance(exclude_paths, list):
                missing_excludes = REQUIRED_PIP_EXCLUDE_PATHS - {
                    value for value in exclude_paths if isinstance(value, str) and value
                }
                for missing in sorted(missing_excludes):
                    errors.append(f"dependabot update ('pip', '/') must exclude unsupported manifest {missing!r}")

            ignores = entry.get("ignore")
            if not isinstance(ignores, list):
                errors.append("dependabot update ('pip', '/') must define governed ignore rules")
            else:
                ignores_by_name = {
                    value.get("dependency-name"): value
                    for value in ignores
                    if isinstance(value, dict) and isinstance(value.get("dependency-name"), str)
                }
                for dependency_name, expected_types in REQUIRED_PIP_IGNORES.items():
                    ignore = ignores_by_name.get(dependency_name)
                    if not isinstance(ignore, dict):
                        errors.append(f"dependabot pip ignores must include {dependency_name!r}")
                        continue
                    update_types = ignore.get("update-types")
                    normalized_types = (
                        {value for value in update_types if isinstance(value, str) and value}
                        if isinstance(update_types, list)
                        else set()
                    )
                    if normalized_types != expected_types:
                        errors.append(
                            f"dependabot pip ignore {dependency_name!r} must use update-types " f"{sorted(expected_types)!r}"
                        )

            mlx_group = groups.get("mlx-runtime") if isinstance(groups, dict) else None
            if not isinstance(mlx_group, dict):
                errors.append("dependabot pip update must define group 'mlx-runtime'")
            else:
                if mlx_group.get("applies-to") != "version-updates":
                    errors.append("dependabot group 'mlx-runtime' must apply to version-updates")
                patterns = mlx_group.get("patterns")
                normalized_patterns = (
                    {value for value in patterns if isinstance(value, str) and value} if isinstance(patterns, list) else set()
                )
                if normalized_patterns != REQUIRED_MLX_GROUP_PATTERNS:
                    errors.append(
                        "dependabot group 'mlx-runtime' must atomically match " f"{sorted(REQUIRED_MLX_GROUP_PATTERNS)!r}"
                    )
                if "group-by" in mlx_group:
                    errors.append("dependabot group 'mlx-runtime' must omit group-by so MLX packages stay coupled")

        if ("github-actions", "/") in entry_pairs:
            codeql_group = groups.get("codeql-actions") if isinstance(groups, dict) else None
            if not isinstance(codeql_group, dict):
                errors.append("dependabot github-actions update must define group 'codeql-actions'")
            else:
                if codeql_group.get("applies-to") != "version-updates":
                    errors.append("dependabot group 'codeql-actions' must apply to version-updates")
                if codeql_group.get("patterns") != ["github/codeql-action/*"]:
                    errors.append("dependabot group 'codeql-actions' must atomically match github/codeql-action/*")

        if entry_pairs == ROOT_WORKER_NPM_PAIRS:
            wrangler_group = groups.get("wrangler-sync") if isinstance(groups, dict) else None
            if not isinstance(wrangler_group, dict):
                errors.append("dependabot root/Worker npm update must define group 'wrangler-sync'")
            else:
                if wrangler_group.get("applies-to") != "version-updates":
                    errors.append("dependabot group 'wrangler-sync' must apply to version-updates")
                patterns = wrangler_group.get("patterns")
                normalized_patterns = (
                    {value for value in patterns if isinstance(value, str) and value} if isinstance(patterns, list) else set()
                )
                if normalized_patterns != REQUIRED_WRANGLER_GROUP_PATTERNS:
                    errors.append(
                        "dependabot group 'wrangler-sync' must atomically match "
                        f"{sorted(REQUIRED_WRANGLER_GROUP_PATTERNS)!r}"
                    )
                if "group-by" in wrangler_group:
                    errors.append(
                        "dependabot group 'wrangler-sync' must omit group-by so Wrangler and Worker types stay coupled"
                    )

            worker_group = groups.get("worker-node-tooling") if isinstance(groups, dict) else None
            if not isinstance(worker_group, dict):
                errors.append("dependabot root/Worker npm update must define group 'worker-node-tooling'")
            else:
                if worker_group.get("applies-to") != "version-updates":
                    errors.append("dependabot group 'worker-node-tooling' must apply to version-updates")
                patterns = worker_group.get("patterns")
                normalized_patterns = (
                    {value for value in patterns if isinstance(value, str) and value} if isinstance(patterns, list) else set()
                )
                if not REQUIRED_NPM_GROUP_PATTERNS.issubset(normalized_patterns):
                    errors.append("dependabot group 'worker-node-tooling' must match npm dependencies")
                exclude_patterns = worker_group.get("exclude-patterns")
                normalized_excludes = (
                    {value for value in exclude_patterns if isinstance(value, str) and value}
                    if isinstance(exclude_patterns, list)
                    else set()
                )
                if not REQUIRED_WRANGLER_GROUP_PATTERNS.issubset(normalized_excludes):
                    errors.append("dependabot group 'worker-node-tooling' must exclude Wrangler and Worker types")
                update_types = worker_group.get("update-types")
                normalized_update_types = (
                    {value for value in update_types if isinstance(value, str) and value}
                    if isinstance(update_types, list)
                    else set()
                )
                if normalized_update_types != REQUIRED_NPM_GROUP_UPDATE_TYPES:
                    errors.append(
                        "dependabot group 'worker-node-tooling' must group only "
                        f"{sorted(REQUIRED_NPM_GROUP_UPDATE_TYPES)!r} updates"
                    )

        for pair in sorted(entry_pairs):
            required_group = REQUIRED_NPM_GROUPS.get(pair)
            if required_group is None:
                continue
            groups = entry.get("groups")
            if not isinstance(groups, dict):
                errors.append(f"dependabot update {pair!r} must define npm version-update groups")
                continue

            group = groups.get(required_group)
            if not isinstance(group, dict):
                errors.append(f"dependabot update {pair!r} must define npm group {required_group!r}")
                continue

            if group.get("applies-to") != "version-updates":
                errors.append(f"dependabot npm group {required_group!r} must apply to version-updates")

            patterns = group.get("patterns")
            if not isinstance(patterns, list):
                errors.append(f"dependabot npm group {required_group!r} must define patterns as a list")
            else:
                missing_patterns = REQUIRED_NPM_GROUP_PATTERNS - {
                    value for value in patterns if isinstance(value, str) and value
                }
                for missing in sorted(missing_patterns):
                    errors.append(f"dependabot npm group {required_group!r} must include pattern {missing!r}")

            update_types = group.get("update-types")
            if not isinstance(update_types, list):
                errors.append(f"dependabot npm group {required_group!r} must define update-types as a list")
            else:
                normalized_update_types = {value for value in update_types if isinstance(value, str) and value}
                if normalized_update_types != REQUIRED_NPM_GROUP_UPDATE_TYPES:
                    errors.append(
                        f"dependabot npm group {required_group!r} must group only "
                        f"{sorted(REQUIRED_NPM_GROUP_UPDATE_TYPES)!r} updates"
                    )

            if pair == ("npm", "/web/secure-landing"):
                security_group = groups.get("frontdoor-security")
                if not isinstance(security_group, dict):
                    errors.append("dependabot frontdoor update must define group 'frontdoor-security'")
                else:
                    if security_group.get("applies-to") != "security-updates":
                        errors.append("dependabot group 'frontdoor-security' must apply to security-updates")
                    security_patterns = security_group.get("patterns")
                    normalized_security_patterns = (
                        {value for value in security_patterns if isinstance(value, str) and value}
                        if isinstance(security_patterns, list)
                        else set()
                    )
                    if not REQUIRED_NPM_GROUP_PATTERNS.issubset(normalized_security_patterns):
                        errors.append("dependabot group 'frontdoor-security' must match npm dependencies")

    missing = REQUIRED_UPDATES - seen_pairs
    for pair in sorted(missing):
        errors.append(f"dependabot config is missing required update target {pair!r}")

    return errors


def validate_repository_references(text: str, repo_root: Path = REPO_ROOT) -> list[str]:
    """Reject declared exact update families that no longer exist in the repo."""
    if CODEQL_ACTION_PATTERN not in text:
        return []

    workflows_dir = repo_root / ".github" / "workflows"
    references: list[str] = []
    if workflows_dir.is_dir():
        workflow_paths = (*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml"))
        for workflow_path in sorted(workflow_paths):
            try:
                workflow_config = _load_workflow(workflow_path)
            except (OSError, ValueError):
                continue
            if _workflow_uses_codeql_action(workflow_config):
                references.append(workflow_path.relative_to(repo_root).as_posix())

    if references:
        return []
    return ["dependabot group 'codeql-actions' is stale: no github/codeql-action/* uses remain under .github/workflows"]


def _load_workflow(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ValueError("PyYAML not installed (pip install PyYAML)")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid workflow YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("workflow must be a YAML mapping")
    return loaded


def _workflow_uses_codeql_action(config: dict[str, Any]) -> bool:
    jobs = config.get("jobs")
    if not isinstance(jobs, dict):
        return False
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            uses = step.get("uses")
            if not isinstance(uses, str):
                continue
            action_name, separator, revision = uses.strip().partition("@")
            normalized_action_name = action_name.lower()
            if (
                separator
                and revision
                and normalized_action_name.startswith("github/codeql-action/")
                and normalized_action_name.removeprefix("github/codeql-action/")
            ):
                return True
    return False


def main() -> int:
    try:
        config_text = DEPENDABOT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: Dependabot config not found at {DEPENDABOT_PATH}", file=sys.stderr)
        return 1

    errors = validate_dependabot_config(config_text)
    if not errors:
        errors.extend(validate_repository_references(config_text))
    if errors:
        print("ERROR: Dependabot config contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("dependabot config contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
