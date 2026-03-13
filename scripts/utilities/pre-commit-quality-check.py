#!/usr/bin/env python3
"""Unified pre-commit quality gate for this repository.

This script consolidates the strongest checks from the prior shell hook,
ad hoc Python runner, and generic pre-commit configuration into one
standalone file that can run directly from ``.git/hooks/pre-commit``.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except ImportError:  # pragma: no cover - packaging is normally available
    from pip._vendor.packaging.requirements import Requirement  # type: ignore[attr-defined]
    from pip._vendor.packaging.version import Version  # type: ignore[attr-defined]

try:
    import yaml
except ImportError:  # pragma: no cover - only needed when YAML files are staged
    yaml = None

MAX_LARGE_FILE_KB = 5000
ROOT_MARKDOWN_LIMIT = 10
CONFLICT_MARKER_RE = re.compile(r"^(<<<<<<< |=======|>>>>>>> )", re.MULTILINE)


@dataclass(frozen=True)
class CheckOutcome:
    label: str
    ok: bool
    blocking: bool = True


def find_repo_root() -> Path:
    result = run_command(["git", "rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        raise SystemExit("This script must run inside a git repository.")
    return Path(result.stdout.strip()).resolve()


def python_can_import(python_bin: str, modules: Sequence[str]) -> bool:
    result = run_command(
        [
            python_bin,
            "-c",
            (
                "import importlib.util, sys; "
                "missing=[name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]; "
                "sys.exit(1 if missing else 0)"
            ),
            *modules,
        ]
    )
    return result.returncode == 0


def choose_python(repo_root: Path) -> str:
    venv_python = repo_root / ".venv" / "bin" / "python"
    candidates = []
    if venv_python.exists():
        candidates.append(str(venv_python))
    candidates.append(sys.executable)

    for candidate in candidates:
        if python_can_import(candidate, ("black", "flake8", "isort")):
            return candidate
    return candidates[0]


def run_command(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(cwd) if cwd else None,
        capture_output=capture_output,
        text=text,
        check=False,
    )


def git_paths(repo_root: Path, *, all_files: bool, diff_filter: str = "ACMR") -> list[Path]:
    if all_files:
        result = run_command(["git", "ls-files"], cwd=repo_root)
    else:
        result = run_command(
            ["git", "diff", "--cached", "--name-only", f"--diff-filter={diff_filter}"],
            cwd=repo_root,
        )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return [repo_root / line for line in result.stdout.splitlines() if line.strip()]


def relpath(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root))


def is_text_file(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:8192]
    except OSError:
        return False
    return b"\0" not in sample


def remove_trailing_whitespace_and_fix_eof(text: str) -> tuple[str, bool]:
    if not text:
        return text, False

    changed = False
    normalized_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        newline = ""
        body = line
        if body.endswith("\r\n"):
            newline = "\r\n"
            body = body[:-2]
        elif body.endswith("\n"):
            newline = "\n"
            body = body[:-1]
        elif body.endswith("\r"):
            newline = "\r"
            body = body[:-1]

        stripped = body.rstrip(" \t\f\v")
        if stripped != body:
            changed = True
        normalized_lines.append(stripped + newline)

    normalized = "".join(normalized_lines)
    if normalized and not normalized.endswith(("\n", "\r")):
        normalized += "\n"
        changed = True

    return normalized, changed


def auto_fix_text_hygiene(repo_root: Path, paths: Iterable[Path]) -> CheckOutcome:
    modified: list[str] = []
    for path in paths:
        if not path.is_file() or not is_text_file(path):
            continue
        original = path.read_text(encoding="utf-8", errors="surrogateescape")
        normalized, changed = remove_trailing_whitespace_and_fix_eof(original)
        if not changed:
            continue
        path.write_text(normalized, encoding="utf-8", errors="surrogateescape")
        modified.append(relpath(repo_root, path))

    if modified:
        run_command(["git", "add", *modified], cwd=repo_root, capture_output=True)
        print("Auto-fixed trailing whitespace / EOF newline in:")
        for path in modified:
            print(f"  - {path}")

    return CheckOutcome("Text hygiene auto-fix", True)


def check_root_file_placement(repo_root: Path, *, all_files: bool) -> CheckOutcome:
    policy_script = repo_root / "scripts" / "setup" / "pre-commit-check.sh"
    if not policy_script.is_file():
        print(f"Root placement policy script not found: {policy_script}")
        return CheckOutcome("Root file placement", False)

    mode = "--all" if all_files else "--staged"
    result = run_command(["bash", str(policy_script), mode], cwd=repo_root)
    if result.returncode == 0:
        return CheckOutcome("Root file placement", True)

    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return CheckOutcome("Root file placement", False)


def check_untracked_core_files(repo_root: Path) -> CheckOutcome:
    result = run_command(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo_root)
    if result.returncode != 0:
        return CheckOutcome("Untracked core files", False)

    blocked = {
        "app.py",
        "portal.html",
        "tests/test_app_orchestrator_runtime.py",
    }
    found = sorted(set(result.stdout.splitlines()) & blocked)
    if not found:
        return CheckOutcome("Untracked core files", True)

    print("Core files are untracked and must be added explicitly before commit:")
    for path in found:
        print(f"  - {path}")
    return CheckOutcome("Untracked core files", False)


def check_markdown_count(repo_root: Path) -> CheckOutcome:
    markdown_count = len(list(repo_root.glob("*.md")))
    if markdown_count <= ROOT_MARKDOWN_LIMIT:
        return CheckOutcome("Root markdown count", True, blocking=False)

    print(
        f"Warning: {markdown_count} markdown files are present in the repo root " f"(recommended max: {ROOT_MARKDOWN_LIMIT})."
    )
    return CheckOutcome("Root markdown count", False, blocking=False)


def check_large_added_files(added_paths: Sequence[Path], repo_root: Path) -> CheckOutcome:
    offenders = []
    for path in added_paths:
        if path.is_file() and path.stat().st_size > MAX_LARGE_FILE_KB * 1024:
            offenders.append((relpath(repo_root, path), path.stat().st_size))

    if not offenders:
        return CheckOutcome("Added large files", True)

    print(f"Large added files exceed {MAX_LARGE_FILE_KB}KB:")
    for path, size in offenders:
        print(f"  - {path} ({size // 1024}KB)")
    return CheckOutcome("Added large files", False)


def check_merge_conflicts(text_paths: Sequence[Path], repo_root: Path) -> CheckOutcome:
    offenders = []
    for path in text_paths:
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="surrogateescape")
        if CONFLICT_MARKER_RE.search(content):
            offenders.append(relpath(repo_root, path))

    if not offenders:
        return CheckOutcome("Merge conflict markers", True)

    print("Merge conflict markers detected:")
    for path in offenders:
        print(f"  - {path}")
    return CheckOutcome("Merge conflict markers", False)


def check_line_endings(text_paths: Sequence[Path], repo_root: Path) -> CheckOutcome:
    offenders = []
    for path in text_paths:
        raw = path.read_bytes()
        has_crlf = b"\r\n" in raw
        has_bare_lf = re.search(rb"(?<!\r)\n", raw) is not None
        has_bare_cr = re.search(rb"\r(?!\n)", raw) is not None
        if (has_crlf and has_bare_lf) or has_bare_cr:
            offenders.append(relpath(repo_root, path))

    if not offenders:
        return CheckOutcome("Mixed line endings", True)

    print("Mixed or unsupported line endings detected:")
    for path in offenders:
        print(f"  - {path}")
    return CheckOutcome("Mixed line endings", False)


def check_yaml_syntax(yaml_paths: Sequence[Path], repo_root: Path) -> CheckOutcome:
    if not yaml_paths:
        return CheckOutcome("YAML syntax", True)
    if yaml is None:
        print("PyYAML is required to validate staged YAML files.")
        return CheckOutcome("YAML syntax", False)

    offenders = []
    for path in yaml_paths:
        try:
            content = path.read_text(encoding="utf-8", errors="surrogateescape")
            list(yaml.safe_load_all(content))
        except yaml.YAMLError as exc:
            offenders.append((relpath(repo_root, path), str(exc).splitlines()[0]))

    if not offenders:
        return CheckOutcome("YAML syntax", True)

    print("YAML syntax errors detected:")
    for path, error in offenders:
        print(f"  - {path}: {error}")
    return CheckOutcome("YAML syntax", False)


def parse_lint_requirements(requirements_path: Path) -> dict[str, Requirement]:
    requirements: dict[str, Requirement] = {}
    for raw in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        requirements[requirement.name.lower()] = requirement
    return requirements


def installed_package_versions(python_bin: str, package_names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in package_names:
        result = run_command(
            [
                python_bin,
                "-c",
                "from importlib.metadata import version; import sys; print(version(sys.argv[1]))",
                package_name,
            ]
        )
        if result.returncode == 0:
            versions[package_name.lower()] = result.stdout.strip()
    return versions


def check_lint_tool_parity(
    repo_root: Path,
    python_bin: str,
    staged_python: Sequence[Path],
    staged_yaml: Sequence[Path],
) -> CheckOutcome:
    if not staged_python and not staged_yaml:
        return CheckOutcome("Lint tool parity", True)

    requirements_path = repo_root / "requirements-lint.txt"
    if not requirements_path.exists():
        print("requirements-lint.txt not found.")
        return CheckOutcome("Lint tool parity", False)

    requirements = parse_lint_requirements(requirements_path)
    installed_versions = installed_package_versions(python_bin, ("black", "flake8", "isort"))
    errors = []
    for tool in ("black", "flake8", "isort"):
        requirement = requirements.get(tool)
        if requirement is None:
            errors.append(f"{tool}: not declared in requirements-lint.txt")
            continue
        actual_version = installed_versions.get(tool)
        if actual_version is None:
            errors.append(f"{tool}: not installed in {python_bin}")
            continue
        actual = Version(actual_version)
        if requirement.specifier and actual not in requirement.specifier:
            errors.append(f"{tool}: installed {actual}, expected {requirement.specifier}")

    if staged_yaml and not python_can_import(python_bin, ("yaml",)):
        errors.append(f"PyYAML: not installed in {python_bin}")

    if not errors:
        return CheckOutcome("Lint tool parity", True)

    print("Lint tool versions do not satisfy requirements-lint.txt:")
    for error in errors:
        print(f"  - {error}")
    return CheckOutcome("Lint tool parity", False)


def run_python_module(
    python_bin: str,
    module: str,
    args: Sequence[str],
    *,
    repo_root: Path,
) -> subprocess.CompletedProcess[str]:
    return run_command([python_bin, "-m", module, *args], cwd=repo_root)


def check_python_syntax_and_undefined_names(
    repo_root: Path,
    python_bin: str,
    staged_python: Sequence[Path],
) -> CheckOutcome:
    if not staged_python:
        return CheckOutcome("Python critical lint", True)

    rel_paths = [relpath(repo_root, path) for path in staged_python]
    result = run_python_module(
        python_bin,
        "flake8",
        ["--select=E9,F63,F7,F82,F821", *rel_paths],
        repo_root=repo_root,
    )
    if result.returncode == 0:
        return CheckOutcome("Python critical lint", True)

    print(result.stdout.strip())
    print(result.stderr.strip())
    return CheckOutcome("Python critical lint", False)


def check_black_and_isort(
    repo_root: Path,
    python_bin: str,
    staged_python: Sequence[Path],
) -> list[CheckOutcome]:
    if not staged_python:
        return [CheckOutcome("Black", True), CheckOutcome("isort", True)]

    rel_paths = [relpath(repo_root, path) for path in staged_python]
    outcomes = []

    black_result = run_python_module(
        python_bin,
        "black",
        ["--check", "--diff", "--line-length=127", *rel_paths],
        repo_root=repo_root,
    )
    if black_result.returncode != 0:
        print(black_result.stdout.strip())
        print(black_result.stderr.strip())
    outcomes.append(CheckOutcome("Black", black_result.returncode == 0))

    isort_result = run_python_module(
        python_bin,
        "isort",
        ["--check-only", "--diff", "--profile=black", "--line-length=127", *rel_paths],
        repo_root=repo_root,
    )
    if isort_result.returncode != 0:
        print(isort_result.stdout.strip())
        print(isort_result.stderr.strip())
    outcomes.append(CheckOutcome("isort", isort_result.returncode == 0))

    return outcomes


def check_import_heuristics(staged_python: Sequence[Path], repo_root: Path) -> CheckOutcome:
    if not staged_python:
        return CheckOutcome("Import heuristics", True)

    offenders = []
    for path in staged_python:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relpath(repo_root, path))
        except SyntaxError:
            continue

        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_names.update(alias.asname or alias.name for alias in node.names)

        used_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for suspect in ("iio", "cv2"):
            if suspect in used_names and suspect not in imported_names:
                offenders.append(f"{relpath(repo_root, path)}: '{suspect}' used without import")

    if not offenders:
        return CheckOutcome("Import heuristics", True)

    print("Import heuristics detected likely missing imports:")
    for error in offenders:
        print(f"  - {error}")
    return CheckOutcome("Import heuristics", False)


def run_optional_quick_tests(repo_root: Path, python_bin: str, enabled: bool) -> CheckOutcome:
    if not enabled:
        return CheckOutcome("Quick tests", True, blocking=False)

    result = run_python_module(
        python_bin,
        "pytest",
        [
            "-q",
            "tests/test_format_utils.py::TestNormalizeExtension",
            "tests/test_error_handling.py::TestFileValidation",
        ],
        repo_root=repo_root,
    )
    if result.returncode == 0:
        return CheckOutcome("Quick tests", True, blocking=False)

    print(result.stdout.strip())
    print(result.stderr.strip())
    return CheckOutcome("Quick tests", False, blocking=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Run against tracked files instead of only staged files.",
    )
    parser.add_argument(
        "--quick-tests",
        action="store_true",
        help="Also run the legacy quick pytest smoke subset.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    repo_root = find_repo_root()
    python_bin = choose_python(repo_root)

    staged_paths = git_paths(repo_root, all_files=args.all_files)
    added_paths = git_paths(repo_root, all_files=False, diff_filter="A")

    text_paths = [path for path in staged_paths if path.is_file() and is_text_file(path)]
    staged_python = [path for path in staged_paths if path.suffix == ".py" and path.is_file()]
    staged_yaml = [path for path in staged_paths if path.suffix in {".yaml", ".yml"} and path.is_file()]

    outcomes = [
        check_untracked_core_files(repo_root),
        check_root_file_placement(repo_root, all_files=args.all_files),
        check_markdown_count(repo_root),
        auto_fix_text_hygiene(repo_root, text_paths),
        check_large_added_files(added_paths, repo_root),
        check_merge_conflicts(text_paths, repo_root),
        check_line_endings(text_paths, repo_root),
        check_yaml_syntax(staged_yaml, repo_root),
        check_lint_tool_parity(repo_root, python_bin, staged_python, staged_yaml),
        check_python_syntax_and_undefined_names(repo_root, python_bin, staged_python),
        *check_black_and_isort(repo_root, python_bin, staged_python),
        check_import_heuristics(staged_python, repo_root),
        run_optional_quick_tests(repo_root, python_bin, args.quick_tests),
    ]

    failures = [outcome.label for outcome in outcomes if not outcome.ok and outcome.blocking]
    warnings = [outcome.label for outcome in outcomes if not outcome.ok and not outcome.blocking]

    print("\nPre-commit quality summary")
    for outcome in outcomes:
        status = "PASS" if outcome.ok else ("WARN" if not outcome.blocking else "FAIL")
        print(f"  [{status}] {outcome.label}")

    if failures:
        print("\nCommit blocked by:")
        for label in failures:
            print(f"  - {label}")
        return 1

    if warnings:
        print("\nWarnings:")
        for label in warnings:
            print(f"  - {label}")

    print("\nPre-commit checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
