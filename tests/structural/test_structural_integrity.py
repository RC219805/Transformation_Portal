"""Structural Integrity & Governance Suite.

Enforces:
1) No exact duplicate Python implementations between src/ and scripts/
2) No exact duplicate Python implementations between src/ and tests/
3) Public API surface freeze (AST-based, __all__ first)
4) Semantic version gating tied to API changes
5) CLI argument surface freeze (subprocess parser introspection)
6) Import boundary enforcement (AST-based static import analysis)
7) Console script entrypoint freeze ([project.scripts] in pyproject.toml)
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
STRUCTURAL_ROOT = Path(__file__).parent

SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
TESTS_ROOT = REPO_ROOT / "tests"

API_SNAPSHOT_FILE = STRUCTURAL_ROOT / "api_surface_snapshot.json"
CLI_SNAPSHOT_FILE = STRUCTURAL_ROOT / "cli_surface_snapshot.json"
CONSOLE_SCRIPTS_SNAPSHOT_FILE = STRUCTURAL_ROOT / "console_scripts_snapshot.json"

PUBLIC_API_POLICY_FILE = STRUCTURAL_ROOT / "public_api_policy.json"
VERSION_POLICY_FILE = STRUCTURAL_ROOT / "version_policy.json"
CLI_SURFACE_POLICY_FILE = STRUCTURAL_ROOT / "cli_surface_policy.json"
IMPORT_BOUNDARY_POLICY_FILE = STRUCTURAL_ROOT / "import_boundary_policy.json"


# ---------------------------------------------------------------------
# Hashing utilities (SHA-256)
# ---------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collect_python_hashes(root: Path) -> dict[str, list[Path]]:
    """Return digest -> [relative paths] for all *.py under root."""
    hashes: dict[str, list[Path]] = {}
    if not root.exists():
        return hashes

    for file in root.rglob("*.py"):
        if "__pycache__" in file.parts or not file.is_file():
            continue
        digest = _sha256(file)
        hashes.setdefault(digest, []).append(file.relative_to(REPO_ROOT))

    return hashes


def _find_exact_duplicates(canonical_hashes: dict[str, list[Path]], candidate_root: Path) -> list[str]:
    duplicates: list[str] = []
    if not candidate_root.exists():
        return duplicates

    for file in candidate_root.rglob("*.py"):
        if "__pycache__" in file.parts or not file.is_file():
            continue

        digest = _sha256(file)
        if digest in canonical_hashes:
            for canon in canonical_hashes[digest]:
                duplicates.append(f"{file.relative_to(REPO_ROOT)} == {canon} (sha256={digest[:12]})")

    return duplicates


# ---------------------------------------------------------------------
# Invariant 1 — src ↔ scripts
# ---------------------------------------------------------------------


def test_no_exact_duplicate_modules_between_src_and_scripts():
    src_hashes = _collect_python_hashes(SRC_ROOT)
    duplicates = _find_exact_duplicates(src_hashes, SCRIPTS_ROOT)

    assert not duplicates, (
        "Exact duplicate Python implementations found between src/ and scripts/.\n"
        "Scripts must be thin wrappers importing canonical modules.\n\n" + "\n".join(sorted(duplicates))
    )


# ---------------------------------------------------------------------
# Invariant 2 — src ↔ tests
# ---------------------------------------------------------------------


def test_no_exact_duplicate_modules_between_src_and_tests():
    src_hashes = _collect_python_hashes(SRC_ROOT)
    duplicates = _find_exact_duplicates(src_hashes, TESTS_ROOT)

    assert not duplicates, (
        "Exact duplicate Python implementations found between src/ and tests/.\n"
        "Tests must import canonical modules instead of copying them.\n\n" + "\n".join(sorted(duplicates))
    )


# ---------------------------------------------------------------------
# Public API surface freeze (AST-based; __all__ authoritative)
# ---------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _module_glob_match(pattern: str, module: str) -> bool:
    return fnmatch(module, pattern)


def _path_to_module(src_py: Path) -> str:
    rel = src_py.relative_to(SRC_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# pylint: disable=too-many-nested-blocks
def _ast_extract___all__(tree: ast.AST) -> set[str] | None:
    """Extract __all__ if assigned as a literal list/tuple of strings."""
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        out: set[str] = set()
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                out.add(elt.value)
                        return out
    return None


# pylint: enable=too-many-nested-blocks


def _ast_extract_declared_public_defs(tree: ast.AST) -> set[str]:
    """Fallback extraction: top-level class/function defs without underscore prefix."""
    out: set[str] = set()
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            out.add(node.name)
    return out


def _collect_public_api_surface() -> dict[str, Any]:
    """Collect public API according to tests/structural/public_api_policy.json."""
    policy = _load_json(PUBLIC_API_POLICY_FILE)
    package = policy["public_package"]

    public_globs: list[str] = policy.get("public_modules", [])
    ignore_globs: list[str] = policy.get("ignore_modules", [])
    ignore_symbols: set[str] = set(policy.get("ignore_symbols", []))
    require___all__ = bool(policy.get("require___all__", True))
    fallback_globs: list[str] = policy.get("allow_fallback_without___all__", [])

    modules: dict[str, list[str]] = {}

    for py in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in py.parts or not py.is_file():
            continue

        module = _path_to_module(py)
        if not module.startswith(package):
            continue

        if public_globs and not any(_module_glob_match(pat, module) for pat in public_globs):
            continue
        if any(_module_glob_match(pat, module) for pat in ignore_globs):
            continue

        tree = ast.parse(py.read_text(encoding="utf-8"))
        explicit_all = _ast_extract___all__(tree)

        if explicit_all is not None:
            symbols = explicit_all
        else:
            allow_fallback = any(_module_glob_match(pat, module) for pat in fallback_globs)
            if require___all__ and not allow_fallback:
                raise AssertionError(
                    f"Public module '{module}' lacks a literal __all__. "
                    f"Define __all__ (list/tuple of strings) or allow fallback in public_api_policy.json."
                )
            symbols = _ast_extract_declared_public_defs(tree)

        if explicit_all is not None:
            # __all__ is authoritative; keep explicit dunder/underscore exports.
            filtered = sorted(s for s in symbols if s not in ignore_symbols)
        else:
            filtered = sorted(s for s in symbols if s not in ignore_symbols and not s.startswith("_"))
        if filtered:
            modules[module] = filtered

    return {
        "snapshot_version": 1,
        "package": package,
        "modules": dict(sorted(modules.items())),
    }


def _extract_version() -> str:
    """Read package version from version policy target."""
    policy = _load_json(VERSION_POLICY_FILE)
    source = policy.get("version_source", "python_var")

    if source == "pyproject":
        import tomllib

        pyproject_file = REPO_ROOT / policy.get("pyproject_file", "pyproject.toml")
        data = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
        try:
            return data["project"]["version"]
        except KeyError as exc:
            raise AssertionError(f"Missing [project].version in {pyproject_file}") from exc

    version_file = REPO_ROOT / policy["version_file"]
    version_variable = policy["version_variable"]
    text = version_file.read_text(encoding="utf-8")
    match = re.search(rf'^\s*{re.escape(version_variable)}\s*=\s*["\']([^"\']+)["\']\s*$', text, re.MULTILINE)
    if not match:
        raise AssertionError(
            f"Could not locate {version_variable} assignment in {version_file}. "
            f'Expected form: {version_variable} = "X.Y.Z"'
        )
    return match.group(1)


def _parse_semver(v: str) -> tuple[int, int, int]:
    parts = v.split(".")
    if len(parts) != 3:
        raise AssertionError(f"Version '{v}' is not strict SemVer 'X.Y.Z'")
    major, minor, patch = parts
    return int(major), int(minor), int(patch)


def _classify_api_change(old: dict[str, Any], new: dict[str, Any]) -> str:
    """Return one of NONE, PATCH, MINOR, MAJOR."""
    old_modules: dict[str, list[str]] = old.get("modules", {})
    new_modules: dict[str, list[str]] = new.get("modules", {})

    if old_modules == new_modules:
        return "NONE"

    old_keys = set(old_modules)
    new_keys = set(new_modules)
    if old_keys - new_keys:
        return "MAJOR"

    breaking = False
    additive = bool(new_keys - old_keys)
    for key in sorted(old_keys & new_keys):
        old_symbols = set(old_modules.get(key, []))
        new_symbols = set(new_modules.get(key, []))
        if old_symbols - new_symbols:
            breaking = True
        if new_symbols - old_symbols:
            additive = True

    if breaking:
        return "MAJOR"
    if additive:
        return "MINOR"
    return "PATCH"


def _require_bump(old_v: str, new_v: str, level: str) -> None:
    old = _parse_semver(old_v)
    new = _parse_semver(new_v)
    if new <= old:
        raise AssertionError(f"Version did not increase: old={old_v} new={new_v} (required for {level} change)")

    old_major, old_minor, _old_patch = old
    new_major, new_minor, _new_patch = new

    if level == "PATCH":
        return
    if level == "MINOR":
        if new_major == old_major and new_minor == old_minor:
            raise AssertionError(f"API additive change requires MINOR or MAJOR bump. old={old_v} new={new_v}")
        return
    if level == "MAJOR":
        if new_major == old_major:
            raise AssertionError(f"API breaking change requires MAJOR bump. old={old_v} new={new_v}")
        return
    raise AssertionError(f"Unknown required level: {level}")


def test_public_api_surface_is_frozen_and_version_gated():
    """Enforce public API freeze and SemVer gating."""
    current = _collect_public_api_surface()
    current_version = _extract_version()

    if os.getenv("UPDATE_API_SNAPSHOT") == "1":
        current["_version_marker"] = current_version
        API_SNAPSHOT_FILE.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        return

    if not API_SNAPSHOT_FILE.exists():
        raise AssertionError("API snapshot missing. Generate with:\n  UPDATE_API_SNAPSHOT=1 pytest -q tests/structural\n")

    expected = _load_json(API_SNAPSHOT_FILE)
    expected_norm = dict(expected)
    expected_norm.pop("_version_marker", None)

    if current != expected_norm:
        old_version = expected.get("_version_marker")
        if not old_version:
            raise AssertionError(
                "API snapshot missing _version_marker. Regenerate with:\n"
                "  UPDATE_API_SNAPSHOT=1 pytest -q tests/structural\n"
            )

        change_level = _classify_api_change(expected_norm, current)
        if change_level != "NONE":
            _require_bump(old_version, current_version, change_level)

        raise AssertionError(
            "Public API surface changed.\n\n"
            f"Classified change level: {change_level}\n"
            f"Old version: {old_version}\n"
            f"New version: {current_version}\n\n"
            "If intentional, regenerate snapshot with:\n"
            "  UPDATE_API_SNAPSHOT=1 pytest -q tests/structural\n"
        )


# ---------------------------------------------------------------------
# CLI argument surface freeze (subprocess-based)
# ---------------------------------------------------------------------


CLI_INTROSPECTOR = r"""
import argparse
import json
import importlib
import sys
from pathlib import Path

repo_root = Path.cwd()
src_root = repo_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

policy = json.loads((repo_root / "tests" / "structural" / "cli_surface_policy.json").read_text(encoding="utf-8"))
cli_module = policy["cli_module"]
parser_builder = policy.get("parser_builder", "build_parser")

module = importlib.import_module(cli_module)
if not hasattr(module, parser_builder):
    raise SystemExit(f"CLI module {cli_module} lacks required parser builder {parser_builder}()")

parser = getattr(module, parser_builder)()
if not isinstance(parser, argparse.ArgumentParser):
    raise SystemExit(f"{cli_module}.{parser_builder}() must return argparse.ArgumentParser")

def normalize_default(value):
    if value is argparse.SUPPRESS:
        return "__SUPPRESS__"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    return repr(value)

surface = {}
positionals = []
for action in parser._actions:
    option_strings = list(getattr(action, "option_strings", []) or [])
    type_obj = getattr(action, "type", None)
    payload = {
        "required": bool(getattr(action, "required", False)),
        "choices": sorted(str(choice) for choice in (getattr(action, "choices", None) or [])) or None,
        "default": normalize_default(getattr(action, "default", None)),
        "dest": getattr(action, "dest", None),
    }
    if option_strings:
        for opt in option_strings:
            surface[opt] = payload
    else:
        positionals.append({
            "dest": getattr(action, "dest", None),
            "nargs": getattr(action, "nargs", None),
            "choices": payload["choices"],
            "default": payload["default"],
            "type": type_obj.__name__ if callable(type_obj) and hasattr(type_obj, "__name__") else None,
        })

print(json.dumps({
    "snapshot_version": 1,
    "cli_module": cli_module,
    "parser_builder": parser_builder,
    "positionals": positionals,
    "arguments": dict(sorted(surface.items())),
}, sort_keys=True))
"""


def _collect_cli_surface() -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    python_executable = sys.executable
    if not python_executable or not Path(python_executable).exists():
        python_executable = shutil.which("python3") or shutil.which("python")
    if not python_executable:
        raise AssertionError(
            "No usable Python executable found for CLI surface introspection. "
            "Install Python 3 or restore the project virtualenv."
        )
    result = subprocess.run(
        [python_executable, "-c", CLI_INTROSPECTOR],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip())


def test_cli_argument_surface_is_frozen():
    current = _collect_cli_surface()

    if os.getenv("UPDATE_CLI_SNAPSHOT") == "1":
        CLI_SNAPSHOT_FILE.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        return

    if not CLI_SNAPSHOT_FILE.exists():
        raise AssertionError("CLI snapshot missing. Generate with:\n  UPDATE_CLI_SNAPSHOT=1 pytest -q tests/structural\n")

    expected = _load_json(CLI_SNAPSHOT_FILE)
    assert current == expected, (
        "CLI argument surface changed.\n\n"
        "If intentional, regenerate snapshot with:\n"
        "  UPDATE_CLI_SNAPSHOT=1 pytest -q tests/structural\n"
    )


def _pyproject_path() -> Path:
    """Resolve the canonical pyproject.toml path from version policy settings."""
    policy = _load_json(VERSION_POLICY_FILE)
    if policy.get("version_source") == "pyproject":
        return REPO_ROOT / policy.get("pyproject_file", "pyproject.toml")
    return REPO_ROOT / "pyproject.toml"


def _collect_console_scripts() -> dict[str, Any]:
    """Collect deterministic [project.scripts] mapping from pyproject.toml."""
    import tomllib

    pyproject_file = _pyproject_path()
    data = tomllib.loads(pyproject_file.read_text(encoding="utf-8"))
    scripts = (data.get("project", {}) or {}).get("scripts", {}) or {}
    if not isinstance(scripts, dict):
        raise AssertionError(f"[project.scripts] must be a table/dict in {pyproject_file}")

    return {
        "snapshot_version": 1,
        "scripts": {key: scripts[key] for key in sorted(scripts)},
    }


def _classify_console_scripts_change(old: dict[str, str], new: dict[str, str]) -> str:
    """Return one of NONE, MINOR, MAJOR for [project.scripts] changes."""
    if old == new:
        return "NONE"

    old_keys = set(old)
    new_keys = set(new)
    if old_keys - new_keys:
        return "MAJOR"

    for key in old_keys & new_keys:
        if old[key] != new[key]:
            return "MAJOR"

    if new_keys - old_keys:
        return "MINOR"

    return "MAJOR"


def _validate_console_script_targets_resolve_to_src_modules(scripts: dict[str, str]) -> None:
    """Statically validate entrypoint target shape and module existence under src/."""
    module_re = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")
    callable_re = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")

    for name, target in scripts.items():
        if ":" not in target:
            raise AssertionError(f"Console script '{name}' target must be 'module:callable' (got {target!r})")
        module, callable_name = target.split(":", 1)
        module = module.strip()
        callable_name = callable_name.strip()

        if not module or not callable_name:
            raise AssertionError(f"Console script '{name}' target malformed: {target!r}")
        if not module_re.fullmatch(module):
            raise AssertionError(f"Console script '{name}' module is not a valid dotted identifier: {module!r}")
        if not callable_re.fullmatch(callable_name):
            raise AssertionError(f"Console script '{name}' callable is not a valid dotted identifier: {callable_name!r}")

        module_rel = Path(*module.split("."))
        module_file = SRC_ROOT / (str(module_rel) + ".py")
        module_init = SRC_ROOT / module_rel / "__init__.py"
        if module_file.exists():
            module_path = module_file
        elif module_init.exists():
            module_path = module_init
        else:
            raise AssertionError(
                f"Console script '{name}' points to missing module '{module}'. "
                f"Expected one of: {module_file.relative_to(REPO_ROOT)}, {module_init.relative_to(REPO_ROOT)}"
            )

        root_symbol = callable_name.split(".", 1)[0]
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        top_level_symbols: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                top_level_symbols.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        top_level_symbols.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                top_level_symbols.add(node.target.id)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    symbol = alias.asname or alias.name.split(".", 1)[0]
                    top_level_symbols.add(symbol)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    symbol = alias.asname or alias.name
                    top_level_symbols.add(symbol)

        if root_symbol not in top_level_symbols:
            raise AssertionError(
                f"Console script '{name}' target callable root '{root_symbol}' "
                f"not found as a top-level symbol in '{module}'."
            )


def test_console_scripts_are_frozen_and_version_gated():
    """Freeze [project.scripts] and enforce SemVer bump requirements for contract changes."""
    version_policy = _load_json(VERSION_POLICY_FILE)
    if version_policy.get("version_source") != "pyproject":
        raise AssertionError("Console script SemVer gating requires version_source='pyproject'.")

    current = _collect_console_scripts()
    _validate_console_script_targets_resolve_to_src_modules(current["scripts"])
    current_version = _extract_version()

    if os.getenv("UPDATE_CONSOLE_SCRIPTS_SNAPSHOT") == "1":
        payload = dict(current)
        payload["_version_marker"] = current_version
        CONSOLE_SCRIPTS_SNAPSHOT_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return

    if not CONSOLE_SCRIPTS_SNAPSHOT_FILE.exists():
        raise AssertionError(
            "Console scripts snapshot missing. Generate with:\n"
            "  UPDATE_CONSOLE_SCRIPTS_SNAPSHOT=1 pytest -q tests/structural\n"
        )

    expected = _load_json(CONSOLE_SCRIPTS_SNAPSHOT_FILE)
    old_version = expected.get("_version_marker")
    expected_norm = dict(expected)
    expected_norm.pop("_version_marker", None)

    if current != expected_norm:
        if not old_version:
            raise AssertionError(
                "Console scripts snapshot missing _version_marker. Regenerate with:\n"
                "  UPDATE_CONSOLE_SCRIPTS_SNAPSHOT=1 pytest -q tests/structural\n"
            )

        change_level = _classify_console_scripts_change(
            expected_norm.get("scripts", {}),
            current.get("scripts", {}),
        )
        if change_level != "NONE":
            _require_bump(old_version, current_version, change_level)

        raise AssertionError(
            "Console script entrypoints changed ([project.scripts]).\n\n"
            f"Classified change level: {change_level}\n"
            f"Old version: {old_version}\n"
            f"New version: {current_version}\n\n"
            "If intentional, regenerate snapshot with:\n"
            "  UPDATE_CONSOLE_SCRIPTS_SNAPSHOT=1 pytest -q tests/structural\n"
        )


# ---------------------------------------------------------------------
# Import boundary enforcement (AST-based)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class _BoundaryRule:
    source_root: Path
    forbid_import_globs: tuple[str, ...]
    allow_import_globs: tuple[str, ...]


def _extract_imports_from_ast(py: Path) -> list[str]:
    tree = ast.parse(py.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def _load_boundary_rules() -> list[_BoundaryRule]:
    policy = _load_json(IMPORT_BOUNDARY_POLICY_FILE)
    rules: list[_BoundaryRule] = []
    for rule in policy.get("rules", []):
        rules.append(
            _BoundaryRule(
                source_root=REPO_ROOT / rule["source_root"],
                forbid_import_globs=tuple(rule.get("forbid_import_globs", [])),
                allow_import_globs=tuple(rule.get("allow_import_globs", [])),
            )
        )
    return rules


def _matches_any_glob(value: str, globs: list[str] | tuple[str, ...]) -> bool:
    return any(fnmatch(value, pattern) for pattern in globs)


def test_import_boundaries_are_respected():
    rules = _load_boundary_rules()
    for rule in rules:
        if not rule.source_root.exists():
            continue

        for py in rule.source_root.rglob("*.py"):
            if "__pycache__" in py.parts or not py.is_file():
                continue

            violations: list[str] = []
            for imported_module in _extract_imports_from_ast(py):
                if _matches_any_glob(imported_module, rule.allow_import_globs):
                    continue
                if _matches_any_glob(imported_module, rule.forbid_import_globs):
                    violations.append(imported_module)

            assert not violations, (
                f"Import boundary violations in {py.relative_to(REPO_ROOT)}\n"
                f"Forbidden imports: {sorted(set(violations))}\n"
            )
