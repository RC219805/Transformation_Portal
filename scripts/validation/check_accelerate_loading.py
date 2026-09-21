#!/usr/bin/env python3
"""Reject repository use of Accelerate's unpatched checkpoint-index loaders.

GHSA-4j2p-28q2-5m79 affects checkpoint loading, not dispatch_model, device maps,
or CPU/disk offload. This source gate preserves those supported features. It
checks statically resolvable imports, attribute aliases and literal getattr;
it is not a Python sandbox or an audit of installed third-party packages.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UNSAFE_APIS = frozenset({"load_checkpoint_and_dispatch", "load_checkpoint_in_model"})
MAX_ALIAS_PASSES = 16
MAX_IDENTITIES_PER_ALIAS = 128
MAX_TOTAL_ALIAS_IDENTITIES = 4096


class _AliasBudgetExceeded(ValueError):
    """Alias expansion exceeded the source audit's bounded-work contract."""


def _add_identity(identities: set[str], identity: str) -> bool:
    """Retain an identity only when the set can grow within its budget."""
    if identity in identities:
        return False
    if len(identities) >= MAX_IDENTITIES_PER_ALIAS:
        raise _AliasBudgetExceeded("Accelerate alias resolution exceeded its per-alias identity budget")
    identities.add(identity)
    return True


def check_source(source: str) -> list[tuple[int, str]]:
    """Return unsafe API references without matching comments or string examples."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [(exc.lineno or 1, f"cannot audit invalid Python: {exc.msg}")]
    try:
        return _check_tree(tree)
    except _AliasBudgetExceeded as exc:
        return [(1, str(exc))]


def _check_tree(tree: ast.AST) -> list[tuple[int, str]]:
    """Audit parsed source using monotone, budgeted alias provenance."""
    # This is a conservative source audit, not a scope-sensitive interpreter.
    # Keep every possible identity: an unrelated import in another scope (or a
    # later reimport) must not erase an earlier Accelerate binding.
    aliases: dict[str, set[str]] = {}
    total_identities = 0

    def retain_alias(name: str, identity: str) -> bool:
        nonlocal total_identities
        identities = aliases.setdefault(name, set())
        if identity in identities:
            return False
        if total_identities >= MAX_TOTAL_ALIAS_IDENTITIES:
            raise _AliasBudgetExceeded("Accelerate alias resolution exceeded its total identity budget")
        _add_identity(identities, identity)
        total_identities += 1
        return True

    def qualified(node: ast.AST) -> set[str]:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, {node.id})
        if isinstance(node, ast.Attribute):
            return {f"{name}.{node.attr}" for name in qualified(node.value)}
        if isinstance(node, ast.Call):
            names = qualified(node.func)
            result: set[str] = set()
            if "getattr" in names and len(node.args) >= 2:
                attr = node.args[1]
                if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
                    for name in qualified(node.args[0]):
                        _add_identity(result, f"{name}.{attr.value}")
            if names & {"importlib.import_module", "__import__"} and node.args:
                module = node.args[0]
                if isinstance(module, ast.Constant) and isinstance(module.value, str):
                    _add_identity(result, module.value)
            return result
        return set()

    nodes = list(ast.walk(tree))
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                identity = alias.name if alias.asname else alias.name.split(".")[0]
                retain_alias(name, identity)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                retain_alias(alias.asname or alias.name, f"{node.module}.{alias.name}")
    # Resolve alias chains independent of function nesting/traversal order.
    # Retain imported names conservatively even if code later rebinds them.
    for _ in range(MAX_ALIAS_PASSES):
        changed = False
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                values = {
                    value
                    for value in qualified(node.value)
                    if value.startswith(("accelerate.", "importlib.")) or value == "accelerate"
                }
                for target in targets:
                    if isinstance(target, ast.Name) and values:
                        for value in values:
                            if retain_alias(target.id, value):
                                changed = True
        if not changed:
            break
    else:
        return [(1, "Accelerate alias resolution did not converge within its 16-pass bound")]

    errors: set[tuple[int, str]] = set()
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "accelerate":
            for alias in node.names:
                if alias.name in UNSAFE_APIS or alias.name == "*":
                    errors.add((node.lineno, f"unsafe Accelerate import: {alias.name}"))
        if isinstance(node, (ast.Attribute, ast.Call, ast.Name)):
            for name in qualified(node):
                if name.startswith("accelerate.") and name.rsplit(".", 1)[-1] in UNSAFE_APIS:
                    errors.add((node.lineno, f"unsafe Accelerate checkpoint API: {name}"))
    return sorted(errors)


def main() -> int:
    """Scan tracked and new, non-ignored Python source without traversing runtimes."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.py"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        files = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
        failures = 0
        scanned = 0
        for filename in files:
            path = Path(filename)
            if set(path.parts) & {"archive", "_archive", "historical", "fixtures"}:
                continue
            scanned += 1
            for line, reason in check_source((REPO_ROOT / path).read_text(encoding="utf-8")):
                print(f"{filename}:{line}: {reason}", file=sys.stderr)
                failures += 1
    except (OSError, UnicodeError, subprocess.CalledProcessError) as exc:
        print(f"Accelerate loading audit failed: {exc}", file=sys.stderr)
        return 2
    print(f"Accelerate loading audit: {scanned} Python files, {failures} violations")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
