"""Utility for auditing modules against high-level codebase principles.

The :class:`CodebasePhilosophyAuditor` inspects a Python module, extracts
``# Decision:`` annotations, and then applies a set of simple rules derived
from the repository's philosophy guidelines.  The goal isn't to perform heavy
static analysis, but rather to provide lightweight guardrails that surface
common policy violations and highlight where explicit decisions were made to
bend the rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional


_DECISION_PATTERN = re.compile(
    r"#\s*Decision\s*:\s*(?P<name>[A-Za-z0-9_\-]+)(?:\s*-\s*(?P<text>.*))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Decision:
    """Represents an explicit decision documented in the source code."""

    name: str
    line: int
    rationale: Optional[str] = None


@dataclass(frozen=True)
class Violation:
    """Information about a principle violation discovered during the audit."""

    principle: str
    message: str
    line: Optional[int] = None
    decision: Optional[Decision] = None


@dataclass
class _AuditContext:
    """Runtime context shared by the auditing rules."""

    source_lines: List[str]
    decisions: List[Decision]

    def global_decision(self, name: str) -> Optional[Decision]:
        name = name.lower()
        for decision in self.decisions:
            if decision.name == name:
                return decision
        return None

    def decision_for_line(
        self, name: str, line: int, *, max_distance: int = 2
    ) -> Optional[Decision]:
        name = name.lower()
        for decision in reversed(self.decisions):
            if decision.name != name:
                continue
            if 0 <= line - decision.line <= max_distance:
                return decision
        return None


def _extract_decisions(source_lines: Iterable[str]) -> List[Decision]:
    decisions: List[Decision] = []
    for index, line in enumerate(source_lines, start=1):
        match = _DECISION_PATTERN.search(line)
        if not match:
            continue
        name = match.group("name").strip().lower()
        rationale = match.group("text")
        if rationale is not None:
            rationale = rationale.strip() or None
        decisions.append(Decision(name=name, line=index, rationale=rationale))
    return decisions


def _check_module_docstring(tree: ast.AST, context: _AuditContext) -> List[Violation]:
    if ast.get_docstring(tree, clean=False) is not None:
        return []
    if context.global_decision("allow_missing_docstring"):
        return []
    return [
        Violation(
            principle="module_docstring",
            message="Module is missing a top-level docstring",
            line=1,
        )
    ]


def _check_public_api_docstrings(tree: ast.Module, context: _AuditContext) -> List[Violation]:
    violations: List[Violation] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            if ast.get_docstring(node, clean=False) is not None:
                continue
            decision = context.decision_for_line("undocumented_public_api", node.lineno)
            if decision:
                continue
            violations.append(
                Violation(
                    principle="public_api_documentation",
                    message=f"Public {type(node).__name__.lower()} '{node.name}' lacks a docstring",
                    line=node.lineno,
                )
            )
    return violations


def _check_wildcard_imports(tree: ast.Module, context: _AuditContext) -> List[Violation]:
    violations: List[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    decision = context.decision_for_line("allow_wildcard_import", node.lineno)
                    if decision:
                        continue
                    violations.append(
                        Violation(
                            principle="no_wildcard_imports",
                            message=f"Wildcard import from '{node.module or ''}' violates import policy",
                            line=node.lineno,
                        )
                    )
    return violations


Rule = Callable[[ast.Module, _AuditContext], List[Violation]]


class CodebasePhilosophyAuditor:
    """Audit Python modules for high-level codebase philosophy violations."""

    def __init__(self, rules: Optional[Iterable[Rule]] = None) -> None:
        self._rules: List[Rule] = list(rules) if rules is not None else [
            _check_module_docstring,
            _check_public_api_docstrings,
            _check_wildcard_imports,
        ]

    def audit_module(self, module_path: Path) -> List[Violation]:
        """Inspect *module_path* and return any principle violations discovered."""

        source = module_path.read_text()
        tree = ast.parse(source, filename=str(module_path))
        source_lines = source.splitlines()
        decisions = _extract_decisions(source_lines)
        context = _AuditContext(source_lines=source_lines, decisions=decisions)

        violations: List[Violation] = []
        for rule in self._rules:
            violations.extend(rule(tree, context))
        return violations
