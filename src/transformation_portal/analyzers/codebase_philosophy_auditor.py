"""
Utility for auditing modules against high-level codebase principles.

The :class:`CodebasePhilosophyAuditor` inspects a Python module, extracts
``# Decision:`` annotations, and then applies a set of simple rules derived
from the repository's philosophy guidelines.

Design goals:
- Lightweight guardrails (no heavy static analysis).
- Explicit, local opt-outs via `# Decision:` annotations.
- Stable violation codes for CI filtering and developer ergonomics.

Decision annotations
--------------------
Format:

    # Decision: <name> - optional rationale text...

Examples:

    # Decision: allow_missing_docstring - generated file
    # Decision: undocumented_public_api - compatibility shim
    # Decision: allow_wildcard_import - re-export legacy API

Module-wide rule disabling (header-only):

    # Decision: disable_rule - public_api_documentation
    # Decision: disable_rule - no_wildcard_imports

The `disable_rule` decision must appear in the module header area (see
`HEADER_DECISION_MAX_LINE`) to avoid “buried waivers”.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

# Decisions intended to apply module-wide should appear near the top of file.
HEADER_DECISION_MAX_LINE = 20

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

    # Stable identifiers (good for CI filtering, dashboards, and docs).
    code: str

    # Human-oriented grouping and message.
    principle: str
    message: str

    # Optional source location and relevant decision (if any).
    line: Optional[int] = None
    decision: Optional[Decision] = None


@dataclass
class _AuditContext:
    """Runtime context shared by the auditing rules."""

    source_lines: List[str]
    decisions: List[Decision]
    disabled_rules: Set[str]

    def header_decisions(self) -> List[Decision]:
        return [d for d in self.decisions if d.line <= HEADER_DECISION_MAX_LINE]

    def global_decision(self, name: str) -> Optional[Decision]:
        """A “global” decision must appear in the header region of the module."""
        wanted = name.lower()
        for decision in self.header_decisions():
            if decision.name == wanted:
                return decision
        return None

    def decision_for_line(self, name: str, line: int, *, max_distance: int = 2) -> Optional[Decision]:
        """Find a decision immediately above (or on) a given line."""
        wanted = name.lower()
        for decision in reversed(self.decisions):
            if decision.name != wanted:
                continue
            if 0 <= line - decision.line <= max_distance:
                return decision
        return None

    def is_rule_disabled(self, rule_id: str) -> bool:
        return rule_id in self.disabled_rules


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


def _extract_disabled_rules(decisions: Sequence[Decision]) -> Set[str]:
    """
    Collect module-wide disabled rule IDs.

    To avoid “buried waivers”, `disable_rule` is honored only in the header region.
    Use format: # Decision: disable_rule - <rule_id>
    """
    disabled: Set[str] = set()
    for d in decisions:
        if d.name != "disable_rule":
            continue
        if d.line > HEADER_DECISION_MAX_LINE:
            continue
        if not d.rationale:
            continue
        rule_id = d.rationale.strip()
        if rule_id:
            disabled.add(rule_id)
    return disabled


# ---------------------------------------------------------------------------
# Rule framework
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleSpec:
    """Metadata for a single audit rule."""

    rule_id: str
    principle: str
    description: str
    func: Callable[[ast.Module, _AuditContext], List[Violation]]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def _check_module_docstring(tree: ast.Module, context: _AuditContext) -> List[Violation]:
    if context.is_rule_disabled("module_docstring"):
        return []
    if ast.get_docstring(tree, clean=False) is not None:
        return []
    decision = context.global_decision("allow_missing_docstring")
    if decision:
        return []
    return [
        Violation(
            code="TPA001",
            principle="module_docstring",
            message="Module is missing a top-level docstring",
            line=1,
            decision=None,
        )
    ]


def _check_public_api_docstrings(tree: ast.Module, context: _AuditContext) -> List[Violation]:
    if context.is_rule_disabled("public_api_documentation"):
        return []

    violations: List[Violation] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name.startswith("_"):
            continue
        if ast.get_docstring(node, clean=False) is not None:
            continue

        decision = context.decision_for_line("undocumented_public_api", node.lineno)
        if decision:
            continue

        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        violations.append(
            Violation(
                code="TPA010",
                principle="public_api_documentation",
                message=f"Public {kind} '{node.name}' lacks a docstring",
                line=node.lineno,
                decision=None,
            )
        )
    return violations


def _check_wildcard_imports(tree: ast.Module, context: _AuditContext) -> List[Violation]:
    if context.is_rule_disabled("no_wildcard_imports"):
        return []

    violations: List[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if alias.name != "*":
                continue

            decision = context.decision_for_line("allow_wildcard_import", node.lineno)
            if decision:
                continue

            module_name = node.module if node.module is not None else "<relative>"
            violations.append(
                Violation(
                    code="TPA020",
                    principle="no_wildcard_imports",
                    message=f"Wildcard import from '{module_name}' violates import policy",
                    line=node.lineno,
                    decision=None,
                )
            )
    return violations


DEFAULT_RULES: List[RuleSpec] = [
    RuleSpec(
        rule_id="module_docstring",
        principle="module_docstring",
        description="Require a module-level docstring unless explicitly waived.",
        func=_check_module_docstring,
    ),
    RuleSpec(
        rule_id="public_api_documentation",
        principle="public_api_documentation",
        description="Require docstrings on public top-level functions/classes unless explicitly waived.",
        func=_check_public_api_docstrings,
    ),
    RuleSpec(
        rule_id="no_wildcard_imports",
        principle="no_wildcard_imports",
        description="Disallow wildcard imports unless explicitly waived.",
        func=_check_wildcard_imports,
    ),
]


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------


class CodebasePhilosophyAuditor:
    """Audit Python modules for high-level codebase philosophy violations."""

    def __init__(self, rules: Optional[Iterable[RuleSpec]] = None) -> None:
        self._rules: List[RuleSpec] = list(rules) if rules is not None else list(DEFAULT_RULES)
        self._rules_by_id: Dict[str, RuleSpec] = {r.rule_id: r for r in self._rules}

    @property
    def rules(self) -> Sequence[RuleSpec]:
        """Return the configured rules (including rule IDs)."""
        return tuple(self._rules)

    def audit_source(self, source: str, *, filename: str = "<memory>") -> List[Violation]:
        """Audit Python source code and return any principle violations discovered."""
        tree = ast.parse(source, filename=filename)
        source_lines = source.splitlines()
        decisions = _extract_decisions(source_lines)
        disabled_rules = _extract_disabled_rules(decisions)
        context = _AuditContext(
            source_lines=source_lines,
            decisions=decisions,
            disabled_rules=disabled_rules,
        )

        violations: List[Violation] = []
        for rule in self._rules:
            if context.is_rule_disabled(rule.rule_id):
                continue
            violations.extend(rule.func(tree, context))
        return violations

    def audit_module(self, module_path: Path) -> List[Violation]:
        """Inspect *module_path* and return any principle violations discovered."""
        source = module_path.read_text(encoding="utf-8")
        return self.audit_source(source, filename=str(module_path))
