from pathlib import Path
from textwrap import dedent

import pytest

from codebase_philosophy_auditor import CodebasePhilosophyAuditor, Violation

# pylint: disable=redefined-outer-name  # pytest fixtures


@pytest.fixture()
def auditor() -> CodebasePhilosophyAuditor:
    return CodebasePhilosophyAuditor()


def _write_module(tmp_path: Path, content: str) -> Path:
    module_path = tmp_path / "module_under_test.py"
    module_path.write_text(content)
    return module_path


def test_audit_module_detects_missing_docstring(tmp_path: Path, auditor: CodebasePhilosophyAuditor) -> None:
    module_path = _write_module(
        tmp_path,
        dedent(
            '''
            from math import sqrt


            def area(radius):
                return 3.14 * radius * radius
            '''
        ).strip(),
    )

    violations = auditor.audit_module(module_path)

    assert any(v.principle == "module_docstring" for v in violations)


def test_audit_module_detects_undocumented_public_api(tmp_path: Path, auditor: CodebasePhilosophyAuditor) -> None:
    module_path = _write_module(
        tmp_path,
        dedent(
            '''
            """Feature module."""


            class Service:
                def execute(self):
                    return True


            async def orchestrate():
                return None
            '''
        ),
    )

    violations = auditor.audit_module(module_path)

    assert {(v.principle, v.line) for v in violations} == {
        ("public_api_documentation", 5),
        ("public_api_documentation", 10),
    }


def test_audit_module_respects_documented_decisions(tmp_path: Path, auditor: CodebasePhilosophyAuditor) -> None:
    module_path = _write_module(
        tmp_path,
        dedent(
            '''
            """Experiment module."""

            # Decision: allow_wildcard_import - tight integration with plugin API
            from plugin_api import *  # noqa: F403

            # Decision: undocumented_public_api - docstring inherited from base class
            class Plugin:
                def run(self):
                    """Runtime hook."""
                    return True


            def _helper():
                """Private helper is ignored."""
                return False
            '''
        ),
    )

    violations = auditor.audit_module(module_path)

    assert violations == []


def test_custom_rules_can_be_supplied(tmp_path: Path) -> None:
    def always_fail_rule(*_args, **_kwargs) -> list[Violation]:
        return [Violation(principle="custom", message="forced failure")]

    auditor = CodebasePhilosophyAuditor(rules=[always_fail_rule])
    module_path = _write_module(
        tmp_path,
        dedent(
            '''
            """Doc."""

            '''
        ),
    )

    violations = auditor.audit_module(module_path)

    assert violations == [Violation(principle="custom", message="forced failure")]
