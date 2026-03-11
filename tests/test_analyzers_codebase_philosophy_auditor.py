from textwrap import dedent

from transformation_portal.analyzers.codebase_philosophy_auditor import (
    CodebasePhilosophyAuditor,
    Violation,
)


def test_audit_source_ignores_decision_examples_inside_docstrings() -> None:
    auditor = CodebasePhilosophyAuditor()
    source = dedent('''
        """Example module docs.

        # Decision: disable_rule - public_api_documentation
        # Decision: undocumented_public_api - example only
        """

        class PublicThing:
            pass
        ''')

    violations = auditor.audit_source(source)

    assert violations == [
        Violation(
            code="TPA010",
            principle="public_api_documentation",
            message="Public class 'PublicThing' lacks a docstring",
            line=8,
            decision=None,
        )
    ]


def test_audit_source_ignores_decisions_inside_string_literals() -> None:
    auditor = CodebasePhilosophyAuditor()
    source = dedent('''
        BANNER = """
        # Decision: disable_rule - public_api_documentation
        """


        def PublicThing():
            return True
        ''')

    violations = auditor.audit_source(source)

    assert {(violation.code, violation.principle) for violation in violations} == {
        ("TPA001", "module_docstring"),
        ("TPA010", "public_api_documentation"),
    }


def test_custom_rule_callables_remain_supported() -> None:
    def always_fail_rule(*_args, **_kwargs) -> list[Violation]:
        return [Violation(code="CUSTOM001", principle="custom", message="forced failure")]

    auditor = CodebasePhilosophyAuditor(rules=[always_fail_rule])

    violations = auditor.audit_source('"""Doc."""\n')

    assert auditor.rules[0].func is always_fail_rule
    assert violations == [Violation(code="CUSTOM001", principle="custom", message="forced failure")]


def test_disabled_rules_are_matched_case_insensitively() -> None:
    auditor = CodebasePhilosophyAuditor()
    source = dedent('''
        # Decision: disable_rule - PUBLIC_API_DOCUMENTATION
        """Module doc."""


        def PublicThing():
            return True
        ''')

    violations = auditor.audit_source(source)

    assert violations == []
