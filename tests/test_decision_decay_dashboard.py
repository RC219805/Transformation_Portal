from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path


from decision_decay_dashboard import (
    collect_color_token_report,
    collect_outdated_valid_until_records,
    collect_philosophy_violations,
    collect_valid_until_records,
)
from codebase_philosophy_auditor import Violation


def test_collect_valid_until_records_sorted(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    module = tests_dir / "test_example.py"
    deadline_a = (date.today() + timedelta(days=10)).isoformat()
    deadline_b = (date.today() + timedelta(days=5)).isoformat()
    module.write_text(
        "from tests.documentation import valid_until\n\n"
        f"@valid_until(\"{deadline_a}\", reason=\"Longer horizon\")\n"
        "def test_future_a():\n    pass\n\n"
        f"@valid_until(\"{deadline_b}\", reason=\"Soon\")\n"
        "def test_future_b():\n    pass\n"
    )

    records = collect_valid_until_records(tests_dir)

    assert [record.target for record in records] == ["test_future_b", "test_future_a"]
    assert records[0].reason == "Soon"


def test_collect_outdated_valid_until_records(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    module = tests_dir / "test_expirations.py"
    expired = (date.today() - timedelta(days=1)).isoformat()
    upcoming = (date.today() + timedelta(days=3)).isoformat()
    module.write_text(
        "from tests.documentation import valid_until\n\n"
        f"@valid_until(\"{expired}\", reason=\"Expired contract\")\n"
        "def test_outdated():\n    pass\n\n"
        f"@valid_until(\"{upcoming}\", reason=\"Still valid\")\n"
        "def test_current():\n    pass\n"
    )

    records = collect_outdated_valid_until_records(tests_dir)

    assert [record.target for record in records] == ["test_outdated"]
    assert records[0].reason == "Expired contract"


class DummyAuditor:
    def __init__(self, violations: list[Violation]):
        self._violations = violations
        self.audited: list[Path] = []

    def audit_module(self, module_path: Path):
        self.audited.append(module_path)
        return self._violations


def test_collect_philosophy_violations_aggregates(tmp_path):
    module = tmp_path / "module.py"
    module.write_text("def sample():\n    pass\n")

    violations = [
        Violation(principle="module_docstring", message="Missing", line=1),
        Violation(principle="module_docstring", message="Missing", line=1),
        Violation(principle="public_api_documentation", message="Doc", line=2),
    ]

    auditor = DummyAuditor(violations)

    summaries = collect_philosophy_violations([module], auditor=auditor)

    assert auditor.audited == [module]
    assert summaries["module_docstring"].count == 2
    assert any("module.py" in example for example in summaries["module_docstring"].examples)
    assert summaries["public_api_documentation"].count == 1


def test_collect_color_token_report_identifies_orphans(tmp_path):
    tokens_path = tmp_path / "lantern_tokens.json"
    tokens_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "color": {
                        "brand": {
                            "active": {"value": "#112233"},
                            "idle": {"value": "#445566"},
                        }
                    }
                }
            }
        )
    )

    css_file = tmp_path / "example.css"
    css_file.write_text(".button { color: #112233; }")

    report = collect_color_token_report(tokens_path)

    assert {usage.token for usage in report.tokens} == {"active", "idle"}
    orphan_tokens = {usage.token for usage in report.orphans}
    assert orphan_tokens == {"idle"}
    active_usage = next(usage for usage in report.tokens if usage.token == "active")
    assert active_usage.used_in == ["example.css"]
