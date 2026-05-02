from __future__ import annotations

import ast
import builtins
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from transformation_portal.analyzers.decision_decay_dashboard import (
    ColorTokenReport,
    ColorTokenUsage,
    PrincipleSummary,
    ValidUntilRecord,
    _render_plain_dashboard,
    _valid_until_from_decorator,
    collect_color_token_report,
    collect_philosophy_violations,
    export_json,
    main,
    parse_args,
    render_dashboard,
    render_github_annotations,
)
from transformation_portal.analyzers.codebase_philosophy_auditor import Violation

pytestmark = [pytest.mark.unit]


def test_valid_until_from_decorator_extracts_deadline_and_reason(tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    decorator = ast.parse('@valid_until("2030-01-02", reason="cleanup")\ndef test_case():\n    pass\n').body[0].decorator_list[0]

    deadline, reason = _valid_until_from_decorator(decorator, path)

    assert deadline == date(2030, 1, 2)
    assert reason == "cleanup"


def test_valid_until_from_decorator_requires_reason_keyword(tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    decorator = ast.parse('@valid_until("2030-01-02")\ndef test_case():\n    pass\n').body[0].decorator_list[0]

    with pytest.raises(ValueError, match="missing required 'reason'"):
        _valid_until_from_decorator(decorator, path)


def test_collect_philosophy_violations_caps_examples(tmp_path: Path) -> None:
    good = tmp_path / "good.py"
    good.write_text("def good():\n    pass\n", encoding="utf-8")

    class DummyAuditor:
        def audit_module(self, module_path: Path):  # noqa: ANN001
            return [
                Violation(code="P1", principle="p1", message="a", line=1),
                Violation(code="P1", principle="p1", message="b", line=2),
                Violation(code="P1", principle="p1", message="c", line=3),
                Violation(code="P1", principle="p1", message="d", line=4),
            ]

    summaries = collect_philosophy_violations([good], auditor=DummyAuditor())

    assert summaries["p1"].count == 4
    assert len(summaries["p1"].examples) == 3
    assert all("good.py" in example for example in summaries["p1"].examples)


def test_collect_philosophy_violations_propagates_auditor_failures(tmp_path: Path) -> None:
    module = tmp_path / "bad.py"
    module.write_text("def bad():\n    pass\n", encoding="utf-8")

    class DummyAuditor:
        def audit_module(self, module_path: Path):  # noqa: ANN001
            raise SyntaxError(f"cannot parse {module_path.name}")

    with pytest.raises(SyntaxError, match="cannot parse bad.py"):
        collect_philosophy_violations([module], auditor=DummyAuditor())


def test_collect_color_token_report_handles_missing_or_invalid_files(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        collect_color_token_report(tmp_path / "missing.json")

    with pytest.raises(json.JSONDecodeError):
        collect_color_token_report(invalid_path)


def test_collect_color_token_report_detects_hex_token_ref_and_css_var_usage(tmp_path: Path) -> None:
    tokens_path = tmp_path / "lantern_tokens.json"
    tokens_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "color": {
                        "brand": {
                            "accent_blue": {"value": "#112233"},
                            "accent_gold": {"value": "#445566"},
                            "unused": {"value": "#778899"},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "hex.css").write_text(".a { color: #112233; }", encoding="utf-8")
    (tmp_path / "ref.js").write_text('const ref = "color.brand.accent_gold";', encoding="utf-8")
    (tmp_path / "vars.css").write_text(".b { color: var(--brand-accent-gold); }", encoding="utf-8")

    report = collect_color_token_report(tokens_path)

    usage = {item.token: item.used_in for item in report.tokens}
    assert usage["accent_blue"] == ["hex.css"]
    assert sorted(usage["accent_gold"]) == ["ref.js", "vars.css"]
    assert {item.token for item in report.orphans} == {"unused"}


def test_render_plain_dashboard_emits_expected_sections(capsys: pytest.CaptureFixture[str]) -> None:
    expired = ValidUntilRecord(
        target="test_old",
        deadline=date.today() - timedelta(days=1),
        reason="cleanup",
        path=Path("tests/test_old.py"),
        line=7,
    )
    summary = PrincipleSummary(principle="docs", count=2, examples=["a.py:1 – missing"])
    report = ColorTokenReport(tokens=[ColorTokenUsage(token="blue", hex_value="#112233", used_in=[])], orphans=[])

    _render_plain_dashboard([expired], {"docs": summary}, report)
    output = capsys.readouterr().out

    assert "=== Temporal Contracts ===" in output
    assert "!! test_old" in output
    assert "=== Philosophy Violations ===" in output
    assert "=== Orphan Brand Colors ===" in output


def test_render_dashboard_falls_back_when_rich_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):  # noqa: ANN002, ANN003
        if name.startswith("rich"):
            raise ModuleNotFoundError("rich unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    render_dashboard([], {}, ColorTokenReport(tokens=[], orphans=[]))

    assert "No valid_until decorators discovered." in capsys.readouterr().out


def test_render_github_annotations_emits_errors_and_warnings(capsys: pytest.CaptureFixture[str]) -> None:
    expired = ValidUntilRecord(
        target="expired_case",
        deadline=date.today() - timedelta(days=2),
        reason="expired",
        path=Path("tests/test_expired.py"),
        line=11,
    )
    soon = ValidUntilRecord(
        target="soon_case",
        deadline=date.today() + timedelta(days=3),
        reason="soon",
        path=Path("tests/test_soon.py"),
        line=12,
    )
    later = ValidUntilRecord(
        target="later_case",
        deadline=date.today() + timedelta(days=20),
        reason="later",
        path=Path("tests/test_later.py"),
        line=13,
    )

    render_github_annotations([expired, soon, later])
    output = capsys.readouterr().out

    assert "::error file=tests/test_expired.py,line=11,title=Expired Tech Debt::" in output
    assert "::warning file=tests/test_soon.py,line=12,title=Tech Debt Expiring Soon::" in output
    assert "test_later.py" not in output


def test_export_json_writes_expected_payload(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    valid_until = [
        ValidUntilRecord(
            target="case",
            deadline=date.today(),
            reason="track",
            path=Path("tests/test_case.py"),
            line=9,
        )
    ]
    summaries = {"docs": PrincipleSummary(principle="docs", count=1, examples=["x.py:3 – missing"])}
    report = ColorTokenReport(
        tokens=[
            ColorTokenUsage(token="used", hex_value="#111111", used_in=["app.css"]),
            ColorTokenUsage(token="unused", hex_value="#222222", used_in=[]),
        ],
        orphans=[ColorTokenUsage(token="unused", hex_value="#222222", used_in=[])],
    )

    export_json(destination, valid_until, summaries, report)
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["valid_until"][0]["target"] == "case"
    assert payload["philosophy_violations"]["docs"]["count"] == 1
    unused_flags = {entry["token"]: entry["unused"] for entry in payload["color_tokens"]}
    assert unused_flags == {"used": False, "unused": True}


def test_parse_args_and_main_wire_collectors_and_renderers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parsed = parse_args(["--root", str(tmp_path), "--json", str(tmp_path / "out.json"), "--github-actions"])
    assert parsed.root == tmp_path
    assert parsed.github_actions is True

    calls: dict[str, object] = {}
    valid_until = [
        ValidUntilRecord(
            target="case",
            deadline=date.today(),
            reason="track",
            path=Path("tests/test_case.py"),
            line=4,
        )
    ]
    summaries = {"docs": PrincipleSummary(principle="docs", count=1, examples=["example"])}
    report = ColorTokenReport(tokens=[ColorTokenUsage(token="used", hex_value="#111111", used_in=["a.css"])], orphans=[])

    def _collect_valid_until(tests_root: Path):
        calls["tests_root"] = tests_root
        return valid_until

    def _collect_philosophy(roots: list[Path]):
        calls["roots"] = roots
        return summaries

    def _collect_colors(tokens_path: Path):
        calls["tokens_path"] = tokens_path
        return report

    def _export(destination: Path, *_args):
        calls["json_path"] = destination

    def _annotate(records):
        calls["annotations"] = records

    def _render(records, principle_summaries, color_report):
        calls["render"] = (records, principle_summaries, color_report)

    monkeypatch.setattr("transformation_portal.analyzers.decision_decay_dashboard.collect_valid_until_records", _collect_valid_until)
    monkeypatch.setattr("transformation_portal.analyzers.decision_decay_dashboard.collect_philosophy_violations", _collect_philosophy)
    monkeypatch.setattr("transformation_portal.analyzers.decision_decay_dashboard.collect_color_token_report", _collect_colors)
    monkeypatch.setattr("transformation_portal.analyzers.decision_decay_dashboard.export_json", _export)
    monkeypatch.setattr("transformation_portal.analyzers.decision_decay_dashboard.render_github_annotations", _annotate)
    monkeypatch.setattr("transformation_portal.analyzers.decision_decay_dashboard.render_dashboard", _render)

    main(["--root", str(tmp_path), "--json", str(tmp_path / "out.json"), "--github-actions"])

    assert calls["tests_root"] == tmp_path / "tests"
    assert calls["roots"] == [tmp_path]
    assert calls["tokens_path"] == tmp_path / "assets" / "brand" / "lantern_logo" / "lantern_tokens.json"
    assert calls["json_path"] == tmp_path / "out.json"
    assert calls["annotations"] == valid_until
    assert calls["render"] == (valid_until, summaries, report)
