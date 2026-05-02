"""Unit tests for the jsonschema_formats helper.

Covers ``build_jsonschema_format_checker`` and the RFC3339 ``date-time``
checker it installs. The checker is contract-bearing: it is the source of
truth for whether a run-card timestamp is accepted by Draft2020-12 schema
validation, so each accept/reject path is covered explicitly.
"""

from __future__ import annotations

import pytest

from transformation_portal.lux_depth_v3.validators.jsonschema_formats import (
    build_jsonschema_format_checker,
)

pytestmark = [pytest.mark.unit]


@pytest.fixture(scope="module")
def checker():
    return build_jsonschema_format_checker()


class TestCheckerConstruction:
    def test_returns_format_checker_with_date_time_registered(self, checker):
        from jsonschema import FormatChecker

        assert isinstance(checker, FormatChecker)
        assert "date-time" in checker.checkers

    def test_each_call_returns_independent_instance(self):
        first = build_jsonschema_format_checker()
        second = build_jsonschema_format_checker()
        assert first is not second


class TestDateTimeAccepts:
    @pytest.mark.parametrize(
        "value",
        [
            "2026-03-20T12:00:00Z",
            "2026-03-20T12:00:00.123Z",
            "2026-03-20T12:00:00.123456Z",
            "2026-03-20T12:00:00+00:00",
            "2026-03-20T12:00:00-05:00",
            "2026-03-20T12:00:00.5+09:30",
            "2000-02-29T00:00:00Z",
        ],
    )
    def test_accepts_valid_rfc3339(self, checker, value):
        assert checker.conforms(value, "date-time") is True

    def test_non_string_short_circuits_to_pass(self, checker):
        # Per spec, a format checker only constrains strings; non-string
        # values are accepted so they fall through to the type validator.
        assert checker.conforms(42, "date-time") is True
        assert checker.conforms(None, "date-time") is True
        assert checker.conforms({"x": 1}, "date-time") is True


class TestDateTimeRejects:
    @pytest.mark.parametrize(
        "value",
        [
            "",
            " 2026-03-20T12:00:00Z",
            "2026-03-20T12:00:00Z ",
            "\t2026-03-20T12:00:00Z",
            "2026-03-20T12:00:00",
            "2026-03-20",
            "12:00:00Z",
            "2026-03-20 12:00:00Z",
            "2026-03-20T12:00:00+0000",
            "2026-03-20T12:00:00+5:00",
            "2026-13-01T00:00:00Z",
            "2026-02-30T00:00:00Z",
            "2026-03-20T25:00:00Z",
            "not-a-timestamp",
        ],
    )
    def test_rejects_malformed_or_invalid(self, checker, value):
        assert checker.conforms(value, "date-time") is False


class TestIntegrationWithDraft202012Validator:
    def test_format_failure_surfaces_through_draft202012_validator(self, checker):
        import jsonschema

        schema = {"type": "string", "format": "date-time"}
        validator = jsonschema.Draft202012Validator(schema, format_checker=checker)
        assert list(validator.iter_errors("2026-03-20T12:00:00Z")) == []
        errors = list(validator.iter_errors("not-a-timestamp"))
        assert len(errors) == 1
        assert "date-time" in errors[0].message
