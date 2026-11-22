"""Unit tests for the portal JSON payload request-boundary check."""

from __future__ import annotations

import json

import pytest

from transformation_portal.portal.json_payload import (
    INVALID_UNICODE_REASON,
    NON_FINITE_NUMBER_REASON,
    json_payload_rejection_reason,
    json_safe_echo,
    unechoable_arg_fields,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"args": {"max_workers": 4, "ratio": 0.5, "name": "café", "flag": True, "none": None}},
        {"list": [1, 2.5, "x", [{"deep": "\U0001f600"}]]},
        {"big": 10**400},
    ],
)
def test_echoable_payloads_pass(payload):
    assert json_payload_rejection_reason(payload) is None


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ('{"a": NaN}', NON_FINITE_NUMBER_REASON),
        ('{"a": {"b": [1, Infinity]}}', NON_FINITE_NUMBER_REASON),
        ('{"a": -1e400}', NON_FINITE_NUMBER_REASON),
        ('{"a": "\\ud800"}', INVALID_UNICODE_REASON),
        ('{"\\udfff": 1}', INVALID_UNICODE_REASON),
        ('[["\\ud800x"]]', INVALID_UNICODE_REASON),
    ],
)
def test_unechoable_payloads_are_rejected(raw, reason):
    assert json_payload_rejection_reason(json.loads(raw)) == reason


def test_deeply_nested_payload_does_not_exhaust_the_stack():
    payload: object = float("nan")
    for _ in range(50_000):
        payload = [payload]
    assert json_payload_rejection_reason(payload) == NON_FINITE_NUMBER_REASON


def test_safe_echo_returns_echoable_values_unchanged():
    payload = {"args": {"ratio": 0.5, "names": ["café", "\U0001f600"], "big": 10**400}}
    assert json_safe_echo(payload) is payload


def test_safe_echo_nulls_unechoable_values_and_keeps_everything_else():
    payload = json.loads(
        '{"keep": "ok", "nan": NaN, "list": [1, Infinity, "x", ["\\ud800", {"deep": -1e400, "flag": true}]],'
        ' "\\udc00": "dropped", "text": "\\ud800", "tail": {"n": null, "f": 2.5}}'
    )

    echoed = json_safe_echo(payload)

    assert echoed == {
        "keep": "ok",
        "nan": None,
        "list": [1, None, "x", [None, {"deep": None, "flag": True}]],
        "text": None,
        "tail": {"n": None, "f": 2.5},
    }
    assert list(echoed) == ["keep", "nan", "list", "text", "tail"]
    assert json_payload_rejection_reason(echoed) is None
    json.dumps(echoed, allow_nan=False).encode("utf-8")
    # The source payload is never modified.
    assert payload["list"][3][0] == "\ud800"


@pytest.mark.parametrize("leaf", [float("nan"), "\ud800"])
def test_safe_echo_of_deeply_nested_unechoable_value_does_not_exhaust_the_stack(leaf):
    # json.loads accepts nesting well beyond the interpreter recursion limit,
    # so the null-echo copy must not recurse per level.
    depth = 50_000
    payload: object = leaf
    for _ in range(depth):
        payload = [payload]

    echoed = json_safe_echo({"note": payload, "label": "ok"})

    assert echoed["label"] == "ok"
    node = echoed["note"]
    for _ in range(depth):
        assert isinstance(node, list) and len(node) == 1
        node = node[0]
    assert node is None


def test_unechoable_arg_fields_reports_each_top_level_arg_once():
    args = json.loads('{"ok": 1, "note": [[NaN, NaN]], "name": "\\ud800", "\\udc00": 1, "ratio": 1e400}')

    assert unechoable_arg_fields(args) == [
        ("note", NON_FINITE_NUMBER_REASON),
        ("name", INVALID_UNICODE_REASON),
        ("args", INVALID_UNICODE_REASON),
        ("ratio", NON_FINITE_NUMBER_REASON),
    ]
    assert unechoable_arg_fields(["not", "a", "dict"]) == []
