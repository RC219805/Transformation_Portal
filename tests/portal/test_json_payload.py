"""Unit tests for the portal JSON payload request-boundary check."""

from __future__ import annotations

import json

import pytest

from transformation_portal.portal.json_payload import (
    INVALID_UNICODE_REASON,
    NON_FINITE_NUMBER_REASON,
    json_payload_rejection_reason,
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
