# tests/__init__.py
"""Unit tests for shared architectural helpers and Golden Hour Courtyard utilities."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

# Hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st
except ImportError as exc:
    raise ImportError(
        "Hypothesis is required for these tests. Install via `pip install hypothesis`."
    ) from exc

# Import the helpers under test
try:
    from helpers import documents, demonstrates, valid_until
except ImportError as e:
    raise ImportError(
        "Failed to import 'helpers' module. Ensure 'helpers' is in PYTHONPATH or installed."
    ) from e

# -------------------------
# tests for documents
# -------------------------


def test_documents_appends_docstring():
    @documents("Test note")
    def func():
        pass
    assert func.__doc__ == "Test note"


def test_documents_preserves_existing_docstring():
    @documents("Prefix note")
    def func():
        """Original doc"""
        return True
    assert func.__doc__ == "Prefix note\nOriginal doc"

# -------------------------
# tests for demonstrates
# -------------------------


def test_demonstrates_single_string():
    @demonstrates("ConceptA")
    def func():
        return True
    assert func.__demonstrates__ == ("ConceptA",)
    assert func.__doc__.startswith("Demonstrates: ConceptA")


def test_demonstrates_class():
    class Dummy:
        pass

    @demonstrates(Dummy)
    def func():
        return True
    assert func.__demonstrates__ == (Dummy,)
    doc = func.__doc__
    assert doc is not None
    assert "Dummy" in doc  # pylint: disable=unsupported-membership-test


def test_demonstrates_existing_docstring():
    @demonstrates("ConceptB")
    def func():
        """Existing doc"""
        return True
    assert func.__doc__.startswith("Demonstrates: ConceptB\n")
    assert "Existing doc" in func.__doc__

# -------------------------
# property-based demonstrates tests
# -------------------------


@given(
    st.one_of(
        st.text(min_size=1, max_size=20),
        st.integers(),
        st.builds(type, st.text(min_size=1, max_size=10), st.tuples())
    )
)
def test_demonstrates_with_various_concepts(concept):
    @demonstrates(concept)
    def dummy():
        return True
    assert dummy.__demonstrates__[0] == concept
    assert dummy.__doc__.startswith("Demonstrates:")


@given(
    st.lists(
        st.one_of(
            st.text(min_size=1, max_size=20),
            st.integers(),
            st.builds(type, st.text(min_size=1, max_size=10), st.tuples())
        ),
        min_size=1,
        max_size=5
    )
)
def test_demonstrates_with_multiple_various_concepts(concepts):
    @demonstrates(concepts)
    def dummy():
        return True
    assert dummy.__demonstrates__ == tuple(concepts)
    expected_parts = []
    for c in concepts:
        if hasattr(c, "__name__"):
            expected_parts.append(c.__name__)
        else:
            expected_parts.append(str(c))
    expected_prefix = "Demonstrates: " + ", ".join(expected_parts)
    assert dummy.__doc__.startswith(expected_prefix)

# -------------------------
# tests for valid_until
# -------------------------


def test_valid_until_future_allows_execution():
    future_date = (date.today() + timedelta(days=1)).isoformat()

    @valid_until(future_date, reason="future test")
    def func():
        return 42
    assert func() == 42
    assert func.__doc__.startswith(f"Valid until {future_date}")


def test_valid_until_past_raises():
    past_date = (date.today() - timedelta(days=1)).isoformat()

    @valid_until(past_date, reason="expired test")
    def func():
        return 42
    with pytest.raises(AssertionError) as exc:
        func()
    assert "expired" in str(exc.value)
    assert func.__doc__.startswith(f"Valid until {past_date}")

# -------------------------
# test suite sanity
# -------------------------


def test_basic_callable_property():
    """Sanity check: decorated functions are still callable."""
    @documents("note")
    @demonstrates("C")
    def func():
        return "ok"
    assert func() == "ok"
