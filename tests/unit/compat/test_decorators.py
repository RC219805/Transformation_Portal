"""Unit tests for transformation_portal.compat.decorators module.

Tests the deprecation decorators and introspection utilities for
marking and tracking deprecated code.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from transformation_portal.compat.decorators import (
    deprecated,
    get_deprecation_info,
    is_deprecated,
    moved_to,
    renamed_class,
    renamed_function,
    renamed_module,
)

pytestmark = pytest.mark.unit


class TestDeprecatedDecorator:
    """Test the @deprecated decorator."""

    def test_basic_deprecation_warning(self) -> None:
        """Test that deprecated functions emit DeprecationWarning."""

        @deprecated()
        def old_func() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()

            assert result == "result"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "old_func is deprecated" in str(w[0].message)

    def test_deprecation_with_replacement(self) -> None:
        """Test deprecation warning includes replacement info."""

        @deprecated(replacement="new_func")
        def old_func() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_func()

            assert "Use 'new_func' instead" in str(w[0].message)

    def test_deprecation_with_removal_version(self) -> None:
        """Test deprecation warning includes removal version."""

        @deprecated(removal_version="2.0.0")
        def old_func() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_func()

            assert "removal in v2.0.0" in str(w[0].message)

    def test_deprecation_with_reason(self) -> None:
        """Test deprecation warning includes custom reason."""

        @deprecated(reason="This function is inefficient")
        def old_func() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_func()

            assert "This function is inefficient" in str(w[0].message)

    def test_deprecation_preserves_function_metadata(self) -> None:
        """Test that functools.wraps preserves function metadata."""

        @deprecated()
        def documented_func() -> str:
            """This is the original docstring."""
            return "result"

        assert documented_func.__name__ == "documented_func"
        # Docstring should be modified to include deprecation notice
        assert "DEPRECATED" in (documented_func.__doc__ or "")
        assert "original docstring" in (documented_func.__doc__ or "")

    def test_deprecation_with_arguments(self) -> None:
        """Test deprecated functions with arguments work correctly."""

        @deprecated(replacement="add_numbers")
        def old_add(a: int, b: int) -> int:
            return a + b

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = old_add(2, 3)
            assert result == 5

    def test_deprecation_with_kwargs(self) -> None:
        """Test deprecated functions with keyword arguments."""

        @deprecated()
        def old_func(name: str, value: int = 10) -> str:
            return f"{name}={value}"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = old_func("test", value=42)
            assert result == "test=42"

    def test_custom_warning_category(self) -> None:
        """Test using FutureWarning instead of DeprecationWarning."""

        @deprecated(category=FutureWarning)
        def old_func() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_func()

            assert issubclass(w[0].category, FutureWarning)


class TestDeprecationIntrospection:
    """Test deprecation introspection utilities."""

    def test_get_deprecation_info_returns_metadata(self) -> None:
        """Test get_deprecation_info returns deprecation metadata."""

        @deprecated(replacement="new_func", removal_version="2.0.0", reason="Legacy")
        def old_func() -> str:
            return "result"

        info = get_deprecation_info(old_func)

        assert info is not None
        assert info["replacement"] == "new_func"
        assert info["removal_version"] == "2.0.0"
        assert info["reason"] == "Legacy"
        assert "message" in info

    def test_get_deprecation_info_returns_none_for_non_deprecated(self) -> None:
        """Test get_deprecation_info returns None for non-deprecated functions."""

        def normal_func() -> str:
            return "result"

        assert get_deprecation_info(normal_func) is None

    def test_is_deprecated_returns_true_for_deprecated(self) -> None:
        """Test is_deprecated returns True for deprecated functions."""

        @deprecated()
        def old_func() -> str:
            return "result"

        assert is_deprecated(old_func) is True

    def test_is_deprecated_returns_false_for_normal_function(self) -> None:
        """Test is_deprecated returns False for normal functions."""

        def normal_func() -> str:
            return "result"

        assert is_deprecated(normal_func) is False


class TestMovedToDecorator:
    """Test the @moved_to decorator."""

    def test_moved_to_emits_warning(self) -> None:
        """Test moved_to emits deprecation warning with location."""

        @moved_to("transformation_portal.new_module.new_func")
        def old_func() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()

            assert result == "result"
            assert len(w) == 1
            assert "transformation_portal.new_module.new_func" in str(w[0].message)
            assert "Moved to new namespace" in str(w[0].message)

    def test_moved_to_with_removal_version(self) -> None:
        """Test moved_to includes removal version in warning."""

        @moved_to("new_location", removal_version="3.0.0")
        def old_func() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_func()

            assert "3.0.0" in str(w[0].message)


class TestRenamedFunctionDecorator:
    """Test the @renamed_function decorator."""

    def test_renamed_function_emits_warning(self) -> None:
        """Test renamed_function emits deprecation warning."""

        @renamed_function("calculate_total")
        def calc_total() -> int:
            return 100

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calc_total()

            assert result == 100
            assert len(w) == 1
            assert "calculate_total" in str(w[0].message)
            assert "renamed" in str(w[0].message).lower()

    def test_renamed_function_with_removal_version(self) -> None:
        """Test renamed_function includes removal version."""

        @renamed_function("new_name", removal_version="2.0.0")
        def old_name() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_name()

            assert "2.0.0" in str(w[0].message)


class TestRenamedClassDecorator:
    """Test the @renamed_class decorator."""

    def test_renamed_class_emits_warning_on_instantiation(self) -> None:
        """Test renamed_class emits warning when class is instantiated."""

        @renamed_class("NewProcessor")
        class OldProcessor:
            def __init__(self, value: int) -> None:
                self.value = value

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = OldProcessor(42)

            assert obj.value == 42
            assert len(w) == 1
            assert "OldProcessor is deprecated" in str(w[0].message)
            assert "NewProcessor" in str(w[0].message)

    def test_renamed_class_with_removal_version(self) -> None:
        """Test renamed_class includes removal version."""

        @renamed_class("NewClass", removal_version="3.0.0")
        class OldClass:
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OldClass()

            assert "3.0.0" in str(w[0].message)

    def test_renamed_class_custom_warning_category(self) -> None:
        """Test renamed_class with custom warning category."""

        @renamed_class("NewClass", category=FutureWarning)
        class OldClass:
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OldClass()

            assert issubclass(w[0].category, FutureWarning)

    def test_renamed_class_is_detectable(self) -> None:
        """Test renamed classes are detected by is_deprecated."""

        @renamed_class("NewClass")
        class OldClass:
            pass

        # Suppress warning during check
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert is_deprecated(OldClass) is True


class TestRenamedModule:
    """Test the renamed_module function."""

    def test_renamed_module_emits_warning(self) -> None:
        """Test renamed_module emits module deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            renamed_module(
                "old_module",
                "new_module",
                stacklevel=2,  # Adjusted for test context
            )

            assert len(w) == 1
            assert "old_module is deprecated" in str(w[0].message)
            assert "new_module" in str(w[0].message)


class TestDeprecatedMethod:
    """Test @deprecated on class methods."""

    def test_deprecated_instance_method(self) -> None:
        """Test deprecation on instance methods."""

        class MyClass:
            @deprecated(replacement="new_method")
            def old_method(self) -> str:
                return "result"

            def new_method(self) -> str:
                return "result"

        obj = MyClass()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = obj.old_method()

            assert result == "result"
            assert len(w) == 1
            assert "old_method is deprecated" in str(w[0].message)

    def test_deprecated_class_method(self) -> None:
        """Test deprecation on class methods."""

        class MyClass:
            @classmethod
            @deprecated(replacement="new_cls_method")
            def old_cls_method(cls) -> str:
                return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = MyClass.old_cls_method()

            assert result == "result"
            assert len(w) == 1

    def test_deprecated_static_method(self) -> None:
        """Test deprecation on static methods."""

        class MyClass:
            @staticmethod
            @deprecated(replacement="new_static_method")
            def old_static_method() -> str:
                return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = MyClass.old_static_method()

            assert result == "result"
            assert len(w) == 1
