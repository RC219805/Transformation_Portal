#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for compatibility decorators."""

import warnings
import pytest

from transformation_portal.compat.decorators import (
    deprecated,
    renamed_function,
    renamed_class,
    renamed_module,
    moved_to,
    experimental,
)


class TestDeprecatedDecorator:
    """Tests for @deprecated decorator."""

    def test_deprecated_function_shows_warning(self):
        """Test that deprecated function shows warning."""
        @deprecated()
        def old_function():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_function()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "old_function" in str(w[0].message)
            assert "deprecated" in str(w[0].message).lower()
            assert result == "result"

    def test_deprecated_function_with_replacement(self):
        """Test deprecated function with replacement suggestion."""
        @deprecated(replacement="new_function")
        def old_function():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_function()

            assert len(w) == 1
            assert "new_function" in str(w[0].message)

    def test_deprecated_function_with_removal_version(self):
        """Test deprecated function with removal version."""
        @deprecated(removal_version="2.0.0")
        def old_function():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_function()

            assert len(w) == 1
            assert "2.0.0" in str(w[0].message)

    def test_deprecated_function_with_custom_message(self):
        """Test deprecated function with custom message."""
        custom_msg = "Use the shiny new API instead"

        @deprecated(message=custom_msg)
        def old_function():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_function()

            assert len(w) == 1
            assert custom_msg in str(w[0].message)

    def test_deprecated_class_shows_warning(self):
        """Test that deprecated class shows warning on instantiation."""
        @deprecated()
        class OldClass:
            def __init__(self):
                self.value = 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = OldClass()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "OldClass" in str(w[0].message)
            assert obj.value == 42

    def test_deprecated_class_with_replacement(self):
        """Test deprecated class with replacement."""
        @deprecated(replacement="NewClass", removal_version="3.0.0")
        class OldClass:
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OldClass()

            assert len(w) == 1
            assert "NewClass" in str(w[0].message)
            assert "3.0.0" in str(w[0].message)

    def test_deprecated_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        @deprecated()
        def documented_function():
            """This is a documented function."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."
        assert hasattr(documented_function, "__deprecated__")
        assert documented_function.__deprecated__ is True

    def test_deprecated_with_custom_category(self):
        """Test deprecated with custom warning category."""
        @deprecated(category=FutureWarning)
        def old_function():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_function()

            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)


class TestRenamedFunction:
    """Tests for @renamed_function decorator."""

    def test_renamed_function_shows_warning(self):
        """Test that renamed function shows appropriate warning."""
        @renamed_function("old_process", "new_process", "2.0.0")
        def new_process(x):
            return x * 2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = new_process(5)

            assert result == 10
            assert len(w) == 1
            assert "old_process" in str(w[0].message)
            assert "new_process" in str(w[0].message)
            assert "renamed" in str(w[0].message).lower()


class TestRenamedClass:
    """Tests for @renamed_class decorator."""

    def test_renamed_class_shows_warning(self):
        """Test that renamed class shows appropriate warning."""
        @renamed_class("OldProcessor", "NewProcessor", "2.0.0")
        class NewProcessor:
            def __init__(self, value):
                self.value = value

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = NewProcessor(42)

            assert obj.value == 42
            assert len(w) == 1
            assert "OldProcessor" in str(w[0].message)
            assert "NewProcessor" in str(w[0].message)
            assert "renamed" in str(w[0].message).lower()


class TestRenamedModule:
    """Tests for renamed_module function."""

    def test_renamed_module_shows_warning(self):
        """Test that renamed_module shows warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            renamed_module(
                "transformation_portal.old_module",
                "transformation_portal.new_module",
                "2.0.0"
            )

            assert len(w) == 1
            assert "old_module" in str(w[0].message)
            assert "new_module" in str(w[0].message)
            assert "2.0.0" in str(w[0].message)

    def test_renamed_module_without_version(self):
        """Test renamed_module without removal version."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            renamed_module(
                "transformation_portal.old_module",
                "transformation_portal.new_module"
            )

            assert len(w) == 1
            assert "old_module" in str(w[0].message)
            assert "new_module" in str(w[0].message)


class TestMovedTo:
    """Tests for @moved_to decorator."""

    def test_moved_to_function(self):
        """Test moved_to decorator on function."""
        @moved_to("transformation_portal.processors.new_location", "2.0.0")
        def some_function():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = some_function()

            assert result == "result"
            assert len(w) == 1
            assert "moved" in str(w[0].message).lower()
            assert "transformation_portal.processors.new_location" in str(w[0].message)

    def test_moved_to_class(self):
        """Test moved_to decorator on class."""
        @moved_to("transformation_portal.new_location")
        class SomeClass:
            def __init__(self):
                self.value = 100

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = SomeClass()

            assert obj.value == 100
            assert len(w) == 1
            assert "moved" in str(w[0].message).lower()


class TestExperimental:
    """Tests for @experimental decorator."""

    def test_experimental_function_shows_warning(self):
        """Test that experimental function shows warning."""
        @experimental()
        def new_feature():
            return "feature"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = new_feature()

            assert result == "feature"
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "experimental" in str(w[0].message).lower()
            assert "new_feature" in str(w[0].message)

    def test_experimental_with_custom_message(self):
        """Test experimental decorator with custom message."""
        custom_msg = "This API is not stable yet"

        @experimental(message=custom_msg)
        def unstable_feature():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            unstable_feature()

            assert len(w) == 1
            assert custom_msg in str(w[0].message)

    def test_experimental_class_shows_warning(self):
        """Test that experimental class shows warning on instantiation."""
        @experimental()
        class ExperimentalClass:
            def __init__(self):
                self.value = 123

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = ExperimentalClass()

            assert obj.value == 123
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "experimental" in str(w[0].message).lower()

    def test_experimental_class_with_custom_message(self):
        """Test experimental class with custom message."""
        @experimental(message="Unstable API - use with caution")
        class UnstableClass:
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            UnstableClass()

            assert len(w) == 1
            assert "Unstable API" in str(w[0].message)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
