#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for compatibility shims."""

import warnings
import pytest

from transformation_portal.compat.shims import (
    LegacyAPIShim,
    create_compatibility_wrapper,
    create_alias,
)


class TestLegacyAPIShim:
    """Tests for LegacyAPIShim class."""

    def test_shim_shows_warning_on_attribute_access(self):
        """Test that shim shows warning when accessing attributes."""
        class MockImplementation:
            some_attribute = "value"

        class TestShim(LegacyAPIShim):
            def __init__(self):
                super().__init__(
                    "OldAPI",
                    "transformation_portal.NewAPI",
                    "2.0.0"
                )

            def _get_implementation(self):
                return MockImplementation

        shim = TestShim()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = shim.some_attribute

            assert value == "value"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "OldAPI" in str(w[0].message)
            assert "NewAPI" in str(w[0].message)
            assert "2.0.0" in str(w[0].message)

    def test_shim_shows_warning_only_once(self):
        """Test that shim shows warning only once per instance."""
        class MockImplementation:
            attr1 = "value1"
            attr2 = "value2"

        class TestShim(LegacyAPIShim):
            def __init__(self):
                super().__init__("OldAPI", "transformation_portal.NewAPI")

            def _get_implementation(self):
                return MockImplementation

        shim = TestShim()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = shim.attr1
            _ = shim.attr2

            # Should only warn once
            assert len(w) == 1

    def test_shim_callable(self):
        """Test that shim can be called like the original."""
        class MockCallable:
            def __call__(self, x, y):
                return x + y

        class TestShim(LegacyAPIShim):
            def __init__(self):
                super().__init__("OldFunc", "transformation_portal.NewFunc")

            def _get_implementation(self):
                return MockCallable()

        shim = TestShim()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = shim(10, 20)

            assert result == 30
            assert len(w) == 1

    def test_shim_without_removal_version(self):
        """Test shim warning message without removal version."""
        class MockImplementation:
            value = 42

        class TestShim(LegacyAPIShim):
            def __init__(self):
                super().__init__("OldAPI", "transformation_portal.NewAPI")

            def _get_implementation(self):
                return MockImplementation

        shim = TestShim()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = shim.value

            assert len(w) == 1
            message = str(w[0].message)
            # Should not mention version if not provided
            assert "version" not in message.lower() or "Will be removed" not in message

    def test_shim_not_implemented_error(self):
        """Test that shim raises error if _get_implementation not overridden."""
        shim = LegacyAPIShim("OldAPI", "NewAPI")

        with pytest.raises(NotImplementedError):
            _ = shim.some_attribute


class TestCreateCompatibilityWrapper:
    """Tests for create_compatibility_wrapper function."""

    def test_wrapper_maps_parameters(self):
        """Test that wrapper correctly maps old parameter names to new ones."""
        def new_func(new_param1, new_param2=10):
            return new_param1 + new_param2

        def old_func(old_param1, old_param2=10):
            pass  # Not actually used

        wrapper = create_compatibility_wrapper(
            old_func,
            new_func,
            param_mapping={
                'old_param1': 'new_param1',
                'old_param2': 'new_param2',
            },
            removal_version="2.0.0"
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wrapper(old_param1=5, old_param2=15)

            assert result == 20
            assert len(w) == 1
            assert "old_func" in str(w[0].message)
            assert "new_func" in str(w[0].message)
            assert "2.0.0" in str(w[0].message)

    def test_wrapper_with_positional_args(self):
        """Test wrapper with positional arguments."""
        def new_func(x, y):
            return x * y

        def old_func(x, y):
            pass

        wrapper = create_compatibility_wrapper(old_func, new_func)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wrapper(3, 4)

            assert result == 12
            assert len(w) == 1

    def test_wrapper_with_no_mapping(self):
        """Test wrapper when no parameter mapping is needed."""
        def new_func(a, b):
            return a + b

        def old_func(a, b):
            pass

        wrapper = create_compatibility_wrapper(
            old_func,
            new_func,
            removal_version="3.0.0"
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wrapper(a=10, b=20)

            assert result == 30
            assert len(w) == 1

    def test_wrapper_preserves_function_metadata(self):
        """Test that wrapper preserves function metadata."""
        def old_func(x):
            """Old function documentation."""

        def new_func(x):
            return x

        wrapper = create_compatibility_wrapper(old_func, new_func)

        assert wrapper.__name__ == "old_func"
        assert wrapper.__doc__ == "Old function documentation."

    def test_wrapper_partial_parameter_mapping(self):
        """Test wrapper with partial parameter mapping."""
        def new_func(unchanged_param, renamed_param):
            return f"{unchanged_param}-{renamed_param}"

        def old_func(unchanged_param, old_param):
            pass

        wrapper = create_compatibility_wrapper(
            old_func,
            new_func,
            param_mapping={'old_param': 'renamed_param'}
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wrapper(unchanged_param="a", old_param="b")

            assert result == "a-b"
            assert len(w) == 1


class TestCreateAlias:
    """Tests for create_alias function."""

    def test_alias_for_function(self):
        """Test creating an alias for a function."""
        def original_function(x):
            return x * 2

        aliased = create_alias(
            original_function,
            "old_function_name",
            removal_version="2.0.0"
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = aliased(5)

            assert result == 10
            assert len(w) == 1
            assert "old_function_name" in str(w[0].message)
            assert "original_function" in str(w[0].message)
            assert "2.0.0" in str(w[0].message)

    def test_alias_for_class(self):
        """Test creating an alias for a class."""
        class OriginalClass:
            def __init__(self, value):
                self.value = value

        AliasedClass = create_alias(
            OriginalClass,
            "OldClassName",
            removal_version="3.0.0"
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = AliasedClass(42)

            assert obj.value == 42
            assert len(w) == 1
            assert "OldClassName" in str(w[0].message)
            assert "OriginalClass" in str(w[0].message)
            assert "3.0.0" in str(w[0].message)

    def test_alias_function_name(self):
        """Test that aliased function has correct name."""
        def original():
            pass

        aliased = create_alias(original, "legacy_name")
        assert aliased.__name__ == "legacy_name"

    def test_alias_class_name(self):
        """Test that aliased class has correct name."""
        class Original:
            pass

        Aliased = create_alias(Original, "LegacyClass")
        assert Aliased.__name__ == "LegacyClass"

    def test_alias_without_removal_version(self):
        """Test alias without removal version."""
        def original():
            return "result"

        aliased = create_alias(original, "old_name")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = aliased()

            assert result == "result"
            assert len(w) == 1
            message = str(w[0].message)
            assert "old_name" in message
            assert "original" in message

    def test_aliased_class_inheritance(self):
        """Test that aliased class properly inherits from original."""
        class OriginalClass:
            def method(self):
                return "from_original"

        AliasedClass = create_alias(OriginalClass, "OldClass")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = AliasedClass()
            assert obj.method() == "from_original"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
