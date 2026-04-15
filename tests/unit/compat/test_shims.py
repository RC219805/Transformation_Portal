"""Unit tests for transformation_portal.compat.shims module.

Tests the shim classes and factory functions for maintaining
backward compatibility when API structures change.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from transformation_portal.compat.shims import (
    DeprecatedConstant,
    LegacyAPIShim,
    create_compatibility_wrapper,
    create_module_alias,
)

pytestmark = pytest.mark.unit


class TestLegacyAPIShim:
    """Test the LegacyAPIShim proxy class."""

    def test_basic_attribute_forwarding(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that attributes are forwarded to the real object."""

        class NewAPI:
            def __init__(self) -> None:
                self.value = 42

            def get_data(self) -> str:
                return "data"

        real_obj = NewAPI()
        shim = LegacyAPIShim(real_obj, "OldAPI")

        with caplog.at_level(logging.WARNING):
            assert shim.value == 42
            assert shim.get_data() == "data"

        # Warnings should have been logged
        assert "Accessing deprecated object 'OldAPI'" in caplog.text

    def test_attribute_mapping(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that old attribute names are mapped to new ones."""

        class NewProcessor:
            def execute(self) -> str:
                return "executed"

        shim = LegacyAPIShim(
            NewProcessor(),
            "OldProcessor",
            {"run": "execute"},  # old_name -> new_name
        )

        with caplog.at_level(logging.WARNING):
            result = shim.run()  # Should call execute()

        assert result == "executed"
        assert "'run' -> 'execute'" in caplog.text

    def test_warn_once_behavior(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warn_once=True only warns once per attribute."""

        class NewAPI:
            value = 10

        shim = LegacyAPIShim(NewAPI(), "OldAPI", warn_once=True)

        with caplog.at_level(logging.WARNING):
            # Access same attribute multiple times
            _ = shim.value
            _ = shim.value
            _ = shim.value

        # Should only have one warning
        assert caplog.text.count("Accessing deprecated object") == 1

    def test_warn_always_by_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warn_once=False (default) warns on every access."""

        class NewAPI:
            value = 10

        shim = LegacyAPIShim(NewAPI(), "OldAPI", warn_once=False)

        with caplog.at_level(logging.WARNING):
            _ = shim.value
            _ = shim.value
            _ = shim.value

        # Should have multiple warnings
        assert caplog.text.count("Accessing deprecated object") == 3

    def test_missing_attribute_raises_attribute_error(self) -> None:
        """Test accessing missing attribute raises AttributeError."""

        class NewAPI:
            pass

        shim = LegacyAPIShim(NewAPI(), "OldAPI")

        with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
            _ = shim.nonexistent

    def test_missing_mapped_attribute_shows_mapping(self) -> None:
        """Test error message shows mapping when mapped attr is missing."""

        class NewAPI:
            pass

        shim = LegacyAPIShim(NewAPI(), "OldAPI", {"old_method": "new_method"})

        with pytest.raises(AttributeError, match="'old_method'.*mapped to 'new_method'"):
            _ = shim.old_method

    def test_setattr_forwarding(self) -> None:
        """Test __setattr__ forwards to real object."""

        class NewAPI:
            value: int = 0

        real_obj = NewAPI()
        shim = LegacyAPIShim(real_obj, "OldAPI")

        shim.value = 100

        assert real_obj.value == 100

    def test_setattr_with_mapping(self) -> None:
        """Test __setattr__ uses attribute mapping."""

        class NewAPI:
            new_value: int = 0

        real_obj = NewAPI()
        shim = LegacyAPIShim(real_obj, "OldAPI", {"old_value": "new_value"})

        shim.old_value = 50

        assert real_obj.new_value == 50

    def test_repr(self) -> None:
        """Test __repr__ shows shim information."""

        class NewAPI:
            pass

        shim = LegacyAPIShim(NewAPI(), "OldAPI")

        repr_str = repr(shim)
        assert "LegacyAPIShim" in repr_str
        assert "NewAPI" in repr_str

    def test_str(self) -> None:
        """Test __str__ shows deprecation info."""

        class NewAPI:
            pass

        shim = LegacyAPIShim(NewAPI(), "OldAPI")

        str_repr = str(shim)
        assert "OldAPI" in str_repr
        assert "deprecated" in str_repr
        assert "NewAPI" in str_repr

    def test_get_real_object(self) -> None:
        """Test _get_real_object returns underlying object."""

        class NewAPI:
            value = 42

        real_obj = NewAPI()
        shim = LegacyAPIShim(real_obj, "OldAPI")

        assert shim._get_real_object() is real_obj

    def test_method_call_with_arguments(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test forwarding method calls with arguments."""

        class NewAPI:
            def calculate(self, a: int, b: int, *, multiplier: int = 1) -> int:
                return (a + b) * multiplier

        shim = LegacyAPIShim(NewAPI(), "OldAPI")

        with caplog.at_level(logging.WARNING):
            result = shim.calculate(2, 3, multiplier=10)

        assert result == 50


class TestCreateCompatibilityWrapper:
    """Test the create_compatibility_wrapper function."""

    def test_basic_wrapper(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test basic wrapper creation."""

        def new_process(data: str) -> str:
            return f"processed: {data}"

        old_process = create_compatibility_wrapper(new_process, "old_process", "new_process")

        with caplog.at_level(logging.WARNING):
            result = old_process("test")

        assert result == "processed: test"
        assert "'old_process' is deprecated" in caplog.text
        assert "new_process" in caplog.text

    def test_warn_once_default_true(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warn_once=True is the default."""

        def new_func() -> str:
            return "result"

        old_func = create_compatibility_wrapper(new_func, "old_func", "new_func")

        with caplog.at_level(logging.WARNING):
            _ = old_func()
            _ = old_func()
            _ = old_func()

        # Should only warn once by default
        assert caplog.text.count("'old_func' is deprecated") == 1

    def test_warn_once_false(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warn_once=False warns on every call."""

        def new_func() -> str:
            return "result"

        old_func = create_compatibility_wrapper(new_func, "old_func", "new_func", warn_once=False)

        with caplog.at_level(logging.WARNING):
            _ = old_func()
            _ = old_func()
            _ = old_func()

        # Should warn every time
        assert caplog.text.count("'old_func' is deprecated") == 3

    def test_wrapper_preserves_name(self) -> None:
        """Test wrapper has old function name."""

        def new_func() -> str:
            return "result"

        old_func = create_compatibility_wrapper(new_func, "old_func", "new_func")

        assert old_func.__name__ == "old_func"

    def test_wrapper_preserves_docstring(self) -> None:
        """Test wrapper includes original docstring."""

        def new_func() -> str:
            """Original documentation."""
            return "result"

        old_func = create_compatibility_wrapper(new_func, "old_func", "new_func")

        assert "Deprecated alias" in (old_func.__doc__ or "")
        assert "Original documentation" in (old_func.__doc__ or "")

    def test_wrapper_with_arguments(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test wrapper passes arguments correctly."""

        def new_add(a: int, b: int, *, c: int = 0) -> int:
            return a + b + c

        old_add = create_compatibility_wrapper(new_add, "old_add", "new_add")

        with caplog.at_level(logging.WARNING):
            result = old_add(1, 2, c=3)

        assert result == 6


class TestCreateModuleAlias:
    """Test the create_module_alias function."""

    def test_creates_shim_for_module(self) -> None:
        """Test create_module_alias creates a LegacyAPIShim."""
        # Create a mock module
        mock_module = MagicMock()
        mock_module.some_function.return_value = "result"

        alias = create_module_alias(mock_module, "old_module_name")

        assert isinstance(alias, LegacyAPIShim)

    def test_module_alias_with_attribute_map(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test module alias with attribute mapping."""
        mock_module = MagicMock()
        mock_module.new_func.return_value = "mapped result"

        alias = create_module_alias(
            mock_module,
            "old_module",
            attribute_map={"old_func": "new_func"},
        )

        with caplog.at_level(logging.WARNING):
            result = alias.old_func()

        assert result == "mapped result"

    def test_module_alias_warn_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test module alias with warn_once."""
        mock_module = MagicMock()
        mock_module.func.return_value = "result"

        alias = create_module_alias(mock_module, "old_module", warn_once=True)

        with caplog.at_level(logging.WARNING):
            _ = alias.func()
            _ = alias.func()

        # Only one warning due to warn_once=True (default)
        assert caplog.text.count("Accessing deprecated object") == 1


class TestDeprecatedConstant:
    """Test the DeprecatedConstant descriptor."""

    def test_basic_usage(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test DeprecatedConstant returns value and warns."""

        class Constants:
            OLD_VALUE = DeprecatedConstant(42, "OLD_VALUE", "NEW_VALUE")
            NEW_VALUE = 42

        with caplog.at_level(logging.WARNING):
            value = Constants.OLD_VALUE

        assert value == 42
        assert "OLD_VALUE" in caplog.text
        assert "deprecated" in caplog.text
        assert "NEW_VALUE" in caplog.text

    def test_warns_only_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test DeprecatedConstant only warns once."""

        class Constants:
            OLD_VALUE = DeprecatedConstant(100, "OLD_VALUE", "NEW_VALUE")

        with caplog.at_level(logging.WARNING):
            _ = Constants.OLD_VALUE
            _ = Constants.OLD_VALUE
            _ = Constants.OLD_VALUE

        # Should only have one warning
        assert caplog.text.count("OLD_VALUE") == 1

    def test_repr(self) -> None:
        """Test DeprecatedConstant repr."""
        const = DeprecatedConstant(42, "OLD", "NEW")
        repr_str = repr(const)

        assert "DeprecatedConstant" in repr_str
        assert "42" in repr_str
        assert "'OLD'" in repr_str
        assert "'NEW'" in repr_str

    def test_instance_access(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test DeprecatedConstant access via instance."""

        class Constants:
            OLD_VALUE = DeprecatedConstant("old", "OLD_VALUE", "NEW_VALUE")
            NEW_VALUE = "new"

        obj = Constants()

        with caplog.at_level(logging.WARNING):
            value = obj.OLD_VALUE

        assert value == "old"
        assert "deprecated" in caplog.text

    def test_different_value_types(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test DeprecatedConstant works with various value types."""

        class Constants:
            OLD_DICT = DeprecatedConstant({"key": "value"}, "OLD_DICT", "NEW_DICT")
            OLD_LIST = DeprecatedConstant([1, 2, 3], "OLD_LIST", "NEW_LIST")
            OLD_TUPLE = DeprecatedConstant((1, 2), "OLD_TUPLE", "NEW_TUPLE")

        with caplog.at_level(logging.WARNING):
            assert Constants.OLD_DICT == {"key": "value"}
            assert Constants.OLD_LIST == [1, 2, 3]
            assert Constants.OLD_TUPLE == (1, 2)
