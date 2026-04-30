"""Unit tests for plugins.validator.PluginValidator.

Covers metadata validation, interface compliance, manifest checks,
dependency validation, strict mode, and the convenience functions —
using only in-process mock plugins with no filesystem access.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Test plugin factories
# ---------------------------------------------------------------------------

def _make_valid_plugin(
    name: str = "valid_plugin",
    version: str = "1.0.0",
    plugin_type_str: str = "processor",
):
    """Return a well-formed plugin that should pass all validations."""
    from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

    pt = PluginType(plugin_type_str)

    class _ValidPlugin(PluginInterface):
        def _create_metadata(self):
            return PluginMetadata(name=name, version=version, plugin_type=pt)

        def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
            self._initialized = True

        def execute(self, *args, **kwargs) -> Any:
            return "ok"

    return _ValidPlugin()


def _make_validator(strict: bool = False):
    from transformation_portal.plugins.validator import PluginValidator

    return PluginValidator(strict_mode=strict)


def _make_manifest(
    name: str = "test_plugin",
    version: str = "1.0.0",
    plugin_type: str = "processor",
    entry_point: str = "my_module:MyPlugin",
    dependencies=None,
):
    from transformation_portal.plugins.loader import PluginManifest

    return PluginManifest(
        name=name,
        version=version,
        plugin_type=plugin_type,
        entry_point=entry_point,
        dependencies=dependencies or [],
    )


# ---------------------------------------------------------------------------
# ValidationResult helpers
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_initial_state_is_valid(self):
        from transformation_portal.plugins.validator import ValidationResult

        result = ValidationResult(plugin_name="test", is_valid=True)
        assert result.is_valid is True
        assert result.errors_count == 0
        assert result.warnings_count == 0

    def test_add_error_invalidates_result(self):
        from transformation_portal.plugins.validator import ValidationIssue, ValidationResult, ValidationSeverity

        result = ValidationResult(plugin_name="test", is_valid=True)
        result.add_issue(
            ValidationIssue(code="ERR", message="error", severity=ValidationSeverity.ERROR)
        )
        assert result.is_valid is False
        assert result.errors_count == 1

    def test_add_warning_does_not_invalidate(self):
        from transformation_portal.plugins.validator import ValidationIssue, ValidationResult, ValidationSeverity

        result = ValidationResult(plugin_name="test", is_valid=True)
        result.add_issue(
            ValidationIssue(code="WARN", message="warning", severity=ValidationSeverity.WARNING)
        )
        assert result.is_valid is True
        assert result.warnings_count == 1

    def test_get_issues_by_severity_filters_correctly(self):
        from transformation_portal.plugins.validator import ValidationIssue, ValidationResult, ValidationSeverity

        result = ValidationResult(plugin_name="test", is_valid=True)
        result.add_issue(
            ValidationIssue(code="W1", message="warn", severity=ValidationSeverity.WARNING)
        )
        result.add_issue(
            ValidationIssue(code="E1", message="err", severity=ValidationSeverity.ERROR)
        )
        warnings = result.get_issues_by_severity(ValidationSeverity.WARNING)
        assert len(warnings) == 1
        assert warnings[0].code == "W1"

    def test_str_contains_valid_or_invalid(self):
        from transformation_portal.plugins.validator import ValidationResult

        valid_result = ValidationResult(plugin_name="p", is_valid=True)
        assert "VALID" in str(valid_result)

        invalid_result = ValidationResult(plugin_name="p", is_valid=False)
        assert "INVALID" in str(invalid_result)


# ---------------------------------------------------------------------------
# ValidationIssue
# ---------------------------------------------------------------------------


class TestValidationIssue:
    def test_str_contains_code_and_message(self):
        from transformation_portal.plugins.validator import ValidationIssue, ValidationSeverity

        issue = ValidationIssue(code="MY_CODE", message="something wrong", severity=ValidationSeverity.ERROR)
        text = str(issue)
        assert "MY_CODE" in text
        assert "something wrong" in text

    def test_str_contains_suggestion_when_present(self):
        from transformation_portal.plugins.validator import ValidationIssue, ValidationSeverity

        issue = ValidationIssue(
            code="FIX_IT",
            message="fix this",
            severity=ValidationSeverity.WARNING,
            suggestion="Try doing X instead",
        )
        assert "Try doing X instead" in str(issue)


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------


class TestMetadataValidation:
    def test_valid_plugin_passes_metadata_validation(self):
        validator = _make_validator()
        plugin = _make_valid_plugin()
        result = validator.validate(plugin)
        # Check no metadata-related errors
        codes = [i.code for i in result.issues]
        assert "EMPTY_NAME" not in codes
        assert "EMPTY_VERSION" not in codes

    def test_invalid_name_format_produces_warning(self):
        from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

        class _BadNamePlugin(PluginInterface):
            def _create_metadata(self):
                return PluginMetadata(
                    name="BadCamelCase",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR,
                )

            def initialize(self, config=None):
                self._initialized = True

            def execute(self, *args, **kwargs):
                return None

        validator = _make_validator()
        result = validator.validate(_BadNamePlugin())
        codes = [i.code for i in result.issues]
        assert "INVALID_NAME_FORMAT" in codes

    def test_deprecated_without_replacement_produces_warning(self):
        from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

        class _DepPlugin(PluginInterface):
            def _create_metadata(self):
                return PluginMetadata(
                    name="old_plugin",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR,
                    deprecated=True,
                    # no replacement set
                )

            def initialize(self, config=None):
                self._initialized = True

            def execute(self, *args, **kwargs):
                return None

        validator = _make_validator()
        result = validator.validate(_DepPlugin())
        codes = [i.code for i in result.issues]
        assert "DEPRECATED_NO_REPLACEMENT" in codes

    def test_invalid_version_format_produces_warning(self):
        from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

        class _BadVersionPlugin(PluginInterface):
            def _create_metadata(self):
                return PluginMetadata(
                    name="ok_name",
                    version="not_semver",
                    plugin_type=PluginType.PROCESSOR,
                )

            def initialize(self, config=None):
                self._initialized = True

            def execute(self, *args, **kwargs):
                return None

        validator = _make_validator()
        result = validator.validate(_BadVersionPlugin())
        codes = [i.code for i in result.issues]
        assert "INVALID_VERSION_FORMAT" in codes


# ---------------------------------------------------------------------------
# Interface compliance validation
# ---------------------------------------------------------------------------


class TestInterfaceComplianceValidation:
    def test_valid_plugin_passes_interface_compliance(self):
        validator = _make_validator()
        plugin = _make_valid_plugin()
        result = validator.validate(plugin)
        compliance_codes = [i.code for i in result.issues if "MISSING" in i.code or "NOT_PLUGIN" in i.code]
        assert len(compliance_codes) == 0


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


class TestManifestValidation:
    def test_valid_manifest_produces_no_errors(self):
        validator = _make_validator()
        plugin = _make_valid_plugin()
        manifest = _make_manifest()
        result = validator.validate(plugin, manifest)
        manifest_error_codes = [i.code for i in result.issues if "ENTRY_POINT" in i.code or "PLUGIN_TYPE" in i.code]
        assert len(manifest_error_codes) == 0

    def test_empty_entry_point_produces_error(self):
        validator = _make_validator()
        plugin = _make_valid_plugin()
        manifest = _make_manifest(entry_point="")
        result = validator.validate(plugin, manifest)
        codes = [i.code for i in result.issues]
        assert "MISSING_ENTRY_POINT" in codes

    def test_invalid_entry_point_format_produces_error(self):
        validator = _make_validator()
        plugin = _make_valid_plugin()
        # entry_point must contain ":" separator
        manifest = _make_manifest(entry_point="mymodule_without_colon")
        result = validator.validate(plugin, manifest)
        codes = [i.code for i in result.issues]
        assert "INVALID_ENTRY_POINT" in codes

    def test_invalid_plugin_type_produces_error(self):
        validator = _make_validator()
        plugin = _make_valid_plugin()
        manifest = _make_manifest(plugin_type="nonexistent_type")
        result = validator.validate(plugin, manifest)
        codes = [i.code for i in result.issues]
        assert "INVALID_PLUGIN_TYPE" in codes


# ---------------------------------------------------------------------------
# Dependency validation
# ---------------------------------------------------------------------------


class TestDependencyValidation:
    def test_available_dependency_passes(self):
        validator = _make_validator()
        plugin = _make_valid_plugin()
        # "os" is always available as a stdlib module
        manifest = _make_manifest(dependencies=["os"])
        result = validator.validate(plugin, manifest)
        codes = [i.code for i in result.issues]
        assert "MISSING_DEPENDENCY" not in codes

    def test_missing_dependency_produces_error(self):
        validator = _make_validator()
        plugin = _make_valid_plugin()
        manifest = _make_manifest(dependencies=["__nonexistent_package_xyz__"])
        result = validator.validate(plugin, manifest)
        codes = [i.code for i in result.issues]
        assert "MISSING_DEPENDENCY" in codes


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


class TestStrictMode:
    def test_strict_mode_converts_warnings_to_errors(self):
        from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

        class _WarnPlugin(PluginInterface):
            def _create_metadata(self):
                return PluginMetadata(
                    name="BadCamelCase",  # → INVALID_NAME_FORMAT warning
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR,
                )

            def initialize(self, config=None):
                self._initialized = True

            def execute(self, *args, **kwargs):
                return None

        strict_validator = _make_validator(strict=True)
        result = strict_validator.validate(_WarnPlugin())
        assert result.is_valid is False

    def test_non_strict_mode_allows_warnings(self):
        from transformation_portal.plugins.interface import PluginInterface, PluginMetadata, PluginType

        class _WarnPlugin(PluginInterface):
            def _create_metadata(self):
                return PluginMetadata(
                    name="BadCamelCase",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR,
                )

            def initialize(self, config=None):
                self._initialized = True

            def execute(self, *args, **kwargs):
                return None

        lax_validator = _make_validator(strict=False)
        result = lax_validator.validate(_WarnPlugin())
        # Should still be valid despite warning
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# Custom validators
# ---------------------------------------------------------------------------


class TestCustomValidators:
    def test_custom_validator_called(self):
        from transformation_portal.plugins.validator import PluginValidator

        called = []

        def my_validator(plugin, result):
            called.append(True)

        validator = PluginValidator(custom_validators=[my_validator])
        validator.validate(_make_valid_plugin())
        assert len(called) == 1

    def test_failing_custom_validator_adds_warning(self):
        from transformation_portal.plugins.validator import PluginValidator

        def bad_validator(plugin, result):
            raise RuntimeError("exploded")

        validator = PluginValidator(custom_validators=[bad_validator])
        result = validator.validate(_make_valid_plugin())
        codes = [i.code for i in result.issues]
        assert "CUSTOM_VALIDATOR_ERROR" in codes


# ---------------------------------------------------------------------------
# validate_loaded
# ---------------------------------------------------------------------------


class TestValidateLoaded:
    def test_validate_loaded_with_no_plugin_is_invalid(self):
        from transformation_portal.plugins.loader import LoadedPlugin
        from transformation_portal.plugins.validator import PluginValidator

        loaded = LoadedPlugin(
            plugin=None,
            manifest=_make_manifest(name="broken"),
            source_path=__import__("pathlib").Path("/fake"),
            module_name="fake_broken_module",
            load_errors=["ImportError: module not found"],
        )
        validator = PluginValidator()
        result = validator.validate_loaded(loaded)
        assert result.is_valid is False
        codes = [i.code for i in result.issues]
        assert "PLUGIN_NOT_LOADED" in codes


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_validate_plugin_returns_validation_result(self):
        from transformation_portal.plugins.validator import ValidationResult, validate_plugin

        result = validate_plugin(_make_valid_plugin())
        assert isinstance(result, ValidationResult)

    def test_quick_validate_returns_bool(self):
        from transformation_portal.plugins.validator import quick_validate

        assert isinstance(quick_validate(_make_valid_plugin()), bool)

    def test_quick_validate_valid_plugin_returns_true(self):
        from transformation_portal.plugins.validator import quick_validate

        assert quick_validate(_make_valid_plugin()) is True
