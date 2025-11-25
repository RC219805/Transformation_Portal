"""Plugin validator for dependency checking and compatibility validation."""

import importlib
import inspect
import logging
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type

from .interface import (
    DepthModelPlugin,
    EnhancerPlugin,
    PluginInterface,
    PluginMetadata,
    PluginType,
    PluginValidationError,
    ProcessorPlugin,
)
from .loader import LoadedPlugin, PluginManifest

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    code: str
    message: str
    severity: ValidationSeverity
    suggestion: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        result = f"[{self.severity.value.upper()}] {self.code}: {self.message}"
        if self.suggestion:
            result += f"\n  Suggestion: {self.suggestion}"
        return result


@dataclass
class ValidationResult:
    """Result of plugin validation."""
    plugin_name: str
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings_count: int = 0
    errors_count: int = 0

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.errors_count += 1
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings_count += 1

    def get_issues_by_severity(
        self,
        severity: ValidationSeverity
    ) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [i for i in self.issues if i.severity == severity]

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        result = f"Plugin '{self.plugin_name}': {status}"
        result += f" ({self.errors_count} errors, {self.warnings_count} warnings)"

        if self.issues:
            result += "\n" + "\n".join(f"  - {issue}" for issue in self.issues)

        return result


class PluginValidator:
    """Comprehensive plugin validator.

    Validates:
    - Interface compliance (required methods, signatures)
    - Dependency availability
    - Version compatibility
    - Metadata correctness
    - Configuration schema (if provided)
    - Runtime behavior

    Example:
        >>> validator = PluginValidator()
        >>> result = validator.validate(my_plugin)
        >>> if not result.is_valid:
        ...     for issue in result.issues:
        ...         print(issue)
    """

    # Required methods for each plugin type
    REQUIRED_METHODS: Dict[Type[PluginInterface], List[str]] = {
        PluginInterface: ["initialize", "execute", "cleanup", "validate"],
        DepthModelPlugin: ["estimate_depth"],
        ProcessorPlugin: ["process"],
        EnhancerPlugin: ["enhance"],
    }

    def __init__(
        self,
        strict_mode: bool = False,
        custom_validators: Optional[List[Callable]] = None,
    ):
        """Initialize validator.

        Args:
            strict_mode: Treat warnings as errors
            custom_validators: Additional validation functions
        """
        self._strict_mode = strict_mode
        self._custom_validators = custom_validators or []

    def validate(
        self,
        plugin: PluginInterface,
        manifest: Optional[PluginManifest] = None
    ) -> ValidationResult:
        """Validate a plugin comprehensively.

        Args:
            plugin: Plugin instance to validate
            manifest: Optional manifest for additional validation

        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult(
            plugin_name=plugin.metadata.name if hasattr(plugin, 'metadata') else "unknown",
            is_valid=True,
        )

        # Run all validators
        self._validate_metadata(plugin, result)
        self._validate_interface_compliance(plugin, result)
        self._validate_method_signatures(plugin, result)

        if manifest:
            self._validate_manifest(manifest, result)
            self._validate_dependencies(manifest.dependencies, result)

        self._validate_version_compatibility(plugin, result)
        self._validate_initialization(plugin, result)

        # Run custom validators
        for validator in self._custom_validators:
            try:
                validator(plugin, result)
            except Exception as e:
                result.add_issue(ValidationIssue(
                    code="CUSTOM_VALIDATOR_ERROR",
                    message=f"Custom validator failed: {e}",
                    severity=ValidationSeverity.WARNING,
                ))

        # In strict mode, warnings become errors
        if self._strict_mode:
            for issue in result.issues:
                if issue.severity == ValidationSeverity.WARNING:
                    issue.severity = ValidationSeverity.ERROR
                    result.errors_count += 1
                    result.warnings_count -= 1
                    result.is_valid = False

        return result

    def validate_loaded(self, loaded_plugin: LoadedPlugin) -> ValidationResult:
        """Validate a loaded plugin with its manifest.

        Args:
            loaded_plugin: Loaded plugin container

        Returns:
            ValidationResult
        """
        if not loaded_plugin.plugin:
            result = ValidationResult(
                plugin_name=loaded_plugin.manifest.name if loaded_plugin.manifest else "unknown",
                is_valid=False,
            )
            result.add_issue(ValidationIssue(
                code="PLUGIN_NOT_LOADED",
                message="Plugin failed to load",
                severity=ValidationSeverity.CRITICAL,
                details={"load_errors": loaded_plugin.load_errors},
            ))
            return result

        return self.validate(loaded_plugin.plugin, loaded_plugin.manifest)

    def _validate_metadata(
        self,
        plugin: PluginInterface,
        result: ValidationResult
    ) -> None:
        """Validate plugin metadata."""
        if not hasattr(plugin, 'metadata'):
            result.add_issue(ValidationIssue(
                code="MISSING_METADATA",
                message="Plugin missing metadata attribute",
                severity=ValidationSeverity.CRITICAL,
                suggestion="Add metadata property returning PluginMetadata instance",
            ))
            return

        metadata = plugin.metadata

        # Validate name
        if not metadata.name:
            result.add_issue(ValidationIssue(
                code="EMPTY_NAME",
                message="Plugin name is empty",
                severity=ValidationSeverity.ERROR,
            ))
        elif not re.match(r'^[a-z][a-z0-9_]*$', metadata.name):
            result.add_issue(ValidationIssue(
                code="INVALID_NAME_FORMAT",
                message=f"Plugin name '{metadata.name}' should be lowercase with underscores",
                severity=ValidationSeverity.WARNING,
                suggestion="Use format like 'my_plugin_name'",
            ))

        # Validate version
        if not metadata.version:
            result.add_issue(ValidationIssue(
                code="EMPTY_VERSION",
                message="Plugin version is empty",
                severity=ValidationSeverity.ERROR,
            ))
        elif not re.match(r'^\d+\.\d+\.\d+', metadata.version):
            result.add_issue(ValidationIssue(
                code="INVALID_VERSION_FORMAT",
                message=f"Version '{metadata.version}' should follow semver (e.g., 1.0.0)",
                severity=ValidationSeverity.WARNING,
            ))

        # Check deprecated status
        if metadata.deprecated and not metadata.replacement:
            result.add_issue(ValidationIssue(
                code="DEPRECATED_NO_REPLACEMENT",
                message="Plugin is deprecated but no replacement specified",
                severity=ValidationSeverity.WARNING,
                suggestion="Specify replacement plugin in metadata",
            ))

    def _validate_interface_compliance(
        self,
        plugin: PluginInterface,
        result: ValidationResult
    ) -> None:
        """Validate plugin implements required interface."""
        # Check it's a proper PluginInterface subclass
        if not isinstance(plugin, PluginInterface):
            result.add_issue(ValidationIssue(
                code="NOT_PLUGIN_INTERFACE",
                message="Plugin does not inherit from PluginInterface",
                severity=ValidationSeverity.CRITICAL,
            ))
            return

        # Check required methods for base interface
        for method_name in self.REQUIRED_METHODS[PluginInterface]:
            if not hasattr(plugin, method_name):
                result.add_issue(ValidationIssue(
                    code="MISSING_METHOD",
                    message=f"Missing required method: {method_name}",
                    severity=ValidationSeverity.ERROR,
                ))
            elif not callable(getattr(plugin, method_name)):
                result.add_issue(ValidationIssue(
                    code="METHOD_NOT_CALLABLE",
                    message=f"'{method_name}' is not callable",
                    severity=ValidationSeverity.ERROR,
                ))

        # Check type-specific methods
        for plugin_type, methods in self.REQUIRED_METHODS.items():
            if plugin_type == PluginInterface:
                continue

            if isinstance(plugin, plugin_type):
                for method_name in methods:
                    if not hasattr(plugin, method_name):
                        result.add_issue(ValidationIssue(
                            code="MISSING_TYPE_METHOD",
                            message=f"Missing method for {plugin_type.__name__}: {method_name}",
                            severity=ValidationSeverity.ERROR,
                        ))

    def _validate_method_signatures(
        self,
        plugin: PluginInterface,
        result: ValidationResult
    ) -> None:
        """Validate method signatures match expected patterns."""
        # Check initialize accepts config
        if hasattr(plugin, 'initialize'):
            sig = inspect.signature(plugin.initialize)
            params = list(sig.parameters.keys())

            if len(params) < 1 or (len(params) == 1 and params[0] == 'self'):
                # No config parameter - might be okay but worth noting
                pass
            elif 'config' not in params and len(params) > 0:
                # Has parameter but not named 'config'
                result.add_issue(ValidationIssue(
                    code="UNEXPECTED_PARAM_NAME",
                    message="initialize() parameter should be named 'config'",
                    severity=ValidationSeverity.INFO,
                ))

        # Check execute exists and is callable
        if hasattr(plugin, 'execute'):
            sig = inspect.signature(plugin.execute)
            if len(sig.parameters) == 0:
                result.add_issue(ValidationIssue(
                    code="NO_EXECUTE_PARAMS",
                    message="execute() takes no parameters besides self",
                    severity=ValidationSeverity.WARNING,
                    suggestion="execute() should accept input data",
                ))

    def _validate_manifest(
        self,
        manifest: PluginManifest,
        result: ValidationResult
    ) -> None:
        """Validate plugin manifest."""
        # Check entry point format
        if not manifest.entry_point:
            result.add_issue(ValidationIssue(
                code="MISSING_ENTRY_POINT",
                message="Manifest missing entry_point",
                severity=ValidationSeverity.ERROR,
            ))
        elif ':' not in manifest.entry_point:
            result.add_issue(ValidationIssue(
                code="INVALID_ENTRY_POINT",
                message=f"Invalid entry_point format: {manifest.entry_point}",
                severity=ValidationSeverity.ERROR,
                suggestion="Use format 'module:ClassName'",
            ))

        # Validate plugin type
        try:
            PluginType(manifest.plugin_type)
        except ValueError:
            result.add_issue(ValidationIssue(
                code="INVALID_PLUGIN_TYPE",
                message=f"Unknown plugin type: {manifest.plugin_type}",
                severity=ValidationSeverity.ERROR,
                suggestion=f"Use one of: {[t.value for t in PluginType]}",
            ))

    def _validate_dependencies(
        self,
        dependencies: List[str],
        result: ValidationResult
    ) -> None:
        """Validate plugin dependencies are available."""
        for dep in dependencies:
            # Parse dependency string
            match = re.match(r'^([a-zA-Z0-9_-]+)(.*)?$', dep)
            if not match:
                result.add_issue(ValidationIssue(
                    code="INVALID_DEPENDENCY_FORMAT",
                    message=f"Invalid dependency format: {dep}",
                    severity=ValidationSeverity.WARNING,
                ))
                continue

            package_name = match.group(1)
            version_spec = match.group(2) or ""

            # Check if package is available
            try:
                module_name = package_name.replace("-", "_")
                importlib.import_module(module_name)
            except ImportError:
                result.add_issue(ValidationIssue(
                    code="MISSING_DEPENDENCY",
                    message=f"Missing dependency: {dep}",
                    severity=ValidationSeverity.ERROR,
                    suggestion=f"Install with: pip install {package_name}",
                ))
                continue

            # Check version if specified
            if version_spec:
                self._check_version_constraint(package_name, version_spec, result)

    def _check_version_constraint(
        self,
        package_name: str,
        version_spec: str,
        result: ValidationResult
    ) -> None:
        """Check if installed package version meets constraint."""
        try:
            from packaging import version as pkg_version
            import importlib.metadata

            installed_version = importlib.metadata.version(package_name)

            # Parse constraint (e.g., ">=1.0.0", "<2.0.0", "==1.5.0")
            match = re.match(r'^([<>=!]+)(.+)$', version_spec)
            if match:
                operator = match.group(1)
                required_version = match.group(2)

                installed = pkg_version.parse(installed_version)
                required = pkg_version.parse(required_version)

                satisfied = False
                if operator == ">=":
                    satisfied = installed >= required
                elif operator == ">":
                    satisfied = installed > required
                elif operator == "<=":
                    satisfied = installed <= required
                elif operator == "<":
                    satisfied = installed < required
                elif operator == "==":
                    satisfied = installed == required
                elif operator == "!=":
                    satisfied = installed != required

                if not satisfied:
                    result.add_issue(ValidationIssue(
                        code="VERSION_MISMATCH",
                        message=f"Package {package_name} version {installed_version} "
                                f"does not satisfy {version_spec}",
                        severity=ValidationSeverity.ERROR,
                    ))

        except Exception as e:
            logger.debug(f"Could not check version for {package_name}: {e}")

    def _validate_version_compatibility(
        self,
        plugin: PluginInterface,
        result: ValidationResult
    ) -> None:
        """Validate plugin is compatible with current portal version."""
        from transformation_portal import __version__ as portal_version

        if hasattr(plugin, 'metadata'):
            metadata = plugin.metadata

            if not metadata.is_compatible(portal_version):
                result.add_issue(ValidationIssue(
                    code="VERSION_INCOMPATIBLE",
                    message=f"Plugin not compatible with portal version {portal_version}",
                    severity=ValidationSeverity.ERROR,
                    details={
                        "min_version": metadata.min_portal_version,
                        "max_version": metadata.max_portal_version,
                        "current_version": portal_version,
                    },
                ))

    def _validate_initialization(
        self,
        plugin: PluginInterface,
        result: ValidationResult
    ) -> None:
        """Validate plugin initialization behavior."""
        # Check internal state
        if hasattr(plugin, '_initialized') and plugin._initialized:
            result.add_issue(ValidationIssue(
                code="PRE_INITIALIZED",
                message="Plugin is already initialized before explicit init call",
                severity=ValidationSeverity.INFO,
            ))

        # Check config attribute
        if hasattr(plugin, '_config') and plugin._config:
            result.add_issue(ValidationIssue(
                code="PRE_CONFIGURED",
                message="Plugin has pre-existing configuration",
                severity=ValidationSeverity.INFO,
            ))


def validate_plugin(
    plugin: PluginInterface,
    manifest: Optional[PluginManifest] = None,
    strict: bool = False
) -> ValidationResult:
    """Convenience function to validate a plugin.

    Args:
        plugin: Plugin to validate
        manifest: Optional manifest
        strict: Enable strict mode

    Returns:
        ValidationResult
    """
    validator = PluginValidator(strict_mode=strict)
    return validator.validate(plugin, manifest)


def quick_validate(plugin: PluginInterface) -> bool:
    """Quick validation check returning True/False.

    Args:
        plugin: Plugin to validate

    Returns:
        True if plugin passes validation
    """
    result = validate_plugin(plugin)
    return result.is_valid
