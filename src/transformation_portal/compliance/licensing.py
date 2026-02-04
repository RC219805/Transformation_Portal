"""Licensing compliance enforcement and validation.

Provides decorators and validators to ensure non-commercial models
(e.g., Depth Anything V3.1 with CC BY-NC 4.0) are used only with
explicit authorization.
"""

import functools
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

import yaml


class LicenseRestrictionError(Exception):
    """Raised when a licensing restriction is violated.

    This exception is raised when attempting to use a non-commercial model
    without explicit opt-in via `non_commercial_ok=True`.
    """

    pass


F = TypeVar("F", bound=Callable[..., Any])


def require_non_commercial(reason: str = "") -> Callable[[F], F]:
    """Decorator enforcing non-commercial usage authorization.

    This decorator ensures that functions using non-commercial models
    (e.g., DA3 1.1) only execute when the caller has explicitly set
    `non_commercial_ok=True` in their configuration.

    Args:
        reason: Human-readable explanation of the licensing restriction
                (e.g., "DA3 1.1 uses CC BY-NC 4.0 models")

    Raises:
        LicenseRestrictionError: If the configuration does not have
                                `non_commercial_ok=True`

    Example:
        ```python
        @require_non_commercial(reason="DA3 1.1 uses CC BY-NC 4.0 models")
        def load_da3_1_1_preset(config: EnhanceConfig):
            ...
        ```
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract config from first positional arg or 'config' kwarg
            config = None
            if args and hasattr(args[0], "non_commercial_ok"):
                config = args[0]
            elif "config" in kwargs and hasattr(kwargs["config"], "non_commercial_ok"):
                config = kwargs["config"]

            if config is None:
                raise TypeError(
                    f"@require_non_commercial decorator on {func.__name__} "
                    "expects first argument or 'config' kwarg to have "
                    "'non_commercial_ok' attribute"
                )

            if not config.non_commercial_ok:
                raise LicenseRestrictionError(
                    f"Function '{func.__name__}' requires "
                    "non_commercial_ok=True in your EnhanceConfig.\n"
                    f"Reason: {reason}\n"
                    "This model uses CC BY-NC 4.0 (non-commercial research only).\n"
                    "For commercial applications, use a commercially-licensed "
                    "depth model instead."
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def validate_non_commercial_preset(preset_dict: Dict[str, Any]) -> bool:
    """Validate that non-commercial presets have required licensing markers.

    Checks if a preset dictionary (from YAML) contains a known non-commercial
    model. If so, ensures the preset has explicit `license_restriction: non_commercial`
    marker.

    Args:
        preset_dict: Dictionary loaded from a preset YAML file
                    (should have 'model' key with 'hf_id')

    Returns:
        True if preset is valid (either commercial or properly marked non-commercial)

    Raises:
        LicenseRestrictionError: If preset uses non-commercial model
                                without proper marker
    """
    model = preset_dict.get("model", {})
    hf_id = model.get("hf_id", "")

    # Check for known non-commercial models
    non_commercial_identifiers = [
        "DA3-Large-1.1",
        "DA3-Base-1.1",
        "DA3-Small-1.1",
        "DA3NESTED-GIANT-LARGE-1.1",
    ]

    is_non_commercial_model = any(identifier in hf_id for identifier in non_commercial_identifiers)

    if is_non_commercial_model:
        # Verify marker exists
        license_restriction = preset_dict.get("license_restriction")
        if license_restriction != "non_commercial":
            raise LicenseRestrictionError(
                f"Preset uses non-commercial model (hf_id={hf_id}) "
                "but lacks license_restriction='non_commercial' marker.\n"
                "Please add this marker to acknowledge CC BY-NC 4.0 restrictions."
            )

    return True


def load_and_validate_preset(preset_path: Path) -> Dict[str, Any]:
    """Load a preset YAML file and validate licensing compliance.

    Args:
        preset_path: Path to preset YAML file

    Returns:
        Loaded preset dictionary

    Raises:
        FileNotFoundError: If preset file does not exist
        yaml.YAMLError: If YAML is malformed
        LicenseRestrictionError: If licensing markers are missing
    """
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_path}")

    with open(preset_path) as f:
        preset = yaml.safe_load(f)

    validate_non_commercial_preset(preset)
    return preset
