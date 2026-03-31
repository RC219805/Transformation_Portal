"""Preset health validation for APEX Research presets.

Validates YAML preset files against registry contracts and governance rules:
- Backend IDs must exist in their respective registries.
- Placeholder strings (NEEDS_VERIFICATION_*, PLACEHOLDER_*) are flagged.
- Obvious hash placeholders (including long all-zero runs) are flagged.
- Missing/unimplemented pipeline stages are explicitly reported.

Produces a ``PresetHealthReport`` JSON artifact for CI and runtime consumption.

See ADR-026 §M1.1 for specification.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Patterns that indicate unresolved placeholders
_PLACEHOLDER_PATTERNS: list[re.Pattern] = [
    re.compile(r"NEEDS_VERIFICATION", re.IGNORECASE),
    re.compile(r"PLACEHOLDER", re.IGNORECASE),
    re.compile(r"TODO_REPLACE", re.IGNORECASE),
    re.compile(r"^0{20,}$"),  # All-zero hashes
]


@dataclass
class HealthIssue:
    """A single validation issue found in a preset.

    Attributes:
        severity: "error" (blocks execution) or "warning" (advisory).
        category: Issue category (e.g., "placeholder", "backend_id", "stage_missing").
        message: Human-readable description of the issue.
        path: Dot-separated path to the offending key in the preset YAML.
    """

    severity: str  # "error" or "warning"
    category: str
    message: str
    path: str = ""


@dataclass
class StageStatus:
    """Status of a pipeline stage as declared in the preset.

    Attributes:
        name: Stage name (e.g., "depth", "segmentation", "materials").
        declared: Whether the stage is declared in the preset.
        backend: Backend ID declared for this stage, if any.
        backend_available: Whether the backend ID is resolvable in the registry.
        skipped_reason: If not None, the stage is explicitly skipped with this reason.
    """

    name: str
    declared: bool = False
    backend: Optional[str] = None
    backend_available: Optional[bool] = None
    skipped_reason: Optional[str] = None


@dataclass
class PresetHealthReport:
    """Result of preset health validation.

    Attributes:
        preset_path: Path to the validated preset file.
        preset_name: Name from the preset metadata.
        issues: List of HealthIssue objects.
        stages: Status of each pipeline stage.
        healthy: True if no errors (warnings are OK).
    """

    preset_path: str
    preset_name: str
    issues: List[HealthIssue] = field(default_factory=list)
    stages: List[StageStatus] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        """True if there are no error-severity issues."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "preset_path": self.preset_path,
            "preset_name": self.preset_name,
            "healthy": self.healthy,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [asdict(i) for i in self.issues],
            "stages": [asdict(s) for s in self.stages],
        }

    def save(self, path: Path) -> None:
        """Write report as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Preset health report written to %s", path)


# ---------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------

# The six ADR-026 pipeline stages and where their backend key lives in YAML
_PIPELINE_STAGES = [
    ("depth", "depth.backend"),
    ("segmentation", "segmentation.backend"),
    ("materials", "materials.backend"),
    ("reconstruction", "reconstruction.backend"),
    ("enhancement", None),  # Enhancement has no backend registry (yet)
    ("validation", "validation.backend"),
]


def _resolve_yaml_path(data: Dict, dotted_path: str) -> Any:
    """Resolve a dot-separated path in a nested dict, returning None if missing."""
    parts = dotted_path.split(".")
    current = data
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _check_placeholders(data: Dict, prefix: str = "") -> List[HealthIssue]:
    """Recursively scan YAML values for placeholder strings."""
    issues: List[HealthIssue] = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            issues.extend(_check_placeholders(value, path))
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                item_path = f"{path}[{idx}]"
                if isinstance(item, dict):
                    issues.extend(_check_placeholders(item, item_path))
                elif isinstance(item, str):
                    for pat in _PLACEHOLDER_PATTERNS:
                        if pat.search(item):
                            issues.append(
                                HealthIssue(
                                    severity="error",
                                    category="placeholder",
                                    message=f"Placeholder value detected: '{item}'",
                                    path=item_path,
                                )
                            )
                            break
        elif isinstance(value, str):
            for pat in _PLACEHOLDER_PATTERNS:
                if pat.search(value):
                    issues.append(
                        HealthIssue(
                            severity="error",
                            category="placeholder",
                            message=f"Placeholder value detected: '{value}'",
                            path=path,
                        )
                    )
                    break
    return issues


def _check_depth_backend_ids(
    data: Dict,
    available_backend_ids: Optional[List[str]] = None,
) -> tuple[List[HealthIssue], Optional[List[str]]]:
    """Validate that depth model names map to registered backends."""
    issues: List[HealthIssue] = []

    # Get available backend IDs from registry if not provided
    if available_backend_ids is None:
        try:
            from transformation_portal.depth.backends.registry import DepthBackendRegistry

            registry = DepthBackendRegistry()
            available_backend_ids = registry.available_backend_ids()
        except Exception as exc:
            logger.warning("Could not load depth backend registry for validation: %s", exc)
            issues.append(
                HealthIssue(
                    severity="warning",
                    category="backend_registry_unavailable",
                    message=("Depth backend registry could not be loaded; " "backend ID validation was skipped."),
                    path="depth",
                )
            )
            return issues, None

    # Check top-level depth backend
    depth_section = data.get("depth", {})
    top_backend = depth_section.get("backend")
    if top_backend and top_backend not in available_backend_ids:
        issues.append(
            HealthIssue(
                severity="error",
                category="backend_id",
                message=(f"Depth backend '{top_backend}' is not registered. " f"Available: {available_backend_ids}"),
                path="depth.backend",
            )
        )

    # Check per-model names in ensemble models list
    models = depth_section.get("models", [])
    for idx, model in enumerate(models):
        if not isinstance(model, dict):
            continue
        name = model.get("name")
        if name and name not in available_backend_ids:
            issues.append(
                HealthIssue(
                    severity="error",
                    category="backend_id",
                    message=(
                        f"Depth model name '{name}' is not registered as a backend. " f"Available: {available_backend_ids}"
                    ),
                    path=f"depth.models[{idx}].name",
                )
            )

    return issues, available_backend_ids


def _check_stages(
    data: Dict,
    available_depth_backend_ids: Optional[List[str]] = None,
) -> List[StageStatus]:
    """Determine the status of each pipeline stage in the preset."""
    statuses: List[StageStatus] = []

    for stage_name, backend_path in _PIPELINE_STAGES:
        section = data.get(stage_name)
        declared = section is not None and isinstance(section, dict)

        backend_id = None
        backend_available = None
        if declared and backend_path:
            backend_id = _resolve_yaml_path(data, backend_path)
            if stage_name == "depth" and backend_id is not None and available_depth_backend_ids is not None:
                backend_available = backend_id in available_depth_backend_ids

        # Determine skip reason for unimplemented stages
        skipped_reason = None
        if not declared:
            skipped_reason = f"Stage '{stage_name}' not declared in preset"

        statuses.append(
            StageStatus(
                name=stage_name,
                declared=declared,
                backend=backend_id,
                backend_available=backend_available,
                skipped_reason=skipped_reason,
            )
        )

    return statuses


def validate_preset(
    preset_path: str | Path,
    *,
    available_depth_backend_ids: Optional[List[str]] = None,
    strict: bool = False,
) -> PresetHealthReport:
    """Validate a preset YAML file and produce a health report.

    Args:
        preset_path: Path to the YAML preset file.
        available_depth_backend_ids: Override list of valid depth backend IDs.
            If None, the depth backend registry is queried at runtime.
        strict: If True, treat warnings as errors.

    Returns:
        PresetHealthReport with all issues and stage statuses.
    """
    preset_path = Path(preset_path)
    if not preset_path.exists():
        report = PresetHealthReport(
            preset_path=str(preset_path),
            preset_name="(unknown)",
        )
        report.issues.append(
            HealthIssue(
                severity="error",
                category="file_missing",
                message=f"Preset file not found: {preset_path}",
            )
        )
        return report

    with open(preset_path, encoding="utf-8") as f:
        # YAML_GOVERNANCE_EXEMPT: diagnostic preset scanner reads YAML for health checks without executing presets.
        data = yaml.safe_load(f) or {}

    preset_name = data.get("name", preset_path.stem)
    report = PresetHealthReport(
        preset_path=str(preset_path),
        preset_name=preset_name,
    )

    # 1. Check for placeholder strings
    report.issues.extend(_check_placeholders(data))

    # 2. Check depth backend IDs against registry
    depth_issues, resolved_depth_backend_ids = _check_depth_backend_ids(
        data,
        available_depth_backend_ids,
    )
    report.issues.extend(depth_issues)

    # 3. Assess pipeline stage status
    report.stages = _check_stages(data, resolved_depth_backend_ids)

    # 4. Strict mode: promote warnings to errors
    if strict:
        for issue in report.issues:
            if issue.severity == "warning":
                issue.severity = "error"

    logger.info(
        "Preset '%s' health: %s (%d errors, %d warnings)",
        preset_name,
        "HEALTHY" if report.healthy else "UNHEALTHY",
        report.error_count,
        report.warning_count,
    )

    return report
