"""Static semantic stage identifiers for canonical execution plans.

This registry is deliberately *not* an executor registry.  Entries contain
only immutable contract metadata; they never contain Python classes,
callables, module paths, commands, or plugin hooks.  Construction of
``StageGraph`` objects remains gated by ADR-051's production vertical slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Tuple


class UnknownStageRegistryIdentifier(ValueError):
    """Raised when a plan names a stage outside the static allowlist."""


class UnknownOutputKind(ValueError):
    """Raised when a plan names an output outside the static allowlist."""


class StageRegistryIdentifier(str, Enum):
    """Stable external identifiers accepted by ``tp.execution.plan.v1``."""

    LUX_PREPROCESS = "tp.stage.lux.preprocess.v1"
    LUX_DEPTH = "tp.stage.lux.depth.v1"
    LUX_MATERIALS_V3 = "tp.stage.lux.materials_v3.v1"
    LUX_PBR = "tp.stage.lux.pbr.v1"
    LUX_V2 = "tp.stage.lux.v2.v1"
    LUX_RECONSTRUCTION = "tp.stage.lux.reconstruction.v1"
    LUX_OUTPUT = "tp.stage.lux.output.v1"


class OutputScope(str, Enum):
    """Stable scope values for logical plan outputs."""

    PER_INPUT = "per_input"
    PER_RUN = "per_run"


class OutputCardinality(str, Enum):
    """Stable cardinality values for logical plan outputs."""

    ONE = "one"
    MANY = "many"


@dataclass(frozen=True)
class OutputKindDefinition:
    """Closed structural semantics for one allowlisted artifact kind."""

    artifact_kind: str
    scope: OutputScope
    cardinality: OutputCardinality


@dataclass(frozen=True)
class ResourceRange:
    """Bounded resource request carried by a semantic stage node."""

    minimum: int
    maximum: int

    def to_payload(self) -> dict[str, int]:
        return {"minimum": self.minimum, "maximum": self.maximum}


@dataclass(frozen=True)
class StageResourceProfile:
    """Conservative declarative limits for a stage family.

    These values constrain plans but do not activate resource enforcement.
    The designated executor must enforce them before an ADR-051 cutover.
    """

    cpu_cores: ResourceRange
    gpu_count: ResourceRange
    memory_mib: ResourceRange
    disk_mib: ResourceRange
    wall_time_seconds: ResourceRange

    def to_payload(self) -> dict[str, dict[str, int]]:
        return {
            "cpu_cores": self.cpu_cores.to_payload(),
            "gpu_count": self.gpu_count.to_payload(),
            "memory_mib": self.memory_mib.to_payload(),
            "disk_mib": self.disk_mib.to_payload(),
            "wall_time_seconds": self.wall_time_seconds.to_payload(),
        }


@dataclass(frozen=True)
class StageDefinition:
    """Closed semantic metadata for one allowlisted stage identifier."""

    identifier: StageRegistryIdentifier
    configuration_schema: str
    allowed_output_kinds: Tuple[str, ...]
    resources: StageResourceProfile


_CPU_ONLY = StageResourceProfile(
    cpu_cores=ResourceRange(1, 64),
    gpu_count=ResourceRange(0, 0),
    memory_mib=ResourceRange(256, 262_144),
    disk_mib=ResourceRange(0, 1_048_576),
    wall_time_seconds=ResourceRange(1, 86_400),
)
_ACCELERATOR_OPTIONAL = StageResourceProfile(
    cpu_cores=ResourceRange(1, 64),
    gpu_count=ResourceRange(0, 8),
    memory_mib=ResourceRange(256, 262_144),
    disk_mib=ResourceRange(0, 1_048_576),
    wall_time_seconds=ResourceRange(1, 86_400),
)


def _output(
    artifact_kind: str,
    *,
    scope: OutputScope = OutputScope.PER_INPUT,
    cardinality: OutputCardinality = OutputCardinality.ONE,
) -> OutputKindDefinition:
    return OutputKindDefinition(
        artifact_kind=artifact_kind,
        scope=scope,
        cardinality=cardinality,
    )


OUTPUT_DEFINITIONS: Mapping[str, OutputKindDefinition] = MappingProxyType(
    {
        definition.artifact_kind: definition
        for definition in (
            _output("preprocessed_image"),
            _output("depth_map"),
            _output("depth_u16_png"),
            _output("depth_metadata_json"),
            _output("depth_float_npy"),
            _output("materials_v3_masks"),
            _output("pbr_maps", cardinality=OutputCardinality.MANY),
            _output("v2_enhanced_image"),
            _output(
                "reconstruction_bundle",
                scope=OutputScope.PER_RUN,
                cardinality=OutputCardinality.MANY,
            ),
            _output("combined_manifest_json"),
            _output("batch_manifest_json", scope=OutputScope.PER_RUN),
            _output("run_card", scope=OutputScope.PER_RUN),
        )
    }
)


STAGE_DEFINITIONS: Mapping[StageRegistryIdentifier, StageDefinition] = MappingProxyType(
    {
        StageRegistryIdentifier.LUX_PREPROCESS: StageDefinition(
            identifier=StageRegistryIdentifier.LUX_PREPROCESS,
            configuration_schema="tp.stage.config.lux.preprocess.v1",
            allowed_output_kinds=("preprocessed_image",),
            resources=_CPU_ONLY,
        ),
        StageRegistryIdentifier.LUX_DEPTH: StageDefinition(
            identifier=StageRegistryIdentifier.LUX_DEPTH,
            configuration_schema="tp.stage.config.lux.depth.v1",
            allowed_output_kinds=(
                "depth_map",
                "depth_u16_png",
                "depth_metadata_json",
                "depth_float_npy",
            ),
            resources=_ACCELERATOR_OPTIONAL,
        ),
        StageRegistryIdentifier.LUX_MATERIALS_V3: StageDefinition(
            identifier=StageRegistryIdentifier.LUX_MATERIALS_V3,
            configuration_schema="tp.stage.config.lux.materials_v3.v1",
            allowed_output_kinds=("materials_v3_masks",),
            resources=_ACCELERATOR_OPTIONAL,
        ),
        StageRegistryIdentifier.LUX_PBR: StageDefinition(
            identifier=StageRegistryIdentifier.LUX_PBR,
            configuration_schema="tp.stage.config.lux.pbr.v1",
            allowed_output_kinds=("pbr_maps",),
            resources=_CPU_ONLY,
        ),
        StageRegistryIdentifier.LUX_V2: StageDefinition(
            identifier=StageRegistryIdentifier.LUX_V2,
            configuration_schema="tp.stage.config.lux.v2.v1",
            allowed_output_kinds=("v2_enhanced_image",),
            resources=_ACCELERATOR_OPTIONAL,
        ),
        StageRegistryIdentifier.LUX_RECONSTRUCTION: StageDefinition(
            identifier=StageRegistryIdentifier.LUX_RECONSTRUCTION,
            configuration_schema="tp.stage.config.lux.reconstruction.v1",
            allowed_output_kinds=("reconstruction_bundle",),
            resources=_ACCELERATOR_OPTIONAL,
        ),
        StageRegistryIdentifier.LUX_OUTPUT: StageDefinition(
            identifier=StageRegistryIdentifier.LUX_OUTPUT,
            configuration_schema="tp.stage.config.lux.output.v1",
            allowed_output_kinds=(
                "combined_manifest_json",
                "batch_manifest_json",
                "run_card",
            ),
            resources=_CPU_ONLY,
        ),
    }
)


def stage_registry_identifiers() -> tuple[str, ...]:
    """Return the complete stable allowlist in declaration order."""

    return tuple(identifier.value for identifier in StageRegistryIdentifier)


def get_stage_definition(identifier: str | StageRegistryIdentifier) -> StageDefinition:
    """Resolve an exact registry identifier without aliases or imports."""

    try:
        typed_identifier = (
            identifier if isinstance(identifier, StageRegistryIdentifier) else StageRegistryIdentifier(identifier)
        )
    except (TypeError, ValueError) as exc:
        raise UnknownStageRegistryIdentifier(f"Unknown stage registry identifier: {identifier!r}") from exc
    return STAGE_DEFINITIONS[typed_identifier]


def get_output_definition(artifact_kind: str) -> OutputKindDefinition:
    """Resolve exact output structure without aliases or dynamic handlers."""

    try:
        return OUTPUT_DEFINITIONS[artifact_kind]
    except (KeyError, TypeError) as exc:
        raise UnknownOutputKind(f"Unknown output artifact kind: {artifact_kind!r}") from exc


__all__ = [
    "OUTPUT_DEFINITIONS",
    "OutputCardinality",
    "OutputKindDefinition",
    "OutputScope",
    "ResourceRange",
    "STAGE_DEFINITIONS",
    "StageDefinition",
    "StageRegistryIdentifier",
    "StageResourceProfile",
    "UnknownStageRegistryIdentifier",
    "UnknownOutputKind",
    "get_output_definition",
    "get_stage_definition",
    "stage_registry_identifiers",
]
