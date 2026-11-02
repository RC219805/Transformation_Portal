"""Runtime-bound native SDR resource qualification, never an admission bypass."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import math
import os
import platform
import re
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformation_portal.ingest.canonical_json import dumps_json
from transformation_portal.photographic_delivery.native_resource_contract import RESOURCE_CONTRACT
from transformation_portal.photographic_delivery.native_resource_inputs import NativeRasterEvidence

PROFILE_ID = "tp.photo.native_sdr_local.v1"
GIB, MIB = 1024**3, 1024**2
MEMORY_BYTES = 6 * GIB
ACCEPTANCE_PEAK_BYTES = 4 * GIB  # An acceptance ceiling, never a claimed observation.
MAX_PIXELS, MAX_REGIONS = 30_000_000, 4
MAX_STATE_BYTES, MAX_STATE_NODES, MAX_STATE_DEPTH = 8 * MIB, 65536, 16
_FIXED_BYTES = 1444 * MIB + GIB
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_PACKAGES = (
    "numpy",
    "Pillow",
    "tifffile",
    "scipy",
    "psutil",
    "fastapi",
    "starlette",
    "pydantic",
    "SQLAlchemy",
    "asyncpg",
    "packaging",
)
_EXTENSIONS = ("numpy._core._multiarray_umath", "PIL._imaging", "PIL._imagingcms", "scipy.ndimage._nd_image", "zlib")
_SOURCE_PATHS = (
    "pyproject.toml",
    "requirements/base.in",
    "app.py",
    "src/transformation_portal/photographic_delivery/native_resources.py",
    "src/transformation_portal/photographic_delivery/native_resource_inputs.py",
    "src/transformation_portal/photographic_delivery/materials.py",
    "src/transformation_portal/photographic_delivery/__init__.py",
    "src/transformation_portal/photographic_delivery/models.py",
    "src/transformation_portal/photographic_delivery/native.py",
    "src/transformation_portal/photographic_delivery/presentation.py",
    "src/transformation_portal/core/image_artifact.py",
    "src/transformation_portal/core/photographic_export.py",
    "src/transformation_portal/core/security/frontdoor_identity.py",
    "src/transformation_portal/core/security/tenant.py",
    "src/transformation_portal/core/security/serialization.py",
    "src/transformation_portal/core/security/fs_guard.py",
    "src/transformation_portal/core/security/path_safety.py",
    "src/transformation_portal/core/execution_plan.py",
    "src/transformation_portal/ingest/canonical_json.py",
    "src/transformation_portal/materials_v4/engine.py",
    "src/transformation_portal/materials_v4/contracts.py",
    "src/transformation_portal/materials_v4/operations.py",
    "src/transformation_portal/lux_depth_v4/photography.py",
    "src/transformation_portal/lux_depth_v4/color_preparation.py",
    "src/transformation_portal/lux_depth_v4/io.py",
    "src/transformation_portal/lux_depth_v5/preview.py",
    "src/transformation_portal/lux_depth_v6/color.py",
    "src/transformation_portal/orchestrator/photographic_review_service.py",
    "src/transformation_portal/orchestrator/photographic_admission.py",
    "src/transformation_portal/orchestrator/photographic_review_access.py",
    "src/transformation_portal/orchestrator/photographic_review_store.py",
    "src/transformation_portal/orchestrator/photographic_scene_catalogue.py",
    "src/transformation_portal/orchestrator/dispatch.py",
    "src/transformation_portal/orchestrator/storage/__init__.py",
    "src/transformation_portal/orchestrator/storage/base.py",
    "src/transformation_portal/orchestrator/storage/postgres.py",
    "src/transformation_portal/orchestrator/storage/operational.py",
    "src/transformation_portal/orchestrator/models.py",
    "src/transformation_portal/orchestrator/artifact_store/base.py",
    "src/transformation_portal/orchestrator/artifact_store/__init__.py",
    "src/transformation_portal/orchestrator/artifact_store/local.py",
    "src/transformation_portal/orchestrator/artifact_store/_filesystem.py",
    "src/transformation_portal/orchestrator/artifact_store/generation.py",
    "src/transformation_portal/api/routes/photographic_review.py",
    "src/transformation_portal/api/routes/__init__.py",
    "src/transformation_portal/api/v1/photographic_review.py",
    "src/transformation_portal/api/v1/envelopes.py",
    "src/transformation_portal/api/v1/__init__.py",
    "src/transformation_portal/portal/photographic_review_access.py",
    "src/transformation_portal/portal/job_artifacts.py",
    "src/transformation_portal/portal/path_security.py",
)


def _canonical(value: Any) -> bytes:
    return dumps_json(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    sources = {}
    for relative in _SOURCE_PATHS:
        expected = _file_hash(root / relative)
        if not relative.endswith(".py"):
            sources[relative] = expected
            continue
        name = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
        name = name.removesuffix(".__init__")
        loaded = sys.modules.get(name)
        if loaded is not None:
            origin = getattr(loaded, "__file__", None)
            if not isinstance(origin, str) or _file_hash(Path(origin)) != expected:
                raise ValueError(f"Loaded native resource source differs: {name}")
        sources[relative] = expected
    return sources


def _requirement_line(line: str) -> str:
    """Remove plain requirements-file comments while preserving quoted markers."""
    quote = None
    for index, character in enumerate(line):
        if character in {"'", '"'}:
            if quote == character:
                quote = None
            elif quote is None:
                quote = character
        elif character == "#" and quote is None and (index == 0 or line[index - 1].isspace()):
            return line[:index].strip()
    return line.strip()


def _active_runtime_requirements(root: Path | None = None, environment: dict[str, str] | None = None) -> tuple[Any, ...]:
    # The parser is optional for the ordinary runtime; its absence denies only this profile.
    from packaging.markers import UndefinedEnvironmentName
    from packaging.requirements import Requirement

    root = root or Path(__file__).resolve().parents[3]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream).get("project")
    dependencies = project.get("dependencies") if type(project) is dict else None
    if type(dependencies) is not list or not dependencies or any(type(value) is not str for value in dependencies):
        raise ValueError("Native resources require declared core dependency requirements")
    lines = [_requirement_line(line) for line in (root / "requirements/base.in").read_text(encoding="utf-8").splitlines()]
    requirements = tuple(Requirement(value) for value in (*dependencies, *(line for line in lines if line)))
    try:
        return tuple(value for value in requirements if value.marker is None or value.marker.evaluate(environment))
    except UndefinedEnvironmentName as error:
        raise ValueError("Native resources do not support the declared marker context") from error


def _validate_runtime_packages(packages: dict[str, str], requirements: tuple[Any, ...]) -> None:
    from packaging.utils import canonicalize_name
    from packaging.version import Version

    versions = {canonicalize_name(name): Version(value) for name, value in packages.items()}
    for requirement in requirements:
        version = versions.get(canonicalize_name(requirement.name))
        if requirement.url is not None or version is None or version not in requirement.specifier:
            raise ValueError(f"Native resource package does not satisfy declared core requirement: {requirement}")


def _runtime_packages() -> dict[str, str]:
    from packaging.utils import canonicalize_name

    requirements = _active_runtime_requirements()
    names = {canonicalize_name(name): name for name in _PACKAGES}
    for requirement in requirements:
        names.setdefault(canonicalize_name(requirement.name), requirement.name)
    packages = {name: importlib.metadata.version(name) for _, name in sorted(names.items())}
    _validate_runtime_packages(packages, requirements)
    return packages


def _runtime_identity() -> dict[str, Any] | None:
    if (platform.python_implementation(), platform.python_version(), platform.system(), platform.machine()) != (
        "CPython",
        "3.12.14",
        "Darwin",
        "arm64",
    ) or any(os.environ.get(name) != "1" for name in _THREAD_ENV):
        return None
    packages = _runtime_packages()
    extensions = {}
    for name in _EXTENSIONS:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or not Path(spec.origin).is_file():
            return None
        extensions[name] = _file_hash(Path(spec.origin))
    return {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
        "version": platform.version(),
        "executable_sha256": _file_hash(Path(sys.executable).resolve()),
        "packages": packages,
        "extensions": extensions,
        "thread_environment": {name: os.environ[name] for name in _THREAD_ENV},
    }


def _limits() -> dict[str, int]:
    return {
        "pixels": MAX_PIXELS,
        "regions": MAX_REGIONS,
        "memory_bytes": MEMORY_BYTES,
        "acceptance_peak_bytes": ACCEPTANCE_PEAK_BYTES,
        "state_heap_bytes": MAX_STATE_BYTES,
        "state_nodes": MAX_STATE_NODES,
        "state_depth": MAX_STATE_DEPTH,
        "fixed_overhead_bytes": _FIXED_BYTES,
    }


def capture_contract() -> dict[str, Any]:
    """Explicitly capture a review template; never install or approve it."""
    runtime = _runtime_identity()
    if runtime is None:
        raise ValueError("Native resources require the pinned runtime and one-thread environment")
    return {
        "schema": "tp.photo.native_resource_contract.v1",
        "profile_id": PROFILE_ID,
        "limits": _limits(),
        "runtime": runtime,
        "sources": _source_identity(),
    }


def profile_identity() -> str:
    """Identify the reviewed contract independently of current runtime matching."""
    return hashlib.sha256(_canonical(RESOURCE_CONTRACT)).hexdigest()


def runtime_profile_identity() -> str | None:
    """Missing or drifted runtime/source authority always selects fallback."""
    try:
        return profile_identity() if RESOURCE_CONTRACT and RESOURCE_CONTRACT == capture_contract() else None
    except (ValueError, TypeError, OSError, ImportError, AttributeError):
        return None


def _bounded_state(state: dict) -> bool:
    if type(state) is not dict:
        return False
    stack: list[tuple[Any, int, bool]] = [(state, 0, False)]
    active: set[int] = set()
    size = nodes = 0
    while stack:
        value, depth, leaving = stack.pop()
        if leaving:
            active.remove(id(value))
            continue
        kind = type(value)
        if kind not in {dict, list, str, int, float, bool, type(None)}:
            return False
        nodes += 1
        size += sys.getsizeof(value)
        if nodes > MAX_STATE_NODES or size > MAX_STATE_BYTES or depth > MAX_STATE_DEPTH:
            return False
        if kind in {dict, list}:
            children = len(value) * (2 if kind is dict else 1)
            if id(value) in active or nodes + len(stack) + children > MAX_STATE_NODES:
                return False
            active.add(id(value))
            stack.append((value, depth, True))
            if kind is dict:
                for key, item in value.items():
                    if type(key) is not str:
                        return False
                    stack.extend(((key, depth + 1, False), (item, depth + 1, False)))
            else:
                stack.extend((item, depth + 1, False) for item in value)
        elif kind not in {str, int, float, bool, type(None)} or (kind is float and not math.isfinite(value)):
            return False
    return True


def metadata_heap_is_bounded(state: dict) -> bool:
    """Share the closed graph ceiling with authenticated provider metadata reads."""
    return _bounded_state(state)


def _valid_raster(evidence: NativeRasterEvidence) -> bool:
    if type(evidence) is not NativeRasterEvidence:
        return False
    hashes = ("source_sha256", "icc_sha256", "profile_sha256", "layout_sha256", "header_sha256")
    numbers = (
        "width",
        "height",
        "source_width",
        "source_height",
        "source_bit_depth",
        "orientation",
        "size_bytes",
        "header_bytes",
        "tag_count",
        "strip_count",
        "raster_bytes",
    )
    if not all(_digest(getattr(evidence, name)) for name in hashes) or not all(
        type(getattr(evidence, name)) is int for name in numbers
    ):
        return False
    if type(evidence.alpha) is not bool or evidence.source_bit_depth not in {8, 16} or evidence.orientation not in range(1, 9):
        return False
    if evidence.container not in {"classic_tiff", "bigtiff"} or evidence.byte_order not in {"little", "big"}:
        return False
    pixels = evidence.source_width * evidence.source_height
    if not (0 < evidence.source_width <= 65536 and 0 < evidence.source_height <= 65536 and pixels <= MAX_PIXELS):
        return False
    shape = (
        (evidence.source_width, evidence.source_height)
        if evidence.orientation < 5
        else (evidence.source_height, evidence.source_width)
    )
    prefix, entry, trailer = (8, 12, 6) if evidence.container == "classic_tiff" else (16, 20, 16)
    from transformation_portal.lux_depth_v4.color_preparation import output_srgb_icc

    return (
        (evidence.width, evidence.height) == shape
        and 10 + int(evidence.alpha) <= evidence.tag_count <= 32
        and prefix + entry * evidence.tag_count + trailer <= evidence.header_bytes <= 65536
        and 1 <= evidence.strip_count <= min(64, evidence.source_height)
        and evidence.raster_bytes == pixels * (3 + int(evidence.alpha)) * (evidence.source_bit_depth // 8)
        and evidence.size_bytes == evidence.raster_bytes + evidence.header_bytes <= 512 * MIB
        and evidence.profile_sha256 == hashlib.sha256(output_srgb_icc()).hexdigest()
    )


def analytical_memory_bytes(evidence: NativeRasterEvidence, regions: int) -> int:
    """Max simultaneous phase allocations plus frozen, conservative allowances."""
    if not _valid_raster(evidence) or type(regions) is not int or not 1 <= regions <= MAX_REGIONS:
        raise ValueError("Analytical native resources require closed raster evidence and one to four regions")
    pixels, alpha = evidence.width * evidence.height, int(evidence.alpha)
    master, masks = (12 + 4 * alpha) * pixels, 4 * regions * pixels
    tiff, preview, descriptor = (6 + 2 * alpha) * pixels + 65536, 1600**2 * 4 + MIB, 256 * 1024
    small = min(pixels, 1600**2)
    phases = (
        63 * pixels,
        master + masks + 4 * pixels,
        master + masks + 12 * pixels,
        (48 + 8 * alpha) * pixels + masks,
        (24 + 8 * alpha) * pixels + masks + math.ceil(tiff * 25 / 8),
        2 * tiff + (57 + 6 * alpha) * pixels + masks,
        tiff + master + masks + 12 * pixels + 64 * small + 3 * preview,
        tiff + master + masks + 12 * pixels + 2 * preview,
        tiff + masks + 4 * pixels + 2 * preview + descriptor,
    )
    return max(phases) + _FIXED_BYTES


@dataclass(frozen=True, slots=True)
class NativeMaterialsQualification:
    profile_id: str
    profile_identity: str
    source_sha256: str
    raster_identity: str
    provider_scope_sha256: str
    regions: int
    width: int
    height: int
    alpha: bool
    source_bit_depth: int
    analytical_memory_bytes: int
    memory_bytes: int = MEMORY_BYTES

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def qualify_native_resources(
    evidence: NativeRasterEvidence, *, regions: int, provider_scope_sha256: str, state: dict, response_owned: bool
) -> NativeMaterialsQualification | None:
    """Select a fixed reservation only; the caller still acquires normal admission."""
    if (
        response_owned is not True
        or type(regions) is not int
        or not 1 <= regions <= MAX_REGIONS
        or not _digest(provider_scope_sha256)
    ):
        return None
    if not metadata_heap_is_bounded(state):
        return None
    identity = runtime_profile_identity()
    if identity is None or not _valid_raster(evidence):
        return None
    analytical = analytical_memory_bytes(evidence, regions)
    if analytical > MEMORY_BYTES:
        return None
    return NativeMaterialsQualification(
        PROFILE_ID,
        identity,
        evidence.source_sha256,
        evidence.content_hash(),
        provider_scope_sha256,
        regions,
        evidence.width,
        evidence.height,
        evidence.alpha,
        evidence.source_bit_depth,
        analytical,
    )
