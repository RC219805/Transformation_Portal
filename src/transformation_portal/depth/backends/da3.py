"""Depth Anything V3 backend adapter for unified backend registry.

Wraps existing DA3InferenceEngine to provide DepthBackend interface with
consistent contract and license governance.

Supports two execution modes:
- In-process when DA3 dependencies are installed in the active environment
- Subprocess mode via a dedicated Python executable for isolated
  depth-anything-3 environments

See ADR-019 for architectural rationale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from PIL import Image

from ...core.da3_runtime import REPO_LOCAL_DA3_PYTHON, find_repo_root
from ...core.ml_dependency_health import (
    _installed_version,
    detect_transformers_torch_version_issue,
    ensure_dependency_importable,
)
from ...core.platform_matrix import CURRENT_PLATFORM
from ...lux_depth_v3.config_resolver import (
    _compat_model_variant_for_resolved_key,
    preset_model_key_for_selection,
    with_typed_preset_provenance,
)
from ...lux_depth_v3.execution_plan_adapter import LuxExecutionPlanAuthorityError
from ...lux_depth_v3.model_resolution import (
    ModelRequest,
    direct_model_contract,
    refresh_direct_model_acknowledgement,
    resolve_model_contract,
    restore_stale_direct_model_selection,
    validate_authoritative_model_contract,
)
from .protocol import DepthResult, LicenseType

if TYPE_CHECKING:
    from ...lux_depth_v3.config import EnhanceConfig, ModelVariant
    from ...lux_depth_v3.depth_cache_runtime import PreparedDepthCacheRuntimeEvidence
    from ...lux_depth_v3.execution_lifecycle import BackendCandidateAuthority
    from ...lux_depth_v3.inference import DA3InferenceEngine

logger = logging.getLogger(__name__)

DA3_RECOMMENDED_VENV = REPO_LOCAL_DA3_PYTHON
DA3_SETUP_SCRIPT = "./scripts/setup/install_da3_runtime.sh"
DEFAULT_DA3_SUBPROCESS_TIMEOUT_SECONDS = 900
_MAX_DA3_DEPTH_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
_MAX_NPY_HEADER_BYTES = 64 * 1024

_DEPENDENCY_FAILURE_MARKERS = (
    "modulenotfounderror",
    "no module named",
    "importerror",
    "cannot import name",
    "package not installed",
    "required for",
)
_STARTUP_FAILURE_MARKERS = (
    "omp: error",
    "libomp",
    "kmp_duplicate_lib_ok",
    "segmentation fault",
    "abort trap",
    "dyld:",
)


def _subprocess_text(value: str | bytes | None) -> str:
    """Normalize timeout output while retaining UTF-8 diagnostics."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _load_verified_worker_depth(path: Path, artifact: Any) -> np.ndarray:
    """Digest, validate, and load one NPY from the same open regular file."""

    expected_keys = {"sha256", "size_bytes", "shape", "dtype", "fortran_order"}
    if not isinstance(artifact, dict) or set(artifact) != expected_keys:
        raise ValueError("DA3 worker depth artifact has an unknown shape")
    expected_sha256 = artifact.get("sha256")
    expected_size = artifact.get("size_bytes")
    expected_shape = artifact.get("shape")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or type(expected_size) is not int
        or expected_size <= 0
        or expected_size > _MAX_DA3_DEPTH_ARTIFACT_BYTES
        or not isinstance(expected_shape, list)
        or len(expected_shape) != 2
        or any(type(value) is not int or value <= 0 for value in expected_shape)
        or artifact.get("dtype") != "float32"
        or type(artifact.get("fortran_order")) is not bool
    ):
        raise ValueError("DA3 worker depth artifact is incomplete or unbounded")
    expected_data_bytes = expected_shape[0] * expected_shape[1] * np.dtype(np.float32).itemsize
    if expected_size < expected_data_bytes or expected_size - expected_data_bytes > _MAX_NPY_HEADER_BYTES:
        raise ValueError("DA3 worker depth artifact has an invalid NPY header or payload size")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise ValueError("DA3 worker depth artifact is not the expected regular file")
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("DA3 worker depth artifact digest mismatch")
        handle.seek(0)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            actual_shape, actual_fortran_order, actual_dtype = np.lib.format.read_array_header_1_0(
                handle,
                max_header_size=_MAX_NPY_HEADER_BYTES,
            )
        elif version == (2, 0):
            actual_shape, actual_fortran_order, actual_dtype = np.lib.format.read_array_header_2_0(
                handle,
                max_header_size=_MAX_NPY_HEADER_BYTES,
            )
        else:
            raise ValueError("DA3 worker depth artifact uses an unsupported NPY version")
        actual_header_bytes = handle.tell()
        if (
            list(actual_shape) != expected_shape
            or actual_dtype != np.dtype(np.float32)
            or actual_fortran_order is not artifact["fortran_order"]
            or actual_header_bytes + expected_data_bytes != expected_size
        ):
            raise ValueError("DA3 worker depth artifact header disagrees with its bounded echo")
        handle.seek(0)
        depth_map = np.load(
            handle,
            allow_pickle=False,
            max_header_size=_MAX_NPY_HEADER_BYTES,
        )
        if handle.tell() != expected_size:
            raise ValueError("DA3 worker depth artifact contains trailing or unread bytes")
        after = os.fstat(handle.fileno())
    before_projection = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_projection = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_projection != after_projection:
        raise ValueError("DA3 worker depth artifact changed while loading")
    if depth_map.dtype != np.dtype(np.float32) or list(depth_map.shape) != expected_shape:
        raise ValueError("DA3 worker depth artifact disagrees with its dtype or shape echo")
    return depth_map


def _model_id_for_variant(model_variant: "ModelVariant") -> str:
    """Return the configured Hugging Face model ID for a variant."""
    return str(model_variant.value.huggingface_id)


def _model_requires_custom_da3_library(model_variant: "ModelVariant") -> bool:
    """Return whether the selected model requires the depth-anything-3 package."""
    model_id = _model_id_for_variant(model_variant).lower()
    return model_id.startswith("depth-anything/da3") or "da3nested" in model_id


def _model_requires_custom_da3_library_for_model_id(model_id: str) -> bool:
    """Return whether a model ID requires the depth-anything-3 package."""
    normalized = str(model_id).lower()
    return normalized.startswith("depth-anything/da3") or "da3nested" in normalized


def _infer_source_depth_units(metadata: dict[str, Any]) -> str:
    """Infer source depth unit semantics from backend metadata."""
    resolved_model_id = str(metadata.get("resolved_model_id", "")).lower()
    requested_model_id = str(metadata.get("requested_model_id", "")).lower()
    model_hint = resolved_model_id or requested_model_id

    depth_tokens = ("metric", "da3nested", "nested-giant")
    if any(token in model_hint for token in depth_tokens):
        return "meters"
    return "relative"


def _normalize_relative_depth_metadata(
    metadata: dict[str, Any],
    *,
    runner_mode: str,
    python_executable: str,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize relative-depth provenance across local and subprocess execution."""
    normalized_metadata = dict(metadata)
    source_depth_units = _infer_source_depth_units(normalized_metadata)
    normalized_metadata["source_depth_units"] = source_depth_units
    normalized_metadata["output_depth_units"] = "relative"
    normalized_metadata["output_normalization"] = "minmax_0_1_per_image"
    normalized_metadata["runner"] = {
        "mode": runner_mode,
        "python_executable": python_executable,
    }

    warnings: list[str] = []
    if source_depth_units == "meters":
        warnings.append("source metric depth normalized to relative [0,1] for unified pipeline output")
    return normalized_metadata, warnings


def _classify_subprocess_failure(
    *,
    phase: str,
    stdout: str,
    stderr: str,
    timed_out: bool = False,
    protocol_error: bool = False,
) -> tuple[str, str]:
    """Classify subprocess failures into stable operator-facing categories."""
    if timed_out:
        return "timeout", f"DA3 worker timed out during {phase}"
    if protocol_error:
        return "protocol_error", "DA3 worker did not produce the expected output contract"

    combined = "\n".join(part for part in (stdout, stderr) if part).lower()
    if any(marker in combined for marker in _DEPENDENCY_FAILURE_MARKERS):
        return "dependency_missing", "isolated DA3 environment is missing one or more Python dependencies"
    if any(marker in combined for marker in _STARTUP_FAILURE_MARKERS):
        return "startup_failed", "isolated DA3 environment failed during native library startup"
    if phase == "availability_check":
        return "startup_failed", "DA3 worker exited before reporting a ready state"
    return "inference_failed", "DA3 worker exited non-zero during inference"


class DA3Backend:
    """Depth Anything V3 backend adapter implementing DepthBackend protocol."""

    name = "da3"
    license_type = LicenseType.MODEL_DEPENDENT
    requires_checkpoint = False
    WORKER_MODULE = "transformation_portal.depth.backends.da3_worker"

    def __init__(
        self,
        config: Optional["EnhanceConfig"] = None,
        *,
        candidate_authority: Optional["BackendCandidateAuthority"] = None,
        canonical_plan_bytes: Optional[bytes] = None,
    ):
        """Initialize DA3 backend."""
        if (candidate_authority is None) != (canonical_plan_bytes is None):
            raise ValueError("candidate_authority and canonical_plan_bytes must be provided together")
        if candidate_authority is not None and config is None:
            raise ValueError("Canonical DA3 authority requires its projected runtime config")
        self._config = config
        self._candidate_authority = candidate_authority
        self._canonical_plan_bytes = canonical_plan_bytes
        self._engine: Optional[DA3InferenceEngine] = None
        self._device = self._resolve_device(config, candidate_authority)
        # P0-1 (issue #2065): consume the authoritative single-resolution
        # contract when the invocation carries one, instead of re-resolving.
        # Re-resolution here is the seam where the legacy model_variant
        # compatibility mapping could silently turn a commercial selection
        # (da3_metric) back into the research model (da3_research).
        authoritative_contract = candidate_authority.resolved_model_contract if candidate_authority is not None else None
        if candidate_authority is not None:
            if candidate_authority.backend_id != self.name or candidate_authority.model_contract is None:
                raise ValueError("Canonical DA3 authority does not select a DA3 model contract")
            if candidate_authority.model_contract.backend_id != self.name:
                raise ValueError("Canonical DA3 model contract names a different backend")
            if authoritative_contract is None:
                raise ValueError("Canonical DA3 authority lacks a re-anchored runtime model contract")
        elif config is not None:
            restore_stale_direct_model_selection(config)
            refresh_direct_model_acknowledgement(config, stacklevel=3)
            invocation = getattr(config, "resolved_invocation", None)
            if invocation is not None:
                authoritative_contract = getattr(invocation, "resolved_model", None)
            if authoritative_contract is None:
                authoritative_contract = direct_model_contract(config)
        if authoritative_contract is not None:
            self._resolved_model_contract = validate_authoritative_model_contract(
                authoritative_contract,
                non_commercial_ok=bool(getattr(config, "non_commercial_ok", False)),
            )
        else:
            preset_model_key = preset_model_key_for_selection(config) if config is not None else None
            self._resolved_model_contract = resolve_model_contract(
                ModelRequest(
                    model_key=(getattr(config, "model_key", None) or preset_model_key if config is not None else None),
                    raw_model_id=getattr(config, "raw_model_id", None) if config is not None else None,
                    # Selection must come from what the config actually
                    # carries — NOT self._model_variant, whose METRIC_LARGE
                    # fallback is a display/compat default. Fabricating an
                    # explicit legacy variant here would turn "no selection"
                    # into the research model instead of DEFAULT_MODEL_KEY
                    # (repair 1.2, #2066).
                    model_variant=getattr(config, "model_variant", None) if config is not None else None,
                    use_coreml_backend=bool(getattr(config, "use_coreml_backend", False)) if config is not None else False,
                    non_commercial_ok=bool(getattr(config, "non_commercial_ok", False)) if config is not None else False,
                    enforce_license=config is not None,
                )
            )
            if config is not None:
                self._resolved_model_contract = with_typed_preset_provenance(
                    config,
                    self._resolved_model_contract,
                    preset_model_key,
                )
        self._model_variant = self._resolve_model_variant(config, authoritative_contract=self._resolved_model_contract)
        self._model_id = self._resolved_model_contract.spec.repo_id
        self._repo_root = self._find_repo_root()
        self._repo_src = self._repo_root / "src" if self._repo_root is not None else None
        self._python_executable = self._resolve_python_executable(config)
        self._subprocess_timeout_seconds = self._resolve_subprocess_timeout_seconds(config)
        self._subprocess_available_checked = False
        self._prepared_cache_runtime_identity: Optional["PreparedDepthCacheRuntimeEvidence"] = None
        self._prepared_cache_runtime_verification_token: Optional[dict[str, Any]] = None
        self._prepared_cache_runtime_verification_token_sha256: Optional[str] = None
        self._prepared_worker_runtime_identity_sha256: Optional[str] = None
        self._prepared_parent_runtime_identity: Any = None
        self._cache_runtime_authority_disabled = False

    def _clear_prepared_cache_runtime_identity(self) -> None:
        self._prepared_cache_runtime_identity = None
        self._prepared_cache_runtime_verification_token = None
        self._prepared_cache_runtime_verification_token_sha256 = None
        self._prepared_worker_runtime_identity_sha256 = None
        self._prepared_parent_runtime_identity = None

    def _disable_cache_runtime_authority(self, reason: str) -> None:
        """Require a process restart after retained runtime materialization drifts."""

        logger.warning("DA3 cache runtime authority disabled until restart: %s", reason)
        self._cache_runtime_authority_disabled = True
        self._clear_prepared_cache_runtime_identity()

    def _find_repo_root(self) -> Optional[Path]:
        """Find repository root by walking parent directories when in a checkout."""
        return find_repo_root(Path(__file__))

    def _worker_cwd(self) -> Path:
        """Choose a stable subprocess working directory."""
        if self._repo_root is not None:
            return self._repo_root
        return Path.cwd()

    def _resolve_device(
        self,
        config: Optional["EnhanceConfig"],
        candidate_authority: Optional["BackendCandidateAuthority"] = None,
    ) -> str:
        """Resolve device from config or auto-detect."""
        if candidate_authority is not None:
            carried = str(candidate_authority.device or "").strip().lower()
            if carried and carried != "auto":
                return carried
        if config is not None:
            device = getattr(config, "depth_device", None)
            if device:
                return device
        return "cpu"

    def _resolve_model_variant(
        self,
        config: Optional["EnhanceConfig"],
        *,
        authoritative_contract: Any = None,
    ) -> "ModelVariant":
        """Resolve model variant from config."""
        if authoritative_contract is not None:
            return _compat_model_variant_for_resolved_key(authoritative_contract.canonical_key)
        if config is not None:
            variant = getattr(config, "model_variant", None)
            if variant:
                return variant

        from ...lux_depth_v3.config import ModelVariant

        return ModelVariant.METRIC_LARGE

    def _resolve_python_executable(
        self,
        config: Optional["EnhanceConfig"],
    ) -> Optional[str]:
        """Resolve optional subprocess Python executable for DA3."""
        candidate: Optional[str] = None
        if config is not None:
            configured = getattr(config, "da3_python_executable", None)
            if configured is not None:
                try:
                    configured_path = os.fspath(configured).strip()
                except TypeError:
                    configured_path = ""
                if configured_path:
                    candidate = configured_path

        if candidate is None and config is not None and getattr(config, "execution_plan_authority", None) is not None:
            # Execution-complete plans freeze absence as well as presence.
            # A later environment mutation cannot add a subprocess runtime.
            return None

        if candidate is None:
            env_candidate = os.environ.get("TRANSFORMATION_PORTAL_DA3_PYTHON")
            if env_candidate and env_candidate.strip():
                candidate = env_candidate.strip()

        if candidate is None:
            return None

        has_separator = os.sep in candidate or (os.altsep is not None and os.altsep in candidate)
        if candidate.startswith(".") or has_separator:
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                base_dir = self._repo_root or Path.cwd()
                path = base_dir / path
            if not path.exists():
                raise FileNotFoundError(f"DA3 Python executable not found: {path}")
            return str(path.absolute())

        resolved = shutil.which(candidate)
        if resolved is None:
            raise FileNotFoundError(f"DA3 Python executable not found on PATH: {candidate}")
        return resolved

    def _uses_subprocess(self) -> bool:
        """Return whether DA3 should execute in a dedicated subprocess."""
        return self._python_executable is not None

    def _resolve_subprocess_timeout_seconds(
        self,
        config: Optional["EnhanceConfig"],
    ) -> int:
        """Resolve subprocess timeout for DA3 worker execution."""
        candidate: Any = None
        if config is not None:
            candidate = getattr(config, "da3_subprocess_timeout_seconds", None)

        if candidate in (None, ""):
            env_candidate = os.environ.get("TRANSFORMATION_PORTAL_DA3_TIMEOUT_SECONDS")
            if env_candidate and env_candidate.strip():
                candidate = env_candidate.strip()

        if candidate in (None, ""):
            return DEFAULT_DA3_SUBPROCESS_TIMEOUT_SECONDS

        try:
            timeout_seconds = int(candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError("DA3 subprocess timeout must be a positive integer number of seconds") from exc

        if timeout_seconds <= 0:
            raise ValueError("DA3 subprocess timeout must be greater than zero")
        return timeout_seconds

    def _build_worker_env(self) -> dict[str, str]:
        """Build a deterministic import environment for the subprocess worker."""
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("PYTHON") and key.upper() not in {"VIRTUAL_ENV", "__PYVENV_LAUNCHER__"}
        }
        env.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1",
            }
        )
        if self._repo_src is not None and self._repo_src.exists():
            env["PYTHONPATH"] = str(self._repo_src)
        if "MPLCONFIGDIR" not in env:
            if self._repo_root is not None:
                mpl_config_dir = self._repo_root / ".runtime" / "mplconfig"
            else:
                mpl_config_dir = Path(tempfile.gettempdir()) / "transformation_portal_mplconfig"
            mpl_config_dir.mkdir(parents=True, exist_ok=True)
            env["MPLCONFIGDIR"] = str(mpl_config_dir)
        if CURRENT_PLATFORM is not None and CURRENT_PLATFORM.is_macos:
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        return env

    def _build_worker_command(self, *args: str) -> list[str]:
        """Build the subprocess worker command."""
        if self._python_executable is None:
            raise RuntimeError("DA3 subprocess requested without a Python executable")
        return [
            self._python_executable,
            "-m",
            self.WORKER_MODULE,
            *args,
        ]

    def _canonical_worker_args(self) -> list[str]:
        """Return the closed argv carrier for canonical subprocess mode."""

        authority = self._candidate_authority
        if authority is None:
            return []
        args = [
            "--execution-plan-stdin",
            "--candidate-id",
            authority.candidate_id,
        ]
        if authority.constituent_backend_id is not None:
            args.extend(["--model-backend-id", authority.constituent_backend_id])
        return args

    def _canonical_worker_input(self) -> Optional[str]:
        """Return exact canonical JSON text for the subprocess stdin pipe."""

        if self._candidate_authority is None:
            return None
        if type(self._canonical_plan_bytes) is not bytes:
            raise RuntimeError("Canonical DA3 execution is missing immutable plan bytes")
        return self._canonical_plan_bytes.decode("utf-8", errors="strict")

    def _effective_device(self, override: Optional[str]) -> str:
        """Reject runtime overrides that disagree with carried authority."""

        if override is None:
            return self._device
        normalized = str(override).strip().lower()
        if self._candidate_authority is not None and normalized != self._device:
            raise ValueError(f"DA3 device override {normalized!r} disagrees with carried authority {self._device!r}")
        return normalized

    def _verify_worker_authority_echo(self, payload: Any) -> Optional[str]:
        """Verify worker authority before loading or accepting its array."""

        authority = self._candidate_authority
        if authority is not None:
            expected = {
                "plan_fingerprint_sha256": authority.plan_fingerprint_sha256,
                "candidate_id": authority.candidate_id,
                "model_backend_id": authority.constituent_backend_id,
                "executed_backend_id": self.name,
            }
            if not isinstance(payload, dict) or payload.get("execution_authority") != expected:
                raise LuxExecutionPlanAuthorityError("DA3 worker execution-authority echo does not match the carried plan")

        prepared = getattr(self, "_prepared_cache_runtime_identity", None)
        if prepared is None:
            return None
        expected_runtime_identity = prepared.runtime_identity_sha256
        if not isinstance(payload, dict) or payload.get("runtime_identity_sha256") != expected_runtime_identity:
            raise LuxExecutionPlanAuthorityError("DA3 worker runtime-identity echo does not match cache preparation")
        return expected_runtime_identity

    def _apple_coreml_opt_in_enabled(self) -> bool:
        """Return whether this backend is configured to prefer Apple CoreML."""
        if self._candidate_authority is not None:
            contract = self._candidate_authority.model_contract
            return bool(contract is not None and contract.model.accelerator_kind == "coreml")
        return bool(
            self._config is not None
            and getattr(self._config, "use_coreml_backend", False)
            and CURRENT_PLATFORM is not None
            and CURRENT_PLATFORM.is_apple_silicon
        )

    def _non_commercial_opt_in_enabled(self) -> bool:
        """Return whether the caller acknowledged non-commercial registry selections."""
        return bool(self._config is not None and getattr(self._config, "non_commercial_ok", False))

    def _cache_device_tag(self) -> str:
        """Return the cache key device/runtime tag for this backend config."""
        if self._apple_coreml_opt_in_enabled():
            return f"{self._device}_coremlopt"
        return self._device

    def _ensure_local_package_available(self) -> None:
        """Ensure depth-anything-3 is installed for DA3 nested models."""
        if not _model_requires_custom_da3_library_for_model_id(self._model_id):
            return

        try:
            from depth_anything_3.api import DepthAnything3  # noqa: F401
        except ImportError:
            try:
                from depth_anything_3 import DepthAnything3  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "depth_anything_3 package not installed in the active environment.\n\n"
                    "Preferred repo-local setup:\n"
                    f"  1. Run {DA3_SETUP_SCRIPT}\n"
                    f"  2. Use --da3-python {DA3_RECOMMENDED_VENV}\n"
                    f"  3. Or set TRANSFORMATION_PORTAL_DA3_PYTHON={DA3_RECOMMENDED_VENV}\n\n"
                    "Legacy in-process install:\n"
                    "  pip install -e /path/to/Depth-Anything-3 --no-deps"
                ) from exc

    def _ensure_subprocess_available(self) -> None:
        """Ensure the dedicated DA3 subprocess environment is usable."""
        if self._subprocess_available_checked:
            return

        if self._candidate_authority is not None:
            command = self._build_worker_command("--check", *self._canonical_worker_args())
        else:
            command = self._build_worker_command(
                "--check",
                "--model-variant",
                self._model_variant.name,
                "--model-key",
                self._resolved_model_contract.canonical_key,
                "--device",
                self._device,
            )
            if self._resolved_model_contract.revision:
                command.extend(["--model-revision", self._resolved_model_contract.revision])
            if self._non_commercial_opt_in_enabled():
                command.append("--non-commercial-ok")
            if self._apple_coreml_opt_in_enabled():
                command.append("--use-coreml")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=self._worker_cwd(),
                env=self._build_worker_env(),
                check=False,
                timeout=self._subprocess_timeout_seconds,
                input=self._canonical_worker_input(),
            )
        except FileNotFoundError as exc:
            raise ImportError(
                self._build_subprocess_failure_message(
                    title="DA3 subprocess environment is not ready.",
                    category="executable_not_found",
                    summary="configured DA3 Python executable could not be launched",
                    python_executable=self._python_executable or "(unset)",
                    command=command,
                    stdout="",
                    stderr=str(exc),
                )
            ) from exc
        except OSError as exc:
            raise ImportError(
                self._build_subprocess_failure_message(
                    title="DA3 subprocess environment is not ready.",
                    category="startup_failed",
                    summary="configured DA3 Python executable could not be launched",
                    python_executable=self._python_executable or "(unset)",
                    command=command,
                    stdout="",
                    stderr=str(exc),
                )
            ) from exc
        except subprocess.TimeoutExpired as exc:
            timeout_stdout = _subprocess_text(exc.stdout)
            timeout_stderr = _subprocess_text(exc.stderr)
            category, summary = _classify_subprocess_failure(
                phase="availability_check",
                stdout=timeout_stdout,
                stderr=timeout_stderr,
                timed_out=True,
            )
            raise ImportError(
                self._build_subprocess_failure_message(
                    title="DA3 subprocess environment is not ready.",
                    category=category,
                    summary=summary,
                    python_executable=self._python_executable or "(unset)",
                    command=command,
                    stdout=timeout_stdout,
                    stderr=timeout_stderr,
                )
            ) from exc
        if result.returncode != 0:
            category, summary = _classify_subprocess_failure(
                phase="availability_check",
                stdout=result.stdout,
                stderr=result.stderr,
            )
            raise ImportError(
                self._build_subprocess_failure_message(
                    title="DA3 subprocess environment is not ready.",
                    category=category,
                    summary=summary,
                    python_executable=self._python_executable or "(unset)",
                    command=command,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                )
            )
        self._subprocess_available_checked = True

    @staticmethod
    def _format_subprocess_output(stdout: str, stderr: str) -> str:
        """Format subprocess output for actionable error messages."""
        stdout_clean = stdout.strip()
        stderr_clean = stderr.strip()
        if stdout_clean and stderr_clean:
            return f"stdout:\n{stdout_clean}\n\nstderr:\n{stderr_clean}"
        if stderr_clean:
            return stderr_clean
        if stdout_clean:
            return stdout_clean
        return "(no output)"

    @classmethod
    def _build_subprocess_failure_message(
        cls,
        *,
        title: str,
        category: str,
        summary: str,
        python_executable: str,
        command: list[str],
        stdout: str,
        stderr: str,
        returncode: int | None = None,
    ) -> str:
        """Build a stable, reviewable subprocess failure message."""
        lines = [
            title,
            "",
            f"Failure category: {category}",
            f"Failure summary: {summary}",
            f"Python: {python_executable}",
            f"Command: {' '.join(command)}",
        ]
        if returncode is not None:
            lines.append(f"Return code: {returncode}")
        lines.extend(
            [
                "Output:",
                cls._format_subprocess_output(stdout, stderr),
            ]
        )
        return "\n".join(lines)

    def ensure_available(self) -> None:
        """Ensure DA3 dependencies are available."""
        if self._uses_subprocess():
            self._ensure_subprocess_available()
            return

        transformers_version = _installed_version("transformers")
        if transformers_version is None:
            raise ImportError(
                "transformers package not installed.\n\n"
                "Install with:\n"
                "  pip install transformers\n\n"
                "See: https://huggingface.co/docs/transformers"
            )

        torch_version = _installed_version("torch")
        if torch_version is None:
            raise ImportError(
                "torch package not installed.\n\n"
                "Install with:\n"
                "  pip install torch\n\n"
                "See: https://pytorch.org/get-started/locally/"
            )

        ensure_dependency_importable("transformers")
        ensure_dependency_importable("torch")
        self._ensure_local_package_available()

        runtime_issue = detect_transformers_torch_version_issue(torch_version, transformers_version)
        if runtime_issue:
            raise ImportError(runtime_issue)

        logger.debug("DA3 backend dependencies available")

    def prepare_cache_runtime_identity(
        self,
        *,
        execution_plan: Optional[Any] = None,
        candidate_id: Optional[str] = None,
        canonical_plan_bytes: Optional[bytes] = None,
    ) -> Optional["PreparedDepthCacheRuntimeEvidence"]:
        """Prepare isolated-worker evidence before a depth-cache lookup.

        This is an optional capability.  Missing local model bytes, a missing
        governed dependency lock, unsupported in-process execution, or any
        malformed worker report returns ``None`` and therefore cannot authorize
        either a cache read or write.  Carried-plan mismatches remain authority
        errors.
        """

        if self._candidate_authority is None:
            return None
        if self._cache_runtime_authority_disabled:
            return None

        from ...lux_depth_v3.depth_cache_runtime import PreparedDepthCacheRuntimeEvidence
        from ...lux_depth_v3.execution_lifecycle import consume_lux_worker_execution_plan
        from .da3_runtime_identity import (
            DA3RuntimeIdentityEvidence,
            bind_parent_output_dependency_identity,
            build_prepared_cache_runtime_evidence,
            da3_cache_runtime_governance_identity,
            load_da3_worker_runtime_handshake,
            merge_runtime_verification_entries,
            prepare_parent_output_runtime_identity,
            runtime_verification_token_sha256,
            verify_parent_output_runtime_identity,
            verify_runtime_verification_token,
        )

        carried_bytes = self._canonical_plan_bytes
        if type(carried_bytes) is not bytes:
            raise LuxExecutionPlanAuthorityError("DA3 cache runtime preparation is missing carried canonical plan bytes")
        if canonical_plan_bytes is not None and (
            type(canonical_plan_bytes) is not bytes or canonical_plan_bytes != carried_bytes
        ):
            raise LuxExecutionPlanAuthorityError("DA3 cache runtime preparation received different canonical plan bytes")
        if candidate_id is not None and candidate_id != self._candidate_authority.candidate_id:
            raise LuxExecutionPlanAuthorityError("DA3 cache runtime preparation received a different candidate")
        carried_plan = consume_lux_worker_execution_plan(carried_bytes)
        if execution_plan is not None:
            try:
                supplied_bytes = execution_plan.to_canonical_json().encode("utf-8")
            except (AttributeError, TypeError, ValueError) as exc:
                raise LuxExecutionPlanAuthorityError(
                    "DA3 cache runtime preparation received an invalid execution plan"
                ) from exc
            if supplied_bytes != carried_bytes:
                raise LuxExecutionPlanAuthorityError("DA3 cache runtime preparation received a different execution plan")
            consume_lux_worker_execution_plan(supplied_bytes)

        # Production stays cheap and fail-closed until the checked-in contract
        # points at a real repo-confined exact dependency lock whose bytes match
        # the declared digest.  In-process DA3 cannot provide a worker echo.
        governance_identity = (
            da3_cache_runtime_governance_identity(self._python_executable)
            if self._uses_subprocess() and self._python_executable is not None
            else None
        )
        if governance_identity is None:
            self._clear_prepared_cache_runtime_identity()
            return None

        prepared = self._prepared_cache_runtime_identity
        verification_token = self._prepared_cache_runtime_verification_token
        verification_token_sha256 = self._prepared_cache_runtime_verification_token_sha256
        worker_runtime_sha256 = self._prepared_worker_runtime_identity_sha256
        parent_runtime_identity = self._prepared_parent_runtime_identity
        if prepared is not None:
            expected_backend = {"cpu": "pytorch_cpu", "mps": "pytorch_mps"}.get(self._device)
            retained_is_valid = (
                verification_token is not None
                and verification_token_sha256 is not None
                and worker_runtime_sha256 is not None
                and parent_runtime_identity is not None
                and expected_backend is not None
                and verify_runtime_verification_token(
                    verification_token,
                    expected_token_sha256=verification_token_sha256,
                    expected_worker_runtime_identity_sha256=worker_runtime_sha256,
                    expected_prepared_runtime_identity_sha256=prepared.runtime_identity_sha256,
                    expected_requested_device=self._device,
                    expected_actual_device=self._device,
                    expected_executed_backend=expected_backend,
                )
                and verify_parent_output_runtime_identity(parent_runtime_identity)
            )
            if retained_is_valid:
                return prepared
            self._disable_cache_runtime_authority("retained runtime evidence changed after preparation")
            return None

        self._clear_prepared_cache_runtime_identity()

        with tempfile.TemporaryDirectory(prefix="tp_da3_identity_") as tmpdir:
            output_path = Path(tmpdir) / "runtime_identity.json"
            command = self._build_worker_command(
                "--prepare-runtime-identity",
                "--output-runtime-identity",
                str(output_path),
                *self._canonical_worker_args(),
            )
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    cwd=self._worker_cwd(),
                    env=self._build_worker_env(),
                    check=True,
                    timeout=self._subprocess_timeout_seconds,
                    input=self._canonical_worker_input(),
                )
                payload = load_da3_worker_runtime_handshake(output_path)
            except (FileNotFoundError, OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as exc:
                logger.warning("DA3 cache runtime preparation is non-authorizing: %s", exc)
                return None

        expected_keys = {
            "schema",
            "runtime_evidence",
            "prepared_cache_runtime",
            "runtime_identity_sha256",
            "runtime_verification_token",
            "runtime_verification_token_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            logger.warning("DA3 cache runtime preparation returned an unknown report shape")
            return None
        if payload.get("schema") != "tp.da3.worker-runtime-handshake.v1":
            logger.warning("DA3 cache runtime preparation returned an unsupported schema")
            return None

        try:
            evidence = DA3RuntimeIdentityEvidence.from_mapping(payload["runtime_evidence"])
        except (TypeError, ValueError) as exc:
            logger.warning("DA3 cache runtime evidence is non-authorizing: %s", exc)
            return None
        if not evidence.cacheable:
            reasons = evidence.to_mapping().get("incomplete_reasons", [])
            logger.info("DA3 cache runtime identity unavailable: %s", ",".join(reasons))
            return None

        verification_token = payload.get("runtime_verification_token")
        verification_token_sha256 = payload.get("runtime_verification_token_sha256")
        worker_runtime_sha256 = evidence.runtime_identity_sha256
        if (
            not isinstance(verification_token, dict)
            or not isinstance(verification_token_sha256, str)
            or worker_runtime_sha256 is None
            or not verify_runtime_verification_token(
                verification_token,
                expected_token_sha256=verification_token_sha256,
                expected_worker_runtime_identity_sha256=worker_runtime_sha256,
            )
        ):
            logger.warning("DA3 cache runtime preparation returned a stale or invalid verification token")
            return None

        evidence_payload = evidence.to_mapping()
        backend_identity = evidence_payload["backend_identity"]
        expected_contract = self._resolved_model_contract
        detailed_evidence = evidence_payload["evidence"]
        interpreter_evidence = detailed_evidence["interpreter"]
        platform_evidence = detailed_evidence["platform"]
        runtime_baseline = governance_identity.runtime_baseline
        if (
            backend_identity.get("model_canonical_key") != expected_contract.canonical_key
            or backend_identity.get("model_repo_id") != expected_contract.spec.repo_id
            or backend_identity.get("model_lock_revision") != expected_contract.revision
            or backend_identity.get("requested_device") != self._device
            or backend_identity.get("actual_device") != self._device
            or evidence_payload.get("governance_contract_sha256") != governance_identity.governance_contract_sha256
            or evidence_payload.get("dependency_lock_sha256") != governance_identity.dependency_lock_sha256
            or detailed_evidence.get("source_revision") != governance_identity.source_revision
            or interpreter_evidence.get("runtime_authority_sha256") != governance_identity.runtime_authority_sha256
            or interpreter_evidence.get("implementation") != runtime_baseline["implementation"]
            or interpreter_evidence.get("version") != runtime_baseline["python_version"]
            or interpreter_evidence.get("executable_sha256") != runtime_baseline["executable_sha256"]
            or interpreter_evidence.get("executable_size_bytes") != runtime_baseline["executable_size_bytes"]
            or platform_evidence.get("system") != runtime_baseline["system"]
            or platform_evidence.get("release") != runtime_baseline["release"]
            or platform_evidence.get("version") != runtime_baseline["platform_version"]
            or platform_evidence.get("machine") != runtime_baseline["machine"]
        ):
            raise LuxExecutionPlanAuthorityError("DA3 prepared runtime evidence disagrees with carried model/device authority")

        try:
            expected_prepared = build_prepared_cache_runtime_evidence(
                evidence,
                plan=carried_plan,
                candidate_authority=self._candidate_authority,
            )
            reported_payload = payload["prepared_cache_runtime"]
            if not isinstance(reported_payload, dict):
                return None
            reported_prepared = PreparedDepthCacheRuntimeEvidence.from_payload(reported_payload)
        except (TypeError, ValueError) as exc:
            logger.warning("DA3 cache runtime preparation cannot complete core evidence: %s", exc)
            return None
        if (
            expected_prepared is None
            or reported_prepared != expected_prepared
            or payload.get("runtime_identity_sha256") != expected_prepared.runtime_identity_sha256
        ):
            raise LuxExecutionPlanAuthorityError("DA3 worker prepared runtime evidence does not match the carried plan")

        try:
            parent_runtime_identity = prepare_parent_output_runtime_identity()
            final_prepared = bind_parent_output_dependency_identity(
                expected_prepared,
                parent_runtime_identity=parent_runtime_identity,
            )
            prepared_runtime_binding = {
                "schema": "tp.da3.prepared-runtime-binding.v1",
                "runtime_identity_sha256": final_prepared.runtime_identity_sha256,
                "requested_device": str(backend_identity["requested_device"]),
                "actual_device": str(backend_identity["actual_device"]),
                "executed_backend": str(backend_identity["executed_backend"]),
            }
            verification_token = merge_runtime_verification_entries(
                verification_token,
                parent_runtime_identity.verification_entries,
                prepared_runtime_binding=prepared_runtime_binding,
            )
            verification_token_sha256 = runtime_verification_token_sha256(verification_token)
            if not verify_runtime_verification_token(
                verification_token,
                expected_token_sha256=verification_token_sha256,
                expected_worker_runtime_identity_sha256=worker_runtime_sha256,
                expected_prepared_runtime_identity_sha256=final_prepared.runtime_identity_sha256,
                expected_requested_device=str(backend_identity["requested_device"]),
                expected_actual_device=str(backend_identity["actual_device"]),
                expected_executed_backend=str(backend_identity["executed_backend"]),
            ):
                raise ValueError("DA3 final runtime verification token is inconsistent")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("DA3 parent output runtime identity is non-authorizing: %s", exc)
            return None

        self._prepared_cache_runtime_identity = final_prepared
        self._prepared_cache_runtime_verification_token = verification_token
        self._prepared_cache_runtime_verification_token_sha256 = verification_token_sha256
        self._prepared_worker_runtime_identity_sha256 = worker_runtime_sha256
        self._prepared_parent_runtime_identity = parent_runtime_identity
        return final_prepared

    def verify_prepared_cache_runtime_identity(
        self,
        *,
        runtime_identity_sha256: str,
    ) -> bool:
        """Revalidate retained DA3 materialization at cache get/store boundaries."""

        prepared = self._prepared_cache_runtime_identity
        verification_token = self._prepared_cache_runtime_verification_token
        verification_token_sha256 = self._prepared_cache_runtime_verification_token_sha256
        worker_runtime_sha256 = self._prepared_worker_runtime_identity_sha256
        parent_runtime_identity = getattr(self, "_prepared_parent_runtime_identity", None)
        if prepared is None or prepared.runtime_identity_sha256 != runtime_identity_sha256:
            return False
        from .da3_runtime_identity import verify_parent_output_runtime_identity, verify_runtime_verification_token

        expected_backend = {"cpu": "pytorch_cpu", "mps": "pytorch_mps"}.get(self._device)
        runtime_inputs_valid = (
            verification_token is not None
            and verification_token_sha256 is not None
            and worker_runtime_sha256 is not None
            and parent_runtime_identity is not None
            and expected_backend is not None
            and verify_runtime_verification_token(
                verification_token,
                expected_token_sha256=verification_token_sha256,
                expected_worker_runtime_identity_sha256=worker_runtime_sha256,
                expected_prepared_runtime_identity_sha256=prepared.runtime_identity_sha256,
                expected_requested_device=self._device,
                expected_actual_device=self._device,
                expected_executed_backend=expected_backend,
            )
            and verify_parent_output_runtime_identity(parent_runtime_identity)
        )
        if not runtime_inputs_valid:
            self._disable_cache_runtime_authority("runtime evidence changed at a cache boundary")
            return False
        return True

    @classmethod
    def required_packages(cls) -> list[str]:
        """Return required import module names for the active DA3 runtime mode."""
        configured_python = os.environ.get("TRANSFORMATION_PORTAL_DA3_PYTHON")
        if configured_python and configured_python.strip():
            return []
        return ["transformers"]

    def runtime_required_packages(self) -> list[str]:
        """Return the complete host-process requirements for this runtime mode."""
        if self._uses_subprocess():
            return []
        return ["torch", "transformers"]

    def _prepare_image(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> tuple[Image.Image, np.ndarray]:
        """Normalize input image for both local and subprocess execution."""
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                arr = image
            elif np.issubdtype(image.dtype, np.floating):
                image_max = image.max()
                if image_max <= 1.0:
                    image_min = image.min()
                    if image_min >= 0.0:
                        arr = (image * 255).astype(np.uint8)
                    else:
                        arr = image.astype(np.uint8, copy=False)
                else:
                    arr = image.astype(np.uint8, copy=False)
            else:
                arr = image.astype(np.uint8, copy=False)
            image_pil = Image.fromarray(arr)
            image_array = arr
        else:
            image_pil = image.convert("RGB")
            image_array = np.array(image_pil)
        return image_pil, image_array

    def compute(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> DepthResult:
        """Estimate relative depth from image."""
        self.ensure_available()
        if self._uses_subprocess():
            return self._compute_subprocess(image, device)
        return self._compute_local(image, device)

    def _compute_local(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> DepthResult:
        """Run DA3 inference in the active Python process."""
        use_device = self._effective_device(device)
        if self._engine is None or device is not None:
            self._load_engine(use_device)

        image_pil, image_array = self._prepare_image(image)
        if self._engine is None:
            raise RuntimeError("DA3 engine failed to initialize")

        result = self._engine.predict(image_pil)
        return self._build_depth_result(
            depth_map=result.depth_map.astype(np.float32),
            image_array=image_array,
            metadata=result.metadata,
            use_device=use_device,
            runner_mode="in_process",
            python_executable=sys.executable,
            dtype="float32",
            input_size=(image_array.shape[0], image_array.shape[1]),
        )

    def _compute_subprocess(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> DepthResult:
        """Run DA3 inference in a dedicated Python subprocess."""
        image_pil, image_array = self._prepare_image(image)
        use_device = self._effective_device(device)

        with tempfile.TemporaryDirectory(prefix="tp_da3_") as tmpdir:
            tmp_root = Path(tmpdir)
            input_path = tmp_root / "input.png"
            output_depth_path = tmp_root / "depth.npy"
            output_json_path = tmp_root / "result.json"
            image_pil.save(input_path)

            common_args = [
                "--input-image",
                str(input_path),
                "--output-depth",
                str(output_depth_path),
                "--output-json",
                str(output_json_path),
            ]
            if self._candidate_authority is not None:
                command = self._build_worker_command(*common_args, *self._canonical_worker_args())
            else:
                command = self._build_worker_command(
                    *common_args,
                    "--model-variant",
                    self._model_variant.name,
                    "--model-key",
                    self._resolved_model_contract.canonical_key,
                    "--device",
                    str(use_device),
                )
                if self._resolved_model_contract.revision:
                    command.extend(["--model-revision", self._resolved_model_contract.revision])
                if self._non_commercial_opt_in_enabled():
                    command.append("--non-commercial-ok")
                if self._apple_coreml_opt_in_enabled():
                    command.append("--use-coreml")
            if self._prepared_cache_runtime_identity is not None:
                from ...ingest.canonical_json import canonicalize_json

                verification_token_path = tmp_root / "runtime-verification-token.json"
                verification_token = self._prepared_cache_runtime_verification_token
                verification_token_sha256 = self._prepared_cache_runtime_verification_token_sha256
                if verification_token is None or verification_token_sha256 is None:
                    raise LuxExecutionPlanAuthorityError("DA3 cache runtime identity is missing its verification token")
                verification_token_path.write_bytes(canonicalize_json(verification_token))
                command.extend(
                    [
                        "--expected-runtime-identity-sha256",
                        self._prepared_cache_runtime_identity.runtime_identity_sha256,
                        "--runtime-verification-token",
                        str(verification_token_path),
                        "--runtime-verification-token-sha256",
                        verification_token_sha256,
                    ]
                )

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    cwd=self._worker_cwd(),
                    env=self._build_worker_env(),
                    check=True,
                    timeout=self._subprocess_timeout_seconds,
                    input=self._canonical_worker_input(),
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    self._build_subprocess_failure_message(
                        title="DA3 subprocess failed.",
                        category="executable_not_found",
                        summary="configured DA3 Python executable could not be launched",
                        python_executable=self._python_executable or "(unset)",
                        command=command,
                        stdout="",
                        stderr=str(exc),
                    )
                ) from exc
            except OSError as exc:
                raise RuntimeError(
                    self._build_subprocess_failure_message(
                        title="DA3 subprocess failed.",
                        category="startup_failed",
                        summary="configured DA3 Python executable could not be launched",
                        python_executable=self._python_executable or "(unset)",
                        command=command,
                        stdout="",
                        stderr=str(exc),
                    )
                ) from exc
            except subprocess.TimeoutExpired as exc:
                timeout_stdout = _subprocess_text(exc.stdout)
                timeout_stderr = _subprocess_text(exc.stderr)
                category, summary = _classify_subprocess_failure(
                    phase="inference",
                    stdout=timeout_stdout,
                    stderr=timeout_stderr,
                    timed_out=True,
                )
                raise RuntimeError(
                    self._build_subprocess_failure_message(
                        title="DA3 subprocess failed.",
                        category=category,
                        summary=summary,
                        python_executable=self._python_executable or "(unset)",
                        command=command,
                        stdout=timeout_stdout,
                        stderr=timeout_stderr,
                    )
                ) from exc
            except subprocess.CalledProcessError as exc:
                category, summary = _classify_subprocess_failure(
                    phase="inference",
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                )
                raise RuntimeError(
                    self._build_subprocess_failure_message(
                        title="DA3 subprocess failed.",
                        category=category,
                        summary=summary,
                        python_executable=self._python_executable or "(unset)",
                        command=command,
                        stdout=exc.stdout or "",
                        stderr=exc.stderr or "",
                        returncode=exc.returncode,
                    )
                ) from exc

            if not output_depth_path.exists() or not output_json_path.exists():
                category, summary = _classify_subprocess_failure(
                    phase="inference",
                    stdout=result.stdout,
                    stderr=result.stderr,
                    protocol_error=True,
                )
                raise RuntimeError(
                    self._build_subprocess_failure_message(
                        title="DA3 subprocess failed.",
                        category=category,
                        summary=summary,
                        python_executable=self._python_executable or "(unset)",
                        command=command,
                        stdout=result.stdout,
                        stderr=result.stderr,
                    )
                )

            try:
                if self._prepared_cache_runtime_identity is not None:
                    from .da3_runtime_identity import load_da3_worker_runtime_handshake

                    payload = load_da3_worker_runtime_handshake(output_json_path)
                else:
                    with output_json_path.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                verified_runtime_identity = self._verify_worker_authority_echo(payload)
                if verified_runtime_identity is not None:
                    depth_map = _load_verified_worker_depth(output_depth_path, payload.get("depth_artifact"))
                else:
                    depth_map = np.load(output_depth_path, allow_pickle=False).astype(np.float32)
            except LuxExecutionPlanAuthorityError:
                raise
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                category, summary = _classify_subprocess_failure(
                    phase="inference",
                    stdout=result.stdout,
                    stderr=f"{result.stderr}\n{exc}".strip(),
                    protocol_error=True,
                )
                raise RuntimeError(
                    self._build_subprocess_failure_message(
                        title="DA3 subprocess failed.",
                        category=category,
                        summary=summary,
                        python_executable=self._python_executable or "(unset)",
                        command=command,
                        stdout=result.stdout,
                        stderr=f"{result.stderr}\n{exc}".strip(),
                    )
                ) from exc

        input_size = payload.get("input_size")
        normalized_input_size = (image_array.shape[0], image_array.shape[1])
        if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
            normalized_input_size = (int(input_size[0]), int(input_size[1]))

        result_metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {}
        if verified_runtime_identity is not None:
            result_metadata["runtime_identity_sha256"] = verified_runtime_identity

        return self._build_depth_result(
            depth_map=depth_map,
            image_array=image_array,
            metadata=result_metadata,
            use_device=str(payload.get("device") or use_device),
            runner_mode="subprocess",
            python_executable=self._python_executable or "",
            dtype=str(payload.get("dtype") or "float32"),
            input_size=normalized_input_size,
        )

    def _build_depth_result(
        self,
        *,
        depth_map: np.ndarray,
        image_array: np.ndarray,
        metadata: Any,
        use_device: str,
        runner_mode: str,
        python_executable: str,
        dtype: str,
        input_size: tuple[int, int],
    ) -> DepthResult:
        """Normalize DA3 outputs into the shared backend contract."""
        normalized_metadata, warnings = _normalize_relative_depth_metadata(
            dict(metadata) if isinstance(metadata, dict) else {},
            runner_mode=runner_mode,
            python_executable=python_executable,
        )
        if self._candidate_authority is not None:
            normalized_metadata["execution_authority"] = {
                "plan_fingerprint_sha256": self._candidate_authority.plan_fingerprint_sha256,
                "candidate_id": self._candidate_authority.candidate_id,
                "model_backend_id": self._candidate_authority.constituent_backend_id,
                "executed_backend_id": self.name,
            }

        effective_device = normalized_metadata.get("device", use_device)
        if effective_device is not None:
            effective_device = str(effective_device)

        return DepthResult(
            depth_map=depth_map,
            original_image=image_array,
            metadata=normalized_metadata,
            depth_units="relative",
            focal_length_px=None,
            field_of_view_deg=None,
            backend_id=self.name,
            device=effective_device,
            dtype=dtype,
            input_size=input_size,
            warnings=warnings,
        )

    def get_cache_key(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Generate deterministic cache key for this image."""
        if isinstance(image, np.ndarray):
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
        else:
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]

        model_name = self._resolved_model_contract.canonical_key

        runner_mode = "subp" if self._uses_subprocess() else "local"
        runner_path = self._python_executable or sys.executable
        runner_hash = hashlib.sha256(runner_path.encode("utf-8")).hexdigest()[:8]

        return f"da3_{model_name}_{image_hash}_{self._cache_device_tag()}_{runner_mode}_{runner_hash}_v2"

    def _load_engine(self, device: str) -> None:
        """Lazy-load DA3InferenceEngine."""
        from ...lux_depth_v3.config import DA3Config, DeviceConfig
        from ...lux_depth_v3.inference import DA3InferenceEngine

        use_coreml = self._apple_coreml_opt_in_enabled()
        device_config = DeviceConfig(device=device, use_coreml=use_coreml)
        da3_config = DA3Config(
            model_variant=self._model_variant,
            model_key=self._resolved_model_contract.canonical_key,
            raw_model_id=self._resolved_model_contract.spec.repo_id,
            non_commercial_ok=bool(getattr(self._config, "non_commercial_ok", False)) if self._config else False,
            # Inject the authoritative contract so the engine consumes it
            # instead of performing another resolution (P0-1, issue #2065),
            # and pin the planned revision for the load path.
            resolved_model_contract=self._resolved_model_contract,
            model_revision=self._resolved_model_contract.revision,
            device=device_config,
        )

        self._engine = DA3InferenceEngine(
            config=da3_config,
            commercial_use=True,
            validate_license_strict=False,
            model_key=self._resolved_model_contract.canonical_key,
            raw_model_id=self._resolved_model_contract.spec.repo_id,
            non_commercial_ok=bool(getattr(self._config, "non_commercial_ok", False)) if self._config else False,
        )

        logger.info(
            "Loaded DA3 backend: model=%s device=%s",
            self._model_id,
            device,
        )
