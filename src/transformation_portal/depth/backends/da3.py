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
    preset_model_key_for_selection,
    with_typed_preset_provenance,
)
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
    from ...lux_depth_v3.inference import DA3InferenceEngine

logger = logging.getLogger(__name__)

DA3_RECOMMENDED_VENV = REPO_LOCAL_DA3_PYTHON
DA3_SETUP_SCRIPT = "./scripts/setup/install_da3_runtime.sh"
DEFAULT_DA3_SUBPROCESS_TIMEOUT_SECONDS = 900

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

    def __init__(self, config: Optional["EnhanceConfig"] = None):
        """Initialize DA3 backend."""
        self._config = config
        self._engine: Optional[DA3InferenceEngine] = None
        self._device = self._resolve_device(config)
        self._model_variant = self._resolve_model_variant(config)
        # P0-1 (issue #2065): consume the authoritative single-resolution
        # contract when the invocation carries one, instead of re-resolving.
        # Re-resolution here is the seam where the legacy model_variant
        # compatibility mapping could silently turn a commercial selection
        # (da3_metric) back into the research model (da3_research).
        authoritative_contract = None
        if config is not None:
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
        self._model_id = self._resolved_model_contract.spec.repo_id
        self._repo_root = self._find_repo_root()
        self._repo_src = self._repo_root / "src" if self._repo_root is not None else None
        self._python_executable = self._resolve_python_executable(config)
        self._subprocess_timeout_seconds = self._resolve_subprocess_timeout_seconds(config)
        self._subprocess_available_checked = False

    def _find_repo_root(self) -> Optional[Path]:
        """Find repository root by walking parent directories when in a checkout."""
        return find_repo_root(Path(__file__))

    def _worker_cwd(self) -> Path:
        """Choose a stable subprocess working directory."""
        if self._repo_root is not None:
            return self._repo_root
        return Path.cwd()

    def _resolve_device(self, config: Optional["EnhanceConfig"]) -> str:
        """Resolve device from config or auto-detect."""
        if config is not None:
            device = getattr(config, "depth_device", None)
            if device:
                return device
        return "cpu"

    def _resolve_model_variant(self, config: Optional["EnhanceConfig"]) -> "ModelVariant":
        """Resolve model variant from config."""
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
        """Build environment for the subprocess worker."""
        env = os.environ.copy()
        if self._repo_src is not None and self._repo_src.exists():
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = f"{self._repo_src}{os.pathsep}{existing}" if existing else str(self._repo_src)
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

    def _apple_coreml_opt_in_enabled(self) -> bool:
        """Return whether this backend is configured to prefer Apple CoreML."""
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
                cwd=self._worker_cwd(),
                env=self._build_worker_env(),
                check=False,
                timeout=self._subprocess_timeout_seconds,
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
            category, summary = _classify_subprocess_failure(
                phase="availability_check",
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                timed_out=True,
            )
            raise ImportError(
                self._build_subprocess_failure_message(
                    title="DA3 subprocess environment is not ready.",
                    category=category,
                    summary=summary,
                    python_executable=self._python_executable or "(unset)",
                    command=command,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
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
        use_device = device or self._device
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
        use_device = device or self._device

        with tempfile.TemporaryDirectory(prefix="tp_da3_") as tmpdir:
            tmp_root = Path(tmpdir)
            input_path = tmp_root / "input.png"
            output_depth_path = tmp_root / "depth.npy"
            output_json_path = tmp_root / "result.json"
            image_pil.save(input_path)

            command = self._build_worker_command(
                "--input-image",
                str(input_path),
                "--output-depth",
                str(output_depth_path),
                "--output-json",
                str(output_json_path),
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

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    cwd=self._worker_cwd(),
                    env=self._build_worker_env(),
                    check=True,
                    timeout=self._subprocess_timeout_seconds,
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
                category, summary = _classify_subprocess_failure(
                    phase="inference",
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    timed_out=True,
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
                with output_json_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                depth_map = np.load(output_depth_path, allow_pickle=False).astype(np.float32)
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

        return self._build_depth_result(
            depth_map=depth_map,
            image_array=image_array,
            metadata=payload.get("metadata", {}),
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
