"""Depth Pro backend adapter for unified backend registry.

Wraps DepthProStage to provide the DepthBackend interface with
consistent contract and license governance.

Supports two execution modes:
- In-process when ``depth_pro`` is installed in the active environment
- Subprocess mode via a dedicated Python executable for isolated
  NumPy 1.x / Depth Pro environments

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

from ...core.platform_matrix import CURRENT_PLATFORM
from ...lux_depth_v3.execution_plan_adapter import LuxExecutionPlanAuthorityError
from .protocol import DepthResult, LicenseRestrictionError, LicenseType

if TYPE_CHECKING:
    from ...lux_depth_v3.config import EnhanceConfig
    from ...lux_depth_v3.execution_lifecycle import BackendCandidateAuthority
    from ...stage_graph.stages.depth_pro import DepthProStage

logger = logging.getLogger(__name__)


class DepthProBackend:
    """Depth Pro backend adapter implementing DepthBackend protocol.

    Wraps ``DepthProStage`` for use with ``DepthBackendRegistry`` and can
    optionally execute that stage in a separate Python environment.
    Provides metric depth (meters) with checkpoint provenance.

    License Requirements:
        Depth Pro requires BOTH flags to be True:
        - non_commercial_ok: Acknowledge non-commercial use only
        - accept_apple_depth_pro_research_license: Accept Apple AMLR license
    """

    name = "depth_pro"
    license_type = LicenseType.RESEARCH_ONLY
    requires_checkpoint = True

    CHECKPOINT_URL = "https://ml-site.cdn-apple.com/" "models/depth-pro/depth_pro.pt"
    DEFAULT_CHECKPOINT = Path("checkpoints/depth_pro.pt")
    EXPECTED_SHA256 = "3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb" "2686259c80685c0ce"
    WORKER_MODULE = "transformation_portal.depth.backends.depth_pro_worker"

    def __init__(
        self,
        config: Optional["EnhanceConfig"] = None,
        *,
        candidate_authority: Optional["BackendCandidateAuthority"] = None,
        canonical_plan_bytes: Optional[bytes] = None,
    ):
        """Initialize Depth Pro backend."""
        if (candidate_authority is None) != (canonical_plan_bytes is None):
            raise ValueError("candidate_authority and canonical_plan_bytes must be provided together")
        carried_contract = candidate_authority.model_contract if candidate_authority is not None else None
        if candidate_authority is not None:
            if config is None:
                raise ValueError("Canonical Depth Pro authority requires its projected runtime config")
            if (
                candidate_authority.backend_id != self.name
                or carried_contract is None
                or carried_contract.backend_id != self.name
            ):
                raise ValueError("Canonical Depth Pro authority does not select a Depth Pro model contract")
            if carried_contract.artifact_path is None:
                raise ValueError("Canonical Depth Pro authority lacks a checkpoint path")
            if carried_contract.artifact_sha256 != self.EXPECTED_SHA256:
                raise ValueError("Canonical Depth Pro authority carries an unexpected checkpoint digest")
            if type(canonical_plan_bytes) is not bytes or not canonical_plan_bytes:
                raise ValueError("Canonical Depth Pro authority requires non-empty immutable plan bytes")
        self._config = config
        self._candidate_authority = candidate_authority
        self._canonical_plan_bytes = canonical_plan_bytes
        self._stage: Optional["DepthProStage"] = None
        self._repo_root = self._find_repo_root()
        self._repo_src = self._repo_root / "src" if self._repo_root is not None else None
        self._device = self._resolve_device(config, candidate_authority)
        if carried_contract is not None:
            if carried_contract.artifact_path is None:  # pragma: no cover - narrowed above
                raise ValueError("Canonical Depth Pro authority lacks a checkpoint path")
            self._checkpoint_path = Path(carried_contract.artifact_path)
        else:
            self._checkpoint_path = self._resolve_checkpoint_path(config)
        self._checkpoint_hash_cached: Optional[str] = None
        self._python_executable = self._resolve_python_executable(config)
        self._subprocess_available_devices: set[str] = set()

    def _find_repo_root(self) -> Optional[Path]:
        """Find repository root by walking parent directories when in a checkout."""
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists() and (parent / "src").exists():
                return parent
        return None

    def _worker_cwd(self) -> Path:
        """Choose a stable subprocess working directory.

        Source checkouts use the repo root so a sibling dedicated Depth Pro
        environment can import the local package via ``PYTHONPATH``. Installed
        package layouts fall back to the caller's current working directory.
        """
        if self._repo_root is not None:
            return self._repo_root
        return Path.cwd()

    def _resolve_device(
        self,
        config: Optional["EnhanceConfig"],
        candidate_authority: Optional["BackendCandidateAuthority"] = None,
    ) -> str:
        """Resolve device from config, defaulting to CPU.

        This method does NOT auto-detect accelerators (MPS/CUDA) because
        importing torch here would load libomp.dylib on macOS. If the
        Depth Pro subprocess (running in .venv-depth-pro) later loads its
        own libomp, the process aborts with "OMP: Error #15".

        The orchestrator passes an explicit depth_device, so production
        workflows always get the correct device. CPU is the safe default
        for ad-hoc or test instantiation.
        """
        if candidate_authority is not None:
            carried = self._normalize_device(candidate_authority.device)
            if carried != "auto":
                return carried
        if config is not None:
            device = getattr(config, "depth_device", None)
            if device:
                return self._normalize_device(device)

        return "cpu"

    @staticmethod
    def _normalize_device(device: Any) -> str:
        """Normalize explicit or override device strings to worker-safe tokens."""
        return str(device or "").strip().lower() or "cpu"

    def _resolve_checkpoint_path(
        self,
        config: Optional["EnhanceConfig"],
    ) -> Path:
        """Resolve checkpoint path from config, env var, or default."""
        if config is not None:
            path = getattr(config, "depth_pro_checkpoint_path", None)
            if path:
                return Path(path).expanduser()

        env_path = os.environ.get("TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT")
        if env_path:
            return Path(env_path).expanduser()

        return self.DEFAULT_CHECKPOINT

    def _resolve_python_executable(
        self,
        config: Optional["EnhanceConfig"],
    ) -> Optional[str]:
        """Resolve optional subprocess Python executable for Depth Pro."""
        candidate: Optional[str] = None
        if config is not None:
            configured = getattr(config, "depth_pro_python_executable", None)
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
            env_candidate = os.environ.get("TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON")
            if env_candidate and env_candidate.strip():
                candidate = env_candidate.strip()

        if candidate is None:
            return None

        has_separator = os.sep in candidate or (os.altsep is not None and os.altsep in candidate)
        if candidate.startswith(".") or has_separator:
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = self._worker_cwd() / path
            if not path.exists():
                raise FileNotFoundError(f"Depth Pro Python executable not found: {path}")
            # Preserve virtualenv interpreter symlink paths such as
            # ``.venv-depth-pro/bin/python`` so subprocess execution keeps the
            # venv context instead of collapsing to the underlying system
            # interpreter.
            return os.path.abspath(os.fspath(path))

        resolved = shutil.which(candidate)
        if resolved is None:
            raise FileNotFoundError(f"Depth Pro Python executable not found on PATH: {candidate}")
        return resolved

    def _uses_subprocess(self) -> bool:
        """Return whether Depth Pro should execute in a dedicated subprocess."""
        return self._python_executable is not None

    def _build_worker_env(self) -> dict[str, str]:
        """Build environment for the subprocess worker."""
        env = os.environ.copy()
        if self._repo_src is not None and self._repo_src.exists():
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = f"{self._repo_src}{os.pathsep}{existing}" if existing else str(self._repo_src)
        return env

    def _build_worker_command(self, *args: str) -> list[str]:
        """Build the subprocess worker command."""
        if self._python_executable is None:
            raise RuntimeError("Depth Pro subprocess requested without a Python executable")
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
            raise RuntimeError("Canonical Depth Pro execution is missing immutable plan bytes")
        return self._canonical_plan_bytes.decode("utf-8", errors="strict")

    def _effective_device(self, override: Optional[str]) -> str:
        """Reject runtime overrides that disagree with carried authority."""

        if override is None:
            return self._device
        normalized = self._normalize_device(override)
        if self._candidate_authority is not None and normalized != self._device:
            raise ValueError(f"Depth Pro device override {normalized!r} disagrees with carried authority {self._device!r}")
        return normalized

    def _verify_worker_authority_echo(self, payload: Any) -> None:
        """Verify worker authority before loading or accepting its array."""

        authority = self._candidate_authority
        if authority is None:
            return
        expected = {
            "plan_fingerprint_sha256": authority.plan_fingerprint_sha256,
            "candidate_id": authority.candidate_id,
            "model_backend_id": authority.constituent_backend_id,
            "executed_backend_id": self.name,
        }
        if not isinstance(payload, dict) or payload.get("execution_authority") != expected:
            raise LuxExecutionPlanAuthorityError("Depth Pro worker execution-authority echo does not match the carried plan")

    def _ensure_checkpoint_exists(self) -> None:
        """Ensure the Depth Pro checkpoint exists locally."""
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"Depth Pro checkpoint not found: {self._checkpoint_path}\n\n"
                f"Download checkpoint (1.9 GB) with:\n"
                f"  mkdir -p {self._checkpoint_path.parent}\n"
                f"  curl -L {self.CHECKPOINT_URL}"
                f" -o {self._checkpoint_path}\n\n"
                f"Or set path via:\n"
                f"  - Config: depth_pro_checkpoint_path='path/to/checkpoint.pt'\n"
                f"  - Env: TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT='path/to/checkpoint.pt'"
            )

    def _ensure_local_package_available(self) -> None:
        """Ensure depth_pro is installed in the active environment."""
        try:
            import depth_pro  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "depth_pro package not installed in the active environment.\n\n"
                "Preferred setup:\n"
                "  1. Bootstrap the repo-local Depth Pro runtime with\n"
                "     ./scripts/setup/install_depth_pro_runtime.sh\n"
                "  2. Set depth_pro_python_executable or TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON\n\n"
                "Legacy in-process install:\n"
                "  pip install depth-pro\n\n"
                "See: https://github.com/apple/ml-depth-pro"
            ) from exc

    def _ensure_subprocess_available(self, device: Optional[str] = None) -> None:
        """Ensure the dedicated Depth Pro subprocess environment is usable."""
        use_device = self._normalize_device(device or self._device)
        if use_device in self._subprocess_available_devices:
            return

        if self._candidate_authority is not None:
            command = self._build_worker_command("--check", *self._canonical_worker_args())
        else:
            command = self._build_worker_command(
                "--check",
                "--checkpoint",
                str(self._checkpoint_path.resolve()),
                "--device",
                use_device,
            )
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=self._worker_cwd(),
            env=self._build_worker_env(),
            check=False,
            input=self._canonical_worker_input(),
        )
        if result.returncode != 0:
            output = self._format_subprocess_output(result.stdout, result.stderr)
            raise ImportError(
                "Depth Pro subprocess environment is not ready.\n\n"
                f"Python: {self._python_executable}\n"
                f"Command: {' '.join(command)}\n"
                f"Output:\n{output}"
            )
        self._subprocess_available_devices.add(use_device)

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

    def _ensure_runtime_available(self, device: Optional[str] = None) -> None:
        """Ensure Depth Pro dependencies and checkpoint are available for a device."""
        self._ensure_checkpoint_exists()
        if self._uses_subprocess():
            self._ensure_subprocess_available(device=device)
            return
        self._ensure_local_package_available()

    def ensure_available(self) -> None:
        """Ensure Depth Pro dependencies and checkpoint are available."""
        self._ensure_runtime_available(device=self._device)

    @classmethod
    def required_packages(cls) -> list[str]:
        """Return required import module names for the local backend path."""
        return ["depth_pro"]

    def compute(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> DepthResult:
        """Estimate metric depth from image."""
        self._validate_license_runtime()
        use_device = self._effective_device(device)
        self._ensure_runtime_available(device=use_device)

        if self._uses_subprocess():
            return self._compute_subprocess(image, use_device)
        return self._compute_local(image, use_device)

    def _prepare_image(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> tuple[Image.Image, np.ndarray]:
        """Normalize input image for both local and subprocess execution."""
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                arr = (image * 255).astype(np.uint8)
            else:
                arr = image.astype(np.uint8)
            image_pil = Image.fromarray(arr)
            image_array = arr
        else:
            image_pil = image.convert("RGB")
            image_array = np.array(image_pil)
        return image_pil, image_array

    def _compute_local(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> DepthResult:
        """Run Depth Pro inference in the active Python process."""
        if self._stage is None:
            self._load_stage()
            assert self._stage is not None

        image_pil, image_array = self._prepare_image(image)

        from ...stage_graph.stage import StageContext, StageStatus

        use_device = self._effective_device(device)
        context = StageContext(
            artifacts={"image": image_pil},
            device=use_device,
        )
        result = self._stage.compute(context)

        if result.status != StageStatus.COMPLETED:
            raise RuntimeError(f"Depth Pro inference failed: {result.error}\nTraceback:\n{result.error_traceback}")

        depth_map = result.artifacts.get("depth_map")
        if depth_map is None:
            raise RuntimeError("Depth Pro did not return depth_map artifact")

        metric_metadata = self._build_metric_metadata(
            result.artifacts.get("depth_provenance", {}),
            runner_mode="in_process",
            python_executable=sys.executable,
        )
        self._cache_checkpoint_hash_from_metadata(metric_metadata)
        if self._candidate_authority is not None:
            metric_metadata["execution_authority"] = {
                "plan_fingerprint_sha256": self._candidate_authority.plan_fingerprint_sha256,
                "candidate_id": self._candidate_authority.candidate_id,
                "model_backend_id": self._candidate_authority.constituent_backend_id,
                "executed_backend_id": self.name,
            }

        return DepthResult(
            depth_map=depth_map.astype(np.float32),
            original_image=image_array,
            metadata=metric_metadata,
            depth_units="meters",
            focal_length_px=result.metadata.get("focal_length_px"),
            field_of_view_deg=result.metadata.get("fov_deg"),
            backend_id=self.name,
            device=use_device,
            dtype="float32",
            input_size=(image_array.shape[0], image_array.shape[1]),
        )

    def _compute_subprocess(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> DepthResult:
        """Run Depth Pro inference in a dedicated Python subprocess."""
        image_pil, image_array = self._prepare_image(image)
        use_device = self._effective_device(device)

        with tempfile.TemporaryDirectory(prefix="tp_depth_pro_") as tmpdir:
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
                    "--checkpoint",
                    str(self._checkpoint_path.resolve()),
                    "--device",
                    use_device,
                )

            try:
                subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    cwd=self._worker_cwd(),
                    env=self._build_worker_env(),
                    check=True,
                    input=self._canonical_worker_input(),
                )
            except subprocess.CalledProcessError as exc:
                output = self._format_subprocess_output(exc.stdout or "", exc.stderr or "")
                raise RuntimeError(
                    "Depth Pro subprocess failed.\n\n"
                    f"Python: {self._python_executable}\n"
                    f"Command: {' '.join(command)}\n"
                    f"Output:\n{output}"
                ) from exc

            if not output_depth_path.exists() or not output_json_path.exists():
                raise RuntimeError("Depth Pro subprocess completed without producing the expected output files.")

            with output_json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._verify_worker_authority_echo(payload)
            depth_map = np.load(output_depth_path, allow_pickle=False).astype(np.float32)

        metric_metadata = self._build_metric_metadata(
            payload.get("provenance", {}),
            runner_mode="subprocess",
            python_executable=self._python_executable or "",
        )
        self._cache_checkpoint_hash_from_metadata(metric_metadata)
        if self._candidate_authority is not None:
            metric_metadata["execution_authority"] = payload["execution_authority"]

        input_size = payload.get("input_size")
        normalized_input_size = (image_array.shape[0], image_array.shape[1])
        if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
            normalized_input_size = (int(input_size[0]), int(input_size[1]))

        warnings = payload.get("warnings")
        if not isinstance(warnings, list):
            warnings = []

        return DepthResult(
            depth_map=depth_map,
            original_image=image_array,
            metadata=metric_metadata,
            depth_units="meters",
            focal_length_px=self._coerce_float(payload.get("focal_length_px")),
            field_of_view_deg=self._coerce_float(payload.get("field_of_view_deg")),
            backend_id=self.name,
            device=str(payload.get("device") or use_device),
            dtype=str(payload.get("dtype") or "float32"),
            input_size=normalized_input_size,
            warnings=warnings,
        )

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        """Best-effort float coercion for subprocess payload values."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_metric_metadata(
        self,
        provenance: Any,
        *,
        runner_mode: str,
        python_executable: str,
    ) -> dict[str, Any]:
        """Normalize metric-depth metadata across execution modes."""
        metric_metadata = dict(provenance) if isinstance(provenance, dict) else {}
        metric_metadata["source_depth_units"] = "meters"
        metric_metadata["output_depth_units"] = "meters"
        metric_metadata["output_normalization"] = "none"
        metric_metadata["runner"] = {
            "mode": runner_mode,
            "python_executable": python_executable,
        }
        return metric_metadata

    @staticmethod
    def _normalize_sha256(value: Any) -> Optional[str]:
        """Normalize SHA-256 strings for internal checkpoint caching."""
        if not isinstance(value, str):
            return None
        candidate = value.strip().lower()
        if len(candidate) == 64 and all(ch in "0123456789abcdef" for ch in candidate):
            return candidate
        return None

    def _cache_checkpoint_hash_from_metadata(self, metadata: dict[str, Any]) -> None:
        """Populate cached checkpoint hash from backend metadata when available."""
        checkpoint = metadata.get("checkpoint")
        if isinstance(checkpoint, dict):
            sha256 = self._normalize_sha256(checkpoint.get("sha256"))
            if sha256 is not None:
                self._checkpoint_hash_cached = sha256

    def get_cache_key(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Generate deterministic cache key for this image."""
        if isinstance(image, np.ndarray):
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
        else:
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]

        if self._checkpoint_path.exists():
            ckpt_hash = self._get_checkpoint_hash()[:16]
        else:
            ckpt_hash = "no_ckpt"

        runner_mode = "subp" if self._uses_subprocess() else "local"
        runner_path = self._python_executable or sys.executable
        runner_hash = hashlib.sha256(runner_path.encode("utf-8")).hexdigest()[:8]

        return f"depthpro_{ckpt_hash}_{image_hash}_{self._device}_{runner_mode}_{runner_hash}_v2"

    def _load_stage(self) -> None:
        """Lazy-load DepthProStage for local in-process execution."""
        from ...stage_graph.stages.depth_pro import DepthProStage

        self._stage = DepthProStage(
            checkpoint_path=self._checkpoint_path,
            device=self._device,
            strict_validation=True,
        )

    def _validate_license_runtime(self) -> None:
        """Runtime license validation (Layer 3: defense-in-depth)."""
        if self._config is None:
            raise LicenseRestrictionError(
                "Depth Pro requires EnhanceConfig with license flags.\n"
                "Create config with:\n"
                "  config = EnhanceConfig(\n"
                "      non_commercial_ok=True,\n"
                "      accept_apple_depth_pro_research_license=True,\n"
                "  )"
            )

        if not getattr(self._config, "non_commercial_ok", False):
            raise LicenseRestrictionError("Depth Pro requires non_commercial_ok=True in config.")

        if not getattr(
            self._config,
            "accept_apple_depth_pro_research_license",
            False,
        ):
            raise LicenseRestrictionError("Depth Pro requires accept_apple_depth_pro_research_license=True in config.")

        logger.debug("Runtime license validation passed for depth_pro")

    def _get_checkpoint_hash(self) -> str:
        """Get SHA256 of checkpoint file (cached)."""
        if self._checkpoint_hash_cached is None:
            h = hashlib.sha256()
            with open(self._checkpoint_path, "rb") as f:
                for chunk in iter(
                    lambda: f.read(1024 * 1024),
                    b"",
                ):
                    h.update(chunk)
            self._checkpoint_hash_cached = h.hexdigest()
        return self._checkpoint_hash_cached
