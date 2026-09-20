"""Prepared, local-only material inference experiments with nonauthorizing scores.

This adapter is deliberately outside production semantic acceptance. Its observed
runtime identity and rejection thresholds are reproducibility evidence, not a
governed model selection or calibrated probability claim. No cache is consulted.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import platform
import stat
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np
from PIL import Image

from transformation_portal.core.image_artifact import ImageMaster, metadata_payload
from transformation_portal.depth.backends import da3_runtime_identity as runtime_identity
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3.execution_evidence import (
    _open_confined_artifact,
    _pin_output_root,
    _validate_confined_entry_identity,
)
from transformation_portal.lux_depth_v4.io import directory_path, snapshot
from transformation_portal.lux_depth_v4.photography import create_proxy

from .contracts import MaterialEvidence, MaterialLimits, MaterialsError, RegionEvidence, _metadata, validate_digest
from .proposal import SAM2_LARGE_SHA256, SAM2_MODEL_CONFIG, _CandidateBudget, check_cancelled, generate_proposals
from .semantics import (
    MATERIAL_PROMPTS,
    PREPROCESS_CONTRACT,
    SEMANTIC_SCORE_TYPE,
    classify_proposals,
)

_PACKAGES = (
    "sam2",
    "open-clip-torch",
    "torch",
    "torchvision",
    "numpy",
    "pillow",
    "safetensors",
    "timm",
    "hydra-core",
    "omegaconf",
)
_MODULES = ("sam2", "open_clip", "torch", "torchvision", "numpy", "PIL", "safetensors", "timm", "hydra", "omegaconf")


def _digest(payload: Any) -> str:
    return hashlib.sha256(canonicalize_json(metadata_payload(payload))).hexdigest()


def _file_identity(path: Path, *, maximum_bytes: int = 2_000_000_000) -> dict[str, Any]:
    root = directory_path(path.parent)
    _, receipt = snapshot(root, root / path.name, maximum_bytes=maximum_bytes, retain_bytes=False)
    return {"path": str(root / path.name), "sha256": receipt["sha256"], "size_bytes": receipt["size_bytes"]}


@contextmanager
def _private_weights(path: Path, receipt: Mapping[str, Any], *, suffix: str) -> Iterator[Path]:
    """Loaders consume an exclusive private copy, never the mutable source path.

    Copying is bounded and streamed through a pinned source descriptor. The copy
    must match the frozen receipt before it is made available to any ML loader.
    """
    root_path = directory_path(path.parent)
    if suffix not in {".pt", ".safetensors"}:
        raise MaterialsError("Unsupported private model checkpoint format")
    with tempfile.TemporaryDirectory(prefix="tp-materials-weights-") as directory:
        destination = Path(directory) / f"checkpoint{suffix}"
        with _pin_output_root(root_path) as root:
            source, relative = _open_confined_artifact(root, root_path / path.name)
            try:
                before = os.fstat(source)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or before.st_size != receipt["size_bytes"]
                    or not 0 < before.st_size <= 2_000_000_000
                ):
                    raise MaterialsError("Model snapshot source is not a bounded, unlinked regular file")
                descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                digest, count = hashlib.sha256(), 0
                with os.fdopen(descriptor, "wb") as target:
                    while True:
                        chunk = os.read(source, 1024 * 1024)
                        if not chunk:
                            break
                        count += len(chunk)
                        if count > before.st_size:
                            raise MaterialsError("Model snapshot grew during admission")
                        digest.update(chunk)
                        target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                _validate_confined_entry_identity(root, relative, before, context="material model snapshot")
                if count != before.st_size or digest.hexdigest() != receipt["sha256"]:
                    raise MaterialsError("Model snapshot differs from frozen weight identity")
                os.chmod(destination, 0o400)
            finally:
                os.close(source)
        yield destination


def _runtime_payload(device: str) -> dict[str, Any]:
    """Observe direct distribution bytes, actual module origins, and adapter code.

    This scope is explicit: it is not a trusted DA3 lock closure and cannot admit
    inferred edits. Reusing the hardened distribution reader avoids trusting a
    wheel version or RECORD declaration instead of the actual installed bytes.
    """
    modules = {name: importlib.import_module(name) for name in _MODULES}
    torch = modules["torch"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise MaterialsError("Requested MPS is unavailable; explicit device fallback is forbidden")
    if device == "cuda" and not torch.cuda.is_available():
        raise MaterialsError("Requested CUDA is unavailable; explicit device fallback is forbidden")
    distributions = [runtime_identity._distribution_record(name, verify_record_hashes=False) for name in _PACKAGES]
    origins = {}
    for name, module in modules.items():
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not origin:
            raise MaterialsError(f"Material runtime module lacks a source origin: {name}")
        origins[name] = str(Path(origin).resolve())
    sam_config = Path(origins["sam2"]).parent / SAM2_MODEL_CONFIG
    sources = {}
    for name in ("contracts.py", "taxonomy.py", "proposal.py", "semantics.py", "inference.py"):
        source = Path(__file__).parent / name
        sources[name] = _file_identity(source, maximum_bytes=1_000_000)["sha256"]
    photographic_origin = getattr(importlib.import_module("transformation_portal.lux_depth_v4.photography"), "__file__", None)
    if not isinstance(photographic_origin, str) or not photographic_origin:
        raise MaterialsError("Photographic preprocessing lacks a source origin")
    photographic_source = Path(photographic_origin)
    sources["photography.py"] = _file_identity(photographic_source, maximum_bytes=1_000_000)["sha256"]
    return {
        "scope": "observed_direct_distributions_and_adapter_sources_v1",
        "python": {
            "version": platform.python_version(),
            "executable": _file_identity(Path(sys.executable).resolve(), maximum_bytes=64_000_000),
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "actual_device": device,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "packages": distributions,
        "module_origins": origins,
        "adapter_sources": sources,
        "sam2_config": _file_identity(sam_config, maximum_bytes=1_000_000),
    }


@dataclass(frozen=True)
class InferenceConfig:
    """Bounded prototype options; local CLIP weights must use safetensors format."""

    sam2_checkpoint: Path
    clip_checkpoint: Path
    device: str = "cpu"
    proxy_longest_side: int = 1024
    max_proposals: int = 64
    classifier_batch_size: int = 8
    min_top_probability: float = 0.5
    min_similarity: float = 0.2
    min_margin: float = 0.05
    points_per_side: int = 16
    points_per_batch: int = 1
    crop_n_layers: int = 0
    max_mask_bytes: int = 512_000_000
    max_proxy_mask_bytes: int = 64_000_000
    max_candidate_bytes: int = 536_870_912

    def __post_init__(self) -> None:
        for name in ("sam2_checkpoint", "clip_checkpoint"):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)):
                raise MaterialsError(f"{name} must be an explicit local path")
            object.__setattr__(self, name, Path(value).expanduser().absolute())
        if self.device not in {"cpu", "mps", "cuda"}:
            raise MaterialsError("Inference requires explicit cpu, mps or cuda device")
        bounds = {
            "proxy_longest_side": (14, 2048),
            "max_proposals": (1, 256),
            "classifier_batch_size": (1, 32),
            "points_per_side": (1, 32),
            "points_per_batch": (1, 64),
            "max_mask_bytes": (1, 512_000_000),
            "max_proxy_mask_bytes": (1, 512_000_000),
            "max_candidate_bytes": (1, 2_147_483_648),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise MaterialsError(f"{name} must be an integer in [{low},{high}]")
        if type(self.crop_n_layers) is not int or self.crop_n_layers != 0:
            raise MaterialsError("Experimental bounded inference supports crop_n_layers=0 only")
        for name in ("min_top_probability", "min_similarity", "min_margin"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise MaterialsError(f"{name} must be finite in [0,1]")
            object.__setattr__(self, name, float(value))

    def to_payload(self) -> dict[str, Any]:
        return {name: str(value) if isinstance(value, Path) else value for name, value in vars(self).items()}


@dataclass(frozen=True)
class PreparedMaterialInference:
    """Frozen experimental semantics, model file receipts, and observed runtime."""

    source_sha256: str
    master_sha256: str
    config: InferenceConfig
    proxy_sha256: str
    proxy_transform: Mapping[str, Any]
    sam2_weights: Mapping[str, Any]
    clip_weights: Mapping[str, Any]
    runtime: Mapping[str, Any]
    runtime_sha256: str
    prompt_sha256: str
    preprocessing_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_sha256",
            "master_sha256",
            "proxy_sha256",
            "runtime_sha256",
            "prompt_sha256",
            "preprocessing_sha256",
        ):
            validate_digest(getattr(self, name), name)
        if not isinstance(self.config, InferenceConfig):
            raise MaterialsError("Prepared inference requires InferenceConfig")
        for name in ("proxy_transform", "sam2_weights", "clip_weights", "runtime"):
            object.__setattr__(self, name, _metadata(getattr(self, name)))
        if _digest(self.runtime) != self.runtime_sha256:
            raise MaterialsError("Prepared runtime digest does not bind runtime content")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "tp.materials.inference_plan.v1",
            "source_sha256": self.source_sha256,
            "master_sha256": self.master_sha256,
            "config": self.config.to_payload(),
            "proxy_sha256": self.proxy_sha256,
            "proxy_transform": metadata_payload(self.proxy_transform),
            "sam2_weights": metadata_payload(self.sam2_weights),
            "clip_weights": metadata_payload(self.clip_weights),
            "runtime": metadata_payload(self.runtime),
            "runtime_sha256": self.runtime_sha256,
            "prompt_sha256": self.prompt_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
            "cache_policy": "off",
            "semantic_authority": "uncalibrated",
        }

    def content_hash(self) -> str:
        return _digest(self.to_payload())


def prepare_inference(master: ImageMaster, config: InferenceConfig) -> PreparedMaterialInference:
    """Freeze local model/runtime identities; never load or download model weights."""
    if not isinstance(master, ImageMaster) or not isinstance(config, InferenceConfig):
        raise MaterialsError("Inference preparation requires an ImageMaster and InferenceConfig")
    if math.prod(master.shape) > MaterialLimits().max_pixels:
        raise MaterialsError("Photograph exceeds experimental inference pixel budget")
    try:
        proxy = create_proxy(master, config.proxy_longest_side)
        _CandidateBudget(proxy.pixels.shape[:2], config).before_batch(min(config.points_per_batch, config.points_per_side**2))
        sam2_weights = _file_identity(config.sam2_checkpoint)
        clip_weights = _file_identity(config.clip_checkpoint)
        if sam2_weights["sha256"] != SAM2_LARGE_SHA256:
            raise MaterialsError("SAM2 checkpoint differs from pinned SAM2.1 Hiera Large bytes")
        # Empty/unsafe local safetensors files fail before expensive model loading.
        from safetensors import safe_open

        with safe_open(str(config.clip_checkpoint), framework="pt", device="cpu") as checkpoint:
            if not checkpoint.keys() or len(checkpoint.keys()) > 1024:
                raise MaterialsError("CLIP requires a bounded nonempty local safetensors checkpoint")
        runtime = _runtime_payload(config.device)
        return PreparedMaterialInference(
            master.source_sha256,
            master.content_hash(),
            config,
            hashlib.sha256(proxy.pixels.tobytes()).hexdigest(),
            proxy.transform.to_payload(),
            sam2_weights,
            clip_weights,
            runtime,
            _digest(runtime),
            _digest(MATERIAL_PROMPTS),
            _digest(PREPROCESS_CONTRACT),
        )
    except MaterialsError:
        raise
    except Exception as exc:
        raise MaterialsError(f"Material inference preparation failed without fallback: {exc}") from exc


def _verify_prepared(master: ImageMaster, prepared: PreparedMaterialInference) -> None:
    if master.source_sha256 != prepared.source_sha256 or master.content_hash() != prepared.master_sha256:
        raise MaterialsError("Photographic master differs from prepared inference identity")
    if prepared.prompt_sha256 != _digest(MATERIAL_PROMPTS) or prepared.preprocessing_sha256 != _digest(PREPROCESS_CONTRACT):
        raise MaterialsError("Semantic prompt or preprocessing contract changed after preparation")
    if _file_identity(prepared.config.sam2_checkpoint) != dict(prepared.sam2_weights) or _file_identity(
        prepared.config.clip_checkpoint
    ) != dict(prepared.clip_weights):
        raise MaterialsError("Model weight bytes changed after inference preparation")
    if prepared.sam2_weights["sha256"] != SAM2_LARGE_SHA256:
        raise MaterialsError("Prepared SAM2 checkpoint lacks the pinned model identity")
    if _digest(_runtime_payload(prepared.config.device)) != prepared.runtime_sha256:
        raise MaterialsError("Material inference runtime changed after preparation")


def infer_materials(
    master: ImageMaster,
    prepared: PreparedMaterialInference,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> MaterialEvidence:
    """Execute the frozen experiment; inferred scores never authorize edits."""
    if not isinstance(master, ImageMaster) or not isinstance(prepared, PreparedMaterialInference):
        raise MaterialsError("Inference execution requires an ImageMaster and PreparedMaterialInference")
    check_cancelled(cancelled)
    try:
        _verify_prepared(master, prepared)
        proxy = create_proxy(master, prepared.config.proxy_longest_side)
        if hashlib.sha256(
            proxy.pixels.tobytes()
        ).hexdigest() != prepared.proxy_sha256 or proxy.transform.to_payload() != metadata_payload(prepared.proxy_transform):
            raise MaterialsError("Model proxy changed after preparation")
        with _private_weights(prepared.config.sam2_checkpoint, prepared.sam2_weights, suffix=".pt") as sam2_copy:
            proposal_config = replace(prepared.config, sam2_checkpoint=sam2_copy)
            proposals = generate_proposals(proxy.pixels, proposal_config, master_shape=master.shape, cancelled=cancelled)
        check_cancelled(cancelled)
        with _private_weights(prepared.config.clip_checkpoint, prepared.clip_weights, suffix=".safetensors") as clip_copy:
            classifier_config = replace(prepared.config, clip_checkpoint=clip_copy)
            observations = classify_proposals(proxy.pixels, proposals, classifier_config, cancelled=cancelled)
        if len(observations) != len(proposals):
            raise MaterialsError("Semantic output count differs from geometric proposals")
        if len(proposals) * math.prod(master.shape) * 4 > prepared.config.max_mask_bytes:
            raise MaterialsError("Restored full-resolution evidence exceeds mask budget")
        regions, diagnostics = [], []
        height, width = proxy.transform.resized_shape
        for index, (proposal, observation) in enumerate(zip(proposals, observations)):
            check_cancelled(cancelled)
            crop = proposal.mask[:height, :width]
            lifted = np.asarray(
                Image.fromarray(crop).resize((master.shape[1], master.shape[0]), Image.Resampling.NEAREST), dtype=bool
            )
            region_id = f"region-{index:04d}"
            regions.append(
                RegionEvidence(
                    region_id,
                    observation.label,
                    lifted,
                    observation.confidence,
                    proposal.geometric_quality,
                    provenance="inferred",
                    score_type=SEMANTIC_SCORE_TYPE,
                )
            )
            diagnostics.append(
                {
                    "region_id": region_id,
                    "cosine_similarity": observation.cosine_similarity,
                    "top2_margin": observation.margin,
                    "rejection": observation.rejection,
                    "geometric_stability": proposal.stability,
                }
            )
        check_cancelled(cancelled)
        # Reject mutations during loading/inference before admitting any evidence.
        _verify_prepared(master, prepared)
        return MaterialEvidence(
            master.source_sha256,
            master.shape,
            tuple(regions),
            producer={
                "inference_plan_sha256": prepared.content_hash(),
                "runtime_sha256": prepared.runtime_sha256,
                "sam2_weights_sha256": prepared.sam2_weights["sha256"],
                "clip_weights_sha256": prepared.clip_weights["sha256"],
                "prompt_sha256": prepared.prompt_sha256,
                "preprocessing_sha256": prepared.preprocessing_sha256,
                "actual_proposal_device": prepared.config.device,
                "actual_classifier_device": prepared.config.device,
                "model_selection": "local_research_prototype_observed_identity",
                "semantic_authority": "uncalibrated",
                "cache_policy": "off",
                "proxy_transform": metadata_payload(prepared.proxy_transform),
                "mask_restore": "crop_padding_then_nearest_pixel_centers",
                "diagnostics": diagnostics,
            },
        )
    except MaterialsError:
        raise
    except Exception as exc:
        raise MaterialsError(f"Material inference failed without fallback: {exc}") from exc
