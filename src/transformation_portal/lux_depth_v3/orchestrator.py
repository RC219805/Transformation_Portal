"""Orchestrator for V3 depth + V2 enhancement pipeline.

Two-stage pipeline:
1. Stage A (V3): Generate depth assets using DA3 (Inference -> Post-Processing -> Write)
2. Stage B (V2): Consume depth assets -> V2 Subprocess -> Output
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any
import time
import logging
import datetime

# Note: Imports adjusted to relative for package context compatibility
from .config import DA3Config, ModelVariant, EnhanceConfig
from .inference import DA3InferenceEngine
from .postprocessing import Postprocessor
from .input_manager import ImageInput
from .depth_writer import atomic_write_depth_u16_png_with_stats
from .v2_runner import V2Runner, find_v2_report
from .security import (
    HashMode,
    sanitize_file_stem,
    sanitize_path_component_nonlossy,
)
from .manifest import (
    CombinedManifest,
    ConfigFingerprint,
    InputMetadata,
    DepthMetadata,
    V2Metadata,
    TimingMetadata,
    ReproMetadata,
    BatchManifest,
    compute_file_sha256,
    get_git_revision,
    capture_environment,
)
from .batch_stats import compute_batch_runtime_stats

logger = logging.getLogger(__name__)


def make_output_key(input_path: Path, input_root: Path) -> Path:
    """Generate collision-free output key preserving directory structure."""
    try:
        relpath = input_path.relative_to(input_root)
    except ValueError:
        logger.warning(f"{input_path} is not relative to {input_root}, using flat naming")
        relpath = Path(input_path.name)

    sanitized_parts = [sanitize_path_component_nonlossy(p) for p in relpath.parent.parts]
    stem_sanitized = sanitize_path_component_nonlossy(relpath.stem)

    if sanitized_parts:
        return Path(*sanitized_parts) / stem_sanitized
    else:
        return Path(stem_sanitized)


class EnhanceOrchestrator:
    """Orchestrates V3 depth generation + V2 enhancement pipeline."""

    def __init__(self, config: EnhanceConfig, output_root: Path):
        self.config = config
        self.output_root = Path(output_root)

        if config.hash_mode == HashMode.NEVER:
            logger.warning("Hash mode set to 'never' - manifests will lack integrity verification.")

        # Create output directories
        self.depth_dir = self.output_root / "depth"
        self.v2_dir = self.output_root / "v2"
        self.manifests_dir = self.output_root / "manifests"
        self.logs_dir = self.output_root / "logs"
        self.zones_dir = self.output_root / "zones"

        for d in [self.depth_dir, self.v2_dir, self.manifests_dir, self.logs_dir, self.zones_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize V3 Configuration (Priority: User Override > Preset > Default)
        if config.preset is not None:
            da3_config = DA3Config.from_preset(config.preset)
            if config.model_variant is not None:
                logger.info(f"Overriding preset model with user choice: {config.model_variant.value.display_name}")
                da3_config.model_variant = config.model_variant
            else:
                config.model_variant = da3_config.model_variant
        else:
            model = config.model_variant if config.model_variant is not None else ModelVariant.METRIC_LARGE
            da3_config = DA3Config(model_variant=model)
            config.model_variant = model

        da3_config.device.device = config.depth_device

        # Initialize Inference Engine
        self.inference_engine = DA3InferenceEngine(
            config=da3_config,
            commercial_use=not config.non_commercial_ok,
            validate_license_strict=True,
        )

        # Initialize Postprocessor (FIX: Ensures refine_edges/bilateral settings from preset are applied)
        self.postprocessor = Postprocessor(da3_config.postprocessing)

        # Initialize V2 Runner and Environment
        self.v2_runner = V2Runner()
        # Adjusted path logic for src/transformation_portal/lux_depth_v3 location
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        git_rev = get_git_revision(repo_root)
        self.v3_git = git_rev
        self.v2_git = git_rev
        self.environment = capture_environment()

    def run_pipeline(
        self,
        input_path: Path,
        output_dir: Optional[Path] = None,
        run_v2: bool = True
    ) -> Dict[str, Any]:
        """Run the complete depth pipeline on a single image.

        Simplified entry point for processing:
        1. Validate image format
        2. Preprocess image
        3. Run DA3 inference
        4. Save depth output
        5. Optionally run V2 enhancement

        Args:
            input_path: Path to input image
            output_dir: Optional override for output directory
            run_v2: Whether to run V2 enhancement (default: True)

        Returns:
            Dict containing status, paths, and runtime information
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        image_input = ImageInput(path=input_path)
        return self.enhance_image(image_input)

    def compute_config_fingerprint(self) -> ConfigFingerprint:
        return ConfigFingerprint(
            model_variant=self.config.model_variant.value.name,
            depth_quantization=self.config.depth_quantization,
            depth_device=self.config.depth_device,
            preset=self.config.preset.value if self.config.preset else None,
            v2_preset=self.config.v2_preset,
            v2_device=self.config.v2_device,
            v2_upscaler_backend=self.config.v2_upscaler_backend,
        )

    def _compute_or_skip_hash(self, image_path: Path, manifest_path: Optional[Path] = None) -> Optional[str]:
        if self.config.hash_mode == HashMode.NEVER:
            return None

        # logic for IF_MANIFEST_EXISTS is handled by caller or defaults to compute
        try:
            return compute_file_sha256(image_path)
        except Exception as e:
            logger.error(f"Hash computation failed for {image_path}: {e}")
            raise IOError(f"Hash computation failed: {e}") from e

    def should_skip_depth(self, depth_path: Path, manifest_path: Path, image_input: ImageInput) -> bool:
        if not depth_path.exists() or not manifest_path.exists():
            return False

        try:
            manifest = CombinedManifest.load(manifest_path)

            # Input Integrity Check
            if manifest.input and manifest.input.image_sha256 and self.config.hash_mode != HashMode.NEVER:
                current_hash = self._compute_or_skip_hash(image_input.path)
                if current_hash and current_hash != manifest.input.image_sha256:
                    logger.info(f"Input image changed - regenerating depth: {image_input.path}")
                    return False

            # Config Integrity Check
            if not manifest.config_fingerprint:
                return False

            current_fp = self.compute_config_fingerprint()
            manifest_fp = ConfigFingerprint(
                model_variant=manifest.depth.model if manifest.depth else "",
                depth_quantization=(manifest.depth.scaling.get("method", "") if manifest.depth else ""),
                depth_device=self.config.depth_device,
                preset=self.config.preset.value if self.config.preset else None,
                v2_preset=manifest.v2.preset if manifest.v2 else "",
                v2_device=self.config.v2_device,
                v2_upscaler_backend=self.config.v2_upscaler_backend,
            )

            if current_fp.depth_only() != manifest_fp.depth_only():
                logger.info("Depth config changed - regenerating")
                return False

            # File Integrity Check
            if not manifest.depth:
                return False

            # Quick read check
            from .depth_writer import read_depth_u16_png
            d = read_depth_u16_png(depth_path)
            if d.ndim != 2:
                return False

            logger.debug(f"Resuming with existing depth: {depth_path}")
            return True
        except Exception:
            return False

    def should_skip_v2(
        self,
        v2_report_path: Optional[Path],
        manifest_path: Path,
        image_input: ImageInput,
        depth_was_skipped: bool
    ) -> bool:
        if not v2_report_path or not v2_report_path.exists() or not manifest_path.exists():
            return False

        try:
            manifest = CombinedManifest.load(manifest_path)

            # Config Check
            if not manifest.config_fingerprint:
                return False

            current_fp = self.compute_config_fingerprint()
            manifest_fp = ConfigFingerprint(
                model_variant=manifest.depth.model if manifest.depth else "",
                depth_quantization=(
                    manifest.depth.scaling.get("method", "") if manifest.depth else ""
                ),
                depth_device=self.config.depth_device,
                preset=self.config.preset.value if self.config.preset else None,
                v2_preset=manifest.v2.preset if manifest.v2 else "",
                v2_device=self.config.v2_device,
                v2_upscaler_backend=self.config.v2_upscaler_backend,
            )

            if current_fp.v2_only() != manifest_fp.v2_only():
                return False

            # Consistency Check
            if not depth_was_skipped:
                logger.info("Depth was regenerated - V2 must rerun")
                return False

            if not manifest.v2 or manifest.v2.status != "ok":
                return False

            return True
        except Exception:
            return False

    def enhance_image(self, image_input: ImageInput, input_root: Optional[Path] = None) -> Dict[str, Any]:
        """Process a single image through V3 depth generation + optional V2 enhancement.

        Workflow:
        1. Validate: Call preprocessing.validate_image_format
        2. Preprocess: Call preprocessing.preprocess_image
        3. Inference: Pass preprocessed data to DA3InferenceEngine.predict
        4. Save: Pass result to depth_writer.atomic_write_depth_u16_png_with_stats
        5. Legacy (Optional): If configured, invoke V2Runner.run

        Args:
            image_input: ImageInput wrapper containing path to input image
            input_root: Optional root directory for relative path calculation

        Returns:
            Dict containing status, paths, and runtime information
        """
        from .preprocessing import validate_image_format, preprocess_image

        if input_root:
            output_key = make_output_key(image_input.path, input_root)
        else:
            output_key = Path(sanitize_file_stem(image_input.path.stem))
        logger.info(f"Processing {output_key}...")

        # Paths
        depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.png"
        manifest_path = self.manifests_dir / output_key.parent / f"{output_key.name}_combined.json"
        v2_log_path = self.logs_dir / output_key.parent / f"v2_{output_key.name}.log"

        # Ensure dirs
        for p in [depth_path, manifest_path, v2_log_path]:
            p.parent.mkdir(parents=True, exist_ok=True)

        # --- STEP 1: VALIDATE ---
        try:
            validated_path = validate_image_format(image_input.path)
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError(f"Image validation failed for {image_input.path}") from e

        # --- STEP 2: PREPROCESS ---
        try:
            preprocessed_image, original_shape = preprocess_image(validated_path)
        except Exception as e:
            raise RuntimeError(f"Image preprocessing failed for {image_input.path}") from e

        # --- STAGE A: DEPTH ---
        skip_depth = not self.config.force_depth and self.should_skip_depth(depth_path, manifest_path, image_input)
        depth_runtime_s = 0.0
        depth_metadata = None

        if not skip_depth:
            logger.info(f"Stage A: Generating depth for {output_key}...")
            t0 = time.time()
            try:
                # 3. Inference - pass preprocessed numpy array directly
                result = self.inference_engine.predict(preprocessed_image)

                # 4. Post-Processing (Refinement)
                result = self.postprocessor.process(result)

                depth_runtime_s = time.time() - t0

                # 5. Write
                _, _, depth_stats = atomic_write_depth_u16_png_with_stats(
                    depth_path,
                    result.depth,
                    method=self.config.depth_quantization,
                    debug_verify=self.config.verify_depth_writes
                )

                depth_metadata = DepthMetadata(
                    model=self.config.model_variant.value.name,
                    depth_path=str(depth_path),
                    runtime_seconds=depth_runtime_s,
                    scaling=depth_stats._asdict(),
                    stats={
                        "original_shape": original_shape,
                        "processed_shape": list(result.depth.shape[:2]),
                        "dtype": "uint16",
                        "non_commercial_ok": self.config.non_commercial_ok,
                    }
                )
            except Exception as e:
                logger.error(f"Depth inference failed: {e}")
                if self.config.depth_fallback == "fail":
                    raise RuntimeError(f"V3 depth inference failed for {image_input.path}") from e
                if self.config.depth_fallback == "skip":
                    return {"status": "skipped", "reason": str(e), "image": str(image_input.path)}
                if self.config.depth_fallback == "v2-auto":
                    logger.warning(f"V3 depth failed, attempting V2 fallback: {e}")
                    if depth_path.exists():
                        depth_path.unlink()
                    depth_path = None
        else:
            if manifest_path.exists():
                try:
                    depth_metadata = CombinedManifest.load(manifest_path).depth
                except Exception:
                    pass

        # --- STAGE B: V2 ENHANCE ---

        v2_report_path = find_v2_report(self.v2_dir, output_key.name)
        skip_v2 = not self.config.force_v2 and self.should_skip_v2(v2_report_path, manifest_path, image_input, skip_depth)
        v2_runtime_s = 0.0

        if skip_v2:
            logger.info("V2 outputs valid, skipping.")
            v2_result = {"status": "ok"}
        else:
            try:
                v2_result = self.v2_runner.run(
                    input_path=image_input.path,
                    depth_dir=self.depth_dir if depth_path else None,
                    output_dir=self.v2_dir,
                    preset=self.config.v2_preset,
                    device=self.config.v2_device,
                    upscaler_backend=self.config.v2_upscaler_backend,
                    log_file=v2_log_path,
                    timeout=self.config.v2_timeout
                )
                v2_runtime_s = v2_result.get("runtime_s", 0.0)
                v2_report_path = find_v2_report(self.v2_dir, output_key.name)
            except (FileNotFoundError, RuntimeError, TimeoutError) as e:
                # V2 fallback failed - if V3 also failed, raise with full context
                if depth_path is None:
                    raise RuntimeError(
                        f"Both V3 and V2 pipelines failed for {image_input.path}. "
                        f"V2 error: {e}"
                    ) from e
                # V3 succeeded but V2 failed - log warning but continue
                logger.warning(f"V2 enhancement failed (V3 depth available): {e}")
                v2_result = {"status": "error", "error": str(e)}

        # Manifest
        v2_metadata = V2Metadata(
            preset=self.config.v2_preset,
            status=v2_result.get("status", "ok"),
            runtime_seconds=v2_runtime_s,
            output_paths=[str(self.v2_dir)],
        )

        input_sha = self._compute_or_skip_hash(image_input.path)

        import datetime as dt
        manifest = CombinedManifest(
            input=InputMetadata(
                image_path=str(image_input.path),
                image_sha256=input_sha,
            ),
            depth=depth_metadata,
            v2=v2_metadata,
            timing=TimingMetadata(
                total_seconds=depth_runtime_s + v2_runtime_s,
                depth_seconds=depth_runtime_s,
                v2_seconds=v2_runtime_s,
                timestamp_utc=dt.datetime.utcnow().isoformat(),
            ),
            repro=ReproMetadata(
                v3_git_revision=self.v3_git,
                v2_git_revision=self.v2_git,
                environment=self.environment,
            ),
            config_fingerprint=self.compute_config_fingerprint(),
        )
        manifest.save(manifest_path)

        return {
            "status": "ok",
            "image": str(image_input.path),
            "depth_path": str(depth_path) if depth_path else None,
            "manifest": str(manifest_path),
            "runtime_s": depth_runtime_s + v2_runtime_s
        }

    def enhance_batch(self, input_dir: Path, image_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

        batch_id = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        logger.info(f"Batch {batch_id}: Scanning {input_dir}")

        images = []
        for ext in image_extensions:
            images.extend(input_dir.rglob(f"*{ext}"))
            images.extend(input_dir.rglob(f"*{ext.upper()}"))

        results = []
        for img in sorted(images):
            try:
                results.append(self.enhance_image(ImageInput(img), input_root=input_dir))
            except Exception as e:
                logger.error(f"Failed {img}: {e}")
                results.append({"status": "error", "image": str(img), "error": str(e)})

        # Write batch summary
        runtime_stats = compute_batch_runtime_stats(results)
        bm = BatchManifest(
            batch_id, datetime.datetime.now().isoformat(), datetime.datetime.now().isoformat(),
            {"model": self.config.model_variant.value.name},
            results,
            {"total": len(results), **runtime_stats}
        )
        bm.write(self.manifests_dir / f"batch_{batch_id}.json")
        return results
