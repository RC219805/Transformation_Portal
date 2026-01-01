"""Orchestrator for V3 depth + V2 enhancement pipeline.

Two-stage pipeline:
1. Stage A (V3): Generate depth assets using DA3
2. Stage B (V2): Consume depth assets → weights → grade → upscale → export → report
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import time
import logging

from lux_depth_v3.config import DA3Config, ModelVariant, Preset
from lux_depth_v3.inference import DA3InferenceEngine
from lux_depth_v3.input_manager import InputManager, ImageInput
from lux_depth_v3.enhance.depth_writer import write_depth_u16_png
from lux_depth_v3.enhance.v2_runner import V2Runner, find_v2_report
from lux_depth_v3.enhance.manifest import (
    CombinedManifest,
    InputMetadata,
    DepthMetadata,
    V2Metadata,
    TimingMetadata,
    ReproMetadata,
    compute_file_sha256,
    get_git_revision,
)

logger = logging.getLogger(__name__)


@dataclass
class EnhanceConfig:
    """Configuration for enhance orchestrator."""
    # Model config
    model_variant: ModelVariant = ModelVariant.METRIC_LARGE
    preset: Optional[Preset] = None

    # V2 config
    v2_preset: str = "production_ultra"
    v2_device: str = "auto"
    v2_upscaler_backend: str = "torch"

    # Depth config
    depth_device: str = "auto"
    depth_quantization: str = "p1p99"

    # Execution control
    execution_mode: str = "sequential"  # "sequential" or "pipelined"
    depth_fallback: str = "fail"  # "fail", "skip", "v2-auto"
    force_depth: bool = False
    force_v2: bool = False

    # License
    non_commercial_ok: bool = False

    # Timeout
    v2_timeout: Optional[float] = 600.0  # 10 minutes default


class EnhanceOrchestrator:
    """Orchestrates V3 depth generation + V2 enhancement pipeline."""

    def __init__(self, config: EnhanceConfig, output_root: Path):
        """Initialize orchestrator.

        Args:
            config: Enhance configuration
            output_root: Root output directory
        """
        self.config = config
        self.output_root = Path(output_root)

        # Create output directories
        self.depth_dir = self.output_root / "depth"
        self.v2_dir = self.output_root / "v2"
        self.manifests_dir = self.output_root / "manifests"
        self.logs_dir = self.output_root / "logs"

        for dir_path in [self.depth_dir, self.v2_dir, self.manifests_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize V3 inference engine
        da3_config = DA3Config(
            model_variant=config.model_variant,
            preset=config.preset,
        )
        da3_config.device.device = config.depth_device

        self.inference_engine = DA3InferenceEngine(
            config=da3_config,
            commercial_use=not config.non_commercial_ok,
            validate_license_strict=True,
        )

        # Initialize V2 runner
        self.v2_runner = V2Runner()

        # Track git revisions for reproducibility
        self.v3_git = get_git_revision(Path(__file__).parent.parent)
        self.v2_git = get_git_revision(self.output_root.parent / "lux_depth_v2")

    def enhance_image(
        self,
        image_input: ImageInput,
    ) -> Dict[str, Any]:
        """Process single image through V3 + V2 pipeline.

        Args:
            image_input: Input image metadata

        Returns:
            Dictionary with processing results and paths
        """
        stem = image_input.path.stem
        logger.info(f"Processing {stem}...")

        # Paths
        depth_path = self.depth_dir / f"{stem}_depth.png"
        depth_manifest_path = self.depth_dir / f"{stem}_depth_manifest.json"
        combined_manifest_path = self.manifests_dir / f"{stem}_combined.json"
        v2_log_path = self.logs_dir / f"v2_{stem}.log"

        # Check resume conditions
        skip_depth = False
        skip_v2 = False

        if depth_path.exists() and not self.config.force_depth:
            logger.info(f"Depth exists, skipping: {depth_path}")
            skip_depth = True

        # Stage A: Generate depth
        depth_result = None
        depth_runtime_s = 0.0
        depth_metadata = None

        if not skip_depth:
            logger.info(f"Stage A: Generating depth for {stem}...")
            start_time = time.time()

            try:
                depth_result = self.inference_engine.predict(image_input)
                depth_runtime_s = time.time() - start_time

                # Write depth
                p1, p99 = write_depth_u16_png(
                    depth_path,
                    depth_result.depth,
                    method=self.config.depth_quantization,
                    debug_verify=True,
                )

                # Create depth metadata
                depth_metadata = DepthMetadata(
                    backend="da3",
                    model=self.config.model_variant.value,
                    license="CC-BY-NC",
                    non_commercial_ok=self.config.non_commercial_ok,
                    depth_path=f"depth/{stem}_depth.png",
                    dtype="uint16",
                    shape=list(depth_result.depth.shape[:2]),
                    scaling={
                        "method": self.config.depth_quantization,
                        "p1": p1,
                        "p99": p99,
                    },
                    runtime_ms=depth_runtime_s * 1000,
                )

                logger.info(f"Depth generated in {depth_runtime_s:.2f}s")

            except Exception as e:
                logger.error(f"Depth generation failed: {e}")

                # Handle failure according to policy
                if self.config.depth_fallback == "fail":
                    raise
                elif self.config.depth_fallback == "skip":
                    logger.warning(f"Skipping {stem} due to depth failure")
                    return {
                        "status": "skipped",
                        "reason": f"depth_failed: {str(e)}",
                        "image": str(image_input.path),
                    }
                elif self.config.depth_fallback == "v2-auto":
                    logger.warning(f"Depth failed, V2 will auto-generate depth")
                    # Clear depth_dir for this image so V2 uses its own depth
                    # Also clean up any partially written depth file
                    if depth_path and depth_path.exists():
                        try:
                            depth_path.unlink()
                            logger.debug(f"Removed failed depth file: {depth_path}")
                        except Exception as unlink_exc:
                            logger.warning(f"Could not remove failed depth file: {unlink_exc}")
                    depth_path = None
                else:
                    raise ValueError(f"Unknown depth_fallback: {self.config.depth_fallback}")

        # Load existing depth metadata if skipped
        else:
            if depth_manifest_path.exists():
                # Load from previous run
                try:
                    manifest = CombinedManifest.load(depth_manifest_path)
                    depth_metadata = manifest.depth
                except Exception:
                    logger.warning("Could not load existing depth metadata")

        # Stage B: Run V2 enhancement
        logger.info(f"Stage B: Running V2 enhancement for {stem}...")
        v2_start_time = time.time()

        # Check if V2 outputs exist
        v2_report_path = find_v2_report(self.v2_dir, stem)
        if v2_report_path and not self.config.force_v2:
            logger.info(f"V2 outputs exist, skipping: {v2_report_path}")
            v2_runtime_s = 0.0
            v2_result = {"status": "ok"}
        else:
            v2_result = self.v2_runner.run(
                input_path=image_input.path,
                depth_dir=self.depth_dir if depth_path else None,
                output_dir=self.v2_dir,
                preset=self.config.v2_preset,
                device=self.config.v2_device,
                upscaler_backend=self.config.v2_upscaler_backend,
                log_file=v2_log_path,
                timeout=self.config.v2_timeout,
            )
            v2_runtime_s = v2_result.get("runtime_s", 0.0)

        # Find V2 report
        v2_report_path = find_v2_report(self.v2_dir, stem)

        # Create V2 metadata
        v2_metadata = V2Metadata(
            preset=self.config.v2_preset,
            strict_depth=depth_path is not None,
            output_dir="v2/",
            report_path=f"v2/{v2_report_path.name}" if v2_report_path else "",
            status=v2_result["status"],
            error_message=v2_result.get("error"),
        )

        # Compute input hash
        input_sha256 = compute_file_sha256(image_input.path)

        # Build combined manifest
        manifest = CombinedManifest(
            input=InputMetadata(
                image_path=str(image_input.path),
                image_sha256=input_sha256,
            ),
            depth=depth_metadata,
            v2=v2_metadata,
            timing=TimingMetadata(
                depth_s=depth_runtime_s,
                v2_s=v2_runtime_s,
                total_s=depth_runtime_s + v2_runtime_s,
            ),
            repro=ReproMetadata(
                v3_git=self.v3_git,
                v2_git=self.v2_git,
                device=self.config.depth_device,
            ),
        )

        # Write manifest
        manifest.write(combined_manifest_path)

        logger.info(f"Completed {stem} in {depth_runtime_s + v2_runtime_s:.2f}s")

        return {
            "status": "ok",
            "image": str(image_input.path),
            "depth_path": str(depth_path) if depth_path else None,
            "v2_report": str(v2_report_path) if v2_report_path else None,
            "manifest": str(combined_manifest_path),
            "runtime_s": depth_runtime_s + v2_runtime_s,
        }

    def enhance_batch(
        self,
        input_dir: Path,
        image_extensions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Process batch of images through V3 + V2 pipeline.

        Args:
            input_dir: Input directory
            image_extensions: Image extensions to process (default: common formats)

        Returns:
            List of results for each image
        """
        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

        # Collect images
        images = []
        for ext in image_extensions:
            images.extend(input_dir.glob(f"*{ext}"))
            images.extend(input_dir.glob(f"*{ext.upper()}"))

        logger.info(f"Found {len(images)} images in {input_dir}")

        # Process images
        results = []
        for img_path in sorted(images):
            image_input = ImageInput(path=img_path)
            try:
                result = self.enhance_image(image_input)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                results.append({
                    "status": "error",
                    "image": str(img_path),
                    "error": str(e),
                })

        # Summary
        succeeded = sum(1 for r in results if r["status"] == "ok")
        failed = sum(1 for r in results if r["status"] == "error")
        skipped = sum(1 for r in results if r["status"] == "skipped")

        logger.info(
            f"Batch complete: {succeeded} succeeded, {failed} failed, {skipped} skipped"
        )

        return results
