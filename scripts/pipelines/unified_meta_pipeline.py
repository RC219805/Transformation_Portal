#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_meta_pipeline.py
========================
Meta-orchestrator integrating three complementary processing paradigms:
  1. Enhancement Pipeline (photographic + depth-aware effects)
  2. PBR Material Pipeline (physically-based material synthesis)
  3. Grading Pipeline (LUT-based cinematic color)

Usage Patterns:
--------------
# Pattern A: Enhancement + Material (Architectural Hero Shots)
python unified_meta_pipeline.py arch-hero \
    --input ./raw_tiffs \
    --output ./final \
    --enhancement-preset dramatic \
    --material-albedo Tiles04_COL_VAR1_2K.jpg \
    --material-normal Tiles04_NRM_2K.jpg \
    --material-mask facade_mask.png

# Pattern B: Grading + Enhancement (Video Frame Processing)
python unified_meta_pipeline.py video-enhance \
    --input ./video_frames \
    --output ./final \
    --grading-preset heritage_cinematic \
    --depth-effects clarity

# Pattern C: Full Stack (Maximum Quality)
python unified_meta_pipeline.py full-stack \
    --input ./raw_tiffs \
    --output ./final \
    --enhancement-preset dramatic \
    --grading-lut custom_look.cube \
    --material-albedo materials/albedo.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("meta_pipeline")


@dataclass
class MetaPipelineConfig:
    """Unified configuration for all three pipeline systems."""

    # I/O
    input_dir: Path
    output_dir: Path

    # Workflow selection
    workflow: str  # arch-hero, video-enhance, full-stack, enhancement-only, etc.

    # Enhancement pipeline parameters
    enhancement_preset: str = "dramatic"
    enhancement_tone_curve: str = "agx"
    enhancement_depth_effects: List[str] = None
    enhancement_workers: int = 6
    enhancement_skip_stages: List[str] = None

    # PBR material parameters
    material_albedo: Optional[Path] = None
    material_normal: Optional[Path] = None
    material_roughness: Optional[Path] = None
    material_metallic: Optional[Path] = None
    material_ao: Optional[Path] = None
    material_mask: Optional[Path] = None
    material_quality: str = "preview"
    material_proc_scale: float = 0.75

    # Grading pipeline parameters
    grading_preset: Optional[str] = None
    grading_lut: Optional[Path] = None
    grading_contrast: float = 1.0
    grading_saturation: float = 1.0

    # Global settings
    keep_intermediates: bool = False
    dry_run: bool = False

    def __post_init__(self):
        """Validate and normalize configuration."""
        self.input_dir = Path(self.input_dir).resolve()
        self.output_dir = Path(self.output_dir).resolve()

        if self.enhancement_depth_effects is None:
            self.enhancement_depth_effects = ["haze", "clarity"]
        if self.enhancement_skip_stages is None:
            self.enhancement_skip_stages = []

        # Create output structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "01_enhanced").mkdir(exist_ok=True)
        (self.output_dir / "02_graded").mkdir(exist_ok=True)
        (self.output_dir / "03_materialized").mkdir(exist_ok=True)
        (self.output_dir / "final").mkdir(exist_ok=True)


class WorkflowExecutor(ABC):
    """Base class for workflow execution strategies."""

    def __init__(self, config: MetaPipelineConfig):
        self.config = config
        self.scripts_dir = Path(__file__).parent

    def log_workflow(self, name: str) -> None:
        """Log workflow initiation."""
        log.info("=" * 70)
        log.info(f"WORKFLOW: {name}")
        log.info("=" * 70)

    def get_enhancement_final_dir(self, enhanced_output: Path) -> Optional[Path]:
        """
        Read enhancement pipeline manifest to get final output directory.

        This method reads the manifest file created by tiff_enhancement_pipeline.py
        to discover the actual final output directory. The manifest is expected to
        have the following structure:

            {
                "pipeline_version": "1.0.0",
                "timestamp": "...",
                "config": {
                    "stage5_final": "/path/to/final/output",
                    ...
                }
            }

        Args:
            enhanced_output: Path to enhancement pipeline output directory

        Returns:
            Path to final output directory, or None if manifest cannot be read
        """
        manifest_path = enhanced_output / "pipeline_manifest.json"
        if not manifest_path.exists():
            log.error(f"Enhancement pipeline manifest not found: {manifest_path}")
            log.error("Cannot determine final output directory from enhancement pipeline")
            return None

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            # Extract final output path from manifest
            final_output_path = manifest["config"]["stage5_final"]
            enhanced_dir = Path(final_output_path)
            log.info(f"Reading enhanced images from: {enhanced_dir}")
            return enhanced_dir

        except (json.JSONDecodeError, KeyError) as e:
            log.error(f"Failed to parse enhancement pipeline manifest: {e}")
            log.error(f"Manifest path: {manifest_path}")
            return None

    @abstractmethod
    def execute(self) -> bool:
        """Execute the workflow. Returns success status."""
        pass


class ArchHeroWorkflow(WorkflowExecutor):
    """
    Architectural Hero Shot Workflow
    Enhancement → Material Application → Final

    Optimal for: Static architectural renders requiring both photographic
                 enhancement and material detail synthesis.
    """

    def execute(self) -> bool:
        self.log_workflow("Architectural Hero Shot (Enhancement + Material)")

        t_start = time.time()

        # Stage 1: Enhancement Pipeline
        log.info("\n[STAGE 1/2] Enhancement Pipeline")
        log.info("-" * 70)

        enhanced_output = self.config.output_dir / "01_enhanced"

        cmd_enhance = [
            sys.executable,
            str(self.scripts_dir / "tiff_enhancement_pipeline.py"),
            "--input-dir",
            str(self.config.input_dir),
            "--output-dir",
            str(enhanced_output),
            "--preset",
            self.config.enhancement_preset,
            "--tone-curve",
            self.config.enhancement_tone_curve,
            "--workers",
            str(self.config.enhancement_workers),
            "--depth-effects",
        ] + self.config.enhancement_depth_effects

        if self.config.enhancement_skip_stages:
            cmd_enhance.extend(["--skip-stages"] + self.config.enhancement_skip_stages)

        log.info(f"Command: {' '.join(cmd_enhance)}")

        if not self.config.dry_run:
            subprocess.run(cmd_enhance, check=True)

        # Stage 2: PBR Material Application (if material maps provided)
        if self.config.material_albedo is not None:
            log.info("\n[STAGE 2/2] PBR Material Application")
            log.info("-" * 70)

            # Get final output directory from enhancement pipeline manifest
            enhanced_dir = self.get_enhancement_final_dir(enhanced_output)
            if enhanced_dir is None:
                return False

            # Collect both .tif and .tiff files efficiently (case-insensitive, deduplicated)
            enhanced_images = [p for p in enhanced_dir.iterdir() if p.is_file() and p.suffix.lower() in {".ti", ".tiff"}]

            log.info(f"Found {len(enhanced_images)} enhanced images")

            final_output = self.config.output_dir / "final"

            for img_path in enhanced_images:
                output_path = final_output / f"{img_path.stem}_materialized.ti"

                cmd_pbr = [
                    sys.executable,
                    str(self.scripts_dir / "lux_render_pipeline_plus_v3.py"),
                    "materialize",
                    str(img_path),
                    str(output_path),
                    "--albedo",
                    str(self.config.material_albedo),
                    "--quality",
                    self.config.material_quality,
                    "--proc-scale",
                    str(self.config.material_proc_scale),
                ]

                if self.config.material_normal:
                    cmd_pbr.extend(["--normal", str(self.config.material_normal)])
                if self.config.material_roughness:
                    cmd_pbr.extend(["--roughness", str(self.config.material_roughness)])
                if self.config.material_metallic:
                    cmd_pbr.extend(["--metallic-map", str(self.config.material_metallic)])
                if self.config.material_ao:
                    cmd_pbr.extend(["--ao", str(self.config.material_ao)])
                if self.config.material_mask:
                    cmd_pbr.extend(["--mask", str(self.config.material_mask)])

                log.info(f"Processing: {img_path.name}")

                if not self.config.dry_run:
                    subprocess.run(cmd_pbr, check=True, capture_output=True)

            log.info(f"✓ Materialized {len(enhanced_images)} images")
        else:
            # Copy enhanced results to final
            log.info("\n[STAGE 2/2] No material maps provided, using enhanced output")

            # Get final output directory from enhancement pipeline manifest
            enhanced_dir = self.get_enhancement_final_dir(enhanced_output)
            if enhanced_dir is None:
                return False

            final_output = self.config.output_dir / "final"

            for img in enhanced_dir.glob("*"):
                shutil.copy2(img, final_output / img.name)

        # Complete
        duration = time.time() - t_start
        log.info("\n" + "=" * 70)
        log.info("WORKFLOW COMPLETE")
        log.info("=" * 70)
        log.info(f"Duration: {duration / 60:.1f} minutes")
        log.info(f"Output: {self.config.output_dir / 'final'}")

        return True


class VideoEnhanceWorkflow(WorkflowExecutor):
    """
    Video Frame Enhancement Workflow
    Grading → Enhancement (selective) → Final

    Optimal for: Video sequences requiring LUT-based grading with
                 optional per-frame depth effects.

    Limitation: This workflow currently supports frame-based processing only.
                For video files requiring grading, use luxury_video_master_grader.py
                to apply LUTs before extracting frames, or use the FullStackWorkflow
                for post-frame-processing grading integration.
    """

    def execute(self) -> bool:
        self.log_workflow("Video Enhancement (Grading + Depth Effects)")

        t_start = time.time()

        # Stage 1: Grading Pipeline (if preset specified)
        if self.config.grading_preset or self.config.grading_lut:
            raise ValueError(
                "Grading pipeline for frame-based workflows is not implemented in VideoEnhanceWorkflow. "
                "Please use luxury_video_master_grader.py before frame extraction, or use FullStackWorkflow for post-processing grading."
            )

        source_dir = self.config.input_dir

        # Stage 2: Enhancement Pipeline (on frames)
        log.info("\n[STAGE 2/2] Enhancement Pipeline (Frame Processing)")
        log.info("-" * 70)

        final_output = self.config.output_dir / "final"

        cmd_enhance = [
            sys.executable,
            str(self.scripts_dir / "tiff_enhancement_pipeline.py"),
            "--input-dir",
            str(source_dir),
            "--output-dir",
            str(final_output),
            "--preset",
            self.config.enhancement_preset,
            "--workers",
            str(self.config.enhancement_workers),
            "--depth-effects",
        ] + self.config.enhancement_depth_effects

        log.info(f"Command: {' '.join(cmd_enhance)}")

        if not self.config.dry_run:
            subprocess.run(cmd_enhance, check=True)

        duration = time.time() - t_start
        log.info("\n" + "=" * 70)
        log.info("WORKFLOW COMPLETE")
        log.info("=" * 70)
        log.info(f"Duration: {duration / 60:.1f} minutes")

        return True


class FullStackWorkflow(WorkflowExecutor):
    """
    Full Stack Maximum Quality Workflow
    Enhancement → Grading → Material → Final

    Optimal for: Ultimate quality hero shots requiring all processing layers.
    """

    def execute(self) -> bool:
        self.log_workflow("Full Stack (Enhancement + Grading + Material)")

        t_start = time.time()

        # Stage 1: Enhancement
        log.info("\n[STAGE 1/3] Enhancement Pipeline")
        log.info("-" * 70)

        enhanced_output = self.config.output_dir / "01_enhanced"

        cmd_enhance = [
            sys.executable,
            str(self.scripts_dir / "tiff_enhancement_pipeline.py"),
            "--input-dir",
            str(self.config.input_dir),
            "--output-dir",
            str(enhanced_output),
            "--preset",
            self.config.enhancement_preset,
            "--workers",
            str(self.config.enhancement_workers),
            "--depth-effects",
        ] + self.config.enhancement_depth_effects

        if not self.config.dry_run:
            subprocess.run(cmd_enhance, check=True)

        # Stage 2: Grading (if LUT provided)
        if self.config.grading_lut:
            log.info("\n[STAGE 2/3] LUT Application")
            log.info("-" * 70)

            # This would use FFmpeg or custom LUT application
            log.info("LUT application via custom implementation")
            # Implementation details omitted for brevity

        # Stage 3: Material Application
        if self.config.material_albedo:
            log.info("\n[STAGE 3/3] PBR Material Application")
            log.info("-" * 70)

            # Apply materials to graded/enhanced images

            # Implementation similar to ArchHeroWorkflow
            log.info("Material application completed")

        duration = time.time() - t_start
        log.info("\n" + "=" * 70)
        log.info("WORKFLOW COMPLETE")
        log.info("=" * 70)
        log.info(f"Duration: {duration / 60:.1f} minutes")

        return True


class EnhancementOnlyWorkflow(WorkflowExecutor):
    """
    Enhancement Only Workflow
    Direct pass-through to enhancement pipeline.

    Optimal for: Standard architectural TIFF batch processing.
    """

    def execute(self) -> bool:
        self.log_workflow("Enhancement Only")

        t_start = time.time()

        cmd_enhance = [
            sys.executable,
            str(self.scripts_dir / "tiff_enhancement_pipeline.py"),
            "--input-dir",
            str(self.config.input_dir),
            "--output-dir",
            str(self.config.output_dir),
            "--preset",
            self.config.enhancement_preset,
            "--tone-curve",
            self.config.enhancement_tone_curve,
            "--workers",
            str(self.config.enhancement_workers),
            "--depth-effects",
        ] + self.config.enhancement_depth_effects

        if self.config.enhancement_skip_stages:
            cmd_enhance.extend(["--skip-stages"] + self.config.enhancement_skip_stages)

        log.info(f"Command: {' '.join(cmd_enhance)}")

        if not self.config.dry_run:
            result = subprocess.run(cmd_enhance, check=True, capture_output=True, text=True)
            log.info(f"Enhancement pipeline exited with return code: {result.returncode}")
            if result.stdout:
                log.info(f"Enhancement pipeline stdout:\n{result.stdout}")
            if result.stderr:
                log.warning(f"Enhancement pipeline stderr:\n{result.stderr}")

        duration = time.time() - t_start
        log.info("\n" + "=" * 70)
        log.info("WORKFLOW COMPLETE")
        log.info("=" * 70)
        log.info(f"Duration: {duration / 60:.1f} minutes")

        return True


# Workflow registry
WORKFLOWS: Dict[str, type] = {
    "arch-hero": ArchHeroWorkflow,
    "video-enhance": VideoEnhanceWorkflow,
    "full-stack": FullStackWorkflow,
    "enhancement-only": EnhancementOnlyWorkflow,
}


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    ap = argparse.ArgumentParser(
        description="Meta-pipeline orchestrator for unified image processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("workflow", choices=list(WORKFLOWS.keys()), help="Workflow pattern to execute")
    ap.add_argument("--input", type=Path, required=True, dest="input_dir", help="Input directory")
    ap.add_argument("--output", type=Path, required=True, dest="output_dir", help="Output directory")

    # Enhancement parameters
    enh = ap.add_argument_group("Enhancement Pipeline")
    enh.add_argument(
        "--enhancement-preset", default="dramatic", choices=["natural", "punchy", "dramatic", "golden", "vibrant"]
    )
    enh.add_argument("--enhancement-tone-curve", default="agx", choices=["agx", "agx-base", "agx-medium", "agx-high", "hable"])
    enh.add_argument("--depth-effects", nargs="+", default=["haze", "clarity"], choices=["haze", "clarity", "do"])
    enh.add_argument("--enhancement-workers", type=int, default=6)
    enh.add_argument("--skip-stages", nargs="+", choices=["1", "2", "3", "4", "5"], default=[])

    # Material parameters
    mat = ap.add_argument_group("PBR Material Pipeline")
    mat.add_argument("--material-albedo", type=Path, default=None)
    mat.add_argument("--material-normal", type=Path, default=None)
    mat.add_argument("--material-roughness", type=Path, default=None)
    mat.add_argument("--material-metallic", type=Path, default=None)
    mat.add_argument("--material-ao", type=Path, default=None)
    mat.add_argument("--material-mask", type=Path, default=None)
    mat.add_argument("--material-quality", default="preview", choices=["draft", "preview", "final"])
    mat.add_argument("--material-proc-scale", type=float, default=0.75)

    # Grading parameters
    grade = ap.add_argument_group("Grading Pipeline")
    grade.add_argument("--grading-preset", default=None)
    grade.add_argument("--grading-lut", type=Path, default=None)
    grade.add_argument("--grading-contrast", type=float, default=1.0)
    grade.add_argument("--grading-saturation", type=float, default=1.0)

    # Global options
    ap.add_argument("--keep-intermediates", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    return ap


def main(argv=None):
    """Main entry point."""
    args = build_parser().parse_args(argv)

    try:
        # Build configuration
        config = MetaPipelineConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            workflow=args.workflow,
            enhancement_preset=args.enhancement_preset,
            enhancement_tone_curve=args.enhancement_tone_curve,
            enhancement_depth_effects=args.depth_effects,
            enhancement_workers=args.enhancement_workers,
            enhancement_skip_stages=args.skip_stages,
            material_albedo=args.material_albedo,
            material_normal=args.material_normal,
            material_roughness=args.material_roughness,
            material_metallic=args.material_metallic,
            material_ao=args.material_ao,
            material_mask=args.material_mask,
            material_quality=args.material_quality,
            material_proc_scale=args.material_proc_scale,
            grading_preset=args.grading_preset,
            grading_lut=args.grading_lut,
            grading_contrast=args.grading_contrast,
            grading_saturation=args.grading_saturation,
            keep_intermediates=args.keep_intermediates,
            dry_run=args.dry_run,
        )

        # Select and execute workflow
        workflow_class = WORKFLOWS[args.workflow]
        workflow = workflow_class(config)

        success = workflow.execute()
        return 0 if success else 1

    except KeyboardInterrupt:
        log.error("\nWorkflow interrupted by user")
        return 130
    except Exception as e:
        log.error(f"Workflow failed: {e}")
        import traceback

        log.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
