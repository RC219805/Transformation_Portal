#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tiff_enhancement_pipeline.py
============================
Comprehensive orchestration pipeline for dramatic enhancement of large 16/32-bit TIFFs.

Pipeline Stages:
1. HDR Enhancement (realize_v8_unified) - Initial color grading, exposure, tone curves
2. Depth Prediction (CoreML) - Generate depth maps optimized for Apple Silicon
3. AgX Tone Mapping - Professional tone mapping with auto-exposure
4. Panoptic Segmentation - Generate semantic masks (sky, building)
5. Depth Effects - Final atmospheric effects (haze, clarity, DOF)

Optimizations for M4 Max:
- Memory-mapped file operations for 100-200MB TIFFs
- Adaptive chunk sizing based on available RAM
- CoreML ANE acceleration for depth prediction
- Parallel processing where beneficial
- Automatic cleanup of intermediate files

Usage:
    python tiff_enhancement_pipeline.py \\
        --input-dir /path/to/tiffs \\
        --output-dir /path/to/outputs \\
        --preset dramatic \\
        --depth-effects haze clarity \\
        --workers 4
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback

# ----------------------------------------------------------
# Configuration & Logging
# ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("pipeline")


@dataclass
class PipelineConfig:
    """Pipeline configuration and paths."""
    # Input/Output
    input_dir: Path
    output_dir: Path
    
    # Stage directories
    stage1_enhance: Path  # realize_v8 output
    stage2_depth: Path    # depth maps
    stage3_tonemap: Path  # AgX tone-mapped
    stage4_masks: Path    # segmentation masks
    stage5_final: Path    # final depth effects
    
    # Processing options
    preset: str = "dramatic"
    tone_curve: str = "agx"
    ocio_config: Optional[str] = None
    depth_effects: List[str] = None  # ["haze", "clarity", "dof"]
    workers: int = 4
    device: str = "cpu"  # cpu/cuda/mps
    
    # Quality settings
    quality_jpeg: int = 95
    out_bitdepth: int = 16  # 8/16/32
    
    # Pipeline control
    skip_stages: List[str] = None
    keep_intermediates: bool = False
    dry_run: bool = False
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        self.input_dir = Path(self.input_dir).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        
        if self.depth_effects is None:
            self.depth_effects = ["haze", "clarity"]
        if self.skip_stages is None:
            self.skip_stages = []
            
        # Create stage directories
        for stage in ["stage1_enhance", "stage2_depth", "stage3_tonemap", 
                      "stage4_masks", "stage5_final"]:
            path = getattr(self, stage)
            if not self.dry_run:
                path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict:
        """Serialize config to dict."""
        d = asdict(self)
        # Convert Path objects to strings
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d
    
    def save_manifest(self, filepath: Path) -> None:
        """Save pipeline manifest to JSON."""
        manifest = {
            "pipeline_version": "1.0.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.to_dict()
        }
        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2)
        log.info(f"Saved pipeline manifest: {filepath}")


# ----------------------------------------------------------
# Stage Executors
# ----------------------------------------------------------

class StageExecutor:
    """Base class for pipeline stage execution."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stage_name = self.__class__.__name__
    
    def should_skip(self) -> bool:
        """Check if this stage should be skipped."""
        stage_num = self.stage_name.lower().replace("stage", "").replace("executor", "")
        return stage_num in self.config.skip_stages
    
    def log_start(self) -> None:
        """Log stage start."""
        log.info("=" * 70)
        log.info(f"STAGE: {self.stage_name}")
        log.info("=" * 70)
    
    def log_complete(self, duration: float, files_processed: int) -> None:
        """Log stage completion."""
        log.info(f"✓ {self.stage_name} complete")
        log.info(f"  Processed: {files_processed} files")
        log.info(f"  Duration: {duration:.1f}s")
        log.info("")
    
    def execute(self) -> Tuple[bool, int]:
        """
        Execute the stage.
        Returns (success, files_processed)
        """
        raise NotImplementedError


class Stage1Enhance(StageExecutor):
    """Stage 1: HDR Enhancement with realize_v8_unified."""
    
    def execute(self) -> Tuple[bool, int]:
        self.log_start()
        
        if self.should_skip():
            log.info("Skipping Stage 1 (realize_v8_unified)")
            return True, 0
        
        t0 = time.time()
        
        # Find realize_v8_unified.py
        script = Path(__file__).parent / "realize_v8_unified.py"
        if not script.exists():
            log.error(f"realize_v8_unified.py not found at {script}")
            return False, 0
        
        # Build command
        cmd = [
            sys.executable, str(script),
            "batch",
            str(self.config.input_dir),
            str(self.config.stage1_enhance),
            "--preset", self.config.preset,
            "--tone-curve", self.config.tone_curve,
            "--suffix", "_ENH",
            "--out-bitdepth", str(self.config.out_bitdepth),
            "--quality-jpeg", str(self.config.quality_jpeg),
            "--jobs", str(self.config.workers)
        ]
        
        if self.config.ocio_config:
            cmd.extend(["--ocio-config", self.config.ocio_config])
        
        log.info(f"Command: {' '.join(cmd)}")
        
        if self.config.dry_run:
            log.info("[DRY RUN] Would execute realize_v8_unified")
            return True, 0
        
        # Execute
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                log.debug(result.stdout)
            
            # Count output files
            files = list(self.config.stage1_enhance.glob("*_ENH.*"))
            
            self.log_complete(time.time() - t0, len(files))
            return True, len(files)
            
        except subprocess.CalledProcessError as e:
            log.error(f"realize_v8_unified failed: {e}")
            if e.stderr:
                log.error(e.stderr)
            return False, 0


class Stage2Depth(StageExecutor):
    """Stage 2: Depth Map Generation with CoreML."""
    
    def execute(self) -> Tuple[bool, int]:
        self.log_start()
        
        if self.should_skip():
            log.info("Skipping Stage 2 (depth_predict_coreml)")
            return True, 0
        
        t0 = time.time()
        
        # Find depth_predict_coreml.py
        script = Path(__file__).parent / "depth_predict_coreml.py"
        if not script.exists():
            log.error(f"depth_predict_coreml.py not found at {script}")
            return False, 0
        
        # Read and modify script to use our paths
        with open(script, 'r') as f:
            script_content = f.read()
        
        # Create temporary modified script
        temp_script = Path("/tmp/depth_predict_temp.py")
        modified = script_content.replace(
            'IN_DIR     = "/Users/rc/Desktop/my_project/images/750_Picacho"',
            f'IN_DIR     = "{self.config.stage1_enhance}"'
        ).replace(
            'OUT_DIR    = "/Users/rc/Desktop/my_project/outputs/depth/750_Picacho"',
            f'OUT_DIR    = "{self.config.stage2_depth}"'
        )
        
        with open(temp_script, 'w') as f:
            f.write(modified)
        
        log.info(f"Running depth prediction on {self.config.stage1_enhance}")
        
        if self.config.dry_run:
            log.info("[DRY RUN] Would execute depth_predict_coreml")
            return True, 0
        
        # Execute
        try:
            result = subprocess.run(
                [sys.executable, str(temp_script)],
                check=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                log.debug(result.stdout)
            
            # Count depth maps
            depth_maps = list(self.config.stage2_depth.glob("*_depth16.png"))
            
            # Cleanup temp script
            temp_script.unlink()
            
            self.log_complete(time.time() - t0, len(depth_maps))
            return True, len(depth_maps)
            
        except subprocess.CalledProcessError as e:
            log.error(f"depth_predict_coreml failed: {e}")
            if e.stderr:
                log.error(e.stderr)
            return False, 0


class Stage3Tonemap(StageExecutor):
    """Stage 3: AgX Tone Mapping with auto-exposure."""
    
    def execute(self) -> Tuple[bool, int]:
        self.log_start()
        
        if self.should_skip():
            log.info("Skipping Stage 3 (agx_batch_processor)")
            return True, 0
        
        t0 = time.time()
        
        # Find agx_batch_processor.py
        script = Path(__file__).parent / "agx_batch_processor.py"
        if not script.exists():
            log.error(f"agx_batch_processor.py not found at {script}")
            return False, 0
        
        # Build command
        cmd = [
            sys.executable, str(script),
            "--input-dir", str(self.config.stage1_enhance),
            "--output-dir", str(self.config.stage3_tonemap),
            "--tone", self.config.tone_curve,
            "--contrast", "1.10",
            "--saturation", "1.05",
            "--auto-exposure", "logmean",
            "--workers", str(self.config.workers),
            "--quality", str(self.config.quality_jpeg)
        ]
        
        if self.config.ocio_config:
            cmd.extend(["--ocio-config", self.config.ocio_config])
        
        log.info(f"Command: {' '.join(cmd)}")
        
        if self.config.dry_run:
            log.info("[DRY RUN] Would execute agx_batch_processor")
            return True, 0
        
        # Execute
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                log.debug(result.stdout)
            
            # Count output files
            files = list(self.config.stage3_tonemap.glob("*"))
            
            self.log_complete(time.time() - t0, len(files))
            return True, len(files)
            
        except subprocess.CalledProcessError as e:
            log.error(f"agx_batch_processor failed: {e}")
            if e.stderr:
                log.error(e.stderr)
            return False, 0


class Stage4Segmentation(StageExecutor):
    """Stage 4: Panoptic Segmentation for masks."""
    
    def execute(self) -> Tuple[bool, int]:
        self.log_start()
        
        if self.should_skip():
            log.info("Skipping Stage 4 (run_detectron2_panoptic_batch)")
            return True, 0
        
        t0 = time.time()
        
        # Find run_detectron2_panoptic_batch.py
        script = Path(__file__).parent / "run_detectron2_panoptic_batch.py"
        if not script.exists():
            log.error(f"run_detectron2_panoptic_batch.py not found at {script}")
            return False, 0
        
        # Build command
        cmd = [
            sys.executable, str(script),
            "--images-root", str(self.config.stage3_tonemap),
            "--depths-root", str(self.config.stage2_depth),
            "--mask-root", str(self.config.stage4_masks),
            "--device", self.config.device
        ]
        
        log.info(f"Command: {' '.join(cmd)}")
        
        if self.config.dry_run:
            log.info("[DRY RUN] Would execute run_detectron2_panoptic_batch")
            return True, 0
        
        # Execute
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                log.debug(result.stdout)
            
            # Count masks
            masks = list(self.config.stage4_masks.glob("*_mask_*.png"))
            
            self.log_complete(time.time() - t0, len(masks))
            return True, len(masks)
            
        except subprocess.CalledProcessError as e:
            log.error(f"run_detectron2_panoptic_batch failed: {e}")
            if e.stderr:
                log.error(e.stderr)
            return False, 0


class Stage5DepthEffects(StageExecutor):
    """Stage 5: Final depth-aware atmospheric effects."""
    
    def execute(self) -> Tuple[bool, int]:
        self.log_start()
        
        if self.should_skip():
            log.info("Skipping Stage 5 (depth_tools)")
            return True, 0
        
        # Find depth_tools.py
        script = Path(__file__).parent / "depth_tools.py"
        if not script.exists():
            log.error(f"depth_tools.py not found at {script}")
            return False, 0
        
        total_files = 0
        
        for effect in self.config.depth_effects:
            log.info(f"Applying depth effect: {effect}")
            t0 = time.time()
            
            # Build command
            cmd = [
                sys.executable, str(script),
                effect,
                str(self.config.stage3_tonemap),
                str(self.config.stage2_depth),
                str(self.config.stage5_final),
                "--mask-root", str(self.config.stage4_masks),
                "--fmt", "tiff" if self.config.out_bitdepth > 8 else "jpg",
                "--workers", str(self.config.workers)
            ]
            
            # Add effect-specific parameters
            if effect == "haze":
                cmd.extend([
                    "--strength", "0.20",
                    "--near", "15.0",
                    "--far", "85.0"
                ])
            elif effect == "clarity":
                cmd.extend([
                    "--amount", "0.15",
                    "--radius", "3"
                ])
            elif effect == "dof":
                cmd.extend([
                    "--focus", "35.0",
                    "--aperture", "0.25"
                ])
            
            log.info(f"Command: {' '.join(cmd)}")
            
            if self.config.dry_run:
                log.info(f"[DRY RUN] Would execute depth_tools {effect}")
                continue
            
            # Execute
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if result.stdout:
                    log.debug(result.stdout)
                
                log.info(f"  ✓ {effect} effect applied ({time.time() - t0:.1f}s)")
                
            except subprocess.CalledProcessError as e:
                log.error(f"depth_tools {effect} failed: {e}")
                if e.stderr:
                    log.error(e.stderr)
                return False, 0
        
        # Count final outputs
        files = list(self.config.stage5_final.glob("*"))
        total_files = len(files)
        
        self.log_complete(time.time() - t0, total_files)
        return True, total_files


# ----------------------------------------------------------
# Pipeline Orchestrator
# ----------------------------------------------------------

class Pipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stages = [
            Stage1Enhance(config),
            Stage2Depth(config),
            Stage3Tonemap(config),
            Stage4Segmentation(config),
            Stage5DepthEffects(config)
        ]
    
    def validate_environment(self) -> bool:
        """Validate required dependencies and paths."""
        log.info("Validating environment...")
        
        # Check input directory
        if not self.config.input_dir.exists():
            log.error(f"Input directory not found: {self.config.input_dir}")
            return False
        
        # Check for TIFF files
        tiff_files = list(self.config.input_dir.glob("*.tif")) + \
                     list(self.config.input_dir.glob("*.tiff")) + \
                     list(self.config.input_dir.glob("*.TIF")) + \
                     list(self.config.input_dir.glob("*.TIFF"))
        
        if not tiff_files:
            log.error(f"No TIFF files found in {self.config.input_dir}")
            return False
        
        log.info(f"Found {len(tiff_files)} TIFF files")
        
        # Check Python dependencies
        required = ["numpy", "PIL", "coremltools"]
        missing = []
        for mod in required:
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
        
        if missing:
            log.error(f"Missing required Python packages: {', '.join(missing)}")
            return False
        
        log.info("✓ Environment validated")
        return True
    
    def execute(self) -> bool:
        """Execute the complete pipeline."""
        log.info("")
        log.info("╔" + "═" * 68 + "╗")
        log.info("║" + " TIFF ENHANCEMENT PIPELINE ".center(68) + "║")
        log.info("╚" + "═" * 68 + "╝")
        log.info("")
        
        # Validate
        if not self.validate_environment():
            return False
        
        # Save manifest
        manifest_path = self.config.output_dir / "pipeline_manifest.json"
        self.config.save_manifest(manifest_path)
        
        # Execute stages
        pipeline_start = time.time()
        
        for i, stage in enumerate(self.stages, 1):
            log.info(f"[{i}/{len(self.stages)}] Executing {stage.stage_name}...")
            
            success, files = stage.execute()
            
            if not success:
                log.error(f"Pipeline failed at {stage.stage_name}")
                return False
        
        # Pipeline complete
        total_duration = time.time() - pipeline_start
        
        log.info("")
        log.info("=" * 70)
        log.info("PIPELINE COMPLETE")
        log.info("=" * 70)
        log.info(f"Total duration: {total_duration / 60:.1f} minutes")
        log.info(f"Output directory: {self.config.stage5_final}")
        log.info("")
        
        # Cleanup intermediates if requested
        if not self.config.keep_intermediates:
            log.info("Cleaning up intermediate files...")
            for stage in ["stage1_enhance", "stage2_depth", "stage3_tonemap", "stage4_masks"]:
                stage_path = getattr(self.config, stage)
                if stage_path.exists() and stage_path != self.config.stage5_final:
                    shutil.rmtree(stage_path)
                    log.info(f"  Removed: {stage_path}")
        
        return True


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    ap = argparse.ArgumentParser(
        description="Comprehensive TIFF enhancement pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Input directory containing 16/32-bit TIFFs")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Output directory for final enhanced images")
    
    # Enhancement options
    ap.add_argument("--preset", choices=["natural", "punchy", "dramatic", "golden", "vibrant"],
                    default="dramatic", help="Enhancement preset")
    ap.add_argument("--tone-curve", choices=["agx", "agx-base", "agx-medium", "agx-high", "hable"],
                    default="agx", help="Tone mapping method")
    ap.add_argument("--ocio-config", type=str, default=None,
                    help="Path to OpenColorIO config for AgX")
    
    # Depth effects
    ap.add_argument("--depth-effects", nargs="+",
                    choices=["haze", "clarity", "dof"],
                    default=["haze", "clarity"],
                    help="Depth-based effects to apply")
    
    # Performance
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of parallel workers")
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"],
                    default="cpu", help="Device for neural networks")
    
    # Quality
    ap.add_argument("--quality", type=int, default=95,
                    help="JPEG quality (1-100)")
    ap.add_argument("--bitdepth", type=int, choices=[8, 16, 32],
                    default=16, help="Output bit depth")
    
    # Pipeline control
    ap.add_argument("--skip-stages", nargs="+",
                    choices=["1", "2", "3", "4", "5"],
                    default=[], help="Stages to skip")
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="Keep intermediate files from each stage")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be executed without running")
    
    return ap


def main(argv=None):
    """Main entry point."""
    args = build_parser().parse_args(argv)
    
    try:
        # Build configuration
        config = PipelineConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            stage1_enhance=args.output_dir / "01_enhance",
            stage2_depth=args.output_dir / "02_depth",
            stage3_tonemap=args.output_dir / "03_tonemap",
            stage4_masks=args.output_dir / "04_masks",
            stage5_final=args.output_dir / "05_final",
            preset=args.preset,
            tone_curve=args.tone_curve,
            ocio_config=args.ocio_config,
            depth_effects=args.depth_effects,
            workers=args.workers,
            device=args.device,
            quality_jpeg=args.quality,
            out_bitdepth=args.bitdepth,
            skip_stages=args.skip_stages,
            keep_intermediates=args.keep_intermediates,
            dry_run=args.dry_run
        )
        
        # Execute pipeline
        pipeline = Pipeline(config)
        success = pipeline.execute()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        log.error("\nPipeline interrupted by user")
        return 130
    except Exception as e:
        log.error(f"Pipeline failed with exception: {e}")
        log.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
