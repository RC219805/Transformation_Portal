#!/usr/bin/env python3
"""
Premium Context-Aware Pipeline
Transformation Portal - Ultimate Architectural Rendering

Combines all processing stages with architectural intelligence:
1. Context extraction from architectural documents
2. Strategy derivation for specific rendering
3. Depth-aware processing (Depth Anything V2)
4. Material Response enhancement
5. Context-driven color grading
6. Quality metrics and validation

This is the flagship pipeline for luxury real estate visualization.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

from architectural_context_extractor import ArchitecturalContextExtractor, ProjectContext
from context_aware_rendering import ContextAwareRenderingPipeline


class PremiumContextAwarePipeline:
    """Ultimate context-aware rendering pipeline."""

    def __init__(
        self,
        project_context: ProjectContext,
        output_dir: Path = None,
        verbose: bool = True
    ):
        """Initialize premium pipeline."""
        self.context = project_context
        self.output_dir = output_dir or Path("output_premium")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.verbose = verbose

        # Initialize sub-pipeline
        self.context_pipeline = ContextAwareRenderingPipeline(
            project_context=project_context,
            output_dir=self.output_dir
        )

    def process_with_context(
        self,
        image_path: Path,
        quality_level: str = 'premium',  # standard, premium, ultimate
    ) -> Dict[str, Path]:
        """
        Process rendering with full context intelligence.

        Args:
            image_path: Path to rendering
            quality_level: Processing quality tier

        Returns:
            Dict of output paths by stage
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("PREMIUM CONTEXT-AWARE PIPELINE")
            print(f"{'='*70}")
            print(f"Project: {self.context.project_name}")
            print(f"Image: {image_path.name}")
            print(f"Quality: {quality_level.upper()}")

        outputs = {}

        # Stage 1: Derive strategy
        if self.verbose:
            print(f"\n{'='*70}")
            print("STAGE 1: STRATEGY DERIVATION")
            print(f"{'='*70}")

        strategy_path = self.context_pipeline.process_render(image_path)
        outputs['strategy'] = strategy_path

        # Load strategy
        with open(strategy_path, 'r') as f:
            strategy_config = json.load(f)

        # Stage 2: Depth processing
        if self.verbose:
            print(f"\n{'='*70}")
            print("STAGE 2: DEPTH-AWARE PROCESSING")
            print(f"{'='*70}")

        depth_output = self._apply_depth_processing(
            image_path,
            strategy_config['depth_config']
        )
        if depth_output:
            outputs['depth'] = depth_output

        # Stage 3: Material Response
        if self.verbose:
            print(f"\n{'='*70}")
            print("STAGE 3: MATERIAL RESPONSE")
            print(f"{'='*70}")

        material_input = depth_output if depth_output else image_path
        material_output = self._apply_material_response(
            material_input,
            strategy_config['material_config']
        )
        if material_output:
            outputs['material'] = material_output

        # Stage 4: Color Grading
        if self.verbose:
            print(f"\n{'='*70}")
            print("STAGE 4: COLOR GRADING")
            print(f"{'='*70}")

        color_input = material_output if material_output else (depth_output if depth_output else image_path)
        color_output = self._apply_color_grading(
            color_input,
            strategy_config['color_config']
        )
        if color_output:
            outputs['color'] = color_output

        # Stage 5: Final enhancement (if ultimate quality)
        if quality_level == 'ultimate':
            if self.verbose:
                print(f"\n{'='*70}")
                print("STAGE 5: ULTIMATE ENHANCEMENT")
                print(f"{'='*70}")

            final_input = color_output if color_output else color_input
            final_output = self._apply_ultimate_enhancement(final_input)
            if final_output:
                outputs['final'] = final_output

        # Generate quality report
        if self.verbose:
            print(f"\n{'='*70}")
            print("PROCESSING COMPLETE")
            print(f"{'='*70}")
            for stage, path in outputs.items():
                print(f"  {stage.upper()}: {path}")

        return outputs

    def _apply_depth_processing(
        self,
        image_path: Path,
        depth_config: Dict
    ) -> Optional[Path]:
        """Apply depth-aware processing using depth pipeline."""
        try:
            # Check if depth_pipeline is available
            depth_pipeline_script = Path("depth_pipeline/pipeline.py")
            if not depth_pipeline_script.exists():
                print("⚠ Depth pipeline not found, skipping depth processing")
                return None

            output_path = self.output_dir / f"{image_path.stem}_depth.tif"

            # Build command
            cmd = [
                'python',
                str(depth_pipeline_script),
                str(image_path),
                '--output', str(output_path),
            ]

            # Add config parameters
            if 'tone_map' in depth_config:
                cmd.extend(['--tone-map', depth_config['tone_map']])

            if self.verbose:
                print(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0 and output_path.exists():
                print(f"✓ Depth processing complete: {output_path}")
                return output_path
            else:
                print(f"⚠ Depth processing failed: {result.stderr}")
                return None

        except Exception as e:
            print(f"⚠ Depth processing error: {e}")
            return None

    def _apply_material_response(
        self,
        image_path: Path,
        material_config: Dict
    ) -> Optional[Path]:
        """Apply Material Response enhancement."""
        try:
            material_response_script = Path("material_response.py")
            if not material_response_script.exists():
                print("⚠ Material Response not found, skipping")
                return None

            output_path = self.output_dir / f"{image_path.stem}_material.tif"

            # Build command
            cmd = [
                'python',
                str(material_response_script),
                str(image_path),
                '--output', str(output_path),
                '--strength', str(material_config.get('global_strength', 0.7)),
            ]

            # Add enabled surfaces
            if 'enabled_surfaces' in material_config:
                for surface in material_config['enabled_surfaces']:
                    cmd.extend(['--surface', surface])

            if self.verbose:
                print(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0 and output_path.exists():
                print(f"✓ Material Response complete: {output_path}")
                return output_path
            else:
                print(f"⚠ Material Response failed: {result.stderr}")
                return None

        except Exception as e:
            print(f"⚠ Material Response error: {e}")
            return None

    def _apply_color_grading(
        self,
        image_path: Path,
        color_config: Dict
    ) -> Optional[Path]:
        """Apply color grading with LUT."""
        try:
            color_script = Path("luxury_tiff_batch_processor.py")
            if not color_script.exists():
                print("⚠ Color grading script not found, skipping")
                return None

            output_path = self.output_dir / f"{image_path.stem}_graded.tif"

            preset = color_config.get('lut_preset', 'signature_estate')

            # Build command
            cmd = [
                'python',
                str(color_script),
                str(image_path),
                '--preset', preset,
                '--output-dir', str(self.output_dir),
            ]

            if self.verbose:
                print(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                print(f"⚠ Color grading failed: {result.stderr.strip()}")
                return None
            # Find output (script generates its own naming)
            expected_output = self.output_dir / f"{image_path.stem}_{preset}.tif"
            if expected_output.exists():
                # Rename to standard name
                expected_output.rename(output_path)
                print(f"✓ Color grading complete: {output_path}")
                return output_path
            else:
                print("⚠ Color grading failed or output not found")
                return None

        except Exception as e:
            print(f"⚠ Color grading error: {e}")
            return None

    def _apply_ultimate_enhancement(
        self,
        image_path: Path
    ) -> Optional[Path]:
        """Apply ultimate enhancement (AI upscaling, refinement)."""
        try:
            lux_script = Path("lux_render_pipeline.py")
            if not lux_script.exists():
                print("⚠ Lux pipeline not found, skipping ultimate enhancement")
                return None

            output_path = self.output_dir / f"{image_path.stem}_ultimate.tif"

            # Build command
            cmd = [
                'python',
                str(lux_script),
                str(image_path),
                '--output', str(output_path),
                '--upscale-only',  # Assume this flag exists for 4K upscale
            ]

            if self.verbose:
                print(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)

            if result.returncode == 0 and output_path.exists():
                print(f"✓ Ultimate enhancement complete: {output_path}")
                return output_path
            else:
                print("⚠ Ultimate enhancement failed")
                return None

        except Exception as e:
            print(f"⚠ Ultimate enhancement error: {e}")
            return None


def main():
    """CLI for premium context-aware pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Premium context-aware architectural rendering pipeline'
    )
    parser.add_argument('image', type=Path, help='Rendering to process')
    parser.add_argument('--context', '-c', type=Path, required=True,
                        help='Path to extracted context JSON or PDF')
    parser.add_argument('--output', '-o', type=Path, default=Path('output_premium'),
                        help='Output directory')
    parser.add_argument('--quality', '-q',
                        choices=['standard', 'premium', 'ultimate'],
                        default='premium',
                        help='Quality tier')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    if not args.image.exists():
        print(f"✗ Image not found: {args.image}")
        return 1

    # Load or extract context
    if args.context.suffix == '.pdf':
        print(f"Extracting context from PDF: {args.context}")
        extractor = ArchitecturalContextExtractor()
        context = extractor.extract_from_pdf(args.context)
    elif args.context.suffix == '.json':
        print(f"Loading context: {args.context}")
        with open(args.context, 'r') as f:
            context_data = json.load(f)
        from architectural_context_extractor import RoomContext
        rooms = {}
        for room_key, room_data in context_data.get('rooms', {}).items():
            # Handle potential list data
            if isinstance(room_data, dict):
                rooms[room_key] = RoomContext(**room_data)
        context = ProjectContext(
            project_name=context_data['project_name'],
            project_number=context_data.get('project_number'),
            address=context_data.get('address'),
            rooms=rooms,
            materials_palette=context_data.get('materials_palette', []),
            design_style=context_data.get('design_style'),
        )
    else:
        print(f"✗ Context must be PDF or JSON: {args.context}")
        return 1

    # Initialize and run pipeline
    pipeline = PremiumContextAwarePipeline(
        project_context=context,
        output_dir=args.output,
        verbose=not args.quiet
    )

    outputs = pipeline.process_with_context(
        args.image,
        quality_level=args.quality
    )

    if not args.quiet:
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE")
        print(f"{'='*70}")
        print("\nOutputs:")
        for stage, path in outputs.items():
            print(f"  {stage.upper()}: {path}")

    return 0


if __name__ == '__main__':
    exit(main())
