#!/usr/bin/env python3
"""
Context-Aware Pro Pipeline
Transformation Portal - Architectural Intelligence Integration

Integrates architectural context (floor plans, elevations, dimensions) into
the professional rendering pipeline for superior quality and accuracy.

Features:
- Architectural document parsing and context extraction
- Context-enhanced AI prompting
- Space-aware material response
- Dimension-informed depth processing
- Design intent preservation
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Import architectural context engine
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.utilities.architectural_context_engine import ArchitecturalContext, ContextAwareRenderingPipeline, SpaceType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ContextAwareProPipeline:
    """Professional pipeline with architectural context awareness."""

    def __init__(
        self,
        output_dir: Path = Path("/tmp/tp-context-aware-pro"),
        context_dir: Path = Path("/tmp/tp-context-aware-pro-context"),
    ):
        """
        Initialize context-aware pipeline.

        Args:
            output_dir: Output directory for processed images
            context_dir: Directory for cached architectural contexts
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        self.context_pipeline = ContextAwareRenderingPipeline(context_dir)

        logger.info("Context-Aware Pro Pipeline initialized")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Context cache: {context_dir}")

    def process_image(
        self,
        image_path: Path,
        pdf_documents: Optional[List[Path]] = None,
        base_prompt: str = "photorealistic luxury architectural rendering",
        enable_depth: bool = True,
        enable_material_response: bool = True,
        enable_ai_enhancement: bool = True,
        upscale_4x: bool = False,
    ) -> Dict[str, Path]:
        """
        Process image with architectural context awareness.

        Args:
            image_path: Input image path
            pdf_documents: Optional list of architectural PDFs
            base_prompt: Base AI prompt (will be enhanced with context)
            enable_depth: Enable depth-aware processing
            enable_material_response: Enable material response
            enable_ai_enhancement: Enable AI enhancement
            upscale_4x: Enable 4x upscaling

        Returns:
            Dictionary of output paths by processing stage
        """
        logger.info("=" * 80)
        logger.info(f"PROCESSING: {image_path.name}")
        logger.info("=" * 80)

        # 1. Extract/load architectural context
        logger.info("\n[1/5] Extracting architectural context...")
        context = self.context_pipeline.prepare_context(image_path, pdf_documents)

        logger.info(f"  Project: {context.project_name}")
        logger.info(f"  Space: {context.space_type.value if context.space_type else 'Unknown'}")
        logger.info(f"  Materials: {len(context.materials)} specifications")
        logger.info(f"  Design intent: {', '.join(context.design_intent[:3])}")

        # 2. Generate context-enhanced prompt
        enhanced_prompt = context.to_enhanced_prompt(base_prompt)
        logger.info("\n[2/5] Enhanced AI prompt generated")
        logger.info(f"  Base: {base_prompt}")
        logger.info(f"  Enhanced: {enhanced_prompt}")

        outputs = {}

        # 3. Depth-aware processing (if enabled)
        if enable_depth:
            logger.info("\n[3/5] Depth-aware processing...")
            depth_output = self._process_depth_stage(image_path, context, enhanced_prompt)
            outputs["depth"] = depth_output
        else:
            logger.info("\n[3/5] Depth processing: SKIPPED")
            outputs["depth"] = image_path

        # 4. Material response (if enabled)
        if enable_material_response:
            logger.info("\n[4/5] Material response processing...")
            material_output = self._process_material_stage(outputs["depth"], context, enhanced_prompt)
            outputs["material"] = material_output
        else:
            logger.info("\n[4/5] Material response: SKIPPED")
            outputs["material"] = outputs["depth"]

        # 5. AI enhancement (if enabled)
        if enable_ai_enhancement:
            logger.info("\n[5/5] AI enhancement...")
            ai_output = self._process_ai_stage(outputs["material"], context, enhanced_prompt, upscale_4x)
            outputs["ai_enhanced"] = ai_output
        else:
            logger.info("\n[5/5] AI enhancement: SKIPPED")
            outputs["ai_enhanced"] = outputs["material"]

        # Save final context summary
        summary_path = self.output_dir / f"{image_path.stem}_context_summary.txt"
        self._save_processing_summary(context, outputs, summary_path)

        logger.info("\n" + "=" * 80)
        logger.info("✅ Processing complete!")
        logger.info(f"   Final output: {outputs['ai_enhanced']}")
        logger.info("=" * 80)

        return outputs

    def _process_depth_stage(self, image_path: Path, context: ArchitecturalContext, prompt: str) -> Path:
        """Process with depth-aware pipeline."""
        try:
            from depth_pipeline.pipeline import ArchitecturalDepthPipeline

            # Select config based on space type
            config_map = {
                SpaceType.KITCHEN: "config/interior_preset.yaml",
                SpaceType.LIVING: "config/interior_preset.yaml",
                SpaceType.BEDROOM: "config/interior_preset.yaml",
                SpaceType.EXTERIOR: "config/exterior_preset.yaml",
                SpaceType.POOL_AREA: "config/exterior_preset.yaml",
            }

            config_path = config_map.get(context.space_type, "config/interior_preset.yaml")

            logger.info(f"  Using depth config: {config_path}")

            # Load pipeline
            pipeline = ArchitecturalDepthPipeline.from_config(config_path)

            # Process
            result = pipeline.process_render(str(image_path))

            # Save
            output_path = self.output_dir / f"{image_path.stem}_depth.png"
            pipeline.save_result(result, output_path)

            logger.info(f"  ✓ Depth output: {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"  ⚠ Depth processing failed: {e}")
            logger.info("  Continuing with original image...")
            return image_path

    def _process_material_stage(self, image_path: Path, context: ArchitecturalContext, prompt: str) -> Path:
        """Process with material response."""
        try:
            from material_response import MaterialResponse, SurfaceType

            # Map architectural materials to surface types
            surface_map = {
                "wood": SurfaceType.WOOD,
                "metal": SurfaceType.METAL,
                "glass": SurfaceType.GLASS,
                "stone": SurfaceType.STONE,
                "fabric": SurfaceType.FABRIC,
            }

            surfaces = []
            for mat_spec in context.materials:
                if mat_spec.material_type in surface_map:
                    surfaces.append(surface_map[mat_spec.material_type])

            # Default surfaces for common space types
            if not surfaces:
                space_defaults = {
                    SpaceType.KITCHEN: [SurfaceType.STONE, SurfaceType.METAL, SurfaceType.GLASS],
                    SpaceType.LIVING: [SurfaceType.WOOD, SurfaceType.FABRIC, SurfaceType.GLASS],
                    SpaceType.POOL_AREA: [SurfaceType.STONE, SurfaceType.WATER],
                }
                surfaces = space_defaults.get(context.space_type, [SurfaceType.WOOD, SurfaceType.STONE])

            logger.info(f"  Target surfaces: {', '.join(s.value for s in surfaces)}")

            # Initialize material response
            mr = MaterialResponse()

            # Load and process image
            import numpy as np
            from PIL import Image

            image = Image.open(image_path)
            image_array = np.array(image)

            # Process with material response
            enhanced = mr.enhance(
                image_array, surfaces=surfaces, strength=0.75, preserve_highlights=True  # Strong material response
            )

            # Save
            output_path = self.output_dir / f"{image_path.stem}_material.png"
            Image.fromarray(enhanced).save(output_path)

            logger.info(f"  ✓ Material output: {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"  ⚠ Material processing failed: {e}")
            logger.info("  Continuing with depth-processed image...")
            return image_path

    def _process_ai_stage(
        self, image_path: Path, context: ArchitecturalContext, prompt: str, upscale_4x: bool = False
    ) -> Path:
        """Process with AI enhancement."""
        try:
            # Import AI pipeline components
            logger.info(f"  Prompt: {prompt}")

            # For now, use existing lux_render_pipeline
            # In future, integrate context directly into Stable Diffusion pipeline

            from lux_render_pipeline import process_image

            output_path = self.output_dir / f"{image_path.stem}_ai_enhanced.png"

            # Process with enhanced prompt
            result = process_image(image_path, prompt=prompt, output_path=output_path, upscale=upscale_4x)

            logger.info(f"  ✓ AI output: {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"  ⚠ AI enhancement failed: {e}")
            logger.info("  Continuing with material-processed image...")
            return image_path

    def _save_processing_summary(self, context: ArchitecturalContext, outputs: Dict[str, Path], summary_path: Path):
        """Save processing summary with context details."""

        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("CONTEXT-AWARE PRO PIPELINE - PROCESSING SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write("ARCHITECTURAL CONTEXT:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Project: {context.project_name}\n")
            if context.project_address:
                f.write(f"Address: {context.project_address}\n")
            if context.space_type:
                f.write(f"Space Type: {context.space_type.value}\n")
            if context.space_name:
                f.write(f"Space Name: {context.space_name}\n")

            f.write("\n")

            if context.dimensions:
                f.write("DIMENSIONS:\n")
                f.write(f"  {context.dimensions.to_prompt_fragment()}\n")
                f.write("\n")

            if context.materials:
                f.write("MATERIALS:\n")
                for mat in context.materials[:10]:
                    f.write(f"  - {mat.to_prompt_fragment()} ({mat.location})\n")
                f.write("\n")

            if context.design_intent:
                f.write("DESIGN INTENT:\n")
                for intent in context.design_intent:
                    f.write(f"  - {intent}\n")
                f.write("\n")

            if context.style_notes:
                f.write("STYLE NOTES:\n")
                for note in context.style_notes:
                    f.write(f"  - {note}\n")
                f.write("\n")

            f.write("PROCESSING OUTPUTS:\n")
            f.write("-" * 80 + "\n")
            for stage, path in outputs.items():
                f.write(f"{stage:20s}: {path}\n")

            f.write("\n")
            f.write("SOURCE DOCUMENTS:\n")
            for doc in context.source_documents:
                f.write(f"  - {doc}\n")

        logger.info(f"  Summary saved: {summary_path}")


def main():
    """CLI for context-aware pro pipeline."""

    parser = argparse.ArgumentParser(description="Context-Aware Pro Pipeline - Architectural Intelligence")

    parser.add_argument("image", type=Path, help="Input image path")

    parser.add_argument(
        "--pdf",
        "--pd",
        dest="pdf",
        type=Path,
        action="append",
        help="Architectural PDF document(s) for context extraction (can specify multiple)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/tp-context-aware-pro"),
        help="Output directory (default: /tmp/tp-context-aware-pro)",
    )

    parser.add_argument(
        "--prompt",
        default="photorealistic luxury architectural rendering",
        help="Base AI prompt (will be enhanced with context)",
    )

    parser.add_argument("--no-depth", action="store_true", help="Disable depth-aware processing")

    parser.add_argument("--no-material", action="store_true", help="Disable material response")

    parser.add_argument("--no-ai", action="store_true", help="Disable AI enhancement")

    parser.add_argument("--upscale-4x", action="store_true", help="Enable 4x upscaling (slower, higher quality)")

    args = parser.parse_args()

    # Validate input
    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Check PDFs
    pdf_docs = []
    if args.pdf:
        for pdf_path in args.pdf:
            if pdf_path.exists():
                pdf_docs.append(pdf_path)
            else:
                print(f"Warning: PDF not found: {pdf_path}")

    # Initialize pipeline
    pipeline = ContextAwareProPipeline(output_dir=args.output_dir)

    # Process
    outputs = pipeline.process_image(
        image_path=args.image,
        pdf_documents=pdf_docs if pdf_docs else None,
        base_prompt=args.prompt,
        enable_depth=not args.no_depth,
        enable_material_response=not args.no_material,
        enable_ai_enhancement=not args.no_ai,
        upscale_4x=args.upscale_4x,
    )

    print("\n✅ Processing complete!")
    print(f"Final output: {outputs['ai_enhanced']}")


if __name__ == "__main__":
    main()
