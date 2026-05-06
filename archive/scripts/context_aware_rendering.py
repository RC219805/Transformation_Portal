#!/usr/bin/env python3
"""
Context-Aware Rendering Pipeline
Transformation Portal - Intelligent Architectural Rendering

Integrates architectural context (plans, elevations, specs) into rendering pipeline:
- Room-specific material enhancement
- Dimension-aware composition
- Style-consistent color grading
- Context-driven depth processing
- Intelligent quality metrics

Uses extracted architectural intelligence to inform every processing decision.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from architectural_context_extractor import ArchitecturalContextExtractor, ProjectContext, RoomContext
from PIL import Image


@dataclass
class RenderingStrategy:
    """Rendering strategy derived from architectural context."""

    room_type: str
    primary_materials: List[str]
    lighting_style: str  # natural, dramatic, soft, ambient
    depth_emphasis: str  # foreground, balanced, atmospheric
    color_temperature: str  # warm, neutral, cool
    enhancement_strength: float  # 0.0 - 1.0
    lut_preset: Optional[str] = None
    material_response_config: Optional[Dict] = None


class ContextAwareRenderingPipeline:
    """
    Intelligent rendering pipeline guided by architectural context.
    """

    # Room-specific rendering strategies
    ROOM_STRATEGIES = {
        "kitchen": RenderingStrategy(
            room_type="kitchen",
            primary_materials=["metal", "stone", "wood", "glass"],
            lighting_style="bright",
            depth_emphasis="balanced",
            color_temperature="neutral",
            enhancement_strength=0.75,
            lut_preset="signature_estate",
        ),
        "bathroom": RenderingStrategy(
            room_type="bathroom",
            primary_materials=["stone", "glass", "metal", "tile"],
            lighting_style="soft",
            depth_emphasis="balanced",
            color_temperature="neutral",
            enhancement_strength=0.7,
            lut_preset="serene_spa",
        ),
        "bedroom": RenderingStrategy(
            room_type="bedroom",
            primary_materials=["wood", "fabric", "leather"],
            lighting_style="soft",
            depth_emphasis="atmospheric",
            color_temperature="warm",
            enhancement_strength=0.6,
            lut_preset="warm_invitation",
        ),
        "living": RenderingStrategy(
            room_type="living",
            primary_materials=["wood", "fabric", "stone", "leather"],
            lighting_style="ambient",
            depth_emphasis="balanced",
            color_temperature="warm",
            enhancement_strength=0.7,
            lut_preset="golden_hour_interior",
        ),
        "outdoor": RenderingStrategy(
            room_type="outdoor",
            primary_materials=["stone", "concrete", "wood", "metal"],
            lighting_style="natural",
            depth_emphasis="atmospheric",
            color_temperature="neutral",
            enhancement_strength=0.8,
            lut_preset="golden_hour_courtyard",
        ),
    }

    def __init__(self, project_context: ProjectContext, output_dir: Path = None):
        """
        Initialize context-aware pipeline.

        Args:
            project_context: Extracted architectural context
            output_dir: Output directory for processed renders
        """
        self.context = project_context
        self.output_dir = output_dir or Path("output_context_aware")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        print(f"\n{'='*70}")
        print("CONTEXT-AWARE RENDERING PIPELINE")
        print(f"{'='*70}")
        print(f"Project: {self.context.project_name}")
        if self.context.design_style:
            print(f"Style: {self.context.design_style}")
        print(f"Rooms: {len(self.context.rooms)}")
        print(f"Materials: {', '.join(self.context.materials_palette[:5])}")

    def identify_room_from_filename(self, image_path: Path) -> Optional[str]:
        """
        Identify room type from image filename.

        Args:
            image_path: Path to rendering

        Returns:
            Room type key (e.g., 'kitchen', 'bedroom') or None
        """
        filename_lower = image_path.stem.lower()

        # Direct matches
        for room_key in self.ROOM_STRATEGIES.keys():
            if room_key in filename_lower:
                return room_key

        # Alias matching
        aliases = {
            "kitchen": ["kitch", "cook"],
            "bathroom": ["bath", "powder"],
            "bedroom": ["bed", "master", "primary"],
            "living": ["great room", "living", "family"],
            "outdoor": ["pool", "patio", "deck", "terrace", "courtyard", "exterior"],
        }

        for room_key, alias_list in aliases.items():
            if any(alias in filename_lower for alias in alias_list):
                return room_key

        return None

    def get_room_context(self, room_type: str) -> Optional[RoomContext]:
        """Get room context from project context."""
        # Find matching room in project context
        for room_key, room in self.context.rooms.items():
            if room_key.startswith(room_type):
                return room
        return None

    def derive_strategy(self, image_path: Path) -> RenderingStrategy:
        """
        Derive optimal rendering strategy from context.

        Args:
            image_path: Path to rendering

        Returns:
            RenderingStrategy optimized for this specific render
        """
        # Identify room type
        room_type = self.identify_room_from_filename(image_path)

        if not room_type:
            # Default strategy
            print(f"⚠ Could not identify room type from: {image_path.name}")
            print("  Using balanced default strategy")
            return RenderingStrategy(
                room_type="unknown",
                primary_materials=self.context.materials_palette[:4] if self.context.materials_palette else ["wood", "stone"],
                lighting_style="ambient",
                depth_emphasis="balanced",
                color_temperature="neutral",
                enhancement_strength=0.7,
            )

        # Get base strategy for room type
        base_strategy = self.ROOM_STRATEGIES.get(room_type)
        if not base_strategy:
            return self.derive_strategy(image_path)  # Fallback to default

        # Customize based on project context
        room_context = self.get_room_context(room_type)

        # Adjust materials based on project palette
        if self.context.materials_palette:
            # Prioritize materials that appear in both strategy and project
            matched_materials = [mat for mat in base_strategy.primary_materials if mat in self.context.materials_palette]
            if matched_materials:
                base_strategy.primary_materials = matched_materials

        # Adjust based on design style
        if self.context.design_style:
            style_lower = self.context.design_style.lower()
            if "modern" in style_lower or "contemporary" in style_lower:
                base_strategy.color_temperature = "neutral"
                base_strategy.enhancement_strength = min(base_strategy.enhancement_strength + 0.1, 1.0)
            elif "traditional" in style_lower:
                base_strategy.color_temperature = "warm"
                base_strategy.enhancement_strength = max(base_strategy.enhancement_strength - 0.1, 0.5)

        print(f"\n✓ Derived strategy for {room_type}:")
        print(f"  Materials: {', '.join(base_strategy.primary_materials)}")
        print(f"  Lighting: {base_strategy.lighting_style}")
        print(f"  Depth: {base_strategy.depth_emphasis}")
        print(f"  Temperature: {base_strategy.color_temperature}")
        print(f"  Enhancement: {base_strategy.enhancement_strength:.2f}")

        return base_strategy

    def generate_depth_config(self, strategy: RenderingStrategy) -> Dict:
        """Generate depth pipeline configuration from strategy."""
        base_config = {
            "model_size": "small",
            "device": "mps",  # Apple Silicon
        }

        # Depth emphasis
        if strategy.depth_emphasis == "foreground":
            base_config["zone_weights"] = {
                "foreground": 1.0,
                "midground": 0.6,
                "background": 0.3,
            }
        elif strategy.depth_emphasis == "atmospheric":
            base_config["zone_weights"] = {
                "foreground": 0.6,
                "midground": 0.8,
                "background": 1.0,
            }
        else:  # balanced
            base_config["zone_weights"] = {
                "foreground": 0.8,
                "midground": 1.0,
                "background": 0.8,
            }

        # Tone mapping based on lighting style
        tone_map_operators = {
            "bright": "reinhard",
            "soft": "hable",
            "dramatic": "filmic",
            "natural": "agx",
            "ambient": "agx",
        }
        base_config["tone_map"] = tone_map_operators.get(strategy.lighting_style, "agx")

        return base_config

    def generate_material_config(self, strategy: RenderingStrategy) -> Dict:
        """Generate material response configuration from strategy."""
        config = {
            "enabled_surfaces": strategy.primary_materials,
            "global_strength": strategy.enhancement_strength,
            "preserve_highlights": True,
            "micro_contrast": 0.15,
        }

        # Per-material strengths
        material_strengths = {}
        for i, material in enumerate(strategy.primary_materials):
            # Primary materials get higher strength
            strength = strategy.enhancement_strength * (1.0 - i * 0.1)
            material_strengths[material] = max(strength, 0.5)

        config["material_strengths"] = material_strengths

        return config

    def generate_color_config(self, strategy: RenderingStrategy) -> Dict:
        """Generate color grading configuration from strategy."""
        config = {
            "lut_preset": strategy.lut_preset,
            "lut_strength": 0.7,
        }

        # Temperature adjustments
        temp_adjustments = {
            "warm": {"saturation": 1.08, "tint": 5},
            "neutral": {"saturation": 1.05, "tint": 0},
            "cool": {"saturation": 1.03, "tint": -5},
        }

        adjustments = temp_adjustments.get(strategy.color_temperature, temp_adjustments["neutral"])
        config.update(adjustments)

        return config

    def process_render(
        self,
        image_path: Path,
        apply_depth: bool = True,
        apply_material: bool = True,
        apply_color: bool = True,
    ) -> Path:
        """
        Process render with context-aware intelligence.

        Args:
            image_path: Path to rendering
            apply_depth: Apply depth-aware processing
            apply_material: Apply material response
            apply_color: Apply color grading

        Returns:
            Path to processed output
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING: {image_path.name}")
        print(f"{'='*70}")

        # Derive strategy
        strategy = self.derive_strategy(image_path)

        # Generate configurations
        depth_config = self.generate_depth_config(strategy) if apply_depth else None
        material_config = self.generate_material_config(strategy) if apply_material else None
        color_config = self.generate_color_config(strategy) if apply_color else None

        # Save strategy and configs
        strategy_path = self.output_dir / f"{image_path.stem}_strategy.json"
        with open(strategy_path, "w") as f:
            json.dump(
                {
                    "strategy": {
                        "room_type": strategy.room_type,
                        "primary_materials": strategy.primary_materials,
                        "lighting_style": strategy.lighting_style,
                        "depth_emphasis": strategy.depth_emphasis,
                        "color_temperature": strategy.color_temperature,
                        "enhancement_strength": strategy.enhancement_strength,
                        "lut_preset": strategy.lut_preset,
                    },
                    "depth_config": depth_config,
                    "material_config": material_config,
                    "color_config": color_config,
                },
                f,
                indent=2,
            )

        print(f"\n✓ Strategy saved: {strategy_path}")

        # Archived: pipeline integration was superseded by Spatial AI
        # orchestration (Phase 2, ADR-027). See archive/scripts/README.md.
        # This script emits a strategy/config bundle only; downstream
        # processing now lives in src/transformation_portal/spatial_ai/.

        print("\n💡 Next steps:")
        print(f"  1. Apply depth pipeline with: {strategy_path}")
        print("  2. Apply material response")
        print(f"  3. Apply color grading with LUT: {strategy.lut_preset}")

        # For now, return strategy path
        return strategy_path


def main():
    """CLI for context-aware rendering."""
    import argparse

    parser = argparse.ArgumentParser(description="Context-aware architectural rendering")
    parser.add_argument("image", type=Path, help="Rendering to process")
    parser.add_argument("--context", "-c", type=Path, required=True, help="Path to extracted context JSON or PDF")
    parser.add_argument("--output", "-o", type=Path, default=Path("output_context_aware"), help="Output directory")
    parser.add_argument("--no-depth", action="store_true", help="Skip depth processing")
    parser.add_argument("--no-material", action="store_true", help="Skip material response")
    parser.add_argument("--no-color", action="store_true", help="Skip color grading")

    args = parser.parse_args()

    if not args.image.exists():
        print(f"✗ Image not found: {args.image}")
        return 1

    # Load or extract context
    if args.context.suffix == ".pdf":
        print(f"Extracting context from PDF: {args.context}")
        extractor = ArchitecturalContextExtractor()
        context = extractor.extract_from_pdf(args.context)
    elif args.context.suffix == ".json":
        print(f"Loading context: {args.context}")
        with open(args.context, "r") as f:
            context_data = json.load(f)
        # Reconstruct context (simplified)
        from architectural_context_extractor import RoomContext

        rooms = {}
        for room_key, room_data in context_data.get("rooms", {}).items():
            rooms[room_key] = RoomContext(**room_data)
        context = ProjectContext(
            project_name=context_data["project_name"],
            project_number=context_data.get("project_number"),
            address=context_data.get("address"),
            rooms=rooms,
            materials_palette=context_data.get("materials_palette", []),
            design_style=context_data.get("design_style"),
        )
    else:
        print(f"✗ Context must be PDF or JSON: {args.context}")
        return 1

    # Initialize pipeline
    pipeline = ContextAwareRenderingPipeline(project_context=context, output_dir=args.output)

    # Process render
    output = pipeline.process_render(
        args.image,
        apply_depth=not args.no_depth,
        apply_material=not args.no_material,
        apply_color=not args.no_color,
    )

    print("\n✓ Processing complete")
    print(f"  Strategy: {output}")

    return 0


if __name__ == "__main__":
    exit(main())
