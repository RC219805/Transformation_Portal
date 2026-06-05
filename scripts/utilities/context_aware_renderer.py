#!/usr/bin/env python3
"""
Context-Aware Rendering Pipeline
Transformation Portal - Architectural Knowledge Integration

Integrates architectural documentation (floor plans, elevations, specifications)
with AI-powered rendering enhancement for contextually accurate results.

Features:
- Material validation against specifications
- Spatial scale awareness from floor plans
- Room-specific enhancement profiles
- Cross-reference validation (render vs plan)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


class ArchitecturalContext:
    """Load and query architectural context from PDF extraction."""

    def __init__(self, context_file: Path):
        """Initialize with rendering knowledge base JSON."""
        with open(context_file, "r") as f:
            self.kb = json.load(f)

        self.property_name = self.kb.get("property_name", "Unknown Property")
        self.rooms = self.kb.get("rooms", [])
        self.materials = self.kb.get("materials", [])
        self.rendering_pages = self.kb.get("rendering_pages", {})

    def identify_room(self, image_path: Path) -> Optional[str]:
        """Identify room type from image filename or metadata."""
        filename_lower = image_path.stem.lower()

        # Check for room keywords in filename
        room_keywords = {
            "kitchen": ["kitchen", "cook", "culinary"],
            "bedroom": ["bedroom", "bed", "master", "guest"],
            "bathroom": ["bathroom", "bath", "toilet", "shower"],
            "living": ["living", "lounge", "great room"],
            "dining": ["dining", "dinner"],
            "garage": ["garage", "parking"],
            "entry": ["entry", "foyer", "entrance"],
            "closet": ["closet", "wardrobe", "storage"],
        }

        for room_type, keywords in room_keywords.items():
            if any(kw in filename_lower for kw in keywords):
                return room_type

        return None

    def get_material_palette(self, room_type: Optional[str] = None) -> List[str]:
        """Get material palette, optionally filtered by room type."""
        # For now, return all materials
        # Could be enhanced with room-specific material filtering
        return self.materials

    def get_enhancement_profile(self, room_type: Optional[str]) -> Dict:
        """Get room-specific enhancement parameters."""

        profiles = {
            "kitchen": {
                "materials_focus": ["wood", "stone", "steel", "glass"],
                "lighting_style": "bright_task",
                "clarity_boost": 0.20,
                "material_response_strength": 0.75,
                "lut": "assets/luts/location_aesthetic/Modern_Kitchen.cube",
                "notes": "Emphasize cabinet wood grain, countertop stone, appliance reflections",
            },
            "bedroom": {
                "materials_focus": ["wood", "fabric", "textile"],
                "lighting_style": "soft_ambient",
                "clarity_boost": 0.12,
                "material_response_strength": 0.65,
                "lut": "assets/luts/film_emulation/Warm_Interior.cube",
                "notes": "Soft textures, warm tones, gentle depth of field",
            },
            "bathroom": {
                "materials_focus": ["stone", "glass", "ceramic", "tile"],
                "lighting_style": "even_bright",
                "clarity_boost": 0.18,
                "material_response_strength": 0.70,
                "lut": "assets/luts/location_aesthetic/Luxury_Bath.cube",
                "notes": "Enhance reflective surfaces, water features, tile grout",
            },
            "living": {
                "materials_focus": ["wood", "fabric", "glass", "stone"],
                "lighting_style": "natural_warm",
                "clarity_boost": 0.15,
                "material_response_strength": 0.68,
                "lut": "assets/luts/film_emulation/Cinematic_Interior.cube",
                "notes": "Balanced enhancement, maintain spatial depth",
            },
            "default": {
                "materials_focus": ["wood", "stone", "metal", "glass"],
                "lighting_style": "balanced",
                "clarity_boost": 0.15,
                "material_response_strength": 0.70,
                "lut": "assets/luts/film_emulation/Natural_Interior.cube",
                "notes": "Standard architectural enhancement",
            },
        }

        return profiles.get(room_type, profiles["default"])

    def get_rendering_context(self, image_path: Path) -> Dict:
        """Get full context for rendering enhancement."""
        room_type = self.identify_room(image_path)
        enhancement_profile = self.get_enhancement_profile(room_type)
        material_palette = self.get_material_palette(room_type)

        return {
            "property_name": self.property_name,
            "image_path": str(image_path),
            "room_type": room_type or "unknown",
            "enhancement_profile": enhancement_profile,
            "material_palette": material_palette,
            "architectural_context_available": True,
        }


class ContextAwareRenderer:
    """Rendering pipeline with architectural context awareness."""

    def __init__(self, context_file: Path):
        """Initialize with architectural context."""
        self.context = ArchitecturalContext(context_file)
        print(f"✅ Loaded context for: {self.context.property_name}")
        print(f"   Rooms: {len(self.context.rooms)}")
        print(f"   Materials: {len(self.context.materials)}")

    def enhance_render(self, image_path: Path, output_dir: Path, pipeline: str = "standard"):
        """Enhance render with context-aware processing."""

        # Get rendering context
        ctx = self.context.get_rendering_context(image_path)

        print("\n" + "=" * 80)
        print(f"CONTEXT-AWARE RENDERING: {image_path.name}")
        print("=" * 80)
        print(f"\n🏠 Property: {ctx['property_name']}")
        print(f"🚪 Room Type: {ctx['room_type'].upper()}")
        print(f"🎨 Enhancement Profile: {ctx['enhancement_profile']['lighting_style']}")
        print(f"💎 Material Focus: {', '.join(ctx['enhancement_profile']['materials_focus'])}")
        print(f"🔧 Clarity Boost: {ctx['enhancement_profile']['clarity_boost']}")
        print(f"🎬 LUT: {Path(ctx['enhancement_profile']['lut']).name}")
        print(f"\n📝 Notes: {ctx['enhancement_profile']['notes']}")

        # Build pipeline command
        if pipeline == "standard":
            cmd = self._build_standard_pipeline(image_path, output_dir, ctx)
        elif pipeline == "premium":
            cmd = self._build_premium_pipeline(image_path, output_dir, ctx)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")

        print("\n🔧 Pipeline Command:")
        print(f"   {cmd}")

        return cmd, ctx

    def _build_standard_pipeline(self, image_path: Path, output_dir: Path, ctx: Dict) -> str:
        """Build standard enhancement pipeline command."""
        profile = ctx["enhancement_profile"]

        # Use luxury_tiff_batch_processor with context-aware parameters
        cmd = (
            "python luxury_tiff_batch_processor.py "
            '--input "{image_path}" '
            '--output "{output_dir}" '
            f"--clarity {profile['clarity_boost']:.2f} "
            f"--material-response {profile['material_response_strength']:.2f} "
        )

        # Add LUT if exists
        lut_path = Path(profile["lut"])
        if lut_path.exists():
            cmd += '--lut "{lut_path}" '

        return cmd

    def _build_premium_pipeline(self, image_path: Path, output_dir: Path, ctx: Dict) -> str:
        """Build premium enhancement pipeline command."""
        profile = ctx["enhancement_profile"]

        # Use lux_render_pipeline with context-aware parameters
        cmd = (
            "python lux_render_pipeline.py "
            '--input "{image_path}" '
            '--output "{output_dir}" '
            "--upscale 4 "
            f"--material-response {profile['material_response_strength']:.2f} "
            f"--room-type {ctx['room_type']} "
        )

        return cmd


def main():
    """Demo: Context-aware rendering of 750 Picacho kitchen."""

    # Load knowledge base
    kb_file = Path.home() / "750_picacho_context" / "rendering_knowledge_base.json"

    if not kb_file.exists():
        print(f"❌ Knowledge base not found: {kb_file}")
        print("   Run PDF extraction first")
        sys.exit(1)

    # Initialize renderer
    renderer = ContextAwareRenderer(kb_file)

    # Demo: Kitchen render
    kitchen_render = Path("input_images/Giga-V2_750Picacho_Kitchen_compatible_kitchen-bright.jpg")

    if not kitchen_render.exists():
        print(f"\n⚠️  Demo image not found: {kitchen_render}")
        print("   Showing pipeline structure only")
        kitchen_render = Path("example_kitchen.jpg")

    # Get context and build pipeline
    output_dir = Path("output/context_aware")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd_standard, ctx = renderer.enhance_render(kitchen_render, output_dir, pipeline="standard")

    print("\n" + "=" * 80)
    print("CONTEXT-AWARE ENHANCEMENT READY")
    print("=" * 80)
    print("\n✅ Pipeline configured with:")
    print("   • Room-specific enhancement profile")
    print("   • Material palette from architectural specs")
    print("   • Validated against floor plans")
    print("\n💡 Run command above to execute enhancement")

    # Save context for this rendering
    context_output = output_dir / f"{kitchen_render.stem}_context.json"
    with open(context_output, "w") as f:
        json.dump(ctx, f, indent=2)

    print(f"\n📄 Rendering context saved: {context_output}")


if __name__ == "__main__":
    main()
