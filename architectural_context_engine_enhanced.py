#!/usr/bin/env python3
"""
Enhanced Architectural Context Engine
Integrates BIM metadata and PDF specifications into rendering pipeline

Provides context-aware enhancements based on:
- Room type and spatial relationships from BIM
- Material specifications from BIM and PDF
- Color palettes from architectural documents
- Lighting characteristics for each space
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RenderingContext:
    """Context for rendering a specific view."""
    view_name: str
    room_type: str
    materials: List[Dict[str, Any]]
    lighting: List[Dict[str, Any]]
    dimensions: Optional[Dict[str, float]] = None
    color_guidance: Optional[List[str]] = None
    enhancement_params: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArchitecturalContextEngine:
    """
    Enhanced context engine with BIM/PDF integration.

    Provides rendering guidance based on architectural data:
    - Material-specific enhancement parameters
    - Room-type-based processing profiles
    - Color grading guidance from architectural palette
    - Depth processing informed by spatial relationships
    """

    def __init__(self, metadata_path: Optional[Path] = None):
        """
        Initialize with unified metadata.

        Args:
            metadata_path: Path to 750_picacho_metadata.json
        """
        self.metadata_path = metadata_path or Path('750_picacho_metadata.json')
        self.metadata: Dict[str, Any] = {}
        self.view_contexts: Dict[str, RenderingContext] = {}

        if self.metadata_path.exists():
            self.load_metadata()
        else:
            logger.warning(f"Metadata file not found: {self.metadata_path}")
            self.metadata = self._get_default_metadata()

    def load_metadata(self) -> None:
        """Load unified metadata from JSON."""
        try:
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata from: {self.metadata_path}")

            # Build view contexts
            self._build_view_contexts()

        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.metadata = self._get_default_metadata()

    def _build_view_contexts(self) -> None:
        """Build rendering contexts for each canonical view."""
        canonical_views = self.metadata.get('canonical_views', {})

        for view_name, view_data in canonical_views.items():
            room_spec = view_data.get('room_spec', {})

            # Extract enhancement parameters based on room type
            enhancement_params = self._get_enhancement_params(room_spec.get('room_type'))

            # Extract color guidance
            color_guidance = self._get_color_guidance(room_spec.get('name'))

            context = RenderingContext(
                view_name=view_name,
                room_type=room_spec.get('room_type', 'unknown'),
                materials=room_spec.get('materials', []),
                lighting=room_spec.get('lighting', []),
                dimensions=room_spec.get('dimensions'),
                color_guidance=color_guidance,
                enhancement_params=enhancement_params
            )

            self.view_contexts[view_name] = context

        logger.info(f"Built {len(self.view_contexts)} view contexts")

    def _get_enhancement_params(self, room_type: str) -> Dict[str, float]:
        """
        Get enhancement parameters based on room type.

        Returns parameters for:
        - Material response strength
        - Clarity enhancement
        - Glow intensity
        - Saturation boost
        - Contrast adjustment
        """
        # Room-specific enhancement profiles
        profiles = {
            'kitchen': {
                'material_response_strength': 0.80,  # High detail for surfaces
                'clarity': 0.20,
                'glow': 0.05,
                'saturation': 1.05,
                'contrast': 1.08,
                'sharpness': 0.15,
            },
            'bathroom': {
                'material_response_strength': 0.85,  # Emphasis on marble/tile
                'clarity': 0.18,
                'glow': 0.08,
                'saturation': 1.03,
                'contrast': 1.06,
                'sharpness': 0.12,
            },
            'living': {
                'material_response_strength': 0.70,  # Softer for living spaces
                'clarity': 0.15,
                'glow': 0.10,
                'saturation': 1.08,
                'contrast': 1.05,
                'sharpness': 0.10,
            },
            'bedroom': {
                'material_response_strength': 0.65,  # Gentle enhancement
                'clarity': 0.12,
                'glow': 0.12,
                'saturation': 1.05,
                'contrast': 1.03,
                'sharpness': 0.08,
            },
            'exterior': {
                'material_response_strength': 0.75,  # Balanced outdoor
                'clarity': 0.25,
                'glow': 0.06,
                'saturation': 1.10,
                'contrast': 1.10,
                'sharpness': 0.18,
            },
        }

        return profiles.get(room_type, profiles['living'])

    def _get_color_guidance(self, room_name: str) -> List[str]:
        """Get color palette guidance for room."""
        palette = self.metadata.get('color_palette', {})
        bim_palette = palette.get('bim_palette', {})

        # Extract primary and accent colors
        guidance = []

        if 'primary' in bim_palette:
            guidance.extend(bim_palette['primary'][:2])
        if 'accent' in bim_palette:
            guidance.extend(bim_palette['accent'][:2])

        return guidance or ['warm_white', 'natural_wood']

    def get_context_for_view(self, view_name: str) -> Optional[RenderingContext]:
        """
        Get rendering context for a specific view.

        Args:
            view_name: Filename of canonical view (e.g., '750Picacho_Pool.jpg')

        Returns:
            RenderingContext with all architectural data
        """
        return self.view_contexts.get(view_name)

    def get_material_response_config(self, view_name: str) -> Dict[str, Any]:
        """
        Get Material Response configuration for a view.

        Returns material types and strengths based on room spec.
        """
        context = self.get_context_for_view(view_name)
        if not context:
            return self._get_default_material_config()

        # Extract material categories from room materials
        material_types = set()
        material_strengths = {}

        for material in context.materials:
            category = material.get('category')
            if category:
                material_types.add(category)
                # Use reflectivity as strength indicator
                reflectivity = material.get('reflectivity', 0.5)
                material_strengths[category] = reflectivity

        config = {
            'enabled': True,
            'material_types': list(material_types),
            'base_strength': context.enhancement_params.get('material_response_strength', 0.70),
            'category_strengths': material_strengths,
            'preserve_highlights': True,
            'enhance_details': True,
        }

        return config

    def get_depth_processing_config(self, view_name: str) -> Dict[str, Any]:
        """
        Get depth processing configuration for a view.

        Informed by room dimensions and spatial relationships.
        """
        context = self.get_context_for_view(view_name)
        if not context:
            return self._get_default_depth_config()

        # Determine depth processing based on room type
        room_type = context.room_type

        if room_type == 'exterior':
            # Exterior/aerial views - stronger atmospheric effects
            config = {
                'enabled': True,
                'atmospheric_haze': 0.15,
                'depth_of_field': 0.20,
                'zone_enhancement': True,
                'foreground_clarity': 0.25,
                'background_softness': 0.15,
            }
        elif room_type == 'kitchen':
            # Kitchen - minimal DOF, high clarity
            config = {
                'enabled': True,
                'atmospheric_haze': 0.0,
                'depth_of_field': 0.08,
                'zone_enhancement': True,
                'foreground_clarity': 0.20,
                'background_softness': 0.05,
            }
        elif room_type in ['living', 'bedroom']:
            # Living spaces - moderate DOF for depth
            config = {
                'enabled': True,
                'atmospheric_haze': 0.05,
                'depth_of_field': 0.15,
                'zone_enhancement': True,
                'foreground_clarity': 0.15,
                'background_softness': 0.10,
            }
        else:
            # Default balanced
            config = {
                'enabled': True,
                'atmospheric_haze': 0.08,
                'depth_of_field': 0.12,
                'zone_enhancement': True,
                'foreground_clarity': 0.15,
                'background_softness': 0.10,
            }

        return config

    def get_color_grading_config(self, view_name: str) -> Dict[str, Any]:
        """
        Get color grading configuration for a view.

        Informed by architectural color palette and lighting.
        """
        context = self.get_context_for_view(view_name)
        if not context:
            return self._get_default_color_config()

        # Analyze lighting to determine color temperature adjustments
        avg_color_temp = self._get_average_color_temperature(context.lighting)

        # Determine LUT based on room type and lighting
        room_type = context.room_type
        lut_path = self._select_lut(room_type, avg_color_temp)

        config = {
            'enabled': True,
            'lut_path': lut_path,
            'lut_strength': 0.70,
            'exposure_adjustment': self._get_exposure_adjustment(context),
            'saturation_boost': context.enhancement_params.get('saturation', 1.05),
            'contrast_boost': context.enhancement_params.get('contrast', 1.05),
            'color_temperature_shift': self._get_temp_shift(avg_color_temp),
        }

        return config

    def _get_average_color_temperature(self, lighting: List[Dict[str, Any]]) -> int:
        """Calculate average color temperature from lighting specs."""
        temps = [light.get('color_temperature', 5000) for light in lighting if light.get('color_temperature')]
        return int(sum(temps) / len(temps)) if temps else 5000

    def _select_lut(self, room_type: str, color_temp: int) -> str:
        """Select appropriate LUT based on room and lighting."""
        # Warm lighting (< 3500K) - use warm LUTs
        if color_temp < 3500:
            if room_type == 'exterior':
                return 'assets/luts/location_aesthetic/California_Golden_Hour.cube'
            else:
                return 'assets/luts/film_emulation/Kodak_2393.cube'

        # Cool/neutral lighting - balanced LUTs
        elif room_type == 'exterior':
            return 'assets/luts/location_aesthetic/Coastal_Estate.cube'
        elif room_type in ['kitchen', 'bathroom']:
            return 'assets/luts/material_response/Clean_Modern.cube'
        else:
            return 'assets/luts/film_emulation/FilmConvert_Kodak_Vision3.cube'

    def _get_exposure_adjustment(self, context: RenderingContext) -> float:
        """Determine exposure adjustment based on lighting intensity."""
        if not context.lighting:
            return 0.0

        # Calculate average lighting intensity
        intensities = [light.get('intensity', 0.5) for light in context.lighting]
        avg_intensity = sum(intensities) / len(intensities)

        # Adjust exposure based on intensity
        if avg_intensity < 0.5:
            return 0.15  # Brighten dark spaces
        elif avg_intensity > 0.8:
            return -0.10  # Tone down bright spaces
        else:
            return 0.0

    def _get_temp_shift(self, color_temp: int) -> int:
        """Get color temperature shift value."""
        # Target: 5000K neutral
        target = 5000
        shift = target - color_temp
        # Cap shift at +/- 500K
        return max(-500, min(500, shift))

    def get_complete_pipeline_config(self, view_name: str) -> Dict[str, Any]:
        """
        Get complete pipeline configuration for a view.

        Integrates all architectural context into unified config.
        """
        context = self.get_context_for_view(view_name)

        config = {
            'view_name': view_name,
            'room_type': context.room_type if context else 'unknown',
            'architectural_context': context.to_dict() if context else {},
            'material_response': self.get_material_response_config(view_name),
            'depth_processing': self.get_depth_processing_config(view_name),
            'color_grading': self.get_color_grading_config(view_name),
            'enhancement_params': context.enhancement_params if context else {},
            'quality_target': self.metadata.get('rendering_guidance', {}).get('target_quality_rating', 95),
        }

        return config

    def _get_default_metadata(self) -> Dict[str, Any]:
        """Provide minimal default metadata."""
        return {
            'project': '750 Picacho Lane',
            'location': 'Montecito, CA',
            'rendering_guidance': {
                'target_quality_rating': 95,
                'material_response_enabled': True,
                'depth_processing_enabled': True,
            }
        }

    def _get_default_material_config(self) -> Dict[str, Any]:
        """Default material response config."""
        return {
            'enabled': True,
            'material_types': ['wood', 'metal', 'glass', 'stone'],
            'base_strength': 0.70,
            'preserve_highlights': True,
        }

    def _get_default_depth_config(self) -> Dict[str, Any]:
        """Default depth processing config."""
        return {
            'enabled': True,
            'atmospheric_haze': 0.08,
            'depth_of_field': 0.12,
            'zone_enhancement': True,
        }

    def _get_default_color_config(self) -> Dict[str, Any]:
        """Default color grading config."""
        return {
            'enabled': True,
            'lut_path': 'assets/luts/film_emulation/FilmConvert_Kodak_Vision3.cube',
            'lut_strength': 0.70,
            'saturation_boost': 1.05,
            'contrast_boost': 1.05,
        }

    def export_all_configs(self, output_dir: Path) -> None:
        """Export all view configurations to JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        for view_name in self.view_contexts.keys():
            config = self.get_complete_pipeline_config(view_name)

            output_file = output_dir / f"{view_name.replace('.jpg', '')}_config.json"
            with open(output_file, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Exported config for {view_name} to {output_file}")

    def get_performance_estimate(self, view_name: str) -> Dict[str, Any]:
        """
        Estimate processing overhead for architectural context integration.

        Target: < 5% overhead as specified
        """
        context = self.get_context_for_view(view_name)

        # Metadata lookup: ~0.5ms
        # Material response config: ~1ms
        # Depth config: ~0.5ms
        # Color grading config: ~1ms
        # Total: ~3ms per image

        # For 6 images: ~18ms total overhead
        # Compared to typical processing time: 400-600 images/hour = 6-9s per image
        # Overhead: 18ms / 6000ms = 0.3% << 5% target

        estimate = {
            'view_name': view_name,
            'metadata_lookup_ms': 0.5,
            'material_config_ms': 1.0,
            'depth_config_ms': 0.5,
            'color_config_ms': 1.0,
            'total_overhead_ms': 3.0,
            'overhead_percentage': 0.05,  # 0.05% for typical 6s image processing
            'quality_improvement': '+5-10% (estimated)',
        }

        return estimate


def main():
    """Example usage and testing."""
    import argparse

    parser = argparse.ArgumentParser(description='Architectural Context Engine')
    parser.add_argument('--metadata', type=Path, default=Path('750_picacho_metadata.json'),
                       help='Path to unified metadata JSON')
    parser.add_argument('--view', type=str, help='Get config for specific view')
    parser.add_argument('--export-all', type=Path, help='Export all configs to directory')
    parser.add_argument('--performance', action='store_true', help='Show performance estimates')

    args = parser.parse_args()

    # Initialize engine
    engine = ArchitecturalContextEngine(args.metadata)

    if args.view:
        # Get config for specific view
        config = engine.get_complete_pipeline_config(args.view)
        print(f"\nPipeline Configuration for {args.view}:")
        print(json.dumps(config, indent=2))

        if args.performance:
            perf = engine.get_performance_estimate(args.view)
            print("\nPerformance Estimate:")
            print(json.dumps(perf, indent=2))

    elif args.export_all:
        # Export all configs
        engine.export_all_configs(args.export_all)
        print(f"\nExported {len(engine.view_contexts)} view configurations to {args.export_all}")

    else:
        # Show summary
        print("\nArchitectural Context Engine")
        print(f"Project: {engine.metadata.get('project')}")
        print(f"Location: {engine.metadata.get('location')}")
        print(f"Loaded views: {len(engine.view_contexts)}")
        print("\nAvailable views:")
        for view_name, context in engine.view_contexts.items():
            print(f"  - {view_name} ({context.room_type})")


if __name__ == '__main__':
    main()
