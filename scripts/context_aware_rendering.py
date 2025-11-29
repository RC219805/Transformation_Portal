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
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# Lazy import for architectural_context_extractor (requires PyMuPDF in some cases)
ArchitecturalContextExtractor = None
ProjectContext = None
RoomContext = None


def _ensure_context_imports():
    """Lazily import architectural context extractor to handle missing PyMuPDF."""
    global ArchitecturalContextExtractor, ProjectContext, RoomContext
    if ProjectContext is None:
        from architectural_context_extractor import (
            ArchitecturalContextExtractor as _ACE,
            ProjectContext as _PC,
            RoomContext as _RC,
        )
        ArchitecturalContextExtractor = _ACE
        ProjectContext = _PC
        RoomContext = _RC


# Optional imports for processing pipelines - lazy loaded for startup performance
_depth_pipeline_available = None
_tiff_processor_available = None

logger = logging.getLogger(__name__)


def _get_device():
    """Detect available compute device for depth processing.

    Returns:
        Device string for depth pipeline ('pytorch_cuda', 'pytorch_mps', or 'pytorch_cpu')
    """
    try:
        import torch
        # Check CUDA availability
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            return 'pytorch_cuda'
        # Check MPS availability (Apple Silicon)
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_available():
                return 'pytorch_mps'
    except (ImportError, AttributeError):
        pass
    return 'pytorch_cpu'


def _check_depth_pipeline():
    """Check if depth pipeline is available."""
    global _depth_pipeline_available
    if _depth_pipeline_available is None:
        try:
            from transformation_portal.depth.pipeline import ArchitecturalDepthPipeline  # noqa: F401
            _depth_pipeline_available = True
        except ImportError:
            _depth_pipeline_available = False
            logger.warning("Depth pipeline not available - install transformation_portal[ml]")
    return _depth_pipeline_available


def _check_tiff_processor():
    """Check if TIFF processor is available."""
    global _tiff_processor_available
    if _tiff_processor_available is None:
        try:
            from luxury_tiff_batch_processor.adjustments import AdjustmentSettings  # noqa: F401
            from luxury_tiff_batch_processor.adjustments import apply_adjustments  # noqa: F401
            _tiff_processor_available = True
        except ImportError:
            _tiff_processor_available = False
            logger.warning("TIFF processor not available - install luxury_tiff_batch_processor")
    return _tiff_processor_available


def _image_to_array(image_path: Path) -> np.ndarray:
    """Load image as normalized float32 array.

    Args:
        image_path: Path to input image

    Returns:
        Normalized float32 array (H, W, C) with values in [0, 1]
    """
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def _array_to_image(arr: np.ndarray, output_path: Path, quality: int = 95):
    """Save float32 array as image.

    Args:
        arr: Float32 array (H, W, C) with values in [0, 1]
        output_path: Output path
        quality: JPEG quality (if applicable)
    """
    arr_clipped = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr_clipped)
    if output_path.suffix.lower() in ('.jpg', '.jpeg'):
        img.save(output_path, quality=quality)
    else:
        img.save(output_path)


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
        'kitchen': RenderingStrategy(
            room_type='kitchen',
            primary_materials=['metal', 'stone', 'wood', 'glass'],
            lighting_style='bright',
            depth_emphasis='balanced',
            color_temperature='neutral',
            enhancement_strength=0.75,
            lut_preset='signature_estate',
        ),
        'bathroom': RenderingStrategy(
            room_type='bathroom',
            primary_materials=['stone', 'glass', 'metal', 'tile'],
            lighting_style='soft',
            depth_emphasis='balanced',
            color_temperature='neutral',
            enhancement_strength=0.7,
            lut_preset='serene_spa',
        ),
        'bedroom': RenderingStrategy(
            room_type='bedroom',
            primary_materials=['wood', 'fabric', 'leather'],
            lighting_style='soft',
            depth_emphasis='atmospheric',
            color_temperature='warm',
            enhancement_strength=0.6,
            lut_preset='warm_invitation',
        ),
        'living': RenderingStrategy(
            room_type='living',
            primary_materials=['wood', 'fabric', 'stone', 'leather'],
            lighting_style='ambient',
            depth_emphasis='balanced',
            color_temperature='warm',
            enhancement_strength=0.7,
            lut_preset='golden_hour_interior',
        ),
        'outdoor': RenderingStrategy(
            room_type='outdoor',
            primary_materials=['stone', 'concrete', 'wood', 'metal'],
            lighting_style='natural',
            depth_emphasis='atmospheric',
            color_temperature='neutral',
            enhancement_strength=0.8,
            lut_preset='golden_hour_courtyard',
        ),
    }

    def __init__(
        self,
        project_context: ProjectContext,
        output_dir: Path = None
    ):
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
            'kitchen': ['kitch', 'cook'],
            'bathroom': ['bath', 'powder'],
            'bedroom': ['bed', 'master', 'primary'],
            'living': ['great room', 'living', 'family'],
            'outdoor': ['pool', 'patio', 'deck', 'terrace', 'courtyard', 'exterior'],
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
                room_type='unknown',
                primary_materials=self.context.materials_palette[:4] if self.context.materials_palette else ['wood', 'stone'],
                lighting_style='ambient',
                depth_emphasis='balanced',
                color_temperature='neutral',
                enhancement_strength=0.7,
            )

        # Get base strategy for room type
        base_strategy = self.ROOM_STRATEGIES.get(room_type)
        if not base_strategy:
            return self.derive_strategy(image_path)  # Fallback to default

        # Get room-specific context (reserved for future material/feature customization)
        room_context = self.get_room_context(room_type)  # noqa: F841

        # Adjust materials based on project palette
        if self.context.materials_palette:
            # Prioritize materials that appear in both strategy and project
            matched_materials = [
                mat for mat in base_strategy.primary_materials
                if mat in self.context.materials_palette
            ]
            if matched_materials:
                base_strategy.primary_materials = matched_materials

        # Adjust based on design style
        if self.context.design_style:
            style_lower = self.context.design_style.lower()
            if 'modern' in style_lower or 'contemporary' in style_lower:
                base_strategy.color_temperature = 'neutral'
                base_strategy.enhancement_strength = min(base_strategy.enhancement_strength + 0.1, 1.0)
            elif 'traditional' in style_lower:
                base_strategy.color_temperature = 'warm'
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
            'model_size': 'small',
            'device': _get_device(),  # Auto-detect CUDA/MPS/CPU
            'depth_emphasis': strategy.depth_emphasis,  # Store for pipeline config building
        }

        # Depth emphasis
        if strategy.depth_emphasis == 'foreground':
            base_config['zone_weights'] = {
                'foreground': 1.0,
                'midground': 0.6,
                'background': 0.3,
            }
        elif strategy.depth_emphasis == 'atmospheric':
            base_config['zone_weights'] = {
                'foreground': 0.6,
                'midground': 0.8,
                'background': 1.0,
            }
        else:  # balanced
            base_config['zone_weights'] = {
                'foreground': 0.8,
                'midground': 1.0,
                'background': 0.8,
            }

        # Tone mapping based on lighting style
        tone_map_operators = {
            'bright': 'reinhard',
            'soft': 'hable',
            'dramatic': 'filmic',
            'natural': 'agx',
            'ambient': 'agx',
        }
        base_config['tone_map'] = tone_map_operators.get(
            strategy.lighting_style,
            'agx'
        )

        return base_config

    def generate_material_config(self, strategy: RenderingStrategy) -> Dict:
        """Generate material response configuration from strategy."""
        config = {
            'enabled_surfaces': strategy.primary_materials,
            'global_strength': strategy.enhancement_strength,
            'preserve_highlights': True,
            'micro_contrast': 0.15,
        }

        # Per-material strengths
        material_strengths = {}
        for i, material in enumerate(strategy.primary_materials):
            # Primary materials get higher strength
            strength = strategy.enhancement_strength * (1.0 - i * 0.1)
            material_strengths[material] = max(strength, 0.5)

        config['material_strengths'] = material_strengths

        return config

    def generate_color_config(self, strategy: RenderingStrategy) -> Dict:
        """Generate color grading configuration from strategy."""
        config = {
            'lut_preset': strategy.lut_preset,
            'lut_strength': 0.7,
        }

        # Temperature adjustments
        temp_adjustments = {
            'warm': {'saturation': 1.08, 'tint': 5},
            'neutral': {'saturation': 1.05, 'tint': 0},
            'cool': {'saturation': 1.03, 'tint': -5},
        }

        adjustments = temp_adjustments.get(strategy.color_temperature, temp_adjustments['neutral'])
        config.update(adjustments)

        return config

    def process_render(
        self,
        image_path: Path,
        apply_depth: bool = True,
        apply_material: bool = True,
        apply_color: bool = True,
    ) -> Dict:
        """
        Process render with context-aware intelligence.

        Args:
            image_path: Path to rendering
            apply_depth: Apply depth-aware processing
            apply_material: Apply material response (not yet implemented)
            apply_color: Apply color grading

        Returns:
            Dict with processing results including output_path, strategy_path,
            strategy, processing_applied, and config details.
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
        with open(strategy_path, 'w') as f:
            json.dump({
                'strategy': {
                    'room_type': strategy.room_type,
                    'primary_materials': strategy.primary_materials,
                    'lighting_style': strategy.lighting_style,
                    'depth_emphasis': strategy.depth_emphasis,
                    'color_temperature': strategy.color_temperature,
                    'enhancement_strength': strategy.enhancement_strength,
                    'lut_preset': strategy.lut_preset,
                },
                'depth_config': depth_config,
                'material_config': material_config,
                'color_config': color_config,
            }, f, indent=2)

        print(f"\n✓ Strategy saved: {strategy_path}")

        # Load image for processing
        image_arr = _image_to_array(image_path)
        processing_log = []

        # 1. Apply depth-aware processing if enabled and available
        if apply_depth and depth_config:
            result = self._apply_depth_processing(image_arr, depth_config, image_path)
            if result is not None:
                image_arr = result
                processing_log.append("depth_processing")
                print("✓ Applied depth-aware processing")
            else:
                print("⚠ Depth processing skipped (pipeline unavailable)")

        # 2. Apply color grading if enabled and available
        if apply_color and color_config:
            result = self._apply_color_grading(image_arr, color_config, strategy)
            if result is not None:
                image_arr = result
                processing_log.append("color_grading")
                print("✓ Applied color grading")
            else:
                print("⚠ Color grading skipped (processor unavailable)")

        # 3. Material response integration (placeholder - complex integration)
        if apply_material and material_config:
            print("💡 Material response config generated (manual integration available)")
            processing_log.append("material_config_generated")

        # Determine output format based on input
        input_suffix = image_path.suffix.lower()
        output_suffix = input_suffix if input_suffix in ('.png', '.jpg', '.jpeg', '.tif', '.tiff') else '.png'
        output_path = self.output_dir / f"{image_path.stem}_enhanced{output_suffix}"

        # Save processed result
        _array_to_image(image_arr, output_path)
        print(f"\n✓ Enhanced image saved: {output_path}")

        # Return result with metadata
        return {
            'output_path': output_path,
            'strategy_path': strategy_path,
            'strategy': strategy,
            'processing_applied': processing_log,
            'depth_config': depth_config,
            'material_config': material_config,
            'color_config': color_config,
        }

    def _apply_depth_processing(
        self,
        image_arr: np.ndarray,
        depth_config: Dict,
        image_path: Path,
    ) -> Optional[np.ndarray]:
        """Apply depth-aware processing using ArchitecturalDepthPipeline.

        Note: The depth pipeline currently requires a file path as input.
        This method saves the current image_arr state to a temp file before
        processing to preserve any prior modifications in the processing chain.

        Args:
            image_arr: Input image array (H, W, C) normalized to [0, 1]
            depth_config: Depth processing configuration
            image_path: Original image path (used for naming)

        Returns:
            Processed image array, or None if pipeline unavailable
        """
        if not _check_depth_pipeline():
            return None

        try:
            from transformation_portal.depth.pipeline import ArchitecturalDepthPipeline

            # Build YAML-compatible config for the pipeline
            pipeline_config = self._build_depth_pipeline_config(depth_config)

            # Create pipeline instance
            pipeline = ArchitecturalDepthPipeline(pipeline_config)

            # Save current image state to temp file for depth pipeline
            # This preserves any prior processing in the chain
            temp_path = self.output_dir / f"_temp_{image_path.name}"
            _array_to_image(image_arr, temp_path)

            try:
                # Process the image
                result = pipeline.process_render(temp_path)
                return result['image']
            finally:
                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()

        except Exception as e:
            logger.warning(f"Depth processing failed: {e}")
            return None

    def _build_depth_pipeline_config(self, depth_config: Dict) -> Dict:
        """Build full pipeline config from context-derived depth config.

        Args:
            depth_config: Simplified depth config from generate_depth_config

        Returns:
            Full configuration dict for ArchitecturalDepthPipeline
        """
        # Map tone mapping method to pipeline config
        tone_method = depth_config.get('tone_map', 'agx')
        zone_weights = depth_config.get('zone_weights', {
            'foreground': 0.8,
            'midground': 1.0,
            'background': 0.8,
        })

        return {
            'depth_model': {
                'variant': depth_config.get('model_size', 'small'),
                'backend': depth_config.get('device', 'pytorch_cpu'),
                'precision': 'fp16',
                'cache_size': 50,
                'enable_disk_cache': False,
            },
            'processing': {
                'depth_aware_denoise': {
                    'enabled': True,
                    'sigma_spatial': 3.0,
                    'sigma_range': 0.1,
                    'edge_threshold': 0.05,
                    'preserve_strength': 0.8,
                },
                'zone_tone_mapping': {
                    'enabled': True,
                    'num_zones': 3,
                    'method': tone_method,
                    'transition_sigma': 2.0,
                    'zone_params': {
                        'foreground': {'exposure': zone_weights.get('foreground', 0.8)},
                        'midground': {'exposure': zone_weights.get('midground', 1.0)},
                        'background': {'exposure': zone_weights.get('background', 0.8)},
                    },
                },
                'atmospheric_effects': {
                    'enabled': depth_config.get('depth_emphasis') != 'foreground',
                    'haze_density': 0.015,
                    'haze_color': [0.7, 0.8, 0.9],
                    'desaturation_strength': 0.3,
                    'depth_scale': 100.0,
                    'enable_color_shift': True,
                },
                'depth_guided_filters': {
                    'enabled': True,
                    'clarity_strength': 0.5,
                    'edge_preserve_threshold': 0.05,
                    'scale_count': 3,
                    'adaptive_to_depth': True,
                },
            },
            'output': {
                'output_format': 'png',
                'jpeg_quality': 95,
                'depth_colormap': 'turbo',
            },
        }

    def _apply_color_grading(
        self,
        image_arr: np.ndarray,
        color_config: Dict,
        strategy: RenderingStrategy,
    ) -> Optional[np.ndarray]:
        """Apply color grading using luxury_tiff_batch_processor adjustments.

        Args:
            image_arr: Input image array (H, W, C) normalized to [0, 1]
            color_config: Color grading configuration
            strategy: Rendering strategy for additional context

        Returns:
            Color-graded image array, or None if processor unavailable
        """
        if not _check_tiff_processor():
            return None

        try:
            from luxury_tiff_batch_processor.adjustments import (
                AdjustmentSettings,
                apply_adjustments,
            )

            # Map strategy to adjustment settings
            saturation_delta = (color_config.get('saturation', 1.05) - 1.0)
            tint = color_config.get('tint', 0)

            # Base adjustment settings informed by strategy
            adjustments = AdjustmentSettings(
                exposure=0.08 if strategy.lighting_style == 'bright' else 0.0,
                white_balance_temp=self._get_temp_from_strategy(strategy),
                white_balance_tint=float(tint),
                shadow_lift=0.18 if strategy.lighting_style in ('soft', 'ambient') else 0.12,
                highlight_recovery=0.15,
                midtone_contrast=0.08,
                vibrance=0.18 * strategy.enhancement_strength,
                saturation=np.clip(saturation_delta, -1.0, 1.0),
                clarity=0.16 * strategy.enhancement_strength,
                chroma_denoise=0.08,
                glow=0.05 if strategy.lighting_style == 'soft' else 0.02,
            )

            # Apply adjustments
            result = apply_adjustments(image_arr, adjustments)

            return result

        except Exception as e:
            logger.warning(f"Color grading failed: {e}")
            return None

    def _get_temp_from_strategy(self, strategy: RenderingStrategy) -> float:
        """Map color temperature strategy to Kelvin value.

        Args:
            strategy: Rendering strategy

        Returns:
            Color temperature in Kelvin
        """
        temp_map = {
            'warm': 5600.0,
            'neutral': 6500.0,
            'cool': 7500.0,
        }
        return temp_map.get(strategy.color_temperature, 6500.0)


def main():
    """CLI for context-aware rendering."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Context-aware architectural rendering'
    )
    parser.add_argument('image', type=Path, help='Rendering to process')
    parser.add_argument('--context', '-c', type=Path, required=True,
                        help='Path to extracted context JSON or PDF')
    parser.add_argument('--output', '-o', type=Path, default=Path('output_context_aware'),
                        help='Output directory')
    parser.add_argument('--no-depth', action='store_true',
                        help='Skip depth processing')
    parser.add_argument('--no-material', action='store_true',
                        help='Skip material response')
    parser.add_argument('--no-color', action='store_true',
                        help='Skip color grading')

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
        # Reconstruct context (simplified)
        from architectural_context_extractor import RoomContext
        rooms = {}
        for room_key, room_data in context_data.get('rooms', {}).items():
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

    # Initialize pipeline
    pipeline = ContextAwareRenderingPipeline(
        project_context=context,
        output_dir=args.output
    )

    # Process render
    output = pipeline.process_render(
        args.image,
        apply_depth=not args.no_depth,
        apply_material=not args.no_material,
        apply_color=not args.no_color,
    )

    print("\n✓ Processing complete")
    if isinstance(output, dict):
        print(f"  Output: {output.get('output_path', 'N/A')}")
        print(f"  Strategy: {output.get('strategy_path', 'N/A')}")
        print(f"  Processing applied: {', '.join(output.get('processing_applied', []))}")
    else:
        print(f"  Strategy: {output}")

    return 0


if __name__ == '__main__':
    exit(main())
