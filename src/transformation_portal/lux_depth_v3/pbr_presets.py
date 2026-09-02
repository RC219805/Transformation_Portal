"""Production-ready EnhanceConfig presets for PBR map generation.

This module provides three optimized configurations for luxury real estate
architectural visualization workflows:

1. STANDARD_QUALITY: Balanced quality/speed for typical batch processing
2. PREMIUM_QUALITY: Maximum quality for hero shots and client deliverables
3. FAST_PREVIEW: Quick iteration and internal review

All presets use explicit commercial-safe Depth Anything V3 registry models on
Apple Silicon (MPS) or CUDA GPUs.

Example:
    >>> from pathlib import Path
    >>> from transformation_portal.lux_depth_v3.pbr_presets import (
    ...     STANDARD_QUALITY,
    ... )
    >>> from transformation_portal.lux_depth_v3.execution_lifecycle import (
    ...     prepare_lux_execution,
    ... )
    >>> from transformation_portal.lux_depth_v3.orchestrator import (
    ...     EnhanceOrchestrator,
    ... )
    >>> input_root = Path("./input_images")
    >>> input_files = sorted(input_root.glob("*.jpg"))
    >>> prepared = prepare_lux_execution(STANDARD_QUALITY, input_root, input_files)
    >>> orchestrator = EnhanceOrchestrator.from_prepared(prepared, Path("./output"))
"""

from transformation_portal.lux_depth_v3.config import EnhanceConfig

# Standard Quality - Balanced preset for typical real estate imagery
# Throughput: ~200-250 images/hour
# Memory: 4-6 GB peak
# Use Case: Batch workflows (10-100 images), client previews
STANDARD_QUALITY = EnhanceConfig(
    # PBR Generation
    generate_pbr=True,
    save_float_depth=True,  # CRITICAL: High-precision depth for quality PBR
    # Normal Map - Moderate detail with slight smoothing
    pbr_normal_strength=1.2,
    pbr_normal_blur_radius=1,
    # Roughness Map - Balanced detail preservation
    pbr_roughness_strength=1.0,
    pbr_roughness_blur_radius=3,
    # Ambient Occlusion - Natural shadows
    pbr_ao_strength=1.0,
    pbr_ao_blur_radius=5,
    pbr_ao_bias=0.45,
    # Commercial-safe DA3 metric model
    model_key="da3-metric",
    depth_device="mps",  # Use "cuda" for NVIDIA GPUs
)


# Premium Quality - Maximum quality for hero shots and marketing materials
# Throughput: ~100-150 images/hour
# Memory: 5-7 GB peak
# Use Case: Client-facing deliverables, marketing materials
PREMIUM_QUALITY = EnhanceConfig(
    # PBR Generation
    generate_pbr=True,
    save_float_depth=True,  # MANDATORY for premium quality
    # Normal Map - Maximum detail, no pre-blur
    pbr_normal_strength=1.5,
    pbr_normal_blur_radius=0,
    # Roughness Map - High sensitivity to surface detail
    pbr_roughness_strength=1.3,
    pbr_roughness_blur_radius=2,
    # Ambient Occlusion - Deep shadows with wide spread
    pbr_ao_strength=1.2,
    pbr_ao_blur_radius=7,
    pbr_ao_bias=0.40,
    # Commercial-safe DA3 metric model
    model_key="da3-metric",
    depth_device="mps",
)


# Fast Preview - Draft quality for quick iteration and internal review
# Throughput: ~500-700 images/hour
# Memory: 3-4 GB peak
# Use Case: Rapid preview, iteration, non-client work
FAST_PREVIEW = EnhanceConfig(
    # PBR Generation
    generate_pbr=True,
    save_float_depth=False,  # Speed: use PNG depth (lower precision)
    # Normal Map - Skip pre-blur to keep preview generation fast
    # while lower strength limits visual noise.
    pbr_normal_strength=0.8,
    pbr_normal_blur_radius=0,
    # Roughness Map - Light smoothing for fast preview reads
    pbr_roughness_strength=0.7,
    pbr_roughness_blur_radius=2,
    # Ambient Occlusion - Subtle shadows with a bounded blur budget
    pbr_ao_strength=0.8,
    pbr_ao_blur_radius=4,
    pbr_ao_bias=0.50,
    # Commercial-safe DA3 Base model for speed
    model_key="da3-base",
    depth_device="mps",
)


# Material-specific presets for specialized workflows
# These extend STANDARD_QUALITY with tuning for specific material types


WOOD_OPTIMIZED = EnhanceConfig(
    generate_pbr=True,
    save_float_depth=True,
    # Emphasize grain texture and plank boundaries
    pbr_normal_strength=1.3,
    pbr_normal_blur_radius=0,
    # Capture surface variation (satin vs matte finish)
    pbr_roughness_strength=1.2,
    pbr_roughness_blur_radius=2,
    # Natural shadows in plank joints
    pbr_ao_strength=1.0,
    pbr_ao_blur_radius=5,
    pbr_ao_bias=0.45,
    model_key="da3-metric",
    depth_device="mps",
)


METAL_OPTIMIZED = EnhanceConfig(
    generate_pbr=True,
    save_float_depth=True,
    # Moderate strength for smooth reflective surfaces
    pbr_normal_strength=1.0,
    pbr_normal_blur_radius=1,
    # Lower roughness for polished metal
    pbr_roughness_strength=0.8,
    pbr_roughness_blur_radius=4,
    # Strong edge shadows, subtle on flat surfaces
    pbr_ao_strength=1.1,
    pbr_ao_blur_radius=6,
    pbr_ao_bias=0.48,
    model_key="da3-metric",
    depth_device="mps",
)


GLASS_OPTIMIZED = EnhanceConfig(
    generate_pbr=True,
    save_float_depth=True,
    # Low strength for flat glass surfaces
    pbr_normal_strength=0.7,
    pbr_normal_blur_radius=3,
    # Very smooth specular
    pbr_roughness_strength=0.5,
    pbr_roughness_blur_radius=6,
    # Strong frame shadows, bright glass
    pbr_ao_strength=1.2,
    pbr_ao_blur_radius=7,
    pbr_ao_bias=0.55,
    model_key="da3-metric",
    depth_device="mps",
)


STONE_OPTIMIZED = EnhanceConfig(
    generate_pbr=True,
    save_float_depth=True,
    # High strength for texture (veining, surface variation)
    pbr_normal_strength=1.4,
    pbr_normal_blur_radius=0,
    # Natural variation from polished to honed
    pbr_roughness_strength=1.3,
    pbr_roughness_blur_radius=2,
    # Deep grout/joint shadows
    pbr_ao_strength=1.1,
    pbr_ao_blur_radius=5,
    pbr_ao_bias=0.42,
    model_key="da3-metric",
    depth_device="mps",
)


FABRIC_OPTIMIZED = EnhanceConfig(
    generate_pbr=True,
    save_float_depth=True,
    # Moderate for weave patterns and draping
    pbr_normal_strength=1.1,
    pbr_normal_blur_radius=1,
    # Natural fabric variation
    pbr_roughness_strength=1.0,
    pbr_roughness_blur_radius=3,
    # Natural fold shadows with soft spread
    pbr_ao_strength=1.0,
    pbr_ao_blur_radius=6,
    pbr_ao_bias=0.47,
    model_key="da3-metric",
    depth_device="mps",
)


# Preset registry for easy lookup
PRESETS = {
    "standard": STANDARD_QUALITY,
    "premium": PREMIUM_QUALITY,
    "draft": FAST_PREVIEW,
    "wood": WOOD_OPTIMIZED,
    "metal": METAL_OPTIMIZED,
    "glass": GLASS_OPTIMIZED,
    "stone": STONE_OPTIMIZED,
    "fabric": FABRIC_OPTIMIZED,
}


def get_preset(name: str) -> EnhanceConfig:
    """Get preset configuration by name.

    Args:
        name: Preset name (standard, premium, draft,
            wood, metal, glass, stone, fabric)

    Returns:
        EnhanceConfig instance

    Raises:
        ValueError: If preset name not found

    Example:
        >>> config = get_preset("premium")
        >>> config.pbr_normal_strength
        1.5
    """
    name_lower = name.lower()
    if name_lower not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name_lower]


def list_presets() -> list[str]:
    """List all available preset names.

    Returns:
        List of preset names

    Example:
        >>> list_presets()
        ['standard', 'premium', 'draft', 'wood',
         'metal', 'glass', 'stone', 'fabric']
    """
    return list(PRESETS.keys())
