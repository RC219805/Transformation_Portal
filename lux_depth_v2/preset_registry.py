"""Preset Registry for Lux Depth V2 Pipeline.

Provides centralized preset governance with:
- Preset discovery (--list-presets)
- Preset inspection (--describe-preset <name>)
- Validation and metadata

This ensures presets are documented, discoverable, and validated.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .config import Preset
from .schemas import PresetMetadata


# Preset Registry - Single source of truth
PRESET_REGISTRY: Dict[str, PresetMetadata] = {
    Preset.PHOTO_REALISTIC.value: PresetMetadata(
        name=Preset.PHOTO_REALISTIC.value,
        display_name="Photo Realistic",
        description="Balanced photorealistic rendering with moderate enhancements",
        intended_use="General-purpose architectural rendering",
        quality_tier="standard",
        stability="stable",
        performance={"throughput_img_hr": "200-300", "memory_gb": "2-4"},
        parameters={
            "exposure": 0.0,
            "contrast": 1.05,
            "saturation": 1.0,
            "clarity": 0.15,
        },
    ),

    Preset.INTERIOR_LUXURY.value: PresetMetadata(
        name=Preset.INTERIOR_LUXURY.value,
        display_name="Interior Luxury",
        description="Optimized for luxury interior spaces with warm tones",
        intended_use="High-end residential interiors, boutique hotels",
        quality_tier="max",
        stability="stable",
        performance={"throughput_img_hr": "150-200", "memory_gb": "3-5"},
        parameters={
            "exposure": 0.05,
            "contrast": 1.08,
            "saturation": 1.02,
            "clarity": 0.20,
            "warmth": 1.05,
        },
    ),

    Preset.INTERIOR_LUXURY_MAX_QUALITY.value: PresetMetadata(
        name=Preset.INTERIOR_LUXURY_MAX_QUALITY.value,
        display_name="Interior Luxury (Max Quality)",
        description="Maximum quality interior rendering with enhanced details",
        intended_use="Portfolio pieces, hero shots, marketing campaigns",
        quality_tier="max",
        stability="stable",
        performance={"throughput_img_hr": "100-150", "memory_gb": "4-6"},
        parameters={
            "exposure": 0.05,
            "contrast": 1.10,
            "saturation": 1.03,
            "clarity": 0.25,
            "detail": 0.20,
        },
    ),

    Preset.INTERIOR_LUXURY_APEX_QUALITY.value: PresetMetadata(
        name=Preset.INTERIOR_LUXURY_APEX_QUALITY.value,
        display_name="Interior Luxury (Apex Quality)",
        description="Absolute maximum quality with all enhancements enabled",
        intended_use="Award submissions, flagship marketing, archival",
        quality_tier="apex",
        stability="stable",
        performance={"throughput_img_hr": "50-100", "memory_gb": "6-8"},
        parameters={
            "exposure": 0.05,
            "contrast": 1.12,
            "saturation": 1.05,
            "clarity": 0.30,
            "detail": 0.25,
            "materials_v2": True,
        },
    ),

    Preset.EXTERIOR_SHOWCASE.value: PresetMetadata(
        name=Preset.EXTERIOR_SHOWCASE.value,
        display_name="Exterior Showcase",
        description="Optimized for exterior architecture with enhanced sky and landscape",
        intended_use="Building exteriors, landscaping, aerial shots",
        quality_tier="max",
        stability="stable",
        performance={"throughput_img_hr": "150-200", "memory_gb": "3-5"},
        parameters={
            "exposure": 0.0,
            "contrast": 1.10,
            "saturation": 1.08,
            "clarity": 0.22,
            "sky_enhance": True,
        },
    ),

    Preset.ARCHITECTURAL.value: PresetMetadata(
        name=Preset.ARCHITECTURAL.value,
        display_name="Architectural",
        description="Clean, precise rendering for technical presentations",
        intended_use="Design reviews, technical documentation",
        quality_tier="standard",
        stability="stable",
        performance={"throughput_img_hr": "250-350", "memory_gb": "2-3"},
        parameters={
            "exposure": 0.0,
            "contrast": 1.05,
            "saturation": 0.98,
            "clarity": 0.18,
        },
    ),

    Preset.ARCHIVAL_QUALITY.value: PresetMetadata(
        name=Preset.ARCHIVAL_QUALITY.value,
        display_name="Archival Quality",
        description="Maximum fidelity with minimal creative adjustments",
        intended_use="Archival documentation, museum-quality prints",
        quality_tier="apex",
        stability="stable",
        performance={"throughput_img_hr": "50-80", "memory_gb": "6-10"},
        parameters={
            "exposure": 0.0,
            "contrast": 1.02,
            "saturation": 1.0,
            "clarity": 0.10,
            "bit_depth": 16,
        },
    ),

    Preset.CI_BASELINE.value: PresetMetadata(
        name=Preset.CI_BASELINE.value,
        display_name="CI Baseline",
        description="Minimal processing for fast CI/CD validation",
        intended_use="Continuous integration testing, smoke tests",
        quality_tier="standard",
        stability="stable",
        performance={"throughput_img_hr": "400-600", "memory_gb": "1-2"},
        parameters={
            "exposure": 0.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "upscale": 2,
        },
    ),

    Preset.PRODUCTION_STANDARD.value: PresetMetadata(
        name=Preset.PRODUCTION_STANDARD.value,
        display_name="Production Standard",
        description="Balanced production preset for consistent output",
        intended_use="Standard production workflows, client deliverables",
        quality_tier="standard",
        stability="stable",
        performance={"throughput_img_hr": "200-300", "memory_gb": "2-4"},
        parameters={
            "exposure": 0.0,
            "contrast": 1.05,
            "saturation": 1.02,
            "clarity": 0.15,
        },
    ),

    Preset.PRODUCTION_ULTRA.value: PresetMetadata(
        name=Preset.PRODUCTION_ULTRA.value,
        display_name="Production Ultra",
        description="High-quality production preset with enhanced details",
        intended_use="Premium client deliverables, portfolio work",
        quality_tier="max",
        stability="stable",
        performance={"throughput_img_hr": "100-150", "memory_gb": "4-6"},
        parameters={
            "exposure": 0.0,
            "contrast": 1.08,
            "saturation": 1.03,
            "clarity": 0.22,
            "detail": 0.18,
        },
    ),

    # Canary/Experimental presets (marked as such)
    Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM.value: PresetMetadata(
        name=Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM.value,
        display_name="Interior Luxury Apex (EfficientSAM)",
        description="Apex quality with EfficientSAM material segmentation",
        intended_use="Experimental - testing EfficientSAM V3 integration",
        quality_tier="apex",
        stability="canary",
        performance={"throughput_img_hr": "40-80", "memory_gb": "8-12"},
        parameters={
            "segmentation_backend": "efficientsam",
            "materials_v2": True,
        },
    ),

    Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_GLASS.value: PresetMetadata(
        name=Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_GLASS.value,
        display_name="Interior Luxury Apex (Materials V3 Glass)",
        description="Apex quality with Materials V3 glass enhancement",
        intended_use="Experimental - testing Materials V3 glass pipeline",
        quality_tier="apex",
        stability="canary",
        performance={"throughput_img_hr": "30-60", "memory_gb": "8-12"},
        parameters={
            "materials_v3": True,
            "glass_enhancement": True,
        },
    ),

    Preset.EXTERIOR_POOL_APEX_QUALITY.value: PresetMetadata(
        name=Preset.EXTERIOR_POOL_APEX_QUALITY.value,
        display_name="Exterior Pool (Apex Quality)",
        description="Optimized for pool and water features",
        intended_use="Pool renders, water features, reflective surfaces",
        quality_tier="apex",
        stability="stable",
        performance={"throughput_img_hr": "80-120", "memory_gb": "5-7"},
        parameters={
            "water_enhance": True,
            "reflection_enhance": True,
            "clarity": 0.28,
        },
    ),

    Preset.EXTERIOR_POOL_APEX_QUALITY_EFFICIENTSAM.value: PresetMetadata(
        name=Preset.EXTERIOR_POOL_APEX_QUALITY_EFFICIENTSAM.value,
        display_name="Exterior Pool Apex (EfficientSAM)",
        description="Pool rendering with EfficientSAM material segmentation",
        intended_use="Experimental - testing EfficientSAM V3 for pool/water scenes",
        quality_tier="apex",
        stability="canary",
        performance={"throughput_img_hr": "60-100", "memory_gb": "6-10"},
        parameters={
            "segmentation_backend": "efficientsam",
            "water_enhance": True,
        },
    ),

    Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_STONE.value: PresetMetadata(
        name=Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_STONE.value,
        display_name="Interior Luxury Apex (Materials V3 Stone)",
        description="Apex quality with Materials V3 stone enhancement",
        intended_use="Experimental - testing Materials V3 stone pipeline",
        quality_tier="apex",
        stability="canary",
        performance={"throughput_img_hr": "30-60", "memory_gb": "8-12"},
        parameters={
            "materials_v3": True,
            "stone_enhancement": True,
        },
    ),

    Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_GLASS_VALIDATE.value: PresetMetadata(
        name=Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_GLASS_VALIDATE.value,
        display_name="Interior Luxury Apex (Materials V3 Glass - Validation)",
        description="Validation-only preset for Materials V3 glass testing",
        intended_use="Internal validation - not for production use",
        quality_tier="apex",
        stability="experimental",
        performance={"throughput_img_hr": "30-60", "memory_gb": "8-12"},
        parameters={
            "materials_v3": True,
            "glass_enhancement": True,
            "validation_mode": True,
        },
    ),

    Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_STONE_VALIDATE.value: PresetMetadata(
        name=Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_STONE_VALIDATE.value,
        display_name="Interior Luxury Apex (Materials V3 Stone - Validation)",
        description="Validation-only preset for Materials V3 stone testing",
        intended_use="Internal validation - not for production use",
        quality_tier="apex",
        stability="experimental",
        performance={"throughput_img_hr": "30-60", "memory_gb": "8-12"},
        parameters={
            "materials_v3": True,
            "stone_enhancement": True,
            "validation_mode": True,
        },
    ),
}


class PresetRegistry:
    """Preset registry for governance and discovery."""

    def __init__(self):
        self.presets = PRESET_REGISTRY
        # Verify registry completeness on initialization
        self._verify_completeness()

    def _verify_completeness(self):
        """Verify that all Preset enum values have registry entries.

        Raises:
            RuntimeError: If any presets are missing from the registry
        """
        from .config import Preset

        enum_presets = set(p.value for p in Preset)
        registry_presets = set(self.presets.keys())
        missing = enum_presets - registry_presets

        if missing:
            raise RuntimeError(
                f"Preset registry is incomplete. Missing entries for: {sorted(missing)}. "
                f"All Preset enum values must have corresponding PRESET_REGISTRY entries."
            )

    def list_presets(self, stability_filter: Optional[str] = None) -> List[PresetMetadata]:
        """List all presets, optionally filtered by stability.

        Args:
            stability_filter: Filter by stability ('stable', 'canary', 'experimental')

        Returns:
            List of preset metadata
        """
        presets = list(self.presets.values())

        if stability_filter:
            presets = [p for p in presets if p.stability == stability_filter]

        return presets

    def get_preset(self, name: str) -> Optional[PresetMetadata]:
        """Get preset metadata by name.

        Args:
            name: Preset name

        Returns:
            PresetMetadata if found, None otherwise
        """
        return self.presets.get(name)

    def validate_preset(self, name: str) -> bool:
        """Validate that a preset exists.

        Args:
            name: Preset name

        Returns:
            True if preset exists, False otherwise
        """
        return name in self.presets

    def get_stable_presets(self) -> List[PresetMetadata]:
        """Get only stable (production-ready) presets."""
        return self.list_presets(stability_filter="stable")

    def get_canary_presets(self) -> List[PresetMetadata]:
        """Get canary (experimental but usable) presets."""
        return self.list_presets(stability_filter="canary")

    def get_by_quality_tier(self, tier: str) -> List[PresetMetadata]:
        """Get presets by quality tier.

        Args:
            tier: Quality tier ('standard', 'max', 'apex')

        Returns:
            List of presets in that tier
        """
        return [p for p in self.presets.values() if p.quality_tier == tier]

    def format_preset_list(self, presets: List[PresetMetadata], show_details: bool = False) -> str:
        """Format preset list for CLI display.

        Args:
            presets: List of presets to format
            show_details: Include detailed information

        Returns:
            Formatted string for terminal output
        """
        if not presets:
            return "No presets found."

        lines = []

        for preset in presets:
            # Basic info
            stability_marker = {
                "stable": "✅",
                "canary": "🚧",
                "experimental": "⚠️",
            }.get(preset.stability, "❓")

            quality_marker = {
                "standard": "⚡",
                "max": "⭐",
                "apex": "💎",
            }.get(preset.quality_tier, "")

            line = f"{stability_marker} {quality_marker} {preset.name}"

            if show_details:
                line += f"\n    {preset.description}"
                line += f"\n    Quality: {preset.quality_tier.upper()} | Stability: {preset.stability}"
                line += f"\n    Use: {preset.intended_use}"

                if preset.performance:
                    perf = preset.performance
                    line += (
                        f"\n    Performance: {perf.get('throughput_img_hr', 'N/A')} img/hr, "
                        f"{perf.get('memory_gb', 'N/A')} GB"
                    )

            lines.append(line)

        return "\n\n".join(lines)

    def format_preset_detail(self, preset: PresetMetadata) -> str:
        """Format detailed preset information.

        Args:
            preset: Preset metadata

        Returns:
            Formatted detail view
        """
        stability_marker = {
            "stable": "✅ Stable",
            "canary": "🚧 Canary (Experimental)",
            "experimental": "⚠️ Experimental",
        }.get(preset.stability, "❓ Unknown")

        quality_marker = {
            "standard": "⚡ Standard Quality",
            "max": "⭐ Max Quality",
            "apex": "💎 Apex Quality",
        }.get(preset.quality_tier, "")

        lines = [
            f"=== {preset.display_name} ===",
            "",
            f"Name: {preset.name}",
            f"Status: {stability_marker}",
            f"Quality: {quality_marker}",
            "",
            "Description:",
            f"  {preset.description}",
            "",
            "Intended Use:",
            f"  {preset.intended_use}",
        ]

        if preset.performance:
            lines.append("")
            lines.append("Performance:")
            for key, value in preset.performance.items():
                lines.append(f"  {key}: {value}")

        if preset.parameters:
            lines.append("")
            lines.append("Parameters:")
            for key, value in preset.parameters.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


# Singleton instance
_registry = PresetRegistry()


def get_registry() -> PresetRegistry:
    """Get the global preset registry."""
    return _registry


def list_presets(stability_filter: Optional[str] = None) -> List[PresetMetadata]:
    """Convenience function to list presets."""
    return _registry.list_presets(stability_filter)


def get_preset(name: str) -> Optional[PresetMetadata]:
    """Convenience function to get a preset."""
    return _registry.get_preset(name)


def validate_preset(name: str) -> bool:
    """Convenience function to validate a preset."""
    return _registry.validate_preset(name)
