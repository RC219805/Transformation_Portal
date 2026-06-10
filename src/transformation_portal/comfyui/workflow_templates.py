"""Pre-built workflow templates for common enhancement tasks.

Provides ready-to-use workflows for luxury real estate enhancement:
- Full enhancement pipeline with all components
- Quick iterative enhancement for client feedback
- Material-specific processing
- Location-specific atmospheric rendering
- Multi-variant generation
"""

import logging
from pathlib import Path
from typing import List, Optional

from transformation_portal.comfyui.workflow_builder import Workflow, WorkflowBuilder

logger = logging.getLogger(__name__)


class WorkflowTemplates:
    """Pre-built workflow templates for common tasks."""

    @staticmethod
    def full_luxury_estate_pipeline(
        input_path: str,
        output_path: str,
        location: str = "montecito",
        season: str = "summer",
        time_of_day: str = "golden_hour",
        emotional_target: str = "luxury",
        flux_variant: str = "dev",
        flux_strength: float = 0.45,
        quality_threshold: float = 7.0,
    ) -> Workflow:
        """Complete luxury estate enhancement pipeline."""
        logger.info("Creating full luxury estate pipeline workflow")

        builder = WorkflowBuilder(name="Full Luxury Estate Enhancement")

        workflow = (
            builder.add_input(input_path)
            # Analysis phase
            .add_scene_analysis(detailed=True)
            .add_material_segmentation()
            # Enhancement phase
            .add_flux_enhancement(
                strength=flux_strength,
                num_steps=4,
                variant=flux_variant,
                use_controlnet=True,
                controlnet_types=["depth", "canny"],
            )
            # Atmospheric rendering with PHYSICS GUARDRAILS
            .add_skygan_sky(
                location=location,
                season=season,
                time_of_day=time_of_day,
                cloud_coverage=0.3,
                update_reflections=True,
                auto_correct=True,  # Smart Engine Enabled
            )
            # Aesthetic optimization
            .add_neuroaesthetics_optimization(
                emotional_target=emotional_target,
                optimize_composition=True,
                optimize_color_harmony=True,
                optimize_spatial_frequency=True,
            )
            # Quality validation
            .add_quality_validation(pass_threshold=quality_threshold, warning_threshold=5.0)
            # Output
            .add_output(output_path, quality=95)
            .build()
        )

        return workflow

    @staticmethod
    def quick_iterative_enhancement(
        input_path: str,
        output_path: str,
        flux_strength: float = 0.35,
        quality_threshold: float = 6.0,
    ) -> Workflow:
        """Quick enhancement for rapid iteration."""
        logger.info("Creating quick iterative enhancement workflow")

        builder = WorkflowBuilder(name="Quick Iterative Enhancement")

        workflow = (
            builder.add_input(input_path)
            # Fast enhancement only
            .add_flux_enhancement(
                strength=flux_strength,
                num_steps=4,
                variant="schnell",  # Fastest variant
                use_controlnet=False,
            )
            # Quick validation
            .add_quality_validation(pass_threshold=quality_threshold, warning_threshold=4.0)
            .add_output(output_path, quality=90)
            .build()
        )

        return workflow

    @staticmethod
    def material_specific_enhancement(
        input_path: str,
        output_path: str,
        target_materials: Optional[List[str]] = None,
        flux_strength: float = 0.40,
    ) -> Workflow:
        """Material-aware enhancement pipeline."""
        logger.info("Creating material-specific enhancement workflow")

        builder = WorkflowBuilder(name="Material-Specific Enhancement")

        workflow = (
            builder.add_input(input_path)
            # Detailed material analysis
            .add_material_segmentation(materials=target_materials)
            # Material-aware enhancement
            .add_flux_enhancement(
                strength=flux_strength,
                num_steps=4,
                variant="dev",
                use_controlnet=True,
                controlnet_types=["depth", "canny"],
            )
            # Validate material consistency
            .add_quality_validation(pass_threshold=7.0, check_material_consistency=True)
            .add_output(output_path, quality=95)
            .build()
        )

        return workflow

    @staticmethod
    def location_specific_atmospheric(
        input_path: str,
        output_path: str,
        location: str = "montecito",
        season: str = "summer",
        time_of_day: str = "golden_hour",
        marine_layer: bool = False,
        cloud_coverage: float = 0.3,
    ) -> Workflow:
        """Location-specific atmospheric rendering."""
        logger.info(f"Creating location-specific atmospheric workflow ({location})")

        builder = WorkflowBuilder(name=f"Atmospheric Rendering - {location.title()}")

        workflow = (
            builder.add_input(input_path)
            .add_scene_analysis(detailed=True)
            .add_skygan_sky(
                location=location,
                season=season,
                time_of_day=time_of_day,
                cloud_coverage=cloud_coverage,
                update_reflections=True,
                auto_correct=True,  # Ensure shadows match new sky
            )
            .add_atmospheric_model(
                apply_aerial_perspective=True,
                marine_layer=marine_layer,
                max_distance=1000.0,
            )
            .add_neuroaesthetics_optimization(
                emotional_target="serenity",
                optimize_composition=False,
                optimize_color_harmony=True,
                optimize_spatial_frequency=False,
            )
            .add_output(output_path, quality=95)
            .build()
        )

        return workflow

    @staticmethod
    def multi_variant_generation(
        input_path: str,
        output_dir: str,
        num_variants: int = 3,
        emotional_targets: Optional[List[str]] = None,
        flux_strengths: Optional[List[float]] = None,
    ) -> List[Workflow]:
        """Generate multiple enhancement variants."""
        logger.info(f"Creating multi-variant generation workflows ({num_variants} variants)")

        if num_variants < 1:
            raise ValueError("num_variants must be >= 1")
        if emotional_targets is not None and not emotional_targets:
            raise ValueError("emotional_targets must not be empty when provided")
        if flux_strengths is not None and not flux_strengths:
            raise ValueError("flux_strengths must not be empty when provided")

        # Default emotional targets
        if emotional_targets is None:
            emotional_targets = ["luxury", "aspiration", "comfort"][:num_variants]

        # Default strengths
        if flux_strengths is None:
            flux_strengths = [0.35, 0.45, 0.55][:num_variants]

        # Ensure we have enough targets and strengths
        while len(emotional_targets) < num_variants:
            emotional_targets.append(emotional_targets[0])
        while len(flux_strengths) < num_variants:
            flux_strengths.append(0.45)

        workflows = []
        output_path_obj = Path(output_dir)

        for i in range(num_variants):
            emotional_target = emotional_targets[i]
            strength = flux_strengths[i]
            variant_name = f"variant_{i+1}_{emotional_target}"
            output_path = str(output_path_obj / f"{variant_name}.jpg")

            builder = WorkflowBuilder(name=f"Variant {i+1} - {emotional_target.title()}")

            workflow = (
                builder.add_input(input_path)
                .add_scene_analysis(detailed=True)
                .add_flux_enhancement(strength=strength, num_steps=4, variant="dev", use_controlnet=True)
                .add_neuroaesthetics_optimization(emotional_target=emotional_target)
                .add_quality_validation(pass_threshold=6.5)
                .add_output(output_path, quality=92)
                .build()
            )

            workflows.append(workflow)

        return workflows

    @staticmethod
    def coastal_property_golden_hour(
        input_path: str,
        output_path: str,
        location: str = "montecito",
        season: str = "summer",
        include_marine_layer: bool = False,
    ) -> Workflow:
        """Specialized workflow for coastal properties at golden hour."""
        logger.info("Creating coastal property golden hour workflow")

        builder = WorkflowBuilder(name="Coastal Golden Hour Enhancement")

        workflow = (
            builder.add_input(input_path)
            .add_scene_analysis(detailed=True)
            .add_material_segmentation()
            .add_flux_enhancement(
                strength=0.42,
                num_steps=4,
                variant="dev",
                use_controlnet=True,
                controlnet_types=["depth", "canny"],
            )
            .add_skygan_sky(
                location=location,
                season=season,
                time_of_day="golden_hour",
                cloud_coverage=0.2,
                update_reflections=True,
                auto_correct=True,  # Critical for Golden Hour alignment
            )
            .add_atmospheric_model(
                apply_aerial_perspective=True,
                marine_layer=include_marine_layer,
                max_distance=1500.0,
            )
            .add_neuroaesthetics_optimization(
                emotional_target="aspiration",
                optimize_composition=True,
                optimize_color_harmony=True,
                optimize_spatial_frequency=True,
            )
            .add_quality_validation(pass_threshold=7.5, warning_threshold=6.0)
            .add_output(output_path, quality=98)
            .build()
        )

        return workflow

    @staticmethod
    def save_all_templates(output_dir: str) -> None:
        """Save all workflow templates as JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving workflow templates to {output_dir}")

        # Full pipeline
        full_pipeline = WorkflowTemplates.full_luxury_estate_pipeline(input_path="input.jpg", output_path="output.jpg")
        full_pipeline.save(output_path / "full_luxury_estate_pipeline.json")

        # Quick iterative
        quick_enhancement = WorkflowTemplates.quick_iterative_enhancement(input_path="input.jpg", output_path="output.jpg")
        quick_enhancement.save(output_path / "quick_iterative_enhancement.json")

        # Material-specific
        material_enhancement = WorkflowTemplates.material_specific_enhancement(
            input_path="input.jpg", output_path="output.jpg"
        )
        material_enhancement.save(output_path / "material_specific_enhancement.json")

        # Atmospheric
        atmospheric = WorkflowTemplates.location_specific_atmospheric(input_path="input.jpg", output_path="output.jpg")
        atmospheric.save(output_path / "location_specific_atmospheric.json")

        # Coastal golden hour
        coastal = WorkflowTemplates.coastal_property_golden_hour(input_path="input.jpg", output_path="output.jpg")
        coastal.save(output_path / "coastal_property_golden_hour.json")

        # Multi-variant
        variants = WorkflowTemplates.multi_variant_generation(input_path="input.jpg", output_dir="variants")
        for i, variant in enumerate(variants):
            variant.save(output_path / f"multi_variant_{i+1}.json")

        logger.info(f"Saved {5 + len(variants)} workflow templates")


# Export
__all__ = ["WorkflowTemplates"]
