#!/usr/bin/env python3
"""
Hyper-Reality Enhancement Integration Example
Demonstrates how to integrate quantum caustics and neural atmosphere synthesis
with existing Transformation_Portal pipelines
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from enhancements import (
    HyperRealityProcessor,
    EnhancementConfig,
    QualityMode,
    enhance_image
)


def example_basic_enhancement():
    """
    Example 1: Basic enhancement with default settings
    Target: 105/100 quality
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Hyper-Reality Enhancement")
    print("="*60)

    # Simple one-line enhancement
    results = enhance_image(
        image_path="input_images/luxury_estate.jpg",
        output_path="outputs/estate_hyper_105.jpg",
        target_quality=105,
        save_intermediate=False
    )

    print(f"✓ Enhanced to {results['quality_score']}/100 quality")
    print(f"✓ Saved to: {results['output_path']}")


def example_custom_configuration():
    """
    Example 2: Custom configuration for specific effects
    Target: Maximum quantum effects
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Configuration")
    print("="*60)

    # Create custom configuration
    config = EnhancementConfig(
        target_quality=120,
        mode=QualityMode.QUANTUM
    )

    # Customize quantum caustics for pool/water scenes
    config.quantum_caustics['caustic_intensity'] = 3.5
    config.quantum_caustics['entanglement'] = 0.25

    # Enhance atmospheric effects for dramatic skies
    config.neural_atmosphere['enhancement_level'] = 2.2
    config.neural_atmosphere['impossible_colors'] = True

    # Create processor with custom config
    processor = HyperRealityProcessor(config)

    # Process image
    results = processor.process_image(
        image_path="input_images/pool_scene.jpg",
        output_path="outputs/pool_hyper_120.jpg",
        save_intermediate=True
    )

    print(f"✓ Enhanced to {results['quality_score']}/100 quality")


def example_material_specific():
    """
    Example 3: Material-specific enhancement
    Focus on stucco and architectural materials
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Material-Specific Enhancement")
    print("="*60)

    config = EnhancementConfig(target_quality=110)

    # Boost material transcendence for architectural shots
    config.material_transcendence['energy_violation'] = 1.25
    config.material_transcendence['quantum_interference'] = 0.25

    # Reduce water effects (no pool in scene)
    config.material_transcendence['bioluminescence'] = 0.05

    # Enhance spatial harmonics for building surfaces
    config.spatial_harmonics['order'] = 12
    config.spatial_harmonics['directional_boost'] = 2.0

    processor = HyperRealityProcessor(config)

    results = processor.process_image(
        image_path="input_images/architectural_facade.jpg",
        output_path="outputs/facade_hyper_110.jpg"
    )

    print(f"✓ Enhanced to {results['quality_score']}/100 quality")


def example_batch_processing():
    """
    Example 4: Batch process multiple images
    Process entire estate photo set
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Processing")
    print("="*60)

    # Estate image set
    images = [
        "750_picacho_rendering_01.jpg",
        "750_picacho_rendering_02.jpg",
        "750_picacho_rendering_03.jpg",
        "750_picacho_rendering_04.jpg",
        "750_picacho_rendering_05.jpg",
        "750_picacho_rendering_06.jpg"
    ]

    input_dir = Path("input_images/750_picacho")
    output_dir = Path("outputs/hyper_reality/750_picacho")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create processor once for efficiency
    config = EnhancementConfig(target_quality=105)
    processor = HyperRealityProcessor(config)

    results_summary = []

    for img_name in images:
        img_path = input_dir / img_name
        if not img_path.exists():
            print(f"  ⚠️  Skipping {img_name} (not found)")
            continue

        output_path = output_dir / f"{img_path.stem}_hyper_105.jpg"

        print(f"\n→ Processing: {img_name}")

        try:
            results = processor.process_image(
                image_path=str(img_path),
                output_path=str(output_path),
                save_intermediate=False
            )

            results_summary.append({
                'name': img_name,
                'quality': results['quality_score'],
                'time': results['processing_time']
            })

            print(f"  ✓ Quality: {results['quality_score']}/100")
            print(f"  ✓ Time: {results['processing_time']:.2f}s")

        except Exception as e:
            print(f"  ❌ Failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    for r in results_summary:
        print(f"  {r['name']:40s} {r['quality']:3d}/100  {r['time']:5.1f}s")

    avg_quality = sum(r['quality'] for r in results_summary) / len(results_summary)
    total_time = sum(r['time'] for r in results_summary)
    print(f"\n  Average Quality: {avg_quality:.1f}/100")
    print(f"  Total Time: {total_time:.1f}s")


def example_integration_with_existing_pipeline():
    """
    Example 5: Integration with existing Transformation_Portal pipelines
    Demonstrates layering hyper-reality on top of existing processing
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Integration with Existing Pipeline")
    print("="*60)

    # This example shows how to integrate with your existing pipelines
    # Uncomment and adapt to your actual pipeline code:

    # from transformation_portal import YourExistingPipeline

    # # Step 1: Run existing pipeline first
    # existing_pipeline = YourExistingPipeline()
    # intermediate = existing_pipeline.process("input.jpg")

    # # Step 2: Apply hyper-reality enhancement as final polish
    # hyper_results = enhance_image(
    #     image_path=intermediate['output_path'],
    #     target_quality=105,
    #     save_intermediate=False
    # )

    # print(f"✓ Pipeline complete with hyper-reality final polish")
    # print(f"✓ Final quality: {hyper_results['quality_score']}/100")

    print("  See source code for integration template")


def example_selective_enhancement():
    """
    Example 6: Selective enhancement (enable/disable specific stages)
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Selective Enhancement")
    print("="*60)

    # Create config with only specific enhancements enabled
    config = EnhancementConfig(target_quality=95)

    # Disable quantum caustics (no water in scene)
    config.quantum_caustics['enable'] = False

    # Keep neural atmosphere
    config.neural_atmosphere['enable'] = True

    # Disable material transcendence (preserve natural materials)
    config.material_transcendence['enable'] = False

    # Keep spatial harmonics for lighting
    config.spatial_harmonics['enable'] = True

    # Strong synergistic amplification
    config.synergistic['enable'] = True
    config.synergistic['edge_enhancement'] = 1.2
    config.synergistic['local_contrast'] = 1.5

    processor = HyperRealityProcessor(config)

    results = processor.process_image(
        image_path="input_images/interior_shot.jpg",
        output_path="outputs/interior_selective_95.jpg"
    )

    print(f"✓ Selective enhancement complete: {results['quality_score']}/100")
    print(f"  Stages: Atmosphere + Harmonics + Synergistic only")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyper-Reality Enhancement Examples")
    parser.add_argument("--example", type=int, choices=range(1, 7),
                       help="Run specific example (1-6)")
    parser.add_argument("--all", action="store_true",
                       help="Run all examples")

    args = parser.parse_args()

    examples = {
        1: example_basic_enhancement,
        2: example_custom_configuration,
        3: example_material_specific,
        4: example_batch_processing,
        5: example_integration_with_existing_pipeline,
        6: example_selective_enhancement
    }

    if args.all:
        for i in range(1, 7):
            try:
                examples[i]()
            except Exception as e:
                print(f"\n❌ Example {i} failed: {e}")
    elif args.example:
        examples[args.example]()
    else:
        print("\nAvailable examples:")
        print("  1. Basic enhancement (default 105/100)")
        print("  2. Custom configuration")
        print("  3. Material-specific enhancement")
        print("  4. Batch processing")
        print("  5. Integration with existing pipelines")
        print("  6. Selective enhancement stages")
        print("\nRun with: python hyper_reality_example.py --example <number>")
        print("Or run all: python hyper_reality_example.py --all")
