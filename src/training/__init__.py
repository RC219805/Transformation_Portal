"""
Property-Specific Training Module for Transformation Portal.

This module provides comprehensive training infrastructure for property-specific
image enhancement models, integrating architectural data analysis, material-aware
optimization, and depth intelligence.

Key Components:
- PropertyAnalyzer: Architectural feature and material detection
- DepthSynthesis: High-quality depth map generation
- DatasetGenerator: Multi-scale augmented dataset creation
- PropertyTrainer: Multi-stage training pipeline
- PropertyInference: Production 4K TIFF processing

Usage:
    from training.property_specific import (
        PicachoAnalyzer,
        DepthSynthesis,
        DatasetGenerator,
        PicachoTrainer,
        PicachoInference
    )

    # Initialize analyzer
    analyzer = PicachoAnalyzer(property_dir="projects/750_picacho_lane")

    # Analyze property
    report = analyzer.analyze_property()

    # Generate depth maps
    depth_synth = DepthSynthesis()
    depth_maps = depth_synth.synthesize_all(analyzer.images)

    # Create training dataset
    generator = DatasetGenerator(analyzer, depth_synth)
    dataset = generator.generate_dataset(num_samples=600)

    # Train model
    trainer = PicachoTrainer(config_path="config/training/750_picacho_lane_protocol.yaml")
    trainer.train()

    # Production inference
    inference = PicachoInference(model_path="weights/750_picacho/best_model.pth")
    enhanced = inference.process("input.tiff", output_format="16bit_tiff")
"""

from .property_specific import (
    PicachoAnalyzer,
    DepthSynthesis,
    DatasetGenerator,
    PicachoTrainer,
    PicachoInference
)

__all__ = [
    "PicachoAnalyzer",
    "DepthSynthesis",
    "DatasetGenerator",
    "PicachoTrainer",
    "PicachoInference"
]

__version__ = "1.0.0"
