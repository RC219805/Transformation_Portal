"""Example: Reference view selection strategies.

Demonstrates all four reference view selection strategies with practical
use cases for multi-view depth estimation.
"""

from pathlib import Path
import numpy as np
from lux_depth_v3.config import DA3Config, DA3APIConfig, DA3CLIConfig
from lux_depth_v3.reference_view import (
    select_reference_view,
    RefViewStrategy,
    ReferenceViewSelector,
)


def example_1_default_strategy():
    """Example 1: Default strategy (saddle_balanced)."""
    print("=" * 80)
    print("Example 1: Default strategy (saddle_balanced)")
    print("=" * 80)

    # Create config with API settings
    api_config = DA3APIConfig(ref_view_strategy="saddle_balanced")
    cli_config = DA3CLIConfig(use_cli=False)
    config = DA3Config(api=api_config, cli=cli_config)

    print(f"Configuration created with ref_view_strategy: {api_config.ref_view_strategy}")
    print("Note: For actual inference, use DA3InferenceEngine with proper setup")
    print("Output would be saved to: output/default/")
    print()


def example_2_video_sequence():
    """Example 2: Video sequence (middle strategy)."""
    print("=" * 80)
    print("Example 2: Video sequence (middle strategy)")
    print("=" * 80)

    api_config = DA3APIConfig(ref_view_strategy="middle")
    cli_config = DA3CLIConfig(use_cli=False)
    config = DA3Config(api=api_config, cli=cli_config)

    num_frames = 30
    middle_index = num_frames // 2

    print(f"Configuration created with ref_view_strategy: {api_config.ref_view_strategy}")
    print(f"For {num_frames} video frames, middle frame index: {middle_index}")
    print("Note: For actual inference, use DA3InferenceEngine with proper setup")
    print("Output would be saved to: output/video/")
    print()


def example_3_wide_baseline():
    """Example 3: Wide baseline multi-view."""
    print("=" * 80)
    print("Example 3: Wide baseline (saddle_sim_range strategy)")
    print("=" * 80)

    api_config = DA3APIConfig(ref_view_strategy="saddle_sim_range")
    cli_config = DA3CLIConfig(use_cli=False)
    config = DA3Config(api=api_config, cli=cli_config)

    print(f"Configuration created with ref_view_strategy: {api_config.ref_view_strategy}")
    print("Note: For actual inference, use DA3InferenceEngine with proper setup")
    print("Output would be saved to: output/wide/")
    print()


def example_4_manual_selection():
    """Example 4: Manual selection with class tokens."""
    print("=" * 80)
    print("Example 4: Manual selection with class tokens")
    print("=" * 80)

    # Simulate class tokens (in practice, these come from the model)
    num_views = 5
    feature_dim = 768
    np.random.seed(42)  # For reproducibility
    class_tokens = np.random.randn(num_views, feature_dim)

    selector = ReferenceViewSelector(strategy=RefViewStrategy.SADDLE_BALANCED)
    selection_result = selector.select(num_views, class_tokens)

    print(f"Selected view: {selection_result.selected_index}")
    print(f"Strategy: {selection_result.strategy.value}")
    if selection_result.metrics:
        print(f"Available metrics: {list(selection_result.metrics.keys())}")
        print(f"\nSimilarity scores: {selection_result.metrics['similarity_scores']}")
        print(f"Feature norms: {selection_result.metrics['feature_norms']}")
        print(f"Feature variances: {selection_result.metrics['feature_variances']}")
    print()


def example_5_strategy_comparison():
    """Example 5: Compare all strategies."""
    print("=" * 80)
    print("Example 5: Compare all strategies")
    print("=" * 80)

    num_views = 7
    feature_dim = 768
    np.random.seed(42)
    class_tokens = np.random.randn(num_views, feature_dim)

    strategies = [
        RefViewStrategy.SADDLE_BALANCED,
        RefViewStrategy.SADDLE_SIM_RANGE,
        RefViewStrategy.MIDDLE,
        RefViewStrategy.FIRST,
    ]

    print(f"Comparing strategies for {num_views} views:\n")

    for strategy in strategies:
        if strategy in [
            RefViewStrategy.SADDLE_BALANCED,
            RefViewStrategy.SADDLE_SIM_RANGE,
        ]:
            selector = ReferenceViewSelector(strategy=strategy)
            result = selector.select(num_views, class_tokens)
        else:
            selector = ReferenceViewSelector(strategy=strategy)
            result = selector.select(num_views)

        print(f"{strategy.value:20s} -> view {result.selected_index}")
        if result.scores is not None and len(result.scores) > 0:
            print(f"{'':20s}    scores: {result.scores[:3].tolist()}")
    print()


def example_6_convenience_function():
    """Example 6: Using convenience function."""
    print("=" * 80)
    print("Example 6: Using convenience function")
    print("=" * 80)

    num_views = 8
    np.random.seed(42)
    class_tokens = np.random.randn(num_views, 768)

    # Test different strategies with convenience function
    print("Using select_reference_view() convenience function:\n")

    # Saddle balanced
    result = select_reference_view(num_views=num_views, strategy="saddle_balanced", class_tokens=class_tokens)
    print(f"saddle_balanced: view {result.selected_index}")

    # Saddle sim range
    result = select_reference_view(num_views=num_views, strategy="saddle_sim_range", class_tokens=class_tokens)
    print(f"saddle_sim_range: view {result.selected_index}")

    # Middle (no class tokens needed)
    result = select_reference_view(num_views=num_views, strategy="middle")
    print(f"middle: view {result.selected_index}")

    # First (no class tokens needed)
    result = select_reference_view(num_views=num_views, strategy="first")
    print(f"first: view {result.selected_index}")
    print()


def example_7_few_views():
    """Example 7: Behavior with few views (<3)."""
    print("=" * 80)
    print("Example 7: Behavior with few views (<3)")
    print("=" * 80)

    selector = ReferenceViewSelector(strategy=RefViewStrategy.SADDLE_BALANCED)

    # Test with 1 view
    print("Testing with 1 view:")
    result = selector.select(num_views=1)
    print(f"  Selected index: {result.selected_index}")
    print(f"  Reason: {result.metrics['reason']}")

    # Test with 2 views
    print("\nTesting with 2 views:")
    result = selector.select(num_views=2)
    print(f"  Selected index: {result.selected_index}")
    print(f"  Reason: {result.metrics['reason']}")

    print("\nNote: Reference view selection is only applied for ≥3 views")
    print()


def example_8_detailed_metrics():
    """Example 8: Accessing detailed selection metrics."""
    print("=" * 80)
    print("Example 8: Accessing detailed selection metrics")
    print("=" * 80)

    num_views = 5
    np.random.seed(42)
    class_tokens = np.random.randn(num_views, 768)

    selector = ReferenceViewSelector(strategy=RefViewStrategy.SADDLE_BALANCED)
    result = selector.select(num_views, class_tokens)

    print(f"Selected view: {result.selected_index}")
    print(f"Strategy: {result.strategy.value}")
    print(f"\nPer-view distances to median (lower is better):")
    for i, score in enumerate(result.scores):
        marker = " <-- SELECTED" if i == result.selected_index else ""
        print(f"  View {i}: {score:.4f}{marker}")

    print(f"\nNormalized metrics (all in [0, 1] range):")
    print(f"  Similarity: {[f'{x:.3f}' for x in result.metrics['normalized_similarity']]}")
    print(f"  Norms:      {[f'{x:.3f}' for x in result.metrics['normalized_norms']]}")
    print(f"  Variances:  {[f'{x:.3f}' for x in result.metrics['normalized_variances']]}")
    print()


def example_9_production_pipeline():
    """Example 9: Production pipeline with conditional strategy."""
    print("=" * 80)
    print("Example 9: Production pipeline with conditional strategy")
    print("=" * 80)

    def select_strategy_for_dataset(image_dir: Path) -> str:
        """Auto-select strategy based on dataset characteristics."""
        images = list(image_dir.glob("*.jpg"))
        num_views = len(images)

        # Check if filename pattern suggests video frames
        is_video = any("frame" in img.name.lower() for img in images)

        if num_views < 3:
            return "first"
        elif is_video:
            return "middle"
        elif num_views > 10:
            return "saddle_balanced"
        else:
            return "saddle_sim_range"

    # Simulate different datasets
    datasets = [
        ("data/multi_view", "Unordered collection"),
        ("data/video_frames", "Video sequence"),
        ("data/aerial", "Wide baseline"),
    ]

    print("Automatic strategy selection for different datasets:\n")
    for data_dir, description in datasets:
        data_path = Path(data_dir)
        if data_path.exists():
            strategy = select_strategy_for_dataset(data_path)
            num_images = len(list(data_path.glob("*.jpg")))
            print(f"{description:25s} ({num_images:2d} views) -> {strategy}")
        else:
            print(f"{description:25s} -> {data_dir} not found (demo only)")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "Reference View Selection Examples".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    # Run all examples
    example_1_default_strategy()
    example_2_video_sequence()
    example_3_wide_baseline()
    example_4_manual_selection()
    example_5_strategy_comparison()
    example_6_convenience_function()
    example_7_few_views()
    example_8_detailed_metrics()
    example_9_production_pipeline()

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print("\nNote: Examples 1-3 require actual image data in specified directories.")
    print("Examples 4-9 use synthetic data for demonstration purposes.\n")


if __name__ == "__main__":
    main()
