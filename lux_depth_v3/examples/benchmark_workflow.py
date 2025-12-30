"""
Example: Running DA3 benchmark evaluation.

This example demonstrates how to run the official Depth Anything 3
Visual Geometry Benchmark for comprehensive model validation.
"""

from pathlib import Path
from lux_depth_v3.benchmark import (
    DA3BenchmarkEvaluator,
    BenchmarkConfig,
    EvaluationMode,
    download_datasets,
)
from lux_depth_v3 import ModelVariant


def main():
    """Run benchmark evaluation workflow."""

    # Step 1: Download benchmark datasets (one-time setup)
    print("Step 1: Downloading benchmark datasets...")
    data_root = Path("workspace/benchmark_dataset")

    # Download a small dataset first for testing
    download_datasets(["hiroom"], data_root)

    # Step 2: Configure benchmark
    print("\nStep 2: Configuring benchmark...")
    config = BenchmarkConfig(
        datasets=["hiroom"],  # Start with smallest dataset
        modes=[
            EvaluationMode.POSE,  # Pose estimation
            EvaluationMode.RECON_UNPOSED,  # Reconstruction with predicted poses
        ],
        max_frames=50,  # Limit frames for faster evaluation
        data_root=data_root,
        work_dir=Path("workspace/evaluation"),
    )

    # Step 3: Initialize evaluator
    print("\nStep 3: Initializing evaluator...")
    evaluator = DA3BenchmarkEvaluator(
        model_variant=ModelVariant.DA3_METRIC_LARGE,
        config=config,
        use_cli=False,  # Use Python API for inference
    )

    # Step 4: Run evaluation
    print("\nStep 4: Running benchmark evaluation...")
    print("This will:")
    print("  - Run depth inference on all scenes")
    print("  - Evaluate pose estimation accuracy (AUC@3°, AUC@30°)")
    print("  - Evaluate 3D reconstruction quality (F-score, Chamfer)")

    results = evaluator.run_full_evaluation()

    # Step 5: Print and save results
    print("\nStep 5: Results")
    evaluator.print_results(results)

    output_path = Path("workspace/evaluation/benchmark_results.json")
    evaluator.save_results(results, output_path)
    print(f"\n✅ Results saved to {output_path}")

    # Step 6: Load and print results later
    print("\nStep 6: Loading saved results...")
    loaded_results = evaluator.load_results(output_path)
    evaluator.print_results(loaded_results)


def run_full_benchmark():
    """Run full benchmark on all datasets (production use)."""

    # Download all datasets
    data_root = Path("workspace/benchmark_dataset")
    download_datasets(["all"], data_root)

    # Configure full benchmark
    config = BenchmarkConfig(
        datasets=["eth3d", "7scenes", "scannetpp", "hiroom", "dtu", "dtu64"],
        modes=[
            EvaluationMode.POSE,
            EvaluationMode.RECON_UNPOSED,
            EvaluationMode.RECON_POSED,
        ],
        max_frames=-1,  # Use all frames
        data_root=data_root,
        work_dir=Path("workspace/evaluation"),
    )

    # Run evaluation with DA3-GIANT (best accuracy)
    evaluator = DA3BenchmarkEvaluator(model_variant=ModelVariant.DA3_GIANT, config=config, use_cli=False)

    results = evaluator.run_full_evaluation()
    evaluator.print_results(results)
    evaluator.save_results(results)


def compare_models():
    """Compare different model variants on benchmark."""

    data_root = Path("workspace/benchmark_dataset")

    # Test configuration
    config = BenchmarkConfig(
        datasets=["hiroom"],
        modes=[EvaluationMode.POSE, EvaluationMode.RECON_POSED],
        max_frames=50,
        data_root=data_root,
    )

    models = [
        ModelVariant.DA3_BASE,
        ModelVariant.DA3_METRIC_LARGE,
        ModelVariant.DA3_GIANT,
    ]

    all_results = {}

    for model in models:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {model.value}")
        print(f"{'=' * 60}\n")

        config.work_dir = Path(f"workspace/evaluation_{model.value}")

        evaluator = DA3BenchmarkEvaluator(model_variant=model, config=config, use_cli=False)

        results = evaluator.run_full_evaluation()
        all_results[model.value] = results

    # Print comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60 + "\n")

    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        # Extract key metrics
        for dataset_name, dataset_results in results.items():
            for scene_name, scene_results in dataset_results.items():
                if "pose" in scene_results:
                    auc3 = scene_results["pose"]["auc3"]
                    auc30 = scene_results["pose"]["auc30"]
                    print(f"  {scene_name} - AUC@3°: {auc3:.3f}, AUC@30°: {auc30:.3f}")


def benchmark_cli_workflow():
    """Example using CLI commands."""

    print("""
    # Download datasets
    lux-depth-v3 benchmark-download --dataset hiroom --data-root workspace/benchmark_dataset

    # Run benchmark evaluation
    lux-depth-v3 benchmark \\
        --dataset hiroom \\
        --mode pose \\
        --mode recon_posed \\
        --max-frames 50 \\
        --data-root workspace/benchmark_dataset \\
        --work-dir workspace/evaluation \\
        --model-variant da3-metric-large

    # Print saved results
    lux-depth-v3 benchmark \\
        --print-only \\
        --work-dir workspace/evaluation

    # Run evaluation on existing predictions
    lux-depth-v3 benchmark \\
        --eval-only \\
        --dataset hiroom \\
        --work-dir workspace/evaluation
    """)


if __name__ == "__main__":
    # Run simple example
    main()

    # Uncomment to run full benchmark (takes hours)
    # run_full_benchmark()

    # Uncomment to compare models
    # compare_models()

    # Print CLI examples
    # benchmark_cli_workflow()
