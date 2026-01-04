"""Command-line interface for Depth Anything 3 integration.

Provides batch processing interface for monocular and multi-view depth estimation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import sys

import typer
from tqdm import tqdm

from lux_depth_v3.config import (
    DA3Config,
    ModelVariant,
    InferenceMode,
    Preset,
    ExportFormat,
    PostprocessingConfig,
    RefinementConfig,
)
from lux_depth_v3.input_manager import InputManager
from lux_depth_v3.inference import DA3InferenceEngine
from lux_depth_v3.postprocessing import Postprocessor
from lux_depth_v3.validation import DepthValidator, ValidationReport
from lux_depth_v3.export import Exporter
from lux_depth_v3.edge_refinement import create_refinement_preset

app = typer.Typer(
    name="lux-depth-v3",
    help="Depth Anything 3 (DA3) integration for Transformation Portal",
    add_completion=False,
)


# Model variant string mapping for CLI
# Maps user-friendly string names to ModelVariant enum members
MODEL_VARIANT_MAP = {
    # Nested models (v1.1 - recommended)
    "nested-giant-large-v1.1": ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1,
    "da3-nested-giant-large-v1.1": ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1,
    # Nested models (v1.0 - deprecated)
    "nested-giant-large": ModelVariant.DA3_NESTED_GIANT_LARGE,
    "da3-nested-giant-large": ModelVariant.DA3_NESTED_GIANT_LARGE,
    # Any-view models (v1.1)
    "giant-v1.1": ModelVariant.DA3_GIANT_V1_1,
    "da3-giant-v1.1": ModelVariant.DA3_GIANT_V1_1,
    "large-v1.1": ModelVariant.DA3_LARGE_V1_1,
    "da3-large-v1.1": ModelVariant.DA3_LARGE_V1_1,
    # Any-view models (v1.0 - deprecated)
    "giant": ModelVariant.DA3_GIANT,
    "da3-giant": ModelVariant.DA3_GIANT,
    "large": ModelVariant.DA3_LARGE,
    "da3-large": ModelVariant.DA3_LARGE,
    # Base/Small (Apache 2.0)
    "base": ModelVariant.DA3_BASE,
    "da3-base": ModelVariant.DA3_BASE,
    "small": ModelVariant.DA3_SMALL,
    "da3-small": ModelVariant.DA3_SMALL,
    # Metric/Mono models (Apache 2.0)
    "metric-large": ModelVariant.DA3_METRIC_LARGE,
    "da3-metric-large": ModelVariant.DA3_METRIC_LARGE,
    "mono-large": ModelVariant.DA3_MONO_LARGE,
    "da3-mono-large": ModelVariant.DA3_MONO_LARGE,
    # Legacy uppercase variants
    "NESTED_GIANT_LARGE": ModelVariant.NESTED_GIANT_LARGE,
    "GIANT": ModelVariant.GIANT,
    "LARGE": ModelVariant.LARGE,
    "BASE": ModelVariant.BASE,
    "SMALL": ModelVariant.SMALL,
    "METRIC_LARGE": ModelVariant.METRIC_LARGE,
    "MONO_LARGE": ModelVariant.MONO_LARGE,
}


def parse_model_variant(model_str: str) -> ModelVariant:
    """Convert model string to ModelVariant enum.
    
    Args:
        model_str: Model name string (case-insensitive)
        
    Returns:
        ModelVariant enum member
        
    Raises:
        typer.BadParameter: If model name is not recognized
    """
    model_lower = model_str.lower()
    if model_lower in MODEL_VARIANT_MAP:
        return MODEL_VARIANT_MAP[model_lower]
    
    # Also try uppercase for backward compatibility
    if model_str in MODEL_VARIANT_MAP:
        return MODEL_VARIANT_MAP[model_str]
    
    # Provide helpful error message
    valid_names = sorted(set(MODEL_VARIANT_MAP.keys()))
    typer.echo(f"ERROR: Unknown model variant: {model_str}")
    typer.echo(f"\nValid options:")
    typer.echo("  Commercial (Apache 2.0):")
    typer.echo("    - metric-large (recommended for interior scenes)")
    typer.echo("    - mono-large")
    typer.echo("    - base")
    typer.echo("    - small")
    typer.echo("  Non-commercial (CC-BY-NC-4.0, requires --non-commercial-ok):")
    typer.echo("    - nested-giant-large-v1.1 (recommended, best quality)")
    typer.echo("    - giant-v1.1")
    typer.echo("    - large-v1.1")
    raise typer.Exit(1)


@app.command()
def process(
    # Input/Output
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        "-i",
        help="Input directory with images",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        Path("output"),
        "--output-dir",
        "-o",
        help="Output directory for results",
    ),
    # Model configuration
    model: str = typer.Option(
        "metric-large",
        "--model",
        "-m",
        help="DA3 model variant to use (e.g., 'metric-large', 'nested-giant-large-v1.1')",
    ),
    preset: Optional[Preset] = typer.Option(
        None,
        "--preset",
        "-p",
        help="Use preset configuration",
    ),
    # Inference mode
    multi_view: bool = typer.Option(
        False,
        "--multi-view",
        help="Enable multi-view depth estimation",
    ),
    # CLI integration
    use_cli: bool = typer.Option(
        False,
        "--use-cli",
        help="Use official DA3 CLI instead of Python API",
    ),
    use_backend: bool = typer.Option(
        False,
        "--use-backend",
        help="Connect to DA3 backend service (requires --use-cli)",
    ),
    backend_url: str = typer.Option(
        "http://localhost:8008",
        "--backend-url",
        help="Backend service URL",
    ),
    # Processing options
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device to use (auto, cuda, mps, cpu)",
    ),
    precision: str = typer.Option(
        "fp16",
        "--precision",
        help="Model precision (fp32, fp16, bf16)",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Batch size for processing",
    ),
    # Postprocessing
    metric_scaling: bool = typer.Option(
        False,
        "--metric-scaling",
        help="Apply metric scaling to depth",
    ),
    bilateral_filter: bool = typer.Option(
        False,
        "--bilateral-filter",
        help="Apply bilateral filtering",
    ),
    # Edge-aware refinement (new)
    enable_refinement: bool = typer.Option(
        False,
        "--enable-refinement",
        help="Enable edge-aware refinement post-processing",
    ),
    refinement_preset: str = typer.Option(
        "balanced",
        "--refinement-preset",
        help="Refinement preset (balanced, aggressive, conservative, edge_focused)",
    ),
    refinement_stages: Optional[str] = typer.Option(
        None,
        "--refinement-stages",
        help="Comma-separated refinement stages (guided,bilateral,edge,gradient)",
    ),
    # Metric depth conversion
    convert_to_metric: bool = typer.Option(
        False,
        "--metric",
        help="Convert depth to metric depth in meters",
    ),
    focal_length: Optional[float] = typer.Option(
        None,
        "--focal-length",
        help="Focal length in pixels (for metric conversion)",
    ),
    fov: Optional[float] = typer.Option(
        None,
        "--fov",
        help="Horizontal field of view in degrees (for metric estimation)",
    ),
    show_depth_stats: bool = typer.Option(
        False,
        "--depth-stats",
        help="Show depth statistics in meters",
    ),
    # Export options
    export_format: List[ExportFormat] = typer.Option(
        [ExportFormat.PNG],
        "--export-format",
        "-f",
        help="Export formats (can specify multiple)",
    ),
    # Validation
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Enable validation with ground truth",
    ),
    ground_truth_dir: Optional[Path] = typer.Option(
        None,
        "--ground-truth-dir",
        help="Directory with ground truth depth maps",
    ),
    # Pattern matching
    pattern: str = typer.Option(
        "*.jpg",
        "--pattern",
        help="File pattern for input images",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Search input directory recursively",
    ),
    # Verbose output
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    """Process images with Depth Anything 3.

    Examples:

      # Monocular metric depth (native mode)
      lux-depth-v3 process -i renders/ -o output/ --model metric-large

      # Using official CLI
      lux-depth-v3 process -i renders/ -o output/ --use-cli

      # Using backend service for batch processing
      lux-depth-v3 process -i renders/ -o output/ --use-cli --use-backend

      # Multi-view with pose estimation
      lux-depth-v3 process -i views/ -o 3d/ --multi-view --model nested-giant-large

      # Using preset
      lux-depth-v3 process -i images/ --preset interior_luxury

      # Export multiple formats
      lux-depth-v3 process -i renders/ -f png -f npz -f ply
    """
    # Validate CLI options
    if use_backend and not use_cli:
        print("ERROR: --use-backend requires --use-cli")
        sys.exit(1)

    # Create configuration
    model_variant = parse_model_variant(model)
    if preset is not None:
        config = DA3Config.from_preset(preset)
        # Override model if explicitly provided (only when different from default)
        if model != "metric-large":  # Only override if user specified non-default
            config.model_variant = model_variant
    else:
        config = DA3Config(
            model_variant=model_variant,
            inference_mode=InferenceMode.MULTI_VIEW if multi_view else InferenceMode.MONOCULAR,
        )

    # Override configuration with CLI options
    config.cli.use_cli = use_cli
    config.cli.use_backend = use_backend
    config.cli.backend_url = backend_url
    config.device.device = device
    config.device.precision = precision
    config.batch_size = batch_size
    config.postprocessing.apply_metric_scaling = metric_scaling
    config.postprocessing.apply_bilateral_filter = bilateral_filter
    config.export.formats = export_format
    config.export.output_dir = output_dir
    config.validation.enable_validation = validate
    config.validation.ground_truth_path = ground_truth_dir

    # Configure edge-aware refinement
    if enable_refinement:
        # Use preset or create custom config
        refinement_config = create_refinement_preset(refinement_preset)

        # Override stages if specified
        if refinement_stages is not None:
            stages = [s.strip() for s in refinement_stages.split(",")]
            refinement_config.stages = stages

        config.postprocessing.refinement = refinement_config

    if verbose:
        print("Configuration:")
        print(f"  Model: {config.model_variant.value}")
        print(f"  Mode: {config.inference_mode.value}")
        print(f"  Device: {config.device.device}")
        print(f"  Precision: {config.device.precision}")
        print(f"  Output: {output_dir}")
        if enable_refinement:
            print(f"  Refinement: {refinement_preset} ({', '.join(config.postprocessing.refinement.stages)})")

    # Initialize pipeline components
    input_manager = InputManager(inference_mode=config.inference_mode)
    inference_engine = DA3InferenceEngine(config)
    postprocessor = Postprocessor(config.postprocessing)
    exporter = Exporter(config.export)

    # Load inputs
    print(f"Loading images from: {input_dir}")
    num_images = input_manager.add_directory(
        input_dir,
        pattern=pattern,
        recursive=recursive,
    )

    if num_images == 0:
        print(f"No images found matching pattern: {pattern}")
        sys.exit(1)

    print(f"Found {num_images} images")

    # Validate inputs
    try:
        input_manager.validate_inputs()
    except ValueError as e:
        print(f"Input validation failed: {e}")
        sys.exit(1)

    # Load model
    print("Loading DA3 model...")
    inference_engine.load_model()

    # Initialize validation
    validator = None
    validation_report = None
    if validate:
        validator = DepthValidator(ground_truth_dir=ground_truth_dir)
        validation_report = ValidationReport()

    # Process images
    print("Processing images...")
    inputs = input_manager.get_images()

    for img_input in tqdm(inputs, desc="Depth estimation"):
        # Run inference
        result = inference_engine.inference(img_input)

        # Apply postprocessing
        result = postprocessor.process(result)

        # Validate if enabled
        if validator is not None:
            metrics = validator.validate(result)
            validation_report.add_result(metrics)

            if verbose and metrics.metadata.get("has_ground_truth"):
                print(f"\n  Metrics: RMSE={metrics.rmse:.4f}, δ1={metrics.delta_1:.3f}")

        # Export results
        if img_input.path is not None:
            filename_base = img_input.path.stem
        else:
            filename_base = f"depth_{len(inputs)}"

        exported = exporter.export(result, filename_base)

        if verbose:
            for fmt, path in exported.items():
                print(f"  Exported {fmt}: {path}")

    # Save validation report
    if validation_report is not None:
        summary = validation_report.compute_summary()
        report_path = output_dir / "validation_report.json"
        validation_report.save(report_path)

        print("\nValidation Summary:")
        print(f"  Images: {summary['num_images']}")
        print(f"  Mean RMSE: {summary['mean_rmse']:.4f}")
        print(f"  Mean δ1: {summary['mean_delta_1']:.3f}")
        print(f"  Report: {report_path}")

    print(f"\nProcessing complete! Results saved to: {output_dir}")


@app.command()
def benchmark(
    model: ModelVariant = typer.Option(
        ModelVariant.METRIC_LARGE,
        "--model",
        help="Model variant to benchmark",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device to use",
    ),
    num_iterations: int = typer.Option(
        100,
        "--iterations",
        "-n",
        help="Number of iterations",
    ),
):
    """Benchmark DA3 model performance.

    Example:
      lux-depth-v3 benchmark --model metric-large --device cuda -n 100
    """
    import time
    import numpy as np

    print(f"Benchmarking {model.value} on {device}")

    # Create dummy input
    dummy_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    # Initialize engine
    config = DA3Config(model_variant=model)
    config.device.device = device
    engine = DA3InferenceEngine(config)
    engine.load_model()

    # Warm-up
    from lux_depth_v3.input_manager import ImageInput

    img_input = ImageInput(array=dummy_image)
    print("Warming up...")
    for _ in range(10):
        _ = engine.inference(img_input)

    # Benchmark
    print(f"Running {num_iterations} iterations...")
    times = []
    for _ in tqdm(range(num_iterations)):
        start = time.perf_counter()
        _ = engine.inference(img_input)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    # Report
    print("\nBenchmark Results:")
    print(f"  Mean: {np.mean(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min: {np.min(times):.2f} ms")
    print(f"  Max: {np.max(times):.2f} ms")
    print(f"  Std: {np.std(times):.2f} ms")
    print(f"  Throughput: {1000 / np.mean(times):.1f} images/second")


@app.command()
def backend_start(
    model_dir: str = typer.Option(
        ...,
        "--model-dir",
        help="Path to DA3 model directory",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Device to use (cuda, mps, cpu)",
    ),
    port: int = typer.Option(
        8008,
        "--port",
        help="Port for backend service",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host address for backend",
    ),
):
    """Start DA3 backend service.

    The backend service keeps the model loaded in GPU memory, providing
    10-20x speedup for batch processing by avoiding model reload overhead.

    Example:
      lux-depth-v3 backend-start --model-dir ~/.cache/lux_depth_v3/models/depth-anything-3-metric-large
    """
    from lux_depth_v3.da3_wrapper import DA3Backend, check_da3_cli_available

    if not check_da3_cli_available():
        print("ERROR: DA3 CLI not found. Install from:")
        print("  https://github.com/DepthAnything/Depth-Anything-V3")
        sys.exit(1)

    backend = DA3Backend(
        model_dir=model_dir,
        device=device,
        port=port,
        host=host,
    )

    try:
        backend.start()
        print(f"Backend running at {backend.get_url()}")
        print("Press Ctrl+C to stop...")

        # Keep running until interrupted
        import signal

        signal.pause()
    except KeyboardInterrupt:
        print("\nStopping backend...")
        backend.stop()


@app.command()
def backend_stop(
    port: int = typer.Option(
        8008,
        "--port",
        help="Port of backend service to stop",
    ),
):
    """Stop DA3 backend service.

    Example:
      lux-depth-v3 backend-stop --port 8008
    """
    import requests

    url = f"http://localhost:{port}"

    try:
        # Try graceful shutdown via API
        response = requests.post(f"{url}/shutdown", timeout=5)
        if response.status_code == 200:
            print(f"Backend at {url} stopped")
        else:
            print(f"Failed to stop backend: {response.status_code}")
    except requests.RequestException:
        print(f"Could not connect to backend at {url}")
        print("Backend may not be running or is inaccessible")


@app.command()
def backend_status(
    port: int = typer.Option(
        8008,
        "--port",
        help="Port of backend service",
    ),
):
    """Check DA3 backend status.

    Example:
      lux-depth-v3 backend-status --port 8008
    """
    from lux_depth_v3.da3_wrapper import DA3Backend

    backend = DA3Backend(
        model_dir="",  # Not needed for status check
        port=port,
    )

    if backend.is_running():
        print(f"✓ Backend is running at {backend.get_url()}")
    else:
        print(f"✗ Backend is not running at {backend.get_url()}")


@app.command()
def benchmark(
    datasets: List[str] = typer.Option(
        ["eth3d", "7scenes", "scannetpp", "hiroom", "dtu", "dtu64"], "--dataset", "-d", help="Datasets to evaluate"
    ),
    modes: List[str] = typer.Option(
        ["pose", "recon_unposed", "recon_posed"], "--mode", "-m", help="Evaluation modes (pose, recon_unposed, recon_posed)"
    ),
    data_root: Path = typer.Option(Path("workspace/benchmark_dataset"), help="Benchmark data root directory"),
    work_dir: Path = typer.Option(Path("workspace/evaluation"), help="Working directory for outputs"),
    max_frames: int = typer.Option(100, help="Max frames per scene (-1 for all)"),
    eval_only: bool = typer.Option(False, help="Only run evaluation, skip inference"),
    print_only: bool = typer.Option(False, help="Only print saved results"),
    use_cli: bool = typer.Option(False, help="Use DA3 CLI for inference"),
    model_variant: str = typer.Option("da3-giant", help="Model variant (da3-giant, da3-metric-large, da3-base)"),
):
    """Run DA3 benchmark evaluation."""
    from lux_depth_v3.benchmark import (
        DA3BenchmarkEvaluator,
        BenchmarkConfig,
        EvaluationMode,
    )

    # Parse model variant
    variant_map = {
        "da3-giant": ModelVariant.DA3_GIANT,
        "da3-metric-large": ModelVariant.DA3_METRIC_LARGE,
        "da3-base": ModelVariant.DA3_BASE,
    }
    model = variant_map.get(model_variant.lower(), ModelVariant.DA3_GIANT)

    # Parse evaluation modes
    mode_map = {
        "pose": EvaluationMode.POSE,
        "recon_unposed": EvaluationMode.RECON_UNPOSED,
        "recon_posed": EvaluationMode.RECON_POSED,
    }
    eval_modes = [mode_map[m] for m in modes if m in mode_map]

    # Create config
    config = BenchmarkConfig(
        datasets=datasets, modes=eval_modes, max_frames=max_frames, data_root=data_root, work_dir=work_dir
    )

    # Initialize evaluator
    evaluator = DA3BenchmarkEvaluator(model_variant=model, config=config, use_cli=use_cli)

    if print_only:
        typer.echo("Loading saved results...")
        results = evaluator.load_results()
        evaluator.print_results(results)
    elif eval_only:
        typer.echo("Running evaluation on existing predictions...")
        # TODO: Implement evaluation-only mode
        typer.echo("Evaluation-only mode not yet implemented")
    else:
        typer.echo("Running full benchmark evaluation...")
        results = evaluator.run_full_evaluation()
        evaluator.print_results(results)
        evaluator.save_results(results)
        typer.echo(f"\n✅ Results saved to {config.work_dir}/benchmark_results.json")


@app.command()
def benchmark_download(
    datasets: List[str] = typer.Option(
        ["all"], "--dataset", "-d", help="Datasets to download (all, eth3d, 7scenes, scannetpp, hiroom, dtu, dtu64)"
    ),
    data_root: Path = typer.Option(Path("workspace/benchmark_dataset"), help="Download destination"),
):
    """Download DA3 benchmark datasets from HuggingFace."""
    from lux_depth_v3.benchmark import download_datasets

    typer.echo(f"Downloading datasets to {data_root}...")
    download_datasets(datasets, data_root)
    typer.echo(f"✅ Download complete")


def main():
    """Main entry point."""
    app()


@app.command()
def api_process(
    # Input/Output
    input_path: Path = typer.Argument(..., help="Input path (image or directory)"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory"),
    # Model selection with version support
    model_name: str = typer.Option(
        "nested-giant-large-v1.1",
        "--model",
        "-m",
        help=(
            "Model variant. Options:\n"
            "  nested-giant-large-v1.1 (recommended, 1.40B, NC)\n"
            "  nested-giant-large (deprecated, 1.40B, NC)\n"
            "  giant-v1.1 (1.15B, NC)\n"
            "  large-v1.1 (0.35B, NC)\n"
            "  base (0.12B, Apache)\n"
            "  small (0.08B, Apache)\n"
            "  metric-large (0.35B, Apache, commercial-friendly)\n"
            "  mono-large (0.35B, Apache)\n"
        ),
    ),
    # License validation
    commercial_use: bool = typer.Option(
        False, "--commercial", help="Declare this is commercial use (triggers license validation)"
    ),
    strict_license: bool = typer.Option(
        False, "--strict-license", help="Raise error instead of warning on license violations"
    ),
    show_license: bool = typer.Option(False, "--show-license", help="Show license information for selected model and exit"),
    # Export formats
    export_format: str = typer.Option(
        "mini_npz", "--export-format", "-f", help="Export format(s), separated by '-' (e.g., 'mini_npz-glb-gs_ply')"
    ),
    # Pose parameters
    use_ray_pose: bool = typer.Option(False, "--use-ray-pose", help="Use ray-based pose estimation"),
    ref_view_strategy: str = typer.Option(
        "saddle_balanced",
        "--ref-view-strategy",
        help="Reference view strategy (first/middle/saddle_balanced/saddle_sim_range)",
    ),
    align_to_input_ext_scale: bool = typer.Option(True, "--align-scale/--no-align-scale", help="Align to input scale"),
    # Gaussian Splatting
    infer_gs: bool = typer.Option(False, "--infer-gs", help="Enable Gaussian Splatting branch"),
    # Feature extraction
    export_feat_layers: str = typer.Option("", "--export-feat", help="Feature layers (comma-separated, e.g., '0,3,6,9')"),
    feat_vis_fps: int = typer.Option(15, "--feat-fps", help="Feature visualization FPS"),
    # GLB export
    conf_thresh_percentile: float = typer.Option(40.0, "--conf-thresh", help="GLB confidence threshold percentile (0-100)"),
    num_max_points: int = typer.Option(1_000_000, "--max-points", help="GLB max points"),
    show_cameras: bool = typer.Option(True, "--show-cameras/--no-cameras", help="GLB show cameras"),
    # Processing
    process_res: int = typer.Option(504, "--process-res", help="Processing resolution"),
    process_res_method: str = typer.Option("upper_bound_resize", "--resize-method", help="Resize method"),
    # Device
    device: str = typer.Option("cuda", "--device", help="Device (cuda/cpu/mps)"),
    # Metric depth conversion
    convert_to_metric: bool = typer.Option(False, "--metric", help="Convert depth to metric depth in meters"),
    focal_length: Optional[float] = typer.Option(
        None, "--focal-length", help="Focal length in pixels (for metric conversion)"
    ),
    fov: Optional[float] = typer.Option(None, "--fov", help="Horizontal field of view in degrees (for metric estimation)"),
    show_depth_stats: bool = typer.Option(False, "--depth-stats", help="Show depth statistics in meters"),
):
    """Process images with full DA3 Python API support.

    This command exposes all DA3 API features including:
    - Multi-view depth estimation with pose estimation
    - Gaussian Splatting (requires da3-giant or da3nested-giant-large)
    - Feature extraction from intermediate layers
    - Multiple export formats (NPZ, GLB, PLY, videos)
    - License validation for commercial use

    Examples:
        # Basic monocular depth with v1.1 model
        lux-depth-v3 api-process image.jpg -o output

        # Show license information
        lux-depth-v3 api-process image.jpg -o output --show-license

        # Multi-view with GLB export
        lux-depth-v3 api-process images/ -o output -f "mini_npz-glb"

        # Commercial use with Apache-licensed model
        lux-depth-v3 api-process images/ -o output -m metric-large --commercial

        # Gaussian Splatting workflow
        lux-depth-v3 api-process images/ -o output -m giant-v1.1 --infer-gs -f "gs_ply-gs_video"

        # Feature extraction
        lux-depth-v3 api-process images/ -o output --export-feat "0,3,6,9" -f "feat_vis"
    """
    from lux_depth_v3.config import DA3Config, DA3APIConfig, ModelVariant, DeviceConfig
    from lux_depth_v3.inference import DA3InferenceEngine
    from lux_depth_v3.license import LicenseValidator

    # Parse model variant
    model_map = {
        "nested-giant-large-v1.1": ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1,
        "nested-giant-large": ModelVariant.DA3_NESTED_GIANT_LARGE,
        "giant-v1.1": ModelVariant.DA3_GIANT_V1_1,
        "giant": ModelVariant.DA3_GIANT,
        "large-v1.1": ModelVariant.DA3_LARGE_V1_1,
        "large": ModelVariant.DA3_LARGE,
        "base": ModelVariant.DA3_BASE,
        "small": ModelVariant.DA3_SMALL,
        "metric-large": ModelVariant.DA3_METRIC_LARGE,
        "mono-large": ModelVariant.DA3_MONO_LARGE,
        # Legacy aliases
        "da3-giant": ModelVariant.DA3_GIANT,
        "da3-large": ModelVariant.DA3_LARGE,
        "da3-base": ModelVariant.DA3_BASE,
        "da3-small": ModelVariant.DA3_SMALL,
        "da3metric-large": ModelVariant.DA3_METRIC_LARGE,
        "da3mono-large": ModelVariant.DA3_MONO_LARGE,
        "da3nested-giant-large": ModelVariant.DA3_NESTED_GIANT_LARGE,
    }

    variant = model_map.get(model_name.lower())
    if variant is None:
        typer.echo(f"❌ Unknown model: {model_name}", err=True)
        typer.echo(f"Available models: {', '.join(model_map.keys())}")
        raise typer.Exit(1)

    # Show license info if requested
    if show_license:
        validator = LicenseValidator()
        info = validator.get_license_info(variant)

        typer.echo(f"\n📄 License Information")
        typer.echo(f"{'=' * 70}")
        typer.echo(f"Model: {info['model']}")
        typer.echo(f"License: {info['license']}")
        typer.echo(f"Commercial Use: {'✅ Allowed' if info['commercial_allowed'] else '❌ Not Allowed'}")
        typer.echo(f"License URL: {info['license_url']}")

        if info["alternative"]:
            typer.echo(f"\nCommercial Alternative: {info['alternative']}")

        if info["capabilities"]:
            typer.echo(f"\nCapabilities:")
            for cap, supported in info["capabilities"].items():
                status = "✅" if supported else "❌"
                typer.echo(f"  {status} {cap.replace('_', ' ').title()}")

        typer.echo(f"{'=' * 70}\n")
        raise typer.Exit(0)

    typer.echo(f"🚀 DA3 API Processing")
    typer.echo(f"   Model: {variant.info.display_name} ({variant.info.params})")
    typer.echo(f"   License: {variant.info.license.value}")
    if commercial_use and not variant.info.is_commercial:
        typer.echo(f"   ⚠️  Commercial use declared with NC-licensed model")
    typer.echo(f"   Input: {input_path}")
    typer.echo(f"   Output: {output_dir}")
    typer.echo(f"   Export: {export_format}")

    # Parse feature layers
    feat_layers = []
    if export_feat_layers:
        try:
            feat_layers = [int(x.strip()) for x in export_feat_layers.split(",")]
        except ValueError:
            typer.echo(f"❌ Invalid feature layers: {export_feat_layers}", err=True)
            sys.exit(1)

    # Create API config
    api_config = DA3APIConfig(
        model_name=model_name,
        align_to_input_ext_scale=align_to_input_ext_scale,
        infer_gs=infer_gs,
        use_ray_pose=use_ray_pose,
        ref_view_strategy=ref_view_strategy,
        process_res=process_res,
        process_res_method=process_res_method,
        export_format=export_format,
        export_feat_layers=feat_layers,
        conf_thresh_percentile=conf_thresh_percentile,
        num_max_points=num_max_points,
        show_cameras=show_cameras,
        feat_vis_fps=feat_vis_fps,
    )

    # Create main config
    config = DA3Config(
        model_variant=variant,
        api=api_config,
        device=DeviceConfig(device=device),
    )

    # Initialize engine with license validation
    try:
        engine = DA3InferenceEngine(config, commercial_use=commercial_use, validate_license_strict=strict_license)
    except Exception as e:
        typer.echo(f"❌ Failed to initialize engine: {e}", err=True)
        sys.exit(1)

    # Collect images
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))
        if not images:
            typer.echo(f"❌ No images found in {input_path}", err=True)
            sys.exit(1)
    else:
        typer.echo(f"❌ Invalid input path: {input_path}", err=True)
        sys.exit(1)

    typer.echo(f"   Found {len(images)} images")

    # Run inference
    try:
        typer.echo("⚙️  Running inference...")
        result = engine.infer(
            images=images,
            export_dir=output_dir,
            convert_to_metric=convert_to_metric,
            focal_length_px=focal_length,
            fov_degrees=fov,
        )

        typer.echo(f"✅ Processing complete")
        typer.echo(f"   Depth shape: {result.depth.shape}")
        if result.extrinsics is not None:
            typer.echo(f"   Estimated poses: {result.extrinsics.shape[0]}")
        if result.aux:
            typer.echo(f"   Auxiliary outputs: {list(result.aux.keys())}")
        typer.echo(f"   Results saved to: {output_dir}")

        # Show depth statistics if requested
        if show_depth_stats and hasattr(result, "metric_depth"):
            from lux_depth_v3.metric_depth import get_depth_statistics

            stats = get_depth_statistics(result.metric_depth)

            typer.echo("\n📏 Depth Statistics (meters)")
            typer.echo(f"{'=' * 50}")
            typer.echo(f"Min:    {stats['min_m']:.2f} m")
            typer.echo(f"Max:    {stats['max_m']:.2f} m")
            typer.echo(f"Mean:   {stats['mean_m']:.2f} m")
            typer.echo(f"Median: {stats['median_m']:.2f} m")
            typer.echo(f"Range:  {stats['range_m']:.2f} m")
            typer.echo(f"{'=' * 50}\n")

    except Exception as e:
        typer.echo(f"❌ Inference failed: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


@app.command()
def cache_download(
    model_set: str = typer.Option(
        "essential", "--set", "-s", help="Model set to download (essential/production/benchmark/all)"
    ),
    models: Optional[str] = typer.Option(None, "--models", "-m", help="Comma-separated model keys (overrides --set)"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Custom cache directory (default: HF cache)"),
    force: bool = typer.Option(False, "--force", help="Force re-download even if cached"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Verify downloads"),
):
    """Download and cache DA3 models."""
    from lux_depth_v3.model_cache import ModelCacheManager

    manager = ModelCacheManager(cache_dir=cache_dir)

    # Parse models if provided
    model_keys = models.split(",") if models else None

    typer.echo(f"📥 Downloading DA3 models...")
    typer.echo(f"   Set: {model_set if not model_keys else 'custom'}")
    typer.echo(f"   Cache: {manager.cache_dir}")

    results = manager.download_models(model_set=model_set, model_keys=model_keys, force=force, verify=verify)

    # Summary
    total_size = sum(r.size_bytes for r in results)

    typer.echo(f"\n✅ Downloaded {len(results)} models")
    typer.echo(f"   Total size: {total_size / (1024**3):.2f} GB")
    typer.echo(f"   Cache location: {manager.cache_dir}")


@app.command()
def cache_list():
    """List cached models."""
    from lux_depth_v3.model_cache import ModelCacheManager

    manager = ModelCacheManager()
    cached = manager.list_cached_models()

    if not cached:
        typer.echo("No models cached yet.")
        typer.echo(f"Run: lux-depth-v3 cache-download --set essential")
        return

    typer.echo(f"\n📦 Cached Models ({len(cached)})")
    typer.echo(f"{'=' * 70}")

    for model in cached:
        size_gb = model.size_bytes / (1024**3)
        verified = "✓" if model.verified else "?"
        typer.echo(f"{verified} {model.model_id}")
        typer.echo(f"   Size: {size_gb:.2f} GB | Cached: {model.cached_at}")

    stats = manager.get_cache_stats()
    typer.echo(f"\n{'=' * 70}")
    typer.echo(f"Total: {stats['total_size_gb']:.2f} GB in {stats['num_models']} models")
    typer.echo(f"Cache: {stats['cache_dir']}")


@app.command()
def cache_stats():
    """Show cache statistics."""
    from lux_depth_v3.model_cache import ModelCacheManager

    manager = ModelCacheManager()
    stats = manager.get_cache_stats()

    typer.echo(f"\n📊 Cache Statistics")
    typer.echo(f"{'=' * 70}")
    typer.echo(f"Location: {stats['cache_dir']}")
    typer.echo(f"Models: {stats['num_models']}")
    typer.echo(f"Total Size: {stats['total_size_gb']:.2f} GB")
    typer.echo(f"Last Updated: {stats.get('last_updated', 'Never')}")


@app.command()
def enhance(
    # Input/Output
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        "-i",
        help="Input directory with images",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        Path("output"),
        "--output-dir",
        "-o",
        help="Root output directory (will create depth/, v2/, manifests/, logs/ subdirs)",
    ),
    # Model configuration
    model: str = typer.Option(
        "metric-large",
        "--model",
        "-m",
        help="DA3 model variant for depth estimation (e.g., 'metric-large', 'nested-giant-large-v1.1')",
    ),
    preset: Optional[Preset] = typer.Option(
        None,
        "--preset",
        "-p",
        help="V3 preset configuration",
    ),
    # V2 configuration
    v2_preset: str = typer.Option(
        "production_ultra",
        "--v2-preset",
        help="V2 enhancement preset",
    ),
    v2_device: str = typer.Option(
        "auto",
        "--v2-device",
        help="Device for V2 enhancement (auto, cuda, cpu)",
    ),
    v2_upscaler: str = typer.Option(
        "torch",
        "--v2-upscaler",
        help="V2 upscaler backend (torch, onnx, none)",
    ),
    # Depth configuration
    depth_device: str = typer.Option(
        "auto",
        "--depth-device",
        help="Device for depth estimation (auto, cuda, mps, cpu)",
    ),
    depth_quantization: str = typer.Option(
        "p1p99",
        "--depth-quantization",
        help="Depth quantization method (p1p99, p0.5p99.5, minmax)",
    ),
    # Execution control
    execution_mode: str = typer.Option(
        "sequential",
        "--execution-mode",
        help="Execution mode (sequential or pipelined)",
    ),
    depth_fallback: str = typer.Option(
        "fail",
        "--depth-fallback",
        help="Depth failure policy (fail, skip, v2-auto)",
    ),
    force_depth: bool = typer.Option(
        False,
        "--force-depth",
        help="Force depth regeneration even if exists",
    ),
    force_v2: bool = typer.Option(
        False,
        "--force-v2",
        help="Force V2 re-enhancement even if outputs exist",
    ),
    # NEW: Filtering options
    include: Optional[str] = typer.Option(
        None,
        "--include",
        help="Comma-separated glob patterns to include (e.g., '*.jpg,*.png')",
    ),
    exclude: Optional[str] = typer.Option(
        None,
        "--exclude",
        help="Comma-separated glob patterns to exclude (e.g., '*_mask.png,*_depth.png')",
    ),
    max_images: Optional[int] = typer.Option(
        None,
        "--max-images",
        help="Maximum number of images to process (useful for testing)",
    ),
    # NEW: Execution modes
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print processing plan without executing (shows what would be processed)",
    ),
    # License
    non_commercial_ok: bool = typer.Option(
        False,
        "--non-commercial-ok",
        help="Acknowledge non-commercial use of DA3 (required for CC-BY-NC models)",
    ),
    # Timeout
    v2_timeout: Optional[float] = typer.Option(
        600.0,
        "--v2-timeout",
        help="Timeout for V2 enhancement in seconds (default: 600)",
    ),
    # Verbosity
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """Orchestrate V3 depth + V2 enhancement pipeline.

    This command runs a two-stage pipeline:
      1. Stage A (V3): Generate depth assets using Depth Anything 3
      2. Stage B (V2): Consume depth → weights → grade → upscale → export → report

    Output structure:
      <output_dir>/
        depth/         - DA3 depth maps ({stem}_depth.png, uint16)
        v2/           - V2 enhancement outputs (16-bit TIFFs, etc.)
        manifests/    - Combined manifests linking DA3 + V2 ({stem}_combined.json)
        logs/         - Processing logs

    Examples:
      # Basic usage (requires --non-commercial-ok for DA3)
      lux-depth-v3 enhance -i renders/ -o output/ --non-commercial-ok

      # Production quality with specific V2 preset
      lux-depth-v3 enhance -i renders/ -o output/ \\
          --v2-preset production_ultra --non-commercial-ok

      # Resume from previous run (skip existing outputs)
      lux-depth-v3 enhance -i renders/ -o output/ --non-commercial-ok

      # Force complete regeneration
      lux-depth-v3 enhance -i renders/ -o output/ \\
          --force-depth --force-v2 --non-commercial-ok
    """
    import logging
    from lux_depth_v3.enhance import EnhanceOrchestrator, EnhanceConfig

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate license acknowledgement
    if not non_commercial_ok:
        typer.echo("ERROR: DA3 models require license acknowledgement.")
        typer.echo("For non-commercial use, add: --non-commercial-ok")
        typer.echo("For commercial use, contact Depth Anything team for licensing.")
        raise typer.Exit(1)

    # Display configuration
    # Only parse model variant if user provided it (non-default value)
    model_variant = parse_model_variant(model) if model != "metric-large" or preset is None else None
    
    if verbose:
        typer.echo("\n🚀 V3 + V2 Enhancement Pipeline")
        typer.echo("=" * 70)
        typer.echo(f"Input: {input_dir}")
        typer.echo(f"Output: {output_dir}")
        typer.echo(f"\nStage A (V3 Depth):")
        if preset is not None:
            typer.echo(f"  Preset: {preset.value}")
            if model_variant is not None:
                typer.echo(f"  Model override: {model_variant.value.display_name}")
        else:
            display_variant = model_variant or parse_model_variant("metric-large")
            typer.echo(f"  Model: {display_variant.value.display_name}")
        typer.echo(f"  Device: {depth_device}")
        typer.echo(f"  Quantization: {depth_quantization}")
        typer.echo(f"\nStage B (V2 Enhancement):")
        typer.echo(f"  Preset: {v2_preset}")
        typer.echo(f"  Device: {v2_device}")
        typer.echo(f"  Upscaler: {v2_upscaler}")
        typer.echo(f"\nExecution:")
        typer.echo(f"  Mode: {execution_mode}")
        typer.echo(f"  Depth fallback: {depth_fallback}")
        typer.echo("=" * 70 + "\n")

    # Create configuration
    config = EnhanceConfig(
        model_variant=model_variant,
        preset=preset,
        v2_preset=v2_preset,
        v2_device=v2_device,
        v2_upscaler_backend=v2_upscaler,
        depth_device=depth_device,
        depth_quantization=depth_quantization,
        execution_mode=execution_mode,
        depth_fallback=depth_fallback,
        force_depth=force_depth,
        force_v2=force_v2,
        non_commercial_ok=non_commercial_ok,
        v2_timeout=v2_timeout,
    )

    # Initialize orchestrator
    typer.echo("Initializing orchestrator...")
    orchestrator = EnhanceOrchestrator(config, output_dir)

    # Collect images with filtering
    from fnmatch import fnmatch

    image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    all_images = []
    for ext in image_extensions:
        all_images.extend(input_dir.rglob(f"*{ext}"))
        all_images.extend(input_dir.rglob(f"*{ext.upper()}"))

    # Apply include/exclude filters
    filtered_images = []
    include_patterns = include.split(",") if include else None
    exclude_patterns = exclude.split(",") if exclude else None

    for img_path in sorted(all_images):
        relative_path = str(img_path.relative_to(input_dir))

        # Apply include filter (if specified, must match at least one pattern)
        if include_patterns:
            if not any(fnmatch(relative_path, pattern.strip()) for pattern in include_patterns):
                continue

        # Apply exclude filter (if specified, must not match any pattern)
        if exclude_patterns:
            if any(fnmatch(relative_path, pattern.strip()) for pattern in exclude_patterns):
                continue

        filtered_images.append(img_path)

    # Apply max_images limit
    if max_images is not None and len(filtered_images) > max_images:
        filtered_images = filtered_images[:max_images]
        typer.echo(f"⚠️  Limited to first {max_images} images (--max-images)")

    typer.echo(f"Found {len(all_images)} images total")
    if include or exclude or max_images:
        typer.echo(f"Filtered to {len(filtered_images)} images")

    # Dry run mode: print plan and exit
    if dry_run:
        typer.echo("\n🔍 DRY RUN MODE - Processing plan:")
        typer.echo("=" * 70)
        typer.echo(f"Would process {len(filtered_images)} images:")
        for img_path in filtered_images[:20]:  # Show first 20
            typer.echo(f"  - {img_path.relative_to(input_dir)}")
        if len(filtered_images) > 20:
            typer.echo(f"  ... and {len(filtered_images) - 20} more")
        typer.echo("=" * 70)
        typer.echo("\n✓ Dry run complete (no files were processed)")
        return

    # Process batch (custom image list if filtered)
    typer.echo("Processing images...")
    if include or exclude or max_images:
        # Process filtered images individually
        from lux_depth_v3.input_manager import ImageInput
        from lux_depth_v3.enhance.batch_stats import compute_batch_runtime_stats
        from lux_depth_v3.enhance.manifest import BatchManifest
        import datetime

        batch_id = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        start_time = datetime.datetime.now().isoformat()

        results = []
        for img_path in tqdm(filtered_images, desc="Processing"):
            image_input = ImageInput(path=img_path)
            try:
                result = orchestrator.enhance_image(image_input, input_root=input_dir)
                results.append(result)
            except Exception as e:
                logging.error(f"Failed to process {img_path}: {e}")
                results.append({"status": "error", "image": str(img_path), "error": str(e)})

        # Generate batch manifest for filtered runs
        end_time = datetime.datetime.now().isoformat()
        succeeded = sum(1 for r in results if r.get("status") == "ok")
        failed = sum(1 for r in results if r.get("status") == "error")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        runtime_stats = compute_batch_runtime_stats(results)

        batch_manifest = BatchManifest(
            batch_id=batch_id,
            start_time=start_time,
            end_time=end_time,
            config={
                "model_variant": model.value,
                "preset": preset.value if preset else None,
                "depth_quantization": depth_quantization,
                "v2_preset": v2_preset,
                "v2_upscaler_backend": v2_upscaler,
                "execution_mode": execution_mode,
                "depth_fallback": depth_fallback,
                "filtered": True,
                "include": include,
                "exclude": exclude,
                "max_images": max_images,
            },
            images=[
                {
                    "stem": Path(r["image"]).stem,
                    "status": r.get("status", "unknown"),
                    "manifest": str(r.get("manifest", "")) if r.get("status") == "ok" else None,
                    "runtime_s": r.get("runtime_s", 0.0) if r.get("status") == "ok" else None,
                    "error": r.get("error") if r.get("status") == "error" else None,
                }
                for r in results
            ],
            summary={
                "total": len(results),
                "ok": succeeded,
                "error": failed,
                "skipped": skipped,
                **runtime_stats,
            },
        )

        # Write batch manifest
        batch_manifest_path = output_dir / "manifests" / f"batch_{batch_id}.json"
        batch_manifest.write(batch_manifest_path)
        typer.echo(f"Batch manifest written to {batch_manifest_path}")
    else:
        # Use default batch processing
        results = orchestrator.enhance_batch(input_dir)

    # Summary
    typer.echo("\n📊 Processing Summary")
    typer.echo("=" * 70)
    succeeded = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skipped")

    typer.echo(f"✓ Succeeded: {succeeded}")
    typer.echo(f"✗ Failed: {failed}")
    typer.echo(f"⊘ Skipped: {skipped}")
    typer.echo(f"Total: {len(results)}")

    if failed > 0:
        typer.echo("\nFailed images:")
        for r in results:
            if r["status"] == "error":
                typer.echo(f"  - {r['image']}: {r.get('error', 'Unknown error')}")

    typer.echo(f"\nOutput directory: {output_dir}")
    typer.echo("=" * 70)


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
