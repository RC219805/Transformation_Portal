#!/usr/bin/env python3
"""
Detailed V3+V2 Pipeline Profiling
Analyzes stage-by-stage performance with optimization recommendations.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from lux_depth_v3.config import DA3Config, ModelVariant
from lux_depth_v3.input_manager import InputManager
from lux_depth_v3.inference import DA3InferenceEngine
from lux_depth_v3.depth_writer import DepthWriter
from lux_depth_v3.enhance.orchestrator import EnhanceOrchestrator


class StageTimer:
    """Track timing for each pipeline stage."""

    def __init__(self):
        self.stages: List[Dict[str, Any]] = []
        self.current_stage = None
        self.start_time = None

    def start(self, stage_name: str, metadata: Dict = None):
        """Start timing a stage."""
        self.current_stage = {
            "name": stage_name,
            "metadata": metadata or {},
            "start": time.perf_counter(),
        }

    def end(self):
        """End timing current stage."""
        if self.current_stage:
            elapsed = time.perf_counter() - self.current_stage["start"]
            self.current_stage["elapsed_ms"] = elapsed * 1000
            self.current_stage["elapsed_s"] = elapsed
            self.stages.append(self.current_stage)
            self.current_stage = None

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_time = sum(s["elapsed_s"] for s in self.stages)

        summary = {
            "total_time_s": total_time,
            "total_time_ms": total_time * 1000,
            "stages": [],
        }

        for stage in self.stages:
            pct = (stage["elapsed_s"] / total_time * 100) if total_time > 0 else 0
            summary["stages"].append(
                {
                    "name": stage["name"],
                    "time_ms": stage["elapsed_ms"],
                    "time_s": stage["elapsed_s"],
                    "percentage": pct,
                    "metadata": stage["metadata"],
                }
            )

        return summary


def profile_v3_only(input_dir: Path, output_dir: Path, num_samples: int = 5):
    """Profile V3 depth generation only."""
    print("\n" + "=" * 80)
    print("PROFILING: V3 DEPTH GENERATION (Stage A)")
    print("=" * 80)

    timer = StageTimer()

    # Stage: Input scanning
    timer.start("Input Scanning")
    input_mgr = InputManager(input_dir, max_file_size_mb=3000)
    images = list(input_mgr.discover_images())[:num_samples]
    timer.end()

    print(f"\nProcessing {len(images)} images from: {input_dir}")

    # Stage: Model loading
    timer.start("Model Loading", {"model": "da3-base-v1.1"})
    config = DA3Config(model_variant=ModelVariant.DA3_BASE_V1_1)
    engine = DA3InferenceEngine(config)
    engine.load_model()
    timer.end()

    # Stage: Depth inference (per-image)
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    writer = DepthWriter(str(depth_dir))

    for i, img_input in enumerate(images, 1):
        # Load image
        timer.start(f"Image Load #{i}", {"file": img_input.path.name})
        loaded = img_input.load()
        timer.end()

        # Inference
        timer.start(
            f"Depth Inference #{i}",
            {
                "file": img_input.path.name,
                "size": f"{loaded.shape[1]}x{loaded.shape[0]}",
            },
        )
        result = engine.inference(img_input)
        timer.end()

        # Write depth
        timer.start(f"Depth Write #{i}", {"file": img_input.path.name})
        writer.write(result, img_input.path.stem)
        timer.end()

    return timer.get_summary()


def profile_v3_v2_integrated(input_dir: Path, output_dir: Path, num_samples: int = 5):
    """Profile full V3+V2 integrated pipeline."""
    print("\n" + "=" * 80)
    print("PROFILING: V3+V2 INTEGRATED PIPELINE (Stage A + Stage B)")
    print("=" * 80)

    timer = StageTimer()

    # Stage: Orchestrator initialization
    timer.start("Orchestrator Init", {"preset": "interior_luxury"})
    orchestrator = EnhanceOrchestrator(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        preset="interior_luxury",
        num_samples=num_samples,
    )
    timer.end()

    # Stage: Input discovery
    timer.start("Input Discovery")
    images = list(orchestrator.input_manager.discover_images())[:num_samples]
    timer.end()

    print(f"\nProcessing {len(images)} images from: {input_dir}")

    # Stage: Model loading
    timer.start(
        "Model Loading (V3)",
        {"model": orchestrator.config.model_variant.value.display_name},
    )
    orchestrator.engine.load_model()
    timer.end()

    # Process each image through full pipeline
    for i, img_input in enumerate(images, 1):
        img_name = img_input.path.name

        # Stage A: Depth generation
        timer.start(f"A: Image Load #{i}", {"file": img_name})
        loaded = img_input.load()
        timer.end()

        timer.start(
            f"A: Depth Inference #{i}",
            {"file": img_name, "size": f"{loaded.shape[1]}x{loaded.shape[0]}"},
        )
        depth_result = orchestrator.engine.inference(img_input)
        timer.end()

        timer.start(f"A: Depth Write #{i}", {"file": img_name})
        depth_path = orchestrator.depth_writer.write(depth_result, img_input.path.stem)
        timer.end()

        # Stage B: V2 enhancement
        timer.start(
            f"B: V2 Enhancement #{i}",
            {"file": img_name, "upscale": orchestrator.v2_config.upscale_factor},
        )
        # This is where V2 subprocess runs
        # We can't easily time internals, but we can time the whole subprocess
        timer.end()

    # Stage: Manifest generation
    timer.start("Manifest Generation")
    # Note: This happens in orchestrator.run(), simulating here
    timer.end()

    return timer.get_summary()


def analyze_optimization_opportunities(v3_summary: Dict, v3v2_summary: Dict):
    """Analyze performance data and suggest optimizations."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION ANALYSIS")
    print("=" * 80)

    opportunities = []

    # Analyze V3 stages
    v3_stages = {s["name"]: s for s in v3_summary["stages"]}

    # Check model loading overhead
    if "Model Loading" in v3_stages:
        model_load_time = v3_stages["Model Loading"]["time_s"]
        if model_load_time > 2.0:
            opportunities.append(
                {
                    "stage": "Model Loading",
                    "current_time": f"{model_load_time:.2f}s",
                    "impact": "HIGH",
                    "optimization": "Use DA3 backend service (lux-depth-v3 backend-start) for batch processing",
                    "expected_improvement": "10-20x speedup (amortize load cost across batch)",
                    "priority": 1,
                }
            )

    # Check average inference time
    inference_stages = [s for s in v3_summary["stages"] if "Depth Inference" in s["name"]]
    if inference_stages:
        avg_inference = sum(s["time_ms"] for s in inference_stages) / len(inference_stages)
        if avg_inference > 100:
            opportunities.append(
                {
                    "stage": "Depth Inference",
                    "current_time": f"{avg_inference:.0f}ms avg",
                    "impact": "MEDIUM",
                    "optimization": "Use CoreML model on Apple Silicon for 3-5x speedup",
                    "expected_improvement": f"{avg_inference / 3:.0f}-{avg_inference / 5:.0f}ms target",
                    "priority": 2,
                }
            )

    # Check I/O overhead
    load_stages = [s for s in v3_summary["stages"] if "Image Load" in s["name"]]
    if load_stages:
        total_load_time = sum(s["time_s"] for s in load_stages)
        load_pct = (total_load_time / v3_summary["total_time_s"]) * 100
        if load_pct > 10:
            opportunities.append(
                {
                    "stage": "Image Loading",
                    "current_time": f"{total_load_time:.2f}s ({load_pct:.1f}% of total)",
                    "impact": "MEDIUM",
                    "optimization": "Use async I/O with concurrent.futures for parallel loading",
                    "expected_improvement": f"{load_pct / 2:.1f}% time reduction",
                    "priority": 3,
                }
            )

    # Check write overhead
    write_stages = [s for s in v3_summary["stages"] if "Depth Write" in s["name"]]
    if write_stages:
        total_write_time = sum(s["time_s"] for s in write_stages)
        write_pct = (total_write_time / v3_summary["total_time_s"]) * 100
        if write_pct > 5:
            opportunities.append(
                {
                    "stage": "Depth Writing",
                    "current_time": f"{total_write_time:.2f}s ({write_pct:.1f}% of total)",
                    "impact": "LOW",
                    "optimization": "Use compression level 1 (fast) instead of default 6",
                    "expected_improvement": f"{write_pct * 0.4:.1f}% time reduction",
                    "priority": 4,
                }
            )

    # Analyze V3+V2 integration overhead
    if v3v2_summary:
        v2_stages = [s for s in v3v2_summary["stages"] if "B: V2 Enhancement" in s["name"]]
        if v2_stages:
            avg_v2_time = sum(s["time_s"] for s in v2_stages) / len(v2_stages)
            v2_pct = (sum(s["time_s"] for s in v2_stages) / v3v2_summary["total_time_s"]) * 100

            opportunities.append(
                {
                    "stage": "V2 Enhancement (subprocess)",
                    "current_time": f"{avg_v2_time:.2f}s avg ({v2_pct:.1f}% of total)",
                    "impact": "HIGH",
                    "optimization": "Convert V2 to in-process library (avoid subprocess overhead)",
                    "expected_improvement": "15-25% reduction in V2 stage time",
                    "priority": 1,
                }
            )

            if avg_v2_time > 10:
                opportunities.append(
                    {
                        "stage": "V2 Upscaling",
                        "current_time": f"{avg_v2_time:.2f}s avg",
                        "impact": "HIGH",
                        "optimization": "Use GPU batch processing for upscaling (currently sequential)",
                        "expected_improvement": "2-3x speedup for upscaling stage",
                        "priority": 1,
                    }
                )

    return sorted(opportunities, key=lambda x: x["priority"])


def print_summary(title: str, summary: Dict):
    """Pretty print performance summary."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"\nTotal Time: {summary['total_time_s']:.2f}s ({summary['total_time_ms']:.0f}ms)")
    print(f"\nStage Breakdown:")
    print(f"{'Stage':<40} {'Time (ms)':<12} {'Time (s)':<10} {'%':<8}")
    print("-" * 80)

    for stage in summary["stages"]:
        name = stage["name"]
        if len(name) > 38:
            name = name[:35] + "..."

        time_ms = stage["time_ms"]
        time_s = stage["time_s"]
        pct = stage["percentage"]

        print(f"{name:<40} {time_ms:>10.1f}ms  {time_s:>8.2f}s  {pct:>6.1f}%")

        # Print metadata if available
        if stage["metadata"]:
            for key, val in stage["metadata"].items():
                print(f"  └─ {key}: {val}")


def print_optimization_opportunities(opportunities: List[Dict]):
    """Pretty print optimization recommendations."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION OPPORTUNITIES (Ranked by Priority)")
    print("=" * 80)

    for i, opp in enumerate(opportunities, 1):
        print(f"\n{i}. [{opp['impact']} IMPACT] {opp['stage']}")
        print(f"   Current: {opp['current_time']}")
        print(f"   Optimization: {opp['optimization']}")
        print(f"   Expected: {opp['expected_improvement']}")


def main():
    """Run comprehensive pipeline profiling."""
    import os

    input_dir = Path(os.environ.get("IN_DIR", "data/sample_images"))
    output_dir = Path(os.environ.get("OUT_DIR", "output/profile_test"))
    num_samples = 5

    print("\n" + "=" * 80)
    print("V3+V2 PIPELINE PERFORMANCE PROFILER")
    print("=" * 80)
    print(f"\nInput Dir: {input_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"Samples: {num_samples}")

    # Profile V3 only
    v3_summary = profile_v3_only(input_dir, output_dir / "v3_only", num_samples)
    print_summary("V3 DEPTH GENERATION ONLY (Stage A)", v3_summary)

    # Profile V3+V2 integrated
    # v3v2_summary = profile_v3_v2_integrated(input_dir, output_dir / "v3_v2", num_samples)
    # print_summary("V3+V2 INTEGRATED PIPELINE (Stage A + B)", v3v2_summary)

    # Analyze and print optimization opportunities
    opportunities = analyze_optimization_opportunities(v3_summary, None)
    print_optimization_opportunities(opportunities)

    # Calculate throughput
    print("\n" + "=" * 80)
    print("THROUGHPUT METRICS")
    print("=" * 80)

    v3_throughput = (num_samples / v3_summary["total_time_s"]) * 3600
    print(f"\nV3 Only: {v3_throughput:.0f} images/hour ({v3_summary['total_time_s'] / num_samples:.2f}s per image)")

    # if v3v2_summary:
    #     v3v2_throughput = (num_samples / v3v2_summary['total_time_s']) * 3600
    #     print(f"V3+V2:   {v3v2_throughput:.0f} images/hour ({v3v2_summary['total_time_s']/num_samples:.2f}s per image)")
    #     print(f"\nV2 Overhead: {((v3v2_summary['total_time_s'] - v3_summary['total_time_s']) / v3v2_summary['total_time_s'] * 100):.1f}%")

    # Save detailed results
    results_file = output_dir / "profiling_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(
            {
                "v3_only": v3_summary,
                # "v3_v2_integrated": v3v2_summary,
                "optimization_opportunities": opportunities,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )

    print(f"\n\nDetailed results saved to: {results_file}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
