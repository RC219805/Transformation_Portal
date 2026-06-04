#!/usr/bin/env python3
"""Benchmark Unified Luxury Pipeline batch I/O scheduling.

This harness compares the current serial batch path with the opt-in
``parallel_io=True`` path using deterministic CPU-only image fixtures. It is
intentionally scoped to load/save scheduling. GPU and backend acceleration
benchmarks belong in separate harnesses because they have different dependency,
hardware, and failure profiles.
"""

from __future__ import annotations

import argparse
import contextlib
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from transformation_portal.ingest.canonical_json import dump_json
from transformation_portal.pipelines.unified_luxury_pipeline import (
    OutputFormat,
    ProcessingProfile,
    SceneType,
    UnifiedLuxuryPipeline,
    UnifiedPipelineConfig,
)

SCHEMA = "tp.unified_luxury.batch_io_benchmark.v1"
DEFAULT_OUTPUT_FORMATS = (OutputFormat.MASTER_TIFF,)


@dataclass(frozen=True)
class TrialMetrics:
    """One measured batch run."""

    mode: str
    run_index: int
    wall_time_s: float
    images_total: int
    images_succeeded: int
    images_failed: int
    output_files: int
    peak_rss_mib: float | None
    peak_rss_delta_mib: float | None


@dataclass(frozen=True)
class BenchmarkOptions:
    """Configuration captured in the benchmark report."""

    runs: int
    warmup_runs: int
    image_count: int
    width: int
    height: int
    input_format: str
    output_formats: tuple[str, ...]
    io_prefetch_size: int
    io_saver_workers: int
    min_speedup: float
    memory_limit_mib: float | None


class PeakRSSMonitor:
    """Poll process RSS during a run without making psutil mandatory."""

    def __init__(self, *, interval_s: float = 0.005) -> None:
        self.interval_s = interval_s
        self.available = False
        self.baseline_rss_bytes: int | None = None
        self.peak_rss_bytes: int | None = None
        self.samples = 0
        self._process: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()

    def __enter__(self) -> "PeakRSSMonitor":
        try:
            import psutil  # type: ignore
        except ImportError:
            return self

        self._process = psutil.Process()
        self.available = True
        self._stop.clear()
        self._ready.clear()
        self.baseline_rss_bytes = int(self._process.memory_info().rss)
        self.peak_rss_bytes = self.baseline_rss_bytes
        self.samples = 0
        self._thread = threading.Thread(target=self._poll, name="batch-io-rss", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=max(0.05, self.interval_s * 10)):
            self._stop.set()
            self._thread.join(timeout=1.0)
            raise RuntimeError("PeakRSSMonitor first-sample barrier timed out")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.available:
            self._sample()
        return False

    @property
    def peak_rss_mib(self) -> float | None:
        if self.peak_rss_bytes is None:
            return None
        return self.peak_rss_bytes / (1024 * 1024)

    @property
    def peak_rss_delta_mib(self) -> float | None:
        if self.peak_rss_bytes is None or self.baseline_rss_bytes is None:
            return None
        return max(0.0, (self.peak_rss_bytes - self.baseline_rss_bytes) / (1024 * 1024))

    def _poll(self) -> None:
        self._sample()
        self._ready.set()
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        if self._process is None:
            return
        rss = int(self._process.memory_info().rss)
        self.samples += 1
        if self.peak_rss_bytes is None or rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss


def parse_output_formats(raw: str) -> tuple[OutputFormat, ...]:
    """Parse comma-separated OutputFormat values or enum names."""

    aliases: dict[str, OutputFormat] = {}
    for fmt in OutputFormat:
        aliases[fmt.value.lower()] = fmt
        aliases[fmt.name.lower()] = fmt

    parsed = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in aliases:
            allowed = ", ".join(sorted(aliases))
            raise argparse.ArgumentTypeError(f"unknown output format {item!r}; allowed: {allowed}")
        parsed.append(aliases[key])

    if not parsed:
        raise argparse.ArgumentTypeError("--output-formats must include at least one format")
    return tuple(parsed)


def generate_synthetic_inputs(
    input_dir: Path,
    *,
    count: int,
    width: int,
    height: int,
    image_format: str,
    seed: int = 1337,
) -> list[Path]:
    """Create deterministic synthetic RGB fixtures for the benchmark."""

    if count < 1:
        raise ValueError("count must be >= 1")
    if width < 1 or height < 1:
        raise ValueError("width and height must be >= 1")

    normalized_format = image_format.lower().lstrip(".")
    extension = {
        "jpg": ".jpg",
        "jpeg": ".jpg",
        "png": ".png",
        "tif": ".tif",
        "tiff": ".tif",
    }.get(normalized_format)
    if extension is None:
        raise ValueError("image_format must be one of jpg, jpeg, png, tif, tiff")

    input_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x_coords = np.linspace(0, 255, width, dtype=np.float32)
    y_coords = np.linspace(0, 255, height, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)

    paths: list[Path] = []
    for index in range(count):
        noise = rng.integers(0, 24, size=(height, width), dtype=np.uint8)
        red = ((xx + index * 17) % 256).astype(np.uint8)
        green = ((yy + index * 11) % 256).astype(np.uint8)
        blue = (((xx + yy) / 2 + noise + index * 7) % 256).astype(np.uint8)
        image = Image.fromarray(np.stack([red, green, blue], axis=2), mode="RGB")
        path = input_dir / f"synthetic_{index:04d}{extension}"
        if extension == ".jpg":
            image.save(path, quality=95, subsampling=0)
        elif extension == ".png":
            image.save(path, compress_level=3)
        else:
            image.save(path, compression="tiff_lzw")
        paths.append(path)
    return paths


def percentile(values: Sequence[float], q: float) -> float:
    """Return an interpolated percentile for a non-empty sequence."""

    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (q / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_trials(trials: Sequence[TrialMetrics]) -> dict[str, Any]:
    """Build stable summary metrics from measured trials."""

    wall_times = [trial.wall_time_s for trial in trials]
    peak_rss = [trial.peak_rss_mib for trial in trials if trial.peak_rss_mib is not None]
    peak_delta = [trial.peak_rss_delta_mib for trial in trials if trial.peak_rss_delta_mib is not None]
    failures = sum(trial.images_failed for trial in trials)
    outputs = sum(trial.output_files for trial in trials)
    return {
        "runs": len(trials),
        "wall_time_s": {
            "min": min(wall_times) if wall_times else 0.0,
            "mean": sum(wall_times) / len(wall_times) if wall_times else 0.0,
            "p50": percentile(wall_times, 50),
            "p95": percentile(wall_times, 95),
            "max": max(wall_times) if wall_times else 0.0,
        },
        "peak_rss_mib": {
            "max": max(peak_rss) if peak_rss else None,
            "delta_max": max(peak_delta) if peak_delta else None,
        },
        "images_failed": failures,
        "output_files": outputs,
    }


def evaluate_parallel_io_default(
    *,
    serial_summary: dict[str, Any],
    parallel_summary: dict[str, Any],
    min_speedup: float,
    memory_limit_mib: float | None,
) -> dict[str, Any]:
    """Evaluate whether results are strong enough to consider a default flip."""

    serial_mean = float(serial_summary["wall_time_s"]["mean"])
    parallel_mean = float(parallel_summary["wall_time_s"]["mean"])
    speedup = serial_mean / parallel_mean if parallel_mean > 0 else 0.0
    parallel_peak = parallel_summary["peak_rss_mib"]["max"]
    failures = int(serial_summary["images_failed"]) + int(parallel_summary["images_failed"])

    reasons: list[str] = []
    if failures:
        reasons.append("one or more benchmark images failed")
    if speedup < min_speedup:
        reasons.append(f"mean speedup {speedup:.3f}x is below required {min_speedup:.3f}x")
    if memory_limit_mib is None:
        reasons.append("no memory limit supplied")
    elif parallel_peak is None:
        reasons.append("peak RSS was unavailable; install psutil to capture memory")
    elif float(parallel_peak) > memory_limit_mib:
        reasons.append(f"parallel peak RSS {parallel_peak:.1f} MiB exceeds limit {memory_limit_mib:.1f} MiB")

    candidate = not reasons
    return {
        "mean_speedup": speedup,
        "parallel_io_default_candidate": candidate,
        "decision": "candidate_after_representative_runs" if candidate else "keep_false",
        "reason": "measured speedup and memory stayed within thresholds" if candidate else "; ".join(reasons),
        "policy": (
            "Do not change UnifiedPipelineConfig.parallel_io default from False until representative "
            "production-sized batches meet the speedup threshold and a documented memory limit."
        ),
    }


def reuse_assessment() -> dict[str, Any]:
    """Document adjacent pipeline reuse boundaries for this benchmark."""

    return {
        "recipe_pipeline": {
            "surface": "src/transformation_portal/pipeline_unified.py",
            "current_control": "parallel=True uses ThreadPoolExecutor with isolated worker pipeline instances",
            "recommendation": (
                "defer ParallelIOPipeline reuse until a separate benchmark shows load/save overlap "
                "beats whole-image worker parallelism without changing recipe state or RAG indexing semantics"
            ),
        },
        "tiff_batch_processor": {
            "surface": "src/luxury_tiff_batch_processor",
            "current_control": "--workers uses ProcessPoolExecutor for per-image TIFF processing",
            "recommendation": (
                "defer ParallelIOPipeline reuse until TIFF-specific measurements show lower wall time "
                "and acceptable RSS versus process-based parallel execution"
            ),
        },
        "acceleration_scope": (
            "GPU/backend acceleration is out of scope for this I/O benchmark and should stay in "
            "hardware-specific benchmark lanes."
        ),
    }


def _run_single_trial(
    *,
    mode: str,
    run_index: int,
    input_paths: Sequence[Path],
    output_dir: Path,
    output_formats: Sequence[OutputFormat],
    parallel_io: bool,
    io_prefetch_size: int,
    io_saver_workers: int,
) -> TrialMetrics:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = UnifiedPipelineConfig(
        scene_type=SceneType.INTERIOR,
        profile=ProcessingProfile.PERFORMANCE,
        output_formats=list(output_formats),
        output_dir=output_dir,
        enable_depth=False,
        enable_material_response=False,
        enable_vfx=False,
        enable_color_grading=False,
        preserve_metadata=False,
        parallel_outputs=False,
        parallel_io=parallel_io,
        io_prefetch_size=io_prefetch_size,
        io_saver_workers=io_saver_workers,
        device="cpu",
    )
    pipeline = UnifiedLuxuryPipeline(config)
    start = time.perf_counter()
    with PeakRSSMonitor() as rss:
        results = pipeline.batch_process(list(input_paths), show_progress=False)
    wall_time_s = time.perf_counter() - start
    succeeded = sum(1 for outputs in results.values() if outputs)
    output_files = sum(len(outputs) for outputs in results.values())
    return TrialMetrics(
        mode=mode,
        run_index=run_index,
        wall_time_s=wall_time_s,
        images_total=len(input_paths),
        images_succeeded=succeeded,
        images_failed=len(input_paths) - succeeded,
        output_files=output_files,
        peak_rss_mib=rss.peak_rss_mib,
        peak_rss_delta_mib=rss.peak_rss_delta_mib,
    )


def run_mode_trials(
    *,
    mode: str,
    input_paths: Sequence[Path],
    work_dir: Path,
    output_formats: Sequence[OutputFormat],
    parallel_io: bool,
    runs: int,
    warmup_runs: int,
    io_prefetch_size: int,
    io_saver_workers: int,
) -> list[TrialMetrics]:
    """Run warmup and measured trials for one mode."""

    if runs < 1:
        raise ValueError("runs must be >= 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")

    for warmup_index in range(warmup_runs):
        _run_single_trial(
            mode=f"{mode}_warmup",
            run_index=warmup_index,
            input_paths=input_paths,
            output_dir=work_dir / f"{mode}_warmup_{warmup_index}",
            output_formats=output_formats,
            parallel_io=parallel_io,
            io_prefetch_size=io_prefetch_size,
            io_saver_workers=io_saver_workers,
        )

    trials: list[TrialMetrics] = []
    for run_index in range(runs):
        trials.append(
            _run_single_trial(
                mode=mode,
                run_index=run_index,
                input_paths=input_paths,
                output_dir=work_dir / f"{mode}_{run_index}",
                output_formats=output_formats,
                parallel_io=parallel_io,
                io_prefetch_size=io_prefetch_size,
                io_saver_workers=io_saver_workers,
            )
        )
    return trials


def build_report(
    *,
    input_paths: Sequence[Path],
    work_dir: Path,
    options: BenchmarkOptions,
    output_formats: Sequence[OutputFormat],
) -> dict[str, Any]:
    """Run the benchmark and return a JSON-serializable report."""

    serial_trials = run_mode_trials(
        mode="serial",
        input_paths=input_paths,
        work_dir=work_dir,
        output_formats=output_formats,
        parallel_io=False,
        runs=options.runs,
        warmup_runs=options.warmup_runs,
        io_prefetch_size=options.io_prefetch_size,
        io_saver_workers=options.io_saver_workers,
    )
    parallel_trials = run_mode_trials(
        mode="parallel_io",
        input_paths=input_paths,
        work_dir=work_dir,
        output_formats=output_formats,
        parallel_io=True,
        runs=options.runs,
        warmup_runs=options.warmup_runs,
        io_prefetch_size=options.io_prefetch_size,
        io_saver_workers=options.io_saver_workers,
    )
    serial_summary = summarize_trials(serial_trials)
    parallel_summary = summarize_trials(parallel_trials)
    return {
        "schema": SCHEMA,
        "created_at": time.time(),
        "options": asdict(options),
        "inputs": {
            "count": len(input_paths),
            "paths": [str(path) for path in input_paths],
        },
        "trials": {
            "serial": [asdict(trial) for trial in serial_trials],
            "parallel_io": [asdict(trial) for trial in parallel_trials],
        },
        "summary": {
            "serial": serial_summary,
            "parallel_io": parallel_summary,
        },
        "comparison": evaluate_parallel_io_default(
            serial_summary=serial_summary,
            parallel_summary=parallel_summary,
            min_speedup=options.min_speedup,
            memory_limit_mib=options.memory_limit_mib,
        ),
        "reuse_assessment": reuse_assessment(),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Persist a deterministic JSON benchmark report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        dump_json(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return value


def _non_negative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="Benchmark Unified Luxury Pipeline batch I/O scheduling.")
    parser.add_argument(
        "--input-dir", type=Path, help="Existing input directory. If omitted, synthetic fixtures are generated."
    )
    parser.add_argument("--work-dir", type=Path, help="Directory for generated inputs and per-trial outputs.")
    parser.add_argument("--output-json", type=Path, help="Path to write the benchmark JSON report.")
    parser.add_argument("--runs", type=_positive_int, default=3)
    parser.add_argument("--warmup-runs", type=_non_negative_int, default=1)
    parser.add_argument("--image-count", type=_positive_int, default=6)
    parser.add_argument("--width", type=_positive_int, default=512)
    parser.add_argument("--height", type=_positive_int, default=384)
    parser.add_argument("--input-format", choices=("jpg", "jpeg", "png", "tif", "tiff"), default="tif")
    parser.add_argument("--output-formats", type=parse_output_formats, default=DEFAULT_OUTPUT_FORMATS)
    parser.add_argument("--io-prefetch-size", type=_positive_int, default=2)
    parser.add_argument("--io-saver-workers", type=_positive_int, default=2)
    parser.add_argument("--min-speedup", type=float, default=1.10)
    parser.add_argument("--memory-limit-mib", type=float, default=None)
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep generated fixtures and outputs after the run.")
    return parser


def _input_paths_from_dir(input_dir: Path) -> list[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.JPG", "*.JPEG", "*.PNG", "*.TIF", "*.TIFF")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(path for path in input_dir.glob(pattern) if path.is_file()))
    deduped = sorted(set(paths))
    if not deduped:
        raise ValueError(f"no supported image files found in {input_dir}")
    return deduped


def run_from_args(args: argparse.Namespace) -> Path:
    """Execute the benchmark from parsed CLI args and return the report path."""

    temp_ctx: contextlib.AbstractContextManager[str] | None = None
    if args.work_dir is None:
        if args.keep_work_dir:
            work_dir = Path(tempfile.mkdtemp(prefix="tp-unified-luxury-batch-io-"))
        else:
            temp_ctx = tempfile.TemporaryDirectory(prefix="tp-unified-luxury-batch-io-")
            work_dir = Path(temp_ctx.__enter__())
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.input_dir is None:
            input_paths = generate_synthetic_inputs(
                work_dir / "inputs",
                count=args.image_count,
                width=args.width,
                height=args.height,
                image_format=args.input_format,
            )
            image_count = args.image_count
        else:
            input_paths = _input_paths_from_dir(args.input_dir)
            image_count = len(input_paths)

        output_formats = tuple(args.output_formats)
        options = BenchmarkOptions(
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            image_count=image_count,
            width=args.width,
            height=args.height,
            input_format=args.input_format,
            output_formats=tuple(fmt.value for fmt in output_formats),
            io_prefetch_size=args.io_prefetch_size,
            io_saver_workers=args.io_saver_workers,
            min_speedup=args.min_speedup,
            memory_limit_mib=args.memory_limit_mib,
        )
        report = build_report(
            input_paths=input_paths,
            work_dir=work_dir / "runs",
            options=options,
            output_formats=output_formats,
        )
        output_json = args.output_json or (Path(tempfile.gettempdir()) / "tp-unified-luxury-batch-io-benchmark.json")
        write_report(output_json, report)
        return output_json
    finally:
        if temp_ctx is not None and not args.keep_work_dir:
            temp_ctx.__exit__(None, None, None)


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        output_json = run_from_args(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"benchmark error: {exc}\n")
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
