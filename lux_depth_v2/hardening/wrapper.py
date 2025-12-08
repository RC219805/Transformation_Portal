from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2.config import PipelineConfig

from .policy import HardeningPolicy
from .profiling import StageProfiler
from .safe_io import safe_mkdir, validate_image_file
from .stamping import stamp_report, write_run_manifest


@dataclass(frozen=True)
class HardenedRunResult:
    report: Dict[str, Any]
    manifest_path: Optional[Path] = None


class LuxPipelineV2Hardened:
    """
    Opt-in wrapper that hardens LuxPipelineV2.

    - Input validation (extension, size cap, magic bytes)
    - Output root enforcement (optional)
    - Run stamping (git commit, config hash, host info)
    - Optional run manifest emission

    Existing LuxPipelineV2 behavior remains unchanged unless you use this wrapper.
    """

    def __init__(self, config: PipelineConfig, policy: Optional[HardeningPolicy] = None):
        self.config = config
        self.policy = policy or HardeningPolicy.load()
        self.policy.assert_output_allowed(Path(self.config.output_dir))
        safe_mkdir(Path(self.config.output_dir), mode=self.policy.safe_dir_mode)
        self._pipe = LuxPipelineV2(config)

    @property
    def device(self):
        # Mirror existing pipeline shape
        return getattr(self._pipe, "device", None)

    def process_one(self, input_path: Path) -> Dict[str, Any]:
        input_path = Path(input_path)

        # Guardrails
        validate_image_file(input_path, self.policy)

        profiler = StageProfiler(enabled=True)
        with profiler.stage("pipeline_total"):
            report = self._pipe.process_one(input_path)

        # Stamp report (reproducibility best-practice)
        stamped = report
        if self.policy.stamp_reports:
            stamped = stamp_report(
                report,
                config=self.config,
                input_path=input_path,
                output_dir=Path(self.config.output_dir),
                include_input_hash=self.policy.stamp_include_input_hash,
                profiler=profiler.summary(),
            )

        # Emit run manifest (minimal, useful for audit/validation pipelines)
        if self.policy.write_run_manifest:
            manifest = {
                "run_id": stamped.get("meta", {}).get("run_id"),
                "input": stamped.get("meta", {}).get("input_file"),
                "config_hash": stamped.get("meta", {}).get("config_hash"),
                "git_commit": stamped.get("meta", {}).get("runtime", {}).get("git_commit"),
                "output_dir": stamped.get("meta", {}).get("output_dir"),
                "profile_ms": stamped.get("profile_ms", {}),
                # Preserve original report core fields unchanged:
                "report_core": {k: v for k, v in report.items() if k not in {"meta", "profile_ms"}},
            }
            out_path = Path(self.config.output_dir) / self.policy.run_manifest_name
            write_run_manifest(out_path, manifest)

        return stamped

    def process_batch(self, inputs: list[Path]) -> list[Dict[str, Any]]:
        # Conservative: sequential by default to avoid memory spikes.
        # If LuxPipelineV2 already supports batch optimizations, those remain available via its own API.
        return [self.process_one(p) for p in inputs]
