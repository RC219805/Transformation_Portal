"""Performance budget checks for reconstruction rasterizer paths.

These tests are intentionally in the slow + benchmark lanes to avoid impacting
fast PR feedback while still enforcing regression budgets nightly.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is required for reconstruction performance budgets")
pytestmark = [pytest.mark.ml, pytest.mark.slow, pytest.mark.benchmark]

from transformation_portal.spatial_ai.reconstruction.gaussian_rasterizer import (  # pylint: disable=wrong-import-position
    compute_rgb_loss,
    render_gaussians,
)

FORWARD_BUDGET_MS = {"p50": 80.0, "p95": 150.0, "max": 220.0}
BACKWARD_BUDGET_MS = {"p50": 220.0, "p95": 420.0, "max": 650.0}
METRICS_ENV_VAR = "TP_RECON_PERF_METRICS_FILE"
FIXTURE_SEED = 20260305


def _build_fixture(num_gaussians: int = 96, image_size: tuple[int, int] = (64, 64)):
    torch.manual_seed(FIXTURE_SEED)
    np.random.seed(FIXTURE_SEED)
    h, w = image_size
    positions = torch.randn(num_gaussians, 3, dtype=torch.float32)
    positions[:, 2] = positions[:, 2].abs() + 3.0
    colors = torch.sigmoid(torch.randn(num_gaussians, 3, dtype=torch.float32))
    scales = torch.rand(num_gaussians, 3, dtype=torch.float32) * 0.2 + 0.05
    rotations = torch.zeros(num_gaussians, 4, dtype=torch.float32)
    rotations[:, 0] = 1.0
    opacities = torch.rand(num_gaussians, 1, dtype=torch.float32) * 0.4 + 0.5
    intrinsics = torch.tensor(
        [
            [85.0, 0.0, w / 2.0],
            [0.0, 85.0, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    extrinsics = torch.eye(4, dtype=torch.float32)
    target = torch.rand(h, w, 3, dtype=torch.float32)
    return positions, colors, scales, rotations, opacities, intrinsics, extrinsics, target, image_size


def _measure_latency_ms(
    fn: Callable[[], None],
    *,
    warmup: int = 3,
    iterations: int = 12,
) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    timings: List[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - started) * 1000.0)
    values = np.array(timings, dtype=np.float64)
    return {
        "samples": float(iterations),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }


def _record_metrics(case_name: str, metrics: Dict[str, float], budget: Dict[str, float]) -> None:
    output_file = os.getenv(METRICS_ENV_VAR)
    if not output_file:
        return

    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object]
    if destination.exists():
        try:
            payload = json.loads(destination.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}

    payload["schema_version"] = "1.0"
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload.setdefault("cases", {})
    payload["cases"][case_name] = {"metrics_ms": metrics, "budget_ms": budget}
    destination.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")


def test_forward_render_budget_cpu():
    """Forward render p50/p95/max must remain within agreed nightly budget."""
    positions, colors, scales, rotations, opacities, intrinsics, extrinsics, _, image_size = _build_fixture()

    def run_forward() -> None:
        with torch.no_grad():
            render_gaussians(
                positions=positions,
                colors=colors,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                image_size=image_size,
                device="cpu",
            )

    metrics = _measure_latency_ms(run_forward)
    _record_metrics("forward_cpu", metrics, FORWARD_BUDGET_MS)
    assert metrics["p95_ms"] >= metrics["p50_ms"]
    assert metrics["p50_ms"] <= FORWARD_BUDGET_MS["p50"], f"Forward p50 budget exceeded: {metrics}"
    assert metrics["p95_ms"] <= FORWARD_BUDGET_MS["p95"], f"Forward p95 budget exceeded: {metrics}"
    assert metrics["max_ms"] <= FORWARD_BUDGET_MS["max"], f"Forward max budget exceeded: {metrics}"


def test_backward_render_budget_cpu():
    """Backward pass p50/p95/max must remain within agreed nightly budget."""
    positions, colors, scales, rotations, opacities, intrinsics, extrinsics, target, image_size = _build_fixture()
    positions.requires_grad_(True)
    colors.requires_grad_(True)
    scales.requires_grad_(True)
    rotations.requires_grad_(True)
    opacities.requires_grad_(True)
    trainable = [positions, colors, scales, rotations, opacities]

    def run_backward() -> None:
        for tensor in trainable:
            if tensor.grad is not None:
                tensor.grad.zero_()
        rendered = render_gaussians(
            positions=positions,
            colors=colors,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_size=image_size,
            device="cpu",
        )
        loss = compute_rgb_loss(rendered, target)
        loss.backward()

    metrics = _measure_latency_ms(run_backward)
    _record_metrics("backward_cpu", metrics, BACKWARD_BUDGET_MS)
    assert metrics["p95_ms"] >= metrics["p50_ms"]
    assert metrics["p50_ms"] <= BACKWARD_BUDGET_MS["p50"], f"Backward p50 budget exceeded: {metrics}"
    assert metrics["p95_ms"] <= BACKWARD_BUDGET_MS["p95"], f"Backward p95 budget exceeded: {metrics}"
    assert metrics["max_ms"] <= BACKWARD_BUDGET_MS["max"], f"Backward max budget exceeded: {metrics}"
