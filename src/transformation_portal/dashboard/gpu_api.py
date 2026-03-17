"""Live GPU utilization monitoring API.

This module provides real-time GPU metrics streaming via WebSocket,
using NVIDIA NVML for hardware monitoring.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Optional imports
try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None

# Try to import pynvml for GPU monitoring
try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    GPU_COUNT = pynvml.nvmlDeviceGetCount()
    logger.info("NVML initialized: %d GPU(s) detected", GPU_COUNT)
except Exception as exc:
    NVML_AVAILABLE = False
    GPU_COUNT = 0
    logger.warning("NVML not available: %s", exc)


@dataclass
class GPUStats:
    """GPU statistics snapshot."""

    index: int
    name: str
    temperature: int  # Celsius
    gpu_util: int  # Percentage
    memory_used: int  # Bytes
    memory_total: int  # Bytes
    memory_util: float  # Percentage
    power_draw: float  # Watts
    power_limit: float  # Watts


def get_gpu_stats(index: int = 0) -> Optional[GPUStats]:
    """Get current GPU statistics.

    Args:
        index: GPU device index

    Returns:
        GPUStats if available, None otherwise
    """
    if not NVML_AVAILABLE:
        return None

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        except Exception:
            power = 0.0
            power_limit = 0.0

        return GPUStats(
            index=index,
            name=name,
            temperature=temp,
            gpu_util=util.gpu,
            memory_used=mem.used,
            memory_total=mem.total,
            memory_util=(mem.used / mem.total * 100) if mem.total > 0 else 0,
            power_draw=power,
            power_limit=power_limit,
        )

    except Exception as exc:
        logger.warning("Failed to get GPU stats for device %d: %s", index, exc)
        return None


def get_all_gpu_stats() -> list[GPUStats]:
    """Get statistics for all GPUs."""
    stats = []
    for i in range(GPU_COUNT):
        s = get_gpu_stats(i)
        if s:
            stats.append(s)
    return stats


def create_gpu_router() -> "APIRouter":
    """Create the GPU monitoring router.

    Returns:
        FastAPI APIRouter with GPU monitoring endpoints
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for GPU monitoring")

    router = APIRouter(prefix="/api/gpu", tags=["gpu"])

    @router.get("/status")
    async def gpu_status():
        """Get current GPU status."""
        if not NVML_AVAILABLE:
            return JSONResponse(
                {
                    "available": False,
                    "message": "NVML not available",
                }
            )

        stats = get_all_gpu_stats()
        return JSONResponse(
            {
                "available": True,
                "gpu_count": GPU_COUNT,
                "gpus": [
                    {
                        "index": s.index,
                        "name": s.name,
                        "temperature_c": s.temperature,
                        "gpu_util_percent": s.gpu_util,
                        "memory_used_gb": s.memory_used / (1024**3),
                        "memory_total_gb": s.memory_total / (1024**3),
                        "memory_util_percent": s.memory_util,
                        "power_draw_w": s.power_draw,
                        "power_limit_w": s.power_limit,
                    }
                    for s in stats
                ],
            }
        )

    @router.websocket("/stream")
    async def gpu_stream(websocket: WebSocket):
        """Stream GPU stats via WebSocket."""
        await websocket.accept()
        logger.info("GPU monitoring WebSocket connected")

        try:
            while True:
                if NVML_AVAILABLE:
                    stats = get_all_gpu_stats()
                    data = {
                        "gpus": [
                            {
                                "index": s.index,
                                "name": s.name,
                                "temperature": s.temperature,
                                "gpu_util": s.gpu_util,
                                "memory_used_gb": round(s.memory_used / (1024**3), 2),
                                "memory_total_gb": round(s.memory_total / (1024**3), 2),
                                "memory_util": round(s.memory_util, 1),
                                "power_draw": round(s.power_draw, 1),
                            }
                            for s in stats
                        ],
                    }
                else:
                    data = {"error": "NVML not available"}

                await websocket.send_json(data)
                await asyncio.sleep(1)  # 1 second interval

        except WebSocketDisconnect:
            logger.info("GPU monitoring WebSocket disconnected")
        except Exception as exc:
            logger.warning("GPU WebSocket error: %s", exc)

    @router.get("/", response_class=HTMLResponse)
    async def gpu_dashboard():
        """Serve the GPU monitoring dashboard."""
        return get_gpu_dashboard_html()

    return router


def get_gpu_dashboard_html() -> str:
    """Get the GPU dashboard frontend HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Monitor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 1rem 2rem;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 1.25rem; }
        .status {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.75rem;
        }
        .status.connected { background: #00d25b; color: #000; }
        .status.disconnected { background: #ff5252; }
        .container { padding: 2rem; max-width: 1200px; margin: 0 auto; }
        .gpu-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 2rem; }
        .gpu-card {
            background: #16213e;
            border-radius: 0.5rem;
            border: 1px solid #0f3460;
            overflow: hidden;
        }
        .gpu-header {
            padding: 1rem 1.5rem;
            background: #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .gpu-header h2 { font-size: 1rem; }
        .gpu-header .temp {
            font-size: 1.5rem;
            font-weight: bold;
        }
        .temp.hot { color: #ff5252; }
        .temp.warm { color: #ffc107; }
        .temp.cool { color: #00d25b; }
        .gpu-body { padding: 1.5rem; }
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        .metric-label { color: #94a3b8; font-size: 0.875rem; }
        .metric-value { font-size: 1.25rem; font-weight: 500; }
        .progress-bar {
            height: 8px;
            background: #1a1a2e;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        .progress-fill {
            height: 100%;
            transition: width 0.3s ease;
        }
        .progress-fill.gpu { background: linear-gradient(90deg, #00d25b, #ffc107, #ff5252); }
        .progress-fill.memory { background: #0d6efd; }
        .progress-fill.power { background: #e94560; }
        .chart-container {
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid #0f3460;
        }
        .chart-title { font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem; }
        .chart {
            height: 80px;
            display: flex;
            align-items: flex-end;
            gap: 2px;
        }
        .chart-bar {
            flex: 1;
            background: #e94560;
            min-height: 2px;
            transition: height 0.3s ease;
        }
        .no-gpu {
            text-align: center;
            padding: 4rem 2rem;
            color: #94a3b8;
        }
        .no-gpu h2 { margin-bottom: 1rem; }
    </style>
</head>
<body>
    <div class="header">
        <h1>GPU Monitor</h1>
        <span id="status" class="status disconnected">Disconnected</span>
    </div>
    <div class="container">
        <div id="gpus" class="gpu-grid">
            <div class="no-gpu">
                <h2>Connecting...</h2>
                <p>Waiting for GPU data</p>
            </div>
        </div>
    </div>

    <script>
        const history = {};
        const HISTORY_SIZE = 60;
        let ws;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/api/gpu/stream`);

            ws.onopen = () => {
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'status connected';
            };

            ws.onclose = () => {
                document.getElementById('status').textContent = 'Disconnected';
                document.getElementById('status').className = 'status disconnected';
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.error) {
                    showNoGpu(data.error);
                } else {
                    updateGpus(data.gpus);
                }
            };
        }

        function showNoGpu(message) {
            document.getElementById('gpus').innerHTML = `
                <div class="no-gpu">
                    <h2>No GPU Available</h2>
                    <p>${message}</p>
                </div>
            `;
        }

        function updateGpus(gpus) {
            if (gpus.length === 0) {
                showNoGpu('No GPUs detected');
                return;
            }

            const container = document.getElementById('gpus');
            container.innerHTML = gpus.map(gpu => {
                // Update history
                if (!history[gpu.index]) history[gpu.index] = [];
                history[gpu.index].push(gpu.gpu_util);
                if (history[gpu.index].length > HISTORY_SIZE) {
                    history[gpu.index].shift();
                }

                const tempClass = gpu.temperature > 80 ? 'hot' : gpu.temperature > 60 ? 'warm' : 'cool';

                return `
                    <div class="gpu-card">
                        <div class="gpu-header">
                            <h2>GPU ${gpu.index}: ${gpu.name}</h2>
                            <span class="temp ${tempClass}">${gpu.temperature}°C</span>
                        </div>
                        <div class="gpu-body">
                            <div class="metric-row">
                                <span class="metric-label">GPU Utilization</span>
                                <span class="metric-value">${gpu.gpu_util}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill gpu" style="width: ${gpu.gpu_util}%"></div>
                            </div>

                            <div class="metric-row" style="margin-top: 1rem;">
                                <span class="metric-label">Memory</span>
                                <span class="metric-value">${gpu.memory_used_gb} / ${gpu.memory_total_gb} GB</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill memory" style="width: ${gpu.memory_util}%"></div>
                            </div>

                            <div class="metric-row" style="margin-top: 1rem;">
                                <span class="metric-label">Power Draw</span>
                                <span class="metric-value">${gpu.power_draw} W</span>
                            </div>

                            <div class="chart-container">
                                <div class="chart-title">GPU Utilization History (60s)</div>
                                <div class="chart">
                                    ${history[gpu.index].map(v =>
                                        `<div class="chart-bar" style="height: ${v}%"></div>`
                                    ).join('')}
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        connect();
    </script>
</body>
</html>"""
