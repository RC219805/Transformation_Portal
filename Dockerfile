# Transformation Portal - Docker Image
# Multi-stage build for optimized image size

# Stage 1: Base image with system dependencies
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: CPU-only image (lightweight)
FROM base as cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install lux_depth_v2 dependencies (CVE-2024-27763 safe)
COPY lux_depth_v2/requirements-repo.txt ./lux_depth_v2/
RUN pip install --no-cache-dir -r lux_depth_v2/requirements-repo.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Verify basicsr is NOT installed (security check)
RUN python -c "import sys; import importlib.util; sys.exit(0 if importlib.util.find_spec('basicsr') is None else 1)" || \
    (echo "ERROR: basicsr found - CVE-2024-27763 vulnerability present" && exit 1)

ENV DEVICE=cpu
EXPOSE 8000

CMD ["python", "-m", "transformation_portal.cli"]

# Stage 3: GPU image (CUDA support)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as gpu

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install lux_depth_v2 dependencies (CVE-2024-27763 safe)
COPY lux_depth_v2/requirements-repo.txt ./lux_depth_v2/
RUN pip3 install --no-cache-dir -r lux_depth_v2/requirements-repo.txt

COPY . .
RUN pip3 install --no-cache-dir -e .

# Verify basicsr is NOT installed (security check)
RUN python3 -c "import sys; import importlib.util; sys.exit(0 if importlib.util.find_spec('basicsr') is None else 1)" || \
    (echo "ERROR: basicsr found - CVE-2024-27763 vulnerability present" && exit 1)

# Health check for GPU availability
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
  CMD python3 -c "import torch; assert torch.cuda.is_available()" || exit 1

ENV DEVICE=cuda
EXPOSE 8000

CMD ["python3", "-m", "transformation_portal.cli"]

# Stage 4: Apple Silicon optimized (M-series chips)
FROM base as apple-silicon

# Install Apple Silicon optimized dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CoreML tools
RUN pip install --no-cache-dir coremltools

# Install lux_depth_v2 dependencies (CVE-2024-27763 safe)
COPY lux_depth_v2/requirements-repo.txt ./lux_depth_v2/
RUN pip install --no-cache-dir -r lux_depth_v2/requirements-repo.txt

COPY . .
RUN pip install --no-cache-dir -e ".[ml]"

# Verify basicsr is NOT installed (security check)
RUN python -c "import sys; import importlib.util; sys.exit(0 if importlib.util.find_spec('basicsr') is None else 1)" || \
    (echo "ERROR: basicsr found - CVE-2024-27763 vulnerability present" && exit 1)

ENV DEVICE=mps
EXPOSE 8000

CMD ["python", "-m", "transformation_portal.cli"]

# Stage 5: Lux Depth V2 Production (security-hardened)
FROM python:3.11-slim as lux-depth-v2-production

# Security: Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies (CVE-2024-27763 safe)
COPY lux_depth_v2/requirements-repo.txt ./lux_depth_v2/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r lux_depth_v2/requirements-repo.txt

# Copy application code
COPY --chown=appuser:appuser lux_depth_v2/ ./lux_depth_v2/
COPY --chown=appuser:appuser config/ ./config/

# Create directories with correct permissions
RUN mkdir -p /data/input /data/output /app/logs && \
    chown -R appuser:appuser /data /app/logs

# Verify basicsr is NOT installed (security validation)
RUN python -c "import sys; import importlib.util; sys.exit(0 if importlib.util.find_spec('basicsr') is None else 1)" || \
    (echo "ERROR: basicsr found - CVE-2024-27763 vulnerability present" && exit 1)

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8088/health || exit 1

EXPOSE 8088

ENV PYTHONUNBUFFERED=1 \
    LUX_LOG_FORMAT=json \
    LUX_LOG_LEVEL=INFO \
    LUX_OUTPUT_DIR=/data/output

CMD ["python", "-m", "lux_depth_v2.cli", "--output-dir", "/data/output", "--service", "--port", "8088"]
