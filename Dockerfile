# Transformation Portal - Docker Image
# Multi-stage build for optimized image size
#
# Runtime user contract (I-3 hardening, 2026-05-18):
#   All runtime stages drop to an unprivileged user `tp` (UID/GID 10001
#   by default; overridable via --build-arg TP_UID=... / TP_GID=...).
#   Installs run as root; `USER tp` is set as the final directive of
#   each runtime stage so the container's default execution identity
#   is unprivileged. The `gpu` stage does not inherit from `base`
#   (different base image) and therefore re-creates the same user.

# Stage 1: Base image with system dependencies
FROM python:3.11-slim as base

ARG TP_UID=10001
ARG TP_GID=10001

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create the unprivileged runtime identity used by all derived stages.
# /usr/sbin/nologin prevents interactive shell access if the container
# is exec'd into without an explicit shell override.
RUN groupadd --gid ${TP_GID} tp \
    && useradd --uid ${TP_UID} --gid ${TP_GID} --create-home --shell /usr/sbin/nologin tp \
    && mkdir -p /app/input /app/output /app/config /home/tp/.transformation_portal \
    && chown -R tp:tp /app /home/tp

ENV HOME=/home/tp

WORKDIR /app

# Stage 2: CPU-only image (lightweight)
FROM base as cpu

COPY requirements.txt .
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e . \
    && chown -R tp:tp /app

ENV DEVICE=cpu
EXPOSE 8000

# Health probe hits the FastAPI orchestrator's /healthz endpoint (app.py).
# Uses Python stdlib so we don't have to install curl in the slim base.
# start-period gives uvicorn time to bind before failures count.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()" || exit 1

USER tp
CMD ["python", "-m", "transformation_portal.cli"]

# Stage 3: GPU image (CUDA support)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as gpu

ARG TP_UID=10001
ARG TP_GID=10001

RUN apt-get update && apt-get install -y \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Recreate the same unprivileged identity as `base`. The gpu stage does
# not inherit from base (different upstream image), so this is duplicated
# intentionally — keep the UID/GID/home in sync with the base stage above.
RUN groupadd --gid ${TP_GID} tp \
    && useradd --uid ${TP_UID} --gid ${TP_GID} --create-home --shell /usr/sbin/nologin tp \
    && mkdir -p /app/input /app/output /app/config /home/tp/.transformation_portal \
    && chown -R tp:tp /app /home/tp

ENV HOME=/home/tp

WORKDIR /app

COPY requirements.txt .
COPY requirements/ requirements/
RUN pip3 install --no-cache-dir -r requirements.txt torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY . .
RUN pip3 install --no-cache-dir -e . \
    && chown -R tp:tp /app

ENV DEVICE=cuda
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()" || exit 1

USER tp
CMD ["python3", "-m", "transformation_portal.cli"]

# Stage 4: Apple Silicon optimized (M-series chips)
FROM base as apple-silicon

# Install Apple Silicon optimized dependencies
COPY requirements.txt .
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements.txt

# Install CoreML tools
RUN pip install --no-cache-dir coremltools

COPY . .
RUN pip install --no-cache-dir -e ".[ml]" \
    && chown -R tp:tp /app

ENV DEVICE=mps
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()" || exit 1

USER tp
CMD ["python", "-m", "transformation_portal.cli"]
