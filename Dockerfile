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

COPY . .
RUN pip install --no-cache-dir -e .

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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY . .
RUN pip3 install --no-cache-dir -e .

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

COPY . .
RUN pip install --no-cache-dir -e ".[ml]"

ENV DEVICE=mps
EXPOSE 8000

CMD ["python", "-m", "transformation_portal.cli"]
