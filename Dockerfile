# Transformation Portal - Docker Image
# Multi-stage build for optimized image size
#
# Runtime user contract (I-3 hardening, 2026-05-18):
#   All runtime stages drop to an unprivileged user `tp` (UID/GID 10001
#   by default; overridable via --build-arg TP_UID=... / TP_GID=...).
#   Installs run as root; `USER tp` is set as the final directive of
#   each final runtime stage so the container's default execution identity
#   is unprivileged. The `gpu` stage does not inherit from
#   `python-runtime-base` (different base image) and therefore re-creates
#   the same user.

# Stage 1: Python runtime base with system dependencies
FROM python:3.11-slim-trixie AS python-runtime-base

ARG TP_UID=10001
ARG TP_GID=10001

# Install runtime system dependencies. Build tooling is installed only in
# builder stages so final runtime images do not carry compiler toolchains.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
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

# Stage 2: Python builder with compiler tooling
FROM python-runtime-base AS python-build

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv ${VIRTUAL_ENV}
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
COPY requirements/ requirements/
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -m pip install --no-cache-dir -e .

# Stage 3: CPU-only image (lightweight)
FROM python-runtime-base AS cpu

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY --from=python-build --chown=tp:tp /opt/venv /opt/venv
COPY --from=python-build --chown=tp:tp /app /app

ENV DEVICE=cpu
EXPOSE 8000

# Health probe hits the FastAPI orchestrator's /healthz endpoint (app.py).
# Uses Python stdlib so we don't have to install curl in the slim base.
# start-period gives uvicorn time to bind before failures count.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()" || exit 1

USER tp
CMD ["python", "-m", "transformation_portal.cli"]

# Stage 4: GPU runtime base (CUDA support)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS gpu-runtime-base

ARG TP_UID=10001
ARG TP_GID=10001

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Recreate the same unprivileged identity as the Python runtime base. The
# gpu stages use a different upstream image, so this is duplicated
# intentionally — keep the UID/GID/home in sync with the Python stage above.
RUN groupadd --gid ${TP_GID} tp \
    && useradd --uid ${TP_UID} --gid ${TP_GID} --create-home --shell /usr/sbin/nologin tp \
    && mkdir -p /app/input /app/output /app/config /home/tp/.transformation_portal \
    && chown -R tp:tp /app /home/tp

ENV HOME=/home/tp

WORKDIR /app

# Stage 5: GPU builder with compiler tooling
FROM gpu-runtime-base AS gpu-build

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    build-essential \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv ${VIRTUAL_ENV}
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
COPY requirements/ requirements/
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt
RUN python3.11 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY . .
RUN python3.11 -m pip install --no-cache-dir -e .

# Stage 6: GPU image (CUDA support)
FROM gpu-runtime-base AS gpu

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY --from=gpu-build --chown=tp:tp /opt/venv /opt/venv
COPY --from=gpu-build --chown=tp:tp /app /app

ENV DEVICE=cuda
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python3.11 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()" || exit 1

USER tp
CMD ["python3.11", "-m", "transformation_portal.cli"]

# Stage 7: Apple Silicon builder (M-series chips)
FROM python-build AS apple-silicon-build

# Install Apple Silicon optimized dependencies
RUN python -m pip install --no-cache-dir coremltools
RUN python -m pip install --no-cache-dir -e ".[ml]"

# Stage 8: Apple Silicon optimized (M-series chips)
FROM python-runtime-base AS apple-silicon

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY --from=apple-silicon-build --chown=tp:tp /opt/venv /opt/venv
COPY --from=apple-silicon-build --chown=tp:tp /app /app

ENV DEVICE=mps
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()" || exit 1

USER tp
CMD ["python", "-m", "transformation_portal.cli"]
