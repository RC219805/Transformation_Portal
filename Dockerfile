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

# Build the CPU interpreter on the runtime's Debian ABI. A clean source
# prefix prevents the upstream image's old installed/bundled bootstrap
# packages from surviving an application-venv upgrade.
FROM debian:trixie-slim@sha256:62bd423aeee912639ca8ee681c2b762cb3e25137e0a92b20be455898a9f034ad AS cpu-python-build

ENV PYTHON_VERSION=3.11.17
ENV PYTHON_SHA256=bfb74ad39efae27cda510f134ab408e00f9992c56851cfc0b1cdb5646da11599

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libbz2-dev \
    libffi-dev \
    libgdbm-compat-dev \
    libgdbm-dev \
    liblzma-dev \
    libncurses-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    uuid-dev \
    xz-utils \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Keep these verified wheels aligned with the separate Jammy GPU builder.
RUN mkdir -p /opt/python3.11/share/python-wheels \
    && curl --fail --silent --show-error --location \
        https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl \
        --output /opt/python3.11/share/python-wheels/pip-26.2.2-py3-none-any.whl \
    && echo "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70  /opt/python3.11/share/python-wheels/pip-26.2.2-py3-none-any.whl" | sha256sum --check --strict \
    && curl --fail --silent --show-error --location \
        https://files.pythonhosted.org/packages/5d/40/e1e72872c6354b306daef1703549e8e83b4d43cfea356311bf722a043752/setuptools-83.0.0-py3-none-any.whl \
        --output /opt/python3.11/share/python-wheels/setuptools-83.0.0-py3-none-any.whl \
    && echo "29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3  /opt/python3.11/share/python-wheels/setuptools-83.0.0-py3-none-any.whl" | sha256sum --check --strict

RUN curl --fail --silent --show-error --location \
        "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" \
        --output /tmp/python.tar.xz \
    && echo "${PYTHON_SHA256}  /tmp/python.tar.xz" | sha256sum --check --strict \
    && mkdir /tmp/python-source \
    && tar --extract --xz --file /tmp/python.tar.xz --directory /tmp/python-source --strip-components=1 \
    && cd /tmp/python-source \
    && LDFLAGS="-Wl,-rpath=/opt/python3.11/lib" ./configure \
        --enable-loadable-sqlite-extensions \
        --prefix=/opt/python3.11 \
        --enable-shared \
        --with-ensurepip=install \
        --with-wheel-pkg-dir=/opt/python3.11/share/python-wheels \
        --without-static-libpython \
    && make -j2 \
    && make altinstall \
    && /opt/python3.11/bin/python3.11 -c "import sys; assert sys.version_info[:2] == (3, 11) and sys.version_info.releaselevel == 'final', sys.version" \
    && ln -s python3.11 /opt/python3.11/bin/python3 \
    && ln -s python3 /opt/python3.11/bin/python \
    && ln -s pip3.11 /opt/python3.11/bin/pip3 \
    && ln -s pip3 /opt/python3.11/bin/pip \
    && rm -rf /opt/python3.11/lib/python3.11/ensurepip/_bundled \
    && cp -a /opt/python3.11 /opt/python3.11-runtime \
    && rm -rf /opt/python3.11-runtime/include /opt/python3.11-runtime/lib/pkgconfig /tmp/python-source /tmp/python.tar.xz

# Stage 1: Python runtime base with system dependencies
FROM debian:trixie-slim@sha256:62bd423aeee912639ca8ee681c2b762cb3e25137e0a92b20be455898a9f034ad AS python-runtime-base

ARG TP_UID=10001
ARG TP_GID=10001

# Install runtime system dependencies. Build tooling is installed only in
# builder stages so final runtime images do not carry compiler toolchains.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libbz2-1.0 \
    libffi8 \
    libgdbm-compat4t64 \
    libgdbm6t64 \
    libgl1 \
    libglib2.0-0t64 \
    liblzma5 \
    libncursesw6 \
    libreadline8t64 \
    libsm6 \
    libsqlite3-0 \
    libssl3t64 \
    libuuid1 \
    libxext6 \
    libxrender1 \
    netbase \
    tzdata \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=cpu-python-build /opt/python3.11-runtime/ /opt/python3.11/
ENV PATH="/opt/python3.11/bin:${PATH}"
ENV LANG=C.UTF-8

RUN python3.11 -c "import bz2, ctypes, curses, dbm.gnu, dbm.ndbm, lzma, readline, sqlite3, ssl, sys, uuid, venv, zlib; assert sys.version_info[:2] == (3, 11) and sys.version_info.releaselevel == 'final', sys.version; assert hasattr(sqlite3.Connection, 'enable_load_extension'); sqlite3.connect(':memory:').enable_load_extension(False)"

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

# Match the source-built interpreter without shipping development headers.
COPY --from=cpu-python-build /opt/python3.11/include/ /opt/python3.11/include/

RUN python -m venv ${VIRTUAL_ENV}
RUN python -m pip install --no-cache-dir --upgrade \
    pip==26.2.2 setuptools==83.0.0 wheel==0.48.0

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
    CMD python3.11 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()" || exit 1

USER tp
CMD ["python3.11", "-m", "transformation_portal.cli"]

# Jammy's python3.11 package is 3.11.0rc1. Build a final CPython release on
# the same Ubuntu ABI as the CUDA runtime, independently of the CUDA image.
FROM ubuntu:22.04 AS gpu-python-build

ENV PYTHON_VERSION=3.11.17
ENV PYTHON_SHA256=bfb74ad39efae27cda510f134ab408e00f9992c56851cfc0b1cdb5646da11599

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libbz2-dev \
    libffi-dev \
    libgdbm-compat-dev \
    libgdbm-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    uuid-dev \
    xz-utils \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Use patched, hash-verified bootstrap wheels rather than the historical
# versions bundled in CPython. The configured directory also serves venv.
RUN mkdir -p /opt/python3.11/share/python-wheels \
    && curl --fail --silent --show-error --location \
        https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl \
        --output /opt/python3.11/share/python-wheels/pip-26.2.2-py3-none-any.whl \
    && echo "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70  /opt/python3.11/share/python-wheels/pip-26.2.2-py3-none-any.whl" | sha256sum --check --strict \
    && curl --fail --silent --show-error --location \
        https://files.pythonhosted.org/packages/5d/40/e1e72872c6354b306daef1703549e8e83b4d43cfea356311bf722a043752/setuptools-83.0.0-py3-none-any.whl \
        --output /opt/python3.11/share/python-wheels/setuptools-83.0.0-py3-none-any.whl \
    && echo "29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3  /opt/python3.11/share/python-wheels/setuptools-83.0.0-py3-none-any.whl" | sha256sum --check --strict

RUN curl --fail --silent --show-error --location \
        "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" \
        --output /tmp/python.tar.xz \
    && echo "${PYTHON_SHA256}  /tmp/python.tar.xz" | sha256sum --check --strict \
    && mkdir /tmp/python-source \
    && tar --extract --xz --file /tmp/python.tar.xz --directory /tmp/python-source --strip-components=1 \
    && cd /tmp/python-source \
    && LDFLAGS="-Wl,-rpath=/opt/python3.11/lib" ./configure \
        --prefix=/opt/python3.11 \
        --enable-shared \
        --with-ensurepip=install \
        --with-wheel-pkg-dir=/opt/python3.11/share/python-wheels \
        --without-static-libpython \
    && make -j2 \
    && make altinstall \
    && /opt/python3.11/bin/python3.11 -c "import sys; assert sys.version_info[:2] == (3, 11) and sys.version_info.releaselevel == 'final', sys.version" \
    && rm -rf /opt/python3.11/lib/python3.11/ensurepip/_bundled \
    && cp -a /opt/python3.11 /opt/python3.11-runtime \
    && rm -rf /opt/python3.11-runtime/include /opt/python3.11-runtime/lib/pkgconfig /tmp/python-source /tmp/python.tar.xz

# Stage 4: GPU runtime base (CUDA support)
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04@sha256:63a18dd805367dacfb077aeced8384ab2fb569598ec5f5f5220c3f90a5c23650 AS gpu-runtime-base

ARG TP_UID=10001
ARG TP_GID=10001

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libbz2-1.0 \
    libffi8 \
    libgdbm-compat4 \
    libgdbm6 \
    libgl1 \
    liblzma5 \
    libncursesw6 \
    libreadline8 \
    libsqlite3-0 \
    libssl3 \
    libuuid1 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=gpu-python-build /opt/python3.11-runtime/ /opt/python3.11/
ENV PATH="/opt/python3.11/bin:${PATH}"

# Verify extension availability against the actual runtime libraries.
RUN python3.11 -c "import bz2, ctypes, curses, dbm.gnu, dbm.ndbm, lzma, readline, sqlite3, ssl, sys, uuid, venv, zlib; assert sys.version_info[:2] == (3, 11) and sys.version_info.releaselevel == 'final', sys.version"

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
    && rm -rf /var/lib/apt/lists/*

# Headers must match the compiled interpreter and remain builder-only.
COPY --from=gpu-python-build /opt/python3.11/include/ /opt/python3.11/include/

RUN python3.11 -m venv ${VIRTUAL_ENV}
RUN python3.11 -m pip install --no-cache-dir --upgrade \
    pip==26.2.2 setuptools==83.0.0 wheel==0.48.0

COPY requirements.txt .
COPY requirements/ requirements/
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt
RUN python3.11 -m pip install --no-cache-dir \
    torch==2.13.0+cu126 torchvision==0.28.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

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

# Stage 7: Apple Silicon compatibility target.
#
# Docker on Apple Silicon still runs a Linux container, so this target cannot
# consume the governed native Darwin arm64 ML lock or provide MPS/CoreML
# acceleration. Keep the historical target name as a CPU image alias and route
# native Apple Silicon ML installs through `make install-ml-core` or
# `scripts/bootstrap/install_ml_stack.sh --profile core-mps`.
FROM cpu AS apple-silicon

ENV DEVICE=cpu
