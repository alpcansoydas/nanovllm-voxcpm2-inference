# ─── Build args ───────────────────────────────────────────────────────────────
# CUDA base: use -devel so we can compile flash-attn and other native extensions.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu22.04

# Target GPU architecture(s) for flash-attn / native CUDA extensions.
# Compiling for fewer arches dramatically reduces build time.
# Override at build time to match your hardware:
#
#   GPU family           | TORCH_CUDA_ARCH_LIST value
#   ---------------------|---------------------------
#   L4 / RTX 40xx (Ada) | 8.9          ← default (current target)
#   A100 / A30 (Ampere) | 8.0
#   A10 / A40 / RTX 30xx| 8.6
#   H100 / H200 (Hopper)| 9.0
#   T4 / RTX 20xx (Tur.)| 7.5
#   V100 (Volta)         | 7.0
#   Multi-GPU deployment | 8.0;8.6;8.9;9.0
#
# Example: docker build --build-arg TORCH_CUDA_ARCH_LIST="9.0" ...
ARG TORCH_CUDA_ARCH_LIST="8.9"

# ─── Base image ───────────────────────────────────────────────────────────────
FROM ${CUDA_IMAGE}

ARG TORCH_CUDA_ARCH_LIST

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    UV_NO_PROGRESS=1 \
    # HuggingFace Hub cache inside the container; mount /var/cache/nanovllm
    # to a host volume so models persist across container restarts.
    HF_HOME=/var/cache/nanovllm/hf

WORKDIR /app

# ─── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      build-essential \
      ninja-build \
      git \
      python3-dev \
      python3-pip \
      pkg-config \
      ffmpeg \
      libsndfile1-dev \
 && rm -rf /var/lib/apt/lists/*

# uv: fast Python package installer used by the workspace
RUN python3 -m pip install --no-cache-dir uv

# ─── Runtime environment ──────────────────────────────────────────────────────
ENV NANOVLLM_CACHE_DIR=/var/cache/nanovllm

# Default model location.  Mount your model directory to this path:
#   docker run -v /host/path/to/VoxCPM2:/models/VoxCPM2 ...
# Or set NANOVLLM_MODEL_PATH to a HuggingFace repo id (e.g. openbmb/VoxCPM2)
# and the server will download it on first start (requires internet + HF token).
ENV NANOVLLM_MODEL_PATH=/models/VoxCPM2

# Voice preset audio files bundled into the image.
ENV NANOVLLM_VOICE_PRESETS_DIR=/app/voice_presets

# ─── Non-root user ────────────────────────────────────────────────────────────
RUN useradd -m -u 10001 appuser \
 && mkdir -p "$NANOVLLM_CACHE_DIR" \
 && chown -R appuser:appuser "$NANOVLLM_CACHE_DIR" \
 && chown -R appuser:appuser /app

USER appuser

# ─── Copy sources ─────────────────────────────────────────────────────────────
# Build context must be the repo root (nanovllm-voxcpm2-inference/).
COPY --chown=appuser:appuser nanovllm-voxcpm-main/ ./nanovllm-voxcpm-main/
COPY --chown=appuser:appuser voice_presets/         ./voice_presets/

# ─── Install Python dependencies ──────────────────────────────────────────────
# flash-attn is compiled from source here; this step is slow (20-40 min)
# but produces a single self-contained image.
WORKDIR /app/nanovllm-voxcpm-main

RUN uv sync --all-packages --no-dev -v --compile-bytecode --no-cache --no-editable

# ─── Runtime ──────────────────────────────────────────────────────────────────
EXPOSE 8000

# Liveness / readiness probes: GET /health and GET /ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uv", "run", "--no-sync", \
     "fastapi", "run", "deployment/app/main.py", \
     "--host", "0.0.0.0", "--port", "8000"]
