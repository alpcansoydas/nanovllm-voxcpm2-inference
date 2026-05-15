ARG CUDA_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu22.04

FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

WORKDIR /app

# Runtime deps for audio libs. No build-essential / nvcc needed: flash-attn
# arrives as a prebuilt wheel (see requirements.txt).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      python3 \
      python3-pip \
      ffmpeg \
      libsndfile1 \
  && rm -rf /var/lib/apt/lists/* \
  && ln -sf /usr/bin/python3 /usr/local/bin/python \
  && python -m pip install --upgrade pip

ENV NANOVLLM_CACHE_DIR=/var/cache/nanovllm \
    HF_HOME=/var/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/var/cache/huggingface/hub
RUN useradd -m -u 10001 appuser \
 && mkdir -p "$NANOVLLM_CACHE_DIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" \
 && chown -R appuser:appuser "$NANOVLLM_CACHE_DIR" "$HF_HOME" \
 && chown -R appuser:appuser /app

USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

# 1) Install third-party deps (heaviest layer; cached on rebuilds while
#    requirements.txt is unchanged).
COPY --chown=appuser:appuser requirements.txt ./
RUN python -m pip install --user --no-cache-dir -r requirements.txt

# 2) Copy the rest of the source.
COPY --chown=appuser:appuser . .

# 3) Install the engine package (provides the `nanovllm_voxcpm` import).
#    --no-deps because requirements.txt already pinned everything.
RUN python -m pip install --user --no-cache-dir --no-deps .

# Run the FastAPI service directly from the source tree at /app/deployment.
# Keeping it on PYTHONPATH means demo.html (and any other static asset under
# deployment/app/static/) resolves correctly via Path(__file__).
WORKDIR /app/deployment
ENV PYTHONPATH=/app/deployment

EXPOSE 8000

# k8s probes can use /health and /ready.
CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
