# Qwen3-ASR (vLLM backend) FastAPI server.
# Built and started via compose.yaml; see docs/deployment-vllm.md.

ARG CUDA_VERSION=12.8.0
# "runtime" keeps the image small: torch and vLLM ship their own CUDA kernels and libraries.
# "devel" adds the CUDA toolkit (nvcc), which is only needed to compile flash-attn (BUNDLE_FLASH_ATTENTION=true).
ARG CUDA_FLAVOR=runtime
ARG from=nvidia/cuda:${CUDA_VERSION}-${CUDA_FLAVOR}-ubuntu22.04
FROM ${from} AS base

ARG DEBIAN_FRONTEND=noninteractive
# gcc: Triton (used by vLLM's torch.compile) builds a small C launcher at runtime
# (the uv-managed Python below ships its own headers).
RUN <<EOF
apt update -y && apt install -y --no-install-recommends  \
    gcc \
    libc6-dev \
    libsndfile1 \
    ffmpeg \
    ca-certificates \
&& rm -rf /var/lib/apt/lists/*
EOF

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

# Ubuntu 22.04 ships Python 3.10; uv installs a standalone Python 3.12 and the locked dependencies
# (`--frozen`: fail instead of re-resolving if uv.lock is out of date).
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.12 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project --extra vllm

# flash-attn only speeds up the forced aligner (Transformers backend); vLLM brings its own attention
# kernels. Off by default and not part of uv.lock: it is compiled against the installed torch, which
# needs CUDA_FLAVOR=devel and takes a long time (MAX_JOBS limits parallel compile jobs).
ARG BUNDLE_FLASH_ATTENTION=false
ARG MAX_JOBS=8
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then \
        if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then \
            echo "BUNDLE_FLASH_ATTENTION=true needs the CUDA toolkit: build with CUDA_FLAVOR=devel" >&2; exit 1; \
        fi; \
        apt update -y && apt install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/* \
        && uv pip install --python /opt/venv/bin/python setuptools wheel ninja packaging \
        && MAX_JOBS=${MAX_JOBS} NVCC_THREADS=2 uv pip install --python /opt/venv/bin/python --no-build-isolation flash-attn; \
    fi

COPY LICENSE /app/LICENSE
COPY server.py /app/server.py
COPY asr_server /app/asr_server

# FlashInfer's sampling kernels may be JIT-compiled with nvcc, which the runtime image lacks.
# ASR decodes greedily, so vLLM's own sampler is all that's needed.
ENV ASR_BACKEND=vllm \
    VLLM_USE_FLASHINFER_SAMPLER=0

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
