# Qwen3-ASR (vLLM backend) FastAPI server.
# Built and started via compose.yaml; see README.md.

ARG CUDA_VERSION=12.8.0
# "runtime" keeps the image small: torch and vLLM ship their own CUDA kernels and libraries.
# "devel" adds the CUDA toolkit (nvcc), which is only needed to compile flash-attn (BUNDLE_FLASH_ATTENTION=true).
ARG CUDA_FLAVOR=runtime
ARG from=nvidia/cuda:${CUDA_VERSION}-${CUDA_FLAVOR}-ubuntu22.04
FROM ${from} AS base

ARG DEBIAN_FRONTEND=noninteractive
# gcc + python3-dev: Triton (used by vLLM's torch.compile) builds a small C launcher at runtime
RUN <<EOF
apt update -y && apt upgrade -y && apt install -y --no-install-recommends  \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    libc6-dev \
    libsndfile1 \
    ffmpeg \
    ca-certificates \
&& rm -rf /var/lib/apt/lists/*
EOF

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -U pip setuptools wheel

# A distro-installed python3-blinker (no pip metadata) makes pip fail when a dependency upgrades it.
# The runtime base doesn't ship it, but derived bases (e.g. with software-properties-common) may.
RUN if dpkg -s python3-blinker >/dev/null 2>&1; then apt-get remove -y python3-blinker; fi

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -U "qwen-asr[vllm]" fastapi uvicorn python-multipart requests soundfile scipy websockets psutil \
        opencc-python-reimplemented

# flash-attn only speeds up the forced aligner (Transformers backend); vLLM brings its own attention
# kernels. Off by default: without a matching prebuilt wheel it compiles from source, which needs
# CUDA_FLAVOR=devel and takes a long time (MAX_JOBS limits parallel compile jobs).
ARG BUNDLE_FLASH_ATTENTION=false
ARG MAX_JOBS=8
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then \
        if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then \
            echo "BUNDLE_FLASH_ATTENTION=true needs the CUDA toolkit: build with CUDA_FLAVOR=devel" >&2; exit 1; \
        fi; \
        apt update -y && apt install -y --no-install-recommends g++ ninja-build && rm -rf /var/lib/apt/lists/* \
        && MAX_JOBS=${MAX_JOBS} NVCC_THREADS=2 pip3 install -U flash-attn --no-build-isolation; \
    fi

COPY server.py /app/server.py
COPY engines /app/engines

# FlashInfer's sampling kernels may be JIT-compiled with nvcc, which the runtime image lacks.
# ASR decodes greedily, so vLLM's own sampler is all that's needed.
ENV ASR_BACKEND=vllm \
    VLLM_USE_FLASHINFER_SAMPLER=0

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
