# .devcontainer/Dockerfile — uv‑based replacement for the previous Conda image
# -----------------------------------------------------------------------------
# CUDA + cuDNN base with drivers already installed --------------------------------
ARG CUDA_TAG=12.8.0              # <── single source of truth
FROM nvidia/cuda:${CUDA_TAG}-cudnn-devel-ubuntu22.04

# ---------- build-time ARGs ---------------------------------------------------
ARG PYTHON_VER=3.10
ARG ENV_NAME=docker_dev_template
ARG JAX_PREALLOCATE=true
ARG JAX_MEM_FRAC=0.95
ARG JAX_ALLOCATOR=platform
ARG JAX_PREALLOC_LIMIT=8589934592
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# 1) Core OS deps, build tools, & Python (system) -----------------------------
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        bash curl ca-certificates git procps htop util-linux build-essential \
        python3 python3-venv python3-pip python3-dev \
        autoconf automake libtool m4 cmake pkg-config \
        jags iproute2 net-tools lsof \
        && pkg-config --modversion jags \
        && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Node.js for VS Code remote extension host
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get update && apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# 2) Copy a *pinned* uv & uvx binary pair from the official distroless image --
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/

# ----------------------------------------------------------------------------
# 3) Create project dir & copy only the lock/manifest for best layer‑caching --
WORKDIR /app
COPY pyproject.toml uv.lock* ./

# ----------------------------------------------------------------------------
# 4) Create an in-project venv, install deps, then symlink into /workspace
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /workspace && \
    uv venv .venv --python "${PYTHON_VER}" --prompt "${ENV_NAME}" && \
    (uv sync --locked || (echo "⚠️  Lock drift detected – regenerating" \
        && uv lock --upgrade --quiet && uv sync)) && \
    ln -s /app/.venv /workspace/.venv

# Promote venv for all later layers ------------------------------------------------
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# ----------------------------------------------------------------------------
# 5) ---------- CUDA wheels -------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --pre --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    uv pip install --no-cache-dir \
        "jax[cuda12]==0.6.0" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# --- CUDA toolkit sanity check (robust for runtime *and* devel images) ------
RUN set -e; \
    # 1️⃣ First try: any cuda-<ver> folder?
    CUDA_REAL="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -n1 || true)"; \
    # 2️⃣ Fallback: flat layout shipped by some runtime images
    if [ -z "$CUDA_REAL" ] && [ -d /usr/local/cuda ]; then \
        CUDA_REAL="/usr/local/cuda"; \
    fi; \
    # 3️⃣ Bail if still empty
    if [ -z "$CUDA_REAL" ]; then \
        echo '❌  No CUDA toolkit folder found — aborting.' >&2; exit 1; \
    fi; \
    # 4️⃣ Refresh the canonical symlink only when needed
    if [ "$CUDA_REAL" != "/usr/local/cuda" ]; then \
        echo "🔧  Linking /usr/local/cuda -> $CUDA_REAL"; \
        ln -sfn "$CUDA_REAL" /usr/local/cuda; \
    fi; \
    echo "🟢  CUDA toolkit detected at $CUDA_REAL"

# ----------------------------------------------------------------------------
# 6) Install PyJAGS with the cstdint header work‑around -----------------------
RUN CPPFLAGS="-include cstdint" uv pip install --no-build-isolation pyjags==1.3.8

# ----------------------------------------------------------------------------
# 7) Copy *rest* of the project after deps → fast rebuild when code changes ---
COPY . /app

# ----------------------------------------------------------------------------
# 8) GPU‑tuning env vars (carried forward from Conda‑based image) -------------
ENV XLA_PYTHON_CLIENT_PREALLOCATE=${JAX_PREALLOCATE}
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=${JAX_MEM_FRAC}
ENV XLA_PYTHON_CLIENT_ALLOCATOR=${JAX_ALLOCATOR}
ENV JAX_PLATFORM_NAME=gpu
ENV XLA_FLAGS="--xla_force_host_platform_device_count=1"
ENV JAX_DISABLE_JIT=false
ENV JAX_ENABLE_X64=false
ENV TF_FORCE_GPU_ALLOW_GROWTH=false
ENV JAX_PREALLOCATION_SIZE_LIMIT_BYTES=${JAX_PREALLOC_LIMIT}

# ----------------------------------------------------------------------------
# 9) Library path so PyJAGS & CUDA libs resolve correctly ---------------------
ENV LD_LIBRARY_PATH="/app/.venv/lib:${LD_LIBRARY_PATH}"

# ----------------------------------------------------------------------------
# 10) Final working directory & default command ------------------------------
WORKDIR /workspace
CMD ["bash"]

# 11) Force login shells & VS Code terminals to land in /workspace
RUN echo 'cd /workspace' > /etc/profile.d/99-workspace-cd.sh

# 12) Force every IPython / Jupyter kernel to start in /workspace
RUN mkdir -p /root/.ipython/profile_default/startup && \
    printf "import os, sys\nos.chdir('/workspace')\nsys.path.append('/workspace')\n" \
      > /root/.ipython/profile_default/startup/00-cd-workspace.py

# 13) Auto-activate uv venv in every login shell
RUN echo '. /app/.venv/bin/activate' > /etc/profile.d/10-uv-activate.sh



