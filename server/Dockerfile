# HunyuanWorld-1.0 self-contained backend -- STAGE 1 (panorama generation).
#
# Wraps text/image -> 360 panorama (FLUX + HunyuanWorld PanoDiT LoRA) behind an
# HTTP API, self-contained in Docker so install/uninstall never touches the
# host OS. Stage 2 (panorama -> layered 3D meshes: ZIM, GroundingDINO,
# Real-ESRGAN) is added in a later revision of this file.
#
# Requires a HuggingFace token at RUNTIME (not build time) for the gated
# FLUX.1-dev download -- passed via docker-compose env_file (.env), never baked
# into the image. Weights land in the /data/hf-cache bind mount.
#
# CUDA 12.4 / Python 3.10 / torch 2.5.0 -- matching Tencent's tested env.
# NOTE: stage 1 needs NO from-source CUDA builds (no flash-attn/pytorch3d) --
# those are stage-2 deps -- so this image builds cleanly and quickly.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev \
        git curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/local/bin/pip /usr/local/bin/pip3

# Lock torch to the CUDA-12.4-matched build across every pip install, so a
# transitive dep can never swap in a differently-compiled torch (the class of
# failure that plagued the HY-World 2.0 build).
RUN mkdir -p /etc/pip \
    && printf 'torch==2.5.0\ntorchvision==0.20.0\n' > /etc/pip/constraints.txt
ENV PIP_CONSTRAINT=/etc/pip/constraints.txt \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124

WORKDIR /app

RUN git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git /app/hunyuanworld-src

RUN pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124 \
    && python -c "import torch; print('=== TORCH', torch.__version__, 'CUDA', torch.version.cuda, '===')"

COPY requirements-panogen.txt /app/requirements-panogen.txt
RUN pip install -r /app/requirements-panogen.txt \
    && python -c "import torch; print('=== TORCH', torch.__version__, 'CUDA', torch.version.cuda, '===')"

COPY requirements-server.txt /app/requirements-server.txt
RUN pip install -r /app/requirements-server.txt

# Stage-1 isolation: the repo's hy3dworld package __init__ files eagerly import
# the scene-generation modules (WorldComposer, LayerDecomposition), which pull
# in the heavy stage-2 dependency chain (Real-ESRGAN, ZIM, GroundingDINO, MoGe,
# open3d, trimesh, ...). Replace the three package __init__ files with minimal
# versions exposing ONLY the panorama pipelines + Perspective, so stage 1 runs
# on the light deps alone. (Stage 2 restores the full package and adds the rest.)
# Placed after the pip installs so those cached layers aren't invalidated.
RUN printf 'from .pano_generator import Image2PanoramaPipelines, Text2PanoramaPipelines\n' \
      > /app/hunyuanworld-src/hy3dworld/models/__init__.py \
 && printf 'from .perspective_utils import Perspective\n' \
      > /app/hunyuanworld-src/hy3dworld/utils/__init__.py \
 && printf 'from .models import Image2PanoramaPipelines, Text2PanoramaPipelines\nfrom .utils import Perspective\n' \
      > /app/hunyuanworld-src/hy3dworld/__init__.py

ENV PYTHONPATH=/app/hunyuanworld-src:${PYTHONPATH} \
    HYWORLD_REPO_DIR=/app/hunyuanworld-src \
    HYWORLD_JOBS_DIR=/data/jobs \
    HYWORLD_HF_CACHE_DIR=/data/hf-cache \
    HF_HOME=/data/hf-cache

COPY app /app/server_app

EXPOSE 8000
CMD ["uvicorn", "server_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
