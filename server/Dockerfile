# HunyuanWorld-1.0 self-contained backend -- STAGES 1 + 2.
#
# Stage 1: text/image -> 360 panorama (FLUX + HunyuanWorld PanoDiT LoRA).
# Stage 2: panorama -> semantically layered 3D meshes (ZIM segmentation +
#          GroundingDINO + FLUX.1-Fill inpainting + MoGe depth + pytorch3d).
#
# Requires a HuggingFace token at RUNTIME for TWO gated models: FLUX.1-dev
# (panorama) and FLUX.1-Fill-dev (inpainting). Passed via docker-compose
# env_file (.env); weights land in the /data/hf-cache bind mount.
#
# CUDA 12.4 / Python 3.10 / torch 2.5.0 -- matching Tencent's tested env.
# The `devel` base provides nvcc for the one from-source CUDA build (pytorch3d).

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev \
        git curl wget ffmpeg libgl1 libglib2.0-0 \
        build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/local/bin/pip /usr/local/bin/pip3

# Lock torch to the CUDA-12.4-matched build across every pip install, so a
# transitive dep can never swap in a differently-compiled torch.
RUN mkdir -p /etc/pip \
    && printf 'torch==2.5.0\ntorchvision==0.20.0\n' > /etc/pip/constraints.txt
ENV PIP_CONSTRAINT=/etc/pip/constraints.txt \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124

# RTX 4090 (Ada = sm_89) for the pytorch3d from-source build. docker build sees
# no GPU, so the arch must be pinned or torch's autodetect crashes on an empty
# arch list. FORCE_CUDA builds the CUDA extension even without a visible device.
ENV TORCH_CUDA_ARCH_LIST="8.9+PTX" \
    FORCE_CUDA=1

WORKDIR /app

RUN git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git /app/hunyuanworld-src

RUN pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124 \
    && python -c "import torch; print('=== TORCH', torch.__version__, 'CUDA', torch.version.cuda, '===')"

# Stage 1 (panorama) deps.
COPY requirements-panogen.txt /app/requirements-panogen.txt
RUN pip install -r /app/requirements-panogen.txt

# Stage 2 (scene generation) deps.
# --ignore-installed: a dep here (open3d -> Flask) wants a newer blinker than
# the apt/distutils blinker 1.4 in the base image, which pip can't uninstall.
# --ignore-installed installs fresh copies without touching the apt one; torch
# is reinstalled as a side effect but PIP_CONSTRAINT holds it at 2.5.0+cu124.
COPY requirements-scenegen.txt /app/requirements-scenegen.txt
RUN pip install --ignore-installed -r /app/requirements-scenegen.txt \
    && python -c "import torch; print('=== TORCH', torch.__version__, 'CUDA', torch.version.cuda, '===')"

# Real-ESRGAN super-resolution. realesrgan's PyPI deps pull plain `basicsr`,
# which imports torchvision.transforms.functional_tensor (removed in tv 0.20);
# install basicsr-fixed first, then realesrgan --no-deps so it uses the fix.
RUN pip install basicsr-fixed facexlib==0.3.0 gfpgan==1.3.8 \
    && pip install --no-deps realesrgan==0.3.0

# MoGe panorama depth (+ utils3d it depends on).
RUN pip install utils3d "moge @ git+https://github.com/microsoft/MoGe.git"

# pytorch3d -- the only from-source CUDA build. --no-build-isolation so it
# compiles against the torch installed above rather than fetching its own.
RUN pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git" \
    && python -c "import pytorch3d; print('=== pytorch3d', pytorch3d.__version__, '===')"

# ZIM segmentation: cloned INTO the repo dir because layer_decomposer.py hard-codes
# the checkpoint path as "./ZIM/zim_vit_l_2092" relative to the repo root (which
# is the CWD the scene service runs in). Download the vit-l onnx checkpoints there.
RUN git clone https://github.com/naver-ai/ZIM.git /app/hunyuanworld-src/ZIM \
    && cd /app/hunyuanworld-src/ZIM && pip install -e . \
    && mkdir -p zim_vit_l_2092 \
    && curl -L -o zim_vit_l_2092/encoder.onnx \
        https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx \
    && curl -L -o zim_vit_l_2092/decoder.onnx \
        https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx

COPY requirements-server.txt /app/requirements-server.txt
RUN pip install -r /app/requirements-server.txt

# basicsr (pulled in by gfpgan, overriding basicsr-fixed) imports
# torchvision.transforms.functional_tensor, removed in torchvision 0.17+.
# Rewrite that import to the current location in every installed file that uses
# it (basicsr, facexlib, ...). Late step so the pytorch3d compile stays cached.
RUN grep -rl "torchvision.transforms.functional_tensor" /usr/local/lib/python3.10/dist-packages/ \
    | xargs -r sed -i 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/g'

# Pin huggingface-hub back to the version transformers 4.51.0 needs (>=0.30,<1.0).
# The --ignore-installed scenegen install pulled hub 1.24.0, which breaks the
# transformers import (GroundingDINO). 0.30.2 matches Tencent's tested env and
# works for every consumer here (transformers, diffusers, moge). Late pin so the
# expensive earlier layers (pytorch3d compile, ZIM) stay cached.
# utils3d: MoGe calls utils3d.numpy.image_uv, which only exists in the exact
# git commit MoGe pins (the PyPI `utils3d` is a different, renamed API and
# fails at runtime with "module 'utils3d.numpy' has no attribute 'image_uv'").
# Force MoGe's pinned commit. Late step so the pytorch3d compile stays cached.
RUN pip install --force-reinstall --no-deps \
    "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"

# easydict: an undeclared runtime dep of ZIM (its setup.py doesn't list it).
RUN pip install "huggingface-hub==0.30.2" easydict==1.9

# Sanity check: the full hy3dworld package (panorama + scene-gen) imports with
# all deps present. This catches a missing scene-gen dependency at BUILD time
# rather than at first generation.
RUN cd /app/hunyuanworld-src \
    && python -c "import hy3dworld; from hy3dworld import Text2PanoramaPipelines, LayerDecomposition, WorldComposer; print('=== hy3dworld OK ===')" \
    && python -c "import utils3d, utils3d.numpy as u3n, inspect; print('utils3d from:', inspect.getfile(utils3d)); print('uv-ish names:', [n for n in dir(u3n) if 'uv' in n.lower()][:10]); assert hasattr(u3n, 'image_uv'), 'utils3d is the wrong build: MoGe needs utils3d.numpy.image_uv'; print('=== utils3d API OK ===')"

ENV PYTHONPATH=/app/hunyuanworld-src:${PYTHONPATH} \
    HYWORLD_REPO_DIR=/app/hunyuanworld-src \
    HYWORLD_JOBS_DIR=/data/jobs \
    HYWORLD_HF_CACHE_DIR=/data/hf-cache \
    HF_HOME=/data/hf-cache

COPY app /app/server_app

EXPOSE 8000
CMD ["uvicorn", "server_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
