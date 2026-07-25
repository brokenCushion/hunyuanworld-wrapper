# HunyuanWorld-1.0 server (self-contained, Docker)

Wraps [HunyuanWorld-1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) behind an HTTP API, built
as a self-contained Docker image so install/uninstall never touches the host OS.

**Stage 1 (this build): text/image → 360° equirectangular panorama.** Stage 2 (panorama → layered 3D
meshes) is added later.

## Requirements

- Docker + the NVIDIA Container Toolkit (one-time host setup).
- An NVIDIA GPU (developed against a single RTX 4090, 24GB).
- A HuggingFace token with access to the gated **FLUX.1-dev** model.

## One-time HuggingFace setup

1. Accept the license at <https://huggingface.co/black-forest-labs/FLUX.1-dev>.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. `cp .env.example .env` and paste your token into `HUGGING_FACE_HUB_TOKEN`.

`.env` is git-ignored and docker-ignored — the token is passed at runtime, never baked into the image.

## Install

```bash
docker compose up -d --build     # or .\install.ps1 on Windows
```

First run builds the image (clones HunyuanWorld-1.0, installs deps) and, on the first generation, downloads
FLUX.1-dev + the PanoDiT LoRA into `./data/hf-cache`. Note: **CUDA 12.4 / Python 3.10 / torch 2.5.0** here,
and stage 1 needs no from-source CUDA builds, so the image builds cleanly.

## Use

```bash
curl.exe http://localhost:8000/health
# -> { "gpu": {...}, "hf_token_present": true, ... }

curl.exe -F prompt="a snowy mountain village at dusk" http://localhost:8000/generate/text
# -> {"job_id": "..."}

curl.exe -F file=@photo.jpg -F prompt="optional" http://localhost:8000/generate/image

curl.exe http://localhost:8000/jobs/<job_id>
curl.exe -OJ http://localhost:8000/jobs/<job_id>/artifact/panorama.png
```

Form fields: `fp8` (default true — FP8 quantization, recommended on 24GB), `cache` (default true — DeepCache
speedup), `seed`, `negative_prompt`.

## Uninstall

```bash
docker compose down -v --rmi all   # or .\uninstall.ps1
```

Removes the container, image, and volumes. `./data/` (weights + outputs) is left on disk — delete it for a
full wipe.
