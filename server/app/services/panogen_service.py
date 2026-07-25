import gc
import sys
from pathlib import Path
from types import SimpleNamespace

from ..config import JOBS_DIR, REPO_DIR
from ..job_manager import JobContext


def run_panorama(
    ctx: JobContext,
    job_id: str,
    mode: str,
    prompt: str,
    negative_prompt: str,
    image_path: str | None,
    fp8: bool,
    cache: bool,
    seed: int,
) -> dict:
    """Stage 1 of HunyuanWorld-1.0: text/image -> 360 equirectangular panorama.

    Wraps the repo's demo_panogen.py Demo classes (which load FLUX + the
    HunyuanWorld PanoDiT LoRA and already enable CPU offload + VAE tiling).
    fp8 toggles the AngelSlim FP8 attention+GeMM processors; cache toggles
    DeepCache -- both trade a little quality for VRAM/speed on a 24GB card.

    Loads the model fresh and releases GPU memory before returning so the next
    queued job starts clean (single GPU).
    """
    import torch

    # Avoid PyTorch's cuDNN scaled-dot-product-attention backend, whose graph
    # planning fails on some cuDNN builds with
    #   "cuDNN Frontend error: No execution plans support the graph."
    # Disabling it makes SDPA fall back to the flash / memory-efficient / math
    # kernels, which handle FLUX's attention fine.
    torch.backends.cuda.enable_cudnn_sdp(False)

    # The Demo classes live in demo_panogen.py at the repo root, and import the
    # repo's hy3dworld package -- make sure the repo root is importable.
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    from demo_panogen import Image2PanoramaDemo, Text2PanoramaDemo

    out_dir = JOBS_DIR / job_id / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The Demo classes only read fp8_attention / fp8_gemm / cache off args.
    args = SimpleNamespace(fp8_attention=fp8, fp8_gemm=fp8, cache=cache)

    demo = None
    try:
        ctx.set_stage("loading_model", progress=5)
        ctx.log(f"Loading HunyuanWorld PanoDiT ({mode}, fp8={fp8}, cache={cache})")
        if mode == "i2s":
            if not image_path:
                raise ValueError("image-to-panorama job is missing its input image")
            demo = Image2PanoramaDemo(args)
            ctx.set_stage("generating_panorama", progress=30)
            ctx.log(f"Generating panorama from image {Path(image_path).name!r}")
            demo.run(prompt, negative_prompt, image_path, seed, str(out_dir))
        else:
            demo = Text2PanoramaDemo(args)
            ctx.set_stage("generating_panorama", progress=30)
            ctx.log(f"Generating panorama from prompt {prompt!r}")
            demo.run(prompt, negative_prompt, seed, str(out_dir))

        pano_path = out_dir / "panorama.png"
        if not pano_path.exists():
            raise RuntimeError("panorama.png was not produced")
        ctx.set_stage("saving_results", progress=90)
        ctx.set_artifact("panorama.png", str(pano_path))
        ctx.log(f"Wrote panorama to {pano_path}")
        return {"panorama.png": str(pano_path), "output_dir": str(out_dir)}
    finally:
        if demo is not None:
            del demo
        gc.collect()
        torch.cuda.empty_cache()
        ctx.log("Released PanoDiT model from VRAM")
