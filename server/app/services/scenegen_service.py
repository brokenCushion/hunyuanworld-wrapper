import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from ..config import JOBS_DIR, REPO_DIR
from ..job_manager import JobContext


def run_scene(
    ctx: JobContext,
    job_id: str,
    panorama_path: str,
    labels_fg1: list[str],
    labels_fg2: list[str],
    classes: str,
    fp8: bool,
    cache: bool,
    seed: int,
) -> dict:
    """Stage 2 of HunyuanWorld-1.0: panorama -> semantically layered 3D meshes.

    Wraps the repo's demo_scenegen.py HYworldDemo: it segments the panorama
    (ZIM + GroundingDINO), peels off foreground object layers by label and
    inpaints behind them (FLUX.1-Fill), estimates depth (MoGe) and reconstructs
    a mesh per layer (pytorch3d). Outputs mesh_layer0.ply, mesh_layer1.ply, ...

    Runs with CWD = repo root because layer_decomposer hard-codes the ZIM
    checkpoint path as "./ZIM/zim_vit_l_2092". Releases GPU memory before
    returning so the next queued job starts clean (single GPU).
    """
    import torch

    # Avoid the cuDNN SDP attention backend (same graph-planning failure as in
    # panogen) -- fall back to flash / mem-efficient / math kernels.
    torch.backends.cuda.enable_cudnn_sdp(False)

    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    from demo_scenegen import HYworldDemo

    out_dir = JOBS_DIR / job_id / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # fp8_attention stays off (needs sageattention); fp8_gemm is native torch fp8.
    args = SimpleNamespace(fp8_attention=False, fp8_gemm=fp8, cache=cache)

    prev_cwd = os.getcwd()
    demo = None
    try:
        ctx.set_stage("loading_scene_models", progress=62)
        ctx.log(f"Loading scene models (fp8_gemm={fp8}, classes={classes}, "
                f"fg1={labels_fg1}, fg2={labels_fg2})")
        os.chdir(str(REPO_DIR))  # so ./ZIM/zim_vit_l_2092 resolves
        demo = HYworldDemo(args, seed=seed)

        ctx.set_stage("building_layers", progress=70)
        ctx.log("Decomposing layers + reconstructing meshes")
        demo.run(
            image_path=panorama_path,
            labels_fg1=labels_fg1,
            labels_fg2=labels_fg2,
            classes=classes,
            output_dir=str(out_dir),
            export_drc=False,
        )

        ctx.set_stage("saving_results", progress=95)
        meshes = sorted(out_dir.glob("mesh_layer*.ply"))
        if not meshes:
            raise RuntimeError("no mesh_layer*.ply files were produced")
        artifacts: dict[str, str] = {}
        for m in meshes:
            artifacts[m.name] = str(m)
            ctx.set_artifact(m.name, str(m))
        ctx.log(f"Wrote {len(meshes)} layer mesh(es): {[m.name for m in meshes]}")
        return artifacts
    finally:
        os.chdir(prev_cwd)
        if demo is not None:
            del demo
        gc.collect()
        torch.cuda.empty_cache()
        ctx.log("Released scene models from VRAM")
