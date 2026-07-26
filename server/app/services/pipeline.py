from ..job_manager import JobContext
from .panogen_service import run_panorama
from .scenegen_service import run_scene


def run_full(
    ctx: JobContext,
    job_id: str,
    mode: str,
    prompt: str,
    negative_prompt: str,
    image_path: str | None,
    labels_fg1: list[str],
    labels_fg2: list[str],
    classes: str,
    scene: bool,
    fp8: bool,
    cache: bool,
    seed: int,
    steps: int = 50,
    guidance_scale: float = 30.0,
    true_cfg_scale: float = -1.0,
    pano_width: int = 1920,
    fov: float = 80.0,
    mesh_quality: str = "high",
) -> dict:
    """Full HunyuanWorld-1.0 pipeline for one job.

    Stage 1 (panorama) always runs; its panorama.png is set as an artifact
    mid-job so the client can preview it while stage 2 works. If `scene` is
    True, stage 2 (panorama -> layered meshes) runs next. Each stage loads and
    frees its own models, so VRAM is clean between them (single GPU).
    """
    pano_result = run_panorama(
        ctx, job_id, mode, prompt, negative_prompt, image_path, fp8, cache, seed,
        steps=steps, guidance_scale=guidance_scale, true_cfg_scale=true_cfg_scale,
        pano_width=pano_width, fov=fov,
    )
    if not scene:
        return pano_result

    panorama_path = pano_result["panorama.png"]
    scene_result = run_scene(
        ctx, job_id, panorama_path, labels_fg1, labels_fg2, classes, fp8, cache, seed,
        mesh_quality=mesh_quality,
    )
    return {**pano_result, **scene_result}
