import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .config import JOBS_DIR
from .job_manager import job_manager
from .services.pipeline import run_full

app = FastAPI(title="HunyuanWorld-1.0 wrapper", version="0.2.0")


def _labels(raw: str) -> list[str]:
    """Space/comma-separated foreground object labels -> list."""
    return [t for t in raw.replace(",", " ").split() if t]


def _hf_token_present() -> bool:
    for var in ("HUGGING_FACE_HUB_TOKEN", "HF_TOKEN"):
        if os.environ.get(var, "").strip():
            return True
    return False


@app.get("/health")
def health():
    gpu = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            gpu = {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "free_gb": round(free / (1024**3), 2),
                "total_gb": round(total / (1024**3), 2),
            }
    except Exception as exc:  # noqa: BLE001
        gpu = {"available": False, "error": str(exc)}
    running = [j.to_dict() for j in job_manager.list() if j.status.value == "running"]
    return {"gpu": gpu, "hf_token_present": _hf_token_present(), "running_jobs": running}


@app.get("/jobs")
def list_jobs():
    return [j.to_dict() for j in job_manager.list()]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.to_dict()


@app.get("/jobs/{job_id}/artifact/{name}")
def get_artifact(job_id: str, name: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    path = job.artifacts.get(name)
    if path is None:
        raise HTTPException(404, f"artifact '{name}' not found for job {job_id}")
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(410, f"artifact '{name}' no longer on disk")
    return FileResponse(file_path)


@app.post("/generate/text")
def generate_text(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    scene: bool = Form(True),
    labels_fg1: str = Form(""),
    labels_fg2: str = Form(""),
    classes: str = Form("outdoor"),
    fp8: bool = Form(True),
    cache: bool = Form(True),
    seed: int = Form(42),
):
    if not prompt.strip():
        raise HTTPException(400, "prompt is empty")
    job_id = job_manager.new_job_id()
    job_manager.submit(
        "t2s", run_full, job_id, "t2s", prompt, negative_prompt, None,
        _labels(labels_fg1), _labels(labels_fg2), classes, scene, fp8, cache, seed,
        job_id=job_id,
    )
    return {"job_id": job_id}


@app.post("/generate/image")
async def generate_image(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    scene: bool = Form(True),
    labels_fg1: str = Form(""),
    labels_fg2: str = Form(""),
    classes: str = Form("outdoor"),
    fp8: bool = Form(True),
    cache: bool = Form(True),
    seed: int = Form(42),
):
    job_id = job_manager.new_job_id()
    input_dir = JOBS_DIR / job_id / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    dest = input_dir / Path(file.filename or "input.png").name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    job_manager.submit(
        "i2s", run_full, job_id, "i2s", prompt, negative_prompt, str(dest),
        _labels(labels_fg1), _labels(labels_fg2), classes, scene, fp8, cache, seed,
        job_id=job_id,
    )
    return {"job_id": job_id}
