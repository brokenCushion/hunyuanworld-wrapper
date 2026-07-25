import os
from pathlib import Path

REPO_DIR = Path(os.environ.get("HYWORLD_REPO_DIR", "/app/hunyuanworld-src"))
JOBS_DIR = Path(os.environ.get("HYWORLD_JOBS_DIR", "/data/jobs"))
HF_CACHE_DIR = Path(os.environ.get("HYWORLD_HF_CACHE_DIR", "/data/hf-cache"))

JOBS_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
