import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    kind: str
    status: JobStatus = JobStatus.QUEUED
    stage: str = ""
    progress: float = 0.0
    log: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "stage": self.stage,
            "progress": self.progress,
            "log": self.log[-200:],
            "artifacts": self.artifacts,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobContext:
    """Passed into job functions so they can report progress back to the manager."""

    def __init__(self, manager: "JobManager", job_id: str):
        self._manager = manager
        self._job_id = job_id

    def set_stage(self, stage: str, progress: float | None = None) -> None:
        self._manager._update(self._job_id, stage=stage, progress=progress)

    def set_progress(self, progress: float) -> None:
        self._manager._update(self._job_id, progress=progress)

    def log(self, line: str) -> None:
        self._manager._append_log(self._job_id, line)

    def set_artifact(self, name: str, path: str) -> None:
        self._manager._set_artifact(self._job_id, name, path)


class JobManager:
    """Serializes all GPU work through a single worker thread.

    There is exactly one GPU on the box, so only one job may hold it at a
    time. Job functions load their own model(s) and release GPU memory before
    returning, so the next queued job starts from a clean VRAM state.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def new_job_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def submit(self, kind: str, fn: Callable[..., dict], *args, job_id: str | None = None, **kwargs) -> str:
        job_id = job_id or self.new_job_id()
        job = Job(id=job_id, kind=kind)
        with self._lock:
            self._jobs[job_id] = job
        ctx = JobContext(self, job_id)
        self._executor.submit(self._run, job_id, fn, ctx, args, kwargs)
        return job_id

    def _run(self, job_id: str, fn: Callable, ctx: JobContext, args: tuple, kwargs: dict) -> None:
        self._update(job_id, status=JobStatus.RUNNING, stage="starting", progress=0.0)
        try:
            result = fn(ctx, *args, **kwargs)
            artifacts = result or {}
            with self._lock:
                job = self._jobs[job_id]
                job.artifacts.update(artifacts)
            self._update(job_id, status=JobStatus.DONE, stage="done", progress=100.0)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client
            self._append_log(job_id, f"ERROR: {exc!r}")
            self._update(job_id, status=JobStatus.FAILED, error=str(exc))

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                if value is not None:
                    setattr(job, key, value)
            job.updated_at = time.time()

    def _append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.log.append(line)
            job.updated_at = time.time()

    def _set_artifact(self, job_id: str, name: str, path: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.artifacts[name] = path
            job.updated_at = time.time()


job_manager = JobManager()
