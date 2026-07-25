"""Background-thread HTTP work + a main-thread queue drain.

Blender's API is not thread-safe: worker threads never touch bpy directly.
They push plain-dict messages onto a queue; a bpy.app.timers callback drains
it on the main thread and applies results to scene properties, keeping the UI
responsive during long jobs.
"""
import os
import queue
import threading
import time

import bpy

from . import http_client

_MESSAGE_QUEUE: "queue.Queue[dict]" = queue.Queue()
_POLL_INTERVAL_SECONDS = 1.5
_DRAIN_INTERVAL_SECONDS = 0.3
_PANO_IMAGE_NAME = "hyworld_pano_preview"
_MAX_CONSECUTIVE_POLL_FAILURES = 10


def _post(message):
    _MESSAGE_QUEUE.put(message)


def _redraw_ui():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _drain_queue():
    job = bpy.context.scene.hyworld_job
    changed = False
    while True:
        try:
            msg = _MESSAGE_QUEUE.get_nowait()
        except queue.Empty:
            break
        changed = True
        kind = msg.get("type")
        if kind == "status":
            job.job_id = msg.get("job_id", job.job_id)
            job.status = msg.get("status", job.status)
            job.stage = msg.get("stage", job.stage)
            job.progress = msg.get("progress", job.progress)
            job.error = msg.get("error") or ""
            if msg.get("status") in ("done", "failed"):
                job.is_busy = False
        elif kind == "fatal_error":
            job.status = "failed"
            job.error = msg.get("message", "unknown error")
            job.is_busy = False
        elif kind == "artifact_ready":
            callback = msg.get("callback")
            if callback:
                try:
                    callback(msg.get("path"), msg.get("error"))
                except Exception as exc:  # noqa: BLE001
                    job.error = f"Import failed: {exc!r}"
    if changed:
        _redraw_ui()
    return _DRAIN_INTERVAL_SECONDS


def ensure_timer_running():
    if not bpy.app.timers.is_registered(_drain_queue):
        bpy.app.timers.register(_drain_queue, persistent=True)


def stop_timer():
    if bpy.app.timers.is_registered(_drain_queue):
        bpy.app.timers.unregister(_drain_queue)


def submit_and_poll(server_url, submit_fn):
    def _worker():
        try:
            _post({"type": "status", "status": "uploading", "stage": "submitting", "progress": 0.0})
            job_id = submit_fn(server_url)
        except http_client.ServerError as exc:
            _post({"type": "fatal_error", "message": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            _post({"type": "fatal_error", "message": f"Unexpected error: {exc!r}"})
            return
        _post({"type": "status", "job_id": job_id, "status": "queued", "stage": "queued", "progress": 0.0})
        _poll_until_terminal(server_url, job_id)

    threading.Thread(target=_worker, daemon=True).start()


def _poll_until_terminal(server_url, job_id):
    consecutive_failures = 0
    pano_fetched = False
    while True:
        try:
            info = http_client.get_job(server_url, job_id)
            consecutive_failures = 0
        except http_client.ServerError as exc:
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_POLL_FAILURES:
                _post({"type": "fatal_error",
                       "message": f"Lost contact with the server (job may still be running): {exc}"})
                return
            time.sleep(_POLL_INTERVAL_SECONDS)
            continue

        artifacts = info.get("artifacts") or {}
        _post({
            "type": "status", "job_id": job_id,
            "status": info.get("status", "running"), "stage": info.get("stage", ""),
            "progress": info.get("progress", 0.0), "error": info.get("error"),
        })
        if not pano_fetched and "panorama.png" in artifacts:
            pano_fetched = True
            dest = os.path.join(bpy.app.tempdir, "hyworld_downloads", job_id, "panorama.png")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            download_artifact_async(server_url, job_id, "panorama.png", dest,
                                    _load_pano_preview, fatal_on_error=False)
        if info.get("status") in ("done", "failed"):
            return
        time.sleep(_POLL_INTERVAL_SECONDS)


def _load_pano_preview(path, error):
    """Main-thread: load the downloaded panorama as a preview data-block."""
    if error or not path:
        return
    old = bpy.data.images.get(_PANO_IMAGE_NAME)
    if old is not None:
        try:
            bpy.data.images.remove(old)
        except Exception:  # noqa: BLE001
            pass
    img = bpy.data.images.load(path)
    img.name = _PANO_IMAGE_NAME
    img.preview_ensure()
    bpy.context.scene.hyworld_job.pano_image_name = img.name


def download_artifact_async(server_url, job_id, name, dest_path, on_done, fatal_on_error=True):
    """on_done(path_or_none, error_or_none) runs on the main thread via the drain.
    fatal_on_error=False reports download failures via on_done instead of failing the job."""

    def _worker():
        try:
            path = http_client.download_artifact(server_url, job_id, name, dest_path)
            _post({"type": "artifact_ready", "name": name, "path": path, "error": None, "callback": on_done})
        except http_client.ServerError as exc:
            if fatal_on_error:
                _post({"type": "fatal_error", "message": str(exc)})
            else:
                _post({"type": "artifact_ready", "name": name, "path": None, "error": str(exc), "callback": on_done})

    threading.Thread(target=_worker, daemon=True).start()
