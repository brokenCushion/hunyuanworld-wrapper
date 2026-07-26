"""Minimal stdlib-only HTTP client for the HunyuanWorld-1.0 server."""
import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid


class ServerError(Exception):
    pass


def _request(url, data=None, headers=None, method="GET", timeout=10.0):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise ServerError(f"HTTP {exc.code} from server: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ServerError(f"Could not reach server at {url}: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        # Read timeouts surface as bare TimeoutError/socket.timeout -- normalize
        # so callers can treat a slow poll as transient, not a crash.
        raise ServerError(f"Request to {url} timed out or failed: {exc}") from exc
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def _encode_multipart(fields, file_field=None, filepath=None):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append((
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
        ).encode("utf-8") + str(value).encode("utf-8") + b"\r\n")
    if file_field and filepath:
        filename = os.path.basename(filepath)
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        with open(filepath, "rb") as f:
            file_bytes = f.read()
        parts.append((
            f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n'
        ).encode("utf-8") + file_bytes + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _post_job(server_url, path, body, content_type, timeout):
    headers = {"Content-Type": content_type, "Content-Length": str(len(body))}
    result = _request(f"{server_url.rstrip('/')}{path}", data=body, headers=headers, method="POST", timeout=timeout)
    job_id = result.get("job_id")
    if not job_id:
        raise ServerError(f"Server did not return a job_id: {result}")
    return job_id


def health(server_url):
    return _request(f"{server_url.rstrip('/')}/health", timeout=5.0)


def _scene_fields(opts: dict) -> dict:
    return {
        "negative_prompt": opts.get("negative_prompt") or "",
        "scene": str(bool(opts.get("scene", True))).lower(),
        "labels_fg1": opts.get("labels_fg1") or "",
        "labels_fg2": opts.get("labels_fg2") or "",
        "classes": opts.get("classes") or "outdoor",
        "fp8": str(bool(opts.get("fp8", True))).lower(),
        "cache": str(bool(opts.get("cache", True))).lower(),
        "seed": str(int(opts.get("seed", 42))),
        "steps": str(int(opts.get("steps", 50))),
        "guidance_scale": str(float(opts.get("guidance_scale", 30.0))),
        "pano_width": str(int(opts.get("pano_width", 1920))),
        "fov": str(float(opts.get("fov", 80.0))),
        "mesh_quality": opts.get("mesh_quality") or "high",
    }


def submit_text(server_url, prompt, opts):
    if not prompt.strip():
        raise ServerError("Prompt is empty")
    fields = {"prompt": prompt, **_scene_fields(opts)}
    body = urllib.parse.urlencode(fields).encode("utf-8")
    return _post_job(server_url, "/generate/text", body, "application/x-www-form-urlencoded", timeout=30.0)


def submit_image(server_url, image_path, prompt, opts):
    if not image_path or not os.path.isfile(image_path):
        raise ServerError(f"Image not found: {image_path}")
    fields = {"prompt": prompt or "", **_scene_fields(opts)}
    body, ctype = _encode_multipart(fields, file_field="file", filepath=image_path)
    return _post_job(server_url, "/generate/image", body, ctype, timeout=120.0)


def get_job(server_url, job_id):
    # Generous timeout: the single-process server can be briefly unresponsive
    # while a job holds the GIL loading a model.
    return _request(f"{server_url.rstrip('/')}/jobs/{job_id}", timeout=60.0)


def download_artifact(server_url, job_id, name, dest_path):
    url = f"{server_url.rstrip('/')}/jobs/{job_id}/artifact/{name}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=120.0) as resp, open(dest_path, "wb") as out:
            out.write(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise ServerError(f"HTTP {exc.code} downloading '{name}': {detail}") from exc
    except urllib.error.URLError as exc:
        raise ServerError(f"Could not reach server at {url}: {exc.reason}") from exc
    return dest_path
