import os

import bpy

from . import background, http_client
from .preferences import get_server_url

PANO_ARTIFACT = "panorama.png"


def _start_job(context, kind, submit_fn):
    job = context.scene.hyworld_job
    if job.pano_image_name:
        old = bpy.data.images.get(job.pano_image_name)
        if old is not None:
            try:
                bpy.data.images.remove(old)
            except Exception:  # noqa: BLE001
                pass
        job.pano_image_name = ""
    job.job_kind = kind
    job.status = "uploading"
    job.stage = "submitting"
    job.progress = 0.0
    job.error = ""
    job.is_busy = True
    background.ensure_timer_running()
    background.submit_and_poll(get_server_url(context), submit_fn)


class HYWORLD_OT_generate_text(bpy.types.Operator):
    bl_idname = "hyworld.generate_text"
    bl_label = "Generate from Text"
    bl_description = "Send the prompt to the server and generate a 360 panorama"

    def execute(self, context):
        job = context.scene.hyworld_job
        if not job.prompt.strip():
            self.report({"WARNING"}, "Enter a prompt first")
            return {"CANCELLED"}
        if job.is_busy:
            self.report({"WARNING"}, "A job is already running")
            return {"CANCELLED"}
        p, n, fp8, cache, seed = job.prompt, job.negative_prompt, job.fp8, job.cache, job.seed

        def submit(url):
            return http_client.submit_text(url, p, n, fp8, cache, seed)

        _start_job(context, "t2s", submit)
        self.report({"INFO"}, "Text-to-panorama submitted")
        return {"FINISHED"}


class HYWORLD_OT_generate_image(bpy.types.Operator):
    bl_idname = "hyworld.generate_image"
    bl_label = "Generate from Image"
    bl_description = "Send the selected image to the server and generate a 360 panorama"

    def execute(self, context):
        job = context.scene.hyworld_job
        image_path = bpy.path.abspath(job.image_path)
        if not image_path or not os.path.isfile(image_path):
            self.report({"WARNING"}, "Pick a valid image file first")
            return {"CANCELLED"}
        if job.is_busy:
            self.report({"WARNING"}, "A job is already running")
            return {"CANCELLED"}
        p, n, fp8, cache, seed = job.prompt, job.negative_prompt, job.fp8, job.cache, job.seed

        def submit(url):
            return http_client.submit_image(url, image_path, p, n, fp8, cache, seed)

        _start_job(context, "i2s", submit)
        self.report({"INFO"}, "Image-to-panorama submitted")
        return {"FINISHED"}


def _download_pano(context, job, on_done):
    server_url = get_server_url(context)
    dest_dir = os.path.join(bpy.app.tempdir, "hyworld_downloads", job.job_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, PANO_ARTIFACT)
    job.importing = True
    background.download_artifact_async(server_url, job.job_id, PANO_ARTIFACT, dest, on_done)


class HYWORLD_OT_download_pano(bpy.types.Operator):
    bl_idname = "hyworld.download_pano"
    bl_label = "Download Panorama"
    bl_description = "Save the generated 360 panorama (panorama.png) to disk"

    def execute(self, context):
        job = context.scene.hyworld_job
        if job.status != "done" or not job.job_id:
            self.report({"WARNING"}, "No completed job to download from")
            return {"CANCELLED"}

        def _on_done(path, error):
            job.importing = False
            self.report({"ERROR"} if error else {"INFO"}, error or f"Saved to {path}")

        _download_pano(context, job, _on_done)
        self.report({"INFO"}, "Downloading panorama...")
        return {"FINISHED"}


class HYWORLD_OT_set_environment(bpy.types.Operator):
    bl_idname = "hyworld.set_environment"
    bl_label = "Set as 360 Environment"
    bl_description = "Download the panorama and set it as the world's equirectangular environment texture"

    def execute(self, context):
        job = context.scene.hyworld_job
        if job.status != "done" or not job.job_id:
            self.report({"WARNING"}, "No completed job to use")
            return {"CANCELLED"}

        def _on_done(path, error):
            job.importing = False
            if error:
                self.report({"ERROR"}, error)
                return
            try:
                _apply_world_environment(context, path)
            except Exception as exc:  # noqa: BLE001
                job.error = f"Set environment failed: {exc!r}"

        _download_pano(context, job, _on_done)
        self.report({"INFO"}, "Downloading panorama...")
        return {"FINISHED"}


def _apply_world_environment(context, image_path):
    """Set the panorama as the world's equirectangular environment texture."""
    scene = context.scene
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    img = bpy.data.images.load(image_path, check_existing=True)
    env.image = img
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


class HYWORLD_OT_check_health(bpy.types.Operator):
    bl_idname = "hyworld.check_health"
    bl_label = "Check Server"
    bl_description = "Ping the HunyuanWorld-1.0 server's /health endpoint"

    def execute(self, context):
        try:
            info = http_client.health(get_server_url(context))
        except http_client.ServerError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        gpu = info.get("gpu", {})
        if not gpu.get("available"):
            self.report({"WARNING"}, f"Server reachable but no GPU: {gpu}")
        elif not info.get("hf_token_present"):
            self.report({"WARNING"}, f"{gpu.get('name')} OK, but HuggingFace token missing (set server .env)")
        else:
            self.report({"INFO"}, f"OK - {gpu.get('name')} ({gpu.get('free_gb')}GB free), HF token present")
        return {"FINISHED"}


classes = (
    HYWORLD_OT_generate_text,
    HYWORLD_OT_generate_image,
    HYWORLD_OT_download_pano,
    HYWORLD_OT_set_environment,
    HYWORLD_OT_check_health,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
