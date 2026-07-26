import os

import bpy

_INPUT_IMAGE_NAME = "hyworld_input_preview"


def _on_image_path_update(self, context):
    path = bpy.path.abspath(self.image_path) if self.image_path else ""
    old = bpy.data.images.get(_INPUT_IMAGE_NAME)
    if old is not None:
        try:
            bpy.data.images.remove(old)
        except Exception:  # noqa: BLE001
            pass
    self.input_image_name = ""
    if not path or not os.path.isfile(path):
        return
    try:
        img = bpy.data.images.load(path)
        img.name = _INPUT_IMAGE_NAME
        img.preview_ensure()
        self.input_image_name = img.name
    except Exception:  # noqa: BLE001
        self.input_image_name = ""


class HYWORLD_JobState(bpy.types.PropertyGroup):
    # Inputs
    prompt: bpy.props.StringProperty(name="Prompt", default="")
    negative_prompt: bpy.props.StringProperty(name="Negative prompt", default="")
    image_path: bpy.props.StringProperty(
        name="Image", default="", subtype="FILE_PATH", update=_on_image_path_update,
    )
    input_image_name: bpy.props.StringProperty(default="")
    fp8: bpy.props.BoolProperty(
        name="FP8 (low VRAM)",
        description="FP8 quantization of attention + GeMM. Recommended on a 24GB card -- lower VRAM/faster",
        default=True,
    )
    cache: bpy.props.BoolProperty(
        name="Cache (faster)",
        description="DeepCache -- speeds up diffusion by caching steps (slight quality trade-off)",
        default=True,
    )
    seed: bpy.props.IntProperty(name="Seed", default=42, min=0)

    # Advanced generation params (stage 1)
    steps: bpy.props.IntProperty(
        name="Steps",
        description="Diffusion steps. Fewer = faster, more = finer detail (default 50)",
        default=50, min=10, max=100,
    )
    guidance_scale: bpy.props.FloatProperty(
        name="Guidance",
        description="How strictly the panorama follows the prompt (default 30)",
        default=30.0, min=1.0, max=50.0,
    )
    pano_width: bpy.props.IntProperty(
        name="Panorama width",
        description="Equirectangular width in px; height is half. Higher = more detail + VRAM (default 1920)",
        default=1920, min=1024, max=2560, step=128,
    )
    fov: bpy.props.FloatProperty(
        name="Input FOV",
        description="Image mode only: field of view your photo occupies in the 360 (default 80)",
        default=80.0, min=30.0, max=140.0,
    )

    # Scene generation (stage 2)
    mesh_quality: bpy.props.EnumProperty(
        name="Mesh quality",
        description="Detail of the generated meshes. Higher = denser geometry, more VRAM and time",
        items=[
            ("low", "Low", "1920px working res, lighter meshes -- fastest"),
            ("medium", "Medium", "2880px working res -- balanced"),
            ("high", "High", "3840px working res -- Tencent's default, most detail"),
        ],
        default="high",
    )
    scene: bpy.props.BoolProperty(
        name="Build 3D scene",
        description="Run stage 2 (panorama -> layered 3D meshes). Off = panorama only (much faster)",
        default=True,
    )
    classes_type: bpy.props.EnumProperty(
        name="Scene type",
        description="Indoor or outdoor scene (guides layering / sky handling)",
        items=[("outdoor", "Outdoor", ""), ("indoor", "Indoor", "")],
        default="outdoor",
    )
    labels_fg1: bpy.props.StringProperty(
        name="Foreground 1",
        description="Object labels to peel into the first foreground layer, space/comma separated (e.g. trees, stones)",
        default="",
    )
    labels_fg2: bpy.props.StringProperty(
        name="Foreground 2",
        description="Object labels for a second foreground layer (e.g. sculptures, flowers)",
        default="",
    )

    # Job status
    job_id: bpy.props.StringProperty(default="")
    job_kind: bpy.props.StringProperty(default="")  # t2s | i2s
    status: bpy.props.StringProperty(default="idle")
    stage: bpy.props.StringProperty(default="")
    progress: bpy.props.FloatProperty(default=0.0, min=0.0, max=100.0, subtype="PERCENTAGE")
    error: bpy.props.StringProperty(default="")
    is_busy: bpy.props.BoolProperty(default=False)
    importing: bpy.props.BoolProperty(default=False)
    pano_image_name: bpy.props.StringProperty(default="")
    mesh_layers: bpy.props.StringProperty(default="")  # comma-separated mesh_layer*.ply artifact names


classes = (HYWORLD_JobState,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.hyworld_job = bpy.props.PointerProperty(type=HYWORLD_JobState)


def unregister():
    del bpy.types.Scene.hyworld_job
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
