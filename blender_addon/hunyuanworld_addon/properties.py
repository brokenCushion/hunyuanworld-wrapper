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


classes = (HYWORLD_JobState,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.hyworld_job = bpy.props.PointerProperty(type=HYWORLD_JobState)


def unregister():
    del bpy.types.Scene.hyworld_job
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
