import bpy


def _draw_image_preview(layout, context, image_name, max_scale):
    """Draw an image preview sized to (roughly) fill the panel width."""
    img = bpy.data.images.get(image_name)
    if img is None or img.preview is None:
        return
    region = getattr(context, "region", None)
    ui_scale = getattr(context.preferences.system, "ui_scale", 1.0) or 1.0
    width_px = region.width if region is not None else 200
    scale = max(4.0, min(max_scale, (width_px - 25) / (20.0 * ui_scale)))
    layout.template_icon(icon_value=img.preview.icon_id, scale=scale)


class HYWORLD_PT_main_panel(bpy.types.Panel):
    bl_label = "HunyuanWorld 1.0"
    bl_idname = "HYWORLD_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "HunyuanWorld"

    def draw(self, context):
        layout = self.layout
        job = context.scene.hyworld_job

        box = layout.box()
        box.label(text="Server", icon="URL")
        box.operator("hyworld.check_health", icon="PLUGIN")

        box = layout.box()
        box.label(text="Options", icon="PREFERENCES")
        row = box.row(align=True)
        row.prop(job, "fp8", toggle=True)
        row.prop(job, "cache", toggle=True)
        box.prop(job, "seed")

        box = layout.box()
        box.label(text="3D Scene", icon="SCENE_DATA")
        box.prop(job, "scene", toggle=True)
        sub = box.column()
        sub.enabled = job.scene
        sub.prop(job, "mesh_quality", text="Quality")
        sub.prop(job, "classes_type", text="")
        sub.prop(job, "labels_fg1")
        sub.prop(job, "labels_fg2")

        box = layout.box()
        box.label(text="Text to Scene", icon="OUTLINER_OB_FONT")
        box.prop(job, "prompt", text="")
        box.prop(job, "negative_prompt", text="Neg")
        row = box.row()
        row.enabled = bool(job.prompt.strip()) and not job.is_busy
        row.operator("hyworld.generate_text", icon="PLAY")

        box = layout.box()
        box.label(text="Image to Scene", icon="IMAGE_DATA")
        box.prop(job, "image_path", text="")
        if job.input_image_name:
            _draw_image_preview(box, context, job.input_image_name, max_scale=14.0)
        row = box.row()
        row.enabled = bool(job.image_path) and not job.is_busy
        row.operator("hyworld.generate_image", icon="PLAY")

        self._draw_job_status(layout, context, job)

    def _draw_job_status(self, layout, context, job):
        if not job.job_kind or job.status == "idle":
            return
        box = layout.box()
        kind = {"t2s": "Text", "i2s": "Image"}.get(job.job_kind, job.job_kind)
        box.label(text=f"Job ({kind}): {job.status}  ({job.stage})", icon="INFO")
        if job.status in ("uploading", "queued", "running"):
            box.label(text=f"Progress: {job.progress:.0f}%")
        if job.status == "failed" and job.error:
            box.label(text=f"Error: {job.error}", icon="ERROR")

        if job.pano_image_name:
            box.label(text="360 panorama:")
            _draw_image_preview(box, context, job.pano_image_name, max_scale=30.0)

        if job.status == "done":
            if job.importing:
                box.label(text="Downloading & importing...", icon="SORTTIME")
            col = box.column(align=True)
            col.enabled = not job.importing
            if job.mesh_layers:
                n = len([x for x in job.mesh_layers.split(",") if x])
                col.operator("hyworld.import_meshes", icon="IMPORT",
                             text=f"Import 3D Scene ({n} layer{'s' if n != 1 else ''})")
            col.operator("hyworld.set_environment", icon="WORLD")
            col.operator("hyworld.download_pano", icon="EXPORT")


class HYWORLD_PT_advanced_panel(bpy.types.Panel):
    """Collapsed-by-default sub-panel for the generation tuning knobs."""
    bl_label = "Advanced"
    bl_idname = "HYWORLD_PT_advanced_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "HunyuanWorld"
    bl_parent_id = "HYWORLD_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        job = context.scene.hyworld_job
        col = layout.column(align=True)
        col.prop(job, "steps")
        col.prop(job, "guidance_scale")
        col.prop(job, "pano_width")
        col.separator()
        row = col.row()
        row.enabled = bool(job.image_path)
        row.prop(job, "fov")


classes = (HYWORLD_PT_main_panel, HYWORLD_PT_advanced_panel)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
