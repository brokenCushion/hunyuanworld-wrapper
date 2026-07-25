bl_info = {
    "name": "HunyuanWorld 1.0",
    "author": "hunyuanworld-wrapper",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > HunyuanWorld",
    "description": "Client for a self-hosted HunyuanWorld-1.0 backend: text/image -> 360 world.",
    "category": "3D View",
}

import bpy

from . import background, operators, panel, preferences, properties


def register():
    bpy.utils.register_class(preferences.HYWORLD_AddonPreferences)
    properties.register()
    operators.register()
    panel.register()
    background.ensure_timer_running()


def unregister():
    background.stop_timer()
    panel.unregister()
    operators.unregister()
    properties.unregister()
    bpy.utils.unregister_class(preferences.HYWORLD_AddonPreferences)


if __name__ == "__main__":
    register()
