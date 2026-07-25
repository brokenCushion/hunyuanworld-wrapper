import bpy


class HYWORLD_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    server_url: bpy.props.StringProperty(
        name="Server URL",
        description="Base URL of the HunyuanWorld-1.0 backend, e.g. http://192.168.1.50:8000",
        default="http://localhost:8000",
    )

    def draw(self, context):
        self.layout.prop(self, "server_url")


def get_server_url(context) -> str:
    return context.preferences.addons[__package__].preferences.server_url
