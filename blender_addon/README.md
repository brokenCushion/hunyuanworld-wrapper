# HunyuanWorld 1.0 Blender addon

Client for the self-hosted [server](../server). Sends a text prompt or image and shows the generated 360°
panorama — no ML dependencies inside Blender.

**Stage 1: panorama generation.** Stage 2 (import layered 3D meshes) is added later.

## Install

1. Zip the `hunyuanworld_addon/` folder.
2. Blender: Edit > Preferences > Add-ons > Install from Disk…, select the zip, enable **HunyuanWorld 1.0**.
3. Set **Server URL** in the addon preferences (e.g. `http://192.168.1.50:8000`).
4. **Restart Blender after any reinstall** — Blender caches addon code in memory.

## Use

Open the **HunyuanWorld** tab in the View3D sidebar (`N`):

1. **Server** — "Check Server" reports GPU + whether the HuggingFace token is set.
2. **Options** — `FP8` (default on, low-VRAM) · `Cache` (default on, faster) · `Seed`.
3. **Text to Panorama** — prompt (+ optional negative) → Generate.
   **Image to Panorama** — pick an image (shows a preview) → Generate.
4. Mid-job the **360 panorama** appears in the panel. When done:
   - **Set as 360 Environment** — loads it as the world's equirectangular background (immersive viewport).
   - **Download Panorama** — saves `panorama.png`.

All network I/O runs on background threads; the UI never freezes.
