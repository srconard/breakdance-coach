"""Blender script: render an animated GLB to a preview MP4 for quality review.

Runs INSIDE Blender headless:
    blender -b --python render_preview.py -- --input move.glb --output move.mp4
        [--fps 12] [--label "RAW"] [--width 640] [--height 800] [--engine eevee|workbench]

Adds a grid floor at z=0, simple studio lighting, and a 3/4 camera auto-framed
on the animated skeleton's bounding box, then renders the full frame range.
Used by the Phase 1 motion-cleanup validation (before/after comparisons) —
presentation inside the exported GLB itself is Phase 3.
"""

import math
import os
import sys

import bpy
import mathutils


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    opts = {
        "input": None, "output": None, "fps": None,
        "width": 640, "height": 800, "engine": "eevee", "label": None,
        "floor-z": 0.0,
    }
    i = 0
    while i < len(argv):
        key = argv[i].lstrip("-")
        if key in opts and i + 1 < len(argv):
            val = argv[i + 1]
            if key in ("fps", "width", "height"):
                opts[key] = int(val)
            elif key == "floor-z":
                opts[key] = float(val)
            else:
                opts[key] = val
            i += 2
        else:
            i += 1
    if not opts["input"] or not opts["output"]:
        print("Usage: blender -b --python render_preview.py -- --input in.glb --output out.mp4")
        sys.exit(1)
    return opts


def scene_animation_range():
    lo, hi = None, None
    for action in bpy.data.actions:
        fs, fe = action.frame_range
        lo = fs if lo is None else min(lo, fs)
        hi = fe if hi is None else max(hi, fe)
    if lo is None:
        return 1, 1
    return int(lo), int(math.ceil(hi))


def animated_bounds(armature, frame_start, frame_end, samples=24):
    """Bounding box of all pose-bone heads across the animation."""
    lo = mathutils.Vector((1e9, 1e9, 1e9))
    hi = mathutils.Vector((-1e9, -1e9, -1e9))
    step = max(1, (frame_end - frame_start) // samples)
    for f in range(frame_start, frame_end + 1, step):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        mw = armature.matrix_world
        for pb in armature.pose.bones:
            p = mw @ pb.head
            lo = mathutils.Vector(map(min, lo, p))
            hi = mathutils.Vector(map(max, hi, p))
    return lo, hi


def build_floor(z=0.0):
    """Light gray floor with a subtle 0.5 m checker grid."""
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, z))
    floor = bpy.context.active_object
    floor.name = "Floor"
    mat = bpy.data.materials.new("FloorMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    checker = nt.nodes.new("ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = (0.85, 0.85, 0.86, 1.0)
    checker.inputs["Color2"].default_value = (0.78, 0.78, 0.80, 1.0)
    checker.inputs["Scale"].default_value = 2.0  # 0.5 m squares on the 30 m plane? scale is per-UV
    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (30.0, 30.0, 1.0)  # 1 checker = 0.5 m
    nt.links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    nt.links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9
    floor.data.materials.append(mat)
    return floor


def build_lights():
    # Key: sun with soft shadows
    bpy.ops.object.light_add(type="SUN", location=(4, -4, 8))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.data.angle = math.radians(10)
    sun.rotation_euler = (math.radians(35), math.radians(15), math.radians(30))
    # Fill: large soft area light
    bpy.ops.object.light_add(type="AREA", location=(-4, -3, 4))
    fill = bpy.context.active_object
    fill.data.energy = 150.0
    fill.data.size = 6.0
    fill.rotation_euler = (math.radians(55), 0, math.radians(-40))


def build_camera(lo, hi, width, height):
    center = (lo + hi) / 2
    extent = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 1.0)
    dist = extent * 2.1
    azim = math.radians(-35)  # 3/4 front view
    elev = math.radians(12)
    cam_pos = center + mathutils.Vector((
        dist * math.cos(elev) * math.sin(azim),
        -dist * math.cos(elev) * math.cos(azim),
        dist * math.sin(elev),
    ))
    cam_pos.z = max(cam_pos.z, 0.4)
    bpy.ops.object.camera_add(location=cam_pos)
    cam = bpy.context.active_object
    direction = center - cam_pos
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 45
    bpy.context.scene.camera = cam


def main():
    opts = parse_args()

    # Fresh scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Set scene fps BEFORE importing: the glTF importer converts keyframe times
    # (seconds) to frame numbers using the CURRENT scene fps. Importing at the
    # default 24 fps and switching later time-compresses the animation ~20%.
    if opts["fps"]:
        bpy.context.scene.render.fps = opts["fps"]

    print(f"Importing GLB: {opts['input']}")
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(opts["input"]))

    armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if armature is None:
        print("ERROR: no armature found in GLB")
        sys.exit(1)

    fs, fe = scene_animation_range()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = fs, fe
    print(f"Animation frames: {fs}..{fe}")

    lo, hi = animated_bounds(armature, fs, fe)
    print(f"Motion bounds: {tuple(round(v,2) for v in lo)} .. {tuple(round(v,2) for v in hi)}")

    build_floor(z=opts["floor-z"])
    build_lights()
    build_camera(lo, hi, opts["width"], opts["height"])

    # World: soft neutral background
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.92, 0.93, 0.95, 1.0)
        bg.inputs["Strength"].default_value = 0.8

    # Render settings
    scene.render.resolution_x = opts["width"]
    scene.render.resolution_y = opts["height"]
    if opts["fps"]:
        scene.render.fps = opts["fps"]
    engine = opts["engine"].lower()
    if engine == "workbench":
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = 16

    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = os.path.abspath(opts["output"])

    # Metadata label burned into the frame (RAW vs CLEANED)
    if opts["label"]:
        scene.render.use_stamp = True
        scene.render.use_stamp_date = False
        scene.render.use_stamp_time = False
        scene.render.use_stamp_render_time = False
        scene.render.use_stamp_frame = False
        scene.render.use_stamp_scene = False
        scene.render.use_stamp_camera = False
        scene.render.use_stamp_filename = False
        scene.render.use_stamp_note = True
        scene.render.stamp_note_text = opts["label"]
        scene.render.stamp_font_size = 24

    print(f"Rendering {fe - fs + 1} frames ({scene.render.engine}) -> {opts['output']}")
    bpy.ops.render.render(animation=True)
    print("Render done.")


if __name__ == "__main__":
    main()
