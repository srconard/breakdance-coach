"""Blender: convert a skinned FBX (e.g. a Mixamo character+animation) to GLB.

Runs INSIDE Blender headless:
    blender -b --python fbx_to_glb.py -- in.fbx out.glb [floor]

Unlike the segmented mannequin path, this keeps the FBX's own skinned mesh
(Mixamo's Beta / X-Bot character), so the export looks exactly like Mixamo's
preview. Grounds the character to z=0 and exports animation. Used for
reference clips where the source ships a mannequin mesh and the motion is
clean enough that LBS skinning holds up (Mixamo game-quality animation).
"""

import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
IN_FBX, OUT_GLB = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=IN_FBX)

# ground the whole rig to z=0 (Mixamo exports with feet near origin already,
# but small offsets look like floating) using the lowest mesh vertex at rest
scene = bpy.context.scene
scene.frame_set(scene.frame_start)
zmin = None
for ob in bpy.data.objects:
    if ob.type == "MESH":
        for v in ob.bound_box:
            wz = (ob.matrix_world @ __import__("mathutils").Vector(v)).z
            zmin = wz if zmin is None else min(zmin, wz)
if zmin is not None:
    for ob in bpy.data.objects:
        if ob.parent is None:
            ob.location.z -= zmin

bpy.ops.export_scene.gltf(
    filepath=OUT_GLB,
    export_format="GLB",
    export_animations=True,
    export_yup=True,
)
print(f"[fbx_to_glb] {IN_FBX} -> {OUT_GLB}")
print("[fbx_to_glb] DONE")
