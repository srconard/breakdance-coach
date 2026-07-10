"""Blender: play an animation-only Mixamo FBX on the Beta character, export GLB.

Runs INSIDE Blender headless:
    blender -b --python fbx_anim_to_beta_glb.py -- donor.fbx anim.fbx out.glb

Mixamo pack downloads ("without skin") are mesh-less FBXs: just the mixamorig
skeleton + keyframes. Individual downloads ship WITH the skinned Beta mesh.
This script bridges the two so pack animations join the same reference library:

  1. import a donor FBX that carries the Beta mesh (e.g. downloads/mixamo/Flair.fbx)
  2. import the animation-only FBX (identical 65-bone mixamorig skeleton)
  3. assign the animation's action to the Beta armature and delete the donor's
     own action + the bare animation skeleton
  4. ground to z=0 and export GLB exactly like fbx_to_glb.py

Blender 4.4 slotted-actions gotcha (see mannequin_glb.py): assigning
`animation_data.action` alone can leave the action unbound to a slot, and the
glTF exporter then silently writes an empty/missing animation. We explicitly
bind `animation_data.action_slot` to the action's slot. Callers should still
verify the output GLB has a non-empty animations array.

The action is renamed to the donor's original action name
('Armature|mixamo.com|Layer0') so the exported animation name matches the
twelve GLBs converted by fbx_to_glb.py.
"""

import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
DONOR_FBX, ANIM_FBX, OUT_GLB = argv[0], argv[1], argv[2]

bpy.ops.wm.read_factory_settings(use_empty=True)

# --- 1. donor: Beta character (mesh + armature + its own throwaway action) ---
bpy.ops.import_scene.fbx(filepath=DONOR_FBX)
donor_objects = set(bpy.data.objects)
beta_arm = next(ob for ob in donor_objects if ob.type == "ARMATURE")
donor_action = beta_arm.animation_data.action if beta_arm.animation_data else None

# --- 2. animation-only FBX: bare skeleton + the action we actually want ---
bpy.ops.import_scene.fbx(filepath=ANIM_FBX)
anim_objects = [ob for ob in bpy.data.objects if ob not in donor_objects]
anim_arm = next(ob for ob in anim_objects if ob.type == "ARMATURE")
anim_action = anim_arm.animation_data.action

# sanity: the rigs must share bone names for direct action assignment
beta_bones = {b.name for b in beta_arm.data.bones}
animated = {fc.data_path.split('"')[1] for fc in anim_action.fcurves if '"' in fc.data_path}
missing = animated - beta_bones
if missing:
    raise RuntimeError(
        f"{len(missing)} animated bones missing on Beta rig, e.g. {sorted(missing)[:5]}"
    )

# --- 3. move the action onto Beta, drop everything else ---
if beta_arm.animation_data is None:
    beta_arm.animation_data_create()
beta_arm.animation_data.action = anim_action
# Blender 4.4 slotted actions: without a bound slot the fcurves don't evaluate
# and the glTF exporter sees no animation. Bind explicitly.
if hasattr(anim_action, "slots") and len(anim_action.slots):
    beta_arm.animation_data.action_slot = anim_action.slots[0]

# keep the animation name identical to the fbx_to_glb.py conversions
if donor_action is not None:
    keep_name = donor_action.name
    bpy.data.actions.remove(donor_action)  # would export as a 2nd animation
    anim_action.name = keep_name

for ob in anim_objects:  # bare skeleton no longer needed
    bpy.data.objects.remove(ob, do_unlink=True)

scene = bpy.context.scene
start, end = anim_action.frame_range
scene.frame_start, scene.frame_end = int(start), int(end)

# --- 4. ground + export: same behavior as fbx_to_glb.py ---
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
print(f"[fbx_anim_to_beta_glb] {ANIM_FBX} (on {DONOR_FBX}) -> {OUT_GLB}")
print("[fbx_anim_to_beta_glb] DONE")
