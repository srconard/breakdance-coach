"""Blender script: bake any BVH/FBX skeleton animation to the mannequin npz.

Runs INSIDE Blender headless:
    blender -b --python skeleton_bake_npz.py -- in.bvh out.npz cmu [fps] [height]

Maps a named bone per SMPL joint (presets: cmu = cgspeed/una-dinosauria CMU
BVH, mixamo = Adobe Mixamo FBX) and samples per frame:
    JW   world head position of the bone          (F, 24, 3)
    G    world rotation RELATIVE TO REST          (F, 24, 3, 3)
    rest world head positions at rest              (24, 3)
so the output satisfies mannequin_glb.py's contract (rest pose = identity
rotations) without any SMPL parameters or retargeting — the mannequin is
built on the SOURCE skeleton's own proportions.

Also: auto-scales so the rest head joint sits at ~1.58 m, rebases the floor
to y=0 from foot-joint lows, downsamples to <= target fps, and converts
Blender Z-up back to the npz's SMPL Y-up convention.
"""

import sys

import bpy
import numpy as np
from mathutils import Matrix

argv = sys.argv[sys.argv.index("--") + 1:]
IN_PATH, OUT_NPZ, PRESET = argv[0], argv[1], argv[2]
TARGET_FPS = float(argv[3]) if len(argv) > 3 else 60.0
TARGET_HEAD_HEIGHT = float(argv[4]) if len(argv) > 4 else 1.58

# SMPL joint index -> source bone name
PRESETS = {
    "cmu": {
        0: "Hips", 1: "LeftUpLeg", 2: "RightUpLeg", 3: "LowerBack",
        4: "LeftLeg", 5: "RightLeg", 6: "Spine", 7: "LeftFoot",
        8: "RightFoot", 9: "Spine1", 10: "LeftToeBase", 11: "RightToeBase",
        12: "Neck", 13: "LeftShoulder", 14: "RightShoulder", 15: "Head",
        16: "LeftArm", 17: "RightArm", 18: "LeftForeArm", 19: "RightForeArm",
        20: "LeftHand", 21: "RightHand", 22: "LeftFingerBase",
        23: "RightFingerBase",
    },
    "mixamo": {
        0: "Hips", 1: "LeftUpLeg", 2: "RightUpLeg", 3: "Spine",
        4: "LeftLeg", 5: "RightLeg", 6: "Spine1", 7: "LeftFoot",
        8: "RightFoot", 9: "Spine2", 10: "LeftToeBase", 11: "RightToeBase",
        12: "Neck", 13: "LeftShoulder", 14: "RightShoulder", 15: "Head",
        16: "LeftArm", 17: "RightArm", 18: "LeftForeArm", 19: "RightForeArm",
        20: "LeftHand", 21: "RightHand", 22: "LeftHandMiddle1",
        23: "RightHandMiddle1",
    },
}
FOOTS = (7, 8, 10, 11)
# Blender Z-up -> SMPL Y-up (inverse of mannequin_glb's C)
CT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]).T


def find_bone(bones, name):
    """Exact match, or suffix match after a namespace prefix (mixamorig:...)."""
    if name in bones:
        return bones[name]
    for b in bones:
        if b.name.endswith(":" + name) or b.name.endswith("|" + name):
            return b
    return None


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    low = IN_PATH.lower()
    if low.endswith(".bvh"):
        bpy.ops.import_anim.bvh(filepath=IN_PATH, update_scene_fps=True)
    elif low.endswith((".fbx",)):
        bpy.ops.import_scene.fbx(filepath=IN_PATH)
    else:
        raise SystemExit(f"unsupported input: {IN_PATH}")

    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    scene = bpy.context.scene
    src_fps = scene.render.fps / scene.render.fps_base
    f0, f1 = scene.frame_start, scene.frame_end
    if arm.animation_data and arm.animation_data.action:
        af0, af1 = arm.animation_data.action.frame_range
        f0, f1 = int(af0), int(np.ceil(af1))
    step = max(1, int(round(src_fps / TARGET_FPS)))
    frames = list(range(f0, f1 + 1, step))
    out_fps = src_fps / step

    mapping = PRESETS[PRESET]
    MW = arm.matrix_world.copy()
    names = {}
    for j, name in mapping.items():
        b = find_bone(arm.data.bones, name)
        if b is None and j not in (22, 23):
            raise SystemExit(f"bone '{name}' (SMPL {j}) not found; have: "
                             f"{[bb.name for bb in arm.data.bones]}")
        names[j] = b.name if b else None

    # rest (world, Blender Z-up); rotation-only rest matrices
    rest = np.zeros((24, 3))
    q_rest_inv = {}
    for j, name in names.items():
        if name is None:
            continue
        bone = arm.data.bones[name]
        rest[j] = np.array((MW @ Matrix.Translation(bone.head_local)).translation)
        q_rest_inv[j] = (MW @ bone.matrix_local).decompose()[1].inverted()
    for j, wrist, elbow in ((22, 20, 18), (23, 21, 19)):
        if names[j] is None:
            rest[j] = rest[wrist] + (rest[wrist] - rest[elbow]) * 0.30

    F = len(frames)
    JW = np.zeros((F, 24, 3))
    G = np.zeros((F, 24, 3, 3))
    for i, f in enumerate(frames):
        scene.frame_set(f)
        for j, name in names.items():
            if name is None:
                continue
            pb = arm.pose.bones[name]
            M = MW @ pb.matrix
            loc, q, _ = M.decompose()
            JW[i, j] = np.array(loc)
            G[i, j] = np.array((q @ q_rest_inv[j]).to_matrix())
        for j, wrist in ((22, 20), (23, 21)):
            if names[j] is None:
                G[i, j] = G[i, wrist]
                JW[i, j] = JW[i, wrist] + G[i, wrist] @ (rest[j] - rest[wrist])

    # auto-scale to human size (source units are arbitrary: CMU ~5.6 cm/unit)
    height = rest[15][2] - rest[:, 2].min()
    scale = TARGET_HEAD_HEIGHT / max(height, 1e-6)
    JW *= scale
    rest *= scale

    # floor to z=0 from foot-joint lows (2nd percentile absorbs mocap noise)
    floor_z = np.percentile(JW[:, FOOTS, 2].min(axis=1), 2.0)
    JW[:, :, 2] -= floor_z

    # Blender Z-up -> SMPL Y-up
    JW = JW @ CT.T
    rest = rest @ CT.T
    G = CT[None, None] @ G @ CT.T[None, None]

    np.savez(OUT_NPZ, JW=JW, G=G, rest=rest, fps=np.float64(out_fps))
    print(f"[skeleton_bake] {F} frames @ {out_fps:.2f} fps (src {src_fps:.0f}, "
          f"step {step}), scale {scale:.4f}, floor {floor_z:.3f}")
    print("[skeleton_bake] DONE")


main()
