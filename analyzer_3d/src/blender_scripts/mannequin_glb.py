"""Blender script: build the segmented training mannequin and export GLB.

Runs INSIDE Blender headless:
    blender -b --python mannequin_glb.py -- motion.npz out.glb

motion.npz (from analyzer_3d.src.mannequin, all SMPL Y-up world space):
    JW   (F, 24, 3)    joint world positions (transl applied)
    G    (F, 24, 3, 3) joint world rotations
    rest (24, 3)       rest joint positions
    fps  scalar

Each body segment is a rigid cylinder modeled in its parent joint's rest
frame and keyframed with that joint's WORLD transform per frame (object
TRS animation — no armature, no skinning). Ball-joint spheres ride the
joint world positions. Left = teal, right = orange, center = gray.
"""

import sys

import bpy
import numpy as np
from mathutils import Matrix

argv = sys.argv[sys.argv.index("--") + 1:]
NPZ_PATH, OUT_GLB = argv[0], argv[1]

# SMPL joint indices
PARENTS24 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
             16, 17, 18, 19, 20, 21]
LEFT = {1, 4, 7, 10, 13, 16, 18, 20, 22}
RIGHT = {2, 5, 8, 11, 14, 17, 19, 21, 23}

# (child_joint, radius) — segment runs parent(child) -> child; color by child side
SEGMENT_RADIUS = {
    1: 0.055, 2: 0.055,            # pelvis -> hips
    4: 0.055, 5: 0.055,            # thighs
    7: 0.042, 8: 0.042,            # shins
    10: 0.030, 11: 0.030,          # feet (ankle -> toes)
    3: 0.075, 6: 0.080, 9: 0.085,  # spine
    12: 0.032, 15: 0.030,          # neck, neck -> head
    13: 0.045, 14: 0.045,          # collars
    16: 0.042, 17: 0.042,          # collar -> shoulder
    18: 0.036, 19: 0.036,          # upper arms
    20: 0.030, 21: 0.030,          # forearms
    22: 0.024, 23: 0.024,          # wrist -> hand
}
JOINT_SPHERE_RADIUS = {
    1: 0.062, 2: 0.062, 4: 0.060, 5: 0.060, 7: 0.048, 8: 0.048,
    10: 0.034, 11: 0.034, 12: 0.045, 16: 0.052, 17: 0.052,
    18: 0.042, 19: 0.042, 20: 0.036, 21: 0.036, 22: 0.040, 23: 0.040,
}
HEAD_JOINT = 15
HEAD_RADIUS = 0.095
HEAD_OFFSET = 0.055  # head sphere center above the head joint, along rest neck axis

COLORS = {
    "L": (0.10, 0.62, 0.58, 1.0),   # teal
    "R": (0.90, 0.42, 0.18, 1.0),   # orange
    "C": (0.72, 0.73, 0.76, 1.0),   # light gray
    "J": (0.28, 0.29, 0.32, 1.0),   # dark gray ball joints
}

# SMPL Y-up -> Blender Z-up: (x, y, z) -> (x, -z, y)
C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def side_of(j):
    return "L" if j in LEFT else "R" if j in RIGHT else "C"


def make_material(name, rgba):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.45
    bsdf.inputs["Metallic"].default_value = 0.05
    return mat


def cylinder_between(name, a, b, radius, mat):
    """Cylinder from local point a to local point b, baked into mesh data
    so the OBJECT transform stays identity (= the joint's world TRS)."""
    a, b = np.asarray(a), np.asarray(b)
    d = b - a
    length = float(np.linalg.norm(d))
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=max(length, 1e-4),
                                        vertices=20)
    ob = bpy.context.active_object
    ob.name = name
    # default cylinder is centered at origin along +Z; move to a->b
    z = np.array([0.0, 0.0, 1.0])
    axis = d / max(length, 1e-9)
    rot = z_align_matrix(z, axis)
    M = Matrix.Translation((a + d / 2).tolist()) @ rot.to_4x4()
    ob.data.transform(M)
    ob.matrix_world = Matrix.Identity(4)
    ob.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return ob


def z_align_matrix(z, axis):
    v = np.cross(z, axis)
    c = float(np.dot(z, axis))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return Matrix.Identity(3)
        return Matrix.Rotation(np.pi, 3, "X")
    from mathutils import Vector
    return Vector(z.tolist()).rotation_difference(Vector(axis.tolist())).to_matrix()


def sphere_at(name, center, radius, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=20, ring_count=14)
    ob = bpy.context.active_object
    ob.name = name
    ob.data.transform(Matrix.Translation(np.asarray(center).tolist()))
    ob.matrix_world = Matrix.Identity(4)
    ob.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return ob


def keyframe_world(ob, locs, quats, n_frames):
    """Fast object TRS animation.

    Blender 4.4 slotted actions: creating an action + fcurves via the pure
    legacy API leaves them unbound to a slot (the glTF exporter then sees no
    animation). Instead, keyframe_insert() on frame 1 creates a properly
    bound action/slot/fcurves, and the legacy action.fcurves accessor then
    bulk-fills the rest via foreach_set."""
    ob.rotation_mode = "QUATERNION"
    ob.location = locs[0].tolist()
    ob.rotation_quaternion = quats[0].tolist()
    ob.keyframe_insert("location", frame=1)
    ob.keyframe_insert("rotation_quaternion", frame=1)
    action = ob.animation_data.action
    frames = np.arange(1, n_frames + 1, dtype=np.float64)
    try:
        for fc in action.fcurves:
            values = locs if fc.data_path == "location" else quats
            fc.keyframe_points.add(n_frames - 1)
            co = np.empty(2 * n_frames, dtype=np.float64)
            co[0::2] = frames
            co[1::2] = values[:, fc.array_index]
            fc.keyframe_points.foreach_set("co", co)
            fc.update()
    except Exception as e:  # noqa: BLE001 - fall back to the slow universal path
        print(f"[mannequin_glb] foreach_set path failed ({e}); slow keyframing")
        for f in range(2, n_frames + 1):
            ob.location = locs[f - 1].tolist()
            ob.rotation_quaternion = quats[f - 1].tolist()
            ob.keyframe_insert("location", frame=f)
            ob.keyframe_insert("rotation_quaternion", frame=f)


def quats_from_rots(R):
    """(F,3,3) -> (F,4) wxyz with sign continuity."""
    F = R.shape[0]
    out = np.empty((F, 4))
    prev = None
    for f in range(F):
        q = Matrix(R[f].tolist()).to_quaternion()
        v = np.array([q.w, q.x, q.y, q.z])
        if prev is not None and np.dot(v, prev) < 0:
            v = -v
        out[f] = v
        prev = v
    return out


def main():
    data = np.load(NPZ_PATH)
    JW, G, rest, fps = data["JW"], data["G"], data["rest"], float(data["fps"])
    F = JW.shape[0]

    # convert everything to Blender Z-up once
    JWb = JW @ C.T
    Gb = C[None, None] @ G @ C.T[None, None]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = int(round(fps))
    scene.frame_start, scene.frame_end = 1, F

    mats = {k: make_material(f"mannequin_{k}", v) for k, v in COLORS.items()}

    # rest offsets in the PARENT's local frame equal world offsets at rest
    # (rest pose has identity rotations), converted to Z-up
    rest_b = rest @ C.T

    parts = []  # (object, anchor_joint)
    for c_j, radius in SEGMENT_RADIUS.items():
        p_j = PARENTS24[c_j]
        offset = rest_b[c_j] - rest_b[p_j]
        ob = cylinder_between(f"seg_{p_j}_{c_j}", np.zeros(3), offset, radius,
                              mats[side_of(c_j)])
        parts.append((ob, p_j))

    for j, radius in JOINT_SPHERE_RADIUS.items():
        ob = sphere_at(f"joint_{j}", np.zeros(3), radius, mats["J"])
        parts.append((ob, j))

    # head: sphere above the head joint along the rest neck->head axis
    neck_axis = rest_b[HEAD_JOINT] - rest_b[12]
    neck_axis = neck_axis / max(np.linalg.norm(neck_axis), 1e-9)
    ob = sphere_at("head", neck_axis * HEAD_OFFSET, HEAD_RADIUS, mats["C"])
    parts.append((ob, HEAD_JOINT))

    for ob, j in parts:
        keyframe_world(ob, JWb[:, j], quats_from_rots(Gb[:, j]), F)

    bpy.ops.export_scene.gltf(
        filepath=OUT_GLB,
        export_format="GLB",
        export_animations=True,
        export_animation_mode="SCENE",  # one merged clip (simplest for Three.js)
        export_yup=True,
    )
    print(f"[mannequin_glb] parts={len(parts)} frames={F} fps={scene.render.fps}")
    print("[mannequin_glb] DONE")


main()
