"""Blender script to convert SMPL parameters to animated GLB.

This script runs INSIDE Blender's Python environment (bpy).
It is called headless via:
    blender -b --python smpl_to_glb.py -- --input params.pkl --output move.glb

Input: Pickle file with GVHMR output format:
    {
        "smpl_params_global": {
            "body_pose": (F, 63),      # 21 joints x 3 axis-angle
            "global_orient": (F, 3),   # root rotation
            "transl": (F, 3),          # root translation
            "betas": (F, 10),          # shape parameters
        },
        "fps": float,
        "num_frames": int,
    }

Output: Animated .GLB file viewable in Obsidian, Three.js, etc.
"""

import sys
import os
import pickle
import math

import bpy
import mathutils
import numpy as np


# SMPL joint names in order (24 joints total)
# global_orient = joint 0 (root), body_pose = joints 1-21
SMPL_JOINT_NAMES = [
    "Pelvis",           # 0 - root (global_orient)
    "L_Hip",            # 1
    "R_Hip",            # 2
    "Spine1",           # 3
    "L_Knee",           # 4
    "R_Knee",           # 5
    "Spine2",           # 6
    "L_Ankle",          # 7
    "R_Ankle",          # 8
    "Spine3",           # 9
    "L_Foot",           # 10
    "R_Foot",           # 11
    "Neck",             # 12
    "L_Collar",         # 13
    "R_Collar",         # 14
    "Head",             # 15
    "L_Shoulder",       # 16
    "R_Shoulder",       # 17
    "L_Elbow",          # 18
    "R_Elbow",          # 19
    "L_Wrist",          # 20
    "R_Wrist",          # 21
]


def smpl_to_blender(v):
    """Convert SMPL Y-up coords to Blender Z-up: (x, y, z) → (x, -z, y)."""
    return (v[0], -v[2], v[1])


def smpl_to_blender_array(arr):
    """Convert array of SMPL coords to Blender coords.
    Works for (N, 3) arrays or (3,) vectors. Applies to positions AND axis-angle rotation vectors.
    """
    result = arr.copy()
    if arr.ndim == 1:
        result[0] = arr[0]
        result[1] = -arr[2]
        result[2] = arr[1]
    else:
        result[:, 0] = arr[:, 0]
        result[:, 1] = -arr[:, 2]
        result[:, 2] = arr[:, 1]
    return result


def axis_angle_to_quaternion(axis_angle):
    """Convert axis-angle rotation to quaternion (w, x, y, z)."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-6:
        return mathutils.Quaternion((1, 0, 0, 0))
    axis = axis_angle / angle
    return mathutils.Quaternion(axis, angle)


# SMPL kinematic tree: parent_index for each joint (-1 = root)
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# Fallback approximate joint positions (used when SMPL mesh data unavailable)
SMPL_FALLBACK_POSITIONS = [
    (0, 0, 0.95),       # 0 Pelvis
    (0.09, 0, 0.90),    # 1 L_Hip
    (-0.09, 0, 0.90),   # 2 R_Hip
    (0, 0, 1.05),       # 3 Spine1
    (0.09, 0, 0.50),    # 4 L_Knee
    (-0.09, 0, 0.50),   # 5 R_Knee
    (0, 0, 1.20),       # 6 Spine2
    (0.09, 0, 0.08),    # 7 L_Ankle
    (-0.09, 0, 0.08),   # 8 R_Ankle
    (0, 0, 1.35),       # 9 Spine3
    (0.09, 0, 0.02),    # 10 L_Foot
    (-0.09, 0, 0.02),   # 11 R_Foot
    (0, 0, 1.50),       # 12 Neck
    (0.10, 0, 1.45),    # 13 L_Collar
    (-0.10, 0, 1.45),   # 14 R_Collar
    (0, 0, 1.60),       # 15 Head
    (0.20, 0, 1.40),    # 16 L_Shoulder
    (-0.20, 0, 1.40),   # 17 R_Shoulder
    (0.45, 0, 1.40),    # 18 L_Elbow
    (-0.45, 0, 1.40),   # 19 R_Elbow
    (0.65, 0, 1.40),    # 20 L_Wrist
    (-0.65, 0, 1.40),   # 21 R_Wrist
]


def create_smpl_armature(name="SMPL_Armature", joint_positions=None):
    """Create an SMPL skeleton armature in Blender.

    Args:
        name: Name for the armature object
        joint_positions: (24, 3) array of actual SMPL rest pose joint positions.
                        If None, uses approximate fallback positions.

    Creates armature with 22 bones matching SMPL joint hierarchy.

    IMPORTANT: All bones point in the SAME direction (+Y, tiny length) so that
    every bone's local frame equals the world frame (identity rest rotation).
    This is critical because SMPL body_pose rotations are in the parent's
    world-aligned frame. With uniform bone directions, Blender's pose rotations
    match SMPL's convention directly — no per-bone compensation needed.
    (Varying bone directions cause spine vs leg rotation mismatches.)
    """
    # Use real positions (first 22 joints) or fallback
    if joint_positions is not None:
        positions = [tuple(joint_positions[i]) for i in range(22)]
        print(f"  Using real SMPL joint positions from body model")
    else:
        positions = SMPL_FALLBACK_POSITIONS
        print(f"  Using approximate fallback joint positions")

    # Create armature
    armature_data = bpy.data.armatures.new(name)
    armature_obj = bpy.data.objects.new(name, armature_data)
    bpy.context.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj

    bpy.ops.object.mode_set(mode='EDIT')

    bones = []
    for i, jname in enumerate(SMPL_JOINT_NAMES):
        bone = armature_data.edit_bones.new(jname)
        head = mathutils.Vector(positions[i])
        bone.head = head
        # All bones point +Y with tiny length → bone-local frame = world frame.
        # This ensures SMPL rotations (in parent's world-aligned frame) can be
        # applied directly as Blender pose rotations without compensation.
        bone.tail = head + mathutils.Vector((0, 0.02, 0))
        bone.roll = 0.0

        if SMPL_PARENTS[i] >= 0:
            bone.parent = bones[SMPL_PARENTS[i]]

        bones.append(bone)

    bpy.ops.object.mode_set(mode='OBJECT')
    return armature_obj


def apply_smpl_animation(armature_obj, smpl_data, fps=30):
    """Apply SMPL parameters as keyframe animation to the armature.

    With uniform bone directions (+Y), all bone-local frames equal the world frame.
    SMPL's coordinate-transformed rotations can be applied directly as pose rotations
    with no per-bone compensation needed.

    Args:
        armature_obj: Blender armature object
        smpl_data: Dict with SMPL parameters from GVHMR
        fps: Frame rate for the animation
    """
    params = smpl_data["smpl_params_global"]
    body_pose = params["body_pose"]       # (F, 63)
    global_orient = params["global_orient"]  # (F, 3)
    transl = params["transl"]             # (F, 3)

    num_frames = body_pose.shape[0]

    # Set scene frame range and FPS
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames
    bpy.context.scene.render.fps = int(fps)

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    # Root bone rest position for translation offset
    pelvis_rest_pos = mathutils.Vector(armature_obj.data.bones["Pelvis"].head_local)

    for frame_idx in range(num_frames):
        bpy.context.scene.frame_set(frame_idx + 1)  # Blender frames are 1-indexed

        # Apply root translation (world-space offset from rest position)
        root_bone = armature_obj.pose.bones["Pelvis"]
        t = transl[frame_idx]
        target = mathutils.Vector((t[0], t[1], t[2]))
        root_bone.location = target - pelvis_rest_pos
        root_bone.keyframe_insert(data_path="location", frame=frame_idx + 1)

        # Apply root orientation (global_orient) — direct, no compensation needed
        orient = global_orient[frame_idx]
        quat = axis_angle_to_quaternion(orient)
        root_bone.rotation_mode = 'QUATERNION'
        root_bone.rotation_quaternion = quat
        root_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx + 1)

        # Apply body joint rotations (joints 1-21) — direct, no compensation needed
        for joint_idx in range(21):
            joint_name = SMPL_JOINT_NAMES[joint_idx + 1]
            pose_bone = armature_obj.pose.bones.get(joint_name)
            if pose_bone is None:
                continue

            # Extract axis-angle for this joint
            aa = body_pose[frame_idx, joint_idx * 3: (joint_idx + 1) * 3]
            quat = axis_angle_to_quaternion(aa)

            pose_bone.rotation_mode = 'QUATERNION'
            pose_bone.rotation_quaternion = quat
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx + 1)

        if (frame_idx + 1) % 100 == 0:
            print(f"  Keyframed {frame_idx + 1}/{num_frames}")

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Applied {num_frames} frames of animation")


def create_smpl_body_mesh(armature_obj, smpl_data):
    """Create the full SMPL body mesh (6890 vertices) with proper skinning.

    Uses pre-computed mesh data from the Modal inference step:
    - vertices: (6890, 3) shaped rest pose
    - faces: (13776, 3) triangle indices
    - weights: (6890, 24) per-vertex bone weights

    Returns the mesh object, or None if mesh data is not available.
    """
    if "mesh" not in smpl_data:
        return None

    mesh_info = smpl_data["mesh"]
    vertices = mesh_info["vertices"]   # (6890, 3)
    faces = mesh_info["faces"]         # (13776, 3)
    weights = mesh_info["weights"]     # (6890, 24)

    # SMPL has 24 joints but our armature only has 22 (joints 0-21)
    # Joints 22 (L_Hand) and 23 (R_Hand) don't exist in our armature
    # We'll merge those weights into L_Wrist and R_Wrist
    num_armature_joints = len(SMPL_JOINT_NAMES)  # 22

    # Create Blender mesh
    mesh_data = bpy.data.meshes.new("SMPL_Body")
    verts_list = [tuple(v) for v in vertices]
    faces_list = [tuple(f) for f in faces]
    mesh_data.from_pydata(verts_list, [], faces_list)
    mesh_data.update()

    body_obj = bpy.data.objects.new("SMPL_Body", mesh_data)
    bpy.context.collection.objects.link(body_obj)

    # Create vertex groups for each bone and assign weights
    for joint_idx in range(num_armature_joints):
        joint_name = SMPL_JOINT_NAMES[joint_idx]
        vg = body_obj.vertex_groups.new(name=joint_name)

        # Get weights for this joint
        joint_weights = weights[:, joint_idx].copy()

        # For wrists, add hand weights (if they exist in SMPL's 24 joints)
        if joint_idx == 20 and weights.shape[1] > 22:  # L_Wrist + L_Hand
            joint_weights += weights[:, 22]
        elif joint_idx == 21 and weights.shape[1] > 23:  # R_Wrist + R_Hand
            joint_weights += weights[:, 23]

        # Assign non-zero weights
        for vert_idx in range(len(joint_weights)):
            w = float(joint_weights[vert_idx])
            if w > 1e-6:
                vg.add([vert_idx], w, 'REPLACE')

    # Add armature modifier
    mod = body_obj.modifiers.new("Armature", 'ARMATURE')
    mod.object = armature_obj
    body_obj.parent = armature_obj

    # Smooth shading for nicer look
    for poly in mesh_data.polygons:
        poly.use_smooth = True

    # Add material (skin-like color)
    mat = bpy.data.materials.new("Body_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.7, 0.55, 0.45, 1.0)  # skin tone
        bsdf.inputs["Roughness"].default_value = 0.6
        bsdf.inputs["Subsurface Weight"].default_value = 0.1
    body_obj.data.materials.append(mat)

    print(f"  Created SMPL body mesh: {len(vertices)} vertices, {len(faces)} faces")
    return body_obj


def create_smpl_mesh(armature_obj, smpl_data):
    """Create a stick-figure mesh with proper vertex groups for bone deformation.

    Builds a single combined mesh with:
    - Icospheres at each joint (sized by body part)
    - Cylinders connecting parent-child joints (limbs)
    - Vertex groups weighted to corresponding bones
    - Armature modifier for animation

    This produces a visible, animated figure in GLB viewers.
    """
    import bmesh

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='OBJECT')

    # Reuse the same positions and parent hierarchy as the armature
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    positions = [
        (0, 0, 0.95),       # 0 Pelvis
        (0.09, 0, 0.90),    # 1 L_Hip
        (-0.09, 0, 0.90),   # 2 R_Hip
        (0, 0, 1.05),       # 3 Spine1
        (0.09, 0, 0.50),    # 4 L_Knee
        (-0.09, 0, 0.50),   # 5 R_Knee
        (0, 0, 1.20),       # 6 Spine2
        (0.09, 0, 0.08),    # 7 L_Ankle
        (-0.09, 0, 0.08),   # 8 R_Ankle
        (0, 0, 1.35),       # 9 Spine3
        (0.09, 0, 0.02),    # 10 L_Foot
        (-0.09, 0, 0.02),   # 11 R_Foot
        (0, 0, 1.50),       # 12 Neck
        (0.10, 0, 1.45),    # 13 L_Collar
        (-0.10, 0, 1.45),   # 14 R_Collar
        (0, 0, 1.60),       # 15 Head
        (0.20, 0, 1.40),    # 16 L_Shoulder
        (-0.20, 0, 1.40),   # 17 R_Shoulder
        (0.45, 0, 1.40),    # 18 L_Elbow
        (-0.45, 0, 1.40),   # 19 R_Elbow
        (0.65, 0, 1.40),    # 20 L_Wrist
        (-0.65, 0, 1.40),   # 21 R_Wrist
    ]

    # Joint sphere radii — larger for torso/head, smaller for extremities
    joint_radii = [
        0.06,   # Pelvis
        0.05,   # L_Hip
        0.05,   # R_Hip
        0.05,   # Spine1
        0.04,   # L_Knee
        0.04,   # R_Knee
        0.05,   # Spine2
        0.035,  # L_Ankle
        0.035,  # R_Ankle
        0.05,   # Spine3
        0.03,   # L_Foot
        0.03,   # R_Foot
        0.04,   # Neck
        0.03,   # L_Collar
        0.03,   # R_Collar
        0.07,   # Head
        0.045,  # L_Shoulder
        0.045,  # R_Shoulder
        0.035,  # L_Elbow
        0.035,  # R_Elbow
        0.03,   # L_Wrist
        0.03,   # R_Wrist
    ]

    mesh_data = bpy.data.meshes.new("SMPL_Body")
    body_obj = bpy.data.objects.new("SMPL_Body", mesh_data)
    bpy.context.collection.objects.link(body_obj)

    bm = bmesh.new()
    vertex_groups_data = {}  # bone_name -> [vertex_indices]

    # --- Create joint spheres ---
    for i, name in enumerate(SMPL_JOINT_NAMES):
        radius = joint_radii[i]
        result = bmesh.ops.create_icosphere(bm, subdivisions=2, radius=radius)
        verts = result['verts']

        # Move sphere to joint position
        offset = mathutils.Vector(positions[i])
        for v in verts:
            v.co += offset

        vertex_groups_data[name] = [v.index for v in verts]

    # --- Create limb cylinders ---
    for i in range(len(SMPL_JOINT_NAMES)):
        if parents[i] < 0:
            continue

        parent_pos = mathutils.Vector(positions[parents[i]])
        child_pos = mathutils.Vector(positions[i])
        direction = child_pos - parent_pos
        length = direction.length
        if length < 0.01:
            continue

        midpoint = (parent_pos + child_pos) / 2

        # Cylinder radius proportional to connected joints
        limb_radius = min(joint_radii[parents[i]], joint_radii[i]) * 0.4

        # Create cylinder along Z axis at origin
        result = bmesh.ops.create_cone(
            bm, cap_ends=True, cap_tris=True,
            segments=8, radius1=limb_radius, radius2=limb_radius,
            depth=length,
        )
        verts = result['verts']

        # Rotate cylinder to align with limb direction
        z_axis = mathutils.Vector((0, 0, 1))
        dir_norm = direction.normalized()
        if (z_axis - dir_norm).length > 1e-6 and (z_axis + dir_norm).length > 1e-6:
            rotation = z_axis.rotation_difference(dir_norm)
            for v in verts:
                v.co.rotate(rotation)

        # Translate to midpoint
        for v in verts:
            v.co += midpoint

        # Weight cylinder to parent bone (moves with parent)
        parent_name = SMPL_JOINT_NAMES[parents[i]]
        if parent_name not in vertex_groups_data:
            vertex_groups_data[parent_name] = []
        vertex_groups_data[parent_name].extend([v.index for v in verts])

    # Finalize mesh
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()

    # Create vertex groups and assign weights
    for name, indices in vertex_groups_data.items():
        vg = body_obj.vertex_groups.new(name=name)
        vg.add(indices, 1.0, 'REPLACE')

    # Add armature modifier
    mod = body_obj.modifiers.new("Armature", 'ARMATURE')
    mod.object = armature_obj
    body_obj.parent = armature_obj

    # Add material for visibility (blue body)
    mat = bpy.data.materials.new("Body_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.15, 0.45, 0.9, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
    body_obj.data.materials.append(mat)

    num_verts = len(mesh_data.vertices)
    num_groups = len(body_obj.vertex_groups)
    print(f"  Created stick figure: {num_verts} vertices, {num_groups} vertex groups")

    return body_obj


def export_glb(output_path):
    """Export the scene as animated GLB."""
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        export_animations=True,
        export_skins=True,
        export_apply=False,
        export_frame_range=True,
        export_frame_step=1,
        export_anim_single_armature=True,
        export_optimize_animation_size=False,
        export_optimize_animation_keep_anim_armature=True,
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported GLB: {output_path} ({size_mb:.2f} MB)")


def main():
    # Parse arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        print("Usage: blender -b --python smpl_to_glb.py -- --input params.pkl --output move.glb")
        sys.exit(1)

    input_path = None
    output_path = None
    fps = 30

    i = 0
    while i < len(argv):
        if argv[i] == "--input" and i + 1 < len(argv):
            input_path = argv[i + 1]
            i += 2
        elif argv[i] == "--output" and i + 1 < len(argv):
            output_path = argv[i + 1]
            i += 2
        elif argv[i] == "--fps" and i + 1 < len(argv):
            fps = int(argv[i + 1])
            i += 2
        else:
            i += 1

    if not input_path or not output_path:
        print("Error: --input and --output are required")
        sys.exit(1)

    print(f"SMPL to GLB Converter")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  FPS:    {fps}")

    # Load SMPL data
    with open(input_path, "rb") as f:
        smpl_data = pickle.load(f)

    # Use video FPS if available
    if "fps" in smpl_data:
        fps = int(smpl_data["fps"])
        print(f"  Using video FPS: {fps}")

    # Transform SMPL data from Y-up to Blender Z-up coordinate system
    # SMPL (x, y, z) → Blender (x, -z, y)
    print("Converting SMPL Y-up → Blender Z-up coordinates...")
    params = smpl_data["smpl_params_global"]
    params["global_orient"] = smpl_to_blender_array(params["global_orient"])
    params["transl"] = smpl_to_blender_array(params["transl"])
    # body_pose: (F, 63) = 21 joints x 3 axis-angle — transform each joint's 3D vector
    bp = params["body_pose"]
    for j in range(21):
        bp[:, j*3:(j+1)*3] = smpl_to_blender_array(bp[:, j*3:(j+1)*3])
    params["body_pose"] = bp

    if "mesh" in smpl_data:
        mesh = smpl_data["mesh"]
        mesh["vertices"] = smpl_to_blender_array(mesh["vertices"])
        if "joints" in mesh:
            mesh["joints"] = smpl_to_blender_array(mesh["joints"])

    # Clear default scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Get real joint positions from SMPL body model (if available)
    joint_positions = None
    if "mesh" in smpl_data and "joints" in smpl_data["mesh"]:
        joint_positions = smpl_data["mesh"]["joints"]

    # Create armature
    print("Creating SMPL armature...")
    armature = create_smpl_armature(joint_positions=joint_positions)

    # Apply animation
    print("Applying animation keyframes...")
    apply_smpl_animation(armature, smpl_data, fps=fps)

    # Create visual mesh (prefer full SMPL body, fall back to stick figure)
    print("Creating mesh...")
    body = create_smpl_body_mesh(armature, smpl_data)
    if body is None:
        print("  No SMPL mesh data found, using stick figure...")
        create_smpl_mesh(armature, smpl_data)
    else:
        print("  Using full SMPL body mesh")

    # Export
    print("Exporting GLB...")
    export_glb(output_path)

    print("Done!")


if __name__ == "__main__":
    main()
