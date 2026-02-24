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


def axis_angle_to_quaternion(axis_angle):
    """Convert axis-angle rotation to quaternion (w, x, y, z)."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-6:
        return mathutils.Quaternion((1, 0, 0, 0))
    axis = axis_angle / angle
    return mathutils.Quaternion(axis, angle)


def create_smpl_armature(name="SMPL_Armature"):
    """Create an SMPL skeleton armature in Blender.

    Creates a simple armature with 22 bones matching SMPL joint hierarchy.
    """
    # Create armature
    armature_data = bpy.data.armatures.new(name)
    armature_obj = bpy.data.objects.new(name, armature_data)
    bpy.context.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj

    # Enter edit mode to create bones
    bpy.ops.object.mode_set(mode='EDIT')

    # SMPL kinematic tree: parent_index for each joint
    # -1 means root (no parent)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

    # Default bone positions (approximate SMPL rest pose)
    # These are rough positions; the actual mesh comes from SMPL parameters
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

    bones = []
    for i, name in enumerate(SMPL_JOINT_NAMES):
        bone = armature_data.edit_bones.new(name)
        pos = positions[i]
        bone.head = mathutils.Vector(pos)
        # Bone tail slightly offset (bones need non-zero length)
        bone.tail = mathutils.Vector((pos[0], pos[1], pos[2] + 0.05))

        if parents[i] >= 0:
            bone.parent = bones[parents[i]]

        bones.append(bone)

    bpy.ops.object.mode_set(mode='OBJECT')
    return armature_obj


def apply_smpl_animation(armature_obj, smpl_data, fps=30):
    """Apply SMPL parameters as keyframe animation to the armature.

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

    for frame_idx in range(num_frames):
        bpy.context.scene.frame_set(frame_idx + 1)  # Blender frames are 1-indexed

        # Apply root translation
        root_bone = armature_obj.pose.bones["Pelvis"]
        t = transl[frame_idx]
        root_bone.location = mathutils.Vector((t[0], t[1], t[2]))
        root_bone.keyframe_insert(data_path="location", frame=frame_idx + 1)

        # Apply root orientation (global_orient)
        orient = global_orient[frame_idx]
        quat = axis_angle_to_quaternion(orient)
        root_bone.rotation_mode = 'QUATERNION'
        root_bone.rotation_quaternion = quat
        root_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx + 1)

        # Apply body joint rotations (joints 1-21)
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


def create_smpl_mesh(armature_obj, smpl_data):
    """Create a simple mesh representation.

    For the initial version, we create a basic stick-figure mesh
    from the armature. A full SMPL mesh with proper skinning would
    require the smplx library (not available in Blender's Python).

    For proper SMPL mesh: use the smplx library externally to generate
    per-frame vertices, then import as shape keys.
    """
    # Create a simple visualization mesh (icosphere at each joint)
    # This gives a visible 3D representation in GLB viewers
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create a low-poly mesh that follows the armature
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.05,
        segments=8,
        ring_count=6,
        location=(0, 0, 0),
    )
    sphere = bpy.context.active_object
    sphere.name = "SMPL_Body"

    # Add armature modifier so mesh follows the skeleton
    mod = sphere.modifiers.new("Armature", 'ARMATURE')
    mod.object = armature_obj

    # Parent to armature
    sphere.parent = armature_obj

    return sphere


def export_glb(output_path):
    """Export the scene as animated GLB."""
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        export_animations=True,
        export_apply=False,
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

    # Clear default scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Create armature
    print("Creating SMPL armature...")
    armature = create_smpl_armature()

    # Apply animation
    print("Applying animation keyframes...")
    apply_smpl_animation(armature, smpl_data, fps=fps)

    # Create visual mesh
    print("Creating mesh...")
    create_smpl_mesh(armature, smpl_data)

    # Export
    print("Exporting GLB...")
    export_glb(output_path)

    print("Done!")


if __name__ == "__main__":
    main()
