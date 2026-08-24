"""Blender-side scan cleanup and GLB validation.

Run inside Blender:
    blender -b --factory-startup --python prepare_scene_glb.py -- \
        source.ply output.glb --report output.scene-report.json

The exported scene follows the viewer contract: meters, glTF Y-up, ground at
y=0, and the selected dance spot near the origin. Blender itself remains
Z-up; the glTF exporter performs the axis conversion.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-triangles", type=int, default=200_000)
    parser.add_argument("--max-output-mb", type=float, default=25.0)
    parser.add_argument("--max-face-area", type=float, default=0.25)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--level-mode", choices=("pca", "none"), default="pca")
    parser.add_argument("--center-mode", choices=("walkable", "bounds", "none"), default="walkable")
    parser.add_argument("--ground-mode", choices=("local", "min", "none"), default="local")
    parser.add_argument("--ground-radius", type=float, default=2.0)
    parser.add_argument("--ground-percentile", type=float, default=10.0)
    parser.add_argument("--walkable-search-radius", type=float, default=12.0)
    parser.add_argument("--walkable-max-slope", type=float, default=20.0)
    parser.add_argument("--walkable-min-area", type=float, default=1.2)
    parser.add_argument("--origin-x", type=float)
    parser.add_argument("--origin-y", type=float)
    parser.add_argument("--ground-z", type=float)
    return parser.parse_args(argv)


def import_scene(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"Unsupported scene format: {suffix}")


def mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def root_objects(objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    roots = set()
    for obj in objects:
        root = obj
        while root.parent is not None:
            root = root.parent
        roots.add(root)
    return list(roots)


def triangle_count(objects: list[bpy.types.Object]) -> int:
    return sum(
        sum(max(0, len(poly.vertices) - 2) for poly in obj.data.polygons)
        for obj in objects
    )


def world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        raise ValueError("Imported scene contains no mesh bounds")
    minimum = Vector(tuple(min(point[i] for point in points) for i in range(3)))
    maximum = Vector(tuple(max(point[i] for point in points) for i in range(3)))
    return minimum, maximum


def level_scene(objects: list[bpy.types.Object], mode: str) -> dict:
    """Rotate a sheet-like scan so its broad surface is horizontal."""
    if mode == "none":
        return {"mode": mode, "applied": False}

    sample_budget = 100_000
    vertex_count = sum(len(obj.data.vertices) for obj in objects)
    stride = max(1, math.ceil(vertex_count / sample_budget))
    samples = []
    for obj in objects:
        matrix = obj.matrix_world
        samples.extend(
            tuple(matrix @ vertex.co)
            for index, vertex in enumerate(obj.data.vertices)
            if index % stride == 0
        )
    if len(samples) < 3:
        raise ValueError("At least three vertices are required for PCA leveling")

    points = np.asarray(samples, dtype=np.float64)
    covariance = np.cov(points - points.mean(axis=0), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = Vector(eigenvectors[:, 0].tolist()).normalized()
    if normal.z < 0:
        normal.negate()
    angle_degrees = math.degrees(normal.angle(Vector((0.0, 0.0, 1.0))))
    rotation = normal.rotation_difference(Vector((0.0, 0.0, 1.0)))
    rotation_matrix = rotation.to_matrix().to_4x4()
    for root in root_objects(objects):
        root.matrix_world = rotation_matrix @ root.matrix_world
    bpy.context.view_layer.update()
    return {
        "mode": mode,
        "applied": angle_degrees > 0.01,
        "estimated_surface_normal": list(normal),
        "rotation_degrees": angle_degrees,
        "pca_eigenvalues": eigenvalues.tolist(),
        "sample_count": len(samples),
    }


def add_vertex_color_material(obj: bpy.types.Object) -> str | None:
    if obj.data.materials or not obj.data.color_attributes:
        return None

    color_name = obj.data.color_attributes.active_color_name
    if not color_name:
        color_name = obj.data.color_attributes[0].name

    material = bpy.data.materials.new(name="ScanVertexColors")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    color = nodes.new("ShaderNodeVertexColor")
    color.layer_name = color_name
    links.new(color.outputs["Color"], principled.inputs["Base Color"])
    links.new(color.outputs["Alpha"], principled.inputs["Alpha"])
    principled.inputs["Roughness"].default_value = 1.0
    obj.data.materials.append(material)
    return color_name


def local_ground_z(
    objects: list[bpy.types.Object],
    center_x: float,
    center_y: float,
    radius: float,
    percentile: float,
) -> tuple[float, int]:
    if radius <= 0:
        raise ValueError("--ground-radius must be positive")
    if not 0 <= percentile <= 100:
        raise ValueError("--ground-percentile must be between 0 and 100")

    radius_sq = radius * radius
    heights = []
    for obj in objects:
        matrix = obj.matrix_world
        for vertex in obj.data.vertices:
            point = matrix @ vertex.co
            if (point.x - center_x) ** 2 + (point.y - center_y) ** 2 <= radius_sq:
                heights.append(point.z)

    if not heights:
        raise ValueError(
            f"No scene vertices within {radius:.2f} m of the selected origin; "
            "increase --ground-radius or pass --ground-z"
        )
    heights.sort()
    index = round((len(heights) - 1) * percentile / 100)
    return heights[index], len(heights)


def find_walkable_anchor(
    objects: list[bpy.types.Object],
    seed_x: float,
    seed_y: float,
    search_radius: float,
    max_slope_degrees: float,
    min_patch_area: float,
    *,
    cell_size: float = 0.5,
    height_band: float = 0.15,
    target_z: float | None = None,
) -> dict:
    """Find a coherent, near-horizontal patch near the requested origin.

    A single flat triangle is not enough: a dancer's two feet can still land
    off it. Faces are first binned by XY and height, then neighboring bins are
    combined into roughly 1.5 m-wide support patches. The nearest qualifying
    patch wins, with a small preference for more area and lower terrain.
    """
    if search_radius <= 0 or min_patch_area <= 0:
        raise ValueError("walkable search radius and minimum area must be positive")
    if not 0 <= max_slope_degrees < 90:
        raise ValueError("walkable maximum slope must be between 0 and 90 degrees")

    normal_threshold = math.cos(math.radians(max_slope_degrees))
    radius_sq = search_radius * search_radius
    bins = {}

    for obj in objects:
        matrix = obj.matrix_world
        normal_matrix = matrix.to_3x3().inverted().transposed()
        scale = matrix.to_scale()
        area_scale = max(scale.x * scale.y, scale.x * scale.z, scale.y * scale.z)
        for polygon in obj.data.polygons:
            normal = (normal_matrix @ polygon.normal).normalized()
            if abs(normal.z) < normal_threshold:
                continue
            center = matrix @ polygon.center
            distance_sq = (center.x - seed_x) ** 2 + (center.y - seed_y) ** 2
            if distance_sq > radius_sq:
                continue
            key = (
                math.floor(center.x / cell_size),
                math.floor(center.y / cell_size),
                math.floor(center.z / height_band),
            )
            area = polygon.area * area_scale
            entry = bins.setdefault(
                key,
                {"area": 0.0, "weighted": Vector((0.0, 0.0, 0.0)), "faces": 0},
            )
            entry["area"] += area
            entry["weighted"] += center * area
            entry["faces"] += 1

    if not bins:
        raise ValueError("No near-horizontal faces found in the walkable search area")

    patches = []
    for key in bins:
        area = 0.0
        weighted = Vector((0.0, 0.0, 0.0))
        faces = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (0,):
                    neighbor = bins.get((key[0] + dx, key[1] + dy, key[2] + dz))
                    if not neighbor:
                        continue
                    area += neighbor["area"]
                    weighted += neighbor["weighted"]
                    faces += neighbor["faces"]
        if area < min_patch_area:
            continue
        center = weighted / area
        distance = math.hypot(center.x - seed_x, center.y - seed_y)
        patches.append({
            "center": center,
            "area": area,
            "faces": faces,
            "distance": distance,
        })

    if not patches:
        largest = max(bins.values(), key=lambda entry: entry["area"])
        raise ValueError(
            "No coherent walkable patch met the minimum area; "
            f"largest single cell was {largest['area']:.3f} m^2"
        )

    heights = sorted(patch["center"].z for patch in patches)
    low = heights[round((len(heights) - 1) * 0.1)]
    high = heights[round((len(heights) - 1) * 0.9)]
    height_span = max(high - low, 0.1)
    for patch in patches:
        height_penalty = max(0.0, min(1.0, (patch["center"].z - low) / height_span))
        area_credit = min(patch["area"], 2.0) * 0.35
        patch["score"] = patch["distance"] + height_penalty * 0.5 - area_credit
        if target_z is not None:
            patch["score"] += abs(patch["center"].z - target_z) * 4.0

    selected = min(patches, key=lambda patch: patch["score"])
    center = selected["center"]
    return {
        "center": [center.x, center.y, center.z],
        "support_area_m2": selected["area"],
        "support_faces": selected["faces"],
        "distance_from_seed_m": selected["distance"],
        "candidate_patches": len(patches),
        "max_slope_degrees": max_slope_degrees,
        "cell_size_m": cell_size,
        "height_band_m": height_band,
    }


def apply_scene_transform(
    objects: list[bpy.types.Object],
    *,
    scale: float,
    level_mode: str,
    center_mode: str,
    ground_mode: str,
    ground_radius: float,
    ground_percentile: float,
    walkable_search_radius: float,
    walkable_max_slope: float,
    walkable_min_area: float,
    origin_x: float | None,
    origin_y: float | None,
    ground_z: float | None,
) -> dict:
    if scale <= 0:
        raise ValueError("--scale must be positive")

    roots = root_objects(objects)
    for root in roots:
        root.scale *= scale
    bpy.context.view_layer.update()
    leveling = level_scene(objects, level_mode)

    minimum, maximum = world_bounds(objects)
    seed_x = origin_x if origin_x is not None else (
        (minimum.x + maximum.x) / 2 if center_mode in {"walkable", "bounds"} else 0.0
    )
    seed_y = origin_y if origin_y is not None else (
        (minimum.y + maximum.y) / 2 if center_mode in {"walkable", "bounds"} else 0.0
    )
    walkable = None
    if center_mode == "walkable":
        walkable = find_walkable_anchor(
            objects,
            seed_x,
            seed_y,
            walkable_search_radius,
            walkable_max_slope,
            walkable_min_area,
        )
        chosen_x, chosen_y, walkable_z = walkable["center"]
    else:
        chosen_x, chosen_y = seed_x, seed_y
    local_sample_count = 0
    if ground_z is not None:
        chosen_z = ground_z
    elif walkable is not None:
        chosen_z = walkable_z
    elif ground_mode == "local":
        chosen_z, local_sample_count = local_ground_z(
            objects,
            chosen_x,
            chosen_y,
            ground_radius,
            ground_percentile,
        )
    elif ground_mode == "min":
        chosen_z = minimum.z
    else:
        chosen_z = 0.0

    offset = Vector((-chosen_x, -chosen_y, -chosen_z))
    for root in roots:
        root.location += offset
    bpy.context.view_layer.update()
    return {
        "scale": scale,
        "leveling": leveling,
        "source_origin": [chosen_x, chosen_y, chosen_z],
        "offset": list(offset),
        "ground_mode": ground_mode,
        "local_ground_radius_m": ground_radius,
        "local_ground_percentile": ground_percentile,
        "local_ground_sample_count": local_sample_count,
        "walkable_anchor": walkable,
    }


def decimate(objects: list[bpy.types.Object], target_triangles: int) -> tuple[int, int]:
    before = triangle_count(objects)
    if target_triangles <= 0:
        raise ValueError("--target-triangles must be positive")
    if before <= target_triangles:
        return before, before

    ratio = target_triangles / before
    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        modifier = obj.modifiers.new(name="ViewerTriangleBudget", type="DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        obj.data.validate(clean_customdata=False)
        obj.data.update()
        obj.select_set(False)

    after = triangle_count(objects)
    if after > target_triangles * 1.01:
        raise ValueError(
            f"Decimation missed target: {after:,} > {target_triangles:,} triangles"
        )
    return before, after


def remove_oversized_faces(
    objects: list[bpy.types.Object], max_face_area: float
) -> int:
    """Remove reconstruction hole-fill faces that read as giant flat slabs."""
    if max_face_area <= 0:
        return 0
    removed = 0
    for obj in objects:
        scale = obj.matrix_world.to_scale()
        area_scale = max(scale.x * scale.y, scale.x * scale.z, scale.y * scale.z)
        mesh = bmesh.new()
        mesh.from_mesh(obj.data)
        oversized = [
            face for face in mesh.faces
            if face.calc_area() * area_scale > max_face_area
        ]
        removed += len(oversized)
        bmesh.ops.delete(mesh, geom=oversized, context="FACES")
        mesh.to_mesh(obj.data)
        mesh.free()
        obj.data.update()
    return removed


def validate_export(
    output: Path,
    target_triangles: int,
    max_output_mb: float,
    ground_radius: float,
    ground_percentile: float,
    center_mode: str,
    walkable_max_slope: float,
    walkable_min_area: float,
) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(output))
    objects = mesh_objects()
    if not objects:
        raise ValueError("Exported GLB re-imported without mesh objects")

    minimum, maximum = world_bounds(objects)
    triangles = triangle_count(objects)
    material_count = sum(len(obj.data.materials) for obj in objects)
    support = None
    if center_mode == "walkable":
        support = find_walkable_anchor(
            objects,
            0.0,
            0.0,
            min(1.5, ground_radius),
            walkable_max_slope,
            walkable_min_area * 0.75,
            target_z=0.0,
        )
        origin_ground = support["center"][2]
        origin_ground_samples = support["support_faces"]
    else:
        origin_ground, origin_ground_samples = local_ground_z(
            objects,
            0.0,
            0.0,
            ground_radius,
            ground_percentile,
        )
    ground_error = abs(origin_ground)
    size_mb = output.stat().st_size / 1_000_000
    result = {
        "triangles": triangles,
        "mesh_objects": len(objects),
        "materials": material_count,
        "bounds_blender_z_up_m": {
            "min": list(minimum),
            "max": list(maximum),
            "dimensions": list(maximum - minimum),
        },
        "ground_error_m": ground_error,
        "origin_ground_samples": origin_ground_samples,
        "walkable_support": support,
        "size_mb": size_mb,
        "checks": {
            "triangle_budget": triangles <= target_triangles * 1.01,
            "file_size_budget": size_mb <= max_output_mb,
            "ground_at_zero": ground_error <= 0.05,
            "has_material": material_count > 0,
            "has_walkable_support": center_mode != "walkable" or support is not None,
            "nonempty_file": output.stat().st_size > 0,
        },
    }
    result["failed_checks"] = [
        name for name, passed in result["checks"].items() if not passed
    ]
    return result


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    report = Path(args.report).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_scene(source)
    objects = mesh_objects()
    if not objects:
        raise ValueError(f"No mesh objects imported from {source}")

    source_stats = {
        "mesh_objects": len(objects),
        "triangles": triangle_count(objects),
        "vertices": sum(len(obj.data.vertices) for obj in objects),
        "uv_layers": sum(len(obj.data.uv_layers) for obj in objects),
        "color_attributes": [
            attribute.name
            for obj in objects
            for attribute in obj.data.color_attributes
        ],
        "materials": sum(len(obj.data.materials) for obj in objects),
    }
    color_layers = [add_vertex_color_material(obj) for obj in objects]
    transform = apply_scene_transform(
        objects,
        scale=args.scale,
        level_mode=args.level_mode,
        center_mode=args.center_mode,
        ground_mode=args.ground_mode,
        ground_radius=args.ground_radius,
        ground_percentile=args.ground_percentile,
        walkable_search_radius=args.walkable_search_radius,
        walkable_max_slope=args.walkable_max_slope,
        walkable_min_area=args.walkable_min_area,
        origin_x=args.origin_x,
        origin_y=args.origin_y,
        ground_z=args.ground_z,
    )
    before, after = decimate(objects, args.target_triangles)
    oversized_faces_removed = remove_oversized_faces(objects, args.max_face_area)

    if args.center_mode == "walkable":
        support_after_decimation = find_walkable_anchor(
            objects,
            0.0,
            0.0,
            min(1.5, args.ground_radius),
            args.walkable_max_slope,
            args.walkable_min_area * 0.75,
            target_z=0.0,
        )
        ground_adjustment = -support_after_decimation["center"][2]
        for root in root_objects(objects):
            root.location.z += ground_adjustment
        bpy.context.view_layer.update()
        transform["post_decimation_ground_adjustment_m"] = ground_adjustment
        transform["post_decimation_support"] = support_after_decimation

    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        export_animations=False,
        export_yup=True,
        export_apply=True,
    )

    output_stats = validate_export(
        output,
        args.target_triangles,
        args.max_output_mb,
        args.ground_radius,
        args.ground_percentile,
        args.center_mode,
        args.walkable_max_slope,
        args.walkable_min_area,
    )
    if args.center_mode == "walkable" and not output_stats["checks"]["ground_at_zero"]:
        exported_ground_adjustment = -output_stats["walkable_support"]["center"][2]
        for root in root_objects(mesh_objects()):
            root.location.z += exported_ground_adjustment
        bpy.context.view_layer.update()
        bpy.ops.export_scene.gltf(
            filepath=str(output),
            export_format="GLB",
            export_animations=False,
            export_yup=True,
            export_apply=True,
        )
        transform["post_export_ground_adjustment_m"] = exported_ground_adjustment
        output_stats = validate_export(
            output,
            args.target_triangles,
            args.max_output_mb,
            args.ground_radius,
            args.ground_percentile,
            args.center_mode,
            args.walkable_max_slope,
            args.walkable_min_area,
        )
    payload = {
        "source_path": str(source),
        "output_path": str(output),
        "contract": {
            "units": "meters (assumed after --scale)",
            "up_axis": "Y (glTF export)",
            "ground": "y=0",
            "origin": "nearby walkable patch unless overridden",
        },
        "source": source_stats,
        "processing": {
            "target_triangles": args.target_triangles,
            "max_output_mb": args.max_output_mb,
            "max_face_area_m2": args.max_face_area,
            "oversized_faces_removed": oversized_faces_removed,
            "triangles_before": before,
            "triangles_after_blender_decimate": after,
            "vertex_color_layers_used": [layer for layer in color_layers if layer],
            "transform": transform,
        },
        "output": output_stats,
    }
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if output_stats["failed_checks"]:
        raise ValueError(
            f"GLB validation failed: {', '.join(output_stats['failed_checks'])}; "
            f"ground={output_stats['ground_error_m']:.4f} m"
        )

    print(
        "[prepare_scene_glb] "
        f"{before:,} -> {output_stats['triangles']:,} triangles; "
        f"{output_stats['size_mb']:.1f} MB"
    )
    print(f"[prepare_scene_glb] GLB: {output}")
    print(f"[prepare_scene_glb] report: {report}")
    print("[prepare_scene_glb] DONE")


if __name__ == "__main__":
    main()
