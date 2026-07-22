#!/usr/bin/env python3
"""Extract one canonical rig/mesh/animation bundle through Blender's FBX API."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import bpy
import numpy as np


STATIC_SUFFIX = "_Character_output.fbx"
ANIMATION_SUFFIX = "_Animation_Walking_Woman_withSkin.fbx"
CANONICAL_FROM_BLENDER = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def _arguments():
    separator = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(sys.argv[separator + 1:])


def _import_fbx(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.import_scene.fbx(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender failed to import {path}")
    armatures = [
        item for item in bpy.context.scene.objects
        if item.type == "ARMATURE"
    ]
    meshes = [
        item for item in bpy.context.scene.objects if item.type == "MESH"
    ]
    if len(armatures) != 1 or len(meshes) != 1:
        raise ValueError(
            f"expected one armature/mesh, found {len(armatures)}/{len(meshes)}"
        )
    return armatures[0], meshes[0]


def _matrix(value):
    return np.asarray(value, dtype=np.float64)


def _canonical_points(points):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate([
        points, np.ones((*points.shape[:-1], 1), dtype=points.dtype)
    ], axis=-1)
    return (homogeneous @ CANONICAL_FROM_BLENDER.T)[..., :3]


def _canonical_transform(transform):
    inverse = np.linalg.inv(CANONICAL_FROM_BLENDER)
    return CANONICAL_FROM_BLENDER @ transform @ inverse


def _base_mesh(armature, mesh):
    bone_names = [bone.name for bone in armature.data.bones]
    bone_index = {name: index for index, name in enumerate(bone_names)}
    vertices = np.stack([
        _matrix(mesh.matrix_world @ vertex.co)
        for vertex in mesh.data.vertices
    ])
    triangles = []
    corner_uv = []
    if not mesh.data.uv_layers:
        raise ValueError("source mesh has no UV layer")
    uv_data = mesh.data.uv_layers.active.data
    for polygon in mesh.data.polygons:
        if len(polygon.vertices) != 3:
            raise ValueError("source mesh must already be triangulated")
        triangles.append(tuple(polygon.vertices))
        corner_uv.append([
            tuple(uv_data[loop_index].uv)
            for loop_index in polygon.loop_indices
        ])
    weights = np.zeros(
        (len(mesh.data.vertices), len(bone_names)), dtype=np.float64
    )
    group_to_bone = {
        group.index: bone_index[group.name]
        for group in mesh.vertex_groups if group.name in bone_index
    }
    for vertex in mesh.data.vertices:
        for element in vertex.groups:
            if element.group in group_to_bone:
                weights[vertex.index, group_to_bone[element.group]] = (
                    element.weight
                )
    bind_matrices = np.stack([
        CANONICAL_FROM_BLENDER
        @ _matrix(armature.matrix_world @ bone.matrix_local)
        for bone in armature.data.bones
    ])
    rest_bone_endpoints = np.asarray([
        _canonical_points([
            _matrix(armature.matrix_world @ bone.head_local),
            _matrix(armature.matrix_world @ bone.tail_local),
        ])
        for bone in armature.data.bones
    ])
    parents = np.array([
        bone_index[bone.parent.name] if bone.parent else -1
        for bone in armature.data.bones
    ], dtype=np.int64)
    return {
        "vertices": _canonical_points(vertices),
        "triangles": np.asarray(triangles, dtype=np.int64),
        "corner_uv": np.asarray(corner_uv, dtype=np.float64),
        "weights": weights,
        "bone_names": np.asarray(bone_names),
        "bone_parents": parents,
        "bind_matrices": bind_matrices,
        "rest_bone_endpoints": rest_bone_endpoints,
    }


def _evaluated_vertices(mesh):
    dependency_graph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(dependency_graph)
    world = np.stack([
        _matrix(evaluated.matrix_world @ vertex.co)
        for vertex in evaluated.data.vertices
    ])
    return _canonical_points(world)


def _animation(armature, mesh):
    actions = list(bpy.data.actions)
    if len(actions) != 1:
        raise ValueError(f"expected one action, found {len(actions)}")
    action = actions[0]
    first, last = (int(round(value)) for value in action.frame_range)
    frames = np.arange(first, last + 1, dtype=np.int64)
    armature_world = _matrix(armature.matrix_world)
    inverse_armature_world = np.linalg.inv(armature_world)
    inverse_canonical = np.linalg.inv(CANONICAL_FROM_BLENDER)
    vertices = []
    skin_matrices = []
    bone_endpoints = []
    for frame in frames:
        bpy.context.scene.frame_set(int(frame))
        vertices.append(_evaluated_vertices(mesh))
        frame_skin = []
        frame_endpoints = []
        for bone in armature.data.bones:
            pose = armature.pose.bones[bone.name]
            pose_matrix = _matrix(pose.matrix)
            bind_inverse = np.linalg.inv(_matrix(bone.matrix_local))
            world_delta = (
                armature_world @ pose_matrix @ bind_inverse
                @ inverse_armature_world
            )
            frame_skin.append(
                CANONICAL_FROM_BLENDER @ world_delta @ inverse_canonical
            )
            head = _matrix(armature.matrix_world @ pose.head)
            tail = _matrix(armature.matrix_world @ pose.tail)
            frame_endpoints.append(_canonical_points([head, tail]))
        skin_matrices.append(frame_skin)
        bone_endpoints.append(frame_endpoints)
    return {
        "frames": frames,
        "fps": int(bpy.context.scene.render.fps),
        "action_name": action.name,
        "vertices": np.asarray(vertices),
        "skin_matrices": np.asarray(skin_matrices),
        "bone_endpoints": np.asarray(bone_endpoints),
    }


def main():
    args = _arguments()
    archive = args.archive.resolve()
    output = args.output.resolve()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="flesh_h4_extract_") as temporary:
        temporary_path = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.namelist()
            bundle.extractall(temporary_path)
        static_member = next(
            (name for name in members if name.endswith(STATIC_SUFFIX)), None
        )
        animation_member = next(
            (name for name in members if name.endswith(ANIMATION_SUFFIX)), None
        )
        if not static_member or not animation_member:
            raise ValueError("archive lacks expected static/animation FBX members")

        static_armature, static_mesh = _import_fbx(
            temporary_path / static_member
        )
        static = _base_mesh(static_armature, static_mesh)
        static_action = list(bpy.data.actions)
        static_action_name = static_action[0].name if static_action else None

        animated_armature, animated_mesh = _import_fbx(
            temporary_path / animation_member
        )
        animated = _base_mesh(animated_armature, animated_mesh)
        if not np.array_equal(static["triangles"], animated["triangles"]):
            raise ValueError("static and animated triangle topology differ")
        if not np.array_equal(static["bone_names"], animated["bone_names"]):
            raise ValueError("static and animated bone orders differ")
        animation = _animation(animated_armature, animated_mesh)

    weight_sum = static["weights"].sum(axis=1)
    metadata = {
        "format": "flesh-and-bone-h4-rig-v1",
        "source_archive": archive.name,
        "source_sha256": digest,
        "static_member": static_member,
        "animation_member": animation_member,
        "static_action": static_action_name,
        "animation_action": animation["action_name"],
        "fps": animation["fps"],
        "coordinate_system": "right-handed, x right, y up, z back, meters",
        "canonical_from_blender": CANONICAL_FROM_BLENDER.tolist(),
        "vertex_count": int(static["vertices"].shape[0]),
        "triangle_count": int(static["triangles"].shape[0]),
        "bone_count": int(static["bone_names"].shape[0]),
        "weighted_bone_count": int((static["weights"].sum(axis=0) > 0).sum()),
        "unweighted_vertex_count": int((weight_sum == 0).sum()),
        "weight_sum_max_error": float(np.abs(weight_sum - 1).max()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(metadata)),
        rest_vertices=static["vertices"],
        animated_bind_vertices=animated["vertices"],
        triangles=static["triangles"],
        corner_uv=static["corner_uv"],
        weights=static["weights"],
        bone_names=static["bone_names"],
        bone_parents=static["bone_parents"],
        bind_matrices=static["bind_matrices"],
        rest_bone_endpoints=static["rest_bone_endpoints"],
        animation_frames=animation["frames"],
        animation_vertices=animation["vertices"],
        skin_matrices=animation["skin_matrices"],
        bone_endpoints=animation["bone_endpoints"],
    )
    print(json.dumps(metadata, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
