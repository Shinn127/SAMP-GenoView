#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMPL_MODEL = (
    ROOT
    / "272-dim-Motion-Representation"
    / "body_models"
    / "human_model_files"
    / "smpl"
    / "SMPL_NEUTRAL.pkl"
)
DEFAULT_OUTPUT = ROOT / "Genoview" / "resources" / "SMPL.bin"
SMPL_MODEL_NAMES = [
    "Pelvis",
    "Left_hip",
    "Right_hip",
    "Spine1",
    "Left_knee",
    "Right_knee",
    "Spine2",
    "Left_ankle",
    "Right_ankle",
    "Spine3",
    "Left_foot",
    "Right_foot",
    "Neck",
    "Left_collar",
    "Right_collar",
    "Head",
    "Left_shoulder",
    "Right_shoulder",
    "Left_elbow",
    "Right_elbow",
    "Left_wrist",
    "Right_wrist",
    "Left_palm",
    "Right_palm",
]
HUMANML3D_BVH_NAMES = [
    "Pelvis",
    "Left_hip",
    "Left_knee",
    "Left_ankle",
    "Left_foot",
    "Right_hip",
    "Right_knee",
    "Right_ankle",
    "Right_foot",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "Left_collar",
    "Left_shoulder",
    "Left_elbow",
    "Left_wrist",
    "Left_palm",
    "Right_collar",
    "Right_shoulder",
    "Right_elbow",
    "Right_wrist",
    "Right_palm",
]
HUMANML3D_BVH_PARENTS = np.asarray([
    -1,
    0,
    1,
    2,
    3,
    0,
    5,
    6,
    7,
    0,
    9,
    10,
    11,
    12,
    11,
    14,
    15,
    16,
    17,
    11,
    19,
    20,
    21,
    22,
], dtype=np.int32)


def load_smpl_model(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def dense_array(value) -> np.ndarray:
    if hasattr(value, "toarray"):
        return value.toarray()
    return np.asarray(value)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    tris = vertices[faces]
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals.astype(np.float32)


def top4_skinning(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(weights, axis=1)[:, ::-1][:, :4]
    top_weights = np.take_along_axis(weights, order, axis=1)
    top_weights /= np.sum(top_weights, axis=1, keepdims=True) + 1e-8
    order[top_weights <= 0.0] = 0
    return order.astype(np.uint8), top_weights.astype(np.float32)


def write_bin(
    output: Path,
    vertices: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray,
    bone_ids: np.ndarray,
    bone_weights: np.ndarray,
    parents: np.ndarray,
    joints: np.ndarray,
    names: list[str],
) -> None:
    if vertices.shape[0] > np.iinfo(np.uint16).max:
        raise ValueError("Genoview .bin uses uint16 indices; mesh has too many vertices.")

    output.parent.mkdir(parents=True, exist_ok=True)
    texcoords = np.zeros((len(vertices), 2), dtype=np.float32)
    identity_rotation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    unit_scale = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

    with output.open("wb") as handle:
        handle.write(struct.pack("I", len(vertices)))
        handle.write(struct.pack("I", len(faces)))
        handle.write(struct.pack("I", len(parents)))
        handle.write(vertices.astype(np.float32).tobytes())
        handle.write(texcoords.tobytes())
        handle.write(normals.astype(np.float32).tobytes())
        handle.write(bone_ids.tobytes())
        handle.write(bone_weights.tobytes())
        handle.write(faces.astype(np.uint16).tobytes())

        for name, parent in zip(names, parents):
            handle.write(struct.pack("32si", name.encode("ascii"), int(parent)))

        for joint in joints.astype(np.float32):
            handle.write(struct.pack(
                "ffffffffff",
                float(joint[0]),
                float(joint[1]),
                float(joint[2]),
                float(identity_rotation[0]),
                float(identity_rotation[1]),
                float(identity_rotation[2]),
                float(identity_rotation[3]),
                float(unit_scale[0]),
                float(unit_scale[1]),
                float(unit_scale[2]),
            ))


def export_smpl_bin(model_path: Path, output: Path) -> None:
    model = load_smpl_model(model_path)
    vertices = np.asarray(model["v_template"], dtype=np.float32)
    faces = np.asarray(model["f"], dtype=np.uint32)
    model_weights = np.asarray(model["weights"], dtype=np.float32)
    smpl_model_index = {name: i for i, name in enumerate(SMPL_MODEL_NAMES)}
    bvh_to_model = np.asarray([smpl_model_index[name] for name in HUMANML3D_BVH_NAMES])
    weights = model_weights[:, bvh_to_model]
    parents = HUMANML3D_BVH_PARENTS
    model_joints = dense_array(model["J_regressor"]).astype(np.float32) @ vertices
    joints = model_joints[bvh_to_model]
    normals = vertex_normals(vertices, faces)
    bone_ids, bone_weights = top4_skinning(weights)

    write_bin(
        output=output,
        vertices=vertices,
        normals=normals,
        faces=faces,
        bone_ids=bone_ids,
        bone_weights=bone_weights,
        parents=parents,
        joints=joints,
        names=HUMANML3D_BVH_NAMES,
    )
    print(f"Wrote {output}")
    print(f"vertices: {len(vertices)}")
    print(f"triangles: {len(faces)}")
    print(f"bones: {len(parents)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_SMPL_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_smpl_bin(args.model, args.output)


if __name__ == "__main__":
    main()
