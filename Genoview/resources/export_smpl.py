from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import smplx


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent.parent
MODEL_PATH = REPO_DIR / "272-dim-Motion-Representation" / "body_models" / "human_model_files"
UV_OBJ_PATH = Path("/Users/shinn/Downloads/smpl_uv_20200910 (1)/smpl_uv.obj")
OUTPUT_BIN_PATH = BASE_DIR / "SMPL.bin"
OUTPUT_MAP_PATH = BASE_DIR / "SMPL_vertex_source.npy"

SMPL_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)


def parse_obj_with_uv(obj_path: Path):
    vertices = []
    texcoords = []
    vertex_faces = []
    texcoord_faces = []

    with obj_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                texcoords.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                face_vertices = []
                face_texcoords = []
                for token in line.split()[1:]:
                    parts = token.split("/")
                    face_vertices.append(int(parts[0]) - 1)
                    face_texcoords.append(int(parts[1]) - 1)
                vertex_faces.append(face_vertices)
                texcoord_faces.append(face_texcoords)

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(texcoords, dtype=np.float32),
        np.asarray(vertex_faces, dtype=np.int32),
        np.asarray(texcoord_faces, dtype=np.int32),
    )


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    tris = vertices[faces]
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals.astype(np.float32)


def top4_bone_weights(weights: np.ndarray):
    order = np.argsort(weights, axis=1)[:, ::-1][:, :4]
    top_weights = np.take_along_axis(weights, order, axis=1).astype(np.float32)
    top_weights /= np.sum(top_weights, axis=1, keepdims=True) + 1e-8
    top_ids = order.astype(np.uint8)
    top_ids[top_weights <= 0.0] = 0
    return top_ids, top_weights


def build_expanded_mesh(
    template_vertices: np.ndarray,
    template_weights: np.ndarray,
    template_faces: np.ndarray,
    obj_vertex_faces: np.ndarray,
    texcoords: np.ndarray,
    texcoord_faces: np.ndarray,
):
    if not np.array_equal(template_faces, obj_vertex_faces):
        raise ValueError("OBJ face topology does not match the SMPL template faces.")

    vertex_map: dict[tuple[int, int], int] = {}
    expanded_positions = []
    expanded_texcoords = []
    expanded_source_indices = []
    expanded_faces = []

    for face_vertices, face_texcoords in zip(template_faces, texcoord_faces):
        expanded_face = []
        for vertex_index, texcoord_index in zip(face_vertices, face_texcoords):
            key = (int(vertex_index), int(texcoord_index))
            expanded_index = vertex_map.get(key)
            if expanded_index is None:
                expanded_index = len(expanded_positions)
                vertex_map[key] = expanded_index
                expanded_positions.append(template_vertices[vertex_index])
                expanded_texcoords.append(texcoords[texcoord_index])
                expanded_source_indices.append(vertex_index)
            expanded_face.append(expanded_index)
        expanded_faces.append(expanded_face)

    expanded_positions = np.asarray(expanded_positions, dtype=np.float32)
    expanded_texcoords = np.asarray(expanded_texcoords, dtype=np.float32)
    expanded_faces = np.asarray(expanded_faces, dtype=np.uint16)
    expanded_source_indices = np.asarray(expanded_source_indices, dtype=np.int32)
    expanded_normals = vertex_normals(expanded_positions, expanded_faces.astype(np.int32))
    expanded_weights = template_weights[expanded_source_indices]
    expanded_bone_ids, expanded_bone_weights = top4_bone_weights(expanded_weights)

    return (
        expanded_positions,
        expanded_texcoords,
        expanded_normals,
        expanded_bone_ids,
        expanded_bone_weights,
        expanded_faces,
        expanded_source_indices,
    )


def write_bin(
    output_path: Path,
    vertices: np.ndarray,
    texcoords: np.ndarray,
    normals: np.ndarray,
    bone_ids: np.ndarray,
    bone_weights: np.ndarray,
    faces: np.ndarray,
    joint_names: tuple[str, ...],
    parents: np.ndarray,
    joint_positions: np.ndarray,
):
    with output_path.open("wb") as f:
        f.write(struct.pack("I", len(vertices)))
        f.write(struct.pack("I", len(faces)))
        f.write(struct.pack("I", len(joint_names)))
        f.write(vertices.astype(np.float32).tobytes())
        f.write(texcoords.astype(np.float32).tobytes())
        f.write(normals.astype(np.float32).tobytes())
        f.write(bone_ids.astype(np.uint8).tobytes())
        f.write(bone_weights.astype(np.float32).tobytes())
        f.write(faces.astype(np.uint16).tobytes())

        for name, parent in zip(joint_names, parents):
            f.write(struct.pack("32si", name.encode("ascii"), int(parent)))

        for position in joint_positions:
            f.write(
                struct.pack(
                    "ffffffffff",
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                )
            )


if __name__ == "__main__":
    obj_vertices, obj_texcoords, obj_vertex_faces, obj_texcoord_faces = parse_obj_with_uv(UV_OBJ_PATH)

    model = smplx.create(
        model_path=str(MODEL_PATH),
        model_type="smpl",
        gender="NEUTRAL",
        batch_size=1,
    )
    output = model(return_verts=True)

    template_vertices = model.v_template.detach().cpu().numpy().astype(np.float32)
    template_faces = model.faces.astype(np.int32)
    template_weights = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
    parents = model.parents.detach().cpu().numpy().astype(np.int32)
    joint_positions = output.joints[0, : len(SMPL_JOINT_NAMES)].detach().cpu().numpy().astype(np.float32)

    if obj_vertices.shape[0] != template_vertices.shape[0]:
        raise ValueError("OBJ vertex count does not match the SMPL template vertex count.")

    (
        expanded_positions,
        expanded_texcoords,
        expanded_normals,
        expanded_bone_ids,
        expanded_bone_weights,
        expanded_faces,
        expanded_source_indices,
    ) = build_expanded_mesh(
        template_vertices,
        template_weights,
        template_faces,
        obj_vertex_faces,
        obj_texcoords,
        obj_texcoord_faces,
    )

    write_bin(
        OUTPUT_BIN_PATH,
        expanded_positions,
        expanded_texcoords,
        expanded_normals,
        expanded_bone_ids,
        expanded_bone_weights,
        expanded_faces,
        SMPL_JOINT_NAMES,
        parents[: len(SMPL_JOINT_NAMES)],
        joint_positions,
    )
    np.save(OUTPUT_MAP_PATH, expanded_source_indices)

    print(f"Wrote {OUTPUT_BIN_PATH}")
    print(f"  vertex count: {len(expanded_positions)}")
    print(f"  triangle count: {len(expanded_faces)}")
    print(f"  bone count: {len(SMPL_JOINT_NAMES)}")
    print(f"Wrote {OUTPUT_MAP_PATH}")
