# Copyright 2026 InsightOS
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""把已加载的 MuJoCo 模型导出为浏览器可复用的 GLB。

本模块只依赖 ``MjModel``/``MjData`` 的公开数组，不读取 MJCF、URDF 或宿主路径。
native、robosuite 与 LIBERO 都把已经加载好的 model/data 交给同一个导出器。
导出的 body 节点是扁平的世界位姿节点；浏览器按 ``dynamic_node_order`` 更新它们，
因此 Robot Link 与自由物体可以在不重新加载 GLB 的情况下连续运动。

坐标仍保持 MuJoCo 的右手 Z-up。GLB 顶层节点只做一次 Z-up→Three.js Y-up 的
旋转，三个页面不得再自行交换坐标轴或取反，避免语义地图与物理画面左右镜像。
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_FLOAT = 5126
_UINT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_BUFFER = 34963

# MuJoCo 的 geom type 数值在原生 binding 与 mujoco-py 中保持一致。这里使用
# 数据模型常量而不导入任一 binding，保证同一个 Wheel 可以服务三种 Profile。
_GEOM_PLANE = 0
_GEOM_HFIELD = 1
_GEOM_SPHERE = 2
_GEOM_CAPSULE = 3
_GEOM_ELLIPSOID = 4
_GEOM_CYLINDER = 5
_GEOM_BOX = 6
_GEOM_MESH = 7
@dataclass(frozen=True)
class CameraDescriptor:
    camera_id: str
    name: str
    position: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]
    fovy: float


@dataclass(frozen=True)
class ExportedViewerScene:
    content: bytes
    dynamic_node_order: Tuple[str, ...]
    cameras: Tuple[CameraDescriptor, ...]


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * ((-len(data)) % 4)


def _png_rgb(width: int, height: int, channels: int, pixels: np.ndarray) -> bytes:
    """把 MuJoCo 内存纹理编码成无额外依赖的 PNG。

    Runtime Pack 不应为视觉导出引入 Pillow 或 Blender。MuJoCo 纹理是 1～4
    通道 uint8；这里统一为 RGB/RGBA，逐行使用 PNG filter=0。
    """

    source = np.asarray(pixels, dtype=np.uint8).reshape(height, width, channels)
    if channels == 1:
        source = np.repeat(source, 3, axis=2)
    elif channels == 2:
        source = np.concatenate((np.repeat(source[:, :, :1], 3, axis=2), source[:, :, 1:2]), axis=2)
    elif channels > 4:
        source = source[:, :, :4]
    color_type = 6 if source.shape[2] == 4 else 2

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    scanlines = b"".join(b"\x00" + row.tobytes() for row in source)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(scanlines, 6)),
            chunk(b"IEND", b""),
        )
    )


def _xyzw(raw_wxyz: Sequence[float]) -> Tuple[float, float, float, float]:
    return (float(raw_wxyz[1]), float(raw_wxyz[2]), float(raw_wxyz[3]), float(raw_wxyz[0]))


def _quat_matrix(wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in wxyz)
    length = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / length, x / length, y / length, z / length
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_xyzw(raw: Sequence[float]) -> Tuple[float, float, float, float]:
    """把 MuJoCo 的行主序 3x3 世界旋转矩阵转换为归一化 xyzw。"""
    matrix = np.asarray(raw, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x, y, z = (
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x, y, z = (
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x, y, z = (
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    length = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return (x / length, y / length, z / length, w / length)


def _matrix(position: Sequence[float], quaternion_wxyz: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _quat_matrix(quaternion_wxyz)
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    return result


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    triangle = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], triangle)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = (0.0, 0.0, 1.0)
    return normals.astype("<f4")


def _box() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
            [1, -1, 1],
        ],
        dtype="<f4",
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 6, 5],
            [4, 7, 6],
            [0, 1, 7],
            [0, 7, 4],
            [3, 5, 6],
            [3, 6, 2],
            [0, 4, 5],
            [0, 5, 3],
            [1, 2, 6],
            [1, 6, 7],
        ],
        dtype="<u4",
    )
    # 立方体的六个面不能共享顶点法线。共享后的平均法线会让平面在光照下
    # 呈现为圆滑曲面，并产生明显的三角形明暗分界。这里将每个三角面展开，
    # 为同一面的顶点写入一致法线，确保箱体、托盘等资产保持真实硬边。
    flat_vertices = vertices[faces.reshape(-1)]
    triangles = flat_vertices.reshape(-1, 3, 3).astype(np.float64)
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > 1e-12
    face_normals[valid] /= lengths[valid, None]
    face_normals[~valid] = (0.0, 0.0, 1.0)
    flat_normals = np.repeat(face_normals, 3, axis=0).astype("<f4")
    flat_faces = np.arange(len(flat_vertices), dtype="<u4").reshape(-1, 3)
    return flat_vertices.astype("<f4"), flat_normals, flat_faces


def _uv_sphere(rows: int = 12, columns: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: List[Tuple[float, float, float]] = []
    for row in range(rows + 1):
        phi = math.pi * row / rows
        for column in range(columns):
            theta = 2.0 * math.pi * column / columns
            vertices.append(
                (math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi))
            )
    faces: List[Tuple[int, int, int]] = []
    for row in range(rows):
        for column in range(columns):
            nxt = (column + 1) % columns
            a, b = row * columns + column, row * columns + nxt
            c, d = (row + 1) * columns + column, (row + 1) * columns + nxt
            faces.extend(((a, c, b), (b, c, d)))
    positions = np.asarray(vertices, dtype="<f4")
    indices = np.asarray(faces, dtype="<u4")
    normals = positions / np.maximum(np.linalg.norm(positions, axis=1, keepdims=True), 1e-12)
    return positions, normals.astype("<f4"), indices


def _cylinder(columns: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: List[Tuple[float, float, float]] = []
    for z in (-1.0, 1.0):
        for index in range(columns):
            angle = 2.0 * math.pi * index / columns
            vertices.append((math.cos(angle), math.sin(angle), z))
    vertices.extend(((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)))
    faces: List[Tuple[int, int, int]] = []
    for index in range(columns):
        nxt = (index + 1) % columns
        faces.extend(((index, columns + index, nxt), (nxt, columns + index, columns + nxt)))
        faces.append((2 * columns, nxt, index))
        faces.append((2 * columns + 1, columns + index, columns + nxt))
    positions = np.asarray(vertices, dtype="<f4")
    indices = np.asarray(faces, dtype="<u4")
    return positions, _vertex_normals(positions, indices), indices


def _capsule(radius: float, half_length: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造沿局部 Z 轴的真实 capsule，而不是用圆柱近似。

    MuJoCo capsule 的两端球心位于 z=±half_length。移动球体上下半区后，
    中间两圈自然组成圆柱侧面，因此不会再把 Robot Link 显示成平头圆柱。
    """

    positions, normals, indices = _uv_sphere(rows=16, columns=24)
    positions = positions * float(radius)
    positions[:, 2] += np.where(positions[:, 2] >= 0.0, half_length, -half_length)
    return positions.astype("<f4"), normals, indices


def _plane() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """局部 XY 平面，包含可供 MuJoCo material texture 使用的 UV。"""

    positions = np.asarray([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype="<f4")
    normals = np.asarray([[0, 0, 1]] * 4, dtype="<f4")
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype="<u4")
    uv = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="<f4")
    return positions, normals, faces, uv


class _Builder:
    def __init__(self) -> None:
        self.binary = bytearray()
        self.views: List[Dict[str, Any]] = []
        self.accessors: List[Dict[str, Any]] = []
        self.meshes: List[Dict[str, Any]] = []
        self.materials: List[Dict[str, Any]] = []
        self.nodes: List[Dict[str, Any]] = []
        self.images: List[Dict[str, Any]] = []
        self.textures: List[Dict[str, Any]] = []
        self.samplers: List[Dict[str, Any]] = []
        self.extensions_used: set[str] = set()

    def material(
        self,
        rgba: Sequence[float],
        *,
        specular: float = 0.5,
        shininess: float = 0.0,
        emission: float = 0.0,
        reflectance: float = 0.0,
        texture: Optional[int] = None,
        texture_repeat: Tuple[float, float] = (1.0, 1.0),
        name: str = "",
    ) -> int:
        values = [float(np.clip(value, 0.0, 1.0)) for value in rgba]
        normalized_shine = max(float(shininess), 0.0) * 128.0
        roughness = float(np.clip(math.sqrt(2.0 / (normalized_shine + 2.0)), 0.04, 1.0))
        pbr: Dict[str, Any] = {
            "baseColorFactor": values,
            "metallicFactor": 0.0,
            "roughnessFactor": roughness,
        }
        if texture is not None:
            texture_info: Dict[str, Any] = {"index": texture}
            if texture_repeat != (1.0, 1.0):
                texture_info["extensions"] = {
                    "KHR_texture_transform": {
                        "offset": [0.0, 0.0],
                        "scale": [float(texture_repeat[0]), float(texture_repeat[1])],
                    }
                }
                self.extensions_used.add("KHR_texture_transform")
            pbr["baseColorTexture"] = texture_info
        item: Dict[str, Any] = {"pbrMetallicRoughness": pbr}
        if name:
            item["name"] = name
        item["extensions"] = {
            "KHR_materials_specular": {
                "specularFactor": float(np.clip(max(specular, reflectance), 0.0, 1.0))
            }
        }
        self.extensions_used.add("KHR_materials_specular")
        if emission > 0.0:
            item["emissiveFactor"] = [
                float(np.clip(values[index] * emission, 0.0, 1.0)) for index in range(3)
            ]
        if values[3] < 0.999:
            item["alphaMode"] = "BLEND"
        self.materials.append(item)
        return len(self.materials) - 1

    def texture(self, png: bytes) -> int:
        if not self.samplers:
            self.samplers.append(
                {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}
            )
        image = len(self.images)
        self.images.append({"bufferView": self._view(png, None), "mimeType": "image/png"})
        self.textures.append({"sampler": 0, "source": image})
        return len(self.textures) - 1

    def _view(self, raw: bytes, target: Optional[int]) -> int:
        while len(self.binary) % 4:
            self.binary.append(0)
        offset = len(self.binary)
        self.binary.extend(raw)
        item = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target is not None:
            item["target"] = target
        self.views.append(item)
        return len(self.views) - 1

    def _accessor(
        self, values: np.ndarray, component: int, kind: str, target: int, bounds: bool = False
    ) -> int:
        values = np.ascontiguousarray(values)
        item: Dict[str, Any] = {
            "bufferView": self._view(values.tobytes(order="C"), target),
            "componentType": component,
            "count": int(values.shape[0]),
            "type": kind,
        }
        if bounds:
            item["min"] = [float(value) for value in values.min(axis=0)]
            item["max"] = [float(value) for value in values.max(axis=0)]
        self.accessors.append(item)
        return len(self.accessors) - 1

    def mesh(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        faces: np.ndarray,
        material: int,
        uv: Optional[np.ndarray] = None,
    ) -> int:
        self.meshes.append(
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": self._accessor(
                                np.asarray(vertices, dtype="<f4"),
                                _FLOAT,
                                "VEC3",
                                _ARRAY_BUFFER,
                                True,
                            ),
                            "NORMAL": self._accessor(
                                np.asarray(normals, dtype="<f4"), _FLOAT, "VEC3", _ARRAY_BUFFER
                            ),
                        },
                        "indices": self._accessor(
                            np.asarray(faces, dtype="<u4").reshape(-1),
                            _UINT,
                            "SCALAR",
                            _ELEMENT_BUFFER,
                        ),
                        "material": material,
                    }
                ]
            }
        )
        if uv is not None:
            self.meshes[-1]["primitives"][0]["attributes"]["TEXCOORD_0"] = self._accessor(
                np.asarray(uv, dtype="<f4"), _FLOAT, "VEC2", _ARRAY_BUFFER
            )
        return len(self.meshes) - 1

    def finish(self, root_index: int) -> bytes:
        document = {
            "asset": {"version": "2.0", "generator": "semantic-mujoco-visuals"},
            "scene": 0,
            "scenes": [{"nodes": [root_index]}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.views,
            "buffers": [{"byteLength": len(self.binary)}],
        }
        if self.images:
            document["images"] = self.images
        if self.textures:
            document["textures"] = self.textures
        if self.samplers:
            document["samplers"] = self.samplers
        if self.extensions_used:
            document["extensionsUsed"] = sorted(self.extensions_used)

        encoded = _pad4(json.dumps(document, separators=(",", ":")).encode("utf-8"), b" ")
        binary = _pad4(bytes(self.binary))
        total = 12 + 8 + len(encoded) + 8 + len(binary)
        return b"".join(
            (
                struct.pack("<4sII", b"glTF", 2, total),
                struct.pack("<I4s", len(encoded), b"JSON"),
                encoded,
                struct.pack("<I4s", len(binary), b"BIN\x00"),
                binary,
            )
        )


class MujocoVisualExporter:
    """从任意已加载 MjModel 生成一次 GLB，并从 MjData 读取同序位姿。"""

    def __init__(
        self,
        model: Any,
        data: Any,
        mujoco_module: Any = None,
        source_for_body: Optional[Callable[[int], Optional[str]]] = None,
    ) -> None:
        self.model = model
        self.data = data
        self.mj = mujoco_module
        self.source_for_body = source_for_body or (lambda _body_id: None)
        self._body_ids = tuple(range(int(model.nbody)))
        self._node_ids = tuple("node-%06d" % index for index in range(len(self._body_ids)))

    @property
    def dynamic_node_order(self) -> Tuple[str, ...]:
        return self._node_ids

    def export(self) -> ExportedViewerScene:
        builder = _Builder()
        # glTF/Three 使用 Y-up；这一根节点是唯一坐标转换点。
        root_index = len(builder.nodes)
        builder.nodes.append(
            {
                "name": "semantic-z-up-root",
                "rotation": [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
                "children": [],
                "extras": {"coordinate_frame": "z_up_right_handed"},
            }
        )
        material_cache: Dict[Tuple[Any, ...], int] = {}
        texture_cache: Dict[int, int] = {}
        mesh_cache: Dict[Tuple[Any, ...], int] = {}
        hidden_duplicate_geoms = self._hidden_duplicate_geoms()
        for order, body_id in enumerate(self._body_ids):
            source_id = self.source_for_body(body_id)
            body_node = {
                "name": self._node_ids[order],
                "translation": [float(value) for value in self.data.xpos[body_id]],
                "rotation": list(_xyzw(self.data.xquat[body_id])),
                "children": [],
                "extras": {
                    "render_node_id": self._node_ids[order],
                    "source_id": source_id,
                    "selectable": source_id is not None,
                },
            }
            body_index = len(builder.nodes)
            builder.nodes.append(body_node)
            builder.nodes[root_index]["children"].append(body_index)
            for geom_id in np.flatnonzero(np.asarray(self.model.geom_bodyid) == body_id):
                geom_name = None
                if self.mj is not None:
                    geom_name = self.mj.mj_id2name(
                        self.model, self.mj.mjtObj.mjOBJ_GEOM, int(geom_id)
                    )
                if (
                    int(geom_id) in hidden_duplicate_geoms
                    or int(self.model.geom_group[geom_id]) > 2
                    or float(self.model.geom_rgba[geom_id][3]) <= 0.0
                    # Robot 资产中的显式 *_collision 是控制/接触几何，不是视觉
                    # 内容。场景物体本身仍可同时参与碰撞和显示，不能按 contype
                    # 一刀切过滤。
                    or bool(geom_name and geom_name.lower().endswith("_collision"))
                ):
                    continue
                material_key, material_values = self._material_spec(
                    builder, int(geom_id), texture_cache
                )
                material = material_cache.get(material_key)
                if material is None:
                    material = builder.material(**material_values)
                    material_cache[material_key] = material
                vertices, normals, faces, scale, uv = self._geometry(int(geom_id))
                key = (
                    int(self.model.geom_type[geom_id]),
                    int(self.model.geom_dataid[geom_id]),
                    tuple(round(float(v), 6) for v in scale),
                    material,
                )
                mesh = mesh_cache.get(key)
                if mesh is None:
                    mesh = builder.mesh(
                        vertices * np.asarray(scale, dtype=np.float32),
                        normals,
                        faces,
                        material,
                        uv,
                    )
                    mesh_cache[key] = mesh
                local = _matrix(self.model.geom_pos[geom_id], self.model.geom_quat[geom_id])
                geom_node = {
                    "name": "visual-%06d" % len(builder.nodes),
                    "mesh": mesh,
                    "matrix": [float(value) for value in local.T.reshape(-1)],
                }
                builder.nodes.append(geom_node)
                body_node["children"].append(len(builder.nodes) - 1)
        cameras = tuple(self._cameras())
        return ExportedViewerScene(builder.finish(root_index), self._node_ids, cameras)

    def poses(self) -> np.ndarray:
        """返回与 dynamic_node_order 严格同序的 ``xyz + xyzw`` float32。"""
        result = np.empty((len(self._body_ids), 7), dtype="<f4")
        for index, body_id in enumerate(self._body_ids):
            result[index, :3] = self.data.xpos[body_id]
            result[index, 3:] = _xyzw(self.data.xquat[body_id])
        return result

    def _material_spec(
        self, builder: _Builder, geom_id: int, texture_cache: Dict[int, int]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """解析 MuJoCo 的 material 引用；默认 geom_rgba 不能覆盖显式材质。"""
        rgba = tuple(float(value) for value in self.model.geom_rgba[geom_id])
        geom_matid = getattr(self.model, "geom_matid", None)
        mat_id = int(geom_matid[geom_id]) if geom_matid is not None else -1
        specular = 0.5
        shininess = 0.0
        emission = 0.0
        reflectance = 0.0
        texture: Optional[int] = None
        texture_repeat = (1.0, 1.0)
        name = ""

        if mat_id >= 0 and hasattr(self.model, "mat_rgba"):
            material_rgba = tuple(float(value) for value in self.model.mat_rgba[mat_id])
            default_geom_rgba = np.allclose(rgba, (0.5, 0.5, 0.5, 1.0), atol=1e-6)
            if default_geom_rgba:
                rgba = material_rgba
            specular = float(self.model.mat_specular[mat_id])
            shininess = float(self.model.mat_shininess[mat_id])
            emission = float(self.model.mat_emission[mat_id])
            reflectance = float(self.model.mat_reflectance[mat_id])
            repeat_values = np.asarray(self.model.mat_texrepeat[mat_id]).reshape(-1)
            if len(repeat_values) >= 2:
                texture_repeat = (float(repeat_values[0]), float(repeat_values[1]))
                # MuJoCo 的 texuniform 表示纹理尺寸使用世界单位。地面通常只有
                # 一张很小的棋盘纹理；若直接写 texrepeat，200m 平面只铺一张图。
                texuniform = getattr(self.model, "mat_texuniform", None)
                if (
                    int(self.model.geom_type[geom_id]) == _GEOM_PLANE
                    and texuniform is not None
                    and bool(texuniform[mat_id])
                ):
                    half_size = np.abs(np.asarray(self.model.geom_size[geom_id])[:2])
                    texture_repeat = (
                        texture_repeat[0] * max(float(half_size[0]), 1.0),
                        texture_repeat[1] * max(float(half_size[1]), 1.0),
                    )
            mat_texid = getattr(self.model, "mat_texid", None)
            texture_ids = (
                np.asarray(mat_texid[mat_id]).reshape(-1)
                if mat_texid is not None
                else np.asarray([-1])
            )
            candidates = list(texture_ids[1:2]) + list(texture_ids)
            tex_id = next((int(value) for value in candidates if int(value) >= 0), -1)
            if (
                tex_id >= 0
                and hasattr(self.model, "tex_type")
                and int(self.model.tex_type[tex_id]) == 0
            ):
                texture = texture_cache.get(tex_id)
                if texture is None:
                    width = int(self.model.tex_width[tex_id])
                    height = int(self.model.tex_height[tex_id])
                    channels = int(self.model.tex_nchannel[tex_id])
                    address = int(self.model.tex_adr[tex_id])
                    count = width * height * channels
                    pixels = np.asarray(
                        self.model.tex_data[address : address + count], dtype=np.uint8
                    )
                    texture = builder.texture(_png_rgb(width, height, channels, pixels))
                    texture_cache[tex_id] = texture
            if self.mj is not None:
                raw_name = self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_MATERIAL, mat_id)
                name = raw_name or ""
        key = (
            tuple(round(value, 6) for value in rgba),
            mat_id,
            round(specular, 6),
            round(shininess, 6),
            round(emission, 6),
            round(reflectance, 6),
            texture,
            tuple(round(value, 6) for value in texture_repeat),
        )
        return key, {
            "rgba": rgba,
            "specular": specular,
            "shininess": shininess,
            "emission": emission,
            "reflectance": reflectance,
            "texture": texture,
            "texture_repeat": texture_repeat,
            "name": name,
        }

    def _hidden_duplicate_geoms(self) -> set[int]:
        """同位同 Mesh 的重复对保留 MuJoCo 原生画面使用的普通 geom。

        R1 Pro 的部分 Link 同时声明普通 geom 与 ``visualgeom``。旧实现保留
        带材质项并隐藏普通项，导致整台 Robot 在浏览器中变成统一绿色。
        这里只隐藏重复的带材质项；没有重复普通 geom 的末端视觉仍然保留。
        """
        groups: Dict[Tuple[Any, ...], List[int]] = {}
        matids = getattr(self.model, "geom_matid", None)
        for geom_id in range(len(self.model.geom_bodyid)):
            if int(self.model.geom_type[geom_id]) != _GEOM_MESH:
                continue
            key = (
                int(self.model.geom_bodyid[geom_id]),
                int(self.model.geom_dataid[geom_id]),
                tuple(round(float(value), 7) for value in self.model.geom_pos[geom_id]),
                tuple(round(float(value), 7) for value in self.model.geom_quat[geom_id]),
                tuple(round(float(value), 7) for value in self.model.geom_size[geom_id]),
            )
            groups.setdefault(key, []).append(geom_id)

        hidden: set[int] = set()
        for geom_ids in groups.values():
            materialized = [
                geom_id for geom_id in geom_ids if matids is not None and int(matids[geom_id]) >= 0
            ]
            plain = [geom_id for geom_id in geom_ids if matids is None or int(matids[geom_id]) < 0]
            if not materialized or not plain:
                continue
            hidden.update(materialized)
        return hidden

    def _geometry(
        self, geom_id: int
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Tuple[float, float, float],
        Optional[np.ndarray],
    ]:
        geom_type = int(self.model.geom_type[geom_id])
        size = tuple(float(value) for value in self.model.geom_size[geom_id])
        if geom_type == _GEOM_MESH:
            mesh_id = int(self.model.geom_dataid[geom_id])
            start, count = (
                int(self.model.mesh_vertadr[mesh_id]),
                int(self.model.mesh_vertnum[mesh_id]),
            )
            face_start, face_count = (
                int(self.model.mesh_faceadr[mesh_id]),
                int(self.model.mesh_facenum[mesh_id]),
            )
            vertices = np.asarray(self.model.mesh_vert[start : start + count], dtype="<f4")
            faces = np.asarray(
                self.model.mesh_face[face_start : face_start + face_count], dtype="<u4"
            )
            mesh_scale = getattr(self.model, "mesh_scale", None)
            scale = (
                tuple(float(value) for value in mesh_scale[mesh_id])
                if mesh_scale is not None
                else (1.0, 1.0, 1.0)
            )
            normal_start = int(self.model.mesh_normaladr[mesh_id])
            normal_count = int(self.model.mesh_normalnum[mesh_id])
            if normal_start >= 0 and normal_count > 0:
                normals = np.asarray(
                    self.model.mesh_normal[normal_start : normal_start + normal_count], dtype="<f4"
                )
            else:
                normals = _vertex_normals(vertices, faces)
            uv: Optional[np.ndarray] = None
            tex_start = int(self.model.mesh_texcoordadr[mesh_id])
            if tex_start >= 0:
                tex_count = int(self.model.mesh_texcoordnum[mesh_id])
                if tex_count > 0:
                    texcoords = np.asarray(
                        self.model.mesh_texcoord[tex_start : tex_start + tex_count], dtype="<f4"
                    )
                    facet_uv = np.asarray(
                        self.model.mesh_facetexcoord[face_start : face_start + face_count]
                    )
                    if np.all(facet_uv >= 0):
                        facet_normals = np.asarray(
                            self.model.mesh_facenormal[face_start : face_start + face_count]
                        )
                        if not np.all(facet_normals >= 0):
                            facet_normals = faces
                        vertices = vertices[faces.reshape(-1)]
                        normals = normals[facet_normals.reshape(-1)]
                        uv = texcoords[facet_uv.reshape(-1)]
                        faces = np.arange(len(vertices), dtype="<u4").reshape(-1, 3)
            return vertices, normals, faces, scale, uv
        if geom_type in {_GEOM_SPHERE, _GEOM_ELLIPSOID}:
            vertices, normals, faces = _uv_sphere()
            scale = (size[0], size[0], size[0]) if geom_type == _GEOM_SPHERE else size
            return vertices, normals, faces, scale, None
        if geom_type == _GEOM_CAPSULE:
            vertices, normals, faces = _capsule(size[0], size[1])
            return vertices, normals, faces, (1.0, 1.0, 1.0), None
        if geom_type == _GEOM_CYLINDER:
            vertices, normals, faces = _cylinder()
            return vertices, normals, faces, (size[0], size[0], size[1]), None
        if geom_type == _GEOM_PLANE:
            vertices, normals, faces, uv = _plane()
            plane = tuple(max(value, 10.0) if value > 0 else 10.0 for value in size[:2])
            return vertices, normals, faces, (plane[0], plane[1], 1.0), uv
        vertices, normals, faces = _box()
        return vertices, normals, faces, size, None

    def _cameras(self) -> List[CameraDescriptor]:
        result: List[CameraDescriptor] = []
        for camera_id in range(int(self.model.ncam)):
            if hasattr(self.model, "camera_id2name"):
                name = self.model.camera_id2name(camera_id)
            elif self.mj is not None:
                name = self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_CAMERA, camera_id)
            else:
                name = None
            public_name = name or "camera-%d" % camera_id
            result.append(
                CameraDescriptor(
                    camera_id=public_name,
                    name=public_name,
                    position=tuple(float(value) for value in self.data.cam_xpos[camera_id]),
                    quaternion_xyzw=_matrix_xyzw(self.data.cam_xmat[camera_id]),
                    fovy=float(self.model.cam_fovy[camera_id]),
                )
            )
        return result
