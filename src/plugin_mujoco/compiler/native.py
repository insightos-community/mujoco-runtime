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

"""把 Studio SceneDocument 编译为可注册的 MuJoCo 场景定义。"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from plugin_mujoco.models import SceneDescriptor, StrictModel
from plugin_mujoco.scene.catalog import AssetSpec, RobotSpec, SceneDefinition, SensorSpec

from .rules import asset_file_issues, document_rule_issues

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class DocumentTransform(StrictModel):
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


class DocumentAsset(StrictModel):
    id: str
    asset_key: str
    kind: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentNode(StrictModel):
    id: str
    parent_id: str | None = None
    name: str
    kind: Literal[
        "group",
        "object",
        "asset",
        "robot",
        "joint",
        "actuator",
        "sensor",
        "camera",
        "light",
        "material",
        "collision",
        "region",
    ]
    asset_id: str | None = None
    transform: DocumentTransform = Field(default_factory=DocumentTransform)
    properties: dict[str, Any] = Field(default_factory=dict)
    extensions: dict[str, Any] = Field(default_factory=dict)


class SceneDocument(StrictModel):
    """Runtime 收到的纯场景内容，不包含 Framework 的 Project 保存状态。"""

    id: str
    name: str
    description: str | None = None
    scene_kind: str = "scene_document"
    assets: list[DocumentAsset] = Field(default_factory=list)
    nodes: list[DocumentNode] = Field(default_factory=list)
    regions: list[DocumentNode] = Field(default_factory=list)
    physics: dict[str, Any] = Field(default_factory=dict)
    extensions: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AuthoringIssue(StrictModel):
    level: Literal["error", "warning"]
    message: str
    node_id: str | None = None
    field: str | None = None


class ValidationResult(StrictModel):
    valid: bool
    issues: list[AuthoringIssue] = Field(default_factory=list)


class RuntimeBundle(StrictModel):
    """Framework 交给 Runtime 编译的不可变场景输入。"""

    runtime_bundle_id: str
    document_id: str
    revision: int = Field(ge=1)
    scene_version: int = Field(ge=1)
    scene_key: str
    runtime_profile_id: str
    document: SceneDocument
    validation: ValidationResult
    created_at: datetime

    @property
    def build_id(self) -> str:
        """编译器内部沿用的目录标识，不属于公开协议。"""
        return self.runtime_bundle_id

    @property
    def version(self) -> int:
        return self.scene_version


class RuntimeBundleResult(StrictModel):
    runtime_bundle_id: str
    scene_key: str
    runtime_profile_id: str
    runtime_scene_key: str
    valid: bool
    issues: list[AuthoringIssue]
    descriptor: SceneDescriptor | None = None


def validate_scene_document(document: SceneDocument) -> list[AuthoringIssue]:
    issues: list[AuthoringIssue] = []
    if not document.name.strip():
        issues.append(AuthoringIssue(level="error", field="name", message="场景名称不能为空"))
    assets: dict[str, DocumentAsset] = {}
    for asset in document.assets:
        path = Path(asset.asset_key)
        if path.is_absolute() or ".." in path.parts:
            issues.append(
                AuthoringIssue(
                    level="error",
                    field="asset_key",
                    message="资产必须使用资产仓内的相对标识",
                )
            )
        if asset.id in assets:
            issues.append(
                AuthoringIssue(level="error", field="assets", message=f"资产 ID 重复: {asset.id}")
            )
        assets[asset.id] = asset

    nodes: dict[str, DocumentNode] = {}
    all_nodes = [*document.nodes, *document.regions]
    for node in all_nodes:
        if node.id in nodes:
            issues.append(
                AuthoringIssue(level="error", node_id=node.id, field="id", message="节点 ID 重复")
            )
        nodes[node.id] = node
        if node.asset_id and node.asset_id not in assets:
            issues.append(
                AuthoringIssue(
                    level="error",
                    node_id=node.id,
                    field="asset_id",
                    message="节点引用的资产不存在",
                )
            )
        if not any(abs(value) > 1e-12 for value in node.transform.quaternion_xyzw):
            issues.append(
                AuthoringIssue(
                    level="error",
                    node_id=node.id,
                    field="quaternion_xyzw",
                    message="四元数不能全为零",
                )
            )
        if any(value <= 0 for value in node.transform.scale):
            issues.append(
                AuthoringIssue(
                    level="error",
                    node_id=node.id,
                    field="scale",
                    message="缩放必须大于零",
                )
            )
        unsupported_extensions = sorted(set(node.extensions) - {"mujoco"})
        if unsupported_extensions:
            issues.append(
                AuthoringIssue(
                    level="error",
                    node_id=node.id,
                    field="extensions",
                    message=f"不支持的节点扩展: {unsupported_extensions}",
                )
            )
        if "mujoco" in node.extensions:
            issues.append(
                AuthoringIssue(
                    level="error",
                    node_id=node.id,
                    field="extensions.mujoco",
                    message="当前编译器尚未实现节点级 MuJoCo 扩展，不能静默忽略",
                )
            )
    for node in all_nodes:
        if node.parent_id and node.parent_id not in nodes:
            issues.append(
                AuthoringIssue(
                    level="error",
                    node_id=node.id,
                    field="parent_id",
                    message="父节点不存在",
                )
            )
    for node in all_nodes:
        norm = math.sqrt(sum(value * value for value in node.transform.quaternion_xyzw))
        if abs(norm - 1.0) > 1e-4:
            issues.append(
                AuthoringIssue(
                    level="error",
                    node_id=node.id,
                    field="quaternion_xyzw",
                    message="四元数必须归一化",
                )
            )
        if node.kind in {"asset", "joint", "actuator", "material", "collision", "sensor"}:
            issues.append(
                AuthoringIssue(
                    level="error",
                    node_id=node.id,
                    field="kind",
                    message=f"native 编译器尚不支持独立 {node.kind} 节点，请通过资产或 Camera 属性声明",
                )
            )
        seen: set[str] = set()
        current = node
        while current.parent_id:
            if current.id in seen:
                issues.append(
                    AuthoringIssue(
                        level="error",
                        node_id=node.id,
                        field="parent_id",
                        message="场景层级存在循环引用",
                    )
                )
                break
            seen.add(current.id)
            parent = nodes.get(current.parent_id)
            if parent is None:
                break
            current = parent
    unsupported_document_extensions = sorted(set(document.extensions) - {"mujoco"})
    if unsupported_document_extensions:
        issues.append(
            AuthoringIssue(
                level="error",
                field="extensions",
                message=f"不支持的场景扩展: {unsupported_document_extensions}",
            )
        )
    if "mujoco" in document.extensions:
        issues.append(
            AuthoringIssue(
                level="error",
                field="extensions.mujoco",
                message="当前编译器尚未实现场景级 MuJoCo 扩展，不能静默忽略",
            )
        )
    if not any(node.kind == "robot" for node in document.nodes):
        issues.append(
            AuthoringIssue(
                level="error",
                field="nodes",
                message="可运行的 MuJoCo 场景至少需要一个 Robot 节点",
            )
        )
    for node_id, field, message in document_rule_issues(document):
        issues.append(AuthoringIssue(level="error", node_id=node_id, field=field, message=message))
    return issues


def compile_scene_document(
    asset_root: Path,
    authoring_root: Path,
    request: RuntimeBundle,
) -> tuple[SceneDefinition | None, RuntimeBundleResult]:
    if not _SAFE_ID.fullmatch(request.build_id) or not _SAFE_ID.fullmatch(request.scene_key):
        issue = AuthoringIssue(
            level="error",
            field="runtime_bundle_id",
            message="runtime_bundle_id 或 scene_key 格式无效",
        )
        return None, _result(request, [issue])
    issues = [*request.validation.issues, *validate_scene_document(request.document)]
    for node_id, field, message in asset_file_issues(asset_root, request.document):
        issues.append(AuthoringIssue(level="error", node_id=node_id, field=field, message=message))
    if request.document_id != request.document.id:
        issues.append(
            AuthoringIssue(
                level="error", field="document_id", message="document_id 与 document.id 不一致"
            )
        )
    if request.runtime_profile_id != "native-mujoco":
        issues.append(
            AuthoringIssue(
                level="error",
                field="runtime_profile_id",
                message="SceneDocument 编译当前只支持 native-mujoco Profile",
            )
        )
    if not request.validation.valid:
        issues.append(
            AuthoringIssue(level="error", field="validation", message="Framework 校验未通过")
        )
    if any(issue.level == "error" for issue in issues):
        return None, _result(request, issues)

    build_root = (authoring_root / request.build_id).resolve()
    expected_root = authoring_root.resolve()
    build_root.relative_to(expected_root)
    build_root.mkdir(parents=True, exist_ok=True)
    base_xml = build_root / "base.xml"
    resolved = _resolved_nodes(request.document)
    document_nodes = [resolved[node.id] for node in request.document.nodes]
    region_nodes = [resolved[node.id] for node in request.document.regions]
    resolved_document = request.document.model_copy(
        update={"nodes": document_nodes, "regions": region_nodes}
    )
    _write_base_xml(base_xml, resolved_document)

    assets_by_id = {asset.id: asset for asset in request.document.assets}
    camera_nodes = [node for node in document_nodes if node.kind == "camera"]
    sensors = tuple(_sensor_spec(node) for node in camera_nodes)
    robots = tuple(
        _robot_spec(node, assets_by_id, camera_nodes)
        for node in document_nodes
        if node.kind == "robot"
    )
    assets = tuple(
        _asset_spec(node, assets_by_id)
        for node in [*document_nodes, *region_nodes]
        if node.kind in {"object", "region"}
    )
    physics = request.document.physics
    gravity = _tuple(physics.get("gravity_m_s2"), 3, (0.0, 0.0, -9.81))
    timestep = float(physics.get("timestep_seconds", 0.002))
    if timestep <= 0 or timestep > 0.1:
        issue = AuthoringIssue(
            level="error",
            field="physics.timestep_seconds",
            message="物理步长必须大于 0 且不超过 0.1 秒",
        )
        return None, _result(request, [*issues, issue])

    definition = SceneDefinition(
        scene_key=request.scene_key,
        name=request.document.name,
        layout=f"version-{request.version}",
        base_xml=base_xml,
        robot_root=asset_root / "robot",
        object_root=asset_root / "assets" / "objects",
        robots=robots,
        sensors=sensors,
        assets=assets,
        timestep=timestep,
        gravity=(float(gravity[0]), float(gravity[1]), float(gravity[2])),
    )
    descriptor = SceneDescriptor(
        scene_key=request.scene_key,
        name=request.document.name,
        scene_kind="scene_document",
        layouts=[definition.layout],
        robot_models=sorted({robot.model for robot in robots}),
        compatible_runtime_profiles=["native-mujoco"],
        read_only=False,
    )
    result = _result(request, issues)
    result.valid = True
    result.descriptor = descriptor
    request_path = build_root / "build-request.json"
    temporary_path = build_root / "build-request.json.tmp"
    temporary_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(request_path)
    return definition, result


def _resolved_nodes(document: SceneDocument) -> dict[str, DocumentNode]:
    """把父子局部变换组合为世界变换，Group 不会在编译时被悄悄丢弃。"""
    nodes = {node.id: node for node in [*document.nodes, *document.regions]}
    cache: dict[str, DocumentNode] = {}

    def resolve(node: DocumentNode) -> DocumentNode:
        if node.id in cache:
            return cache[node.id]
        if not node.parent_id:
            cache[node.id] = node
            return node
        parent = resolve(nodes[node.parent_id])
        pt, ct = parent.transform, node.transform
        scaled = tuple(ct.position[i] * pt.scale[i] for i in range(3))
        rotated = _rotate_vector(pt.quaternion_xyzw, scaled)
        world = DocumentTransform(
            position=tuple(pt.position[i] + rotated[i] for i in range(3)),
            quaternion_xyzw=_quaternion_multiply(pt.quaternion_xyzw, ct.quaternion_xyzw),
            scale=tuple(pt.scale[i] * ct.scale[i] for i in range(3)),
        )
        cache[node.id] = node.model_copy(update={"transform": world})
        return cache[node.id]

    for item in nodes.values():
        resolve(item)
    return cache


def _quaternion_multiply(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    value = (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )
    norm = math.sqrt(sum(item * item for item in value))
    return tuple(item / norm for item in value)


def _rotate_vector(
    q: tuple[float, float, float, float], v: tuple[float, float, float]
) -> tuple[float, float, float]:
    x, y, z, w = q
    vx, vy, vz = v
    # q * (v, 0) * conjugate(q) 的展开式，场景编译无需引入 NumPy。
    tx, ty, tz = (2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx))
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def _result(request: RuntimeBundle, issues: list[AuthoringIssue]) -> RuntimeBundleResult:
    return RuntimeBundleResult(
        runtime_bundle_id=request.runtime_bundle_id,
        scene_key=request.scene_key,
        runtime_profile_id=request.runtime_profile_id,
        runtime_scene_key=request.scene_key,
        valid=not any(issue.level == "error" for issue in issues),
        issues=issues,
    )


def _robot_spec(
    node: DocumentNode,
    assets: dict[str, DocumentAsset],
    camera_nodes: list[DocumentNode],
) -> RobotSpec:
    asset = assets.get(node.asset_id or "")
    model = str(
        node.properties.get("model")
        or (asset.metadata.get("model") if asset else "")
        or _asset_model(asset)
        or "r1_pro_chassis"
    )
    sensor_names = node.properties.get("sensor_names")
    if not sensor_names:
        sensor_names = [
            camera.name
            for camera in camera_nodes
            if camera.properties.get("robot_id") in {None, "", node.id}
        ]
    return RobotSpec(
        robot_id=str(node.properties.get("robot_id") or node.id),
        model=model,
        position=node.transform.position,
        quaternion_xyzw=node.transform.quaternion_xyzw,
        sensor_names=tuple(str(value) for value in sensor_names),
    )


def _asset_spec(node: DocumentNode, assets: dict[str, DocumentAsset]) -> AssetSpec:
    asset = assets.get(node.asset_id or "")
    model = str(
        node.properties.get("model")
        or (asset.metadata.get("model") if asset else "")
        or _asset_model(asset)
        or ("target" if node.kind == "region" else "box")
    )
    raw_size = node.properties.get("size")
    size = _tuple(raw_size, 3, (0.4, 0.4, 0.4)) if raw_size else None
    if size:
        size = tuple(size[index] * node.transform.scale[index] for index in range(3))
    material = dict(node.properties.get("material") or {})
    collision = dict(node.properties.get("collision") or {})
    raw_color = (
        node.properties.get("rgba") or node.properties.get("color_rgba") or material.get("rgba")
    )
    rgba = _tuple(raw_color, 4, (0.7, 0.7, 0.7, 1.0)) if raw_color else None
    raw_friction = collision.get("friction") or node.properties.get("friction")
    friction = _tuple(raw_friction, 3, (1.0, 0.005, 0.0001)) if raw_friction else None
    return AssetSpec(
        source_id=node.id,
        model=model,
        category="region"
        if node.kind == "region"
        else str(node.properties.get("category") or model),
        position=node.transform.position,
        quaternion_xyzw=node.transform.quaternion_xyzw,
        size=size,
        rgba=rgba,
        mass=(float(node.properties["mass"]) if node.properties.get("mass") is not None else None),
        inertia=(
            _tuple(node.properties["inertia"], 3, (0.01, 0.01, 0.01))
            if node.properties.get("inertia") is not None
            else None
        ),
        friction=friction,
        collision_enabled=bool(collision.get("enabled", True)),
        contype=int(collision["contype"]) if collision.get("contype") is not None else None,
        conaffinity=int(collision["conaffinity"])
        if collision.get("conaffinity") is not None
        else None,
        static=bool(node.properties.get("static", node.kind == "region")),
        interactive=bool(node.properties.get("interactive", node.kind == "object")),
        pose_reference=str(node.properties.get("pose_reference") or "center"),
        properties=dict(node.properties),
    )


def _sensor_spec(node: DocumentNode) -> SensorSpec:
    return SensorSpec(
        sensor_id=node.name,
        kind=str(node.properties.get("sensor_kind") or "rgb"),
        width=int(node.properties.get("width", 640)),
        height=int(node.properties.get("height", 480)),
        fps=float(node.properties.get("fps", 15)),
        intrinsics=dict(node.properties.get("intrinsics") or {}),
    )


def _write_base_xml(path: Path, document: SceneDocument) -> None:
    root = ET.Element("mujoco", {"model": document.id})
    ET.SubElement(
        root,
        "compiler",
        {"angle": "radian", "eulerseq": "zyx", "autolimits": "true"},
    )
    assets = ET.SubElement(root, "asset")
    ET.SubElement(
        assets,
        "texture",
        {
            "name": "authoring_ground_texture",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.15 0.2",
            "width": "256",
            "height": "256",
        },
    )
    ET.SubElement(
        assets,
        "material",
        {"name": "authoring_ground_material", "texture": "authoring_ground_texture"},
    )
    # R1 Pro 可视几何显式引用该公共材质；场景编译器必须提供它，不能依赖
    # 某个旧场景文件偶然声明。后续 Robot Profile 若增加其他外部材质，应在
    # 资产检查阶段明确列出并由 RuntimeBundle 提供。
    ET.SubElement(
        assets,
        "material",
        {"name": "visualgeom", "rgba": "0.5 0.9 0.2 1"},
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "geom",
        {
            "name": "authoring_ground",
            "type": "plane",
            "size": "20 20 0.1",
            "material": "authoring_ground_material",
        },
    )
    lights = [node for node in document.nodes if node.kind == "light"]
    if not lights:
        ET.SubElement(
            world,
            "light",
            {"name": "authoring_sun", "pos": "3 -4 8", "dir": "-0.3 0.4 -1"},
        )
    for node in lights:
        ET.SubElement(
            world,
            "light",
            {
                "name": node.id,
                "pos": _vec(node.transform.position),
                "dir": _vec(node.properties.get("direction") or (0.0, 0.0, -1.0)),
                "diffuse": _vec(node.properties.get("diffuse") or (0.8, 0.8, 0.8)),
                "specular": _vec(node.properties.get("specular") or (0.2, 0.2, 0.2)),
                "castshadow": (
                    "true" if bool(node.properties.get("castshadow", True)) else "false"
                ),
                "active": "true" if bool(node.properties.get("active", True)) else "false",
            },
        )
    for node in document.nodes:
        if node.kind != "camera" or node.properties.get("robot_id"):
            continue
        ET.SubElement(
            world,
            "camera",
            {
                "name": node.name,
                "mode": "fixed",
                "pos": _vec(node.transform.position),
                "quat": _wxyz(node.transform.quaternion_xyzw),
                "fovy": str(float(node.properties.get("fovy", 50))),
            },
        )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _asset_model(asset: DocumentAsset | None) -> str:
    if asset is None:
        return ""
    value = Path(asset.asset_key).stem
    return value if value not in {"", "."} else ""


def _tuple(value: Any, size: int, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = list(value or default)
    return tuple(float(raw[index]) if index < len(raw) else default[index] for index in range(size))


def _vec(value: Any) -> str:
    return " ".join(f"{float(item):.9g}" for item in value)


def _wxyz(value: tuple[float, float, float, float]) -> str:
    x, y, z, w = value
    return _vec((w, x, y, z))
