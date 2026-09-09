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

"""SceneDocument 的 native MuJoCo 校验规则。

规则不依赖 FastAPI 或 Runtime 状态，因此构建、单元测试和未来命令行工具可以
复用同一份结果。每个问题都携带 node_id 与字段路径，Studio 能直接定位错误。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def document_rule_issues(document: Any) -> list[tuple[str | None, str, str]]:
    issues: list[tuple[str | None, str, str]] = []
    assets = {asset.id: asset for asset in document.assets}

    for asset in document.assets:
        if asset.kind not in {"object", "robot"}:
            issues.append((None, f"assets.{asset.id}.kind", "资产类型必须是 object 或 robot"))
        for key in asset.metadata:
            if key not in {"model", "category"}:
                issues.append(
                    (None, f"assets.{asset.id}.metadata.{key}", f"不支持的资产元数据 {key}")
                )
        model = asset.metadata.get("model")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            issues.append(
                (None, f"assets.{asset.id}.metadata.model", "资产 model 必须是非空字符串")
            )

    for node in [*document.nodes, *document.regions]:
        if node.kind in {"robot", "object"} and not node.asset_id:
            issues.append(
                (node.id, "asset_id", "Robot 和 Object 必须通过 AttachAsset 使用发布资产")
            )
        asset = assets.get(node.asset_id or "")
        if asset is not None:
            if node.kind == "robot" and asset.kind != "robot":
                issues.append((node.id, "asset_id", "Robot 节点只能引用 robot 资产"))
            elif node.kind == "object" and asset.kind != "object":
                issues.append((node.id, "asset_id", "Object 节点只能引用 object 资产"))
            elif node.kind not in {"robot", "object"}:
                issues.append((node.id, "asset_id", f"{node.kind} 节点不能直接引用资产"))
        issues.extend(_property_issues(node))

    for node in document.nodes:
        if node.kind == "region":
            issues.append((node.id, "kind", "Region 必须保存在 regions 集合中"))
    for node in document.regions:
        if node.kind != "region":
            issues.append((node.id, "kind", "regions 集合只能包含 Region"))

    for key in document.physics:
        if key not in {"gravity_m_s2", "timestep_seconds"}:
            issues.append((None, f"physics.{key}", f"不支持的物理属性 {key}"))
    gravity = document.physics.get("gravity_m_s2", (0.0, 0.0, -9.81))
    if not _vector(gravity, 3):
        issues.append((None, "physics.gravity_m_s2", "gravity_m_s2 必须是三个有限数值"))
    timestep = document.physics.get("timestep_seconds", 0.002)
    if not _number(timestep, minimum=0, maximum=0.1, exclusive_minimum=True):
        issues.append((None, "physics.timestep_seconds", "物理步长必须大于 0 且不超过 0.1 秒"))
    return issues


def asset_file_issues(asset_root: Path, document: Any) -> list[tuple[str | None, str, str]]:
    """检查逻辑资产标识确实对应资产仓中的受支持 MJCF 文件。"""
    issues: list[tuple[str | None, str, str]] = []
    root = asset_root.expanduser().resolve()
    referenced_by: dict[str, list[str]] = {}
    for node in [*document.nodes, *document.regions]:
        if node.asset_id:
            referenced_by.setdefault(node.asset_id, []).append(node.id)

    for asset in document.assets:
        try:
            candidate = (root / asset.asset_key).resolve()
            candidate.relative_to(root)
        except ValueError:
            issues.append((None, f"assets.{asset.id}.asset_key", "资产路径越过资产仓"))
            continue
        node_id = referenced_by.get(asset.id, [None])[0]
        if not candidate.is_file():
            issues.append((node_id, "asset_id", f"资产文件不存在: {asset.asset_key}"))
            continue
        if candidate.suffix.lower() != ".xml":
            issues.append((node_id, "asset_id", "native MuJoCo 资产必须是 MJCF XML"))
            continue
        model = str(asset.metadata.get("model") or candidate.stem)
        expected = (
            root / "robot" / model / "config" / f"{model}.xml"
            if asset.kind == "robot"
            else root / "assets" / "objects" / f"{model}.xml"
        ).resolve()
        if candidate != expected:
            issues.append(
                (
                    node_id,
                    "asset_id",
                    f"资产标识与 model 不一致，应为 {expected.relative_to(root)}",
                )
            )
    return issues


def _property_issues(node: Any) -> list[tuple[str | None, str, str]]:
    allowed = {
        "group": set(),
        "object": {
            "model",
            "category",
            "size",
            "rgba",
            "color_rgba",
            "static",
            "interactive",
            "mass",
            "inertia",
            "friction",
            "material",
            "collision",
        },
        "robot": {"model", "robot_id", "sensor_names"},
        "camera": {"robot_id", "sensor_kind", "width", "height", "fps", "intrinsics", "fovy"},
        "light": {"direction", "diffuse", "specular", "castshadow", "active"},
        "region": {
            "model",
            "category",
            "size",
            "rgba",
            "color_rgba",
            "static",
            "interactive",
            "material",
            "collision",
        },
    }
    if node.kind not in allowed:
        return []
    issues: list[tuple[str | None, str, str]] = []

    def problem(field: str, message: str) -> None:
        issues.append((node.id, f"properties.{field}", message))

    for key in node.properties:
        if key not in allowed[node.kind]:
            problem(key, f"该节点类型不支持属性 {key}")
    if "size" in node.properties and not _vector(
        node.properties["size"], 3, minimum=0, exclusive_minimum=True
    ):
        problem("size", "size 必须是三个大于零的米制数值")
    for field in ("rgba", "color_rgba"):
        if field in node.properties and not _vector(
            node.properties[field], 4, minimum=0, maximum=1
        ):
            problem(field, f"{field} 必须是四个 0 到 1 的数值")
    if "mass" in node.properties and not _number(
        node.properties["mass"], minimum=0, exclusive_minimum=True
    ):
        problem("mass", "mass 必须是大于零的千克数值")
    if "inertia" in node.properties:
        if "mass" not in node.properties:
            problem("inertia", "设置 inertia 时必须同时提供 mass")
        if not _vector(node.properties["inertia"], 3, minimum=0, exclusive_minimum=True):
            problem("inertia", "inertia 必须是三个大于零的主惯量")
    if "friction" in node.properties and not _vector(node.properties["friction"], 3, minimum=0):
        problem("friction", "friction 必须是三个非负数值")
    for field in ("static", "interactive"):
        if field in node.properties and not isinstance(node.properties[field], bool):
            problem(field, f"{field} 必须是布尔值")
    if "sensor_names" in node.properties:
        names = node.properties["sensor_names"]
        if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
            problem("sensor_names", "sensor_names 必须是字符串数组")
    for field in ("width", "height", "fps", "fovy"):
        if field in node.properties and not _number(
            node.properties[field], minimum=0, exclusive_minimum=True
        ):
            problem(field, f"{field} 必须是大于零的数值")
    for field in ("direction", "diffuse", "specular"):
        if field in node.properties and not _vector(node.properties[field], 3):
            problem(field, f"{field} 必须是三个数值")
    for field in ("castshadow", "active"):
        if field in node.properties and not isinstance(node.properties[field], bool):
            problem(field, f"{field} 必须是布尔值")

    material = node.properties.get("material")
    if material is not None:
        if not isinstance(material, dict):
            problem("material", "material 必须是对象")
        else:
            for key in material:
                if key != "rgba":
                    problem(f"material.{key}", f"不支持的材质属性 {key}")
            if "rgba" in material and not _vector(material["rgba"], 4, minimum=0, maximum=1):
                problem("material.rgba", "material.rgba 必须是四个 0 到 1 的数值")

    collision = node.properties.get("collision")
    if collision is not None:
        if not isinstance(collision, dict):
            problem("collision", "collision 必须是对象")
        else:
            for key in collision:
                if key not in {"enabled", "contype", "conaffinity", "friction"}:
                    problem(f"collision.{key}", f"不支持的碰撞属性 {key}")
            if "enabled" in collision and not isinstance(collision["enabled"], bool):
                problem("collision.enabled", "collision.enabled 必须是布尔值")
            for field in ("contype", "conaffinity"):
                value = collision.get(field)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                ):
                    problem(f"collision.{field}", "碰撞分组必须是非负整数")
            if "friction" in collision and not _vector(collision["friction"], 3, minimum=0):
                problem("collision.friction", "collision.friction 必须是三个非负数值")
    return issues


def _number(
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    number = float(value)
    if not math.isfinite(number):
        return False
    if minimum is not None and (number <= minimum if exclusive_minimum else number < minimum):
        return False
    return maximum is None or number <= maximum


def _vector(
    value: Any,
    size: int,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> bool:
    if not isinstance(value, list | tuple) or len(value) != size:
        return False
    return all(
        _number(
            item,
            minimum=minimum,
            maximum=maximum,
            exclusive_minimum=exclusive_minimum,
        )
        for item in value
    )
