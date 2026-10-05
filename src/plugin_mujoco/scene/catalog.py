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

"""从外部资产仓读取场景配置。

场景目录只描述公共名称、初态和资源位置。MuJoCo 的 body/geom ID 不会离开
native backend。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from plugin_mujoco.errors import NotFoundError, ValidationRuntimeError
from plugin_mujoco.models import SceneDescriptor
from plugin_mujoco.robots.profile import RobotMapping


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    model: str
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    sensor_names: tuple[str, ...]


@dataclass(frozen=True)
class SensorSpec:
    sensor_id: str
    kind: str
    width: int
    height: int
    fps: float
    intrinsics: dict[str, Any]


@dataclass(frozen=True)
class AssetSpec:
    source_id: str
    model: str
    category: str
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    size: tuple[float, float, float] | None
    rgba: tuple[float, float, float, float] | None
    mass: float | None = None
    inertia: tuple[float, float, float] | None = None
    friction: tuple[float, float, float] | None = None
    collision_enabled: bool = True
    contype: int | None = None
    conaffinity: int | None = None
    static: bool = False
    interactive: bool = False
    pose_reference: str = "center"
    properties: dict[str, Any] | None = None


@dataclass(frozen=True)
class SceneDefinition:
    scene_key: str
    name: str
    layout: str
    base_xml: Path
    robot_root: Path
    object_root: Path
    robots: tuple[RobotSpec, ...]
    sensors: tuple[SensorSpec, ...]
    assets: tuple[AssetSpec, ...]
    timestep: float
    gravity: tuple[float, float, float]


def _tuple(values: Any, size: int, default: float = 0.0) -> tuple[float, ...]:
    raw = list(values or [])
    return tuple(float(raw[index]) if index < len(raw) else default for index in range(size))


def _rpy_to_xyzw(values: tuple[float, ...]) -> tuple[float, float, float, float]:
    """旧资产 YAML 使用 roll/pitch/yaw；只在读取边界转换为公共 xyzw。"""
    roll, pitch, yaw = values
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class SceneCatalog:
    """扫描并加载资产仓中的可运行场景。"""

    def __init__(self, asset_root: Path) -> None:
        self.asset_root = asset_root.expanduser().resolve()
        self.scene_root = self.asset_root / "scene"
        self.robot_root = self.asset_root / "robot"
        self.object_root = self.asset_root / "assets" / "objects"
        self._built: dict[tuple[str, str], SceneDefinition] = {}

    def list(self) -> list[SceneDescriptor]:
        result: list[SceneDescriptor] = []
        if not self.scene_root.is_dir():
            return self._built_descriptors()
        for info_path in sorted(self.scene_root.glob("*/scene_info.yaml")):
            scene_dir = info_path.parent
            layouts = sorted(path.stem for path in scene_dir.glob("layout*.yaml"))
            if not layouts:
                continue
            raw = self._read_yaml(info_path)
            basic = raw.get("basic_info") or {}
            robot_models: set[str] = set()
            for item in raw.get("robots", []):
                asset_model = str(item.get("robot_name", "")).strip()
                if not asset_model:
                    continue
                # scene_info.robot_name 是用于拼装 MJCF 的资产包名；对外目录必须
                # 使用 Profile 声明的硬件型号，避免工具变体被误识别为新型号。
                profile_path = (
                    self.robot_root / asset_model / "config" / "semantic_robot_profile.yaml"
                )
                robot_models.add(RobotMapping.load(profile_path).model)
            result.append(
                SceneDescriptor(
                    scene_key=str(basic.get("scene_id") or scene_dir.name),
                    name=str(basic.get("scene_name") or scene_dir.name),
                    scene_kind="asset_scene",
                    layouts=layouts,
                    robot_models=sorted(robot_models),
                    compatible_runtime_profiles=["native-mujoco"],
                    read_only=True,
                )
            )
        return [*result, *self._built_descriptors()]

    def register(self, definition: SceneDefinition) -> None:
        self._built[(definition.scene_key, definition.layout)] = definition

    def _built_descriptors(self) -> list[SceneDescriptor]:
        grouped: dict[str, list[SceneDefinition]] = {}
        for definition in self._built.values():
            grouped.setdefault(definition.scene_key, []).append(definition)
        return [
            SceneDescriptor(
                scene_key=scene_key,
                name=values[-1].name,
                scene_kind="scene_document",
                layouts=sorted(value.layout for value in values),
                robot_models=sorted({robot.model for value in values for robot in value.robots}),
                compatible_runtime_profiles=["native-mujoco"],
                read_only=False,
            )
            for scene_key, values in sorted(grouped.items())
        ]

    def load(self, scene_key: str, layout: str) -> SceneDefinition:
        built = self._built.get((scene_key, layout))
        if built is not None:
            return built
        scene_dir = self._safe_child(self.scene_root, scene_key)
        info_path = scene_dir / "scene_info.yaml"
        layout_path = scene_dir / f"{layout}.yaml"
        if not info_path.is_file():
            raise NotFoundError(f"场景不存在: {scene_key}")
        if not layout_path.is_file():
            raise NotFoundError(
                f"场景布局不存在: {layout}",
                details={"scene_key": scene_key, "layout": layout},
            )

        info = self._read_yaml(info_path)
        layout_data = self._read_yaml(layout_path)
        basic = info.get("basic_info") or {}
        base_xml = self._safe_child(scene_dir, str(basic.get("base_model_path", "")).strip())
        if not base_xml.is_file():
            raise ValidationRuntimeError(f"基础场景 XML 不存在: {base_xml}")

        robots = tuple(self._parse_robot(item) for item in info.get("robots", []))
        if not robots:
            raise ValidationRuntimeError("场景至少需要一台虚拟 Robot")
        sensors = tuple(self._parse_sensor(item) for item in info.get("sensors", []))
        assets = tuple(self._parse_asset(item) for item in layout_data.get("assets", []))
        simulation = info.get("simulation_params") or {}
        gravity = _tuple(simulation.get("gravity"), 3)

        return SceneDefinition(
            scene_key=str(basic.get("scene_id") or scene_key),
            name=str(basic.get("scene_name") or scene_key),
            layout=layout,
            base_xml=base_xml,
            robot_root=self.robot_root,
            object_root=self.object_root,
            robots=robots,
            sensors=sensors,
            assets=assets,
            timestep=float(simulation.get("timestep", 0.001)),
            gravity=(float(gravity[0]), float(gravity[1]), float(gravity[2])),
        )

    def _parse_robot(self, raw: dict[str, Any]) -> RobotSpec:
        model = str(raw.get("robot_name", "")).strip()
        robot_id = str(raw.get("robot_id", "")).strip()
        if not model or not robot_id:
            raise ValidationRuntimeError("robot_name 和 robot_id 不能为空")
        sensor_names = raw.get("sensor_names") or [raw.get("sensor_name")]
        return RobotSpec(
            robot_id=f"{model}-{robot_id}",
            model=model,
            position=_tuple(raw.get("position"), 3),
            quaternion_xyzw=_rpy_to_xyzw(_tuple(raw.get("rotation"), 3)),
            sensor_names=tuple(str(item) for item in sensor_names if item),
        )

    def _parse_sensor(self, raw: dict[str, Any]) -> SensorSpec:
        sensor_id = str(raw.get("sensor_name", "")).strip()
        if not sensor_id:
            raise ValidationRuntimeError("sensor_name 不能为空")
        return SensorSpec(
            sensor_id=sensor_id,
            kind=str(raw.get("sensor_type") or "camera"),
            width=int(raw.get("width", 640)),
            height=int(raw.get("height", 480)),
            fps=float(raw.get("fps", 15)),
            intrinsics=dict(raw.get("camera_intrinsics") or {}),
        )

    def _parse_asset(self, raw: dict[str, Any]) -> AssetSpec:
        source_id = str(raw.get("asset_name", "")).strip()
        if not source_id:
            raise ValidationRuntimeError("asset_name 不能为空")
        model = str(raw.get("asset_model") or source_id).strip()
        # 新场景应显式声明 category。早期拆码垛 YAML 用同一个 box primitive
        # 表达箱体和托盘，但通过稳定 source_id 区分 pallet。这里只保留受限的
        # 迁移规则，避免场景快照和 Semantic Map 继续把托盘标成 box；不使用
        # 几何尺寸或宿主文件路径猜测领域类别。
        category = str(raw.get("category") or "").strip()
        if not category:
            if model == "target":
                category = "region"
            elif model == "box" and source_id.lower().startswith("pallet"):
                category = "pallet"
            else:
                category = model
        raw_size = raw.get("size")
        raw_rgba = raw.get("rgba")
        collision = raw.get("collision") or {}
        if not isinstance(collision, dict):
            raise ValidationRuntimeError("asset collision 必须是对象")
        raw_friction = collision.get("friction")
        return AssetSpec(
            source_id=source_id,
            model=model,
            category=category,
            position=_tuple(raw.get("position"), 3),
            quaternion_xyzw=_rpy_to_xyzw(_tuple(raw.get("rotation"), 3)),
            size=_tuple(raw_size, 3) if raw_size else None,
            rgba=_tuple(raw_rgba, 4) if raw_rgba else None,
            friction=_tuple(raw_friction, 3) if raw_friction else None,
            collision_enabled=bool(collision.get("enabled", True)),
            contype=int(collision["contype"]) if collision.get("contype") is not None else None,
            conaffinity=(
                int(collision["conaffinity"])
                if collision.get("conaffinity") is not None
                else None
            ),
            static=bool(raw.get("static", False)),
            interactive=bool(raw.get("interactive", False)),
            pose_reference=str(raw.get("pose_reference") or "center"),
            properties=dict(raw.get("properties") or {}),
        )

    def _safe_child(self, parent: Path, raw: str) -> Path:
        candidate = (parent / raw).resolve()
        try:
            candidate.relative_to(self.asset_root)
        except ValueError as exc:
            raise ValidationRuntimeError("资产路径越过 MUJOCO_ASSET_ROOT") from exc
        return candidate

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValidationRuntimeError(f"无法读取场景配置: {path}") from exc
        if not isinstance(value, dict):
            raise ValidationRuntimeError(f"场景配置必须是对象: {path}")
        return value
