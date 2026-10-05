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

"""将场景、Robot 与布局对象拼装为可由 MuJoCo 加载的临时 XML。"""

from __future__ import annotations

import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from plugin_mujoco.errors import ValidationRuntimeError

from .catalog import AssetSpec, RobotSpec, SceneDefinition

_REFERENCE_ATTRS = {
    "actuator",
    "body",
    "body1",
    "body2",
    "camera",
    "crankbody",
    "cranksite",
    "framename",
    "geom",
    "geom1",
    "geom2",
    "hfield",
    "joint",
    "joint1",
    "joint2",
    "material",
    "mesh",
    "objname",
    "site",
    "site1",
    "site2",
    "slidersite",
    "target",
    "tendon",
    "texture",
}


@dataclass(frozen=True)
class RobotBinding:
    robot_id: str
    model: str
    prefix: str
    joint_names: tuple[str, ...]
    camera_names: tuple[str, ...]


@dataclass
class RuntimeSceneBuild:
    runtime_dir: Path
    xml_path: Path
    robots: tuple[RobotBinding, ...]
    object_body_names: dict[str, str]
    definition: SceneDefinition

    def cleanup(self) -> None:
        shutil.rmtree(self.runtime_dir, ignore_errors=True)


def build_runtime_scene(definition: SceneDefinition) -> RuntimeSceneBuild:
    runtime_dir = Path(tempfile.mkdtemp(prefix=f"semantic_mujoco_{definition.scene_key}_"))
    robot_bindings: list[RobotBinding] = []
    object_body_names: dict[str, str] = {}

    for index, robot in enumerate(definition.robots):
        source = definition.robot_root / robot.model / "config" / f"{robot.model}.xml"
        if not source.is_file():
            raise ValidationRuntimeError(f"Robot 模型不存在: {source}")
        root = ET.parse(source).getroot()
        _absolute_files(root, source.parent)
        prefix = _prefix(robot.robot_id) if len(definition.robots) > 1 else ""
        if prefix:
            _prefix_tree(root, prefix)
        _spawn_robot(root, robot, prefix)
        _inject_camera(root, definition, robot, prefix)
        target = runtime_dir / f"robot_{index + 1}.xml"
        ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
        joint_names = tuple(
            item.attrib["name"] for item in root.iter("joint") if item.attrib.get("name")
        )
        camera_names = tuple(
            item.attrib["name"] for item in root.iter("camera") if item.attrib.get("name")
        )
        robot_bindings.append(
            RobotBinding(
                robot_id=robot.robot_id,
                model=robot.model,
                prefix=prefix,
                joint_names=joint_names,
                camera_names=camera_names,
            )
        )

    asset_paths: list[Path] = []
    for index, asset in enumerate(definition.assets):
        source = definition.object_root / f"{asset.model}.xml"
        if not source.is_file():
            raise ValidationRuntimeError(f"对象模板不存在: {source}")
        root = ET.parse(source).getroot()
        _absolute_files(root, source.parent)
        body = _first_body(root)
        if body is None:
            raise ValidationRuntimeError(f"对象模板没有 body: {source}")
        prefix = f"asset_{index + 1}__"
        _prefix_tree(root, prefix, {body.attrib.get("name", ""): asset.source_id})
        body = _named_body(root, asset.source_id)
        if body is None:
            raise ValidationRuntimeError(f"无法定位对象 body: {asset.source_id}")
        _apply_asset(body, asset)
        target = runtime_dir / f"asset_{index + 1}.xml"
        ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
        asset_paths.append(target)
        object_body_names[asset.source_id] = asset.source_id

    wrapper = ET.Element(
        "mujoco",
        {"model": f"{_safe(definition.scene_key)}_{_safe(definition.layout)}"},
    )
    ET.SubElement(wrapper, "include", {"file": str(definition.base_xml.resolve())})
    for index in range(len(robot_bindings)):
        ET.SubElement(wrapper, "include", {"file": str(runtime_dir / f"robot_{index + 1}.xml")})
    for path in asset_paths:
        ET.SubElement(wrapper, "include", {"file": str(path)})
    ET.SubElement(
        wrapper,
        "option",
        {
            "timestep": f"{definition.timestep:.9g}",
            "gravity": " ".join(f"{value:.9g}" for value in definition.gravity),
            "solver": "CG",
        },
    )
    xml_path = runtime_dir / "scene.xml"
    ET.ElementTree(wrapper).write(xml_path, encoding="utf-8", xml_declaration=True)
    return RuntimeSceneBuild(
        runtime_dir=runtime_dir,
        xml_path=xml_path,
        robots=tuple(robot_bindings),
        object_body_names=object_body_names,
        definition=definition,
    )


def _safe(raw: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_") or "unnamed"


def _prefix(raw: str) -> str:
    value = _safe(raw)
    if value[0].isdigit():
        value = f"r_{value}"
    return f"{value}__"


def _absolute_files(root: ET.Element, source_dir: Path) -> None:
    for element in root.iter():
        raw = element.attrib.get("file")
        if raw and not Path(raw).is_absolute():
            element.attrib["file"] = str((source_dir / raw).resolve())


def _prefix_tree(
    root: ET.Element,
    prefix: str,
    overrides: dict[str, str] | None = None,
) -> None:
    name_map: dict[str, str] = {}
    class_map: dict[str, str] = {}
    overrides = overrides or {}
    for element in root.iter():
        if name := element.attrib.get("name"):
            name_map[name] = overrides.get(name, f"{prefix}{name}")
        if element.tag == "default" and (class_name := element.attrib.get("class")):
            class_map[class_name] = f"{prefix}{class_name}"

    for element in root.iter():
        for attr, value in list(element.attrib.items()):
            if attr == "name" and value in name_map:
                element.attrib[attr] = name_map[value]
            elif element.tag == "mujoco" and attr == "model":
                element.attrib[attr] = f"{prefix}{value}"
            elif attr in {"class", "childclass"} and value in class_map:
                element.attrib[attr] = class_map[value]
            elif attr in _REFERENCE_ATTRS:
                element.attrib[attr] = " ".join(
                    name_map.get(token, token) for token in value.split()
                )


def _spawn_robot(root: ET.Element, robot: RobotSpec, prefix: str) -> None:
    body = _named_body(root, f"{prefix}root")
    if body is None:
        body = _first_body(root)
    if body is None:
        raise ValidationRuntimeError(f"Robot {robot.model} 没有根 body")
    body.attrib["pos"] = _vec(robot.position)
    body.attrib["quat"] = _wxyz_quat(robot.quaternion_xyzw)


def _inject_camera(
    root: ET.Element,
    definition: SceneDefinition,
    robot: RobotSpec,
    prefix: str,
) -> None:
    by_name = {sensor.sensor_id: sensor for sensor in definition.sensors}
    for public_name in robot.sensor_names:
        sensor = by_name.get(public_name)
        if sensor is None:
            continue
        camera = next(
            (
                item
                for item in root.iter("camera")
                if item.attrib.get("name") in {public_name, f"{prefix}{public_name}"}
            ),
            None,
        )
        if camera is None:
            continue
        camera.attrib.pop("fovy", None)
        camera.attrib["resolution"] = f"{sensor.width} {sensor.height}"
        intrinsics = sensor.intrinsics
        if values := intrinsics.get("sensorsize"):
            camera.attrib["sensorsize"] = _vec_n(values, 2)
        if values := intrinsics.get("focalpixel"):
            camera.attrib["focalpixel"] = _vec_n(values, 2)
        if values := intrinsics.get("principalpixel"):
            cx, cy = float(values[0]), float(values[1])
            camera.attrib["principalpixel"] = _vec_n(
                (cx - sensor.width * 0.5, cy - sensor.height * 0.5),
                2,
            )


def _apply_asset(body: ET.Element, asset: AssetSpec) -> None:
    body.attrib["name"] = asset.source_id
    body.attrib["pos"] = _vec(asset.position)
    body.attrib["quat"] = _wxyz_quat(asset.quaternion_xyzw)
    if asset.static:
        for parent in body.iter():
            for child in list(parent):
                if child.tag in {"joint", "freejoint"}:
                    parent.remove(child)
    geoms = list(body.iter("geom"))
    # Layout 的 size 用于覆盖单几何 primitive（box/target）的外形。多几何
    # 资产中的 Box 是模型内部碰撞代理，例如箱体的底板和箱沿，不能被语义
    # extent 全部改写成整物体尺寸，否则会在加载时产生大面积自相交。
    resizable_primitive = len(geoms) == 1 and geoms[0].attrib.get("type") == "box"
    if asset.inertia is not None:
        if asset.mass is None:
            raise ValidationRuntimeError("设置 inertia 时必须同时提供 mass")
        inertial = next(iter(body.findall("inertial")), None)
        if inertial is None:
            inertial = ET.Element("inertial")
            body.insert(0, inertial)
        inertial.attrib.update(
            {
                "pos": "0 0 0",
                "mass": f"{asset.mass:.9g}",
                "diaginertia": _vec_n(asset.inertia, 3),
            }
        )

    for index, geom in enumerate(geoms):
        if asset.rgba:
            geom.attrib["rgba"] = _vec_n(asset.rgba, 4)
        if asset.size and resizable_primitive and geom.attrib.get("type") == "box":
            geom.attrib["size"] = _vec(tuple(value * 0.5 for value in asset.size))
        if asset.friction:
            geom.attrib["friction"] = _vec_n(asset.friction, 3)
        if not asset.collision_enabled:
            geom.attrib["contype"] = "0"
            geom.attrib["conaffinity"] = "0"
        else:
            if asset.contype is not None:
                geom.attrib["contype"] = str(asset.contype)
            if asset.conaffinity is not None:
                geom.attrib["conaffinity"] = str(asset.conaffinity)
        # 没有显式惯量时，质量只写入第一个 geom，避免多 geom 模型重复累计质量。
        if asset.mass is not None and asset.inertia is None:
            if index == 0:
                geom.attrib["mass"] = f"{asset.mass:.9g}"
            else:
                geom.attrib.pop("mass", None)


def _first_body(root: ET.Element) -> ET.Element | None:
    world = root.find("worldbody")
    if world is None:
        return None
    return next((child for child in list(world) if child.tag == "body"), None)


def _named_body(root: ET.Element, name: str) -> ET.Element | None:
    return next((body for body in root.iter("body") if body.attrib.get("name") == name), None)


def _vec(values: tuple[float, float, float]) -> str:
    return _vec_n(values, 3)


def _vec_n(values: object, size: int) -> str:
    return " ".join(f"{float(value):.9g}" for value in list(values)[:size])


def _wxyz_quat(quaternion_xyzw: tuple[float, float, float, float]) -> str:
    """MuJoCo XML 使用 wxyz；公共模型始终保持 xyzw。"""
    x, y, z, w = quaternion_xyzw
    return _vec_n((w, x, y, z), 4)
