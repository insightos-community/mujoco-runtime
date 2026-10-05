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

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import replace

import pytest

from plugin_mujoco.errors import NotFoundError, ValidationRuntimeError
from plugin_mujoco.robots.profile import RobotMapping
from plugin_mujoco.scene import SceneCatalog, build_runtime_scene
from plugin_mujoco.scene.builder import _apply_asset
from plugin_mujoco.scene.catalog import AssetSpec


def test_catalog_lists_three_layouts_and_two_robots(asset_root):
    catalog = SceneCatalog(asset_root)
    descriptors = catalog.list()

    assert len(descriptors) == 1
    assert descriptors[0].layouts == ["layout001", "layout002", "layout003"]
    definition = catalog.load("palletizing_depalletizing_001", "layout001")
    assert [robot.robot_id for robot in definition.robots] == [
        "r1_pro_chassis-1",
        "r1_pro_chassis-2",
    ]
    assert definition.assets[0].source_id == "box-a1"
    categories = {asset.source_id: asset.category for asset in definition.assets}
    assert categories["box-a1"] == "box"
    legacy_pallet = catalog._parse_asset(
        {
            "asset_name": "pallet1",
            "asset_model": "box",
            "position": [0, 0, 0.075],
            "rotation": [0, 0, 0],
        }
    )
    assert legacy_pallet.category == "pallet"
    filtered_pallet = catalog._parse_asset(
        {
            "asset_name": "pallet-a",
            "asset_model": "box",
            "collision": {
                "contype": 8,
                "conaffinity": 5,
                "friction": [0.8, 0.01, 0.001],
            },
        }
    )
    assert filtered_pallet.contype == 8
    assert filtered_pallet.conaffinity == 5
    assert filtered_pallet.friction == (0.8, 0.01, 0.001)
    assert categories["target"] == "region"


def test_catalog_rejects_unknown_layout_and_path_escape(asset_root):
    catalog = SceneCatalog(asset_root)
    with pytest.raises(NotFoundError):
        catalog.load("palletizing_depalletizing_001", "layout999")
    with pytest.raises(NotFoundError):
        catalog.load("../outside", "layout001")


def test_builder_prefixes_multiple_robots_and_hides_engine_ids(asset_root):
    definition = SceneCatalog(asset_root).load("palletizing_depalletizing_001", "layout001")
    build = build_runtime_scene(definition)
    try:
        tree = ET.parse(build.xml_path)
        includes = [item.attrib["file"] for item in tree.getroot().findall("include")]
        assert len(includes) == 5
        assert build.robots[0].prefix != build.robots[1].prefix
        assert build.object_body_names == {"box-a1": "box-a1", "target": "target"}
        for binding in build.robots:
            parsed = ET.parse(build.runtime_dir / f"robot_{build.robots.index(binding) + 1}.xml")
            names_by_type = {}
            for item in parsed.getroot().iter():
                if "name" in item.attrib:
                    names_by_type.setdefault(item.tag, []).append(item.attrib["name"])
            assert all(len(names) == len(set(names)) for names in names_by_type.values())
            assert all(
                name.startswith(binding.prefix)
                for names in names_by_type.values()
                for name in names
                if name not in {"visualgeom"}
            )
    finally:
        build.cleanup()
    assert not build.runtime_dir.exists()


def test_layout_extent_only_resizes_single_geom_primitives():
    primitive = ET.fromstring('<body><geom type="box" size="0.1 0.1 0.1"/></body>')
    compound = ET.fromstring(
        '<body><geom name="bottom" type="box" size="0.3 0.2 0.0025"/>'
        '<geom name="rim" type="box" size="0.009 0.189 0.009"/></body>'
    )
    asset = AssetSpec(
        source_id="tote", model="compound", category="tote",
        position=(0.0, 0.0, 0.0), quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        size=(0.6, 0.4, 0.34), rgba=None,
    )

    _apply_asset(primitive, replace(asset, source_id="box", model="box"))
    _apply_asset(compound, asset)

    assert primitive.find("geom").attrib["size"] == "0.3 0.2 0.17"
    assert [geom.attrib["size"] for geom in compound.iter("geom")] == [
        "0.3 0.2 0.0025",
        "0.009 0.189 0.009",
    ]


def test_builder_preserves_robot_and_object_roll_pitch(asset_root):
    """SceneDocument 的完整姿态不能在 Runtime 拼装时退化成只有 yaw。"""
    definition = SceneCatalog(asset_root).load("palletizing_depalletizing_001", "layout001")
    half_turn = math.pi / 4
    quaternion_xyzw = (math.sin(half_turn), 0.0, 0.0, math.cos(half_turn))
    definition = replace(
        definition,
        robots=(
            replace(definition.robots[0], quaternion_xyzw=quaternion_xyzw),
            *definition.robots[1:],
        ),
        assets=(
            replace(definition.assets[0], quaternion_xyzw=quaternion_xyzw),
            *definition.assets[1:],
        ),
    )
    build = build_runtime_scene(definition)
    try:
        robot_tree = ET.parse(build.runtime_dir / "robot_1.xml")
        robot_body = next(
            item
            for item in robot_tree.getroot().iter("body")
            if item.attrib.get("name") == f"{build.robots[0].prefix}root"
        )
        asset_tree = ET.parse(build.runtime_dir / "asset_1.xml")
        asset_body = next(
            item
            for item in asset_tree.getroot().iter("body")
            if item.attrib.get("name") == definition.assets[0].source_id
        )
        expected_wxyz = (
            quaternion_xyzw[3],
            quaternion_xyzw[0],
            quaternion_xyzw[1],
            quaternion_xyzw[2],
        )
        assert tuple(map(float, robot_body.attrib["quat"].split())) == pytest.approx(expected_wxyz)
        assert tuple(map(float, asset_body.attrib["quat"].split())) == pytest.approx(expected_wxyz)
    finally:
        build.cleanup()


def test_robot_profile_is_the_single_public_to_mujoco_mapping(asset_root):
    path = asset_root / "robot" / "r1_pro_chassis" / "config" / "semantic_robot_profile.yaml"
    mapping = RobotMapping.load(path)
    assert mapping.base_joints["x_joint"] == "root_x_translate"
    assert mapping.kinematic_root_frame == "base_link"
    assert mapping.joint_groups["left_arm"] == ("left_arm_joint1",)
    assert mapping.end_effectors["right"] == "right_ee_site"

    prefix = "robot_1__"
    public_names = {
        mapping.root_body,
        *mapping.joint_names,
        *mapping.end_effectors.values(),
        *mapping.sensors.values(),
    }
    mapping.validate_model_names({f"{prefix}{name}" for name in public_names}, prefix=prefix)
    with pytest.raises(ValidationRuntimeError):
        mapping.validate_model_names(set(), prefix=prefix)
