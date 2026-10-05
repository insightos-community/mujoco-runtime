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

from pathlib import Path

import pytest


@pytest.fixture
def asset_root(tmp_path: Path) -> Path:
    scene = tmp_path / "scene" / "palletizing_depalletizing_001"
    robot = tmp_path / "robot" / "r1_pro_chassis" / "config"
    objects = tmp_path / "assets" / "objects"
    scene.mkdir(parents=True)
    robot.mkdir(parents=True)
    objects.mkdir(parents=True)

    (scene / "scene_info.yaml").write_text(
        """
basic_info:
  scene_id: palletizing_depalletizing_001
  scene_name: 拆码垛测试
  base_model_path: ./palletizing_depalletizing_001.xml
robots:
  - robot_name: r1_pro_chassis
    robot_id: 1
    position: [0, 0, 0.01]
    rotation: [0, 0, 0]
    sensor_name: camera
  - robot_name: r1_pro_chassis
    robot_id: 2
    position: [2, 0, 0.01]
    rotation: [0, 0, 0]
    sensor_name: camera
sensors:
  - sensor_name: camera
    sensor_type: camera
    width: 16
    height: 12
    fps: 10
simulation_params:
  timestep: 0.002
  gravity: [0, 0, -9.81]
""".strip(),
        encoding="utf-8",
    )
    (scene / "palletizing_depalletizing_001.xml").write_text(
        """<mujoco><worldbody><geom name="ground" type="plane" size="5 5 .1"/></worldbody></mujoco>""",
        encoding="utf-8",
    )
    for name, x in (("layout001", 0), ("layout002", 1), ("layout003", 2)):
        (scene / f"{name}.yaml").write_text(
            f"""
assets:
  - asset_name: box-a1
    asset_model: box
    position: [{x}, 1.5, 0.3]
    rotation: [0, 0, 0]
    size: [0.6, 0.4, 0.34]
    rgba: [1, 1, 1, 1]
    static: false
    interactive: true
  - asset_name: target
    asset_model: target
    position: [0, -1.5, 0.1]
    rotation: [0, 0, 0]
    static: true
    interactive: true
""".strip(),
            encoding="utf-8",
        )
    (robot / "r1_pro_chassis.xml").write_text(
        """
<mujoco model="r1">
  <worldbody>
    <body name="root">
      <joint name="root_x_translate" type="slide" axis="1 0 0"/>
      <joint name="root_y_translate" type="slide" axis="0 1 0"/>
      <joint name="root_z_rotate" type="hinge" axis="0 0 1"/>
      <body name="left_arm"><joint name="left_arm_joint1"/><site name="left_ee_site"/></body>
      <body name="right_arm"><joint name="right_arm_joint1"/><site name="right_ee_site"/></body>
      <camera name="camera"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="left_arm_joint1" joint="left_arm_joint1"/>
    <motor name="right_arm_joint1" joint="right_arm_joint1"/>
  </actuator>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    (robot / "semantic_robot_profile.yaml").write_text(
        """
schema_version: 1
model: r1_pro_chassis
kind: mobile_manipulator
coordinate_frame: world
root_body: root
base:
  x_joint: root_x_translate
  y_joint: root_y_translate
  yaw_joint: root_z_rotate
joint_groups:
  left_arm: [left_arm_joint1]
  right_arm: [right_arm_joint1]
end_effectors:
  left: left_ee_site
  right: right_ee_site
grippers: {}
sensors:
  camera: camera
""".strip(),
        encoding="utf-8",
    )
    (objects / "box.xml").write_text(
        """<mujoco><worldbody><body name="box"><freejoint/><geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    (objects / "target.xml").write_text(
        """<mujoco><worldbody><body name="target"><geom type="sphere" size=".1"/></body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    return tmp_path
