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

"""R1 Pro 周转箱专用夹具的 Runtime 低层边界测试。"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest
from test_native_robot_sdk import _pose_matrix, _urdf_fk

from plugin_mujoco.application import RuntimeManager
from plugin_mujoco.errors import ConflictError, ValidationRuntimeError
from plugin_mujoco.models import (
    CommandState,
    RobotCommand,
    RobotCommandRequest,
    SceneStartRequest,
    utc_now,
)
from plugin_mujoco.native.backend import (
    MujocoBackend,
    _interpolate_base,
    _nearest_equivalent_angle,
)
from plugin_mujoco.robots.profile import GripperToolMapping, RobotMapping
from plugin_mujoco.scene import SceneCatalog
from plugin_mujoco.settings import Settings
from plugin_mujoco.testing.fake_backend import FakeBackend


def _tool(hook: str, clamp: str) -> GripperToolMapping:
    return GripperToolMapping(
        tool_ref="component://tool/test",
        frame="test_tote_load_frame",
        joint="test_tote_clamp_joint",
        kind="tote_clamp",
        software_min_position_m=0.0,
        software_max_position_m=0.035,
        normal_force_n=60.0,
        peak_force_n=120.0,
        control_force_scale_n_per_unit=120.0,
        hook_bodies=(hook,),
        clamp_bodies=(clamp,),
    )


def test_base_yaw_interpolation_remains_continuous_across_pi():
    """跨过 +pi 时控制目标必须继续增大，不能瞬间跳到 -pi。"""

    sample, finished = _interpolate_base(
        [
            {"time_from_start": 0.0, "x": 0.0, "y": 0.0, "yaw": 3.1},
            {"time_from_start": 1.0, "x": 0.0, "y": 0.0, "yaw": -3.1},
        ],
        0.75,
    )

    assert finished is False
    assert sample["yaw"] == pytest.approx(3.16238898)


def test_wrapped_pose_yaw_is_rebased_near_continuous_joint_angle():
    """新轨迹的 -pi/2 必须延续上一段已经到达的 3*pi/2。"""

    assert _nearest_equivalent_angle(1.5 * np.pi, -0.5 * np.pi) == pytest.approx(
        1.5 * np.pi
    )


def test_insert_joint_trajectory_can_finish_on_expected_tool_contact():
    backend = MujocoBackend.__new__(MujocoBackend)
    backend._tool_contacts = {"r1": {"left": {"hook_contact": False, "hook_force_n": 0.0}}}
    mapping = SimpleNamespace(
        gripper_tools={
            "left": SimpleNamespace(
                tool_ref="component://tool/left",
                peak_force_n=120.0,
            )
        }
    )
    target = {
        "contact_tool_refs": ["component://tool/left"],
        "stop_on_contact": True,
        "max_contact_force_n": 20.0,
    }
    command = SimpleNamespace(command_id="insert-1", target=target)

    assert backend._joint_contact_tick("r1", mapping, command) is None
    backend._tool_contacts["r1"]["left"]["hook_contact"] = True
    backend._tool_contacts["r1"]["left"]["hook_force_n"] = 12.0
    tick = backend._joint_contact_tick("r1", mapping, command)
    assert tick is not None and tick.completed

    backend._tool_contacts["r1"]["left"]["hook_force_n"] = 21.0
    tick = backend._joint_contact_tick("r1", mapping, command)
    assert tick is not None and tick.failed


def test_insert_joint_trajectory_does_not_finish_without_contact():
    """insert 到达几何终点但没有钩爪接触时，必须继续等待并最终由超时收敛。"""

    backend = MujocoBackend.__new__(MujocoBackend)
    backend._lock = RLock()
    backend._in_hold = defaultdict(bool)
    backend._tool_contacts = {"r1": {"left": {"hook_contact": False, "hook_force_n": 0.0}}}
    mapping = SimpleNamespace(
        command_joint_names=("left_arm_joint",),
        gripper_tools={
            "left": SimpleNamespace(
                tool_ref="component://tool/left",
                peak_force_n=120.0,
            )
        },
    )
    backend._binding = lambda _robot_id: SimpleNamespace()
    backend._mapping = lambda _robot_id: mapping
    backend._drive_public_joint = lambda _binding, _name, _value: 0.0
    backend._insert_contact_geometry = lambda *_args: ["left_hook->outer_wall"]
    command = SimpleNamespace(
        command_id="insert-no-contact",
        type="joint_trajectory",
        details={},
        target={
            "points": [
                {
                    "time_from_start": 0.0,
                    "positions": {"left_arm_joint": 0.0},
                },
                {
                    "time_from_start": 1.0,
                    "positions": {"left_arm_joint": 0.2},
                },
            ],
            "tolerance": 0.002,
            "stop_on_contact": True,
            "contact_tool_refs": ["component://tool/left"],
            "max_contact_force_n": 20.0,
        },
    )

    tick = backend.advance_command("r1", command, elapsed=1.1)

    assert tick.completed is False
    assert tick.failed is True
    assert "left_hook->outer_wall" in tick.reason


def _gripper_command_backend(*, error: float, contact: dict) -> MujocoBackend:
    backend = MujocoBackend.__new__(MujocoBackend)
    backend._lock = RLock()
    backend._in_hold = defaultdict(bool)
    backend._load_support_latched = defaultdict(bool)
    backend._stable_upper_body_positions = {}
    backend._gripper_targets = defaultdict(dict)
    backend._tool_contacts = {"r1": {"left": contact}}
    backend._binding = lambda _robot_id: SimpleNamespace(prefix="")
    backend._joint_control_gains = {
        "left_clamp_joint": SimpleNamespace(kp=5000.0, kd=55.0)
    }
    tool = SimpleNamespace(
        normal_force_n=60.0,
        peak_force_n=120.0,
        control_force_scale_n_per_unit=120.0,
        minimum_clamp_force_n=2.0,
    )
    backend._mapping = lambda _robot_id: SimpleNamespace(
        grippers={"left": ("left_clamp_joint",)},
        validate_gripper_target=lambda _side, _opening, _force: tool,
    )
    backend._driven_gripper_targets = []

    def record_gripper_target(_binding, _joint_name, target, **_kwargs):
        backend._driven_gripper_targets.append(target)
        return error

    backend._drive_public_joint = record_gripper_target
    backend._public_joint_position = lambda *_args: 0.01
    return backend


def _close_until_contact_command(command_id: str = "close-1") -> SimpleNamespace:
    return SimpleNamespace(
        command_id=command_id,
        type="gripper_command",
        target={
            "gripper": "left",
            "opening": 0.0,
            "force_limit_n": 60.0,
            "close_until_contact": True,
            "tolerance": 0.002,
        },
    )


def test_close_until_contact_completes_at_position_without_claiming_load():
    """闭合到目标是低层命令终态，不要求此时已经形成承载接触。"""

    backend = _gripper_command_backend(
        error=0.0,
        contact={
            "contact_object": None,
            "hook_contact": False,
            "clamp_contact": False,
            "hook_force_n": 0.0,
            "clamp_force_n": 0.0,
        },
    )

    tick = backend.advance_command("r1", _close_until_contact_command(), elapsed=0.0)

    assert tick.completed
    assert backend._gripper_targets["r1"]["left"]["contact_latched"] is False


def test_close_until_contact_latches_clamp_contact_without_requiring_hook_load():
    """压紧块接触后锁存当前位置，但不能在Runtime中冒充钩爪承载结论。"""

    backend = _gripper_command_backend(
        error=0.02,
        contact={
            "contact_object": "tote-large-smoke",
            "hook_contact": False,
            "clamp_contact": True,
            "hook_force_n": 0.0,
            "clamp_force_n": 12.0,
        },
    )

    tick = backend.advance_command("r1", _close_until_contact_command(), elapsed=0.0)

    assert tick.completed
    assert backend._gripper_targets["r1"]["left"]["contact_latched"] is True
    # 60N是允许上限，不是持续目标；实测12N已超过Profile的2N最低预紧，
    # 因而保持首次接触开度，不能再额外内收。
    expected_hold_position = 0.01
    assert backend._gripper_targets["r1"]["left"]["position"] == pytest.approx(
        expected_hold_position
    )
    assert backend._driven_gripper_targets == [pytest.approx(expected_hold_position)]

    # 命令对象仍携带调用方原始零开度；再次推进也不能恢复追踪零开度。
    backend._tool_contacts["r1"]["left"] = {
        "contact_object": None,
        "hook_contact": False,
        "clamp_contact": False,
        "hook_force_n": 0.0,
        "clamp_force_n": 0.0,
    }
    tick = backend.advance_command("r1", _close_until_contact_command(), elapsed=0.1)

    assert tick.completed
    assert backend._gripper_targets["r1"]["left"]["position"] == pytest.approx(
        expected_hold_position
    )
    assert backend._driven_gripper_targets == [
        pytest.approx(expected_hold_position),
        pytest.approx(expected_hold_position),
    ]


def test_close_until_contact_continues_until_profile_preload():
    """首次轻触只开始机械预紧，不能把不足的夹片力锁存为命令成功。"""

    backend = _gripper_command_backend(
        error=0.0,
        contact={
            "contact_object": "tote-large-smoke",
            "hook_contact": True,
            "clamp_contact": True,
            "hook_force_n": 4.0,
            "clamp_force_n": 0.5,
        },
    )
    command = _close_until_contact_command()

    tick = backend.advance_command("r1", command, elapsed=0.0)

    assert tick.completed is False
    target = backend._gripper_targets["r1"]["left"]
    assert target["contact_latched"] is False
    assert target["position"] < 0.01


def test_close_until_contact_latches_preload_without_cycle_ratchet():
    """达到Profile预紧后保持同一控制目标，不按控制周期继续内收。"""

    backend = _gripper_command_backend(
        error=0.0,
        contact={
            "contact_object": "tote-large-smoke",
            "hook_contact": True,
            "clamp_contact": True,
            "hook_force_n": 4.0,
            "clamp_force_n": 1.0,
        },
    )
    tool = backend._mapping("r1").validate_gripper_target("left", 0.0, 60.0)
    tool.control_force_scale_n_per_unit = 1.0
    tool.minimum_clamp_force_n = 15.0
    opening = [0.012]
    backend._public_joint_position = lambda *_args: opening[0]
    command = _close_until_contact_command()

    first = backend.advance_command("r1", command, elapsed=0.0)
    first_target = backend._gripper_targets["r1"]["left"]["position"]
    assert first.completed is False
    assert first_target == pytest.approx(0.012 - 14.0 / 5000.0)

    opening[0] = first_target
    backend._tool_contacts["r1"]["left"]["clamp_force_n"] = 15.0
    second = backend.advance_command("r1", command, elapsed=0.1)
    assert second.completed is True
    assert backend._gripper_targets["r1"]["left"]["position"] == pytest.approx(
        first_target
    )


def test_close_until_contact_replaces_same_command_nominal_zero_after_contact():
    """同一命令由自由运动进入接触后，必须锁存接触预紧目标而非继续追零。"""

    backend = _gripper_command_backend(
        error=0.02,
        contact={
            "contact_object": None,
            "hook_contact": False,
            "clamp_contact": False,
            "hook_force_n": 0.0,
            "clamp_force_n": 0.0,
        },
    )
    command = _close_until_contact_command()

    first = backend.advance_command("r1", command, elapsed=0.0)
    assert first.completed is False
    assert backend._gripper_targets["r1"]["left"]["position"] == pytest.approx(0.0)

    backend._tool_contacts["r1"]["left"] = {
        "contact_object": "tote-large-smoke",
        "hook_contact": True,
        "clamp_contact": True,
        "hook_force_n": 4.0,
        "clamp_force_n": 3.0,
    }
    second = backend.advance_command("r1", command, elapsed=0.1)

    # 实测3N已满足Profile的2N最低预紧，锁存当前接触开度而不继续追零。
    expected_hold_position = 0.01
    assert second.completed is True
    assert backend._gripper_targets["r1"]["left"]["position"] == pytest.approx(
        expected_hold_position
    )


def test_new_close_command_reacquires_contact_instead_of_reusing_old_latch():
    """重新压紧必须建立本命令的接触目标，不能沿用上一命令的锁存终态。"""

    backend = _gripper_command_backend(
        error=0.02,
        contact={
            "contact_object": "tote-large-smoke",
            "hook_contact": True,
            "clamp_contact": True,
            "hook_force_n": 4.0,
            "clamp_force_n": 6.0,
        },
    )
    backend.advance_command("r1", _close_until_contact_command("close-1"), elapsed=0.0)
    first_target = backend._gripper_targets["r1"]["left"]["position"]

    backend._public_joint_position = lambda *_args: 0.009
    backend.advance_command("r1", _close_until_contact_command("close-2"), elapsed=0.0)

    second = backend._gripper_targets["r1"]["left"]
    assert second["command_id"] == "close-2"
    assert second["contact_latched"] is True
    assert second["position"] == pytest.approx(first_target)


def test_new_close_command_without_contact_never_opens_previous_latch():
    """接触暂失时重新夹紧继续向内寻找，不能向名义候选位置反向开夹。"""

    backend = _gripper_command_backend(
        error=0.02,
        contact={
            "contact_object": "tote-large-smoke",
            "hook_contact": True,
            "clamp_contact": True,
            "hook_force_n": 4.0,
            "clamp_force_n": 6.0,
        },
    )
    backend.advance_command("r1", _close_until_contact_command("close-1"), elapsed=0.0)
    first_target = backend._gripper_targets["r1"]["left"]["position"]
    backend._tool_contacts["r1"]["left"] = {
        "contact_object": None,
        "hook_contact": False,
        "clamp_contact": False,
        "hook_force_n": 0.0,
        "clamp_force_n": 0.0,
    }
    command = _close_until_contact_command("close-2")
    command.target["opening"] = 0.015

    tick = backend.advance_command("r1", command, elapsed=0.0)

    second = backend._gripper_targets["r1"]["left"]
    assert tick.completed is False
    assert second["contact_latched"] is False
    assert second["require_contact"] is True
    assert second["position"] == pytest.approx(first_target)


def test_tote_gripper_profile_centralizes_travel_force_and_contact_names(tmp_path):
    """机械暂定值必须来自 Profile，不能散落在 Driver 或 Skill 中。"""
    profile = tmp_path / "semantic_robot_profile.yaml"
    profile.write_text(
        """
schema_version: 1
model: r1_pro_chassis
kind: mobile_manipulator
coordinate_frame: world
root_body: root
base: {x_joint: root_x, y_joint: root_y, yaw_joint: root_yaw}
joint_groups: {left_arm: [left_arm_joint]}
joint_control:
  left_arm: {kp: 240, kd: 24}
  left: {kp: 1600, kd: 40}
end_effectors: {left: left_tool}
grippers: {left: [left_tote_clamp_joint]}
gripper_tools:
  left:
    kind: tote_clamp
    tool_ref: component://tool/left
    frame: left_tote_load_frame
    joint: left_tote_clamp_joint
    software_range_m: [0.0, 0.035]
    normal_force_n: 60
    peak_force_n: 120
    control_force_scale_n_per_unit: 120
    hook_bodies: [left_tote_hook]
    clamp_bodies: [left_tote_clamp]
sensors: {}
""".strip(),
        encoding="utf-8",
    )

    mapping = RobotMapping.load(profile)
    tool = mapping.validate_gripper_target("left", 0.02, 60.0)
    assert tool is not None
    assert tool.tool_ref == "component://tool/left"
    assert tool.frame == "left_tote_load_frame"
    assert tool.joint == "left_tote_clamp_joint"
    assert tool.software_max_position_m == pytest.approx(0.035)
    assert tool.normal_force_n == pytest.approx(60.0)
    assert tool.peak_force_n == pytest.approx(120.0)
    assert tool.hook_bodies == ("left_tote_hook",)
    assert tool.clamp_bodies == ("left_tote_clamp",)
    gains = mapping.control_gains("left_arm_joint")
    assert gains.kp == pytest.approx(240.0)
    assert gains.kd == pytest.approx(24.0)
    assert "left_tote_clamp_joint" not in mapping.command_joint_names
    assert "left_tote_clamp_joint" in mapping.controlled_joint_names
    tool_gains = mapping.control_gains("left_tote_clamp_joint")
    assert tool_gains.kp == pytest.approx(1600.0)
    assert tool_gains.kd == pytest.approx(40.0)

    with pytest.raises(ValidationRuntimeError, match="软件行程"):
        mapping.validate_gripper_target("left", 0.04, 60.0)
    with pytest.raises(ValidationRuntimeError, match="机械峰值"):
        mapping.validate_gripper_target("left", 0.02, 121.0)


def test_tote_load_requires_both_arms_hook_and_clamp_on_same_tote():
    """单侧未插入或脱钩时不能报告稳定承载。"""
    backend = MujocoBackend.__new__(MujocoBackend)
    backend.build = SimpleNamespace(object_body_names={"tote-a"})
    backend._holding = {"r1": None}
    backend._load_support_latched = {"r1": False}
    backend._stable_upper_body_positions = {}
    backend._gripper_targets = {
        "r1": {
            "left": {"contact_latched": True},
            "right": {"contact_latched": True},
        }
    }
    backend._tool_contacts = defaultdict(dict)
    backend._profiles = {
        "r1": SimpleNamespace(
            base_joints={},
            grippers={},
            gripper_tools={
                "left": _tool("left_hook", "left_clamp"),
                "right": _tool("right_hook", "right_clamp"),
            },
        )
    }
    backend._binding = lambda _robot_id: SimpleNamespace(prefix="")
    backend._joint_positions = lambda _binding, _mapping: {}
    public_names = {
        1: "tote-a",
        2: "r1:left_hook",
        3: "r1:left_clamp",
        4: "r1:right_hook",
        5: "r1:right_clamp",
        6: "tote-a",
    }
    backend._public_body_for_geom = public_names.__getitem__
    geom_names = {
        1: "tote_a_collision_wall_neg_above_groove",
        2: "left_tote_hook_lip_collision",
        3: "left_tote_clamp_pad_collision",
        4: "right_tote_hook_lip_collision",
        5: "right_tote_clamp_pad_collision",
        6: "tote_a_collision_outer_wall",
    }
    backend._geom_name = geom_names.__getitem__

    def _normal_force(_index: int, contact: SimpleNamespace) -> float:
        return abs(float(contact.normal_force))

    def _relative_speed(contact: SimpleNamespace) -> float:
        return abs(float(contact.relative_speed))

    backend._contact_normal_force = _normal_force
    backend._contact_relative_speed = _relative_speed
    backend._contact_vertical_ratio = lambda contact: float(getattr(contact, "vertical_ratio", 1.0))
    backend.data = SimpleNamespace(
        ncon=4,
        contact=[
            SimpleNamespace(geom1=1, geom2=2, normal_force=12.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=3, normal_force=12.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=4, normal_force=12.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=5, normal_force=12.0, relative_speed=0.0),
        ],
    )

    backend._detect_tote_tool_contacts()
    assert backend._holding["r1"] == "tote-a"
    assert backend._load_support_latched["r1"] is True
    assert all(
        item["hook_contact"] and item["clamp_contact"]
        for item in backend._tool_contacts["r1"].values()
    )
    assert all(
        item["hook_force_n"] == pytest.approx(12.0) and item["clamp_force_n"] == pytest.approx(12.0)
        for item in backend._tool_contacts["r1"].values()
    )

    # 起升取载时压紧件会沿箱壁产生相对运动，只要承重钩爪没有沿箱沿
    # 滑动、钩入和夹紧接触都还在，就必须继续报告稳定承载。
    backend.data = SimpleNamespace(
        ncon=4,
        contact=[
            SimpleNamespace(geom1=1, geom2=2, normal_force=12.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=3, normal_force=12.0, relative_speed=0.1),
            SimpleNamespace(geom1=1, geom2=4, normal_force=12.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=5, normal_force=12.0, relative_speed=0.1),
        ],
    )
    backend._detect_tote_tool_contacts()
    assert backend._holding["r1"] == "tote-a"
    assert all(
        item["hook_tangential_speed_m_s"] == pytest.approx(0.0)
        for item in backend._tool_contacts["r1"].values()
    )

    # 相同速度若发生在承重钩爪接触平面，Runtime只上报实测相对速度；
    # 是否需要停止和hold由消费该状态的Ability按任务上下文决定。
    backend.data = SimpleNamespace(
        ncon=4,
        contact=[
            SimpleNamespace(geom1=1, geom2=2, normal_force=12.0, relative_speed=0.1),
            SimpleNamespace(geom1=1, geom2=3, normal_force=12.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=4, normal_force=12.0, relative_speed=0.1),
            SimpleNamespace(geom1=1, geom2=5, normal_force=12.0, relative_speed=0.0),
        ],
    )
    backend._detect_tote_tool_contacts()
    assert backend._holding["r1"] == "tote-a"
    assert all(
        item["hook_tangential_speed_m_s"] == pytest.approx(0.1)
        for item in backend._tool_contacts["r1"].values()
    )

    # 下面各分支应从独立稳定状态开始，避免上一个滑移场景影响 was_stable。
    backend.data = SimpleNamespace(
        ncon=4,
        contact=[
            SimpleNamespace(geom1=1, geom2=2, normal_force=12.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=3, normal_force=12.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=4, normal_force=12.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=5, normal_force=12.0, relative_speed=0.0),
        ],
    )
    for _ in range(3):
        backend._detect_tote_tool_contacts()
    assert backend._holding["r1"] == "tote-a"

    # 一侧收到开夹命令后，夹具离开箱壁前仍可能保留一帧双侧接触。接触
    # 观测可以保持真实，但不能把已经清除的承载锁存和旧上臂姿态重新写回。
    backend._gripper_targets["r1"]["left"]["contact_latched"] = False
    backend._load_support_latched["r1"] = False
    backend._stable_upper_body_positions["r1"] = {"arm_joint": 0.25}
    backend._detect_tote_tool_contacts()
    assert backend._holding["r1"] == "tote-a"
    assert backend._load_support_latched["r1"] is False
    assert backend._stable_upper_body_positions["r1"] == {"arm_joint": 0.25}
    backend._gripper_targets["r1"]["left"]["contact_latched"] = True

    backend.data = SimpleNamespace(
        ncon=3,
        contact=[
            SimpleNamespace(geom1=1, geom2=2, normal_force=12.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=3, normal_force=12.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=4, normal_force=12.0, relative_speed=0.0),
        ],
    )
    backend._detect_tote_tool_contacts()
    assert backend._holding["r1"] is None
    assert backend._tool_contacts["r1"]["right"]["hook_contact"] is True
    assert backend._tool_contacts["r1"]["right"]["clamp_contact"] is False
    assert backend._tool_contacts["r1"]["right"]["hook_force_n"] == pytest.approx(12.0)
    assert backend._tool_contacts["r1"]["right"]["clamp_force_n"] == pytest.approx(0.0)

    # Runtime只报告此刻压块接触、下钩未接触；是否允许继续由Ability决定。
    # 这里不输出“稳定承载”等上层结论。
    backend.data = SimpleNamespace(
        ncon=2,
        contact=[
            SimpleNamespace(geom1=1, geom2=3, normal_force=12.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=5, normal_force=12.0, relative_speed=0.0),
        ],
    )
    backend._detect_tote_tool_contacts()
    assert backend._holding["r1"] is None
    assert all(
        not item["hook_contact"] and item["clamp_contact"]
        for item in backend._tool_contacts["r1"].values()
    )

    # 钩爪碰到箱体外壁是真实接触；Runtime照实上报低承重方向比例，
    # 不把它改写成“没有接触”或“稳定/不稳定”结论。
    # Runtime 不依赖 Scene Package 的 geom 命名；这里用接触物理量表达差异。
    backend.data = SimpleNamespace(
        ncon=4,
        contact=[
            SimpleNamespace(
                geom1=6,
                geom2=2,
                normal_force=12.0,
                relative_speed=0.0,
                vertical_ratio=0.1,
            ),
            SimpleNamespace(geom1=1, geom2=3, normal_force=12.0, relative_speed=0.0),
            SimpleNamespace(
                geom1=6,
                geom2=4,
                normal_force=12.0,
                relative_speed=0.0,
                vertical_ratio=0.1,
            ),
            SimpleNamespace(geom1=1, geom2=5, normal_force=12.0, relative_speed=0.0),
        ],
    )
    backend._detect_tote_tool_contacts()
    assert backend._holding["r1"] is None
    assert all(
        item["hook_contact"] and item["hook_support_ratio"] == pytest.approx(0.1)
        for item in backend._tool_contacts["r1"].values()
    )

    # receiver侧壁上的横向擦碰仍是接触，只是承重方向比例为零。
    # Runtime不代替Ability判断它能否承载，也不依赖具体零件名称。
    backend.data = SimpleNamespace(
        ncon=4,
        contact=[
            SimpleNamespace(
                geom1=1, geom2=2, normal_force=12.0, relative_speed=0.0, vertical_ratio=0.0
            ),
            SimpleNamespace(
                geom1=1, geom2=3, normal_force=12.0, relative_speed=0.0, vertical_ratio=1.0
            ),
            SimpleNamespace(
                geom1=1, geom2=4, normal_force=12.0, relative_speed=0.0, vertical_ratio=0.0
            ),
            SimpleNamespace(
                geom1=1, geom2=5, normal_force=12.0, relative_speed=0.0, vertical_ratio=1.0
            ),
        ],
    )
    backend._detect_tote_tool_contacts()
    assert backend._holding["r1"] is None
    assert all(
        item["hook_contact"] and item["hook_support_ratio"] == pytest.approx(0.0)
        for item in backend._tool_contacts["r1"].values()
    )

    # 超过工具机械峰值时Runtime仍上报实测力；内部低层支撑不会锁存，
    # 是否过载及如何停止由Ability结合工具描述判定。
    backend.data = SimpleNamespace(
        ncon=4,
        contact=[
            SimpleNamespace(geom1=1, geom2=2, normal_force=130.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=3, normal_force=130.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=4, normal_force=130.0, relative_speed=0.0),
            SimpleNamespace(geom1=1, geom2=5, normal_force=130.0, relative_speed=0.0),
        ],
    )
    backend._detect_tote_tool_contacts()
    assert backend._holding["r1"] is None
    assert all(
        item["hook_force_n"] == pytest.approx(130.0)
        and item["clamp_force_n"] == pytest.approx(130.0)
        for item in backend._tool_contacts["r1"].values()
    )


def test_contact_relative_speed_is_measured_at_contact_point() -> None:
    """刚体参考点速度不同但接触点速度相同时，不得误报滑移。"""

    backend = MujocoBackend.__new__(MujocoBackend)
    backend.model = SimpleNamespace(geom_bodyid=np.asarray([0, 1]))
    backend.data = SimpleNamespace(
        xpos=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
    )
    velocities = {
        # body 0 绕 z 轴转动，位于 [1, 0, 0] 的接触点速度为 [0, 1, 0]。
        0: np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        # body 1 不旋转，但其参考点以同样速度平移。
        1: np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    }

    def object_velocity(_model, _data, _kind, body_id, output, _local):
        output[:] = velocities[body_id]

    backend.mj = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_BODY=1),
        mj_objectVelocity=object_velocity,
    )
    contact = SimpleNamespace(
        geom1=0,
        geom2=1,
        pos=[1.0, 0.0, 0.0],
        frame=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )

    assert backend._contact_relative_speed(contact) == pytest.approx(0.0)

    velocities[1] = np.asarray([0.0, 0.0, 0.0, 0.0, 0.9, 0.0])
    assert backend._contact_relative_speed(contact) == pytest.approx(0.1)

    # 接触法向为 x；相同大小的法向压紧速度不属于沿箱沿滑动。
    velocities[1] = np.asarray([0.0, 0.0, 0.0, -0.1, 1.0, 0.0])
    assert backend._contact_relative_speed(contact) == pytest.approx(0.0)


def test_finished_joint_trajectory_reports_compact_tracking_error() -> None:
    """轨迹末端未收敛时指出超差关节，但不复制完整轨迹到状态响应。"""

    backend = MujocoBackend.__new__(MujocoBackend)
    backend._lock = RLock()
    backend._in_hold = {"robot-1": True}
    backend._binding = lambda _robot_id: object()
    backend._mapping = lambda _robot_id: SimpleNamespace(command_joint_names=("joint-a", "joint-b"))
    errors = iter((0.003, 0.0004))
    backend._drive_public_joint = lambda *_args, **_kwargs: next(errors)
    backend._joint_contact_tick = lambda *_args, **_kwargs: None

    command = RobotCommand(
        command_id="tracking-diagnostic",
        robot_id="robot-1",
        scene_generation=1,
        type="joint_trajectory",
        joint_trajectory={
            "resources": ["joints:arm"],
            "frame_id": "world",
            "points": [
                {
                    "time_from_start_seconds": 0.0,
                    "positions": {"joint-a": 0.0, "joint-b": 0.0},
                },
                {
                    "time_from_start_seconds": 1.0,
                    "positions": {"joint-a": 0.1, "joint-b": 0.2},
                },
            ],
            "position_tolerance_rad": 0.002,
        },
        timeout_seconds=2.0,
        status=CommandState.RUNNING,
        accepted_at=utc_now(),
        updated_at=utc_now(),
    )

    tick = backend.advance_command("robot-1", command, elapsed=1.0)

    assert not tick.completed
    assert command.details["joint_tracking"] == {
        "position_tolerance_rad": 0.002,
        "max_error_rad": 0.003,
        "errors_rad": {"joint-a": 0.003},
    }


def test_hold_preserves_each_contact_latched_tool_without_claiming_bilateral_load() -> None:
    """上臂动作结束时，单侧真实接触不能被通用hold改写成回弹开度。"""

    backend = MujocoBackend.__new__(MujocoBackend)
    backend._lock = RLock()
    backend.mj = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_JOINT=1, mjOBJ_ACTUATOR=2),
        mj_forward=lambda *_args: None,
    )
    backend.model = SimpleNamespace(
        jnt_dofadr=np.asarray([0, 1]),
    )
    backend.data = SimpleNamespace(
        qvel=np.asarray([0.3, 0.2]),
        ctrl=np.asarray([1.0, 1.0]),
    )
    backend._binding = lambda _robot_id: SimpleNamespace(prefix="")
    tool = SimpleNamespace(control_force_scale_n_per_unit=1.0)
    backend._mapping = lambda _robot_id: SimpleNamespace(
        joint_names=("arm_joint", "left_clamp_joint"),
        gripper_tools={"left": tool},
        grippers={"left": ("left_clamp_joint",)},
    )
    ids = {"arm_joint": 0, "left_clamp_joint": 1}
    backend._id = lambda _kind, name: ids[name]
    backend._optional_id = lambda _kind, name: ids[name]
    backend._joint_positions = lambda *_args: {
        "arm_joint": 0.2,
        "left_clamp_joint": 0.014,
    }
    backend._tool_contacts = {
        "r1": {
            "left": {
                "hook_contact": True,
                "clamp_contact": True,
            }
        }
    }
    backend._gripper_targets = {
        "r1": {
            "left": {
                "position": 0.012,
                "force_limit_n": 60.0,
                "contact_latched": True,
            }
        }
    }
    backend._holding = {"r1": None}
    backend._load_support_latched = {"r1": False}
    backend._stable_upper_body_positions = {}
    backend._hold_targets = {}
    backend._hold_effort_limits = {}
    backend._in_hold = {"r1": False}

    backend.hold_robot("r1")

    assert backend._holding["r1"] is None
    assert backend._hold_targets["r1"]["arm_joint"] == pytest.approx(0.2)
    assert backend._hold_targets["r1"]["left_clamp_joint"] == pytest.approx(0.012)
    assert backend._hold_effort_limits["left_clamp_joint"] == pytest.approx(60.0)


def test_joint_control_compensates_passive_and_contact_constraint_forces() -> None:
    """持物接触力应由动力学前馈承担，不能靠关节长期偏离目标来平衡。"""

    backend = MujocoBackend.__new__(MujocoBackend)
    backend.mj = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_JOINT=1, mjOBJ_ACTUATOR=2))
    backend._id = lambda _kind, _name: 0
    backend._optional_id = lambda _kind, _name: 0
    backend._joint_control_gains = {}
    backend._joint_robot_ids = {"joint-a": "robot-1"}
    backend._holding = {"robot-1": "tote-a"}
    backend._load_support_latched = {"robot-1": True}
    backend._profiles = {
        "robot-1": SimpleNamespace(gripper_tools={"left": object(), "right": object()})
    }
    backend._tool_contacts = {
        "robot-1": {
            side: {
                "hook_contact": True,
                "clamp_contact": True,
                "contact_object": "tote-a",
                "hook_force_n": 12.0,
                "clamp_force_n": 12.0,
            }
            for side in ("left", "right")
        }
    }
    backend.model = SimpleNamespace(
        jnt_qposadr=np.asarray([0]),
        jnt_dofadr=np.asarray([0]),
        actuator_ctrllimited=np.asarray([True]),
        actuator_ctrlrange=np.asarray([[-100.0, 100.0]]),
    )
    backend.data = SimpleNamespace(
        qpos=np.asarray([0.0]),
        qvel=np.asarray([0.0]),
        qfrc_bias=np.asarray([10.0]),
        qfrc_passive=np.asarray([2.0]),
        qfrc_constraint=np.asarray([3.0]),
        ctrl=np.asarray([0.0]),
    )

    backend._object_has_external_contact = lambda *_args: False
    error = backend._drive_joint("joint-a", 0.0)

    assert error == pytest.approx(0.0)
    assert backend.data.ctrl[0] == pytest.approx(5.0)

    # 箱体仍受托盘支撑时，当前夹具接触仍会把约束力传回上臂；忽略它会
    # 让单侧外拉的hold姿态在下一次重规划前漂移。
    backend._object_has_external_contact = lambda *_args: True
    backend._drive_joint("joint-a", 0.0)
    assert backend.data.ctrl[0] == pytest.approx(5.0)

    # 夹具限力控制只使用位置误差和动力学偏置；接触约束不能再次叠加到输出。
    backend._object_has_external_contact = lambda *_args: False
    backend._drive_joint("joint-a", 0.0, control_limit=100.0)
    assert backend.data.ctrl[0] == pytest.approx(8.0)

    backend._holding["robot-1"] = None
    backend._drive_joint("joint-a", 0.0)
    assert backend.data.ctrl[0] == pytest.approx(5.0)

    # 当前工具接触全部消失后，历史锁存不能继续参与关节控制。
    backend._load_support_latched["robot-1"] = False
    backend._tool_contacts["robot-1"] = {}
    backend._gripper_targets = {"robot-1": {}}
    backend._drive_joint("joint-a", 0.0)
    assert backend.data.ctrl[0] == pytest.approx(8.0)


def test_fake_backend_exposes_low_level_gripper_target_and_force(asset_root):
    """测试客户端可读目标、力限制和到位反馈，但不伪造承载。"""
    definition = SceneCatalog(asset_root).load("palletizing_depalletizing_001", "layout001")
    backend = FakeBackend(definition)
    robot_id = backend.robot_ids()[0]
    command = SimpleNamespace(
        type="gripper_command",
        target={"gripper": "left", "opening": 0.025, "force_limit_n": 60.0},
    )

    tick = backend.advance_command(robot_id, command, 0.0)
    state = backend.robot_state(robot_id, generation=1)
    assert tick.completed is True
    assert state.gripper_states["left"].position == pytest.approx(0.025)
    assert state.gripper_states["left"].target_position == pytest.approx(0.025)
    assert state.gripper_states["left"].effort == pytest.approx(60.0)
    assert state.gripper_states["left"].reached_target is True
    assert state.gripper_states["left"].hook_contact is False
    assert state.gripper_states["left"].clamp_contact is False


def _wait_command(instance, robot_id: str, command_id: str, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        command = instance.command(robot_id, command_id)
        if command.status.value in {"succeeded", "failed", "cancelled", "unknown"}:
            return command
        time.sleep(0.01)
    raise AssertionError(f"夹具命令未在 {timeout}s 内结束: {command_id}")


def _joint_control_diagnostics(instance, robot_id: str, names):
    """返回失败时的控制平衡量，避免靠猜测调整执行器或完成容差。"""
    backend = instance.components.robots._backend
    binding = backend._binding(robot_id)
    diagnostics = {}
    mass_matrix = np.zeros((backend.model.nv, backend.model.nv))
    backend.mj.mj_fullM(backend.model, mass_matrix, backend.data.qM)
    for public_name in names:
        internal = f"{binding.prefix}{public_name}"
        joint_id = backend._id(backend.mj.mjtObj.mjOBJ_JOINT, internal)
        dof_address = int(backend.model.jnt_dofadr[joint_id])
        actuator_id = backend._id(backend.mj.mjtObj.mjOBJ_ACTUATOR, internal)
        gains = backend._joint_control_gains.get(internal)
        kp = gains.kp if gains is not None else 150.0
        kd = gains.kd if gains is not None else 15.0
        bias = float(backend.data.qfrc_bias[dof_address])
        passive = float(backend.data.qfrc_passive[dof_address])
        velocity = float(backend.data.qvel[dof_address])
        constraint = float(backend.data.qfrc_constraint[dof_address])
        diagnostics[public_name] = {
            "inertia": round(float(mass_matrix[dof_address, dof_address]), 6),
            "bias": round(bias, 4),
            "passive": round(passive, 4),
            "constraint": round(constraint, 4),
            "control": round(float(backend.data.ctrl[actuator_id]), 4),
            "target_control": round(
                bias - passive - constraint + kp * float(names[public_name]) - kd * velocity,
                4,
            ),
            "control_range": [
                round(float(value), 4) for value in backend.model.actuator_ctrlrange[actuator_id]
            ],
        }
    return diagnostics


@pytest.mark.native
def test_native_tote_gripper_scene_layouts_and_lifecycle():
    """三套正式 Layout 必须由真实 MuJoCo 加载并保持多箱物理稳定。"""
    raw_root = os.getenv("MUJOCO_ASSET_ROOT")
    if not raw_root:
        pytest.skip("MUJOCO_ASSET_ROOT 未设置")
    manager = RuntimeManager(
        Settings(
            asset_root=Path(raw_root),
            backend="mujoco",
            render_backend="egl",
            realtime=False,
        )
    )
    descriptor = next(
        item
        for item in manager.catalog.list()
        if item.scene_key == "palletizing_depalletizing_tote_v1"
    )
    assert descriptor.robot_models == ["r1_pro_chassis"]
    large_totes = {
        f"tote-large-l{layer}-r{row}-c{column}"
        for layer in range(1, 4)
        for row in range(1, 3)
        for column in range(1, 3)
    }
    small_totes = {
        f"tote-small-l{layer}-r{row}-c{column}"
        for layer in range(1, 4)
        for row in range(1, 3)
        for column in range(1, 3)
    }
    expected_totes = {
        "layout001": large_totes,
        "layout002": small_totes,
        "layout003": large_totes | small_totes,
    }
    try:
        for index, (layout, tote_ids) in enumerate(expected_totes.items(), start=1):
            started = manager.start(
                "palletizing_depalletizing_tote_v1",
                SceneStartRequest(
                    request_id=f"native-tote-{layout}",
                    layout=layout,
                    seed=index,
                    render_backend="egl",
                ),
            )
            scene = manager.wait_ready(started.instance_id, timeout=30)
            assert scene.state.value == "running", scene.failure_reason
            instance = manager.get(scene.instance_id)
            robot_id = instance.robot_ids()[0]
            profile = instance.components.robots.robot_profile(robot_id)
            assert profile.model == "r1_pro_chassis"
            assert profile.backend == "mujoco"
            assert profile.sdk_package == "semantic-robot-sdk-r1pro"
            assert profile.backend_profile == "r1pro-tote-mujoco-v1"
            assert profile.endpoint == "http://127.0.0.1:8090"
            assert profile.urdf_path and profile.urdf_path.endswith(
                "/robot/r1_pro_tote_gripper/meshes/r1_pro_tote_gripper.urdf"
            )
            assert len(profile.package_directories) == 1
            assert profile.package_directories[0].endswith("/robot")
            assert set(profile.capabilities.commands) >= {"gripper_command"}
            assert {tool.tool_ref for tool in profile.tools} == {
                "component://tool/left",
                "component://tool/right",
            }
            assert {tool.kind for tool in profile.tools} == {"tote_clamp"}
            assert {tool.frame for tool in profile.tools} <= set(profile.capabilities.frames)

            deadline = time.monotonic() + 5.0
            while instance.view().sim_time < 1.0 and time.monotonic() < deadline:
                time.sleep(0.005)
            snapshot = instance.snapshot()
            objects = {item.source_id: item for item in snapshot.objects}
            assert tote_ids <= set(objects)
            regions = {item.source_id: item for item in snapshot.regions}
            assert set(regions) >= {
                "pallet-b-slot-r1-c1",
                "pallet-b-slot-r1-c2",
                "pallet-b-slot-r2-c1",
                "pallet-b-slot-r2-c2",
            }
            assert "pallet-a-approach" not in regions
            assert "pallet-b-approach" not in regions
            for row in (1, 2):
                for column in (1, 2):
                    slot = regions[f"pallet-b-slot-r{row}-c{column}"]
                    assert slot.properties["row"] == row
                    assert slot.properties["column"] == column
                    assert slot.properties["max_layers"] == 3
                    assert slot.properties["support_z_m"] == pytest.approx(0.15)
            # Runtime Snapshot 对外统一返回物体几何中心，MJCF body 原点是否位于
            # 底面只属于资产编译细节，不能泄漏给 SDK、Ability 或 Robot Skill。
            for tote_id in tote_ids:
                layer = int(tote_id.rsplit("-l", 1)[1][0])
                height = 0.34 if tote_id.startswith("tote-large-") else 0.24
                expected_extent = (
                    (0.6, 0.4, 0.34) if tote_id.startswith("tote-large-") else (0.53, 0.41, 0.24)
                )
                assert objects[tote_id].extent == pytest.approx(expected_extent)
                expected_center_z = 0.15 + (layer - 0.5) * height
                assert objects[tote_id].pose.position[2] == pytest.approx(
                    expected_center_z, abs=0.035
                )

            stopped = manager.stop(scene.instance_id)
            assert stopped.state.value == "stopped"
    finally:
        manager.shutdown()


@pytest.mark.native
def test_native_tote_gripper_command_hold_and_generation_guard():
    """低层夹具命令在独立 smoke Layout 验证，避免多箱堆叠规模干扰命令时限。"""
    raw_root = os.getenv("MUJOCO_ASSET_ROOT")
    if not raw_root:
        pytest.skip("MUJOCO_ASSET_ROOT 未设置")
    manager = RuntimeManager(
        Settings(
            asset_root=Path(raw_root),
            backend="mujoco",
            render_backend="egl",
            realtime=False,
        )
    )
    try:
        started = manager.start(
            "palletizing_depalletizing_tote_v1",
            SceneStartRequest(
                request_id="native-tote-layout-smoke",
                layout="layout_smoke",
                seed=7,
                render_backend="egl",
            ),
        )
        scene = manager.wait_ready(started.instance_id, timeout=30)
        assert scene.state.value == "running", scene.failure_reason
        instance = manager.get(scene.instance_id)
        robot_id = instance.robot_ids()[0]

        # SDK 使用 URDF 做多末端 IK，而 Runtime 使用 MJCF 执行关节轨迹。两边
        # 必须对同一个工具负载坐标系给出一致的 base-relative 位姿；否则低层
        # 命令虽然会按关节目标正常结束，上层 VerifyPregrasp 却永远无法通过。
        state = instance.components.robots.robot_state(robot_id, scene.generation)
        asset_root = Path(os.environ["MUJOCO_ASSET_ROOT"])
        urdf = asset_root / "robot/r1_pro_tote_gripper/meshes/r1_pro_tote_gripper.urdf"
        world_from_base = _pose_matrix(state.base_pose)
        for side in ("left", "right"):
            world_from_tool = _pose_matrix(state.end_effectors[side])
            actual_base_from_tool = np.linalg.inv(world_from_base) @ world_from_tool
            expected_base_from_tool = _urdf_fk(
                urdf,
                root_link="base_link",
                target_link=f"{side}_tote_load_frame",
                joints={name: value.position for name, value in state.joints.items()},
            )
            np.testing.assert_allclose(
                actual_base_from_tool[:3, 3],
                expected_base_from_tool[:3, 3],
                atol=5e-4,
            )
            np.testing.assert_allclose(
                actual_base_from_tool[:3, :3],
                expected_base_from_tool[:3, :3],
                atol=5e-4,
            )

        requests = []
        target_positions = {"left": 0.005, "right": 0.02}
        for side, position in target_positions.items():
            request = RobotCommandRequest(
                command_id=f"tote-clamp-{side}",
                scene_generation=scene.generation,
                type="gripper_command",
                gripper_command={
                    "gripper_id": side,
                    "position": position,
                    "max_effort": 60.0,
                },
                timeout_seconds=10,
            )
            instance.submit_command(robot_id, request)
            requests.append(request)

        # 左侧短行程会先完成；它不能触发全 Robot hold 而冻结仍在闭合的右侧。
        for request in requests:
            result = _wait_command(instance, robot_id, request.command_id)
            assert result.status.value == "succeeded", result.failure_reason
        state = instance.components.robots.robot_state(robot_id, scene.generation)
        for side, position in target_positions.items():
            assert state.gripper_states[side].position == pytest.approx(position, abs=0.002)
            assert state.gripper_states[side].target_position == pytest.approx(position)
        assert state.in_hold

        instance.hold_robot(robot_id)
        assert instance.components.robots.robot_state(robot_id, 1).in_hold
        reset = instance.reset()
        assert reset.generation == 2
        with pytest.raises(ConflictError, match="generation"):
            instance.submit_command(
                robot_id,
                RobotCommandRequest(
                    command_id="stale-tote-command",
                    scene_generation=1,
                    type="gripper_command",
                    gripper_command={"gripper_id": "left", "position": 0.01},
                ),
            )
        stopped = manager.stop(scene.instance_id)
        assert stopped.state.value == "stopped"
    finally:
        manager.shutdown()


@pytest.mark.native
def test_native_tote_gripper_tracks_representative_bilateral_approach():
    """双臂接近周转箱时应在公开力矩和原精度约束内稳定收敛。"""
    raw_root = os.getenv("MUJOCO_ASSET_ROOT")
    if not raw_root:
        pytest.skip("MUJOCO_ASSET_ROOT 未设置")
    manager = RuntimeManager(
        Settings(
            asset_root=Path(raw_root),
            backend="mujoco",
            render_backend="egl",
            realtime=False,
        )
    )
    try:
        started = manager.start(
            "palletizing_depalletizing_tote_v1",
            SceneStartRequest(
                request_id="native-tote-bilateral-approach",
                layout="layout_smoke",
                seed=7,
                render_backend="egl",
            ),
        )
        scene = manager.wait_ready(started.instance_id, timeout=30)
        assert scene.state.value == "running", scene.failure_reason
        instance = manager.get(scene.instance_id)
        robot_id = instance.robot_ids()[0]
        state = instance.components.robots.robot_state(robot_id, scene.generation)

        # 该目标来自 Robot SDK 为 direct_bilateral 接近位姿生成的真实轨迹。
        # 只保留首末两点，测试 Runtime 的负载跟踪与到位判定，不把 SDK 的
        # 完整采样算法复制到 Runtime 测试中。
        target = {
            "torso_joint1": -0.6990023494141235,
            "torso_joint2": 2.016551017242034,
            "torso_joint3": -0.10532276104827279,
            "torso_joint4": 0.09051378043584318,
            "left_arm_joint1": -2.6359937722140847,
            "left_arm_joint2": 0.13395573072204603,
            "left_arm_joint3": -0.1495268372972428,
            "left_arm_joint4": 0.16313635045263228,
            "left_arm_joint5": -0.16327670053434012,
            "left_arm_joint6": 1.0242005711572932,
            "left_arm_joint7": -0.416938006644592,
            "right_arm_joint1": -2.799352488486755,
            "right_arm_joint2": -0.20744821779557832,
            "right_arm_joint3": 0.2463279149583444,
            "right_arm_joint4": 0.26854583796141634,
            "right_arm_joint5": 0.27573160391407875,
            "right_arm_joint6": 1.047198,
            "right_arm_joint7": 0.4351321361465715,
        }
        initial = {name: state.joints[name].position for name in target}
        command = RobotCommandRequest(
            command_id="representative-bilateral-approach",
            scene_generation=scene.generation,
            type="joint_trajectory",
            joint_trajectory={
                "resources": ["joints:torso", "joints:left_arm", "joints:right_arm"],
                "frame_id": "world",
                "points": [
                    {"time_from_start_seconds": 0.0, "positions": initial},
                    {"time_from_start_seconds": 9.2, "positions": target},
                ],
                "position_tolerance_rad": 0.002,
            },
            timeout_seconds=14.0,
        )
        instance.submit_command(robot_id, command)
        result = _wait_command(instance, robot_id, command.command_id, timeout=20.0)
        final_state = instance.components.robots.robot_state(robot_id, scene.generation)
        errors = {
            name: round(target[name] - final_state.joints[name].position, 6)
            for name in target
            if abs(target[name] - final_state.joints[name].position) > 0.001
        }
        if result.status.value != "succeeded":
            pytest.fail(
                json.dumps(
                    {
                        "reason": result.failure_reason,
                        "joint_errors_rad": errors,
                        "joint_control": _joint_control_diagnostics(instance, robot_id, errors),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        assert final_state.in_hold
    finally:
        manager.shutdown()


@pytest.mark.native
def test_native_tote_gripper_settles_loaded_torso_at_insert_target():
    """双臂到达箱沿后，躯干的小幅插入修正仍应在原精度内完成。"""
    raw_root = os.getenv("MUJOCO_ASSET_ROOT")
    if not raw_root:
        pytest.skip("MUJOCO_ASSET_ROOT 未设置")
    manager = RuntimeManager(
        Settings(
            asset_root=Path(raw_root),
            backend="mujoco",
            render_backend="egl",
            realtime=False,
        )
    )
    try:
        started = manager.start(
            "palletizing_depalletizing_tote_v1",
            SceneStartRequest(
                request_id="native-tote-insert-settling",
                layout="layout_smoke",
                seed=7,
                render_backend="egl",
            ),
        )
        scene = manager.wait_ready(started.instance_id, timeout=30)
        assert scene.state.value == "running", scene.failure_reason
        instance = manager.get(scene.instance_id)
        robot_id = instance.robot_ids()[0]

        # 轨迹取自真实 AbilityFramework → SDK → Runtime 链中曾经超时的插入动作。
        # 两个点之间只有很小的末端修正，旧的统一低增益会让承载双臂的躯干
        # 留下超过 0.002rad 的稳态误差；测试保留原精度和力矩限制，防止以后
        # 通过放宽完成条件掩盖同类问题。
        initial = {
            "torso_joint1": -0.45229525096806183,
            "torso_joint2": 1.792945638881111,
            "torso_joint3": 0.14115435068723264,
            "torso_joint4": -0.001055802885440343,
            "left_arm_joint1": -2.0557233295722064,
            "left_arm_joint2": 0.1395849345691648,
            "left_arm_joint3": -0.10863011971731275,
            "left_arm_joint4": -0.18799973814659873,
            "left_arm_joint5": -0.07175643231045759,
            "left_arm_joint6": 1.0472134013290573,
            "left_arm_joint7": -0.23463814717754933,
            "right_arm_joint1": -2.0578662412654047,
            "right_arm_joint2": -0.1397307571862986,
            "right_arm_joint3": 0.11110801712793619,
            "right_arm_joint4": -0.1887397020521045,
            "right_arm_joint5": 0.06822262568709427,
            "right_arm_joint6": 1.0472002504485358,
            "right_arm_joint7": 0.23360511948987428,
        }
        target = {
            "torso_joint1": -0.45621344530908114,
            "torso_joint2": 1.7883843016967262,
            "torso_joint3": 0.1445802824154846,
            "torso_joint4": 3.497589178249485e-05,
            "left_arm_joint1": -2.057653790577211,
            "left_arm_joint2": 0.13935189978728582,
            "left_arm_joint3": -0.10847295528098568,
            "left_arm_joint4": -0.18906439817074108,
            "left_arm_joint5": -0.07159102240957255,
            "left_arm_joint6": 1.0471980000000003,
            "left_arm_joint7": -0.23481672104729656,
            "right_arm_joint1": -2.057884304259588,
            "right_arm_joint2": -0.13932674338660442,
            "right_arm_joint3": 0.11109179840094532,
            "right_arm_joint4": -0.18878283463597423,
            "right_arm_joint5": 0.06815704174322187,
            "right_arm_joint6": 1.047198,
            "right_arm_joint7": 0.23373420154358018,
        }
        command = RobotCommandRequest(
            command_id="loaded-torso-insert-settling",
            scene_generation=scene.generation,
            type="joint_trajectory",
            joint_trajectory={
                "resources": ["joints:torso", "joints:left_arm", "joints:right_arm"],
                "frame_id": "world",
                "points": [
                    {"time_from_start_seconds": 0.0, "positions": initial},
                    {"time_from_start_seconds": 0.5, "positions": target},
                ],
                "position_tolerance_rad": 0.002,
            },
            timeout_seconds=10.0,
        )
        instance.submit_command(robot_id, command)
        result = _wait_command(instance, robot_id, command.command_id, timeout=15.0)
        final_state = instance.components.robots.robot_state(robot_id, scene.generation)
        errors = {
            name: round(target[name] - final_state.joints[name].position, 6)
            for name in target
            if abs(target[name] - final_state.joints[name].position) > 0.001
        }
        if result.status.value != "succeeded":
            pytest.fail(
                json.dumps(
                    {
                        "reason": result.failure_reason,
                        "joint_errors_rad": errors,
                        "joint_control": _joint_control_diagnostics(instance, robot_id, errors),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        assert final_state.in_hold
    finally:
        manager.shutdown()


@pytest.mark.native
@pytest.mark.parametrize("layout", ["layout001", "layout002", "layout003"])
def test_native_tote_full_stack_remains_finite_for_product_window(layout: str):
    """正式多箱 Layout 在产品动作时间窗内不得产生坏加速度。"""
    raw_root = os.getenv("MUJOCO_ASSET_ROOT")
    if not raw_root:
        pytest.skip("MUJOCO_ASSET_ROOT 未设置")

    definition = SceneCatalog(Path(raw_root)).load("palletizing_depalletizing_tote_v1", layout)
    backend = MujocoBackend(definition, render_backend="egl", seed=1)
    try:
        initial = {
            item.source_id: np.asarray(item.pose.position, dtype=float)
            for item in backend.scene_objects()
            if item.category == "tote"
        }
        # 旧参数在约 7.6 秒后才出现自由箱体 QACC 发散，普通一秒加载测试
        # 无法发现。这里覆盖一次抓取规划和执行所需的代表性十秒物理窗口；
        # 同时约束静态横向漂移，避免“数值仍有限、整垛却持续爬行”的假通过。
        for _ in range(round(10.0 / backend.timestep)):
            backend.step()
        final = {
            item.source_id: np.asarray(item.pose.position, dtype=float)
            for item in backend.scene_objects()
            if item.category == "tote"
        }
        import mujoco

        bad_qacc = backend.data.warning[mujoco.mjtWarning.mjWARN_BADQACC]
        assert bad_qacc.number == 0, {
            "layout": layout,
            "last_dof": int(bad_qacc.lastinfo),
            "sim_time": backend.sim_time,
        }
        assert np.isfinite(backend.data.qpos).all()
        assert np.isfinite(backend.data.qvel).all()
        assert np.isfinite(backend.data.qacc).all()
        lateral_drift = {
            source_id: float(np.linalg.norm(final[source_id][:2] - pose[:2]))
            for source_id, pose in initial.items()
        }
        assert max(lateral_drift.values(), default=0.0) < 0.002, {
            "layout": layout,
            "lateral_drift_m": lateral_drift,
        }
    finally:
        backend.close()
