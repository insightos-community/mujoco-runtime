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

import time

import pytest

from plugin_mujoco.application import RuntimeManager
from plugin_mujoco.errors import ConflictError
from plugin_mujoco.models import RobotCommandRequest, SceneStartRequest
from plugin_mujoco.settings import Settings


@pytest.fixture
def running(asset_root):
    manager = RuntimeManager(Settings(asset_root=asset_root, backend="fake", realtime=True))
    record = manager.start(
        "palletizing_depalletizing_001",
        SceneStartRequest(request_id="start", layout="layout001", headless=True),
    )
    ready = manager.wait_ready(record.instance_id)
    assert ready.state.value == "running"
    yield manager, manager.get(record.instance_id)
    manager.shutdown()


def base_command(command_id="move", generation=1, duration=0.02, x=1.0):
    return RobotCommandRequest(
        command_id=command_id,
        scene_generation=generation,
        type="base_trajectory",
        base_trajectory={
            "frame_id": "world",
            "points": [
                {"time_from_start_seconds": 0.0, "positions": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                {"time_from_start_seconds": duration, "positions": {"x": x, "y": 2.0, "yaw": 0.2}},
            ],
        },
        timeout_seconds=max(duration * 2, 0.01),
    )


def wait_terminal(instance, robot_id, command_id, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = instance.command(robot_id, command_id)
        if value.status.value in {"succeeded", "failed", "cancelled"}:
            return value
        time.sleep(0.005)
    raise AssertionError("命令没有进入终态")


def test_command_idempotency_and_robot_routing(running):
    _, instance = running
    first_id, second_id = instance.robot_ids()
    first = instance.submit_command(first_id, base_command())
    assert instance.submit_command(first_id, base_command()).command_id == first.command_id
    with pytest.raises(ConflictError):
        instance.submit_command(first_id, base_command(x=9.0))
    instance.submit_command(second_id, base_command(command_id="move-two"))
    assert wait_terminal(instance, first_id, "move").status.value == "succeeded"
    assert wait_terminal(instance, second_id, "move-two").status.value == "succeeded"
    first_state = instance.backend.robot_state(first_id, 1)
    second_state = instance.backend.robot_state(second_id, 1)
    assert first_state.base_pose.position == (1.0, 2.0, 0.01)
    assert second_state.base_pose.position == (1.0, 2.0, 0.01)


def test_timeout_stops_motion_and_enters_hold(running):
    _, instance = running
    robot_id = instance.robot_ids()[0]
    request = base_command(command_id="timeout", duration=10.0)
    request = request.model_copy(update={"timeout_seconds": 0.01})
    instance.submit_command(robot_id, request)
    result = wait_terminal(instance, robot_id, "timeout")
    assert result.status.value == "failed"
    assert "超时" in (result.failure_reason or "")
    assert instance.backend.robot_state(robot_id, 1).in_hold


def test_stop_hold_and_resource_conflict(running):
    _, instance = running
    robot_id = instance.robot_ids()[0]
    instance.submit_command(robot_id, base_command(command_id="long", duration=5.0))
    with pytest.raises(ConflictError):
        instance.submit_command(robot_id, base_command(command_id="conflict", duration=5.0))
    stopped = instance.stop_command(robot_id, "long")
    assert stopped.status.value == "cancelled"
    assert instance.backend.robot_state(robot_id, 1).in_hold

    joint = RobotCommandRequest(
        command_id="joint",
        scene_generation=1,
        type="joint_trajectory",
        joint_trajectory={
            "resources": ["torso", "left_arm"],
            "frame_id": "world",
            "points": [
                {"time_from_start_seconds": 0.0, "positions": {"left_arm_joint1": 0.0}},
                {"time_from_start_seconds": 5.0, "positions": {"left_arm_joint1": 0.4}},
            ],
        },
    )
    # Runtime 的关节完成容差是公开低层契约。抓取用多末端 IK 时，若继续
    # 使用旧的 0.02 rad 隐式容差，多个躯干关节会在未到位时提前结束，并把
    # 小关节误差累计成厘米级末端偏差。
    assert joint.joint_trajectory.position_tolerance_rad == pytest.approx(0.002)
    assert joint.target["tolerance"] == pytest.approx(0.002)
    insert_payload = joint.model_dump(mode="json")
    insert_payload["command_id"] = "joint-insert"
    insert_payload["joint_trajectory"].update(
        {
            "stop_on_contact": True,
            "contact_tool_refs": ["component://tool/left"],
            "max_contact_force_n": 20.0,
        }
    )
    insert_joint = RobotCommandRequest.model_validate(insert_payload)
    assert insert_joint.target["stop_on_contact"] is True
    assert insert_joint.target["contact_tool_refs"] == ["component://tool/left"]
    assert insert_joint.target["max_contact_force_n"] == pytest.approx(20.0)

    contact_command = RobotCommandRequest(
        command_id="contact",
        scene_generation=1,
        type="gripper_command",
        gripper_command={
            "gripper_id": "left",
            "position": 0.015,
            "max_effort": 60,
            "stop_on_contact": True,
        },
    )
    assert contact_command.target["close_until_contact"] is True
    instance.submit_command(robot_id, joint)
    overlap = joint.model_copy(
        update={
            "command_id": "joint-overlap",
            "joint_trajectory": joint.joint_trajectory.model_copy(
                update={"resources": ["torso", "right_arm"]}
            ),
        }
    )
    with pytest.raises(ConflictError):
        instance.submit_command(robot_id, overlap)

    instance.hold_robot(robot_id)
    assert instance.command(robot_id, "joint").status.value == "cancelled"


def test_reset_rejects_active_command_and_invalidates_generation(running):
    _, instance = running
    robot_id = instance.robot_ids()[0]
    instance.submit_command(robot_id, base_command(command_id="active", duration=5.0))
    with pytest.raises(ConflictError):
        instance.reset()
    instance.stop_command(robot_id, "active")
    assert instance.reset().generation == 2
    with pytest.raises(ConflictError):
        instance.submit_command(robot_id, base_command(command_id="old", generation=1))


def test_sensors_and_snapshot_do_not_expose_mujoco_ids(running):
    _, instance = running
    robot_id = instance.robot_ids()[0]
    sensors = instance.components.sensors.sensor_descriptors(robot_id)
    assert {item.kind for item in sensors} == {"rgb", "depth", "contact"}
    snapshot = instance.snapshot()
    dumped = snapshot.model_dump_json()
    assert "geom" not in dumped and "body_id" not in dumped
    assert "quaternion_xyzw" in dumped and "frame_id" in dumped
    assert "quaternion_wxyz" not in dumped and '"frame"' not in dumped
    assert {item.source_id for item in snapshot.objects} == {"box-a1"}
