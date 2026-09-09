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

"""真实 robosuite 门控：低层命令必须由 Robot 实际观测确认完成。

本文件只在 ``profiles/robosuite`` 隔离环境中显式运行，不能加入不安装
robosuite / MuJoCo 的 common 快速测试。测试不启动 HTTP 服务，直接使用同一
Profile Runtime 状态机，确保成功状态不是由轨迹时钟伪造出来的。
"""

from __future__ import annotations

import time
from typing import Any, Dict

import pytest

from semantic_sim_profiles.robosuite import RobosuiteAdapter
from semantic_sim_profiles.runtime_service import (
    FRANKA_JOINT_NAMES,
    ProfileRuntimeInstance,
)


@pytest.fixture
def real_lift_runtime() -> ProfileRuntimeInstance:
    instance = ProfileRuntimeInstance(
        profile_id="robosuite-1.5",
        scene_key="Lift",
        layout="default",
        request={
            "request_id": "real-command-convergence",
            "runtime_profile_id": "robosuite-1.5",
            "layout": "default",
            "seed": 11,
            "headless": True,
            "render_backend": "egl",
        },
        adapter_factory=lambda: RobosuiteAdapter(
            "Lift",
            camera_names=("agentview",),
            width=64,
            height=64,
            horizon=10000,
        ),
    )
    instance.start()
    instance.wait_ready(30.0)
    try:
        yield instance
    finally:
        if instance.state not in {"stopped", "failed"}:
            instance.stop()


def _wait_terminal(
    instance: ProfileRuntimeInstance,
    command_id: str,
    *,
    timeout: float = 12.0,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    result: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        result = instance.command("franka-0", command_id)
        if result["status"] in {"succeeded", "failed", "cancelled", "unknown"}:
            return result
        time.sleep(0.02)
    pytest.fail("真实 robosuite 命令未在门控时间内结束: %s" % result)


def test_real_robosuite_uses_observed_joint_and_gripper_convergence(
    real_lift_runtime: ProfileRuntimeInstance,
) -> None:
    instance = real_lift_runtime
    initial = instance.robot_state("franka-0")
    start_positions = {
        name: float(initial["joints"][name]["position"]) for name in FRANKA_JOINT_NAMES
    }
    target_positions = dict(start_positions)
    target_positions["panda_joint1"] += 0.05

    instance.submit_command(
        "franka-0",
        {
            "command_id": "real-joint-convergence",
            "scene_generation": instance.generation,
            "type": "joint_trajectory",
            "timeout_seconds": 5.0,
            "joint_trajectory": {
                "resources": ["joints:arm"],
                "frame_id": "panda_link0",
                "points": [
                    {"time_from_start_seconds": 0.0, "positions": start_positions},
                    {"time_from_start_seconds": 0.5, "positions": target_positions},
                ],
            },
        },
    )
    joint_result = _wait_terminal(instance, "real-joint-convergence")
    assert joint_result["status"] == "succeeded", joint_result
    actual_joint = instance.robot_state("franka-0")["joints"]["panda_joint1"]["position"]
    assert actual_joint == pytest.approx(
        target_positions["panda_joint1"],
        abs=instance.joint_position_tolerance_rad,
    )
    assert joint_result["position_error_rad"] <= instance.joint_position_tolerance_rad

    # Panda GRIP 的输入是方向，不是目标开度。Runtime 必须持续读取真实 qpos，
    # 在开度连续进入米制容差后才能把命令标记为 succeeded。
    instance.submit_command(
        "franka-0",
        {
            "command_id": "real-gripper-convergence",
            "scene_generation": instance.generation,
            "type": "gripper_command",
            "timeout_seconds": 8.0,
            "gripper_command": {
                "gripper_id": "hand",
                "position": 0.0,
                "max_effort": 20.0,
            },
        },
    )
    gripper_result = _wait_terminal(instance, "real-gripper-convergence")
    assert gripper_result["status"] == "succeeded", gripper_result
    actual_opening = instance.robot_state("franka-0")["grippers"]["hand"]
    assert actual_opening == pytest.approx(0.0, abs=instance.gripper_opening_tolerance_m)
    assert gripper_result["position_error_m"] <= instance.gripper_opening_tolerance_m

    # 命令结束后下一物理周期只能保持实际位置，不应继续积分夹爪方向。
    time.sleep(0.15)
