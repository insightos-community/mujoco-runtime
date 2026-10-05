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

"""独立 Runtime 验收客户端；只验证低层轨迹执行，不伪造 IK 或导航。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

from plugin_mujoco.models import RobotCommandRequest, SceneStartRequest
from plugin_mujoco.testing.runtime_client import RuntimeClient


def run() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo Runtime 低层接口验收")
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--layout", default="layout001")
    parser.add_argument("--output", default=".output/demo")
    parser.add_argument("--scene-smoke", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    client = RuntimeClient(args.url)
    try:
        started = client.start_scene(
            "palletizing_depalletizing_001",
            SceneStartRequest(
                request_id=f"demo-{uuid4()}", layout=args.layout, seed=7,
                headless=True, render_backend="egl", runtime_profile_id="native-mujoco",
            ),
        )
        scene = client.wait_scene(started.instance_id)
        if scene.state.value != "running":
            raise RuntimeError(f"场景启动失败: {scene.failure_reason}")
        robot = client.robot(client.robots(scene.instance_id)[0].robot_id)
        initial = robot.state()
        snapshot = client.snapshot(scene.instance_id)
        _write_json(output / "profile.json", robot.profile().model_dump(mode="json"))
        _write_json(output / "state-before.json", initial.model_dump(mode="json"))
        _write_json(output / "snapshot-before.json", snapshot.model_dump(mode="json"))
        for sensor in robot.sensors():
            _write_sensor(output, robot, sensor)

        commands: list[dict] = []
        if not args.scene_smoke:
            x, y = initial.base_pose.position[:2]
            commands.append(_execute(robot, scene.generation, "base_trajectory", {
                "frame_id": "world",
                "points": [
                    {"time_from_start_seconds": 0.0, "positions": {"x": x, "y": y, "yaw": 0.0}},
                    {"time_from_start_seconds": 0.5, "positions": {"x": x + 0.1, "y": y, "yaw": 0.05}},
                ]
            }, 3.0).model_dump(mode="json"))
            joint_name = next(name for name in initial.joints if "arm_joint" in name)
            current = robot.state().joints[joint_name].position
            commands.append(_execute(robot, scene.generation, "joint_trajectory", {
                "resources": ["arm"],
                "frame_id": "world",
                "points": [
                    {"time_from_start_seconds": 0.0, "positions": {joint_name: current}},
                    {"time_from_start_seconds": 0.5, "positions": {joint_name: current + 0.03}},
                ]
            }, 3.0).model_dump(mode="json"))

        paused = client.pause(scene.instance_id)
        stepped = client.step(scene.instance_id, 3)
        if stepped.step_count != paused.step_count + 3:
            raise RuntimeError("暂停后的单步没有精确推进")
        client.resume(scene.instance_id)
        time.sleep(0.1)
        before_reset = client.snapshot(scene.instance_id)
        reset = client.reset(scene.instance_id)
        if reset.generation != scene.generation + 1:
            raise RuntimeError("reset 后 generation 没有增加")
        _write_json(output / "snapshot-before-reset.json", before_reset.model_dump(mode="json"))
        _write_json(output / "snapshot-after-reset.json", client.snapshot(scene.instance_id).model_dump(mode="json"))
        _write_json(output / "commands.json", commands)
        _write_json(output / "report.json", {
            "layout": args.layout, "robot_id": robot.robot_id,
            "object_count": len(snapshot.objects), "sensor_count": len(robot.sensors()),
            "generation_before": scene.generation, "generation_after": reset.generation,
            "trajectory_commands": len(commands), "result": "passed",
            "note": "IK、导航与抓取由 semantic-robot-sdk/Ability 组合测试验证",
        })
        client.stop(scene.instance_id)
        print(f"Runtime 验收证据已保存到 {output.resolve()}")
    finally:
        client.close()


def _execute(robot, generation: int, command_kind: str, payload: dict, timeout: float):
    command = robot.command(RobotCommandRequest.model_validate({
        "command_id": f"{command_kind}-{uuid4()}",
        "scene_generation": generation,
        "type": command_kind,
        command_kind: payload,
        "timeout_seconds": timeout,
    }))
    result = robot.wait(command.command_id, timeout=timeout + 2)
    if result.status.value != "succeeded":
        raise RuntimeError(f"{command_kind} 失败: {result.failure_reason}")
    return result


def _write_sensor(output: Path, robot, descriptor) -> None:
    if descriptor.kind in {"rgb", "depth"}:
        payload, metadata = robot.sensor_content(descriptor.sensor_id)
        suffix = ".jpg" if descriptor.kind == "rgb" else ".png"
        (output / f"{descriptor.sensor_id}{suffix}").write_bytes(payload)
        _write_json(output / f"{descriptor.sensor_id}.metadata.json", metadata)
    else:
        frame = robot.sensor_frame(descriptor.sensor_id)
        _write_json(output / f"{frame.sensor_id}.json", frame.model_dump(mode="json"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
