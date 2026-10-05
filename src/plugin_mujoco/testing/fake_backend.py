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

"""无需 MuJoCo 的确定性后端。

它只用于模块测试和并行开发样例。产品验收必须再运行 Native MuJoCo 测试。
"""

from __future__ import annotations

import base64
import math
from copy import deepcopy

import numpy as np

from plugin_mujoco.errors import NotFoundError
from plugin_mujoco.imaging import encode_depth_png16, encode_rgb_jpeg
from plugin_mujoco.models import (
    GripperState,
    JointState,
    Pose,
    RobotCapability,
    RobotCommand,
    RobotProfile,
    RobotState,
    SceneObject,
    SceneRegion,
    SensorDescriptor,
    SensorFrame,
    utc_now,
)
from plugin_mujoco.runtime.backend import CommandTick
from plugin_mujoco.scene import SceneDefinition
from plugin_mujoco.streaming import EncodedFrame, SensorFrameMetadata
from plugin_mujoco.visuals import VISUAL_CONTENT_VERSION, public_visual_id


class FakeBackend:
    def __init__(
        self, definition: SceneDefinition, *, endpoint: str = "http://127.0.0.1:8090"
    ) -> None:
        self.definition = definition
        self.endpoint = endpoint
        self._sim_time = 0.0
        self._closed = False
        self._states: dict[str, dict] = {}
        for robot in definition.robots:
            self._states[robot.robot_id] = {
                "x": robot.position[0],
                "y": robot.position[1],
                "z": robot.position[2],
                "yaw": _yaw(robot.quaternion_xyzw),
                "joints": {
                    "left_arm_joint1": 0.0,
                    "right_arm_joint1": 0.0,
                    "left_gripper_finger_joint1": 0.05,
                    "left_gripper_finger_joint2": 0.05,
                    "right_gripper_finger_joint1": 0.05,
                    "right_gripper_finger_joint2": 0.05,
                },
                "gripper_targets": {},
                "hold": True,
            }
        self._initial = deepcopy(self._states)

    @property
    def sim_time(self) -> float:
        return self._sim_time

    @property
    def timestep(self) -> float:
        return self.definition.timestep

    def step(self) -> None:
        if not self._closed:
            self._sim_time += self.timestep

    def reset(self) -> None:
        self._sim_time = 0.0
        self._states = deepcopy(self._initial)

    def close(self) -> None:
        self._closed = True

    def robot_ids(self) -> list[str]:
        return list(self._states)

    def robot_profile(self, robot_id: str) -> RobotProfile:
        state = self._get(robot_id)
        return RobotProfile(
            robot_id=robot_id,
            model="r1_pro_chassis",
            backend="mujoco",
            kind="mobile_manipulator",
            coordinate_frame="world",
            sdk_package="semantic-robot-sdk-r1pro",
            backend_profile="r1pro-tote-mujoco-v1",
            endpoint=self.endpoint,
            joint_names=sorted(name for name in state["joints"] if "_gripper_" not in name),
            end_effectors=["left", "right"],
            grippers=["left", "right"],
            capabilities=RobotCapability(
                commands=["joint_trajectory", "base_trajectory", "gripper_command"],
                sensors=["camera.rgb", "camera.depth", "contact"],
                frames=["world", "left", "right"],
            ),
        )

    def robot_state(self, robot_id: str, generation: int) -> RobotState:
        value = self._get(robot_id)
        joints = {
            name: JointState(position=position, velocity=0.0, effort=0.0)
            for name, position in value["joints"].items()
        }
        pose = Pose(
            position=(value["x"], value["y"], value["z"]),
            quaternion_xyzw=(
                0.0,
                0.0,
                math.sin(value["yaw"] * 0.5),
                math.cos(value["yaw"] * 0.5),
            ),
        )
        return RobotState(
            robot_id=robot_id,
            generation=generation,
            observed_at=utc_now(),
            base_pose=pose,
            joints=joints,
            end_effectors={"left": pose, "right": pose},
            grippers={
                "left": joints["left_gripper_finger_joint1"].position,
                "right": joints["right_gripper_finger_joint1"].position,
            },
            gripper_states={
                side: GripperState(
                    position=joints[f"{side}_gripper_finger_joint1"].position,
                    velocity=0.0,
                    effort=float(value["gripper_targets"].get(side, {}).get("force_limit_n", 0)),
                    target_position=value["gripper_targets"].get(side, {}).get("position"),
                    reached_target=side in value["gripper_targets"],
                )
                for side in ("left", "right")
            },
            in_hold=bool(value["hold"]),
        )

    def advance_command(
        self,
        robot_id: str,
        command: RobotCommand,
        elapsed: float,
    ) -> CommandTick:
        """跟随 Robot SDK 已生成的轨迹；这里不执行 IK 或路径搜索。"""
        value = self._get(robot_id)
        value["hold"] = False
        target = command.target
        if command.type == "base_trajectory":
            sample, finished = _trajectory_sample(target["points"], elapsed)
            value["x"], value["y"], value["yaw"] = sample["x"], sample["y"], sample["yaw"]
            return CommandTick(completed=finished)
        if command.type == "joint_trajectory":
            sample, finished = _trajectory_sample(target["points"], elapsed)
            value["joints"].update(sample["positions"])
            return CommandTick(completed=finished)
        if command.type == "gripper_command":
            side = target["gripper"]
            names = [name for name in value["joints"] if name.startswith(f"{side}_gripper_")]
            if not names:
                return CommandTick(failed=True, reason=f"Robot Profile 中没有夹爪: {side}")
            for name in names:
                value["joints"][name] = target["opening"]
            value["gripper_targets"][side] = {
                "position": target["opening"],
                "force_limit_n": float(target.get("force_limit_n") or 0.0),
            }
            # Runtime Fake 不合成接触事实；要求真实接触的命令保持运行，
            # 由上层超时或停止。Robot SDK 自己的 FakeBackend 另有显式 fixture。
            return CommandTick(completed=not bool(target.get("close_until_contact")))
        return CommandTick(failed=True, reason=f"不支持的低层命令: {command.type}")

    def stop_robot(self, robot_id: str) -> None:
        self.hold_robot(robot_id)

    def hold_robot(self, robot_id: str) -> None:
        self._get(robot_id)["hold"] = True

    def sensor_descriptors(self, robot_id: str) -> list[SensorDescriptor]:
        self._get(robot_id)
        return [
            SensorDescriptor(
                sensor_id="camera.rgb",
                robot_id=robot_id,
                kind="rgb",
                frame_id="camera",
                width=2,
                height=2,
            ),
            SensorDescriptor(
                sensor_id="camera.depth",
                robot_id=robot_id,
                kind="depth",
                frame_id="camera",
                width=2,
                height=2,
            ),
            SensorDescriptor(
                sensor_id="contact", robot_id=robot_id, kind="contact", frame_id="world"
            ),
        ]

    def sensor_frame(self, robot_id: str, sensor_id: str) -> SensorFrame:
        self._get(robot_id)
        descriptor = next(
            (item for item in self.sensor_descriptors(robot_id) if item.sensor_id == sensor_id),
            None,
        )
        if descriptor is None:
            raise NotFoundError(f"传感器不存在: {sensor_id}")
        if descriptor.kind == "contact":
            data: str | dict = {
                "active": False,
                "count": 0,
                "contacts": [],
                "tools": {},
            }
            media_type, encoding = "application/json", "json"
        else:
            data = base64.b64encode(bytes(range(12))).decode("ascii")
            media_type, encoding = "application/octet-stream", "base64"
        return SensorFrame(
            sensor_id=sensor_id,
            kind=descriptor.kind,
            observed_at=utc_now(),
            media_type=media_type,
            encoding=encoding,
            width=descriptor.width,
            height=descriptor.height,
            data=data,
        )

    def encoded_sensor_frame(
        self,
        robot_id: str,
        sensor_id: str,
        *,
        generation: int,
        sequence: int,
        quality: int = 85,
    ) -> EncodedFrame:
        descriptor = next(
            (item for item in self.sensor_descriptors(robot_id) if item.sensor_id == sensor_id),
            None,
        )
        if descriptor is None:
            raise NotFoundError(f"传感器不存在: {sensor_id}")
        width, height = int(descriptor.width or 2), int(descriptor.height or 2)
        if descriptor.kind == "rgb":
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[:, :, 0] = int(self._sim_time * 20) % 255
            payload = encode_rgb_jpeg(image, quality=quality)
            media_type, encoding = "image/jpeg", "jpeg"
        elif descriptor.kind == "depth":
            image = np.full((height, width), 1.25, dtype=np.float32)
            payload = encode_depth_png16(image)
            media_type, encoding = "image/png", "png16-mm"
        else:
            payload = self.sensor_frame(robot_id, sensor_id).model_dump_json().encode("utf-8")
            media_type, encoding = "application/json", "json"
        return EncodedFrame(
            SensorFrameMetadata(
                stream="sensor",
                sequence=sequence,
                generation=generation,
                observed_at=utc_now(),
                media_type=media_type,
                encoding=encoding,
                width=width,
                height=height,
                frame=descriptor.frame,
                sensor_id=sensor_id,
            ),
            payload,
        )

    def scene_objects(self) -> list[SceneObject]:
        return [
            SceneObject(
                source_id=item.source_id,
                category=item.category,
                name=item.source_id,
                pose=Pose(position=item.position, quaternion_xyzw=item.quaternion_xyzw),
                extent=item.size,
                visual_ref={
                    "visual_id": public_visual_id(item.model, item.category),
                    "version": VISUAL_CONTENT_VERSION,
                },
                state={"interactive": item.interactive, "static": item.static},
            )
            for item in self.definition.assets
            if item.category != "region"
        ]

    def scene_regions(self) -> list[SceneRegion]:
        return [
            SceneRegion(
                source_id=item.source_id,
                name=item.source_id,
                pose=Pose(position=item.position, quaternion_xyzw=item.quaternion_xyzw),
                extent=item.size,
                visual_ref={
                    "visual_id": public_visual_id(item.model, item.category),
                    "version": VISUAL_CONTENT_VERSION,
                },
            )
            for item in self.definition.assets
            if item.category == "region"
        ]

    def _get(self, robot_id: str) -> dict:
        try:
            return self._states[robot_id]
        except KeyError as exc:
            raise NotFoundError(f"Robot 不存在: {robot_id}") from exc


def _yaw(quaternion_xyzw: tuple[float, float, float, float]) -> float:
    x, y, z, w = quaternion_xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _trajectory_sample(points: list[dict], elapsed: float) -> tuple[dict, bool]:
    """Fake 后端使用最近采样点，测试幂等和调度而不伪造规划能力。"""
    selected = points[0]
    for point in points[1:]:
        if float(point["time_from_start"]) > elapsed:
            break
        selected = point
    return selected, elapsed >= float(points[-1]["time_from_start"])
