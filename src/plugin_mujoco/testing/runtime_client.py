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

"""仅供 Plugin 独立验收和进程测试使用的同步客户端。

产品代码必须使用 robot-sdk-r1pro，不能依赖这里的模型或调用路径。
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from plugin_mujoco.models import (
    RobotCommand,
    RobotCommandRequest,
    RobotProfile,
    RobotState,
    SceneInstance,
    SceneSnapshot,
    SceneStartRequest,
    SensorDescriptor,
    SensorFrame,
)


class RuntimeClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def start_scene(self, scene_key: str, request: SceneStartRequest) -> SceneInstance:
        response = self._client.post(
            f"/api/v1/scenes/{scene_key}/instances",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()
        return SceneInstance.model_validate(response.json())

    def scene(self, instance_id: str) -> SceneInstance:
        response = self._client.get(f"/api/v1/scene-instances/{instance_id}")
        response.raise_for_status()
        return SceneInstance.model_validate(response.json())

    def wait_scene(self, instance_id: str, *, timeout: float = 30.0) -> SceneInstance:
        """等待异步模型加载结束；failed/stopped 也会立即返回供诊断。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            scene = self.scene(instance_id)
            if scene.state.value in {"running", "failed", "stopped"}:
                return scene
            time.sleep(0.05)
        raise TimeoutError(f"等待场景启动超时: {instance_id}")

    def pause(self, instance_id: str) -> SceneInstance:
        return self._scene_operation(instance_id, "pause")

    def resume(self, instance_id: str) -> SceneInstance:
        return self._scene_operation(instance_id, "resume")

    def step(self, instance_id: str, steps: int = 1) -> SceneInstance:
        response = self._client.post(
            f"/api/v1/scene-instances/{instance_id}/step",
            json={"steps": steps},
        )
        response.raise_for_status()
        return SceneInstance.model_validate(response.json())

    def reset(self, instance_id: str) -> SceneInstance:
        return self._scene_operation(instance_id, "reset")

    def stop(self, instance_id: str) -> SceneInstance:
        return self._scene_operation(instance_id, "stop")

    def snapshot(self, instance_id: str) -> SceneSnapshot:
        response = self._client.get(f"/api/v1/scene-instances/{instance_id}/snapshot")
        response.raise_for_status()
        return SceneSnapshot.model_validate(response.json())

    def robots(self, instance_id: str) -> list[RobotProfile]:
        response = self._client.get(f"/api/v1/scene-instances/{instance_id}/robots")
        response.raise_for_status()
        return [RobotProfile.model_validate(value) for value in response.json()]

    def robot(self, robot_id: str) -> RobotSdkClient:
        return RobotSdkClient(self._client, robot_id)

    def _scene_operation(self, instance_id: str, operation: str) -> SceneInstance:
        response = self._client.post(f"/api/v1/scene-instances/{instance_id}/{operation}")
        response.raise_for_status()
        return SceneInstance.model_validate(response.json())


class RobotSdkClient:
    def __init__(self, client: httpx.Client, robot_id: str) -> None:
        self._client = client
        self.robot_id = robot_id

    def profile(self) -> RobotProfile:
        return self._get_model(f"/robots/{self.robot_id}/profile", RobotProfile)

    def state(self) -> RobotState:
        return self._get_model(f"/robots/{self.robot_id}/state", RobotState)

    def command(self, request: RobotCommandRequest) -> RobotCommand:
        response = self._client.post(
            f"/robots/{self.robot_id}/commands",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()
        return RobotCommand.model_validate(response.json())

    def command_status(self, command_id: str) -> RobotCommand:
        return self._get_model(
            f"/robots/{self.robot_id}/commands/{command_id}",
            RobotCommand,
        )

    def wait(
        self,
        command_id: str,
        *,
        timeout: float = 30.0,
        interval: float = 0.05,
    ) -> RobotCommand:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            command = self.command_status(command_id)
            if command.status.value in {"succeeded", "failed", "cancelled", "unknown"}:
                return command
            time.sleep(interval)
        raise TimeoutError(f"等待命令超时: {command_id}")

    def stop(self, command_id: str) -> RobotCommand:
        response = self._client.post(f"/robots/{self.robot_id}/commands/{command_id}/stop")
        response.raise_for_status()
        return RobotCommand.model_validate(response.json())

    def hold(self, generation: int) -> None:
        response = self._client.post(
            f"/robots/{self.robot_id}/hold",
            json={"scene_generation": generation},
        )
        response.raise_for_status()

    def sensors(self) -> list[SensorDescriptor]:
        response = self._client.get(f"/robots/{self.robot_id}/sensors")
        response.raise_for_status()
        return [SensorDescriptor.model_validate(value) for value in response.json()]

    def sensor_frame(self, sensor_id: str) -> SensorFrame:
        return self._get_model(
            f"/robots/{self.robot_id}/sensors/{sensor_id}/frames/latest",
            SensorFrame,
        )

    def sensor_content(self, sensor_id: str) -> tuple[bytes, dict[str, str]]:
        """读取正式二进制帧；只保留验收证据所需的公开元数据。"""
        response = self._client.get(
            f"/api/v1/robots/{self.robot_id}/sensors/{sensor_id}/frames/latest/content"
        )
        response.raise_for_status()
        names = (
            "content-type",
            "x-semantic-sequence",
            "x-semantic-generation",
            "x-semantic-encoding",
            "x-semantic-frame",
            "x-semantic-width",
            "x-semantic-height",
            "x-semantic-observed-at",
        )
        return response.content, {name: response.headers[name] for name in names}

    def _get_model(self, path: str, model: Any):
        response = self._client.get(path)
        response.raise_for_status()
        return model.model_validate(response.json())
