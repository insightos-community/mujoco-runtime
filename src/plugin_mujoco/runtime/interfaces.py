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

"""Runtime 各职责的最小接口。

接口拆分后，生命周期不再依赖一个无所不包的 Backend。当前 native 适配器
会逐步把同一实现对象拆为独立组件，但调用方向已经固定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from plugin_mujoco.models import (
    RobotCommand,
    RobotProfile,
    RobotState,
    SceneObject,
    SceneRegion,
    SensorDescriptor,
    SensorFrame,
)
from plugin_mujoco.runtime.backend import CommandTick
from plugin_mujoco.streaming import EncodedFrame
from plugin_mujoco.visuals.scene import SceneVisualProvider


class PhysicsRuntime(Protocol):
    @property
    def sim_time(self) -> float: ...
    @property
    def timestep(self) -> float: ...
    def step(self) -> None: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...


class RobotDriver(Protocol):
    def robot_ids(self) -> list[str]: ...
    def robot_profile(self, robot_id: str) -> RobotProfile: ...
    def robot_state(self, robot_id: str, generation: int) -> RobotState: ...
    def advance_command(
        self, robot_id: str, command: RobotCommand, elapsed: float
    ) -> CommandTick: ...
    def stop_robot(self, robot_id: str) -> None: ...
    def hold_robot(self, robot_id: str) -> None: ...


class SensorSource(Protocol):
    def sensor_descriptors(self, robot_id: str) -> list[SensorDescriptor]: ...
    def sensor_frame(self, robot_id: str, sensor_id: str) -> SensorFrame: ...
    def encoded_sensor_frame(
        self, robot_id: str, sensor_id: str, *, generation: int, sequence: int, quality: int = 85
    ) -> EncodedFrame: ...



class SceneSnapshotProvider(Protocol):
    def scene_objects(self) -> list[SceneObject]: ...
    def scene_regions(self) -> list[SceneRegion]: ...


@dataclass(frozen=True)
class RuntimeComponents:
    """一个 Runtime 的独立职责；场景视觉与 Sensor Renderer 严格分离。"""

    physics: PhysicsRuntime
    robots: RobotDriver
    sensors: SensorSource
    snapshot: SceneSnapshotProvider
    visuals: SceneVisualProvider

    @classmethod
    def from_combined_backend(cls, backend: object) -> RuntimeComponents:
        """迁移入口：先切断调用依赖，再逐个搬移 native 具体实现。"""
        return cls(backend, backend, backend, backend, backend)  # type: ignore[arg-type]
