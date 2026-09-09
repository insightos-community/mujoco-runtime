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

"""Runtime 对物理后端的最小调用面。"""

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
from plugin_mujoco.streaming import EncodedFrame


@dataclass(frozen=True)
class CommandTick:
    completed: bool = False
    failed: bool = False
    reason: str | None = None


class SimulationBackend(Protocol):
    @property
    def sim_time(self) -> float: ...

    @property
    def timestep(self) -> float: ...

    def step(self) -> None: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...

    def robot_ids(self) -> list[str]: ...

    def robot_profile(self, robot_id: str) -> RobotProfile: ...

    def robot_state(self, robot_id: str, generation: int) -> RobotState: ...

    def advance_command(
        self,
        robot_id: str,
        command: RobotCommand,
        elapsed: float,
    ) -> CommandTick: ...

    def stop_robot(self, robot_id: str) -> None: ...

    def hold_robot(self, robot_id: str) -> None: ...

    def sensor_descriptors(self, robot_id: str) -> list[SensorDescriptor]: ...

    def sensor_frame(self, robot_id: str, sensor_id: str) -> SensorFrame: ...

    def encoded_sensor_frame(
        self, robot_id: str, sensor_id: str, *, generation: int, sequence: int, quality: int = 85
    ) -> EncodedFrame: ...


    def scene_objects(self) -> list[SceneObject]: ...

    def scene_regions(self) -> list[SceneRegion]: ...
