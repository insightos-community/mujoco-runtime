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

"""Runtime 侧低层 RobotDriver。"""

from __future__ import annotations

from typing import Any

from plugin_mujoco.models import RobotCommand, RobotProfile, RobotState
from plugin_mujoco.runtime.backend import CommandTick


class BackendRobotDriver:
    """只暴露轨迹跟随、状态和停止；不暴露 IK 或导航规划。"""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def robot_ids(self) -> list[str]:
        return self._backend.robot_ids()

    def robot_profile(self, robot_id: str) -> RobotProfile:
        return self._backend.robot_profile(robot_id)

    def robot_state(self, robot_id: str, generation: int) -> RobotState:
        return self._backend.robot_state(robot_id, generation)

    def advance_command(self, robot_id: str, command: RobotCommand, elapsed: float) -> CommandTick:
        return self._backend.advance_command(robot_id, command, elapsed)

    def stop_robot(self, robot_id: str) -> None:
        self._backend.stop_robot(robot_id)

    def hold_robot(self, robot_id: str) -> None:
        self._backend.hold_robot(robot_id)
