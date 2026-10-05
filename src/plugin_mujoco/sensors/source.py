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

"""Robot 传感器数据适配器。"""

from __future__ import annotations

from typing import Any

from plugin_mujoco.models import SensorDescriptor, SensorFrame
from plugin_mujoco.rendering.executor import RenderExecutor
from plugin_mujoco.streaming import EncodedFrame


class BackendSensorSource:
    def __init__(self, backend: Any, executor: RenderExecutor) -> None:
        self._backend = backend
        self._executor = executor

    def sensor_descriptors(self, robot_id: str) -> list[SensorDescriptor]:
        return self._backend.sensor_descriptors(robot_id)

    def sensor_frame(self, robot_id: str, sensor_id: str) -> SensorFrame:
        # 结构化传感器也通过同一入口；相机类型会在专用线程创建并使用 EGL Context。
        return self._executor.call(lambda: self._backend.sensor_frame(robot_id, sensor_id))

    def encoded_sensor_frame(self, robot_id: str, sensor_id: str, *, generation: int, sequence: int, quality: int = 85) -> EncodedFrame:
        return self._executor.call(lambda: self._backend.encoded_sensor_frame(
            robot_id, sensor_id, generation=generation, sequence=sequence, quality=quality
        ))
