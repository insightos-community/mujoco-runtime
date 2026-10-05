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

"""Runtime 二进制流协议。

Sensor、Robot State 与 Scene Pose 复用同一封包方式：四字节大端 JSON 头长度，
随后是 JSON 元数据和二进制 payload。Physics Viewer 不再属于图像帧协议；它只
消费一次 GLB 与 float32 位姿流。
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import Field

from plugin_mujoco.models import StrictModel


class SensorFrameMetadata(StrictModel):
    stream: Literal["sensor"] = "sensor"
    sequence: int = Field(ge=1)
    generation: int = Field(ge=1)
    observed_at: datetime
    media_type: str
    encoding: str
    width: int
    height: int
    frame: str
    sensor_id: str


class RobotStateFrameMetadata(StrictModel):
    stream: Literal["robot_state"] = "robot_state"
    sequence: int = Field(ge=1)
    generation: int = Field(ge=1)
    sim_time: float = Field(ge=0.0)
    observed_at: datetime
    frame_id: str = Field(min_length=1)
    robot_id: str = Field(min_length=1)
    media_type: Literal["application/json"] = "application/json"
    encoding: Literal["json"] = "json"


class PoseFrameMetadata(StrictModel):
    """Scene GLB 动态节点的世界位姿流头。"""

    stream: Literal["scene_pose"] = "scene_pose"
    sequence: int = Field(ge=1)
    generation: int = Field(ge=1)
    scene_revision: str = Field(min_length=1)
    sim_time: float = Field(ge=0.0)
    node_count: int = Field(ge=0)
    coordinate_frame: Literal["world"] = "world"
    encoding: Literal["float32-le-xyz-xyzw"] = "float32-le-xyz-xyzw"


FrameMetadata = SensorFrameMetadata | RobotStateFrameMetadata | PoseFrameMetadata


@dataclass(frozen=True)
class EncodedFrame:
    metadata: FrameMetadata
    payload: bytes

    def packet(self) -> bytes:
        header = self.metadata.model_dump_json().encode("utf-8")
        return struct.pack(">I", len(header)) + header + self.payload

    @classmethod
    def unpack(cls, packet: bytes) -> tuple[dict, bytes]:
        if len(packet) < 4:
            raise ValueError("帧数据不完整")
        length = struct.unpack(">I", packet[:4])[0]
        if length <= 0 or len(packet) < 4 + length:
            raise ValueError("帧头长度无效")
        return json.loads(packet[4 : 4 + length]), packet[4 + length :]
