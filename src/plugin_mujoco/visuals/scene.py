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

"""Runtime 场景视觉内容适配。

模型加载完成时生成一次 GLB；reset 只更新 generation 和位姿，不重新构建内容。
连续位姿只由物理线程调用 ``capture``，API 线程不会接触 MjData。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from semantic_mujoco_visuals import MujocoVisualExporter

from plugin_mujoco.models import StrictModel


class ViewerCameraDescriptor(StrictModel):
    camera_id: str
    name: str
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    fovy: float


class ViewerScene(StrictModel):
    generation: int
    scene_revision: str
    coordinate_frame: str = "world"
    content_url: str
    pose_stream_url: str
    dynamic_node_order: list[str]
    cameras: list[ViewerCameraDescriptor]


@dataclass(frozen=True)
class CapturedScenePose:
    node_count: int
    payload: bytes


class SceneVisualProvider(Protocol):
    @property
    def scene_revision(self) -> str: ...
    @property
    def dynamic_node_order(self) -> tuple[str, ...]: ...
    @property
    def cameras(self) -> tuple[ViewerCameraDescriptor, ...]: ...
    def content(self) -> bytes: ...
    def capture(self) -> CapturedScenePose: ...


class BackendSceneVisualProvider:
    """把 native Backend 的 MjModel/MjData 交给共用导出器。"""

    def __init__(self, backend: object) -> None:
        exporter = MujocoVisualExporter(
            backend.model,  # type: ignore[attr-defined]
            backend.data,  # type: ignore[attr-defined]
            backend.mj,  # type: ignore[attr-defined]
            backend.visual_source_for_body,  # type: ignore[attr-defined]
        )
        exported = exporter.export()
        self._exporter = exporter
        self._content = exported.content
        self._order = exported.dynamic_node_order
        self._cameras = tuple(
            ViewerCameraDescriptor(
                camera_id=item.camera_id,
                name=item.name,
                position=item.position,
                quaternion_xyzw=item.quaternion_xyzw,
                fovy=item.fovy,
            )
            for item in exported.cameras
        )
        self._revision = hashlib.sha256(self._content).hexdigest()[:20]

    @property
    def scene_revision(self) -> str:
        return self._revision

    @property
    def dynamic_node_order(self) -> tuple[str, ...]:
        return self._order

    @property
    def cameras(self) -> tuple[ViewerCameraDescriptor, ...]:
        return self._cameras

    def content(self) -> bytes:
        return self._content

    def capture(self) -> CapturedScenePose:
        poses = np.asarray(self._exporter.poses(), dtype="<f4")
        return CapturedScenePose(int(poses.shape[0]), poses.tobytes(order="C"))
