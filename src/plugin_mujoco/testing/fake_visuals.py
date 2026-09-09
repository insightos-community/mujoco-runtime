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

"""Fake Runtime 的确定性视觉 provider。

它只让不安装 MuJoCo 的生命周期测试继续覆盖 ViewerScene/pose 契约。产品 Runtime
仍统一使用公共 MjModel/MjData 导出器，本实现不参与正式验收。
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
from semantic_mujoco_visuals import MujocoVisualExporter

from plugin_mujoco.visuals.scene import CapturedScenePose, ViewerCameraDescriptor


class FakeSceneVisualProvider:
    def __init__(self, backend: object) -> None:
        self._backend = backend
        definition = backend.definition
        self._entries = [
            *[
                (
                    "robot",
                    item.robot_id,
                    item.position,
                    item.quaternion_xyzw,
                    (0.6, 0.6, 1.6),
                    (0.35, 0.55, 0.85, 1.0),
                )
                for item in definition.robots
            ],
            *[
                (
                    "object",
                    item.source_id,
                    item.position,
                    item.quaternion_xyzw,
                    item.size or (0.4, 0.4, 0.4),
                    item.rgba or (0.7, 0.7, 0.72, 1.0),
                )
                for item in definition.assets
                if item.category != "region"
            ],
        ]
        count = len(self._entries)
        model = SimpleNamespace(
            nbody=count + 1,
            ngeom=count,
            ncam=0,
            geom_bodyid=np.arange(1, count + 1, dtype=np.int32),
            geom_group=np.zeros(count, dtype=np.int32),
            geom_rgba=np.asarray([entry[5] for entry in self._entries], dtype=np.float64),
            geom_type=np.full(count, 6, dtype=np.int32),
            geom_dataid=np.full(count, -1, dtype=np.int32),
            geom_size=np.asarray(
                [[value * 0.5 for value in entry[4]] for entry in self._entries],
                dtype=np.float64,
            ),
            geom_pos=np.zeros((count, 3), dtype=np.float64),
            geom_quat=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (count, 1)),
        )
        data = SimpleNamespace(
            xpos=np.zeros((count + 1, 3), dtype=np.float64),
            xquat=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (count + 1, 1)),
            cam_xpos=np.empty((0, 3), dtype=np.float64),
            cam_xmat=np.empty((0, 9), dtype=np.float64),
        )
        self._data = data
        self._sync()
        exporter = MujocoVisualExporter(
            model,
            data,
            source_for_body=lambda body_id: self._entries[body_id - 1][1],
        )
        exported = exporter.export()
        self._exporter = exporter
        self._content = exported.content
        self._revision = hashlib.sha256(self._content).hexdigest()[:20]
        self._order = exported.dynamic_node_order

    @property
    def scene_revision(self) -> str:
        return self._revision

    @property
    def dynamic_node_order(self) -> tuple[str, ...]:
        return self._order

    @property
    def cameras(self) -> tuple[ViewerCameraDescriptor, ...]:
        return ()

    def content(self) -> bytes:
        return self._content

    def capture(self) -> CapturedScenePose:
        self._sync()
        poses = np.asarray(self._exporter.poses(), dtype="<f4")
        return CapturedScenePose(int(poses.shape[0]), poses.tobytes(order="C"))

    def _sync(self) -> None:
        for index, entry in enumerate(self._entries, start=1):
            kind, source_id, position, quaternion, _size, _rgba = entry
            if kind == "robot":
                state = self._backend.robot_state(source_id, 1)
                position = state.base_pose.position
                quaternion = state.base_pose.quaternion_xyzw
            self._data.xpos[index] = position
            self._data.xquat[index] = (
                quaternion[3],
                quaternion[0],
                quaternion[1],
                quaternion[2],
            )
