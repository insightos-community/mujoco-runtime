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

"""场景公共快照适配器。"""

from __future__ import annotations

from typing import Any

from plugin_mujoco.models import SceneObject, SceneRegion


class BackendSceneSnapshotProvider:
    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def scene_objects(self) -> list[SceneObject]:
        return self._backend.scene_objects()

    def scene_regions(self) -> list[SceneRegion]:
        return self._backend.scene_regions()
