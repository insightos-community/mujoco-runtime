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
