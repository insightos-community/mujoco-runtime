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

"""用统一 MuJoCo 导出器生成单资产 GLB。

场景 Viewer、Scene Editor 和 Semantic Map 都使用 MujocoVisualExporter。
本模块只保留 Asset Drawer 在没有活动场景时所需的单资产入口，不再维护另一套
GLB Builder；因此坐标、材质、Mesh 和 Robot Link 处理与完整场景完全一致。
"""

from __future__ import annotations

import threading
from pathlib import Path

from semantic_mujoco_visuals import MujocoVisualExporter

from plugin_mujoco.errors import BackendFailureError, NotFoundError

VISUAL_CONTENT_VERSION = "3"
_R1_VISUAL_RGBA = [0.5, 0.9, 0.2, 1.0]


def public_visual_id(model: str, category: str) -> str:
    """把 Runtime 物理模型映射为稳定视觉 ID。"""

    return category if category in {"box", "pallet", "target"} else model


class RuntimeVisualAssetStore:
    """生成 Runtime 已登记资产的浏览器视觉内容。

    这里只接受固定视觉 ID，不接受任意文件路径。生成结果缓存于当前 Runtime
    进程；正式场景仍通过 viewer-scene 暴露完整 GLB。
    """

    _PRIMITIVES = {
        "box": ((0.5, 0.5, 0.5), (0.85, 0.55, 0.2, 1.0)),
        "pallet": ((1.2, 1.0, 0.15), (0.15, 0.22, 0.4, 1.0)),
        "target": ((1.4, 1.4, 0.05), (0.25, 0.75, 0.55, 0.38)),
    }

    def __init__(self, asset_root: Path) -> None:
        self.asset_root = asset_root.resolve()
        self._cache: dict[tuple[str, str], bytes] = {}
        self._lock = threading.Lock()

    def content(self, visual_id: str, version: str) -> bytes:
        key = (visual_id.strip(), version.strip())
        if version != VISUAL_CONTENT_VERSION or visual_id not in {
            "r1_pro_chassis",
            *self._PRIMITIVES,
        }:
            raise NotFoundError(f"视觉资产不存在: {visual_id}@{version}")
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            try:
                content = (
                    self._robot_glb()
                    if visual_id == "r1_pro_chassis"
                    else self._primitive_glb(visual_id, *self._PRIMITIVES[visual_id])
                )
            except NotFoundError:
                raise
            except Exception as exc:
                raise BackendFailureError(
                    f"生成视觉资产失败: {visual_id}",
                    details={"visual_id": visual_id, "version": version},
                ) from exc
            self._cache[key] = content
            return content

    def _primitive_glb(
        self,
        visual_id: str,
        size: tuple[float, float, float],
        rgba: tuple[float, float, float, float],
    ) -> bytes:
        import mujoco

        half = tuple(value / 2.0 for value in size)
        xml = (
            "<mujoco><worldbody>"
            f'<body name="{visual_id}" pos="0 0 0">'
            f'<geom name="{visual_id}-visual" type="box" '
            f'size="{half[0]} {half[1]} {half[2]}" '
            f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>'
            "</body></worldbody></mujoco>"
        )
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return (
            MujocoVisualExporter(
                model,
                data,
                mujoco,
                source_for_body=lambda _body_id: visual_id,
            )
            .export()
            .content
        )

    def _robot_glb(self) -> bytes:
        import mujoco

        xml_path = self.asset_root / "robot" / "r1_pro_chassis" / "config" / "r1_pro_chassis.xml"
        if not xml_path.is_file():
            raise NotFoundError("R1 Pro 视觉模型未安装")
        # Robot 是场景 include 的 MJCF fragment，visualgeom 材质由宿主提供。
        # 单资产预览只在内存补同名材质，不修改资产文件或维护另一套模型。
        spec = mujoco.MjSpec.from_file(str(xml_path))
        if spec.material("visualgeom") is None:
            spec.add_material(name="visualgeom", rgba=_R1_VISUAL_RGBA)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return (
            MujocoVisualExporter(
                model,
                data,
                mujoco,
                source_for_body=lambda _body_id: "r1_pro_chassis",
            )
            .export()
            .content
        )
