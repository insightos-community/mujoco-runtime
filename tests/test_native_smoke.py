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

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml

from plugin_mujoco.application import RuntimeManager
from plugin_mujoco.models import SceneStartRequest
from plugin_mujoco.settings import Settings


def _glb_document(content: bytes) -> dict:
    """读取 GLB JSON chunk，验证浏览器实际收到的材质而非导出器内部状态。"""
    assert content[:4] == b"glTF"
    json_length, json_kind = struct.unpack_from("<I4s", content, 12)
    assert json_kind == b"JSON"
    return json.loads(content[20 : 20 + json_length].decode("utf-8"))


@pytest.mark.native
@pytest.mark.parametrize("layout", ["layout001", "layout002", "layout003"])
def test_real_mujoco_loads_each_layout_and_produces_snapshot(layout):
    raw_root = os.getenv("MUJOCO_ASSET_ROOT")
    if not raw_root:
        pytest.skip("MUJOCO_ASSET_ROOT 未设置")
    root = Path(raw_root)
    manager = RuntimeManager(
        Settings(asset_root=root, backend="mujoco", realtime=True)
    )
    try:
        scene = manager.start(
            "palletizing_depalletizing_001",
            SceneStartRequest(
                request_id=f"native-{layout}",
                layout=layout,
                seed=1,
                headless=True,
                render_backend="auto",
            ),
        )
        ready = manager.wait_ready(scene.instance_id, timeout=30)
        assert ready.state.value == "running", ready.failure_reason
        instance = manager.get(scene.instance_id)
        assert instance.robot_ids()
        snapshot = instance.snapshot()
        assert snapshot.objects
        assert snapshot.sensors
        manifest_path = root / "scene" / "palletizing_depalletizing_001" / "asset-manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        expected = set(manifest["layouts"][layout]["expected_objects"])
        actual = {item.source_id for item in (*snapshot.objects, *snapshot.regions)}
        assert actual == expected
        if layout == "layout001":
            viewer_scene = instance.viewer_scene()
            assert viewer_scene.dynamic_node_order
            # 固定相机不是浏览器猜出的视角：位置、FOV 与姿态均直接来自当前
            # MjData。使用 MuJoCo 官方矩阵转四元数函数作独立对照，防止转置或
            # wxyz/xyzw 顺序错误。
            backend = instance.components.physics._backend
            camera = viewer_scene.cameras[0]
            expected_wxyz = np.empty(4, dtype=np.float64)
            backend.mj.mju_mat2Quat(expected_wxyz, backend.data.cam_xmat[0])
            assert camera.position == pytest.approx(backend.data.cam_xpos[0])
            assert camera.quaternion_xyzw == pytest.approx((*expected_wxyz[1:], expected_wxyz[0]))
            assert camera.fovy == pytest.approx(float(backend.model.cam_fovy[0]))
            document = _glb_document(instance.viewer_scene_content())
            assert document["images"], "地面 MuJoCo 纹理必须嵌入 GLB"
            assert "KHR_materials_specular" in document["extensionsUsed"]
            visual = next(
                item for item in document["materials"] if item.get("name") == "visualgeom"
            )
            color = visual["pbrMetallicRoughness"]["baseColorFactor"]
            assert color[:3] == pytest.approx([0.5, 0.9, 0.2], abs=1e-5)
            ground = next(item for item in document["materials"] if item.get("name") == "matplane")
            transform = ground["pbrMetallicRoughness"]["baseColorTexture"]["extensions"][
                "KHR_texture_transform"
            ]
            assert transform["scale"] == pytest.approx([100.0, 100.0])
        assert instance.pause().state.value == "paused"
        assert instance.resume().state.value == "running"
        assert instance.stop().state.value == "stopped"
    finally:
        manager.shutdown()
