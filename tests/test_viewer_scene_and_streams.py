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
import struct
import time

from fastapi.testclient import TestClient

from plugin_mujoco.api import create_app
from plugin_mujoco.application import RuntimeManager
from plugin_mujoco.models import SceneStartRequest
from plugin_mujoco.settings import Settings
from plugin_mujoco.streaming import EncodedFrame


def _running(asset_root):
    manager = RuntimeManager(Settings(asset_root=asset_root, backend="fake", realtime=True))
    scene = manager.start(
        "palletizing_depalletizing_001",
        SceneStartRequest(request_id="viewer-scene-start", layout="layout001"),
    )
    assert manager.wait_ready(scene.instance_id).state.value == "running"
    return manager, manager.get(scene.instance_id)


def test_viewer_scene_glb_and_pose_stream_reset_generation(asset_root):
    manager, instance = _running(asset_root)
    try:
        scene = instance.viewer_scene()
        assert scene.generation == 1
        assert scene.scene_revision
        assert scene.content_url.endswith("/viewer-scene/content")
        assert scene.pose_stream_url.endswith("/pose-stream")
        assert scene.dynamic_node_order
        assert instance.viewer_scene_content().startswith(b"glTF")

        first = instance.scene_pose_frame()
        header, payload = EncodedFrame.unpack(first.packet())
        assert header["stream"] == "scene_pose"
        assert header["generation"] == 1
        assert header["scene_revision"] == scene.scene_revision
        assert header["node_count"] == len(scene.dynamic_node_order)
        assert len(payload) == header["node_count"] * 7 * 4
        assert len(struct.unpack("<" + "f" * header["node_count"] * 7, payload)) > 0

        instance.reset()
        reset = instance.scene_pose_frame()
        assert reset.metadata.generation == 2
        assert reset.metadata.sequence >= 1
        assert instance.viewer_scene().scene_revision == scene.scene_revision
    finally:
        manager.shutdown()


def test_binary_viewer_scene_robot_state_and_sensor_routes(asset_root):
    manager = RuntimeManager(Settings(asset_root=asset_root, backend="fake", realtime=True))
    with TestClient(create_app(manager=manager)) as client:
        scene = client.post(
            "/api/v1/scenes/palletizing_depalletizing_001/instances",
            json={
                "request_id": "binary-scene-flow",
                "layout": "layout001",
                "seed": 1,
                "headless": True,
                "render_backend": "auto",
            },
        ).json()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            record = client.get(
                f"/api/v1/scene-instances/{scene['instance_id']}"
            ).json()
            if record["state"] == "running":
                break
            time.sleep(0.005)

        instance_id = scene["instance_id"]
        viewer_scene = client.get(
            f"/api/v1/scene-instances/{instance_id}/viewer-scene"
        )
        assert viewer_scene.status_code == 200
        descriptor = viewer_scene.json()
        assert descriptor["generation"] == 1
        assert descriptor["dynamic_node_order"]

        content = client.get(
            f"/api/v1/scene-instances/{instance_id}/viewer-scene/content"
        )
        assert content.status_code == 200
        assert content.headers["content-type"].startswith("model/gltf-binary")
        assert content.content.startswith(b"glTF")

        with client.websocket_connect(
            f"/api/v1/scene-instances/{instance_id}/pose-stream"
        ) as socket:
            pose_header, pose_payload = EncodedFrame.unpack(socket.receive_bytes())
        assert pose_header["stream"] == "scene_pose"
        assert pose_header["generation"] == 1
        assert len(pose_payload) == pose_header["node_count"] * 28

        robot_id = client.get(
            f"/api/v1/scene-instances/{instance_id}/robots"
        ).json()[0]["robot_id"]
        with client.websocket_connect(f"/api/v1/robots/{robot_id}/state/stream") as socket:
            state_header, state_payload = EncodedFrame.unpack(socket.receive_bytes())
        state = json.loads(state_payload)
        assert state_header["stream"] == "robot_state"
        assert state_header["robot_id"] == robot_id == state["robot_id"]

        response = client.get(
            f"/robots/{robot_id}/sensors/camera.rgb/frames/latest/content"
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/jpeg")
        assert response.content.startswith(b"\xff\xd8")
