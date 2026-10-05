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

import time

from fastapi.testclient import TestClient

from plugin_mujoco.api import create_app
from plugin_mujoco.application import RuntimeManager
from plugin_mujoco.settings import Settings


def _wait_running(client, instance_id: str):
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        record = client.get(f"/api/v1/scene-instances/{instance_id}").json()
        if record["state"] == "running":
            return record
        time.sleep(0.005)
    raise AssertionError("场景没有进入 running")


def test_complete_api_flow(asset_root):
    manager = RuntimeManager(Settings(asset_root=asset_root, backend="fake", realtime=True))
    with TestClient(create_app(manager=manager)) as client:
        assert client.get("/healthz").status_code == 200
        visual = client.get("/api/v1/visual-assets/box/3.glb")
        assert visual.status_code == 200
        assert visual.headers["content-type"].startswith("model/gltf-binary")
        assert visual.headers["cache-control"].endswith("immutable")
        assert visual.content[:4] == b"glTF"
        assert client.get("/api/v1/visual-assets/box/1.glb").status_code == 404

        scenes = client.get("/api/v1/scenes").json()
        assert scenes[0]["layouts"] == ["layout001", "layout002", "layout003"]

        payload = {
            "request_id": "api-start",
            "layout": "layout001",
            "seed": 1,
            "headless": True,
            "render_backend": "auto",
        }
        response = client.post(
            "/api/v1/scenes/palletizing_depalletizing_001/instances",
            json=payload,
        )
        assert response.status_code == 201
        scene = response.json()
        instance_id = scene["instance_id"]

        same = client.post(
            "/api/v1/scenes/palletizing_depalletizing_001/instances",
            json=payload,
        )
        assert same.json()["instance_id"] == instance_id

        _wait_running(client, instance_id)
        profiles = client.get("/api/v1/runtime-profiles").json()
        assert {item["runtime_profile_id"] for item in profiles} == {
            "native-mujoco",
            "robosuite-1.5",
            "libero-robosuite-1.4",
        }
        robots = client.get(f"/api/v1/scene-instances/{instance_id}/robots").json()
        assert robots[0]["joint_names"] == ["left_arm_joint1", "right_arm_joint1"]
        assert all("gripper" not in name for name in robots[0]["joint_names"])
        robot_id = robots[0]["robot_id"]
        assert client.get(f"/robots/{robot_id}/profile").status_code == 200
        assert client.get(f"/api/v1/robots/{robot_id}/state").status_code == 200

        command = {
            "command_id": "api-command",
            "scene_generation": 1,
            "type": "joint_trajectory",
            "joint_trajectory": {
                "resources": ["left_arm"],
                "frame_id": "world",
                "points": [
                    {"time_from_start_seconds": 0.0, "positions": {"left_arm_joint1": 0.0}},
                    {"time_from_start_seconds": 0.02, "positions": {"left_arm_joint1": 0.3}},
                ],
            },
            "timeout_seconds": 1,
        }
        submitted = client.post(f"/robots/{robot_id}/commands", json=command)
        assert submitted.status_code == 202
        assert submitted.json()["command_id"] == "api-command"
        # Runtime 内部继续持有完整轨迹，但 HTTP 状态回执不能在每次轮询时
        # 重复返回大数组，否则长轨迹会让 SDK 把 accepted 误判为断连。
        assert "joint_trajectory" not in submitted.json()
        status = client.get(f"/robots/{robot_id}/commands/api-command")
        assert status.status_code == 200
        assert "joint_trajectory" not in status.json()
        held = client.post(f"/robots/{robot_id}/hold", json={"scene_generation": 1})
        assert held.status_code == 200
        assert held.json()["status"] == "succeeded"
        assert held.json()["type"] == "hold"
        stale_hold = client.post(f"/robots/{robot_id}/hold", json={"scene_generation": 2})
        assert stale_hold.status_code == 409

        snapshot = client.get(f"/api/v1/scene-instances/{instance_id}/snapshot").json()
        assert snapshot["generation"] == 1
        assert snapshot["robots"][0]["visual_ref"] == {
            "visual_id": "r1_pro_chassis",
            "version": "3",
        }
        target_region = next(item for item in snapshot["regions"] if item["source_id"] == "target")
        assert target_region["visual_ref"] == {
            "visual_id": "target", "version": "3"
        }
        assert (
            client.post(f"/api/v1/scene-instances/{instance_id}/pause").json()["state"] == "paused"
        )
        assert (
            client.post(f"/api/v1/scene-instances/{instance_id}/resume").json()["state"]
            == "running"
        )
        assert (
            client.post(f"/api/v1/scene-instances/{instance_id}/stop").json()["state"] == "stopped"
        )


def test_api_returns_structured_conflict(asset_root):
    manager = RuntimeManager(Settings(asset_root=asset_root, backend="fake"))
    with TestClient(create_app(manager=manager)) as client:
        response = client.get("/robots/unknown/state")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
