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

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from plugin_mujoco.api import create_app
from plugin_mujoco.application import RuntimeManager
from plugin_mujoco.settings import Settings


def build_payload() -> dict:
    return {
        "runtime_bundle_id": "build-1",
        "document_id": "scene-doc-1",
        "revision": 3,
        "scene_version": 1,
        "scene_key": "studio-depalletizing",
        "runtime_profile_id": "native-mujoco",
        "validation": {"valid": True, "issues": []},
        "created_at": "2026-08-09T00:00:00Z",
        "document": {
            "id": "scene-doc-1",
            "name": "Studio 拆码垛草稿",
            "scene_kind": "scene_document",
            "assets": [
                {
                    "id": "robot-r1-pro-chassis-v1",
                    "asset_key": "robot/r1_pro_chassis/config/r1_pro_chassis.xml",
                    "kind": "robot",
                    "metadata": {"model": "r1_pro_chassis"},
                },
                {
                    "id": "object-box-v1",
                    "asset_key": "assets/objects/box.xml",
                    "kind": "object",
                    "metadata": {"model": "box", "category": "box"},
                },
            ],
            "nodes": [
                {
                    "id": "r1pro-1",
                    "name": "R1 Pro",
                    "kind": "robot",
                    "asset_id": "robot-r1-pro-chassis-v1",
                    "transform": {
                        "position": [0, 0, 0.01],
                        "quaternion_xyzw": [0, 0, 0, 1],
                        "scale": [1, 1, 1],
                    },
                    "properties": {
                        "model": "r1_pro_chassis",
                        "sensor_names": ["camera"],
                    },
                },
                {
                    "id": "box-1",
                    "name": "Box",
                    "kind": "object",
                    "asset_id": "object-box-v1",
                    "transform": {
                        "position": [0.5, 0, 0.3],
                        "quaternion_xyzw": [0, 0, 0, 1],
                        "scale": [1, 1, 1],
                    },
                    "properties": {
                        "model": "box",
                        "category": "box",
                        "size": [0.4, 0.4, 0.4],
                        "mass": 0.5,
                        "inertia": [0.02, 0.02, 0.02],
                        "material": {"rgba": [0.85, 0.55, 0.2, 1]},
                        "collision": {"enabled": True, "friction": [1, 0.005, 0.0001]},
                    },
                },
                {
                    "id": "camera-node",
                    "name": "camera",
                    "kind": "camera",
                    "transform": {
                        "position": [1, -2, 1.5],
                        "quaternion_xyzw": [0, 0, 0, 1],
                        "scale": [1, 1, 1],
                    },
                    "properties": {"robot_id": "r1pro-1", "width": 64, "height": 48},
                },
                {
                    "id": "light-node",
                    "name": "key-light",
                    "kind": "light",
                    "transform": {
                        "position": [2, -2, 4],
                        "quaternion_xyzw": [0, 0, 0, 1],
                        "scale": [1, 1, 1],
                    },
                    "properties": {
                        "direction": [0, 0, -1],
                        "diffuse": [0.8, 0.8, 0.8],
                        "specular": [0.2, 0.2, 0.2],
                        "castshadow": True,
                        "active": True,
                    },
                },
            ],
            "regions": [
                {
                    "id": "target-1",
                    "name": "Target",
                    "kind": "region",
                    "transform": {
                        "position": [1.5, 0, 0.05],
                        "quaternion_xyzw": [0, 0, 0, 1],
                        "scale": [1, 1, 1],
                    },
                    "properties": {
                        "model": "target",
                        "category": "region",
                        "size": [1.4, 1.4, 0.05],
                        "static": True,
                        "interactive": True,
                        "material": {"rgba": [0.25, 0.75, 0.55, 0.38]},
                        "collision": {"enabled": False},
                    },
                }
            ],
            "physics": {
                "gravity_m_s2": [0, 0, -9.81],
                "timestep_seconds": 0.002,
            },
        },
    }


def test_scene_document_build_registers_immutable_runtime_scene(asset_root, tmp_path):
    authoring_root = tmp_path / "builds"
    manager = RuntimeManager(
        Settings(
            asset_root=asset_root,
            authoring_root=authoring_root,
            backend="fake",
            realtime=True,
        )
    )
    with TestClient(create_app(manager=manager)) as client:
        built = client.post("/api/v1/runtime-bundles", json=build_payload())
        assert built.status_code == 201
        assert built.json()["runtime_bundle_id"] == "build-1"
        assert built.json()["runtime_scene_key"] == "studio-depalletizing"
        assert built.json()["valid"] is True
        scenes = client.get("/api/v1/scenes").json()
        duplicate = client.post("/api/v1/runtime-bundles", json=build_payload())
        assert duplicate.status_code == 201
        conflicting_payload = build_payload()
        conflicting_payload["scene_version"] = 2
        conflict = client.post("/api/v1/runtime-bundles", json=conflicting_payload)
        assert conflict.status_code == 409
        descriptor = next(item for item in scenes if item["scene_key"] == "studio-depalletizing")
        assert descriptor["layouts"] == ["version-1"]
        assert descriptor["read_only"] is False
        definition = manager.catalog.load("studio-depalletizing", "version-1")
        box = next(item for item in definition.assets if item.source_id == "box-1")
        assert box.mass == 0.5
        assert box.inertia == (0.02, 0.02, 0.02)
        assert box.rgba == (0.85, 0.55, 0.2, 1.0)
        assert box.friction == (1.0, 0.005, 0.0001)
        assert box.collision_enabled is True
        base_xml = definition.base_xml.read_text(encoding="utf-8")
        assert 'name="light-node"' in base_xml
        assert 'castshadow="true"' in base_xml

        started = client.post(
            "/api/v1/scenes/studio-depalletizing/instances",
            json={
                "request_id": "start-built",
                "layout": "version-1",
                "seed": 0,
                "headless": True,
                "render_backend": "auto",
                "runtime_profile_id": "native-mujoco",
                "runtime_bundle_id": "build-1",
            },
        )
        assert started.status_code == 201
        instance_id = started.json()["instance_id"]
        assert manager.wait_ready(instance_id).state.value == "running"
        snapshot = client.get(f"/api/v1/scene-instances/{instance_id}/snapshot").json()
        assert snapshot["generation"] == 1
        assert {item["source_id"] for item in snapshot["objects"]} == {"box-1"}
        assert {item["source_id"] for item in snapshot["regions"]} == {"target-1"}

    restarted = RuntimeManager(
        Settings(asset_root=asset_root, authoring_root=authoring_root, backend="fake")
    )
    with TestClient(create_app(manager=restarted)) as client:
        scenes = client.get("/api/v1/scenes").json()
        descriptor = next(item for item in scenes if item["scene_key"] == "studio-depalletizing")
        assert descriptor["layouts"] == ["version-1"]


def test_scene_document_rejects_missing_robot_and_host_path(asset_root, tmp_path):
    payload = build_payload()
    payload["document"]["nodes"] = [
        {
            "id": "asset-node",
            "name": "Asset Reference",
            "kind": "asset",
            "transform": {
                "position": [0, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
                "scale": [1, 1, 1],
            },
        }
    ]
    payload["document"]["assets"] = [
        {"id": "bad", "asset_key": "/etc/passwd", "kind": "mesh", "metadata": {}}
    ]
    manager = RuntimeManager(
        Settings(asset_root=asset_root, authoring_root=tmp_path / "builds", backend="fake")
    )
    with TestClient(create_app(manager=manager)) as client:
        result = client.post("/api/v1/runtime-bundles", json=payload).json()
    assert result["valid"] is False
    messages = {issue["message"] for issue in result["issues"]}
    assert "资产必须使用资产仓内的相对标识" in messages
    assert "可运行的 MuJoCo 场景至少需要一个 Robot 节点" in messages
    assert any("独立 asset 节点" in message for message in messages)


def test_scene_document_reports_missing_asset_on_referencing_node(asset_root, tmp_path):
    payload = build_payload()
    box_asset = next(
        item for item in payload["document"]["assets"] if item["id"] == "object-box-v1"
    )
    box_asset["asset_key"] = "assets/objects/missing.xml"
    manager = RuntimeManager(
        Settings(asset_root=asset_root, authoring_root=tmp_path / "builds", backend="fake")
    )
    try:
        with TestClient(create_app(manager=manager)) as client:
            result = client.post("/api/v1/runtime-bundles", json=payload).json()
        assert result["valid"] is False
        issue = next(item for item in result["issues"] if "资产文件不存在" in item["message"])
        assert issue["node_id"] == "box-1"
        assert issue["field"] == "asset_id"
    finally:
        manager.shutdown()


@pytest.mark.native
def test_scene_document_build_runs_on_real_mujoco(tmp_path):
    """防止场景编辑链只在 Fake Backend 中成功、真实 MJCF 却无法加载。"""

    raw_root = os.getenv("MUJOCO_ASSET_ROOT")
    if not raw_root:
        pytest.skip("MUJOCO_ASSET_ROOT 未设置")
    manager = RuntimeManager(
        Settings(
            asset_root=Path(raw_root),
            authoring_root=tmp_path / "builds",
            backend="mujoco",
            render_backend="egl",
            realtime=True,
        )
    )
    try:
        with TestClient(create_app(manager=manager)) as client:
            built = client.post("/api/v1/runtime-bundles", json=build_payload())
            assert built.status_code == 201
            assert built.json()["valid"] is True

            started = client.post(
                "/api/v1/scenes/studio-depalletizing/instances",
                json={
                    "request_id": "start-real-authored",
                    "runtime_profile_id": "native-mujoco",
                    "runtime_bundle_id": "build-1",
                    "layout": "version-1",
                    "seed": 7,
                    "headless": True,
                    "render_backend": "egl",
                },
            )
            assert started.status_code == 201
            instance_id = started.json()["instance_id"]
            ready = manager.wait_ready(instance_id, timeout=30)
            assert ready.state.value == "running", ready.failure_reason

            instance = manager.get(instance_id)
            snapshot = instance.snapshot()
            assert snapshot.generation == 1
            assert {item.source_id for item in snapshot.objects} == {"box-1"}
            assert {item.source_id for item in snapshot.regions} == {"target-1"}
            assert instance.robot_ids() == ["r1pro-1"]

            viewer_scene = instance.viewer_scene()
            assert viewer_scene.dynamic_node_order
            assert instance.viewer_scene_content().startswith(b"glTF")
            assert instance.stop().state.value == "stopped"
    finally:
        manager.shutdown()
