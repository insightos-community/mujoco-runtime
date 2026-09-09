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
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient

import semantic_sim_profiles.runtime_service as runtime_module
from semantic_sim_profiles.runtime_api import create_app
from semantic_sim_profiles.runtime_service import (
    FRANKA_JOINT_NAMES,
    ProfileRuntimeError,
    ProfileRuntimeService,
)


class FakeProfileAdapter:
    """不加载 MuJoCo 的确定性 adapter，用于验证 Runtime 的线程和状态语义。"""

    def __init__(self) -> None:
        self.closed = False
        self.steps = 0
        self.positions = {name: 0.01 * index for index, name in enumerate(FRANKA_JOINT_NAMES)}
        self.last_action: Dict[str, Any] | None = None
        self.base_pose_thread_ids: list[int] = []
        self.follow_commands = True
        self.gripper_opening_m = 0.04
        self._visual_model = SimpleNamespace(
            nbody=3,
            body_parentid=np.asarray([0, 0, 0], dtype=np.int32),
            body_pos=np.asarray([[0, 0, 0], [1, 2, 0.3], [0.5, 0.1, 0.04]], dtype=float),
            body_quat=np.asarray([[1, 0, 0, 0]] * 3, dtype=float),
            ngeom=2,
            geom_bodyid=np.asarray([1, 2], dtype=np.int32),
            geom_type=np.asarray([6, 6], dtype=np.int32),
            geom_size=np.asarray([[0.2, 0.2, 0.3], [0.04, 0.04, 0.04]], dtype=float),
            geom_pos=np.zeros((2, 3), dtype=float),
            geom_quat=np.asarray([[1, 0, 0, 0]] * 2, dtype=float),
            geom_group=np.asarray([0, 0], dtype=np.int32),
            geom_rgba=np.asarray([[0.4, 0.6, 0.9, 1], [0.8, 0.5, 0.2, 1]], dtype=float),
            geom_matid=np.asarray([-1, -1], dtype=np.int32),
            geom_dataid=np.asarray([-1, -1], dtype=np.int32),
            ncam=0,
            camera_id2name=lambda _camera_id: None,
            body_id2name=lambda body_id: ("world", "robot0_panda", "cube_main")[body_id],
        )
        self._visual_data = SimpleNamespace(
            xpos=np.zeros((3, 3), dtype=float),
            xquat=np.asarray([[1, 0, 0, 0]] * 3, dtype=float),
            cam_xpos=np.zeros((0, 3), dtype=float),
            cam_xmat=np.zeros((0, 9), dtype=float),
        )
        self._sync_visual_data()

    def reset(self, seed: int) -> Dict[str, Any]:
        self.steps = 0
        self.gripper_opening_m = 0.04
        self._sync_visual_data()
        return self._observation(seed)

    def visual_model_data(self) -> tuple[Any, Any]:
        return self._visual_model, self._visual_data

    def visual_source_for_body(self, body_id: int, _object_source_ids: set[str]) -> str | None:
        return {1: "franka-0", 2: "cube"}.get(body_id)

    def neutral_action(self) -> Dict[str, Any]:
        return {"positions": dict(self.positions), "gripper": 0.0}

    def joint_positions(self) -> Dict[str, float]:
        return dict(self.positions)

    def base_pose(self) -> Dict[str, Any]:
        self.base_pose_thread_ids.append(threading.get_ident())
        return {
            "position": [1.0 + self.steps * 0.01, 2.0, 0.3],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 2.0],
            "frame_id": "world",
        }

    def gripper_opening(self) -> float:
        return self.gripper_opening_m

    def joint_position_action(
        self, target: Dict[str, float], *, gripper_action: float
    ) -> Dict[str, Any]:
        return {"positions": dict(target), "gripper": gripper_action}

    def step(self, action: Dict[str, Any]) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.steps += 1
        self.last_action = action
        if self.follow_commands:
            self.positions.update(action["positions"])
            self.gripper_opening_m = max(
                0.0, min(0.08, self.gripper_opening_m - float(action["gripper"]) * 0.01)
            )
        self._sync_visual_data()
        return self._observation(0), 0.25, False, {"step": self.steps}

    def contact_state(self) -> Dict[str, Any]:
        return {"active": self.steps > 0, "count": int(self.steps > 0), "holding": False}

    def success(self) -> bool:
        return self.steps > 0

    def native_metrics(self) -> Dict[str, Any]:
        return {"environment": "FakeLift", "steps": self.steps}

    def close(self) -> None:
        self.closed = True

    def _sync_visual_data(self) -> None:
        self._visual_data.xpos[:] = np.asarray(
            [[0.0, 0.0, 0.0], [1.0 + self.steps * 0.01, 2.0, 0.3], [0.5, 0.1, 0.04]]
        )

    def _observation(self, seed: int) -> Dict[str, Any]:
        return {
            "agentview_image": np.full((8, 12, 3), seed % 255, dtype=np.uint8),
            "agentview_depth": np.full((8, 12, 1), 0.25, dtype=np.float32),
            "robot0_eef_pos": np.asarray([0.4, 0.0, 0.5]),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "robot0_joint_vel": np.zeros(7),
            "robot0_gripper_qpos": np.asarray(
                [self.gripper_opening_m / 2, -self.gripper_opening_m / 2]
            ),
            "cube_pos": np.asarray([0.5, 0.1, 0.04]),
            "cube_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        }


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch):
    adapters: list[FakeProfileAdapter] = []

    def factory(*_args: Any, **_kwargs: Any):
        def create() -> FakeProfileAdapter:
            adapter = FakeProfileAdapter()
            adapters.append(adapter)
            return adapter

        return create

    monkeypatch.setattr(runtime_module, "_adapter_factory", factory)
    current = ProfileRuntimeService("robosuite-1.5")
    yield current, adapters
    current.shutdown()


def start(service: ProfileRuntimeService, request_id: str = "start-1"):
    instance = service.start_scene(
        "Lift",
        {
            "request_id": request_id,
            "runtime_profile_id": "robosuite-1.5",
            "layout": "default",
            "seed": 7,
            "headless": True,
            "render_backend": "egl",
        },
    )
    instance.wait_ready(2.0)
    return instance


def test_profile_runtime_exposes_worker_cached_non_origin_base_pose(service) -> None:
    current, adapters = service
    instance = start(current)
    instance.pause()
    adapter = adapters[0]
    calls_before = len(adapter.base_pose_thread_ids)
    state = instance.robot_state("franka-0")
    assert state["base_pose"]["position"][1:] == pytest.approx([2.0, 0.3])
    assert state["base_pose"]["position"][0] >= 1.0
    assert state["base_pose"]["quaternion_xyzw"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert state["base_pose"]["frame_id"] == "world"
    assert len(adapter.base_pose_thread_ids) == calls_before
    assert all(thread_id != threading.get_ident() for thread_id in adapter.base_pose_thread_ids)

    # 调用方修改返回值不能污染 Runtime 的跨线程缓存。
    state["base_pose"]["position"][0] = 999.0
    second = instance.robot_state("franka-0")
    assert second["base_pose"]["position"][0] != 999.0
    assert len(adapter.base_pose_thread_ids) == calls_before


def test_profile_runtime_lifecycle_generation_and_idempotency(service) -> None:
    current, adapters = service
    instance = start(current)
    assert current.start_scene("Lift", dict(instance.request)) is instance
    with pytest.raises(ProfileRuntimeError, match="启动输入不同"):
        current.start_scene(
            "Stack",
            {
                **instance.request,
                "runtime_profile_id": "robosuite-1.5",
            },
        )

    time.sleep(0.08)
    instance.pause()
    paused_steps = instance.step_count
    time.sleep(0.12)
    assert instance.step_count == paused_steps

    instance.step_once(2)
    assert instance.step_count == paused_steps + 2
    instance.resume()
    old_generation = instance.generation
    instance.reset()
    assert instance.generation == old_generation + 1
    assert instance.snapshot()["objects"][0]["source_id"] == "cube"

    instance.stop()
    assert instance.state == "stopped"
    assert adapters[0].closed is True


def test_failed_profile_requires_explicit_stop_before_replacement(service) -> None:
    current, _ = service
    instance = start(current)
    with instance._condition:
        instance.state = "failed"
    info = current.runtime_info()
    assert info["state"] == "failed"
    assert info["active_instance_id"] == instance.instance_id
    replacement_request = {**instance.request, "request_id": "replacement-before-stop"}
    with pytest.raises(ProfileRuntimeError, match="活动场景"):
        current.start_scene("Lift", replacement_request)
    assert instance.stop()["state"] == "stopped"
    replacement = start(current, "replacement-before-stop")
    assert replacement.instance_id != instance.instance_id


def test_profile_runtime_executes_low_level_trajectory_and_contact(service) -> None:
    current, _ = service
    instance = start(current)
    initial = instance.robot_state("franka-0")
    target = {
        name: state["position"] + (0.02 if name == "panda_joint1" else 0.0)
        for name, state in initial["joints"].items()
    }
    command = {
        "command_id": "joint-1",
        "scene_generation": instance.generation,
        "type": "joint_trajectory",
        "timeout_seconds": 2.0,
        "joint_trajectory": {
            "resources": ["joints:arm"],
            "frame_id": "panda_link0",
            "points": [
                {
                    "time_from_start_seconds": 0.0,
                    "positions": {
                        name: state["position"] for name, state in initial["joints"].items()
                    },
                },
                {
                    "time_from_start_seconds": 0.1,
                    "positions": target,
                },
            ],
        },
    }
    accepted = instance.submit_command("franka-0", command)
    assert accepted["status"] == "accepted"
    assert instance.submit_command("franka-0", command)["command_id"] == "joint-1"

    deadline = time.monotonic() + 2.0
    result = accepted
    while time.monotonic() < deadline:
        result = instance.command("franka-0", "joint-1")
        if result["status"] == "succeeded":
            break
        time.sleep(0.02)
    assert result["status"] == "succeeded"
    assert instance.robot_state("franka-0")["joints"]["panda_joint1"]["position"] == pytest.approx(
        target["panda_joint1"]
    )

    descriptors = instance.sensor_descriptors("franka-0")
    assert {item["kind"] for item in descriptors} == {"rgb", "depth", "contact"}
    kind, contact, sequence = instance.sensor_payload("franka-0", "robot0_contact")
    assert kind == "contact"
    assert contact["active"] is True
    assert sequence == 1

    with pytest.raises(ProfileRuntimeError, match="generation"):
        instance.submit_command(
            "franka-0", {**command, "command_id": "old", "scene_generation": 999}
        )


def test_profile_gripper_succeeds_only_after_actual_opening_converges(service) -> None:
    current, _ = service
    instance = start(current)
    accepted = instance.submit_command(
        "franka-0",
        {
            "command_id": "gripper-observed",
            "scene_generation": instance.generation,
            "type": "gripper_command",
            "timeout_seconds": 1.0,
            "gripper_command": {
                "gripper_id": "hand",
                "position": 0.02,
                "max_effort": 20.0,
            },
        },
    )
    assert accepted["status"] == "accepted"

    deadline = time.monotonic() + 2.0
    result = accepted
    while time.monotonic() < deadline:
        result = instance.command("franka-0", "gripper-observed")
        if result["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.02)

    assert result["status"] == "succeeded", result
    state = instance.robot_state("franka-0")
    assert state["grippers"]["hand"] == pytest.approx(
        0.02,
        abs=instance.gripper_opening_tolerance_m,
    )
    assert result["position_error_m"] <= instance.gripper_opening_tolerance_m


def test_profile_unreached_target_fails_and_stop_holds_observed_position(service) -> None:
    current, adapters = service
    instance = start(current)
    adapter = adapters[0]
    initial = instance.robot_state("franka-0")
    positions = {name: value["position"] for name, value in initial["joints"].items()}
    target = dict(positions)
    target["panda_joint1"] += 0.4
    command = {
        "command_id": "stuck-target",
        "scene_generation": instance.generation,
        "type": "joint_trajectory",
        "timeout_seconds": 2.0,
        "joint_trajectory": {
            "resources": ["joints:arm"],
            "frame_id": "panda_link0",
            "points": [{"time_from_start_seconds": 0.0, "positions": target}],
        },
    }

    adapter.follow_commands = False
    instance.joint_settle_timeout_s = 0.1
    instance.submit_command("franka-0", command)
    deadline = time.monotonic() + 2.0
    result = {}
    while time.monotonic() < deadline:
        result = instance.command("franka-0", "stuck-target")
        if result.get("status") == "failed":
            break
        time.sleep(0.02)
    assert result["status"] == "failed"
    assert result["failure_reason"] == "target_not_reached"
    assert instance.robot_state("franka-0")["joints"]["panda_joint1"]["position"] == pytest.approx(
        positions["panda_joint1"]
    )

    adapter.follow_commands = True
    instance.joint_settle_timeout_s = 5.0
    moving = {
        **command,
        "command_id": "stop-observed",
        "joint_trajectory": {
            **command["joint_trajectory"],
            "points": [
                {"time_from_start_seconds": 0.0, "positions": positions},
                {"time_from_start_seconds": 2.0, "positions": target},
            ],
        },
    }
    instance.submit_command("franka-0", moving)
    time.sleep(0.08)
    stopped = instance.stop_command("franka-0", "stop-observed")
    observed_after_stop = instance.robot_state("franka-0")["joints"]["panda_joint1"]["position"]
    time.sleep(0.12)
    assert stopped["status"] == "cancelled"
    assert instance.robot_state("franka-0")["in_hold"] is True
    assert instance.robot_state("franka-0")["joints"]["panda_joint1"]["position"] == pytest.approx(
        observed_after_stop
    )


def test_profile_runtime_timeout_finishes_command_and_holds(service) -> None:
    current, _ = service
    instance = start(current)
    state = instance.robot_state("franka-0")
    positions = {name: value["position"] for name, value in state["joints"].items()}
    target = dict(positions)
    target["panda_joint1"] += 0.5
    instance.submit_command(
        "franka-0",
        {
            "command_id": "timeout-1",
            "scene_generation": instance.generation,
            "type": "joint_trajectory",
            "timeout_seconds": 0.05,
            "joint_trajectory": {
                "resources": ["joints:arm"],
                "frame_id": "panda_link0",
                "points": [
                    {"time_from_start_seconds": 0.0, "positions": positions},
                    {"time_from_start_seconds": 10.0, "positions": target},
                ],
            },
        },
    )

    deadline = time.monotonic() + 2.0
    result = {}
    while time.monotonic() < deadline:
        result = instance.command("franka-0", "timeout-1")
        if result["status"] == "failed":
            break
        time.sleep(0.02)
    assert result["status"] == "failed"
    assert result["failure_reason"] == "timeout"
    assert instance.robot_state("franka-0")["in_hold"] is True


def test_profile_runtime_http_and_binary_frames(service) -> None:
    current, _ = service
    app = create_app(current)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/scenes/Lift/instances",
            json={
                "request_id": "http-start",
                "runtime_profile_id": "robosuite-1.5",
                "layout": "default",
                "seed": 3,
                "headless": True,
                "render_backend": "egl",
            },
        )
        assert response.status_code == 201
        instance_id = response.json()["instance_id"]
        current.instance(instance_id).wait_ready(2.0)

        robots = client.get(f"/api/v1/scene-instances/{instance_id}/robots").json()
        assert robots[0]["sdk_package"] == "robot-sdk-franka"

        with client.websocket_connect("/api/v1/robots/franka-0/state/stream") as socket:
            state_packet = socket.receive_bytes()
        state_header_size = int.from_bytes(state_packet[:4], "big")
        state_header = json.loads(state_packet[4 : 4 + state_header_size])
        state_payload = json.loads(state_packet[4 + state_header_size :])
        assert state_header["stream"] == "robot_state"
        assert state_header["robot_id"] == "franka-0"
        assert state_header["generation"] == state_payload["generation"]
        assert state_header["observed_at"] == state_payload["observed_at"]
        assert state_header["frame_id"] == state_payload["base_pose"]["frame_id"]
        assert state_header["encoding"] == "json"
        assert state_header["media_type"] == "application/json"
        assert state_header["sim_time"] >= 0

        rgb = client.get("/api/v1/robots/franka-0/sensors/agentview_rgb/frames/latest/content")
        assert rgb.status_code == 200
        assert rgb.headers["content-type"].startswith("image/jpeg")
        assert rgb.content.startswith(b"\xff\xd8")

        time.sleep(0.06)
        evaluation = client.get(f"/api/v1/scene-instances/{instance_id}/evaluation")
        assert evaluation.status_code == 200
        assert evaluation.json()["reward"] == pytest.approx(0.25)
        assert evaluation.json()["success"] is True
        assert evaluation.json()["metrics"]["environment"] == "FakeLift"

        # 公共样例由四仓共享；这里比较完整键集合，防止 Profile API 悄悄增加、
        # 删除或改名字段，而主 Plugin 的 Pydantic 测试负责校验字段类型。
        contract_path = (
            Path(__file__).resolve().parents[3] / "examples/contracts/scene-evaluation.json"
        )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        assert set(evaluation.json()) == set(contract)

        depth = client.get("/robots/franka-0/sensors/agentview_depth/frames/latest/content")
        assert depth.status_code == 200
        assert depth.headers["x-semantic-encoding"] == "float32-le"
        assert len(depth.content) == 8 * 12 * 4

        viewer_scene = client.get(
            f"/api/v1/scene-instances/{instance_id}/viewer-scene"
        )
        assert viewer_scene.status_code == 200
        descriptor = viewer_scene.json()
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
            packet = socket.receive_bytes()
        header_size = int.from_bytes(packet[:4], "big")
        pose_header = json.loads(packet[4 : 4 + header_size])
        assert pose_header["stream"] == "scene_pose"
        assert pose_header["node_count"] == len(descriptor["dynamic_node_order"])
        assert len(packet[4 + header_size :]) == pose_header["node_count"] * 28
        rejected = client.post("/api/v1/runtime-bundles", json={})
        assert rejected.status_code == 422
