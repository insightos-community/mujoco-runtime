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
import math
import os
import struct
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from plugin_mujoco.application import RuntimeManager
from plugin_mujoco.errors import ConflictError
from plugin_mujoco.models import RobotCommandRequest, SceneStartRequest
from plugin_mujoco.settings import Settings
from plugin_mujoco.streaming import EncodedFrame
from plugin_mujoco.visuals import RuntimeVisualAssetStore


def _manager() -> RuntimeManager:
    raw_root = os.getenv("MUJOCO_ASSET_ROOT")
    if not raw_root:
        pytest.skip("MUJOCO_ASSET_ROOT 未设置")
    return RuntimeManager(
        Settings(
            asset_root=Path(raw_root),
            backend="mujoco",
            render_backend="egl",
            realtime=True,
        )
    )


def _wait(instance, robot_id, command_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        command = instance.command(robot_id, command_id)
        if command.status.value in {"succeeded", "failed", "cancelled", "unknown"}:
            return command
        time.sleep(0.01)
    raise AssertionError(f"命令未在 {timeout}s 内结束: {command_id}")


@pytest.mark.native
def test_native_low_level_trajectory_render_stop_timeout_and_generation():
    """Native 验收只验证轨迹执行；IK 和导航由 semantic-robot-sdk 测试。"""
    manager = _manager()
    try:
        started = manager.start(
            "palletizing_depalletizing_001",
            SceneStartRequest(
                request_id="native-sdk", layout="layout001", seed=2, render_backend="egl"
            ),
        )
        scene = manager.wait_ready(started.instance_id, timeout=30)
        assert scene.state.value == "running", scene.failure_reason
        instance = manager.get(scene.instance_id)
        robot_id = instance.robot_ids()[0]
        profile = instance.components.robots.robot_profile(robot_id)
        assert "base_link" in profile.capabilities.frames
        initial = instance.components.robots.robot_state(robot_id, scene.generation)
        assert len(initial.joints) == 25
        assert initial.base_pose.position[2] == pytest.approx(0.01, abs=1e-6)

        state_frame = instance.encoded_robot_state(robot_id)
        state_header, state_payload = EncodedFrame.unpack(state_frame.packet())
        streamed_state = json.loads(state_payload)
        assert state_header["stream"] == "robot_state"
        assert state_header["robot_id"] == robot_id == streamed_state["robot_id"]
        assert state_header["generation"] == streamed_state["generation"] == scene.generation
        assert state_header["observed_at"] == streamed_state["observed_at"]
        assert state_header["frame_id"] == streamed_state["base_pose"]["frame_id"]

        # SDK 使用 URDF 计算 base_link→left_gripper_link；Runtime 必须返回同一相对
        # 位姿。若 base_pose 丢失场景给 Robot root 的 z 偏移，这里会稳定相差 1cm。
        asset_root = Path(os.environ["MUJOCO_ASSET_ROOT"])
        urdf = asset_root / "robot/r1_pro_chassis/meshes/r1_pro_with_gripper.urdf"
        world_from_base = _pose_matrix(initial.base_pose)
        world_from_ee = _pose_matrix(initial.end_effectors["left"])
        actual_base_from_ee = np.linalg.inv(world_from_base) @ world_from_ee
        expected_base_from_ee = _urdf_fk(
            urdf,
            root_link="base_link",
            target_link="left_gripper_link",
            joints={name: value.position for name, value in initial.joints.items()},
        )
        np.testing.assert_allclose(
            actual_base_from_ee[:3, 3], expected_base_from_ee[:3, 3], atol=5e-4
        )
        np.testing.assert_allclose(
            actual_base_from_ee[:3, :3], expected_base_from_ee[:3, :3], atol=5e-4
        )

        rgb = instance.encoded_sensor_frame(robot_id, "camera.rgb")
        depth = instance.encoded_sensor_frame(robot_id, "camera.depth")
        assert rgb.payload.startswith(b"\xff\xd8")
        assert depth.payload.startswith(b"\x89PNG")
        contact = instance.encoded_sensor_frame(robot_id, "contact")
        contact_payload = json.loads(contact.payload)
        assert set(contact_payload) == {"active", "count", "contacts", "tools"}
        assert isinstance(contact_payload["contacts"], list)
        assert "data" not in contact_payload



        x, y = initial.base_pose.position[:2]
        base = RobotCommandRequest(
            command_id="native-base",
            scene_generation=scene.generation,
            type="base_trajectory",
            base_trajectory={
                "frame_id": "world",
                "points": [
                    {"time_from_start_seconds": 0.0, "positions": {"x": x, "y": y, "yaw": 0.0}},
                    {
                        "time_from_start_seconds": 0.2,
                        "positions": {"x": x + 0.05, "y": y, "yaw": 0.05},
                    },
                ],
            },
            timeout_seconds=2,
        )
        instance.submit_command(robot_id, base)
        assert _wait(instance, robot_id, base.command_id).status.value == "succeeded"

        state = instance.components.robots.robot_state(robot_id, scene.generation)
        current = state.joints["left_arm_joint1"].position
        joint = RobotCommandRequest(
            command_id="native-joint",
            scene_generation=scene.generation,
            type="joint_trajectory",
            joint_trajectory={
                "resources": ["left_arm"],
                "frame_id": "world",
                "points": [
                    {"time_from_start_seconds": 0.0, "positions": {"left_arm_joint1": current}},
                    {
                        "time_from_start_seconds": 0.2,
                        "positions": {"left_arm_joint1": current + 0.02},
                    },
                ],
            },
            timeout_seconds=2,
        )
        instance.submit_command(robot_id, joint)
        assert _wait(instance, robot_id, joint.command_id).status.value == "succeeded"

        timeout = RobotCommandRequest(
            command_id="native-timeout",
            scene_generation=scene.generation,
            type="base_trajectory",
            base_trajectory={
                "frame_id": "world",
                "points": [
                    {
                        "time_from_start_seconds": 0.0,
                        "positions": {"x": x + 0.05, "y": y, "yaw": 0.05},
                    },
                    {
                        "time_from_start_seconds": 10.0,
                        "positions": {"x": x + 2.0, "y": y, "yaw": 0.05},
                    },
                ],
            },
            timeout_seconds=0.02,
        )
        instance.submit_command(robot_id, timeout)
        assert _wait(instance, robot_id, timeout.command_id).status.value == "failed"
        assert instance.components.robots.robot_state(robot_id, scene.generation).in_hold

        reset = instance.reset()
        assert reset.generation == 2
        with pytest.raises(ConflictError):
            instance.submit_command(
                robot_id,
                RobotCommandRequest(
                    command_id="old-generation",
                    scene_generation=1,
                    type="gripper_command",
                    gripper_command={"gripper_id": "left", "position": 0.05},
                ),
            )
    finally:
        manager.shutdown()


@pytest.mark.native
def test_native_image_encoding_does_not_block_physics(monkeypatch):
    """图片编码故意阻塞时，物理线程仍应继续推进 sim_time。"""
    from plugin_mujoco.native import backend as native_backend_module

    manager = _manager()
    release_encoding = threading.Event()
    encoding_started = threading.Event()
    errors: list[BaseException] = []
    try:
        started = manager.start(
            "palletizing_depalletizing_001",
            SceneStartRequest(
                request_id="native-render-lock",
                layout="layout001",
                seed=2,
                render_backend="egl",
            ),
        )
        scene = manager.wait_ready(started.instance_id, timeout=30)
        instance = manager.get(scene.instance_id)
        robot_id = instance.robot_ids()[0]
        original_encode = native_backend_module.encode_rgb_jpeg

        def slow_encode(image, *, quality):
            encoding_started.set()
            if not release_encoding.wait(3):
                raise TimeoutError("测试未释放图片编码")
            return original_encode(image, quality=quality)

        monkeypatch.setattr(native_backend_module, "encode_rgb_jpeg", slow_encode)

        def render_one_frame() -> None:
            try:
                instance.components.sensors.encoded_sensor_frame(
                    robot_id,
                    "camera.rgb",
                    generation=scene.generation,
                    sequence=1,
                )
            except BaseException as exc:  # 线程异常必须回传到测试线程。
                errors.append(exc)

        worker = threading.Thread(target=render_one_frame, daemon=True)
        worker.start()
        assert encoding_started.wait(5), "相机帧未进入编码阶段"
        before = instance.view().sim_time
        time.sleep(0.15)
        after = instance.view().sim_time
        # 旧实现会在 JPEG 编码期间一直占用 Backend 锁，sim_time 完全不变。
        assert after - before >= instance.components.physics.timestep * 5
        release_encoding.set()
        worker.join(5)
        assert not worker.is_alive()
        assert errors == []
    finally:
        release_encoding.set()
        manager.shutdown()


def _pose_matrix(pose) -> np.ndarray:
    """把公共 xyzw Pose 转为齐次变换；测试中禁止依赖引擎内部 wxyz。"""
    x, y, z, w = pose.quaternion_xyzw
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = pose.position
    return result


def _urdf_fk(
    path: Path,
    *,
    root_link: str,
    target_link: str,
    joints: dict[str, float],
) -> np.ndarray:
    """按 URDF joint origin/axis 计算目标 Link 位姿，覆盖本回归所需的通用关节。"""
    document = ET.parse(path).getroot()
    by_child = {
        joint.find("child").attrib["link"]: joint
        for joint in document.findall("joint")
        if joint.find("child") is not None
    }
    chain: list[ET.Element] = []
    link = target_link
    while link != root_link:
        joint = by_child.get(link)
        if joint is None:
            raise AssertionError(f"URDF 中 {target_link} 不属于 {root_link} 子树")
        chain.append(joint)
        parent = joint.find("parent")
        assert parent is not None
        link = parent.attrib["link"]

    transform = np.eye(4)
    for joint in reversed(chain):
        origin = joint.find("origin")
        xyz = _numbers(origin.attrib.get("xyz") if origin is not None else None, (0, 0, 0))
        rpy = _numbers(origin.attrib.get("rpy") if origin is not None else None, (0, 0, 0))
        transform = transform @ _origin_matrix(xyz, rpy)
        joint_type = joint.attrib.get("type", "fixed")
        value = float(joints.get(joint.attrib.get("name", ""), 0.0))
        axis_node = joint.find("axis")
        axis = _numbers(axis_node.attrib.get("xyz") if axis_node is not None else None, (1, 0, 0))
        if joint_type in {"revolute", "continuous"}:
            transform = transform @ _axis_rotation(axis, value)
        elif joint_type == "prismatic":
            motion = np.eye(4)
            motion[:3, 3] = np.asarray(axis) * value
            transform = transform @ motion
    return transform


def _numbers(value: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(item) for item in value.split()) if value else default


def _origin_matrix(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = xyz
    return result


def _axis_rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    vector /= np.linalg.norm(vector)
    x, y, z = vector
    cosine, sine = math.cos(angle), math.sin(angle)
    cross = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    rotation = cosine * np.eye(3) + (1 - cosine) * np.outer(vector, vector) + sine * cross
    result = np.eye(4)
    result[:3, :3] = rotation
    return result


@pytest.mark.native
def test_r1_browser_glb_preserves_compiled_mujoco_geom_colors() -> None:
    """浏览器 GLB 必须继承 MuJoCo 编译后的材质，不能用前端专用颜色替换真实资产。"""
    import mujoco

    raw_root = os.getenv("MUJOCO_ASSET_ROOT")
    if not raw_root:
        pytest.skip("MUJOCO_ASSET_ROOT 未设置")
    asset_root = Path(raw_root)
    content = RuntimeVisualAssetStore(asset_root).content("r1_pro_chassis", "3")
    json_length = struct.unpack_from("<I", content, 12)[0]
    document = json.loads(content[20 : 20 + json_length])
    actual = {
        tuple(
            round(float(value), 6)
            for value in material["pbrMetallicRoughness"]["baseColorFactor"]
        )
        for material in document["materials"]
    }

    model_root = asset_root / "robot/r1_pro_chassis"
    model_path = model_root / "config/r1_pro_chassis.xml"
    root = ET.fromstring(model_path.read_text(encoding="utf-8"))
    asset = root.find("asset")
    assert asset is not None
    if not any(item.get("name") == "visualgeom" for item in asset.findall("material")):
        ET.SubElement(asset, "material", name="visualgeom", rgba="0.5 0.9 0.2 1")
    files: dict[str, bytes] = {}
    for mesh in asset.findall("mesh"):
        source = (model_path.parent / mesh.get("file", "")).resolve()
        assert source.is_relative_to(model_root) and source.is_file()
        mesh.set("file", source.name)
        files[source.name] = source.read_bytes()
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), files)
    expected = {
        tuple(round(float(value), 6) for value in model.geom_rgba[geom_id])
        for geom_id in range(model.ngeom)
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
    }

    assert actual
    assert expected.issubset(actual)
    assert (0.5, 0.9, 0.2, 1.0) in actual
