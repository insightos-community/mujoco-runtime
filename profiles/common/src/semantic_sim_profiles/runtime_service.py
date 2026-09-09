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

"""robosuite / LIBERO 隔离 Runtime 的生命周期和低层 Robot 执行。

物理环境只在 worker 线程中创建、步进、重置和释放。HTTP 线程只能修改本模块
保存的领域状态或向 worker 投递调用，不能直接访问 MuJoCo / robosuite 对象。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
from semantic_mujoco_visuals import MujocoVisualExporter

from semantic_sim_profiles.libero import LiberoAdapter
from semantic_sim_profiles.robosuite import FRANKA_JOINT_NAMES, RobosuiteAdapter


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProfileRuntimeError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 409):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class _WorkerCall:
    callback: Callable[[Any], Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None


class ProfileRuntimeInstance:
    """一个只读 benchmark 场景实例；一个进程同时最多存在一个活动实例。"""

    control_period_s = 0.05
    robot_id = "franka-0"
    # 轨迹时长只表示目标何时应被下发，不能证明 Robot 已到达。完成判定要求
    # worker 连续观察到实际状态进入容差；settle 窗口耗尽则失败并进入 hold。
    joint_position_tolerance_rad = 0.01
    gripper_opening_tolerance_m = 0.004
    settled_observation_samples = 3
    joint_settle_timeout_s = 2.0
    gripper_settle_timeout_s = 8.0

    def __init__(
        self,
        *,
        profile_id: str,
        scene_key: str,
        layout: str,
        request: Dict[str, Any],
        adapter_factory: Callable[[], Any],
    ) -> None:
        now = utc_iso()
        self.profile_id = profile_id
        self.scene_key = scene_key
        self.layout = layout
        self.request = dict(request)
        self.instance_id = str(uuid.uuid4())
        self.generation = 1
        self.state = "starting"
        self.failure_reason: Optional[str] = None
        self.created_at = now
        self.updated_at = now
        self.sim_time = 0.0
        self.step_count = 0
        self._adapter_factory = adapter_factory
        self._condition = threading.Condition(threading.RLock())
        self._calls: Deque[_WorkerCall] = deque()
        self._stop_requested = False
        self._adapter: Any = None
        self._observation: Dict[str, Any] = {}
        self._joint_positions: Dict[str, float] = {}
        self._base_pose: Optional[Dict[str, Any]] = None
        self._gripper_opening_m: Optional[float] = None
        self._contact_state: Dict[str, Any] = {
            "active": False,
            "count": 0,
            "contacts": [],
            "holding": False,
        }
        # reward、结束原因和 success 属于原生环境的评测证据。它们与 Robot Skill
        # 的业务结果不同，但必须跟随当前 generation 保存，供测试和 Studio 回看。
        self._last_reward = 0.0
        self._terminated = False
        self._commands: Dict[str, Dict[str, Any]] = {}
        self._command_fingerprints: Dict[str, str] = {}
        self._active_command_id: Optional[str] = None
        self._active_started_sim_time = 0.0
        self._gripper_action = 0.0
        self._sensor_sequences: Dict[str, int] = {}
        self._visual_exporter: Any = None
        self._visual_content = b""
        self._visual_revision = ""
        self._visual_node_order: Tuple[str, ...] = ()
        self._visual_cameras: Tuple[Any, ...] = ()
        self._pose_sequence = 0
        self._pose_payload = b""
        self._thread = threading.Thread(
            target=self._run,
            name="semantic-profile-" + self.instance_id,
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def wait_ready(self, timeout: float = 30.0) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self.state == "starting" and time.monotonic() < deadline:
                self._condition.wait(timeout=0.1)
            if self.state == "starting":
                raise ProfileRuntimeError("隔离 Profile 场景启动超时")
            return self.view()

    def view(self) -> Dict[str, Any]:
        with self._condition:
            return {
                "instance_id": self.instance_id,
                "scene_key": self.scene_key,
                "layout": self.layout,
                "seed": int(self.request.get("seed", 0)),
                "headless": bool(self.request.get("headless", True)),
                "render_backend": str(self.request.get("render_backend", "egl")),
                "generation": self.generation,
                "state": self.state,
                "sim_time": self.sim_time,
                "step_count": self.step_count,
                "request_id": self.request["request_id"],
                "runtime_profile_id": self.profile_id,
                "runtime_bundle_id": self.request.get("runtime_bundle_id"),
                "progress": 1.0 if self.state not in {"starting", "failed"} else 0.0,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "failure_reason": self.failure_reason,
            }

    def pause(self) -> Dict[str, Any]:
        with self._condition:
            if self.state == "paused":
                return self.view()
            self._require_state("running")
            self._cancel_active("场景暂停")
            self.state = "paused"
            self.updated_at = utc_iso()
            self._condition.notify_all()

        # API 线程不能直接接触仿真对象。同步屏障会在已经进入 adapter.step 的
        # 物理周期结束后由 worker 执行，因此 pause 返回时 sim_time 已稳定。
        self._worker_call(lambda _adapter: None)
        return self.view()

    def resume(self) -> Dict[str, Any]:
        with self._condition:
            if self.state == "running":
                return self.view()
            self._require_state("paused")
            self.state = "running"
            self.updated_at = utc_iso()
            self._condition.notify_all()
            return self.view()

    def step_once(self, steps: int) -> Dict[str, Any]:
        if steps < 1 or steps > 1000:
            raise ProfileRuntimeError("steps 必须在 1 到 1000 之间", status_code=422)
        with self._condition:
            self._require_state("paused")
            if self._active_command_id:
                raise ProfileRuntimeError("存在活动 Robot 命令时不能单步")

        def advance(adapter: Any) -> None:
            for _ in range(steps):
                self._advance_adapter(adapter, self._hold_action(adapter))

        self._worker_call(advance)
        return self.view()

    def reset(self) -> Dict[str, Any]:
        with self._condition:
            self._require_state("running", "paused")
            if self._active_command_id:
                raise ProfileRuntimeError("reset 前必须停止活动 Robot 命令")
            self.state = "resetting"
            self.updated_at = utc_iso()

        def do_reset(adapter: Any) -> None:
            observation = adapter.reset(int(self.request.get("seed", 0)))
            base_pose = adapter.base_pose()
            gripper_opening = adapter.gripper_opening()
            with self._condition:
                self._observation = _copy_observation(observation)
                self._base_pose = _copy_base_pose(base_pose)
                self._gripper_opening_m = _validated_gripper_opening(gripper_opening)
                self._joint_positions = adapter.joint_positions()
                self._contact_state = dict(adapter.contact_state())
                self._last_reward = 0.0
                self._terminated = False
                self.generation += 1
                self.sim_time = 0.0
                self.step_count = 0
                self._sensor_sequences.clear()
                self._pose_sequence = 0
                self._capture_visual_pose_locked()
                self.state = "running"
                self.updated_at = utc_iso()

        self._worker_call(do_reset, timeout=30.0)
        return self.view()

    def stop(self, timeout: float = 10.0) -> Dict[str, Any]:
        with self._condition:
            if self.state == "stopped":
                return self.view()
            if self.state not in {"starting", "running", "paused", "failed"}:
                raise ProfileRuntimeError("当前场景状态不能停止: " + self.state)
            self._cancel_active("场景停止")
            # worker 已因加载或物理异常退出时，不再等待一个不会执行的 finally。
            if self.state == "failed" and not self._thread.is_alive():
                self.state = "stopped"
                self.updated_at = utc_iso()
                return self.view()
            self.state = "stopping"
            self._stop_requested = True
            self.updated_at = utc_iso()
            self._condition.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise ProfileRuntimeError("隔离 Profile 线程未能停止")
        return self.view()

    def submit_command(self, robot_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        self._check_robot(robot_id)
        command_id = str(request.get("command_id", "")).strip()
        if not command_id:
            raise ProfileRuntimeError("command_id 不能为空", status_code=422)
        if int(request.get("scene_generation", 0)) != self.generation:
            raise ProfileRuntimeError("命令 generation 已失效")
        command_type = request.get("type")
        if command_type not in {"joint_trajectory", "gripper_command"}:
            raise ProfileRuntimeError("Franka Profile 只接受关节轨迹和夹爪命令")
        fingerprint = _fingerprint(request)
        with self._condition:
            self._require_state("running")
            previous = self._commands.get(command_id)
            if previous is not None:
                if self._command_fingerprints[command_id] != fingerprint:
                    raise ProfileRuntimeError("相同 command_id 的输入不同")
                return dict(previous)
            if self._active_command_id:
                raise ProfileRuntimeError("Franka 运动资源正在执行其他命令")
            _validate_command(request)
            now = utc_iso()
            command = {
                **request,
                "robot_id": robot_id,
                "status": "accepted",
                "progress": 0.0,
                "message": "命令已接受",
                "accepted_at": now,
                "updated_at": now,
            }
            self._commands[command_id] = command
            self._command_fingerprints[command_id] = fingerprint
            self._active_command_id = command_id
            self._active_started_sim_time = self.sim_time
            self._condition.notify_all()
            return dict(command)

    def command(self, robot_id: str, command_id: str) -> Dict[str, Any]:
        self._check_robot(robot_id)
        with self._condition:
            command = self._commands.get(command_id)
            if command is None:
                raise ProfileRuntimeError("命令不存在: " + command_id, status_code=404)
            return dict(command)

    def stop_command(self, robot_id: str, command_id: str) -> Dict[str, Any]:
        self._check_robot(robot_id)
        with self._condition:
            command = self._commands.get(command_id)
            if command is None:
                raise ProfileRuntimeError("命令不存在: " + command_id, status_code=404)
            if command["status"] in {"succeeded", "failed", "cancelled", "unknown"}:
                return dict(command)
            self._finish_command(command, "cancelled", "命令已停止，Robot 保持当前位置")
            result = dict(command)

        # 等待可能已经进入 adapter.step 的最后一个周期结束。barrier 返回后，
        # worker 会先看到 active command 已清除，再生成保持当前位置的 action。
        self._worker_call(lambda _adapter: None)
        return result

    def hold(self, robot_id: str, generation: int) -> Dict[str, Any]:
        self._check_robot(robot_id)
        if generation != self.generation:
            raise ProfileRuntimeError("hold generation 已失效")
        with self._condition:
            self._cancel_active("Robot 进入 hold")
            return {
                "command_id": "hold-%s-%s" % (robot_id, generation),
                "robot_id": robot_id,
                "scene_generation": generation,
                "type": "hold",
                "status": "succeeded",
                "progress": 1.0,
                "message": "Robot 已进入 hold",
                "updated_at": utc_iso(),
            }

    def robot_profile(self, robot_id: str) -> Dict[str, Any]:
        self._check_robot(robot_id)
        return {
            "robot_id": self.robot_id,
            "model": "franka_panda",
            "kind": "manipulator",
            "coordinate_frame": "world",
            "sdk_package": "robot-sdk-franka",
            "backend_profile": self.profile_id,
            "joint_names": list(FRANKA_JOINT_NAMES),
            "end_effectors": ["hand"],
            "grippers": ["hand"],
            "capabilities": {
                "commands": ["joint_trajectory", "gripper_command"],
                "sensors": ["rgb", "depth", "contact", "robot_state"],
                "frames": ["world", "panda_link0", "panda_hand"],
            },
        }

    def robot_state(self, robot_id: str) -> Dict[str, Any]:
        self._check_robot(robot_id)
        with self._condition:
            eef_position = _vector(self._observation.get("robot0_eef_pos"), 3, 0.0)
            eef_quaternion = _unit_quaternion(
                _vector(self._observation.get("robot0_eef_quat"), 4, 0.0)
            )
            velocities = _vector(self._observation.get("robot0_joint_vel"), 7, 0.0)
            if self._base_pose is None or self._gripper_opening_m is None:
                raise ProfileRuntimeError("Franka Robot 状态尚未就绪", status_code=503)
            # 返回缓存副本，API 线程绝不能为了取状态直接访问 robosuite / MuJoCo。
            base_pose = _copy_base_pose(self._base_pose)
            return {
                "robot_id": self.robot_id,
                "generation": self.generation,
                "observed_at": self.updated_at,
                "base_pose": base_pose,
                "joints": {
                    name: {
                        "position": float(self._joint_positions.get(name, 0.0)),
                        "velocity": float(velocities[index]),
                    }
                    for index, name in enumerate(FRANKA_JOINT_NAMES)
                },
                "end_effectors": {
                    "hand": {
                        "position": eef_position,
                        "quaternion_xyzw": eef_quaternion,
                        "frame_id": "world",
                    }
                },
                "grippers": {"hand": self._gripper_opening_m},
                "holding_object": None,
                "in_hold": self._active_command_id is None,
            }

    def sensor_descriptors(self, robot_id: str) -> List[Dict[str, Any]]:
        self._check_robot(robot_id)
        result = []
        for sensor_id, observation_key, kind in _sensor_bindings(self._observation):
            value = np.asarray(self._observation[observation_key])
            height, width = int(value.shape[0]), int(value.shape[1])
            result.append(
                {
                    "sensor_id": sensor_id,
                    "robot_id": self.robot_id,
                    "kind": kind,
                    "frame_id": sensor_id.replace("_rgb", "").replace("_depth", ""),
                    "encoding": "jpeg" if kind == "rgb" else "float32-le",
                    "width": width,
                    "height": height,
                    "fps": 20.0,
                }
            )
        result.append(
            {
                "sensor_id": "robot0_contact",
                "robot_id": self.robot_id,
                "kind": "contact",
                "frame_id": "panda_hand",
                "encoding": "json",
                "width": None,
                "height": None,
                "fps": 20.0,
            }
        )
        return result

    def sensor_payload(self, robot_id: str, sensor_id: str) -> Tuple[str, Any, int]:
        self._check_robot(robot_id)
        with self._condition:
            if sensor_id == "robot0_contact":
                sequence = self._sensor_sequences.get(sensor_id, 0) + 1
                self._sensor_sequences[sensor_id] = sequence
                return "contact", dict(self._contact_state), sequence
            binding = next(
                (item for item in _sensor_bindings(self._observation) if item[0] == sensor_id),
                None,
            )
            if binding is None:
                raise ProfileRuntimeError("传感器不存在: " + sensor_id, status_code=404)
            self._sensor_sequences[sensor_id] = self._sensor_sequences.get(sensor_id, 0) + 1
            return (
                binding[2],
                np.array(self._observation[binding[1]], copy=True),
                self._sensor_sequences[sensor_id],
            )

    def snapshot(self) -> Dict[str, Any]:
        with self._condition:
            objects = _scene_objects(self._observation)
        return {
            "scene_key": self.scene_key,
            "instance_id": self.instance_id,
            "generation": self.generation,
            "coordinate_frame": "world",
            "robots": [self.robot_state(self.robot_id)],
            "objects": objects,
            "regions": [],
            "sensors": self.sensor_descriptors(self.robot_id),
            "observed_at": utc_iso(),
        }

    def evaluation(self) -> Dict[str, Any]:
        """在物理 worker 上读取原生评测器，避免 API 线程直接访问环境对象。"""

        native = self._worker_call(
            lambda adapter: {
                "success": bool(adapter.success()),
                "metrics": dict(adapter.native_metrics()),
                "language": str(getattr(adapter, "language", "")),
            }
        )
        with self._condition:
            return {
                "scene_key": self.scene_key,
                "instance_id": self.instance_id,
                "generation": self.generation,
                "runtime_profile_id": self.profile_id,
                "reward": self._last_reward,
                "success": native["success"],
                "terminated": self._terminated,
                "language": native["language"],
                "metrics": native["metrics"],
                "observed_at": utc_iso(),
            }

    def viewer_scene(self) -> Dict[str, Any]:
        with self._condition:
            if not self._visual_revision:
                raise ProfileRuntimeError("Viewer Scene 尚未就绪", status_code=503)
            return {
                "generation": self.generation,
                "scene_revision": self._visual_revision,
                "coordinate_frame": "world",
                "content_url": (f"/api/v1/scene-instances/{self.instance_id}/viewer-scene/content"),
                "pose_stream_url": (f"/api/v1/scene-instances/{self.instance_id}/pose-stream"),
                "dynamic_node_order": list(self._visual_node_order),
                "cameras": [
                    {
                        "camera_id": item.camera_id,
                        "name": item.name,
                        "position": list(item.position),
                        "quaternion_xyzw": list(item.quaternion_xyzw),
                        "fovy": item.fovy,
                    }
                    for item in self._visual_cameras
                ],
            }

    def viewer_scene_content(self) -> Tuple[bytes, str]:
        with self._condition:
            if not self._visual_content:
                raise ProfileRuntimeError("Viewer Scene 尚未就绪", status_code=503)
            return self._visual_content, self._visual_revision

    def scene_pose_frame(
        self, after_sequence: int = 0, timeout: float = 2.0
    ) -> Tuple[Dict[str, Any], bytes]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._pose_sequence <= after_sequence and not self._stop_requested:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._pose_sequence == 0:
                raise ProfileRuntimeError("等待 Scene Pose 首帧超时", status_code=503)
            return (
                {
                    "stream": "scene_pose",
                    "sequence": self._pose_sequence,
                    "generation": self.generation,
                    "scene_revision": self._visual_revision,
                    "sim_time": self.sim_time,
                    "node_count": len(self._visual_node_order),
                    "coordinate_frame": "world",
                    "encoding": "float32-le-xyz-xyzw",
                },
                self._pose_payload,
            )

    def _run(self) -> None:
        adapter = None
        try:
            adapter = self._adapter_factory()
            observation = adapter.reset(int(self.request.get("seed", 0)))
            base_pose = adapter.base_pose()
            gripper_opening = adapter.gripper_opening()
            visual_exporter, visual_content, visual_order, visual_cameras = self._prepare_visuals(
                adapter, observation
            )
            with self._condition:
                self._adapter = adapter
                self._observation = _copy_observation(observation)
                self._base_pose = _copy_base_pose(base_pose)
                self._gripper_opening_m = _validated_gripper_opening(gripper_opening)
                self._joint_positions = adapter.joint_positions()
                self._contact_state = dict(adapter.contact_state())
                self._visual_exporter = visual_exporter
                self._visual_content = visual_content
                self._visual_revision = hashlib.sha256(visual_content).hexdigest()[:20]
                self._visual_node_order = visual_order
                self._visual_cameras = visual_cameras
                self._capture_visual_pose_locked()
                self.state = "running"
                self.updated_at = utc_iso()
                self._condition.notify_all()
            deadline = time.monotonic()
            while True:
                call = None
                with self._condition:
                    if self._calls:
                        call = self._calls.popleft()
                    elif self._stop_requested:
                        break
                    elif self.state == "paused":
                        self._condition.wait(timeout=0.5)
                        continue
                if call is not None:
                    try:
                        call.result = call.callback(adapter)
                    except BaseException as error:
                        call.error = error
                    finally:
                        call.done.set()
                    continue
                action = self._next_action(adapter)
                self._advance_adapter(adapter, action)
                deadline += self.control_period_s
                time.sleep(max(0.0, deadline - time.monotonic()))
        except BaseException as error:
            with self._condition:
                self.failure_reason = str(error)
                self.state = "failed"
                self.updated_at = utc_iso()
                while self._calls:
                    call = self._calls.popleft()
                    call.error = error
                    call.done.set()
                self._condition.notify_all()
        finally:
            if adapter is not None:
                try:
                    adapter.close()
                except Exception:
                    pass
            with self._condition:
                if self.state == "stopping":
                    self.state = "stopped"
                    self.updated_at = utc_iso()
                self._condition.notify_all()

    def _prepare_visuals(
        self, adapter: Any, observation: Dict[str, Any]
    ) -> Tuple[Any, bytes, Tuple[str, ...], Tuple[Any, ...]]:
        model, data = adapter.visual_model_data()
        object_source_ids = [item["source_id"] for item in _scene_objects(observation)]
        exporter = MujocoVisualExporter(
            model,
            data,
            source_for_body=lambda body_id: adapter.visual_source_for_body(
                body_id, object_source_ids
            ),
        )
        exported = exporter.export()
        return (
            exporter,
            exported.content,
            exported.dynamic_node_order,
            exported.cameras,
        )

    def _capture_visual_pose_locked(self) -> None:
        """只允许 worker 在已持有状态锁时读取 MjData 并发布不可变 float32。"""
        if self._visual_exporter is None:
            return
        poses = np.asarray(self._visual_exporter.poses(), dtype="<f4")
        self._pose_sequence += 1
        self._pose_payload = poses.tobytes(order="C")
        self._condition.notify_all()

    def _next_action(self, adapter: Any) -> Any:
        with self._condition:
            command = (
                self._commands.get(self._active_command_id)
                if self._active_command_id is not None
                else None
            )
            if command is None:
                return self._hold_action(adapter)
            command["status"] = "running"
            elapsed = self.sim_time - self._active_started_sim_time
            if command["type"] == "gripper_command":
                payload = command["gripper_command"]
                if self._gripper_opening_m is None:
                    raise ProfileRuntimeError("Franka 夹爪状态尚未就绪")
                target_opening = float(payload["position"])
                self._gripper_action = _gripper_control_direction(
                    self._gripper_opening_m,
                    target_opening,
                    self.gripper_opening_tolerance_m,
                )
                command["position_error_m"] = abs(self._gripper_opening_m - target_opening)
                command["updated_at"] = utc_iso()
                target = dict(self._joint_positions)
                return adapter.joint_position_action(target, gripper_action=self._gripper_action)
            points = command["joint_trajectory"]["points"]
            target, progress, _final_target_sent = _interpolate_joint_trajectory(points, elapsed)
            command["progress"] = min(progress, 0.99)
            command["updated_at"] = utc_iso()
            action = adapter.joint_position_action(target, gripper_action=self._gripper_action)
            return action

    def _advance_adapter(self, adapter: Any, action: Any) -> None:
        observation, reward, done, _info = adapter.step(action)
        base_pose = adapter.base_pose()
        gripper_opening = adapter.gripper_opening()
        joint_positions = adapter.joint_positions()
        contact_state = adapter.contact_state()
        with self._condition:
            self._observation = _copy_observation(observation)
            self._base_pose = _copy_base_pose(base_pose)
            self._gripper_opening_m = _validated_gripper_opening(gripper_opening)
            self._joint_positions = dict(joint_positions)
            self._contact_state = dict(contact_state)
            self._last_reward = float(reward)
            self._terminated = bool(done)
            self.sim_time += self.control_period_s
            self.step_count += 1
            self.updated_at = utc_iso()
            self._observe_active_command()
            self._capture_visual_pose_locked()

    def _hold_action(self, adapter: Any) -> Any:
        """保持当前实际关节和夹爪控制目标，不生成新的运动。"""

        return adapter.joint_position_action(
            dict(self._joint_positions),
            gripper_action=0.0,
        )

    def _observe_active_command(self) -> None:
        """用刚完成的物理周期观测判断命令是否真正结束。

        本方法只从 worker 已写入的缓存读取状态。轨迹时间到达只意味着最终目标
        已经发送；只有实际位置连续进入容差，命令才能变成 succeeded。
        """

        if self._active_command_id is None:
            return
        command = self._commands[self._active_command_id]
        elapsed = self.sim_time - self._active_started_sim_time
        if command["type"] == "joint_trajectory":
            points = command["joint_trajectory"]["points"]
            duration = float(points[-1]["time_from_start_seconds"])
            target = points[-1]["positions"]
            error = max(
                abs(float(self._joint_positions[name]) - float(target[name]))
                for name in FRANKA_JOINT_NAMES
            )
            command["position_error_rad"] = error
            eligible = elapsed + 1e-9 >= duration
            tolerance = self.joint_position_tolerance_rad
            settle_deadline = duration + self.joint_settle_timeout_s
            success_message = "实际关节位置已进入目标容差"
        else:
            if self._gripper_opening_m is None:
                raise ProfileRuntimeError("Franka 夹爪状态尚未就绪")
            target_opening = float(command["gripper_command"]["position"])
            error = abs(self._gripper_opening_m - target_opening)
            command["position_error_m"] = error
            eligible = True
            tolerance = self.gripper_opening_tolerance_m
            settle_deadline = self.gripper_settle_timeout_s
            success_message = "实际夹爪开度已进入目标容差"

        settled = int(command.get("_settled_observations", 0))
        command["_settled_observations"] = settled + 1 if eligible and error <= tolerance else 0
        command["updated_at"] = utc_iso()
        if command["_settled_observations"] >= self.settled_observation_samples:
            self._gripper_action = 0.0
            self._finish_command(command, "succeeded", success_message)
            return

        # progress=1 只保留给真实完成。轨迹已经发送完但仍在收敛时最多显示 0.99。
        command["progress"] = min(float(command.get("progress", 0.0)), 0.99)
        timeout = float(command.get("timeout_seconds", 30.0))
        if elapsed + 1e-9 >= timeout:
            self._gripper_action = 0.0
            command["failure_reason"] = "timeout"
            self._finish_command(command, "failed", "命令超时，Robot 已保持当前位置")
            return
        if elapsed + 1e-9 >= settle_deadline:
            self._gripper_action = 0.0
            command["failure_reason"] = "target_not_reached"
            self._finish_command(
                command,
                "failed",
                "目标在收敛窗口内未进入容差，Robot 已保持当前位置",
            )

    def _worker_call(self, callback: Callable[[Any], Any], timeout: float = 10.0) -> Any:
        call = _WorkerCall(callback)
        with self._condition:
            if not self._thread.is_alive():
                raise ProfileRuntimeError("隔离 Profile worker 不可用")
            self._calls.append(call)
            self._condition.notify_all()
        if not call.done.wait(timeout):
            raise ProfileRuntimeError("隔离 Profile worker 未在限定时间内响应")
        if call.error is not None:
            raise call.error
        return call.result

    def _finish_command(self, command: Dict[str, Any], status: str, message: str) -> None:
        command["status"] = status
        command["progress"] = 1.0 if status == "succeeded" else command.get("progress", 0.0)
        command["message"] = message
        command["updated_at"] = utc_iso()
        command["ended_at"] = command["updated_at"]
        if self._active_command_id == command["command_id"]:
            self._active_command_id = None

        # Panda GRIP 输入是方向量；命令终止后必须归零，否则适配器会继续积分，
        # 即使上层状态已经显示 cancelled/failed，夹爪仍会继续运动。
        self._gripper_action = 0.0

    def _cancel_active(self, reason: str) -> None:
        if self._active_command_id:
            command = self._commands[self._active_command_id]
            self._finish_command(command, "cancelled", reason)

    def _require_state(self, *states: str) -> None:
        if self.state not in states:
            raise ProfileRuntimeError(
                "场景状态 %s 不允许该操作，期望 %s" % (self.state, list(states))
            )

    def _check_robot(self, robot_id: str) -> None:
        if robot_id != self.robot_id:
            raise ProfileRuntimeError("Robot 不存在: " + robot_id, status_code=404)


class ProfileRuntimeService:
    """管理隔离 Runtime 进程中的唯一活动场景。"""

    def __init__(self, profile_id: str) -> None:
        if profile_id not in {"robosuite-1.5", "libero-robosuite-1.4"}:
            raise ValueError("不支持的隔离 Runtime Profile: " + profile_id)
        self.profile_id = profile_id
        self.runtime_id = "plugin-mujoco-" + profile_id
        self._lock = threading.RLock()
        self._instance: Optional[ProfileRuntimeInstance] = None
        self._requests: Dict[str, Tuple[str, str]] = {}

    def runtime_info(self) -> Dict[str, Any]:
        with self._lock:
            active = self._instance
            active_id = active.instance_id if active and active.state != "stopped" else None
            runtime_state = (
                "failed"
                if active and active.state == "failed"
                else ("busy" if active_id else "ready")
            )
            return {
                "runtime_id": self.runtime_id,
                "runtime_profile_id": self.profile_id,
                "state": runtime_state,
                "engine": "mujoco",
                "version": "0.4.0-dev",
                "api_version": "v1",
                "active_instance_id": active_id,
                "capabilities": _profile_capabilities(self.profile_id),
                "asset_root": "",
            }

    def scenes(self) -> List[Dict[str, Any]]:
        if self.profile_id == "robosuite-1.5":
            return [
                _scene_descriptor(name, "robosuite", self.profile_id, ["default"])
                for name in ("Lift", "Stack")
            ]
        suite = os.getenv("SEMANTIC_LIBERO_SUITE", "libero_spatial")
        task_id = int(os.getenv("SEMANTIC_LIBERO_TASK_ID", "0"))
        return [
            _scene_descriptor(
                "%s:%s" % (suite, task_id),
                "libero",
                self.profile_id,
                ["init-0"],
            )
        ]

    def start_scene(self, scene_key: str, request: Dict[str, Any]) -> ProfileRuntimeInstance:
        request_id = str(request.get("request_id", "")).strip()
        if not request_id:
            raise ProfileRuntimeError("request_id 不能为空", status_code=422)
        profile = request.get("runtime_profile_id") or self.profile_id
        if profile != self.profile_id:
            raise ProfileRuntimeError("请求的 Runtime Profile 与当前进程不一致")
        layout = str(request.get("layout") or "default")
        fingerprint = _fingerprint({"scene_key": scene_key, **request})
        with self._lock:
            previous = self._requests.get(request_id)
            if previous is not None:
                previous_fingerprint, instance_id = previous
                if previous_fingerprint != fingerprint:
                    raise ProfileRuntimeError("相同 request_id 的启动输入不同")
                if self._instance and self._instance.instance_id == instance_id:
                    return self._instance
            if self._instance and self._instance.state != "stopped":
                raise ProfileRuntimeError("当前 Runtime 已有活动场景")
            factory = _adapter_factory(self.profile_id, scene_key, layout, request)
            instance = ProfileRuntimeInstance(
                profile_id=self.profile_id,
                scene_key=scene_key,
                layout=layout,
                request=request,
                adapter_factory=factory,
            )
            self._instance = instance
            self._requests[request_id] = (fingerprint, instance.instance_id)
            instance.start()
            return instance

    def instance(self, instance_id: str) -> ProfileRuntimeInstance:
        with self._lock:
            if self._instance is None or self._instance.instance_id != instance_id:
                raise ProfileRuntimeError("场景实例不存在: " + instance_id, status_code=404)
            return self._instance

    def active(self) -> ProfileRuntimeInstance:
        with self._lock:
            if self._instance is None or self._instance.state in {"stopped", "failed"}:
                raise ProfileRuntimeError("当前没有活动场景", status_code=404)
            return self._instance

    def shutdown(self) -> None:
        with self._lock:
            instance = self._instance
        if instance and instance.state != "stopped":
            instance.stop()


def _adapter_factory(
    profile_id: str,
    scene_key: str,
    layout: str,
    request: Dict[str, Any],
) -> Callable[[], Any]:
    seed = int(request.get("seed", 0))
    if profile_id == "robosuite-1.5":
        if scene_key not in {"Lift", "Stack"} or layout != "default":
            raise ProfileRuntimeError("robosuite 场景或布局不存在", status_code=404)
        return lambda: RobosuiteAdapter(
            scene_key,
            camera_names=("agentview", "robot0_eye_in_hand"),
            width=640,
            height=480,
            horizon=1000000,
        )

    parts = scene_key.split(":")
    if len(parts) != 2 or not parts[1].isdigit():
        raise ProfileRuntimeError("LIBERO scene_key 必须为 suite:task_id", status_code=422)
    if not layout.startswith("init-") or not layout[5:].isdigit():
        raise ProfileRuntimeError("LIBERO layout 必须为 init-N", status_code=422)
    source_root_text = os.getenv("SEMANTIC_LIBERO_ROOT", "").strip()
    if not source_root_text:
        raise ProfileRuntimeError("SEMANTIC_LIBERO_ROOT 未配置")
    source_root = Path(source_root_text).expanduser()
    if not source_root.is_dir():
        raise ProfileRuntimeError("SEMANTIC_LIBERO_ROOT 未指向固定 LIBERO 源码")
    config_root = Path(os.getenv("SEMANTIC_LIBERO_CONFIG_ROOT", ".output/runtime-libero-config"))
    return lambda: LiberoAdapter(
        source_root=source_root,
        config_root=config_root,
        suite_name=parts[0],
        task_id=int(parts[1]),
        init_state_id=int(layout[5:]),
        seed=seed,
        camera_names=("agentview", "robot0_eye_in_hand"),
        width=640,
        height=480,
        horizon=1000000,
    )


def _profile_capabilities(profile_id: str) -> Dict[str, Any]:
    return {
        "editable_scene": False,
        "native_evaluator": True,
        "viewer": True,
        "viewer_camera_modes": ["fixed"],
        "scene_step": True,
        "scene_reset": True,
        "robot_models": ["franka_panda"],
        "sensor_kinds": ["rgb", "depth", "contact", "robot_state"],
    }


def _scene_descriptor(
    scene_key: str, scene_kind: str, profile_id: str, layouts: List[str]
) -> Dict[str, Any]:
    return {
        "scene_key": scene_key,
        "name": scene_key,
        "scene_kind": scene_kind,
        "layouts": layouts,
        "robot_models": ["franka_panda"],
        "compatible_runtime_profiles": [profile_id],
        "read_only": True,
    }


def _validate_command(request: Dict[str, Any]) -> None:
    timeout = float(request.get("timeout_seconds", 30.0))
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ProfileRuntimeError("Robot 命令 timeout_seconds 必须大于零", status_code=422)
    command_type = request.get("type")
    payload_key = command_type
    present = [
        key
        for key in ("joint_trajectory", "base_trajectory", "gripper_command")
        if request.get(key) is not None
    ]
    if present != [payload_key]:
        raise ProfileRuntimeError("Robot 命令必须且只能携带同名负载", status_code=422)
    if command_type == "joint_trajectory":
        trajectory = request["joint_trajectory"]
        resources = trajectory.get("resources") or []
        if resources != ["joints:arm"]:
            raise ProfileRuntimeError(
                "Franka 关节轨迹 resources 必须为 joints:arm", status_code=422
            )
        points = trajectory.get("points") or []
        if not points or float(points[0].get("time_from_start_seconds", -1)) != 0.0:
            raise ProfileRuntimeError("关节轨迹必须从 0 秒开始", status_code=422)
        previous = -1.0
        for point in points:
            timestamp = float(point.get("time_from_start_seconds", -1))
            positions = point.get("positions") or {}
            if timestamp <= previous or set(positions) != set(FRANKA_JOINT_NAMES):
                raise ProfileRuntimeError("关节轨迹时间或关节集合无效", status_code=422)
            previous = timestamp
    else:
        payload = request["gripper_command"]
        if payload.get("gripper_id") != "hand":
            raise ProfileRuntimeError("Franka gripper_id 必须为 hand", status_code=422)
        opening = float(payload.get("position", -1))
        if opening < 0.0 or opening > 0.08:
            raise ProfileRuntimeError("Franka 夹爪开度必须在 0 到 0.08 米", status_code=422)


def _interpolate_joint_trajectory(
    points: List[Dict[str, Any]], elapsed: float
) -> Tuple[Dict[str, float], float, bool]:
    last_time = float(points[-1]["time_from_start_seconds"])
    if elapsed >= last_time:
        return dict(points[-1]["positions"]), 1.0, True
    for index in range(1, len(points)):
        current_time = float(points[index]["time_from_start_seconds"])
        if elapsed <= current_time:
            previous = points[index - 1]
            previous_time = float(previous["time_from_start_seconds"])
            ratio = (elapsed - previous_time) / max(current_time - previous_time, 1e-9)
            target = {
                name: float(previous["positions"][name])
                + (float(points[index]["positions"][name]) - float(previous["positions"][name]))
                * ratio
                for name in FRANKA_JOINT_NAMES
            }
            return target, min(max(elapsed / max(last_time, 1e-9), 0.0), 1.0), False
    return dict(points[-1]["positions"]), 1.0, True


def _validated_gripper_opening(value: Any) -> float:
    opening = float(value)
    if not math.isfinite(opening) or opening < 0.0 or opening > 0.1:
        raise ProfileRuntimeError("Franka 实际夹爪开度无效")
    return opening


def _gripper_control_direction(current: float, target: float, tolerance: float) -> float:
    """把米制开度误差转换为 Panda GRIP 的方向输入。

    两版 robosuite 都会在 gripper model 内部对输入方向积分，因此这里不能把
    opening 直接缩放成 action。+1 表示继续关闭，-1 表示继续打开，进入容差
    后必须输出 0，让内部 actuator 目标停止继续变化。
    """

    if current > target + tolerance:
        return 1.0
    if current < target - tolerance:
        return -1.0
    return 0.0


def _copy_observation(value: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: np.array(item, copy=True) if hasattr(item, "shape") else item
        for key, item in value.items()
    }


def _copy_base_pose(value: Dict[str, Any]) -> Dict[str, Any]:
    """校验并复制 worker 提供的世界位姿，避免可变数组跨线程共享。"""

    position = np.asarray(value.get("position"), dtype=np.float64)
    quaternion = np.asarray(value.get("quaternion_xyzw"), dtype=np.float64)
    frame_id = str(value.get("frame_id", ""))
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not bool(np.isfinite(position).all())
        or not bool(np.isfinite(quaternion).all())
    ):
        raise ProfileRuntimeError("Franka 基座位姿格式无效")
    if frame_id != "world":
        raise ProfileRuntimeError("Franka Profile 的基座位姿必须位于 world 坐标系")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        raise ProfileRuntimeError("Franka 基座四元数无效")
    quaternion = quaternion / norm
    return {
        "position": [float(item) for item in position],
        "quaternion_xyzw": [float(item) for item in quaternion],
        "frame_id": frame_id,
    }


def _sensor_bindings(observation: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    result = []
    for key in sorted(observation):
        if key.endswith("_image"):
            result.append((key[:-6] + "_rgb", key, "rgb"))
        elif key.endswith("_depth"):
            result.append((key, key, "depth"))
    return result


def _scene_objects(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    objects = []
    for key in sorted(observation):
        if not key.endswith("_pos") or key.startswith("robot0_") or "_to_" in key:
            continue
        prefix = key[:-4]
        quat_key = prefix + "_quat"
        if quat_key not in observation:
            continue
        objects.append(
            {
                "source_id": prefix,
                "category": "object",
                "name": prefix,
                "pose": {
                    "position": _vector(observation[key], 3, 0.0),
                    "quaternion_xyzw": _unit_quaternion(_vector(observation[quat_key], 4, 0.0)),
                    "frame_id": "world",
                },
                "state": {},
            }
        )
    return objects


def _vector(value: Any, size: int, default: float) -> List[float]:
    if value is None:
        result = [default] * size
    else:
        result = [float(item) for item in np.asarray(value).reshape(-1)[:size]]
        result.extend([default] * (size - len(result)))
    return result


def _unit_quaternion(value: List[float]) -> List[float]:
    norm = math.sqrt(sum(item * item for item in value))
    if norm < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    return [item / norm for item in value]


def _fingerprint(value: Dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
