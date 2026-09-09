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

"""单个活动场景的生命周期和命令调度。"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from plugin_mujoco.errors import ConflictError, NotFoundError
from plugin_mujoco.models import (
    CommandState,
    RobotCommand,
    RobotCommandRequest,
    SceneInstance,
    SceneSnapshot,
    SceneState,
    SnapshotRobotState,
    utc_now,
)
from plugin_mujoco.runtime.interfaces import RuntimeComponents
from plugin_mujoco.streaming import (
    EncodedFrame,
    LatestFrameHub,
    PoseFrameMetadata,
    RobotStateFrameMetadata,
)
from plugin_mujoco.visuals import VISUAL_CONTENT_VERSION, ViewerScene


def _realtime_deadline_delay(deadline: float, now: float) -> float:
    """按连续墙钟deadline限速，避免每步sleep误差不断累积。"""
    return max(deadline - now, 0.0)


@dataclass
class _PhysicsCall:
    """交给物理线程执行的一次同步控制请求。"""

    callback: Callable[[], Any]
    completed: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


class RuntimeInstance:
    def __init__(
        self,
        record: SceneInstance,
        components: RuntimeComponents,
        *,
        realtime: bool,
    ) -> None:
        self.record = record
        self.components = components
        # 只为现有测试客户端保留只读兼容入口；Runtime 内部按职责访问组件。
        self.backend = components.robots
        self.realtime = realtime
        self._condition = threading.Condition(threading.RLock())
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        self._commands: dict[tuple[str, str], RobotCommand] = {}
        self._command_fingerprints: dict[tuple[str, str], str] = {}
        self._command_started_monotonic: dict[tuple[str, str], float] = {}
        self._sensor_sequences: dict[tuple[str, str], int] = {}
        self._physics_calls: deque[_PhysicsCall] = deque()
        self._frames = LatestFrameHub()
        self._pose_sequence = 0
        self._pose_generation = record.generation
        self._pose_node_count = 0
        self._pose_payload = b""
        self._pose_last_wall = 0.0
        # Viewer只需要平滑观察，不应在1kHz物理线程内按60Hz复制整棵场景树。
        # 30Hz足以保持连续视觉，同时把动态节点序列化和锁持有开销减半；
        # 相机/传感器仍由独立LatestFrameHub按各自请求频率产生，不受此值影响。
        self._pose_interval = 1.0 / 30.0

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self.record.state = SceneState.RUNNING
            self.record.updated_at = utc_now()
            self._thread = threading.Thread(
                target=self._run,
                name=f"mujoco-{self.record.instance_id}",
                daemon=True,
            )
            self._thread.start()

    def view(self) -> SceneInstance:
        with self._condition:
            self.record.sim_time = self.components.physics.sim_time
            return self.record.model_copy(deep=True)

    def pause(self) -> SceneInstance:
        with self._condition:
            if self.record.state == SceneState.PAUSED:
                return self.view()
            self._require_state(SceneState.RUNNING)
        # hold 会修改 qvel/ctrl，只能由物理线程执行。
        self._physics_call(self._hold_all_robots)
        with self._condition:
            self.record.state = SceneState.PAUSED
            self.record.updated_at = utc_now()
            self._condition.notify_all()
            return self.view()

    def resume(self) -> SceneInstance:
        with self._condition:
            if self.record.state == SceneState.RUNNING:
                return self.view()
            self._require_state(SceneState.PAUSED)
            self.record.state = SceneState.RUNNING
            self.record.updated_at = utc_now()
            self._condition.notify_all()
            return self.view()

    def step(self, steps: int = 1) -> SceneInstance:
        """暂停时由物理线程单步，API 线程永远不直接修改 MjData。"""
        if steps < 1 or steps > 1000:
            raise ValueError("steps 必须在 1 到 1000 之间")
        with self._condition:
            self._require_state(SceneState.PAUSED)
            if self._active_commands():
                raise ConflictError("存在活动 Robot 命令时不能单步")

        def advance() -> None:
            for _ in range(steps):
                self.components.physics.step()
                self.record.step_count += 1
            self._capture_scene_pose(force=True)

        self._physics_call(advance)
        with self._condition:
            self.record.sim_time = self.components.physics.sim_time
            self.record.updated_at = utc_now()
            return self.view()

    def reset(self) -> SceneInstance:
        with self._condition:
            self._require_state(SceneState.RUNNING, SceneState.PAUSED)
            if self._active_commands():
                raise ConflictError(
                    "场景仍有活动命令，必须先 stop 或 hold",
                    details={"commands": [item.command_id for item in self._active_commands()]},
                )
            self.record.state = SceneState.RESETTING
            self.record.updated_at = utc_now()
        self._physics_call(self.components.physics.reset)
        self._frames.stop_all()
        with self._condition:
            self.record.generation += 1
            self.record.step_count = 0
            self._sensor_sequences.clear()
            self._pose_sequence = 0
            self._pose_generation = self.record.generation
            self._pose_last_wall = 0.0
            self.record.state = SceneState.RUNNING
            self.record.updated_at = utc_now()
            self._condition.notify_all()
            return self.view()

    def stop(self, timeout: float = 5.0) -> SceneInstance:
        with self._condition:
            if self.record.state == SceneState.STOPPED:
                return self.view()
            self._require_state(SceneState.RUNNING, SceneState.PAUSED, SceneState.FAILED)
            failed = self.record.state == SceneState.FAILED
        if not failed and self._thread is not None and self._thread.is_alive():
            self._physics_call(self._hold_all_robots)
        with self._condition:
            self.record.state = SceneState.STOPPING
            self.record.updated_at = utc_now()
            self._cancel_all("场景停止")
            self._stop_requested = True
            thread = self._thread
            self._condition.notify_all()
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise ConflictError("场景线程未能在限定时间内停止")
        self._frames.stop_all()
        self.components.physics.close()
        with self._condition:
            self.record.state = SceneState.STOPPED
            self.record.updated_at = utc_now()
            return self.view()

    def viewer_scene(self) -> ViewerScene:
        """返回当前 Scene Instance 的不可变视觉描述。"""
        with self._condition:
            return ViewerScene(
                generation=self.record.generation,
                scene_revision=self.components.visuals.scene_revision,
                content_url=(
                    f"/api/v1/scene-instances/{self.record.instance_id}/viewer-scene/content"
                ),
                pose_stream_url=(f"/api/v1/scene-instances/{self.record.instance_id}/pose-stream"),
                dynamic_node_order=list(self.components.visuals.dynamic_node_order),
                cameras=list(self.components.visuals.cameras),
            )

    def viewer_scene_content(self) -> bytes:
        return self.components.visuals.content()

    def scene_pose_frame(self, *, after_sequence: int = 0, timeout: float = 2.0) -> EncodedFrame:
        """等待物理线程发布的新位姿；超时则返回最近帧而不阻塞物理循环。"""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._pose_sequence <= after_sequence and not self._stop_requested:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._pose_sequence == 0:
                raise TimeoutError("等待 Scene Pose 首帧超时")
            metadata = PoseFrameMetadata(
                sequence=self._pose_sequence,
                generation=self._pose_generation,
                scene_revision=self.components.visuals.scene_revision,
                sim_time=self.record.sim_time,
                node_count=self._pose_node_count,
            )
            return EncodedFrame(metadata, self._pose_payload)

    def encoded_sensor_frame(
        self, robot_id: str, sensor_id: str, *, after_sequence: int = 0
    ) -> EncodedFrame:
        descriptor = next(
            (
                item
                for item in self.components.sensors.sensor_descriptors(robot_id)
                if item.sensor_id == sensor_id
            ),
            None,
        )
        if descriptor is None:
            raise NotFoundError(f"传感器不存在: {sensor_id}")
        key = f"sensor:{robot_id}:{sensor_id}"

        def produce(sequence: int) -> EncodedFrame:
            with self._condition:
                generation = self.record.generation
            return self.components.sensors.encoded_sensor_frame(
                robot_id, sensor_id, generation=generation, sequence=sequence
            )

        self._frames.ensure(key, descriptor.fps, produce)
        # EGL 首次创建 Renderer/编译 shader 可能超过普通帧的 2 秒等待窗口。
        # 这里只放宽首帧等待，不改变后台帧率，也不会让慢客户端形成队列。
        return self._frames.latest(key, after_sequence=after_sequence, timeout=10.0)

    def encoded_robot_state(self, robot_id: str, *, after_sequence: int = 0) -> EncodedFrame:
        """返回 Robot State 的共享最新帧，不让慢客户端阻塞物理线程。

        RobotDriver 只在自己的实现边界内读取 MuJoCo。这里复用传感器/Viewer 已经
        验证过的 LatestFrameHub：每台 Robot 只有一个 20 Hz 生产器，多个 SDK
        客户端只读取同一个单帧缓存。reset 会清除生产器，新 generation 的首帧会
        终止旧 SDK 订阅，绝不会把 reset 前后的状态拼成一条连续轨迹。
        """

        self.components.robots.robot_profile(robot_id)
        key = f"robot-state:{robot_id}"

        def produce(sequence: int) -> EncodedFrame:
            with self._condition:
                generation = self.record.generation
                sim_time = self.components.physics.sim_time
            state = self.components.robots.robot_state(robot_id, generation)
            return EncodedFrame(
                RobotStateFrameMetadata(
                    sequence=sequence,
                    generation=generation,
                    sim_time=sim_time,
                    observed_at=state.observed_at,
                    frame_id=state.base_pose.frame_id,
                    robot_id=robot_id,
                ),
                state.model_dump_json().encode("utf-8"),
            )

        self._frames.ensure(key, 20.0, produce)
        return self._frames.latest(key, after_sequence=after_sequence)

    def robot_ids(self) -> list[str]:
        return self.components.robots.robot_ids()

    def submit_command(self, robot_id: str, request: RobotCommandRequest) -> RobotCommand:
        self.components.robots.robot_profile(robot_id)
        fingerprint = _fingerprint(request.model_dump(mode="json"))
        key = (robot_id, request.command_id)
        with self._condition:
            if self.record.state != SceneState.RUNNING:
                raise ConflictError(
                    "只有 running 场景可以接收 Robot 命令",
                    details={"state": self.record.state.value},
                )
            if request.scene_generation != self.record.generation:
                raise ConflictError(
                    "命令 generation 已失效",
                    details={
                        "requested": request.scene_generation,
                        "current": self.record.generation,
                    },
                )
            if key in self._commands:
                if self._command_fingerprints[key] != fingerprint:
                    raise ConflictError("相同 command_id 的输入不同")
                return self._commands[key].model_copy(deep=True)
            requested_resources = set(request.resources)
            conflicts = [
                item.command_id
                for item in self._active_commands(robot_id)
                if requested_resources.intersection(item.resources)
            ]
            if conflicts:
                raise ConflictError(
                    "Robot 资源正在执行其他命令",
                    details={"resources": sorted(requested_resources), "commands": conflicts},
                )
            now = utc_now()
            command = RobotCommand(
                robot_id=robot_id,
                status=CommandState.ACCEPTED,
                message="命令已接受",
                accepted_at=now,
                updated_at=now,
                **request.model_dump(mode="json"),
            )
            self._commands[key] = command
            self._command_fingerprints[key] = fingerprint
            self._condition.notify_all()
            return command.model_copy(deep=True)

    def command(self, robot_id: str, command_id: str) -> RobotCommand:
        with self._condition:
            try:
                return self._commands[(robot_id, command_id)].model_copy(deep=True)
            except KeyError as exc:
                raise NotFoundError(f"命令不存在: {command_id}") from exc

    def stop_command(self, robot_id: str, command_id: str) -> RobotCommand:
        with self._condition:
            command = self._commands.get((robot_id, command_id))
            if command is None:
                raise NotFoundError(f"命令不存在: {command_id}")
            if command.status in {
                CommandState.SUCCEEDED,
                CommandState.FAILED,
                CommandState.CANCELLED,
                CommandState.UNKNOWN,
            }:
                return command.model_copy(deep=True)
            # 先移出可运行集合，避免物理线程在 stop 调用排队期间再次推进轨迹。
            self._finish(command, CommandState.CANCELLED, "命令已停止，Robot 进入 hold")
        try:
            self._physics_call(lambda: self.components.robots.stop_robot(robot_id))
        except Exception:
            with self._condition:
                command.status = CommandState.UNKNOWN
                command.failure_reason = "停止结果无法确认"
                command.updated_at = utc_now()
            raise
        return command.model_copy(deep=True)

    def hold_robot(self, robot_id: str) -> None:
        with self._condition:
            for command in self._active_commands(robot_id):
                self._finish(command, CommandState.CANCELLED, "Robot 进入 hold")
        self._physics_call(lambda: self.components.robots.hold_robot(robot_id))

    def snapshot(self) -> SceneSnapshot:
        with self._condition:
            robots = [
                SnapshotRobotState(
                    **self.components.robots.robot_state(
                        robot_id, self.record.generation
                    ).model_dump(),
                    visual_ref={
                        "visual_id": self.components.robots.robot_profile(robot_id).model,
                        "version": VISUAL_CONTENT_VERSION,
                    },
                )
                for robot_id in self.components.robots.robot_ids()
            ]
            sensors = [
                sensor
                for robot_id in self.components.robots.robot_ids()
                for sensor in self.components.sensors.sensor_descriptors(robot_id)
            ]
            return SceneSnapshot(
                scene_key=self.record.scene_key,
                instance_id=self.record.instance_id,
                generation=self.record.generation,
                robots=robots,
                objects=self.components.snapshot.scene_objects(),
                regions=self.components.snapshot.scene_regions(),
                sensors=sensors,
                observed_at=utc_now(),
            )

    def fail(self, reason: str) -> None:
        # 物理异常发生在物理线程本身，此处可直接执行 hold，随后终止循环。
        with self._condition:
            self.record.state = SceneState.FAILED
            self.record.failure_reason = reason
            self.record.updated_at = utc_now()
            self._cancel_all(reason)
            for robot_id in self.components.robots.robot_ids():
                self.components.robots.hold_robot(robot_id)
            self._stop_requested = True
            while self._physics_calls:
                call = self._physics_calls.popleft()
                call.error = RuntimeError(f"物理线程已失败: {reason}")
                call.completed.set()
            self._condition.notify_all()

    def _run(self) -> None:
        """物理线程是 Runtime 中唯一可以修改物理模型的线程。"""
        realtime_deadline: float | None = None
        try:
            with self._condition:
                self._capture_scene_pose(force=True)
            while True:
                call: _PhysicsCall | None = None
                should_step = False
                with self._condition:
                    if self._physics_calls:
                        call = self._physics_calls.popleft()
                    elif self._stop_requested:
                        return
                    elif self.record.state == SceneState.PAUSED:
                        self._condition.wait(timeout=1.0)
                        continue
                    elif self.record.state == SceneState.RUNNING:
                        should_step = True
                    else:
                        self._condition.wait(timeout=0.1)
                        continue
                if call is not None:
                    try:
                        call.result = call.callback()
                    except BaseException as exc:
                        call.error = exc
                    finally:
                        call.completed.set()
                        # RGB-D等同步调用可能占用较长墙钟时间；它们结束后从
                        # 当前时刻重新建立节拍，不通过突发物理步追赶传感调用。
                        realtime_deadline = None
                    continue
                if should_step:
                    step_started_at = time.monotonic()
                    with self._condition:
                        self._advance_commands()
                        self.components.physics.step()
                        self.record.step_count += 1
                        self.record.sim_time = self.components.physics.sim_time
                        self.record.updated_at = utc_now()
                        self._capture_scene_pose()
                    if self.realtime:
                        # 每步都从"当前时刻+timestep"重新sleep会把操作系统的
                        # 唤醒误差永久累加，layout001会因此长期低于实时速度。
                        # 连续deadline允许下一步自然补回一次短暂晚唤醒；如果
                        # 已落后20ms以上则重新定基准，避免形成突发追赶。
                        timestep = self.components.physics.timestep
                        if realtime_deadline is None:
                            realtime_deadline = step_started_at + timestep
                        else:
                            realtime_deadline += timestep
                        now = time.monotonic()
                        if now - realtime_deadline > max(0.02, timestep * 5):
                            realtime_deadline = now
                        delay = _realtime_deadline_delay(realtime_deadline, now)
                        if delay > 0:
                            time.sleep(delay)
                    else:
                        time.sleep(0)
        except Exception as exc:
            self.fail(str(exc))

    def _capture_scene_pose(self, *, force: bool = False) -> None:
        """只由物理线程调用，把 MjData 位姿复制成不可变 float32 帧。"""
        now = time.monotonic()
        if not force and now - self._pose_last_wall < self._pose_interval:
            return
        captured = self.components.visuals.capture()
        with self._condition:
            self._pose_sequence += 1
            self._pose_generation = self.record.generation
            self._pose_node_count = captured.node_count
            self._pose_payload = captured.payload
            self._pose_last_wall = now
            self._condition.notify_all()

    def _physics_call(self, callback: Callable[[], Any], timeout: float = 5.0) -> Any:
        if threading.current_thread() is self._thread:
            return callback()
        call = _PhysicsCall(callback)
        with self._condition:
            if self._thread is None or not self._thread.is_alive():
                raise ConflictError("物理线程当前不可用")
            self._physics_calls.append(call)
            self._condition.notify_all()
        if not call.completed.wait(timeout):
            raise ConflictError("物理线程未在限定时间内响应")
        if call.error is not None:
            raise call.error
        return call.result

    def _hold_all_robots(self) -> None:
        for robot_id in self.components.robots.robot_ids():
            self.components.robots.hold_robot(robot_id)

    def _advance_commands(self) -> None:
        completed_robots: set[str] = set()
        for key, command in list(self._commands.items()):
            if command.status == CommandState.ACCEPTED:
                command.status = CommandState.RUNNING
                command.started_at = utc_now()
                command.updated_at = command.started_at
                command.message = "命令执行中"
                self._command_started_monotonic[key] = self.components.physics.sim_time
            if command.status != CommandState.RUNNING:
                continue
            # 使用仿真时间，pause 期间命令进度和超时都不会偷偷推进。
            elapsed = self.components.physics.sim_time - self._command_started_monotonic[key]
            if elapsed >= command.timeout_seconds:
                reason = "命令超时，Robot 已进入 hold"
                tracking = command.details.get("joint_tracking")
                if isinstance(tracking, dict):
                    reason += (
                        f"；最大关节误差 {tracking.get('max_error_rad')} rad，"
                        f"容差 {tracking.get('position_tolerance_rad')} rad，"
                        f"未收敛关节 {tracking.get('errors_rad')}"
                    )
                self._fail_command_and_hold(command, reason)
                continue
            target = command.target
            points = target.get("points", [])
            if points:
                duration = max(float(points[-1]["time_from_start"]), 1e-9)
                command.progress = min(elapsed / duration, 0.999)
            result = self.components.robots.advance_command(command.robot_id, command, elapsed)
            if result.failed:
                self._fail_command_and_hold(command, result.reason or "命令失败")
            elif result.completed:
                self._finish(command, CommandState.SUCCEEDED, None)
                # 接触式 insert 与 close_until_contact 的最终控制目标分别包含
                # 固定钩爪压入 receiver、活动夹板持续夹持所需的微小预载。
                # 若命令一成功就用实测关节位置重新 hold，会把这段
                # 预载清掉，VerifyPregrasp 与随后的夹紧立即看到脱钩。此处只让
                # 原位置控制目标继续生效，下一条 Ability 命令会正常接管；显式
                # stop、失败和普通轨迹完成仍走 hold_robot，安全停止语义不变。
                preserves_contact_preload = (
                    command.type == "joint_trajectory"
                    and bool(command.target.get("stop_on_contact"))
                ) or (
                    command.type == "gripper_command"
                    and bool(command.target.get("close_until_contact"))
                )
                if not preserves_contact_preload:
                    completed_robots.add(command.robot_id)

        for robot_id in completed_robots:
            if not self._active_commands(robot_id):
                # Ability 会把左右工具拆成两个 Runtime command。任一工具先完成
                # 时都不能调用全 Robot hold，否则会把另一侧尚未到位的目标冻结。
                # 只有该 Robot 的全部低层命令都进入终态后才统一锁存当前姿态。
                self.components.robots.hold_robot(robot_id)

    def _fail_command_and_hold(self, command: RobotCommand, reason: str) -> None:
        """一个低层命令失败时停止同 Robot 的其余命令并锁存安全姿态。"""
        self._finish(command, CommandState.FAILED, reason)
        for active in self._active_commands(command.robot_id):
            self._finish(
                active,
                CommandState.CANCELLED,
                f"同一 Robot 的并行命令失败，已进入 hold: {command.command_id}",
            )
        # 两侧工具由独立 command 驱动，但物理上共同承载一个物体。只停止失败
        # 的一侧会让另一侧继续施力，因此失败边界必须收敛到 Robot 级 hold。
        self.components.robots.stop_robot(command.robot_id)

    def _finish(
        self,
        command: RobotCommand,
        status: CommandState,
        reason: str | None,
    ) -> None:
        command.status = status
        command.failure_reason = reason
        command.message = reason or (
            "命令执行成功" if status == CommandState.SUCCEEDED else status.value
        )
        if status == CommandState.SUCCEEDED:
            command.progress = 1.0
        command.ended_at = utc_now()
        command.updated_at = command.ended_at

    def _active_commands(self, robot_id: str | None = None) -> list[RobotCommand]:
        return [
            command
            for command in self._commands.values()
            if command.status in {CommandState.ACCEPTED, CommandState.RUNNING}
            and (robot_id is None or command.robot_id == robot_id)
        ]

    def _cancel_all(self, reason: str) -> None:
        for command in self._active_commands():
            self._finish(command, CommandState.CANCELLED, reason)

    def _require_state(self, *allowed: SceneState) -> None:
        if self.record.state not in allowed:
            raise ConflictError(
                "当前场景状态不允许该操作",
                details={
                    "state": self.record.state.value,
                    "allowed": [item.value for item in allowed],
                },
            )


def _fingerprint(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
