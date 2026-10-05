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

"""Runtime API 与 Robot SDK 共用的数据模型。"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SceneState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    RESETTING = "resetting"
    STOPPING = "stopping"
    FAILED = "failed"


class CommandState(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class Pose(StrictModel):
    """公共位姿使用米、xyzw 四元数和明确坐标帧。"""

    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    frame_id: str = "world"

    @model_validator(mode="after")
    def validate_quaternion(self) -> Pose:
        norm_squared = sum(value * value for value in self.quaternion_xyzw)
        if norm_squared <= 1e-12:
            raise ValueError("四元数不能为零")
        return self


class SceneStartRequest(StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    runtime_profile_id: str = "native-mujoco"
    runtime_bundle_id: str | None = None
    layout: str = Field(min_length=1, max_length=128)
    seed: int = 0
    headless: bool = True
    render_backend: Literal["auto", "egl", "osmesa", "glfw"] = "auto"


class SceneDescriptor(StrictModel):
    scene_key: str
    name: str
    scene_kind: str
    layouts: list[str]
    robot_models: list[str]
    compatible_runtime_profiles: list[str]
    read_only: bool = False


class SceneInstance(StrictModel):
    instance_id: str
    scene_key: str
    layout: str
    seed: int
    headless: bool
    render_backend: str
    generation: int
    state: SceneState
    sim_time: float = 0.0
    step_count: int = 0
    request_id: str
    runtime_profile_id: str = "native-mujoco"
    runtime_bundle_id: str | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    progress_message: str | None = None
    created_at: datetime
    updated_at: datetime
    failure_reason: str | None = None


class RuntimeCapability(StrictModel):
    """Runtime 能力必须显式声明，调用方不能根据 Profile 名称猜测。"""

    editable_scene: bool = False
    native_evaluator: bool = False
    viewer: bool = True
    viewer_camera_modes: list[Literal["free", "fixed"]] = Field(
        default_factory=lambda: ["free", "fixed"]
    )
    scene_step: bool = True
    scene_reset: bool = True
    robot_models: list[str] = Field(default_factory=list)
    sensor_kinds: list[str] = Field(default_factory=lambda: ["rgb", "depth", "contact"])


class RuntimeProfile(StrictModel):
    """Profile 静态说明加当前进程的真实环境探测结果。"""

    runtime_profile_id: str
    name: str
    engine: str
    loader: str
    api_version: str = "v1"
    scene_kinds: list[str]
    capabilities: RuntimeCapability = Field(default_factory=RuntimeCapability)
    environment: str
    environment_ready: bool = False
    available: bool
    unavailable_reason: str | None = None

    @property
    def profile_id(self) -> str:
        return self.runtime_profile_id


class RuntimeInfo(StrictModel):
    runtime_id: str = "plugin-mujoco"
    runtime_profile_id: str = "native-mujoco"
    state: Literal["ready", "busy", "failed"]
    engine: str = "mujoco"
    version: str
    api_version: str = "v1"
    active_instance_id: str | None
    capabilities: RuntimeCapability
    managed: bool = False
    # Framework 不会把资产路径传到 Studio，但 Runtime 本地诊断需要保留该字段。
    asset_root: str


class SceneEvaluation(StrictModel):
    """Profile Runtime 返回的原生评测证据。

    reward 和 success 由 robosuite、LIBERO 等运行环境计算，只用于回归和
    验收取证，不代表上层 Workflow、Robot Skill 或 Ability 的业务结果。
    generation 用于阻止 reset 前的评测数据被错误显示为当前场景结果。
    """

    scene_key: str
    instance_id: str
    generation: int = Field(ge=1)
    runtime_profile_id: str
    reward: float
    success: bool
    terminated: bool
    language: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime


class SceneStepRequest(StrictModel):
    steps: int = Field(default=1, ge=1, le=1000)


class RobotCapability(StrictModel):
    commands: list[Literal["joint_trajectory", "base_trajectory", "gripper_command"]]
    sensors: list[str]
    frames: list[str]


class RobotToolDescriptor(StrictModel):
    """可配置末端工具描述；字段与 Framework 的 VirtualRobot 合同一致。

    行程和力限制直接属于工具，不再套一层 Runtime 私有 ``limits``，这样
    Framework、Pilot 和型号 SDK 使用同一份 JSON 时无需维护转换器。
    """

    tool_ref: str = Field(min_length=1)
    side: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    frame: str = Field(min_length=1)
    joint: str = Field(min_length=1)
    travel_m: float = Field(gt=0)
    normal_force_n: float = Field(gt=0)
    maximum_force_n: float = Field(gt=0)


class VirtualRobotDescriptor(StrictModel):
    """虚拟 Robot 公共描述，不包含 MuJoCo 内部名称。"""

    robot_id: str
    model: str
    backend: str
    kind: str
    coordinate_frame: str
    sdk_package: str
    backend_profile: str
    endpoint: str
    urdf_path: str | None = None
    package_directories: list[str] = Field(default_factory=list)
    joint_names: list[str]
    end_effectors: list[str]
    grippers: list[str]
    tools: list[RobotToolDescriptor] = Field(default_factory=list)
    capabilities: RobotCapability


# 内部短名称与公开模型共用同一份字段，避免维护两套 Robot 描述。
RobotProfile = VirtualRobotDescriptor


class JointState(StrictModel):
    position: float
    velocity: float
    effort: float | None = None


class GripperState(StrictModel):
    """末端夹具的低层实测状态。

    position/velocity 使用米和米每秒，effort 使用牛顿。Runtime 只公开当前
    传感值；稳定承载、滑移和业务持物状态由 Ability 基于时间序列判断。
    """

    position: float
    velocity: float
    effort: float
    target_position: float | None = None
    reached_target: bool = False
    hook_contact: bool = False
    clamp_contact: bool = False
    sensor_fault: bool = False
    hook_force_n: float = 0.0
    clamp_force_n: float = 0.0
    hook_support_ratio: float = 0.0
    hook_tangential_speed_m_s: float = 0.0


class RobotState(StrictModel):
    robot_id: str
    generation: int
    observed_at: datetime
    base_pose: Pose
    joints: dict[str, JointState]
    end_effectors: dict[str, Pose]
    grippers: dict[str, float]
    gripper_states: dict[str, GripperState] = Field(default_factory=dict)
    in_hold: bool


class TrajectoryPoint(StrictModel):
    """Robot SDK 已完成规划的轨迹采样点。

    `time_from_start_seconds` 使用仿真秒；`positions` 的键由 Robot Profile 定义。
    Plugin 不解释末端目标或导航目标，只在物理周期内跟随这些采样点。
    """

    time_from_start_seconds: float = Field(ge=0.0)
    positions: dict[str, float] = Field(min_length=1)
    velocities: dict[str, float] = Field(default_factory=dict)


class JointTrajectory(StrictModel):
    resources: list[str] = Field(min_length=1)
    frame_id: str = Field(min_length=1, max_length=128)
    points: list[TrajectoryPoint] = Field(min_length=1)
    position_tolerance_rad: float = Field(default=0.002, gt=0)
    stop_on_contact: bool = False
    contact_tool_refs: list[str] = Field(default_factory=list)
    max_contact_force_n: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_points(self) -> JointTrajectory:
        _validate_timeline([point.time_from_start_seconds for point in self.points])
        joints = set(self.points[0].positions)
        if any(set(point.positions) != joints for point in self.points[1:]):
            raise ValueError("关节轨迹的每个采样点必须包含相同关节")
        for point in self.points:
            if point.velocities and not set(point.velocities).issubset(joints):
                raise ValueError("velocities 不能包含轨迹之外的关节")
        if self.stop_on_contact and not self.contact_tool_refs:
            raise ValueError("接触完成的关节轨迹必须声明目标工具")
        return self


class BaseTrajectory(StrictModel):
    frame_id: str = Field(min_length=1, max_length=128)
    points: list[TrajectoryPoint] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_points(self) -> BaseTrajectory:
        _validate_timeline([point.time_from_start_seconds for point in self.points])
        required = {"x", "y", "yaw"}
        if any(set(point.positions) != required for point in self.points):
            raise ValueError("底盘轨迹 positions 必须且只能包含 x、y、yaw")
        return self


class GripperCommand(StrictModel):
    gripper_id: str = Field(min_length=1, max_length=64)
    position: float = Field(ge=0.0)
    max_effort: float = Field(default=0.0, ge=0.0)
    stop_on_contact: bool = False


class RobotCommandRequest(StrictModel):
    """Framework 与 Robot SDK 共用的正式低层命令封装。

    三种类型必须且只能携带一个同名负载。stop/hold 使用独立端点，不能伪装
    为轨迹命令。该边界保证 Runtime 不会重新实现 IK、导航或抓取策略。
    """

    command_id: str = Field(min_length=1, max_length=128)
    scene_generation: int = Field(ge=1)
    type: Literal["joint_trajectory", "base_trajectory", "gripper_command"]
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    joint_trajectory: JointTrajectory | None = None
    base_trajectory: BaseTrajectory | None = None
    gripper_command: GripperCommand | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> RobotCommandRequest:
        values = {
            "joint_trajectory": self.joint_trajectory,
            "base_trajectory": self.base_trajectory,
            "gripper_command": self.gripper_command,
        }
        present = [name for name, value in values.items() if value is not None]
        if present != [self.type]:
            raise ValueError(f"{self.type} 命令必须且只能携带同名负载")
        return self

    @property
    def target(self) -> dict[str, Any]:
        """转换为现有低层 Driver 的插值输入，不恢复旧公开协议。"""
        if self.joint_trajectory is not None:
            return {
                "points": [
                    {"time_from_start": point.time_from_start_seconds, "positions": point.positions}
                    for point in self.joint_trajectory.points
                ],
                "tolerance": self.joint_trajectory.position_tolerance_rad,
                "stop_on_contact": self.joint_trajectory.stop_on_contact,
                "contact_tool_refs": list(self.joint_trajectory.contact_tool_refs),
                "max_contact_force_n": self.joint_trajectory.max_contact_force_n,
            }
        if self.base_trajectory is not None:
            return {
                "points": [
                    {"time_from_start": point.time_from_start_seconds, **point.positions}
                    for point in self.base_trajectory.points
                ],
                "position_tolerance": 0.05,
                "yaw_tolerance": 0.08,
            }
        assert self.gripper_command is not None
        return {
            "gripper": self.gripper_command.gripper_id,
            "opening": self.gripper_command.position,
            "force_limit_n": self.gripper_command.max_effort or None,
            "close_until_contact": self.gripper_command.stop_on_contact,
            "tolerance": 0.002,
        }

    @property
    def coordinate_frame(self) -> str:
        if self.joint_trajectory is not None:
            return self.joint_trajectory.frame_id
        if self.base_trajectory is not None:
            return self.base_trajectory.frame_id
        return "world"

    @property
    def resources(self) -> list[str]:
        if self.joint_trajectory is not None:
            return list(self.joint_trajectory.resources)
        if self.base_trajectory is not None:
            return ["base"]
        assert self.gripper_command is not None
        return [f"gripper:{self.gripper_command.gripper_id}"]


class RobotCommand(RobotCommandRequest):
    robot_id: str
    status: CommandState
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    accepted_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    failure_reason: str | None = None


class RobotCommandStatus(StrictModel):
    """命令提交和查询接口返回的轻量状态视图。

    ``RobotCommand`` 在 Runtime 内仍保存完整轨迹，用于幂等指纹、物理推进和
    调试。REST 回执只需要把身份与终态交给 SDK；若把数千个轨迹点在每次
    Feedback 轮询中重复序列化，会让已经接受的物理命令被客户端误判为提交
    超时。这里不是删除执行证据，完整请求仍由 Runtime 按 command_id 持有。
    """

    command_id: str
    scene_generation: int
    type: Literal["joint_trajectory", "base_trajectory", "gripper_command"]
    robot_id: str
    status: CommandState
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    accepted_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    failure_reason: str | None = None


class RobotHoldRequest(StrictModel):
    scene_generation: int = Field(ge=1)


class RobotOperationResult(StrictModel):
    command_id: str
    robot_id: str
    scene_generation: int
    type: Literal["hold"]
    status: CommandState
    progress: float = 1.0
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime


class SensorDescriptor(StrictModel):
    sensor_id: str
    robot_id: str
    kind: Literal["rgb", "depth", "contact"]
    frame_id: str
    encoding: str = ""
    width: int | None = None
    height: int | None = None
    fps: float = Field(default=15.0, gt=0.0, le=120.0)

    @model_validator(mode="after")
    def fill_encoding(self) -> SensorDescriptor:
        if not self.encoding:
            self.encoding = {
                "rgb": "jpeg",
                "depth": "png16-mm",
                "contact": "json",
            }[self.kind]
        return self

    @property
    def frame(self) -> str:
        """帧生产内部兼容属性；公共 JSON 只输出 frame_id。"""
        return self.frame_id


class SensorFrame(StrictModel):
    sensor_id: str
    kind: str
    observed_at: datetime
    sequence: int = 0
    generation: int = 0
    frame: str = "world"
    media_type: str
    encoding: str
    width: int | None = None
    height: int | None = None
    payload_size: int = 0
    data: str | dict[str, Any] | None = None


class VisualReference(StrictModel):
    """浏览器视觉资产的稳定引用，不包含 Runtime 宿主路径。"""

    visual_id: str
    version: str


class SnapshotRobotState(RobotState):
    """场景快照中的 Robot 可携带视觉引用，但 Robot SDK 状态接口保持不变。"""

    visual_ref: VisualReference | None = None


class SceneObject(StrictModel):
    source_id: str
    category: str
    name: str
    pose: Pose
    extent: tuple[float, float, float] | None = None
    visual_ref: VisualReference | None = None
    state: dict[str, Any] = Field(default_factory=dict)


class SceneRegion(StrictModel):
    source_id: str
    name: str
    pose: Pose
    extent: tuple[float, float, float] | None = None
    visual_ref: VisualReference | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class SceneSnapshot(StrictModel):
    scene_key: str
    instance_id: str
    generation: int
    coordinate_frame: str = "world"
    robots: list[SnapshotRobotState]
    objects: list[SceneObject]
    regions: list[SceneRegion]
    sensors: list[SensorDescriptor]
    observed_at: datetime


class ErrorBody(StrictModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


def _validate_timeline(times: list[float]) -> None:
    """拒绝含糊时间线，Runtime 因而只负责跟随而不需要猜测轨迹。"""
    if not times or times[0] != 0.0:
        raise ValueError("轨迹第一个采样点的 time_from_start 必须为 0")
    if any(current <= previous for previous, current in zip(times, times[1:], strict=False)):
        raise ValueError("轨迹采样时间必须严格递增")
