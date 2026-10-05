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

"""原生 MuJoCo 物理后端。"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import threading
from collections import defaultdict
from contextlib import suppress
from typing import Any

import numpy as np

from plugin_mujoco.errors import BackendFailureError, NotFoundError, ValidationRuntimeError
from plugin_mujoco.imaging import encode_depth_png16, encode_rgb_jpeg
from plugin_mujoco.models import (
    GripperState,
    JointState,
    Pose,
    RobotCapability,
    RobotCommand,
    RobotProfile,
    RobotState,
    RobotToolDescriptor,
    SceneObject,
    SceneRegion,
    SensorDescriptor,
    SensorFrame,
    utc_now,
)
from plugin_mujoco.robots.profile import RobotMapping
from plugin_mujoco.runtime.backend import CommandTick
from plugin_mujoco.scene import RuntimeSceneBuild, SceneDefinition, build_runtime_scene
from plugin_mujoco.streaming import EncodedFrame, SensorFrameMetadata
from plugin_mujoco.visuals import VISUAL_CONTENT_VERSION, public_visual_id


class MujocoBackend:
    """加载一个真实 MuJoCo 场景并提供 Robot SDK 所需的原子操作。"""

    def __init__(
        self,
        definition: SceneDefinition,
        *,
        render_backend: str,
        seed: int,
        endpoint: str = "http://127.0.0.1:8090",
    ) -> None:
        configured_backend = os.environ.get("MUJOCO_GL")
        if "mujoco" in sys.modules and configured_backend != render_backend:
            raise BackendFailureError(
                "同一 Runtime 进程不能切换 MuJoCo 渲染后端",
                details={
                    "configured": configured_backend,
                    "requested": render_backend,
                },
            )
        os.environ["MUJOCO_GL"] = render_backend
        try:
            import mujoco
        except Exception as exc:
            raise BackendFailureError(
                "MuJoCo 或所选无头渲染后端无法初始化",
                details={"render_backend": render_backend, "reason": str(exc)},
            ) from exc

        self.mj = mujoco
        self.definition = definition
        self.endpoint = endpoint
        self.build: RuntimeSceneBuild = build_runtime_scene(definition)
        self._lock = threading.RLock()
        self._renderers: dict[tuple[int, int], Any] = {}
        # 物理循环和渲染不能共用同一个 MjData。物理线程是 self.data 的唯一
        # 写入者；渲染线程只使用下面这份独立模型和状态副本。这样 JPEG/Depth
        # 编码再慢也不会长时间占用物理锁，Viewer 只会丢帧而不会拖慢 sim_time。
        self._public_body_cache: dict[int, str | None] = {}
        self._render_model: Any | None = None
        self._render_data: Any | None = None
        self._in_hold = {binding.robot_id: True for binding in self.build.robots}
        self._holding: dict[str, str | None] = {
            binding.robot_id: None for binding in self.build.robots
        }
        self._hold_targets: dict[str, dict[str, float]] = {}
        self._hold_effort_limits: dict[str, float] = {}
        self._gripper_targets: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        self._tool_contacts: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._tool_contact_sample_period_s = 0.004
        self._next_tool_contact_sample_time = 0.0
        # 这是低层关节控制的承载前馈锁存，不是对外的“稳定持物”结论。它只
        # 决定执行器是否继续补偿当前接触约束，并在开夹时清除；Runtime 对外
        # 始终只报告当前接触、力和相对速度。稳定时间窗与抓取成功由 Ability
        # 根据带时间戳的连续采样判断，不能从 MuJoCo 物理帧数向上泄漏。
        self._load_support_latched: dict[str, bool] = {
            binding.robot_id: False for binding in self.build.robots
        }
        self._stable_upper_body_positions: dict[str, dict[str, float]] = {}
        self._bindings_by_robot_id = {
            binding.robot_id: binding for binding in self.build.robots
        }
        self._model_object_ids: dict[tuple[int, str], int] = {}
        np.random.seed(seed)
        try:
            self.model = mujoco.MjModel.from_xml_path(str(self.build.xml_path))
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)
            self._profiles = self._load_profiles()
            self._joint_control_gains = {
                f"{binding.prefix}{joint}": self._profiles[binding.robot_id].control_gains(joint)
                for binding in self.build.robots
                for joint in self._profiles[binding.robot_id].controlled_joint_names
            }
            self._joint_robot_ids = {
                f"{binding.prefix}{joint}": binding.robot_id
                for binding in self.build.robots
                for joint in self._profiles[binding.robot_id].controlled_joint_names
            }
        except Exception as exc:
            self.build.cleanup()
            raise BackendFailureError(
                "MuJoCo 场景加载失败",
                details={"xml": str(self.build.xml_path), "reason": str(exc)},
            ) from exc
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()
        for robot_id in self.robot_ids():
            self.hold_robot(robot_id)

    @property
    def sim_time(self) -> float:
        return float(self.data.time)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def step(self) -> None:
        with self._lock:
            self._apply_holds()
            self.mj.mj_step(self.model, self.data)
            self.data.xfrc_applied[:] = 0
            # MuJoCo接触和低层控制仍以模型timestep推进；这里只把对外工具
            # 传感器的力、方向和相对速度派生限制为250 Hz。它对应真机传感器
            # 采样，不是业务稳定时间窗，也不向SDK/Ability暴露仿真帧计数。
            # insert最多延迟4ms看到hook接触，在当前接触速度下不足0.4mm。
            if float(self.data.time) + 1e-12 >= self._next_tool_contact_sample_time:
                self._detect_holding()
                self._next_tool_contact_sample_time = (
                    float(self.data.time) + self._tool_contact_sample_period_s
                )

    def reset(self) -> None:
        with self._lock:
            self.mj.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self._initial_qpos
            self.data.qvel[:] = self._initial_qvel
            self.data.ctrl[:] = 0
            self.mj.mj_forward(self.model, self.data)
            for robot_id in self._in_hold:
                self._in_hold[robot_id] = True
                self._holding[robot_id] = None
            self._hold_targets.clear()
            self._hold_effort_limits.clear()
            self._gripper_targets.clear()
            self._tool_contacts.clear()
            self._next_tool_contact_sample_time = 0.0
            for robot_id in self._load_support_latched:
                self._load_support_latched[robot_id] = False
            self._stable_upper_body_positions.clear()
            for robot_id in self.robot_ids():
                self.hold_robot(robot_id)

    def close(self) -> None:
        with self._lock:
            for renderer in self._renderers.values():
                close = getattr(renderer, "close", None)
                if close:
                    close()
            self._renderers.clear()
            self._render_data = None
            self._render_model = None
            self.build.cleanup()

    def robot_ids(self) -> list[str]:
        return [binding.robot_id for binding in self.build.robots]

    def robot_profile(self, robot_id: str) -> RobotProfile:
        mapping = self._mapping(robot_id)
        sensor_ids: list[str] = ["contact"]
        for public_name in mapping.sensors.values():
            sensor_ids.extend([f"{public_name}.rgb", f"{public_name}.depth"])
        return RobotProfile(
            robot_id=robot_id,
            model=mapping.model,
            backend=mapping.backend,
            kind=mapping.kind,
            coordinate_frame=mapping.coordinate_frame,
            sdk_package=mapping.sdk_package,
            backend_profile=mapping.backend_profile,
            endpoint=self.endpoint,
            urdf_path=str(mapping.urdf_path) if mapping.urdf_path else None,
            # URDF 使用 package://r1_pro_chassis/... 与
            # package://r1_pro_tote_gripper/...。两者都位于 Asset 的 robot/
            # 目录下，必须把这一事实随 Robot Descriptor 交给部署层；让启动器
            # 根据绝对 URDF 路径猜目录会在只读类型包或远程 Runtime 下失效。
            package_directories=([str(mapping.urdf_path.parents[2])] if mapping.urdf_path else []),
            joint_names=list(mapping.command_joint_names),
            end_effectors=list(mapping.end_effectors),
            grippers=list(mapping.grippers),
            tools=[
                RobotToolDescriptor(
                    tool_ref=tool.tool_ref,
                    side=side,
                    kind=tool.kind,
                    frame=tool.frame,
                    joint=tool.joint,
                    travel_m=tool.software_max_position_m - tool.software_min_position_m,
                    normal_force_n=tool.normal_force_n,
                    maximum_force_n=tool.peak_force_n,
                )
                for side, tool in mapping.gripper_tools.items()
            ],
            capabilities=RobotCapability(
                commands=["joint_trajectory", "base_trajectory", "gripper_command"],
                sensors=sensor_ids,
                frames=[
                    mapping.coordinate_frame,
                    mapping.kinematic_root_frame,
                    *mapping.end_effectors,
                    *(tool.frame for tool in mapping.gripper_tools.values()),
                ],
            ),
        )

    def robot_state(self, robot_id: str, generation: int) -> RobotState:
        binding = self._binding(robot_id)
        mapping = self._mapping(robot_id)
        with self._lock:
            joints: dict[str, JointState] = {}
            for public in mapping.joint_names:
                internal = f"{binding.prefix}{public}"
                joint_id = self._id(self.mj.mjtObj.mjOBJ_JOINT, internal)
                qpos_address = int(self.model.jnt_qposadr[joint_id])
                dof_address = int(self.model.jnt_dofadr[joint_id])
                joints[public] = JointState(
                    position=float(self.data.qpos[qpos_address]),
                    velocity=float(self.data.qvel[dof_address]),
                    effort=float(self.data.qfrc_actuator[dof_address]),
                )
            base_pose = self._base_pose(binding, mapping)
            gripper_states = self._gripper_states(binding, mapping, joints)
            return RobotState(
                robot_id=robot_id,
                generation=generation,
                observed_at=utc_now(),
                base_pose=base_pose,
                joints=joints,
                end_effectors={
                    side: self._site_pose(f"{binding.prefix}{site_name}")
                    for side, site_name in mapping.end_effectors.items()
                },
                grippers={
                    side: sum(joints[name].position for name in joint_names) / len(joint_names)
                    for side, joint_names in mapping.grippers.items()
                },
                gripper_states=gripper_states,
                in_hold=self._in_hold[robot_id],
            )

    def advance_command(
        self,
        robot_id: str,
        command: RobotCommand,
        elapsed: float,
    ) -> CommandTick:
        """在物理周期内跟随 Robot SDK 已生成的轨迹。

        本方法不做 IK、路径搜索或抓取策略。命令中的采样时间和目标已经由
        型号 Robot SDK 检查，Runtime 只负责插值、低层驱动和实际状态确认。
        """
        binding = self._binding(robot_id)
        mapping = self._mapping(robot_id)
        with self._lock:
            self._in_hold[robot_id] = False
            target = command.target
            if command.type == "base_trajectory":
                sample, finished = _interpolate_base(target["points"], elapsed)
                yaw_public = mapping.base_joints["yaw_joint"]
                yaw_internal = f"{binding.prefix}{yaw_public}"
                yaw_joint_id = self._id(self.mj.mjtObj.mjOBJ_JOINT, yaw_internal)
                yaw_qpos_address = int(self.model.jnt_qposadr[yaw_joint_id])
                # 公共 Pose 四元数只能表达 [-pi, pi]，而 MuJoCo 的底盘 yaw
                # joint 保存连续角度。上一段到达 3*pi/2 后，下一段轨迹起点会
                # 等价地写成 -pi/2；必须映射回当前关节附近，否则会瞬间反转
                # 2*pi 并破坏携物接触。
                sample["yaw"] = _nearest_equivalent_angle(
                    float(self.data.qpos[yaw_qpos_address]),
                    float(sample["yaw"]),
                )
                desired = {
                    mapping.base_joints["x_joint"]: sample["x"],
                    mapping.base_joints["y_joint"]: sample["y"],
                    yaw_public: sample["yaw"],
                }
                errors = [
                    self._drive_public_joint(binding, name, value)
                    for name, value in desired.items()
                ]
                if finished and max(abs(value) for value in errors) <= float(
                    target["position_tolerance"]
                ):
                    return CommandTick(completed=True)
                return CommandTick()
            if command.type == "joint_trajectory":
                positions, finished = _interpolate_joints(target["points"], elapsed)
                unknown = sorted(set(positions) - set(mapping.command_joint_names))
                if unknown:
                    return CommandTick(failed=True, reason=f"Robot Profile 中没有关节: {unknown}")
                errors = [
                    self._drive_public_joint(binding, name, value)
                    for name, value in positions.items()
                ]
                contact_tick = self._joint_contact_tick(robot_id, mapping, command)
                if contact_tick is not None:
                    return contact_tick
                if finished:
                    tolerance = float(target["tolerance"])
                    # 轨迹时间走完并不代表 Robot 已经到达目标。真实负载下如果
                    # 某个关节受力矩限制或控制参数影响无法收敛，仅返回“命令超时”
                    # 会把轨迹、控制器和接触问题混在一起。这里只保存超出正式
                    # 位置容差的紧凑诊断，不回传数千个轨迹点，也不改变完成条件。
                    tracking_errors = {
                        name: round(float(error), 7)
                        for (name, _), error in zip(positions.items(), errors, strict=True)
                        if abs(error) > tolerance
                    }
                    command.details["joint_tracking"] = {
                        "position_tolerance_rad": tolerance,
                        "max_error_rad": round(
                            max((abs(value) for value in errors), default=0.0), 7
                        ),
                        "errors_rad": tracking_errors,
                    }
                    # 对普通关节轨迹，到达几何终点就是完成；对 insert，调用方
                    # 已明确要求由真实 hook contact 决定终点，不能把“轨迹走完
                    # 但没有钩入箱沿”伪装成成功。
                    if bool(target.get("stop_on_contact")):
                        blockers = self._insert_contact_geometry(
                            robot_id,
                            mapping,
                            [str(value) for value in target.get("contact_tool_refs", [])],
                        )
                        detail = "、".join(blockers) if blockers else "无工具-箱体接触"
                        return CommandTick(
                            failed=True,
                            reason=f"insert 到达轨迹终点但未建立 receiver 接触；当前接触: {detail}",
                        )
                    if max(abs(value) for value in errors) <= tolerance:
                        return CommandTick(completed=True)
                return CommandTick()
            if command.type == "gripper_command":
                side = str(target["gripper"])
                opening = float(target["opening"])
                requested_force = target.get("force_limit_n")
                try:
                    tool = mapping.validate_gripper_target(side, opening, requested_force)
                except ValidationRuntimeError as exc:
                    return CommandTick(failed=True, reason=str(exc))
                joint_names = mapping.grippers[side]
                force_limit_n = float(requested_force or tool.normal_force_n) if tool else 0.0
                control_limit = (
                    force_limit_n / tool.control_force_scale_n_per_unit if tool else None
                )
                close_until_contact = bool(target.get("close_until_contact"))
                previous_target = self._gripper_targets[robot_id].get(side, {})
                if (
                    not close_until_contact
                    and previous_target
                    and opening > float(previous_target.get("position", opening)) + 0.001
                ):
                    # 开夹是 Runtime 能可靠识别的“主动卸载”边界。从这一刻起
                    # 不再保留携物前馈或上一稳定双臂姿态；逐侧释放的安全性由
                    # 上层实时 ToolState/VerifyPlacement 继续确认。
                    self._load_support_latched[robot_id] = False
                    self._stable_upper_body_positions.pop(robot_id, None)
                contact = self._tool_contacts.get(robot_id, {}).get(side, {})
                same_command = previous_target.get("command_id") == command.command_id
                # force_limit_n 是执行器允许施加的上限，不是抓取成功阈值。
                # close_until_contact 仍需完成工具Profile声明的机械预紧，否则
                # 第一次轻触就结束命令会让夹片在外拉时打滑。这里仅闭合低层
                # 执行器并锁存位置目标；是否持续承载仍由Ability读取原始力、
                # 接触和相对速度判断，Runtime不输出业务级抓取状态。
                minimum_clamp_force_n = (
                    min(force_limit_n, float(tool.minimum_clamp_force_n))
                    if tool is not None
                    else 0.0
                )
                clamp_contact_observed = bool(
                    close_until_contact
                    and tool is not None
                    and contact.get("contact_object")
                    and contact.get("clamp_contact")
                    and float(contact.get("clamp_force_n", 0.0)) > 0.0
                    and float(contact.get("clamp_force_n", 0.0)) <= tool.peak_force_n
                )
                contact_seen = bool(
                    clamp_contact_observed
                    or (same_command and previous_target.get("contact_seen"))
                )
                require_contact = bool(
                    close_until_contact
                    and (
                        (same_command and previous_target.get("require_contact"))
                        or (not same_command and previous_target)
                    )
                )
                contact_opening = (
                    float(previous_target["contact_opening"])
                    if same_command and previous_target.get("contact_opening") is not None
                    else None
                )
                if same_command and previous_target.get("contact_latched"):
                    effective_opening = float(previous_target["position"])
                    contact_latched = True
                elif clamp_contact_observed:
                    current_opening = sum(
                        self._public_joint_position(binding, name) for name in joint_names
                    ) / len(joint_names)
                    if contact_opening is None:
                        contact_opening = current_opening
                    measured_force_n = float(contact.get("clamp_force_n", 0.0))
                    missing_force_n = max(
                        0.0, minimum_clamp_force_n - measured_force_n
                    )
                    preload = max(
                        (
                            missing_force_n
                            / tool.control_force_scale_n_per_unit
                            / self._joint_control_gains[
                                f"{binding.prefix}{name}"
                            ].kp
                        )
                        for name in joint_names
                    )
                    # 同一接触命令每轮按当前缺失力推进控制目标；一旦达到设备
                    # 最低预紧，继续保留上一轮更深的目标来维持力，不能把目标
                    # 改回瞬时关节位置而立即卸载。该反馈只发生在命令活动期，
                    # 不维护全局抓取状态，也不依赖MuJoCo物理步数。
                    previous_preload_target = (
                        float(previous_target["position"])
                        if previous_target.get("contact_seen")
                        else current_opening
                    )
                    contact_latched = measured_force_n >= minimum_clamp_force_n
                    if contact_latched and previous_target.get("contact_latched"):
                        effective_opening = previous_preload_target
                    else:
                        effective_opening = max(
                            opening,
                            min(previous_preload_target, current_opening - preload),
                        )
                else:
                    # 新的恢复命令不能把上一条接触锁存误当成本命令终态，也不能
                    # 因候选名义位置更大而反向开夹。当前无接触时继续向内寻找；
                    # 新接触出现后再由上面的分支锁存当时开度。
                    effective_opening = (
                        min(opening, float(previous_target["position"]))
                        if close_until_contact and previous_target
                        else opening
                    )
                    contact_latched = False
                self._gripper_targets[robot_id][side] = {
                    "command_id": command.command_id,
                    "position": effective_opening,
                    "force_limit_n": force_limit_n,
                    "contact_seen": contact_seen,
                    "contact_opening": contact_opening,
                    "require_contact": require_contact,
                    "contact_latched": contact_latched,
                }
                errors = [
                    self._drive_public_joint(
                        binding, name, effective_opening, control_limit=control_limit
                    )
                    for name in joint_names
                ]
                reached_target = max(abs(value) for value in errors) <= float(target["tolerance"])
                complete = reached_target
                if close_until_contact:
                    # 首次普通闭合仍可在软件目标处结束；一旦观察到夹片接触，
                    # 立即以当前开度结束“闭合到接触”。这里不累计仿真帧，也不
                    # 输出持物结论或等待业务夹紧阈值。
                    complete = bool(
                        contact_latched
                        or (not contact_seen and not require_contact and reached_target)
                    )
                return CommandTick(completed=complete)
            return CommandTick(failed=True, reason=f"不支持的低层命令: {command.type}")

    def _joint_contact_tick(
        self, robot_id: str, mapping: RobotMapping, command: RobotCommand
    ) -> CommandTick | None:
        """把夹具插入的预期接触转换为当前关节命令的终态。

        普通末端运动仍必须进入关节容差。只有 Skill 明确标记的 insert 动作才
        可以由目标工具的 hook 接触提前完成；后续 VerifyPregrasp 会重新读取
        实际末端和接触状态，因此这里不把“碰到任意物体”提升为抓取成功。
        """
        target = command.target
        tool_refs = [str(value) for value in target.get("contact_tool_refs", [])]
        if not tool_refs:
            return None
        sides_by_ref = {tool.tool_ref: side for side, tool in mapping.gripper_tools.items()}
        unknown = sorted(set(tool_refs) - set(sides_by_ref))
        if unknown:
            return CommandTick(failed=True, reason=f"Robot Profile 中没有工具: {unknown}")
        contacts = [
            self._tool_contacts.get(robot_id, {}).get(sides_by_ref[tool_ref], {})
            for tool_ref in tool_refs
        ]
        force_limit = target.get("max_contact_force_n")
        for tool_ref, contact in zip(tool_refs, contacts, strict=True):
            force = max(
                float(contact.get("hook_force_n", 0.0)),
                float(contact.get("clamp_force_n", 0.0)),
            )
            tool = mapping.gripper_tools[sides_by_ref[tool_ref]]
            maximum_force = (
                min(float(force_limit), tool.peak_force_n) if force_limit else tool.peak_force_n
            )
            if force > maximum_force:
                return CommandTick(
                    failed=True,
                    reason=f"{tool_ref} 接触力超过 insert 安全限制",
                )
        if not bool(target.get("stop_on_contact")):
            return None

        # insert 是低层“遇到接触停止”，不是承载判定。当前钩爪接触即可结束
        # 运动，后续 VerifyPregrasp/Ability 会用连续实时采样判断是否可夹紧。
        all_hooks_contact = bool(contacts) and all(
            bool(contact.get("hook_contact")) and float(contact.get("hook_force_n", 0.0)) > 0.0
            for contact in contacts
        )
        return CommandTick(completed=True) if all_hooks_contact else None

    def stop_robot(self, robot_id: str) -> None:
        """立即清零速度并锁定当前关节位置，不能只修改命令状态。"""
        self.hold_robot(robot_id)

    def hold_robot(self, robot_id: str) -> None:
        binding = self._binding(robot_id)
        mapping = self._mapping(robot_id)
        with self._lock:
            stable_tool_targets: dict[str, dict[str, float]] = {}
            for public_name in mapping.joint_names:
                internal = f"{binding.prefix}{public_name}"
                joint_id = self._id(self.mj.mjtObj.mjOBJ_JOINT, internal)
                dof_address = int(self.model.jnt_dofadr[joint_id])
                self.data.qvel[dof_address] = 0.0
                actuator_id = self._optional_id(self.mj.mjtObj.mjOBJ_ACTUATOR, internal)
                if actuator_id >= 0:
                    self.data.ctrl[actuator_id] = 0.0
                self._hold_effort_limits.pop(internal, None)
            for side, _tool in mapping.gripper_tools.items():
                contact = self._tool_contacts.get(robot_id, {}).get(side, {})
                target = self._gripper_targets.get(robot_id, {}).get(side)
                if (
                    target is not None
                    and target.get("contact_latched")
                    and contact.get("hook_contact")
                    and contact.get("clamp_contact")
                ):
                    # 单侧外拉尚未形成Robot级双侧持物，但该工具已经建立了
                    # 真机等价的接触锁存和预紧力。上臂轨迹完成后的通用hold
                    # 必须继续驱动这个工具目标；若改写为瞬时回弹开度，第一侧
                    # 会在下一次观测前自行卸载。Robot级holding判定仍只在双侧
                    # 承载同一物体时成立，这里不提升任何业务抓取状态。
                    stable_tool_targets[side] = target
            self._in_hold[robot_id] = True
            self._hold_targets[robot_id] = self._joint_positions(binding, mapping)
            if self._load_support_latched.get(robot_id):
                # 底盘停在 stop 时的位置；双臂和躯干恢复到最近一次双侧真实
                # 接触时的姿态，避免把失稳后已经下沉的位置变成安全目标。
                # 这只是低层 hold，不是抓取成功判断。
                self._hold_targets[robot_id].update(
                    self._stable_upper_body_positions.get(robot_id, {})
                )
            # 普通 stop/hold 必须锁存当前位置，不能让尚未完成的夹具继续运动。
            # 唯一例外是仍有双侧真实接触的工具：它们继续保持
            # stop_on_contact 已经锁存的接触位置和同一力限制；它不固定箱体，
            # 也不写物体 Pose。
            for side, target in stable_tool_targets.items():
                tool = mapping.gripper_tools[side]
                force_limit = float(target["force_limit_n"]) / tool.control_force_scale_n_per_unit
                for public_name in mapping.grippers[side]:
                    internal = f"{binding.prefix}{public_name}"
                    self._hold_targets[robot_id][internal] = float(target["position"])
                    self._hold_effort_limits[internal] = force_limit
            self.mj.mj_forward(self.model, self.data)

    def sensor_descriptors(self, robot_id: str) -> list[SensorDescriptor]:
        mapping = self._mapping(robot_id)
        configured = {item.sensor_id: item for item in self.definition.sensors}
        result: list[SensorDescriptor] = []
        for public in mapping.sensors.values():
            config = configured.get(public)
            width = config.width if config else 640
            height = config.height if config else 480
            result.extend(
                [
                    SensorDescriptor(
                        sensor_id=f"{public}.rgb",
                        robot_id=robot_id,
                        kind="rgb",
                        frame_id=public,
                        width=width,
                        height=height,
                    ),
                    SensorDescriptor(
                        sensor_id=f"{public}.depth",
                        robot_id=robot_id,
                        kind="depth",
                        frame_id=public,
                        width=width,
                        height=height,
                    ),
                ]
            )
        result.append(
            SensorDescriptor(
                sensor_id="contact", robot_id=robot_id, kind="contact", frame_id="world"
            )
        )
        return result

    def sensor_frame(self, robot_id: str, sensor_id: str) -> SensorFrame:
        binding = self._binding(robot_id)
        descriptor = next(
            (item for item in self.sensor_descriptors(robot_id) if item.sensor_id == sensor_id),
            None,
        )
        if descriptor is None:
            raise NotFoundError(f"传感器不存在: {sensor_id}")
        if descriptor.kind == "contact":
            with self._lock:
                # contact_object 是 MuJoCo 内部用于关节前馈的对象关联，不是
                # 真机通用传感值，不能经 Runtime API 泄漏给 SDK/Skill。
                public_fields = {
                    "hook_contact",
                    "clamp_contact",
                    "hook_force_n",
                    "clamp_force_n",
                    "hook_support_ratio",
                    "hook_tangential_speed_m_s",
                }
                tool_states = {
                    side: {key: value for key, value in state.items() if key in public_fields}
                    for side, state in self._tool_contacts.get(robot_id, {}).items()
                }
                contacts = self._public_contacts(binding)
                data: str | dict[str, Any] = {
                    "active": bool(contacts),
                    "count": len(contacts),
                    "contacts": contacts,
                    "tools": tool_states,
                }
            media_type, encoding = "application/json", "json"
        else:
            camera_public = sensor_id.rsplit(".", 1)[0]
            internal = f"{binding.prefix}{camera_public}"
            array = self._render(
                internal,
                int(descriptor.width or 640),
                int(descriptor.height or 480),
                depth=descriptor.kind == "depth",
            )
            output = io.BytesIO()
            np.save(output, array, allow_pickle=False)
            data = base64.b64encode(output.getvalue()).decode("ascii")
            media_type, encoding = "application/x-npy", "base64"
        return SensorFrame(
            sensor_id=sensor_id,
            kind=descriptor.kind,
            observed_at=utc_now(),
            media_type=media_type,
            encoding=encoding,
            width=descriptor.width,
            height=descriptor.height,
            data=data,
        )

    def encoded_sensor_frame(
        self,
        robot_id: str,
        sensor_id: str,
        *,
        generation: int,
        sequence: int,
        quality: int = 85,
    ) -> EncodedFrame:
        binding = self._binding(robot_id)
        descriptor = next(
            (item for item in self.sensor_descriptors(robot_id) if item.sensor_id == sensor_id),
            None,
        )
        if descriptor is None:
            raise NotFoundError(f"传感器不存在: {sensor_id}")
        width, height = int(descriptor.width or 640), int(descriptor.height or 480)
        if descriptor.kind == "rgb":
            internal = f"{binding.prefix}{sensor_id.rsplit('.', 1)[0]}"
            payload = encode_rgb_jpeg(
                self._render(internal, width, height, depth=False),
                quality=quality,
            )
            media_type, encoding = "image/jpeg", "jpeg"
        elif descriptor.kind == "depth":
            internal = f"{binding.prefix}{sensor_id.rsplit('.', 1)[0]}"
            payload = encode_depth_png16(self._render(internal, width, height, depth=True))
            media_type, encoding = "image/png", "png16-mm"
        else:
            structured = self.sensor_frame(robot_id, sensor_id).data
            if not isinstance(structured, dict):
                raise ValidationRuntimeError("结构化传感器必须返回 JSON 对象")
            payload = json.dumps(structured, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            media_type, encoding = "application/json", "json"
        return EncodedFrame(
            SensorFrameMetadata(
                stream="sensor",
                sequence=sequence,
                generation=generation,
                observed_at=utc_now(),
                media_type=media_type,
                encoding=encoding,
                width=width,
                height=height,
                frame=descriptor.frame,
                sensor_id=sensor_id,
            ),
            payload,
        )

    def _source_id_for_body(self, model: Any, body_id: int) -> str | None:
        while body_id > 0:
            body_name = self.mj.mj_id2name(model, self.mj.mjtObj.mjOBJ_BODY, body_id)
            for source_id, runtime_name in self.build.object_body_names.items():
                if body_name == runtime_name:
                    return source_id
            body_id = int(model.body_parentid[body_id])
        return None

    def visual_source_for_body(self, body_id: int) -> str | None:
        """把视觉 body 归并到公开场景对象或整台虚拟 Robot。

        GLB 节点只写公开 ``source_id``。MuJoCo body 名称和整数 ID 只在
        Runtime 内用于沿父链查找，浏览器点击 Robot 任意 Link 时都会选择整台
        Robot，而不会看到内部 link/body 标识。
        """
        object_source = self._source_id_for_body(self.model, body_id)
        if object_source is not None:
            return object_source
        for binding in self.build.robots:
            mapping = self._mapping(binding.robot_id)
            root_name = f"{binding.prefix}{mapping.root_body}"
            root_id = self._optional_id(self.mj.mjtObj.mjOBJ_BODY, root_name)
            if root_id < 0:
                continue
            current = int(body_id)
            while current > 0:
                if current == root_id:
                    return binding.robot_id
                current = int(self.model.body_parentid[current])
        return None

    def scene_objects(self) -> list[SceneObject]:
        result: list[SceneObject] = []
        with self._lock:
            for item in self.definition.assets:
                if item.category == "region":
                    continue
                body_id = self._id(self.mj.mjtObj.mjOBJ_BODY, item.source_id)
                position = np.asarray(self.data.xpos[body_id], dtype=float)
                if item.pose_reference == "bottom_center" and item.size is not None:
                    # 场景资产可以用支撑点布置，但对外 Snapshot 必须统一为几何中心。
                    # 这里应用的是资产声明的坐标约定，不按箱型或业务类别写特例。
                    rotation = np.asarray(self.data.xmat[body_id], dtype=float).reshape(3, 3)
                    position = position + rotation @ np.asarray(
                        (0.0, 0.0, float(item.size[2]) * 0.5),
                        dtype=float,
                    )
                result.append(
                    SceneObject(
                        source_id=item.source_id,
                        category=item.category,
                        name=item.source_id,
                        pose=Pose(
                            position=tuple(float(value) for value in position),
                            quaternion_xyzw=_xyzw_from_wxyz(self.data.xquat[body_id]),
                        ),
                        extent=item.size,
                        visual_ref={
                            "visual_id": public_visual_id(item.model, item.category),
                            "version": VISUAL_CONTENT_VERSION,
                        },
                        state={
                            "model": item.model,
                            "interactive": item.interactive,
                            "static": item.static,
                            "in_contact": self._object_in_contact(item.source_id),
                        },
                    )
                )
        return result

    def scene_regions(self) -> list[SceneRegion]:
        return [
            SceneRegion(
                source_id=item.source_id,
                name=item.source_id,
                pose=Pose(
                    position=item.position,
                    quaternion_xyzw=item.quaternion_xyzw,
                ),
                extent=item.size,
                visual_ref={
                    "visual_id": public_visual_id(item.model, item.category),
                    "version": VISUAL_CONTENT_VERSION,
                },
                properties=dict(item.properties or {}),
            )
            for item in self.definition.assets
            if item.category == "region"
        ]

    def _load_profiles(self) -> dict[str, RobotMapping]:
        profiles: dict[str, RobotMapping] = {}
        names: set[str] = set()
        for kind, count in (
            (self.mj.mjtObj.mjOBJ_BODY, self.model.nbody),
            (self.mj.mjtObj.mjOBJ_JOINT, self.model.njnt),
            (self.mj.mjtObj.mjOBJ_SITE, self.model.nsite),
            (self.mj.mjtObj.mjOBJ_CAMERA, self.model.ncam),
        ):
            for index in range(int(count)):
                name = self.mj.mj_id2name(self.model, kind, index)
                if name:
                    names.add(name)
        for binding in self.build.robots:
            profile_path = (
                self.definition.robot_root
                / binding.model
                / "config"
                / "semantic_robot_profile.yaml"
            )
            mapping = RobotMapping.load(profile_path)
            # binding.model 是 MuJoCo 资产包名，mapping.model 是对外硬件型号。
            # 末端工具变体允许复用同一硬件型号；名称一致性由下面的完整
            # body/joint/site 校验保证，不能再错误要求资产包名等于硬件型号。
            mapping.validate_model_names(names, prefix=binding.prefix)
            profiles[binding.robot_id] = mapping
        return profiles

    def _mapping(self, robot_id: str) -> RobotMapping:
        try:
            return self._profiles[robot_id]
        except KeyError as exc:
            raise NotFoundError(f"Robot 不存在: {robot_id}") from exc

    def _binding(self, robot_id: str):
        try:
            return self._bindings_by_robot_id[robot_id]
        except KeyError as exc:
            raise NotFoundError(f"Robot 不存在: {robot_id}") from exc

    def _drive_joint(
        self, internal: str, target: float, *, control_limit: float | None = None
    ) -> float:
        joint_id = self._id(self.mj.mjtObj.mjOBJ_JOINT, internal)
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        dof_address = int(self.model.jnt_dofadr[joint_id])
        error = target - float(self.data.qpos[qpos_address])
        actuator_id = self._optional_id(self.mj.mjtObj.mjOBJ_ACTUATOR, internal)
        if actuator_id < 0:
            position_actuator_id = self._optional_id(
                self.mj.mjtObj.mjOBJ_ACTUATOR, f"{internal}_motor"
            )
            if position_actuator_id >= 0:
                if control_limit is not None:
                    raise ValidationRuntimeError(
                        "专用夹具必须使用可限制输出的 motor actuator",
                        details={"joint": internal},
                    )
                self.data.ctrl[position_actuator_id] = target
                return error
        if actuator_id >= 0:
            # MuJoCo 动力学为 M*qacc + bias = passive + actuator + constraint。
            # 双侧工具当前共同支撑同一刚体时，刚体重量会通过接触约束传回双臂；
            # 若遗漏该项，纯 PD 只能依靠持续的位置偏差产生承载力矩。反过来，
            # pregrasp/insert 的 constraint 也可能来自箱沿碰撞或关节限制，
            # 盲目补偿会把 Robot 推向障碍。因此这里只依据当前物理接触和低层
            # 控制关联决定是否补偿约束力。该关联仅服务MuJoCo关节控制，不是
            # “稳定持物”业务状态，也不通过Runtime API暴露；实际接触消失后
            # 必须立即停止补偿。稳定时间窗、滑移和过载结论由Ability使用
            # Runtime上报的力、方向比例和相对速度计算。
            gains = self._joint_control_gains.get(internal)
            kp = gains.kp if gains is not None else 150.0
            kd = gains.kd if gains is not None else 15.0
            robot_id = self._joint_robot_ids.get(internal)
            holding_load = bool(
                robot_id
                and (
                    self._holding.get(robot_id)
                    or self._has_current_latched_load_support(robot_id)
                    or self._has_current_latched_tool_contact(robot_id)
                )
            )
            # 单侧外拉时箱体仍受托盘支撑，但夹具接触力已经通过机械臂关节
            # 传回。上臂若忽略这部分约束力，hold目标会在重规划等待期间漂移，
            # 下一次外拉便从错误高度开始并把承重钩带出。这里对具有当前接触
            # 证据的上臂补偿约束力；夹具自身仍使用限力位置控制，不能叠加
            # 约束前馈，否则会越过工具力上限。这个内部关联不对外输出持物结论。
            constraint = (
                float(self.data.qfrc_constraint[dof_address])
                if holding_load and control_limit is None
                else 0.0
            )
            feedforward = (
                float(self.data.qfrc_bias[dof_address])
                - float(self.data.qfrc_passive[dof_address])
                - constraint
            )
            control = error * kp - float(self.data.qvel[dof_address]) * kd + feedforward
            if self.model.actuator_ctrllimited[actuator_id]:
                low, high = self.model.actuator_ctrlrange[actuator_id]
                control = float(np.clip(control, low, high))
            if control_limit is not None:
                control = float(np.clip(control, -control_limit, control_limit))
            self.data.ctrl[actuator_id] = control
        else:
            if control_limit is not None:
                raise ValidationRuntimeError(
                    "专用夹具关节缺少可限制输出的 actuator",
                    details={"joint": internal},
                )
            self.data.qvel[dof_address] = _clip(error * 5.0, 1.0)
        return error

    def _has_current_latched_tool_contact(self, robot_id: str) -> bool:
        """当前任一锁存工具仍有真实钩入和压紧接触。"""

        contacts = self._tool_contacts.get(robot_id, {})
        for side, target in self._gripper_targets.get(robot_id, {}).items():
            contact = contacts.get(side, {})
            if (
                target.get("contact_latched")
                and contact.get("hook_contact")
                and contact.get("clamp_contact")
                and float(contact.get("hook_force_n", 0.0)) > 0.0
                and float(contact.get("clamp_force_n", 0.0)) > 0.0
            ):
                return True
        return False

    def _has_current_latched_load_support(self, robot_id: str) -> bool:
        """判断当前原始接触是否仍允许维持低层承载前馈。

        该锁存只用于防止控制器突然撤掉重力补偿；每个物理步仍要求双侧
        hook/clamp 接触同一内部对象且力为正。它不输出持续时间、稳定状态或
        对象身份，也不会固定箱体或修改物体位姿。
        """
        if not self._load_support_latched.get(robot_id):
            return False
        mapping = self._profiles.get(robot_id)
        if mapping is None or len(mapping.gripper_tools) < 2:
            return False
        contacts = self._tool_contacts.get(robot_id, {})
        load_objects: set[str] = set()
        for side in mapping.gripper_tools:
            contact = contacts.get(side, {})
            # contact_object 只来自当前 MuJoCo 接触对，永不进入公共协议。
            load_object = contact.get("contact_object")
            if (
                not contact.get("hook_contact")
                or not contact.get("clamp_contact")
                or not load_object
                or float(contact.get("hook_force_n", 0.0)) <= 0.0
                or float(contact.get("clamp_force_n", 0.0)) <= 0.0
            ):
                return False
            load_objects.add(str(load_object))
        return len(load_objects) == 1

    def _drive_public_joint(
        self,
        binding: Any,
        public_name: str,
        target: float,
        *,
        control_limit: float | None = None,
    ) -> float:
        """把公共关节名转换为引擎名；映射只存在于 Runtime 内部。"""
        internal = f"{binding.prefix}{public_name}"
        self._hold_targets[binding.robot_id][internal] = float(target)
        if control_limit is None:
            self._hold_effort_limits.pop(internal, None)
        else:
            self._hold_effort_limits[internal] = float(control_limit)
        return self._drive_joint(internal, float(target), control_limit=control_limit)

    def _public_joint_position(self, binding: Any, public_name: str) -> float:
        """读取公共关节当前位置，供接触约束轨迹锁存本命令涉及的资源。"""
        internal = f"{binding.prefix}{public_name}"
        joint_id = self._id(self.mj.mjtObj.mjOBJ_JOINT, internal)
        return float(self.data.qpos[int(self.model.jnt_qposadr[joint_id])])

    def _base_pose(self, binding: Any, mapping: RobotMapping) -> Pose:
        """读取承载完整底盘变换的 MuJoCo body 世界位姿。

        R1 Pro 的 x/y/yaw 关节采用嵌套 body 表达，`root_body` 是整棵 Robot
        子树的锚点，而 yaw joint 所属 body 才同时包含三个底盘自由度。这里从
        mapping 声明的 yaw joint 解析该 body，并验证它确实位于 root_body 子树。
        这样既保留场景给 root 的 1cm 高度，也不会在移动后丢失 y 或 yaw。
        """
        root_id = self._id(self.mj.mjtObj.mjOBJ_BODY, f"{binding.prefix}{mapping.root_body}")
        yaw_joint = f"{binding.prefix}{mapping.base_joints['yaw_joint']}"
        yaw_joint_id = self._id(self.mj.mjtObj.mjOBJ_JOINT, yaw_joint)
        base_body_id = int(self.model.jnt_bodyid[yaw_joint_id])
        current = base_body_id
        while current != root_id and current > 0:
            current = int(self.model.body_parentid[current])
        if current != root_id:
            raise ValidationRuntimeError(
                "底盘 yaw joint 不在 Robot root_body 子树内",
                details={"robot_id": binding.robot_id, "root_body": mapping.root_body},
            )
        return Pose(
            position=tuple(float(value) for value in self.data.xpos[base_body_id]),
            quaternion_xyzw=_xyzw_from_wxyz(self.data.xquat[base_body_id]),
            frame_id=mapping.coordinate_frame,
        )

    def _site_pose(self, name: str) -> Pose:
        site_id = self._id(self.mj.mjtObj.mjOBJ_SITE, name)
        quaternion = np.zeros(4)
        self.mj.mju_mat2Quat(quaternion, self.data.site_xmat[site_id])
        return Pose(
            position=tuple(float(value) for value in self.data.site_xpos[site_id]),
            quaternion_xyzw=_xyzw_from_wxyz(quaternion),
        )

    def _ensure_offscreen_size(self, width: int, height: int) -> None:
        """按需扩展离屏 framebuffer；扩容前释放旧 Renderer 上下文。"""
        render_model, _ = self._ensure_render_state()
        visual = render_model.vis.global_
        target_width = max(int(visual.offwidth), width)
        target_height = max(int(visual.offheight), height)
        if target_width == int(visual.offwidth) and target_height == int(visual.offheight):
            return
        for renderer in self._renderers.values():
            # 已失效的 OpenGL 上下文不能阻止后续用更大 framebuffer 重建。
            with suppress(Exception):
                renderer.close()
        self._renderers.clear()
        visual.offwidth = target_width
        visual.offheight = target_height

    def _render(self, camera: str, width: int, height: int, *, depth: bool) -> np.ndarray:
        render_model, render_data = self._prepare_render_state()
        self._ensure_offscreen_size(width, height)
        key = (width, height)
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = self.mj.Renderer(render_model, height=height, width=width)
            self._renderers[key] = renderer
        renderer.enable_depth_rendering() if depth else renderer.disable_depth_rendering()
        renderer.update_scene(render_data, camera=camera)
        return np.asarray(renderer.render()).copy()

    def _ensure_render_state(self) -> tuple[Any, Any]:
        """在渲染线程中延迟创建独立的 MuJoCo 模型和状态。

        Runtime 默认只有一个 RenderExecutor 调用本方法，因此 Renderer、渲染模型
        和 OpenGL Context 都不会跨线程。独立模型还避免实验性抓取辅助修改物理模型
        的碰撞掩码时，与 Viewer 同时读取同一块模型内存。
        """
        if self._render_model is None:
            self._render_model = self.mj.MjModel.from_xml_path(str(self.build.xml_path))
            self._render_data = self.mj.MjData(self._render_model)
            self.mj.mj_forward(self._render_model, self._render_data)
        assert self._render_data is not None
        return self._render_model, self._render_data

    def _prepare_render_state(self) -> tuple[Any, Any]:
        """快速复制物理状态，再在物理锁外准备一帧只读渲染状态。

        锁内只复制数值数组；`mj_forward`、OpenGL 渲染和图片编码全部在锁外
        完成。渲染来不及时 LatestFrameHub 会丢弃旧帧，而物理循环继续推进。
        """
        render_model, render_data = self._ensure_render_state()
        fields = (
            "qpos",
            "qvel",
            "act",
            "mocap_pos",
            "mocap_quat",
            "userdata",
            "ctrl",
            "qfrc_applied",
            "xfrc_applied",
            "eq_active",
            "plugin_state",
        )
        with self._lock:
            state = {
                name: np.asarray(getattr(self.data, name)).copy()
                for name in fields
                if hasattr(self.data, name)
            }
            sim_time = float(self.data.time)
        for name, values in state.items():
            target = getattr(render_data, name, None)
            if target is not None and np.shape(target) == np.shape(values):
                target[...] = values
        render_data.time = sim_time
        self.mj.mj_forward(render_model, render_data)
        return render_model, render_data

    def _public_contacts(self, binding: Any) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            names = [
                self._public_body_for_geom(int(contact.geom1)),
                self._public_body_for_geom(int(contact.geom2)),
            ]
            if not names[0] or not names[1]:
                continue
            pair = tuple(sorted((names[0], names[1])))
            if pair in seen:
                continue
            if binding.prefix and not any(value.startswith(binding.prefix) for value in names):
                continue
            seen.add(pair)
            result.append({"first": names[0], "second": names[1]})
        return result

    def _public_body_for_geom(self, geom_id: int) -> str | None:
        # 模型加载后geom/body层级不会改变。密集场景若每步为每个接触重新调用
        # mj_id2name并遍历父链，Python映射成本会超过mj_step。这里只缓存静态
        # 身份；接触、力和速度仍逐物理步读取，物理语义和时间精度不变。
        if geom_id in self._public_body_cache:
            return self._public_body_cache[geom_id]
        body_id = int(self.model.geom_bodyid[geom_id])
        while body_id > 0:
            raw = self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY, body_id)
            if raw in self.build.object_body_names:
                self._public_body_cache[geom_id] = raw
                return raw
            for binding in self.build.robots:
                if raw and (not binding.prefix or raw.startswith(binding.prefix)):
                    public_name = (
                        f"{binding.robot_id}:{self._strip_prefix(raw, binding.prefix)}"
                    )
                    self._public_body_cache[geom_id] = public_name
                    return public_name
            body_id = int(self.model.body_parentid[body_id])
        self._public_body_cache[geom_id] = None
        return None

    def _geom_name(self, geom_id: int) -> str:
        """返回 Scene Package 声明的碰撞几何名称。"""
        return str(self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_GEOM, geom_id) or "")

    def _insert_contact_geometry(
        self,
        robot_id: str,
        mapping: RobotMapping,
        tool_refs: list[str],
    ) -> list[str]:
        """返回 insert 失败瞬间工具碰到的箱体几何，供上层定位真实阻挡。"""
        selected = {
            body
            for tool in mapping.gripper_tools.values()
            if tool.tool_ref in tool_refs
            for body in (*tool.hook_bodies, *tool.clamp_bodies)
        }
        result: set[str] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom_ids = (int(contact.geom1), int(contact.geom2))
            names = tuple(self._public_body_for_geom(geom_id) for geom_id in geom_ids)
            object_index = next(
                (
                    offset
                    for offset, name in enumerate(names)
                    if name in self.build.object_body_names
                ),
                None,
            )
            if object_index is None:
                continue
            robot_name = names[1 - object_index]
            if not robot_name or not robot_name.startswith(f"{robot_id}:"):
                continue
            robot_geom_id = geom_ids[1 - object_index]
            part = robot_name.split(":", 1)[1]
            if part not in selected:
                continue
            x, y, z = (float(value) for value in contact.pos)
            result.add(
                f"{part}[{self._geom_name(robot_geom_id) or 'unnamed-geom'}]"
                f"->{self._geom_name(geom_ids[object_index]) or 'unnamed-geom'}"
                f"@({x:.4f},{y:.4f},{z:.4f})"
            )
        return sorted(result)

    def _object_has_external_contact(self, source_id: str, robot_id: str) -> bool:
        """判断载荷是否仍与当前 Robot 以外的物理实体接触。"""

        for index in range(int(getattr(self.data, "ncon", 0))):
            contact = self.data.contact[index]
            names = (
                self._public_body_for_geom(int(contact.geom1)),
                self._public_body_for_geom(int(contact.geom2)),
            )
            if source_id not in names:
                continue
            other = names[1] if names[0] == source_id else names[0]
            if other is None or not other.startswith(f"{robot_id}:"):
                return True
        return False

    def _object_in_contact(self, source_id: str) -> bool:
        return any(
            source_id
            in {
                self._public_body_for_geom(int(self.data.contact[index].geom1)),
                self._public_body_for_geom(int(self.data.contact[index].geom2)),
            }
            for index in range(self.data.ncon)
        )

    def _detect_holding(self) -> None:
        """只根据同一物体与同侧两根夹指的真实接触更新 Holding。

        Holding 是传感器观察结果，不是抓取执行器。这里不得关闭碰撞、搬动物体
        或根据距离猜测抓取，否则上层会把并未发生的物理抓取误判为成功。
        """
        # 专用工具已经由下方基于Profile的采样覆盖Holding。若当前所有Robot都
        # 声明了工具，再扫描一遍历史双指命名不仅没有语义价值，还会在密集场景
        # 每1ms重复遍历全部接触。混合或普通夹爪场景继续保留原路径。
        if self._holding and all(
            self._profiles.get(robot_id) is not None
            and self._profiles[robot_id].gripper_tools
            for robot_id in self._holding
        ):
            self._detect_tote_tool_contacts()
            return
        contacts: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set))
        )
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            names = [
                self._public_body_for_geom(int(contact.geom1)),
                self._public_body_for_geom(int(contact.geom2)),
            ]
            object_id = next((name for name in names if name in self.build.object_body_names), None)
            robot_part = next((name for name in names if name and ":" in name), None)
            if not object_id or not robot_part or "gripper_finger" not in robot_part:
                continue
            robot_id, part = robot_part.split(":", 1)
            if "left_" in part:
                side = "left"
            elif "right_" in part:
                side = "right"
            else:
                continue
            if "finger_joint1" in part or "finger_link1" in part:
                finger = "finger_1"
            elif "finger_joint2" in part or "finger_link2" in part:
                finger = "finger_2"
            else:
                continue
            contacts[robot_id][side][object_id].add(finger)

        for robot_id in self._holding:
            self._holding[robot_id] = next(
                (
                    object_id
                    for side_contacts in contacts[robot_id].values()
                    for object_id, fingers in side_contacts.items()
                    if {"finger_1", "finger_2"}.issubset(fingers)
                ),
                None,
            )

        self._detect_tote_tool_contacts()

    def _detect_tote_tool_contacts(self) -> None:
        """采集夹具当前接触、法向力、接触方向和切向相对速度。

        Runtime 不累计物理帧，也不输出 stable_load、slipping 或 holding_object。
        这些结论需要真实时间窗口和任务期望，只能由 Ability 根据连续采样判断。
        contact_object 仅供本 Runtime 的关节前馈控制使用，不进入公共状态。
        """
        contacts: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: {
                        "hook_contact": False,
                        "clamp_contact": False,
                        "hook_force_n": 0.0,
                        "clamp_force_n": 0.0,
                        "hook_support_ratio": 0.0,
                        "hook_tangential_speed_m_s": 0.0,
                    }
                )
            )
        )
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom_ids = [int(contact.geom1), int(contact.geom2)]
            names = [self._public_body_for_geom(geom_id) for geom_id in geom_ids]
            object_id = next((name for name in names if name in self.build.object_body_names), None)
            robot_part = next((name for name in names if name and ":" in name), None)
            if not object_id or not robot_part:
                continue
            robot_id, part = robot_part.split(":", 1)
            mapping = self._profiles.get(robot_id)
            if mapping is None or not mapping.gripper_tools:
                continue
            normal_force = self._contact_normal_force(index, contact)
            support_ratio = self._contact_vertical_ratio(contact)
            relative_speed = self._contact_relative_speed(contact)
            for side, tool in mapping.gripper_tools.items():
                metrics = contacts[robot_id][side][object_id]
                if part in tool.hook_bodies:
                    metrics["hook_contact"] = True
                    metrics["hook_force_n"] = max(metrics["hook_force_n"], normal_force)
                    metrics["hook_support_ratio"] = max(
                        metrics["hook_support_ratio"], support_ratio
                    )
                    metrics["hook_tangential_speed_m_s"] = max(
                        metrics["hook_tangential_speed_m_s"], relative_speed
                    )
                if part in tool.clamp_bodies:
                    metrics["clamp_contact"] = True
                    metrics["clamp_force_n"] = max(metrics["clamp_force_n"], normal_force)
                # 非承重的 hook 侧碰和 clamp 接触不能写入滑移速度，否则正常
                # 载荷接管会被误报为脱钩。

        for robot_id, mapping in self._profiles.items():
            if not mapping.gripper_tools:
                continue
            reported: dict[str, dict[str, Any]] = {}
            load_support_objects: list[str] = []
            for side, tool in mapping.gripper_tools.items():
                candidates = [
                    (object_id, metrics)
                    for object_id, metrics in contacts[robot_id][side].items()
                    if metrics["hook_contact"] or metrics["clamp_contact"]
                ]
                contact_object, metrics = (
                    sorted(
                        candidates,
                        key=lambda item: (
                            -int(item[1]["hook_contact"]) - int(item[1]["clamp_contact"]),
                            -max(item[1]["hook_force_n"], item[1]["clamp_force_n"]),
                            item[0],
                        ),
                    )[0]
                    if candidates
                    else (None, {})
                )
                reported[side] = {
                    "hook_contact": bool(metrics.get("hook_contact")),
                    "clamp_contact": bool(metrics.get("clamp_contact")),
                    "hook_force_n": float(metrics.get("hook_force_n", 0.0)),
                    "clamp_force_n": float(metrics.get("clamp_force_n", 0.0)),
                    "hook_support_ratio": float(metrics.get("hook_support_ratio", 0.0)),
                    "hook_tangential_speed_m_s": float(
                        metrics.get("hook_tangential_speed_m_s", 0.0)
                    ),
                    "contact_object": contact_object,
                }
                if (
                    contact_object
                    and metrics.get("hook_contact")
                    and metrics.get("clamp_contact")
                    and float(metrics.get("hook_support_ratio", 0.0)) >= 0.65
                    and max(
                        float(metrics.get("hook_force_n", 0.0)),
                        float(metrics.get("clamp_force_n", 0.0)),
                    )
                    <= tool.peak_force_n
                ):
                    load_support_objects.append(contact_object)
            self._tool_contacts[robot_id] = reported
            self._holding[robot_id] = (
                load_support_objects[0]
                if len(load_support_objects) == len(mapping.gripper_tools)
                and len(set(load_support_objects)) == 1
                else None
            )
            gripper_targets = self._gripper_targets.get(robot_id, {})
            all_tools_commanded_to_hold = all(
                bool(gripper_targets.get(side, {}).get("contact_latched"))
                for side in mapping.gripper_tools
            )
            if (
                self._holding[robot_id] is not None
                and all_tools_commanded_to_hold
            ):
                # 显式开夹后，压片离开箱壁前仍会短暂保留真实接触。这个瞬时
                # 接触可以继续上报给Ability，但不能重新建立已经清除的双侧
                # 承载锁存；否则下一段退钩动作完成时，通用hold会恢复到开夹
                # 前的旧上臂姿态，把已经抬高的空钩重新拉回箱沿。只有两侧
                # 低层控制目标本身都仍锁存在接触位置时，才允许续写稳定姿态。
                self._load_support_latched[robot_id] = True
                binding = self._binding(robot_id)
                excluded = {
                    f"{binding.prefix}{name}"
                    for name in (
                        list(mapping.base_joints.values())
                        + [joint for joints in mapping.grippers.values() for joint in joints]
                    )
                }
                self._stable_upper_body_positions[robot_id] = {
                    name: position
                    for name, position in self._joint_positions(binding, mapping).items()
                    if name not in excluded
                }

    def _contact_normal_force(self, index: int, contact: Any) -> float:
        """读取 MuJoCo 接触坐标系中的实际法向力。"""
        force = np.zeros(6, dtype=float)
        self.mj.mj_contactForce(self.model, self.data, index, force)
        return abs(float(force[0]))

    @staticmethod
    def _contact_vertical_ratio(contact: Any) -> float:
        """返回接触法向的竖直分量比例，用于识别真正承重的钩爪接触。"""
        normal = np.asarray(contact.frame[:3], dtype=float)
        norm = float(np.linalg.norm(normal))
        return abs(float(normal[2])) / norm if norm > 1e-9 else 0.0

    def _contact_relative_speed(self, contact: Any) -> float:
        """计算接触点沿接触平面的相对速度，避免移动中的脱钩被误报稳定。

        MuJoCo ``cvel``/``mj_objectVelocity`` 给出的是刚体参考点速度。双臂抬升
        时夹具和箱体会有微小转动；直接比较两个 body 参考点的线速度，会把
        ``omega × radius`` 产生的正常差异误报成接触面滑移。这里把两侧速度
        都平移到同一个真实接触点，再投影到接触平面。

        沿法向的相对速度是压紧、回弹或分离，不是滑移。它已经通过接触存在、
        接触力和稳定窗口参与承载判断；若再次计入 slip，双侧夹具开始抬升时
        正常的弹性压缩会被随机一侧误报为滑移并触发 hold。真正沿箱沿运动的
        切向速度仍按 Profile 阈值立即上报，安全语义没有放宽。
        """
        body_a = int(self.model.geom_bodyid[int(contact.geom1)])
        body_b = int(self.model.geom_bodyid[int(contact.geom2)])
        position = np.asarray(contact.pos, dtype=float)
        velocity_a = self._body_point_velocity(body_a, position)
        velocity_b = self._body_point_velocity(body_b, position)
        relative = velocity_a - velocity_b
        normal = np.asarray(contact.frame[:3], dtype=float)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-9:
            return float(np.linalg.norm(relative))
        unit_normal = normal / normal_norm
        tangential = relative - unit_normal * float(np.dot(relative, unit_normal))
        return float(np.linalg.norm(tangential))

    def _body_point_velocity(self, body_id: int, position: np.ndarray) -> np.ndarray:
        velocity = np.zeros(6, dtype=float)
        self.mj.mj_objectVelocity(
            self.model,
            self.data,
            self.mj.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        angular = velocity[:3]
        linear = velocity[3:]
        origin = np.asarray(self.data.xpos[body_id], dtype=float)
        return linear + np.cross(angular, position - origin)

    def _gripper_states(
        self, binding: Any, mapping: RobotMapping, joints: dict[str, JointState]
    ) -> dict[str, GripperState]:
        """聚合夹具关节、力限制和真实接触反馈；不推断抓取 Skill 成功。"""
        result: dict[str, GripperState] = {}
        for side, joint_names in mapping.grippers.items():
            positions = [joints[name].position for name in joint_names]
            velocities = [joints[name].velocity for name in joint_names]
            tool = mapping.gripper_tools.get(side)
            effort = max(abs(float(joints[name].effort or 0.0)) for name in joint_names)
            if tool:
                effort *= tool.control_force_scale_n_per_unit
            target = self._gripper_targets.get(binding.robot_id, {}).get(side)
            position = sum(positions) / len(positions)
            contact = self._tool_contacts.get(binding.robot_id, {}).get(side, {})
            result[side] = GripperState(
                position=position,
                velocity=sum(velocities) / len(velocities),
                effort=effort,
                target_position=target["position"] if target else None,
                reached_target=target is not None and abs(position - target["position"]) <= 0.002,
                hook_contact=bool(contact.get("hook_contact")),
                clamp_contact=bool(contact.get("clamp_contact")),
                hook_force_n=float(contact.get("hook_force_n", 0.0)),
                clamp_force_n=float(contact.get("clamp_force_n", 0.0)),
                hook_support_ratio=float(contact.get("hook_support_ratio", 0.0)),
                hook_tangential_speed_m_s=float(contact.get("hook_tangential_speed_m_s", 0.0)),
                sensor_fault=False,
            )
        return result

    def _apply_holds(self) -> None:
        # 运动期间也必须稳定未参与命令的关节；in_hold 只表示 Robot 是否
        # 接受活动命令，不表示关闭底层关节保持控制。
        for targets in self._hold_targets.values():
            for internal, target in targets.items():
                self._drive_joint(
                    internal,
                    target,
                    control_limit=self._hold_effort_limits.get(internal),
                )

    def _joint_positions(self, binding: Any, mapping: RobotMapping) -> dict[str, float]:
        result: dict[str, float] = {}
        for public_name in mapping.joint_names:
            internal = f"{binding.prefix}{public_name}"
            joint_id = self._id(self.mj.mjtObj.mjOBJ_JOINT, internal)
            result[internal] = float(self.data.qpos[int(self.model.jnt_qposadr[joint_id])])
        return result

    def _id(self, kind: Any, name: str) -> int:
        value = self._optional_id(kind, name)
        if value < 0:
            raise ValidationRuntimeError(f"模型元素不存在: {name}")
        return value

    def _optional_id(self, kind: Any, name: str) -> int:
        # MuJoCo模型在Scene Instance生命周期内不可变。1kHz控制循环不应为
        # 同一个关节、执行器反复扫描名称表；缓存包含-1，缺失执行器也只查一次。
        key = (int(kind), name)
        value = self._model_object_ids.get(key)
        if value is None:
            value = int(self.mj.mj_name2id(self.model, kind, name))
            self._model_object_ids[key] = value
        return value

    @staticmethod
    def _strip_prefix(name: str, prefix: str) -> str:
        return name[len(prefix) :] if prefix and name.startswith(prefix) else name


def _segment(
    points: list[dict[str, Any]], elapsed: float
) -> tuple[dict[str, Any], dict[str, Any], float, bool]:
    """返回轨迹当前区间和插值比例；轨迹合法性已在 API 模型中检查。"""
    if elapsed >= float(points[-1]["time_from_start"]):
        return points[-1], points[-1], 1.0, True
    for previous, current in zip(points, points[1:], strict=False):
        end = float(current["time_from_start"])
        if elapsed <= end:
            begin = float(previous["time_from_start"])
            ratio = 0.0 if end == begin else (elapsed - begin) / (end - begin)
            return previous, current, max(0.0, min(1.0, ratio)), False
    return points[0], points[0], 0.0, False


def _interpolate_joints(
    points: list[dict[str, Any]], elapsed: float
) -> tuple[dict[str, float], bool]:
    previous, current, ratio, finished = _segment(points, elapsed)
    return {
        name: float(previous["positions"][name])
        + (float(current["positions"][name]) - float(previous["positions"][name])) * ratio
        for name in previous["positions"]
    }, finished


def _interpolate_base(
    points: list[dict[str, Any]], elapsed: float
) -> tuple[dict[str, float], bool]:
    previous, current, ratio, finished = _segment(points, elapsed)
    result = {
        name: float(previous[name]) + (float(current[name]) - float(previous[name])) * ratio
        for name in ("x", "y")
    }
    yaw_delta = _wrap_angle(float(current["yaw"]) - float(previous["yaw"]))
    # MuJoCo 的无限位 yaw joint 保存连续角度。轨迹越过 +pi 时如果把插值结果
    # 折回 -pi，位置控制器会看到约 2*pi 的瞬时反向误差，Robot 会在转角处
    # 被猛然掀倒。这里只把相邻点的差值归一到最短转向，绝对目标保持连续。
    result["yaw"] = float(previous["yaw"]) + yaw_delta * ratio
    return result, finished


def _xyzw_from_wxyz(values: Any) -> tuple[float, float, float, float]:
    """MuJoCo 内部 wxyz 只在此边界转换，公共模型不接受或输出旧顺序。"""
    w, x, y, z = (float(value) for value in values)
    return (x, y, z, w)


def _clip(value: float, maximum: float) -> float:
    return max(-maximum, min(maximum, float(value)))


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def _nearest_equivalent_angle(reference: float, target: float) -> float:
    """把目标角映射到离连续物理关节当前位置最近的等价角。"""

    return reference + _wrap_angle(target - reference)


def _quat_yaw(quaternion: tuple[float, float, float, float]) -> float:
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
