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

"""R1 Pro 公共 Profile 到 MuJoCo 名称的唯一映射。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from plugin_mujoco.errors import ValidationRuntimeError


@dataclass(frozen=True)
class GripperToolMapping:
    """专用夹具机械边界；接触稳定性与抓取成功不属于 Runtime。"""

    tool_ref: str
    frame: str
    joint: str
    kind: str
    software_min_position_m: float
    software_max_position_m: float
    normal_force_n: float
    peak_force_n: float
    control_force_scale_n_per_unit: float
    hook_bodies: tuple[str, ...]
    clamp_bodies: tuple[str, ...]
    minimum_clamp_force_n: float = 0.0


@dataclass(frozen=True)
class JointControlGains:
    """Robot 型号为一个关节组声明的低层位置控制参数。"""

    kp: float
    kd: float


@dataclass(frozen=True)
class RobotMapping:
    model: str
    kind: str
    coordinate_frame: str
    sdk_package: str
    backend: str
    backend_profile: str
    urdf_path: Path | None
    kinematic_root_frame: str
    root_body: str
    base_joints: dict[str, str]
    joint_groups: dict[str, tuple[str, ...]]
    joint_control: dict[str, JointControlGains]
    end_effectors: dict[str, str]
    grippers: dict[str, tuple[str, ...]]
    gripper_tools: dict[str, GripperToolMapping]
    sensors: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> RobotMapping:
        if not path.is_file():
            raise ValidationRuntimeError(f"缺少 Robot Profile 映射: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValidationRuntimeError(f"无法读取 Robot Profile: {path}") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValidationRuntimeError(f"Robot Profile schema_version 必须为 1: {path}")
        base = raw.get("base") or {}
        required_base = {"x_joint", "y_joint", "yaw_joint"}
        if set(base) != required_base:
            raise ValidationRuntimeError(f"Robot Profile base 必须包含 {sorted(required_base)}")
        kinematics = raw.get("kinematics") or {}
        kinematic_root_frame = str(kinematics.get("root_frame", "base_link")).strip()
        if not kinematic_root_frame:
            raise ValidationRuntimeError("Robot Profile kinematics.root_frame 不能为空")
        joint_groups = _tuple_mapping(raw.get("joint_groups"))
        grippers = _tuple_mapping(raw.get("grippers"))
        joint_control = _joint_control_mapping(
            raw.get("joint_control"), control_groups={**joint_groups, **grippers}
        )
        gripper_tools = _gripper_tool_mapping(raw.get("gripper_tools"), grippers=grippers)
        raw_urdf = str(kinematics.get("urdf", "")).strip()
        # kinematics.urdf 相对 Robot 资产根目录，而不是 config 目录。这里只
        # 解析路径并向 Framework 报告；Runtime 仍由 MJCF 驱动物理，不读取
        # URDF 做另一套运动学。
        urdf_path = (path.parent.parent / raw_urdf).resolve() if raw_urdf else None
        return cls(
            model=str(raw["model"]),
            kind=str(raw["kind"]),
            coordinate_frame=str(raw.get("coordinate_frame", "world")),
            sdk_package=str(raw.get("sdk_package", "semantic-robot-sdk-r1pro")),
            backend=str(raw.get("backend", "mujoco")),
            backend_profile=str(raw.get("backend_profile", "native-mujoco")),
            urdf_path=urdf_path,
            kinematic_root_frame=kinematic_root_frame,
            root_body=str(raw["root_body"]),
            base_joints={str(key): str(value) for key, value in base.items()},
            joint_groups=joint_groups,
            joint_control=joint_control,
            end_effectors={
                str(key): str(value) for key, value in (raw.get("end_effectors") or {}).items()
            },
            grippers=grippers,
            gripper_tools=gripper_tools,
            sensors={str(key): str(value) for key, value in (raw.get("sensors") or {}).items()},
        )

    @property
    def command_joint_names(self) -> tuple[str, ...]:
        """只返回 joint_trajectory 可占用的 torso/arm 关节。

        底盘和夹爪分别由 BaseTrajectory/GripperCommand 控制，不能混入 Robot SDK
        对关节轨迹能力的精确校验清单。
        """
        return tuple(name for group in self.joint_groups.values() for name in group)

    @property
    def controlled_joint_names(self) -> tuple[str, ...]:
        """返回 Runtime 位置控制器实际调节的手臂、躯干和夹具关节。"""
        values = list(self.command_joint_names)
        for group in self.grippers.values():
            values.extend(group)
        return tuple(dict.fromkeys(values))

    @property
    def joint_names(self) -> tuple[str, ...]:
        values = list(self.base_joints.values())
        for group in self.joint_groups.values():
            values.extend(group)
        for group in self.grippers.values():
            values.extend(group)
        return tuple(dict.fromkeys(values))

    def control_gains(self, public_joint_name: str) -> JointControlGains:
        """返回型号配置的关节组增益；未声明的型号沿用稳定默认值。"""
        # joint_groups 是轨迹能力，grippers 是工具能力；两者可以共享低层位置
        # 控制参数，但不能为了配置夹具增益而把工具伪装成轨迹关节。
        for groups in (self.joint_groups, self.grippers):
            for group, joints in groups.items():
                if public_joint_name in joints:
                    return self.joint_control.get(
                        group, JointControlGains(kp=150.0, kd=15.0)
                    )
        return JointControlGains(kp=150.0, kd=15.0)

    def validate_model_names(self, names: set[str], *, prefix: str) -> None:
        required = {
            f"{prefix}{self.root_body}",
            *(f"{prefix}{name}" for name in self.joint_names),
            *(f"{prefix}{name}" for name in self.end_effectors.values()),
            *(f"{prefix}{name}" for name in self.sensors.values()),
            *(
                f"{prefix}{name}"
                for tool in self.gripper_tools.values()
                for name in (tool.frame, tool.joint, *tool.hook_bodies, *tool.clamp_bodies)
            ),
        }
        missing = sorted(required - names)
        if missing:
            raise ValidationRuntimeError(
                "Robot Profile 与 MuJoCo 模型不一致",
                details={"model": self.model, "missing": missing},
            )

    def validate_gripper_target(
        self, side: str, position: float, max_effort: float | None
    ) -> GripperToolMapping | None:
        """校验低层夹具目标，并返回可选的专用工具标定。"""
        if side not in self.grippers:
            raise ValidationRuntimeError(f"Robot Profile 中没有夹爪: {side}")
        tool = self.gripper_tools.get(side)
        if tool is None:
            return None
        if not tool.software_min_position_m <= position <= tool.software_max_position_m:
            raise ValidationRuntimeError(
                "夹具位置超过软件行程",
                details={
                    "gripper_id": side,
                    "position": position,
                    "software_range_m": [
                        tool.software_min_position_m,
                        tool.software_max_position_m,
                    ],
                },
            )
        if max_effort is not None and max_effort > tool.peak_force_n:
            raise ValidationRuntimeError(
                "夹具力限制超过机械峰值",
                details={
                    "gripper_id": side,
                    "max_effort": max_effort,
                    "peak_force_n": tool.peak_force_n,
                },
            )
        return tool


def _tuple_mapping(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): tuple(str(item) for item in items)
        for key, items in value.items()
        if isinstance(items, list)
    }


def _joint_control_mapping(
    value: Any, *, control_groups: dict[str, tuple[str, ...]]
) -> dict[str, JointControlGains]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValidationRuntimeError("Robot Profile joint_control 必须是对象")
    unknown = sorted(set(map(str, value)) - set(control_groups))
    if unknown:
        raise ValidationRuntimeError(
            "joint_control 引用了未声明的控制分组", details={"groups": unknown}
        )
    result: dict[str, JointControlGains] = {}
    for group, item in value.items():
        if not isinstance(item, dict):
            raise ValidationRuntimeError(f"joint_control.{group} 必须是对象")
        kp = float(item.get("kp", 0.0))
        kd = float(item.get("kd", 0.0))
        if kp <= 0 or kd <= 0:
            raise ValidationRuntimeError(
                f"joint_control.{group} 的 kp/kd 必须大于 0"
            )
        result[str(group)] = JointControlGains(kp=kp, kd=kd)
    return result


def _gripper_tool_mapping(
    value: Any, *, grippers: dict[str, tuple[str, ...]]
) -> dict[str, GripperToolMapping]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValidationRuntimeError("Robot Profile gripper_tools 必须是对象")
    unknown = sorted(set(map(str, value)) - set(grippers))
    if unknown:
        raise ValidationRuntimeError(
            "gripper_tools 引用了未声明的夹爪", details={"grippers": unknown}
        )
    result: dict[str, GripperToolMapping] = {}
    for side, item in value.items():
        if not isinstance(item, dict):
            raise ValidationRuntimeError(f"gripper_tools.{side} 必须是对象")
        tool_ref = str(item.get("tool_ref", f"component://tool/{side}")).strip()
        frame = str(item.get("frame", "")).strip()
        joint = str(item.get("joint", "")).strip()
        if not tool_ref or not frame or not joint:
            raise ValidationRuntimeError(
                f"gripper_tools.{side} 必须声明 tool_ref、frame 和 joint"
            )
        if joint not in grippers[str(side)]:
            raise ValidationRuntimeError(f"gripper_tools.{side}.joint 不属于该工具关节")
        software_range = item.get("software_range_m")
        if (
            not isinstance(software_range, list)
            or len(software_range) != 2
            or not all(isinstance(number, int | float) for number in software_range)
        ):
            raise ValidationRuntimeError(f"gripper_tools.{side}.software_range_m 必须包含两个数")
        minimum, maximum = map(float, software_range)
        normal_force = float(item.get("normal_force_n", 0.0))
        peak_force = float(item.get("peak_force_n", 0.0))
        force_scale = float(item.get("control_force_scale_n_per_unit", 1.0))
        minimum_clamp_force = float(item.get("minimum_clamp_force_n", 0.0))
        hook_bodies = _string_tuple(item.get("hook_bodies"))
        clamp_bodies = _string_tuple(item.get("clamp_bodies"))
        if minimum < 0 or maximum <= minimum:
            raise ValidationRuntimeError(f"gripper_tools.{side} 软件行程无效")
        if (
            normal_force <= 0
            or peak_force < normal_force
            or force_scale <= 0
            or minimum_clamp_force < 0
            or minimum_clamp_force > peak_force
        ):
            raise ValidationRuntimeError(f"gripper_tools.{side} 力参数无效")
        if str(item.get("kind", "")) == "tote_clamp" and (not hook_bodies or not clamp_bodies):
            raise ValidationRuntimeError(
                f"gripper_tools.{side} 的 tote_clamp 必须声明 hook_bodies 和 clamp_bodies"
            )
        result[str(side)] = GripperToolMapping(
            tool_ref=tool_ref,
            frame=frame,
            joint=joint,
            kind=str(item.get("kind", "generic")),
            software_min_position_m=minimum,
            software_max_position_m=maximum,
            normal_force_n=normal_force,
            peak_force_n=peak_force,
            control_force_scale_n_per_unit=force_scale,
            hook_bodies=hook_bodies,
            clamp_bodies=clamp_bodies,
            minimum_clamp_force_n=minimum_clamp_force,
        )
    return result


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValidationRuntimeError("Robot Profile body 列表必须由非空字符串组成")
    return tuple(value)
