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

"""robosuite 1.5 的 Lift / Stack 独立运行适配。"""

from __future__ import annotations

from importlib.metadata import version
from typing import Any, Dict, Iterable, Sequence, Tuple

from semantic_sim_profiles.pose import read_root_body_pose
from semantic_sim_profiles.visual_source import public_source_for_body

FRANKA_JOINT_NAMES = tuple("panda_joint" + str(index) for index in range(1, 8))


class RobosuiteAdapter:
    def __init__(
        self,
        environment: str,
        *,
        camera_names: Sequence[str],
        width: int,
        height: int,
        horizon: int,
    ) -> None:
        if environment not in {"Lift", "Stack"}:
            raise ValueError("robosuite profile 只支持 Lift 或 Stack")
        import robosuite as suite

        self._environment = environment
        self._version = version("robosuite")
        self._last_info: Dict[str, Any] = {}
        self._last_observation: Dict[str, Any] = {}
        # Robot SDK 提交的是已规划的绝对关节轨迹。这里明确选择绝对位置控制器，
        # 避免把相同轨迹解释成 OSC 增量；action array 只保留在 Profile 内部。
        joint_controller = suite.load_part_controller_config(default_controller="JOINT_POSITION")
        joint_controller["input_type"] = "absolute"
        joint_controller["gripper"] = {"type": "GRIP"}
        controller = suite.load_composite_controller_config(robot="Panda")
        controller["body_parts"]["right"] = joint_controller
        self._env = suite.make(
            env_name=environment,
            robots="Panda",
            controller_configs=controller,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=list(camera_names),
            camera_heights=height,
            camera_widths=width,
            camera_depths=True,
            reward_shaping=True,
            control_freq=20,
            horizon=max(horizon, 1),
            ignore_done=False,
        )

    @property
    def language(self) -> str:
        return {"Lift": "lift the cube", "Stack": "stack the red cube on the green cube"}[
            self._environment
        ]

    def reset(self, seed: int) -> Dict[str, Any]:
        seeder = getattr(self._env, "seed", None)
        if callable(seeder):
            seeder(seed)
        self._last_observation = _normalize_observation(self._env.reset())
        return dict(self._last_observation)

    def neutral_action(self) -> Any:
        return self.joint_position_action(self.joint_positions(), gripper_action=0.0)

    def joint_positions(self) -> Dict[str, float]:
        """返回 Robot SDK 使用的稳定 Franka 关节名，不暴露 robosuite 前缀。"""

        values = self._env.robots[0]._joint_positions
        return {name: float(values[index]) for index, name in enumerate(FRANKA_JOINT_NAMES)}

    def base_pose(self) -> Dict[str, Any]:
        """读取 Panda 根 body 的当前世界位姿；只能由 Profile worker 调用。"""

        return read_root_body_pose(self._env.robots[0])

    def gripper_opening(self) -> float:
        """返回两根 Panda 夹指之间的实际开度，单位为米。"""

        import numpy as np

        values = np.asarray(self._last_observation.get("robot0_gripper_qpos"), dtype=np.float64)
        if values.size < 2 or not bool(np.isfinite(values[:2]).all()):
            raise RuntimeError("robosuite 未提供有效的 Panda 夹爪位置")
        return float(abs(values[0]) + abs(values[1]))

    def visual_model_data(self) -> Tuple[Any, Any]:
        return self._env.sim.model, self._env.sim.data

    def visual_source_for_body(self, body_id: int, object_source_ids: Iterable[str]) -> str | None:
        return public_source_for_body(
            self._env.sim.model,
            body_id,
            robot_source_id="franka-0",
            object_source_ids=object_source_ids,
        )

    def joint_position_action(self, target: Dict[str, float], *, gripper_action: float) -> Any:
        """把绝对关节目标转换为当前 Profile 的内部 action array。"""

        import numpy as np

        missing = [name for name in FRANKA_JOINT_NAMES if name not in target]
        extra = sorted(set(target).difference(FRANKA_JOINT_NAMES))
        if missing or extra:
            raise ValueError("Franka 关节目标不完整: missing=%s extra=%s" % (missing, extra))
        action = np.asarray(
            [*[float(target[name]) for name in FRANKA_JOINT_NAMES], gripper_action],
            dtype=np.float64,
        )
        robot = self._env.robots[0]
        joint_limits = np.asarray(robot.sim.model.jnt_range)[robot._ref_joint_indexes]
        low, high = self._env.action_spec
        joint_target = action[: len(FRANKA_JOINT_NAMES)]
        if action.shape != low.shape or bool(
            (joint_target < joint_limits[:, 0]).any()
            or (joint_target > joint_limits[:, 1]).any()
            or gripper_action < low[-1]
            or gripper_action > high[-1]
        ):
            raise ValueError("Franka 关节或夹爪目标超出 robosuite 控制范围")
        return action

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        result = self._env.step(action)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            observation, reward, done, info = result
        self._last_info = _json_scalars(info)
        self._last_observation = _normalize_observation(observation)
        return dict(self._last_observation), float(reward), bool(done), self._last_info

    def success(self) -> bool:
        checker = getattr(self._env, "_check_success", None)
        return bool(checker()) if checker is not None else False

    def contact_state(self) -> Dict[str, Any]:
        """只发布接触汇总，不把 MuJoCo geom id 或内部名称带出 Runtime。"""

        count = int(getattr(self._env.sim.data, "ncon", 0))
        return {
            "active": count > 0,
            "count": count,
            "contacts": [],
            "holding": False,
        }

    def native_metrics(self) -> Dict[str, Any]:
        low, _ = self._env.action_spec
        return {
            "robosuite_version": self._version,
            "environment": self._environment,
            "action_dimension": int(len(low)),
            "last_info": self._last_info,
        }

    def close(self) -> None:
        self._env.close()


def _normalize_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np

    normalized = dict(observation)
    for key, value in list(normalized.items()):
        if hasattr(value, "shape") and (key.endswith("_image") or key.endswith("_depth")):
            normalized[key] = np.asarray(value)[::-1].copy()
    return normalized


def _json_scalars(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_scalars(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_scalars(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
