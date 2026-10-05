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

"""LIBERO 及 LIBERO-Pro 的任务加载、初态和评测适配。"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from semantic_sim_profiles.libero_scene import body_sources, scene_metadata
from semantic_sim_profiles.pose import read_root_body_pose

FRANKA_JOINT_NAMES = tuple("panda_joint" + str(index) for index in range(1, 8))


def installed_task_catalog(source_root: Path, config_root: Path) -> List[Dict[str, Any]]:
    """读取当前上游安装的任务目录，不创建仿真环境或改写 BDDL。

    task_id 必须沿用 benchmark 默认任务顺序，不能按文件名重新排序，否则
    Web 选择的 Spatial task 0 会与模型评测中的 task 0 指向不同任务。
    初态数量直接来自上游文件；缺文件的任务仍可浏览，但不可声明为可启动。
    这里的 available 仅表示任务定义和初态存在，不表示任何策略能完成任务。
    """
    revision = _activate_libero_source(source_root, config_root)
    from libero.libero import benchmark

    registered = benchmark.get_benchmark_dict()
    # 上游还注册了 LIBERO_100 类，但固定版本没有为它定义 task_maps。
    # 采用上游正式 libero_suites 清单，不能把注册表中的辅助类当成已安装套件。
    return _benchmark_catalog(
        {name: registered[name] for name in benchmark.libero_suites}, revision
    )


def _benchmark_catalog(benchmarks: Dict[str, Any], revision: Optional[str]) -> List[Dict[str, Any]]:
    scenes = []
    for suite_name, suite_factory in benchmarks.items():
        suite = suite_factory()
        for task_id in range(suite.get_num_tasks()):
            task = suite.get_task(task_id)
            bddl_path = Path(suite.get_task_bddl_file_path(task_id))
            missing = []
            if not bddl_path.is_file():
                missing.append("任务 BDDL 未安装")
            try:
                count = len(suite.get_task_init_states(task_id))
            except FileNotFoundError:
                count = 0
            if count == 0:
                missing.append("任务初态未安装")
            scenes.append({
                "scene_key": "%s:%s" % (suite_name, task_id),
                "name": task.language,
                "scene_kind": "libero",
                "layouts": ["init-%s" % index for index in range(count)],
                "robot_models": ["franka_panda"],
                "compatible_runtime_profiles": ["libero-robosuite-1.4"],
                "read_only": True,
                "available": not missing,
                "unavailable_reason": "；".join(missing) or None,
                "task": {
                    "suite": suite_name,
                    "task_id": task_id,
                    "name": task.name,
                    "language": task.language,
                    "bddl_file": str(bddl_path),
                    "initial_state_count": count,
                    "source_revision": revision,
                },
            })
    return scenes


class LiberoAdapter:
    def __init__(
        self,
        *,
        suite_name: str,
        task_id: int,
        init_state_id: int,
        seed: int,
        camera_names: Sequence[str],
        width: int,
        height: int,
        horizon: int,
        source_root: Path,
        config_root: Path,
        bddl_file: Optional[Path] = None,
        use_init_state: bool = True,
        controller: str = "JOINT_POSITION",
    ) -> None:
        if controller not in {"JOINT_POSITION", "OSC_POSE"}:
            raise ValueError("LIBERO 控制器必须为 JOINT_POSITION 或 OSC_POSE")
        self.controller_name = controller
        commit = _activate_libero_source(source_root, config_root)
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        package_name = "libero"

        benchmark_dict = benchmark.get_benchmark_dict()
        if suite_name not in benchmark_dict:
            raise ValueError("LIBERO suite 不存在: " + suite_name)
        self._suite_name = suite_name
        self._task_id = task_id
        self._init_state_id = init_state_id
        self._seed = seed
        self._package_name = package_name
        self._version = commit
        self._suite = benchmark_dict[suite_name]()
        self._task = self._suite.get_task(task_id)
        self._language = self._task.language
        self.bddl_path = (
            bddl_file
            or Path(get_libero_path("bddl_files"))
            / self._task.problem_folder
            / self._task.bddl_file
        )
        if not self.bddl_path.is_file():
            raise FileNotFoundError("LIBERO BDDL 不存在: " + str(self.bddl_path))
        self._init_states = self._suite.get_task_init_states(task_id) if use_init_state else None
        if self._init_states is not None and not 0 <= init_state_id < len(self._init_states):
            raise ValueError("init_state_id 超出范围")
        self._last_info: Dict[str, Any] = {}
        self._last_observation: Dict[str, Any] = {}
        self._raw_observation: Dict[str, Any] = {}
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(self.bddl_path),
            controller=controller,
            camera_names=list(camera_names),
            camera_heights=height,
            camera_widths=width,
            camera_depths=True,
            has_renderer=False,
            has_offscreen_renderer=True,
            control_freq=20,
            horizon=max(horizon, 1),
            ignore_done=True,
        )

    @property
    def language(self) -> str:
        return self._language

    def reset(self, seed: int) -> Dict[str, Any]:
        seeder = getattr(self._env, "seed", None)
        if callable(seeder):
            seeder(seed)
        observation = self._env.reset()
        if self._init_states is not None:
            observation = self._env.set_init_state(self._init_states[self._init_state_id])
        # set_init_state 只恢复 MuJoCo 状态，不刷新 OSC 的目标。首次空闲周期
        # 必须保持选中的原生初态，不能向 env.reset 随机生成的旧目标运动。
        self.hold_current()
        self._visual_sources = body_sources(self._env.env)
        self._raw_observation = observation
        self._last_observation = _normalize_observation(observation)
        return dict(self._last_observation)

    def reset_initial_state(self, index: int) -> Dict[str, Any]:
        """预览批处理复用加载后的原生任务，恢复规则与正式启动一致。"""
        if self._init_states is None or not 0 <= index < len(self._init_states):
            raise ValueError("init_state_id 超出范围")
        self._init_state_id = index
        return self.reset(self._seed)

    def policy_observation(self) -> Dict[str, Any]:
        """保留上游相机方向、通道与状态精度；只在仿真线程读取。

        Web 的图像经过竖直翻转，不能再作为 LeRobot 输入。这里返回同一次
        reset/step 的原始两路 RGB、EEF 和夹爪状态，模型侧再按固定配置组成
        8 维状态及归一化。不把模型相关字段拼接放到 Framework 中。
        """
        import numpy as np

        keys = (
            "agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
            "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos",
        )
        return {key: np.array(self._raw_observation[key], copy=True) for key in keys}

    def prepare_control_sequence(self, samples: Sequence[Dict[str, Any]]) -> List[Any]:
        """在 worker 上校验整段控制；无效样本不得执行一半后才被发现。"""
        actions = []
        for sample in samples:
            controls = sample.get("controls", [])
            if len(controls) != 2 or {item.get("group") for item in controls} != {"arm", "hand"}:
                raise ValueError("Franka 序列必须同时指定 arm 和 hand")
            arm = next(item for item in controls if item["group"] == "arm")
            hand = next(item for item in controls if item["group"] == "hand")
            if arm.get("kind") != "end_effector_delta" or arm.get("frame_id") != "world":
                raise ValueError("LIBERO OSC 序列只接受世界坐标末端增量")
            if hand.get("kind") != "gripper_direction":
                raise ValueError("LIBERO OSC 序列必须明确夹爪开合方向")
            actions.append(self.end_effector_delta_action(
                arm["translation_m"], arm["rotation_axis_angle_rad"],
                gripper_action=float(hand["closing_direction"]),
            ))
        return actions

    def hold_current(self) -> None:
        """增量序列结束、取消或重置时捕获实测位姿，舍弃未达的旧增量目标。"""
        if self.controller_name == "OSC_POSE":
            controller = self._env.env.robots[0].controller
            controller.update(force=True)
            controller.reset_goal()

    def end_effector_delta_action(
        self, translation_m: Sequence[float], rotation_axis_angle_rad: Sequence[float],
        *, gripper_action: float,
    ) -> Any:
        """世界坐标的米/弧度增量转换为上游 OSC 归一化输入。

        OSC 默认位置缩放为 ±0.05 m、旋转缩放为 ±0.5 rad，但使用实际
        控制器配置换算，不把这两个数写死。越界返回错误，不能默默裁剪模型
        动作后还宣称与上游执行一致。
        """
        import numpy as np

        if self.controller_name != "OSC_POSE":
            raise ValueError("末端增量要求使用 OSC_POSE 部署配置")
        controller = self._env.env.robots[0].controller
        physical = np.asarray([*translation_m, *rotation_axis_angle_rad], dtype=np.float64)
        if physical.shape != (6,) or not np.isfinite(physical).all():
            raise ValueError("末端增量必须包含有限的三维位移与三维旋转")
        low = np.asarray(controller.output_min)
        high = np.asarray(controller.output_max)
        normalized = controller.input_min + (physical - low) * (
            controller.input_max - controller.input_min
        ) / (high - low)
        action = np.concatenate([normalized, [gripper_action]])
        action_low, action_high = self._env.env.action_spec
        if (
            not np.isfinite(action).all()
            or (action < action_low).any() or (action > action_high).any()
        ):
            raise ValueError("末端增量或夹爪方向超出 LIBERO 控制范围")
        return action

    def hold_action(self) -> Any:
        """无新动作时保持上次控制目标，而不是重复位移或持续追随漂移后的实测值。"""
        if self.controller_name == "JOINT_POSITION":
            return self.joint_position_action(self.joint_positions(), gripper_action=0.0)
        import numpy as np
        from scipy.spatial.transform import Rotation

        controller = self._env.env.robots[0].controller
        # robosuite 的相对 OSC 每次 set_goal 都以实测 EEF 为起点。因此零增量
        # 不能保持旧目标；将旧目标相对当前位姿的误差重新编码，夹爪方向归零。
        controller.update(force=True)
        translation = np.asarray(controller.goal_pos) - controller.ee_pos
        rotation = Rotation.from_matrix(controller.goal_ori @ controller.ee_ori_mat.T).as_rotvec()
        physical = np.clip(
            np.concatenate([translation, rotation]), controller.output_min, controller.output_max
        )
        return self.end_effector_delta_action(physical[:3], physical[3:], gripper_action=0.0)

    def neutral_action(self) -> Any:
        import numpy as np

        target = getattr(self._env, "env", self._env)
        spec = getattr(target, "action_spec", None)
        if spec is None:
            return np.zeros(7, dtype=np.float64)
        low, high = spec
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        action = np.zeros_like(low)
        finite = np.isfinite(low) & np.isfinite(high)
        action[finite] = np.clip(action[finite], low[finite], high[finite])
        return action

    def joint_positions(self) -> Dict[str, float]:
        """返回 Robot SDK 使用的稳定 Franka 关节名。"""

        robot = self._env.env.robots[0]
        return {
            name: float(robot._joint_positions[index])
            for index, name in enumerate(FRANKA_JOINT_NAMES)
        }

    def base_pose(self) -> Dict[str, Any]:
        """读取 Panda 根 body 的当前世界位姿；只能由 Profile worker 调用。"""

        return read_root_body_pose(self._env.env.robots[0])

    def gripper_opening(self) -> float:
        """返回两根 Panda 夹指之间的实际开度，单位为米。"""

        import numpy as np

        values = np.asarray(self._last_observation.get("robot0_gripper_qpos"), dtype=np.float64)
        if values.size < 2 or not bool(np.isfinite(values[:2]).all()):
            raise RuntimeError("LIBERO 未提供有效的 Panda 夹爪位置")
        return float(abs(values[0]) + abs(values[1]))

    def visual_model_data(self) -> Tuple[Any, Any]:
        return self._env.env.sim.model, self._env.env.sim.data

    def visual_geom_groups(self) -> Tuple[int, ...]:
        # 直接复用上游相机的可见分组：LIBERO 的 group 0 是接触代理，不能与
        # group 1 外观同时画到 Web，否则机器人会重影、半透明物体会遮挡操作。
        mask = self._env.env.sim._render_context_offscreen.vopt.geomgroup
        return tuple(index for index, visible in enumerate(mask) if visible)

    def visual_source_for_body(self, body_id: int, object_source_ids: Iterable[str]) -> str | None:
        return self._visual_sources.get(body_id)

    def scene_metadata(self) -> Dict[str, Any]:
        return scene_metadata(self._env.env)

    def joint_position_action(self, target: Dict[str, float], *, gripper_action: float) -> Any:
        """把绝对关节目标转换为 LIBERO 固定控制器的归一化增量。

        LIBERO 1.4 在包装器内部创建控制器，不能像 robosuite 1.5 一样传入
        absolute 配置。因此 Profile 每个控制周期读取实际关节角，将轨迹目标
        变成控制器允许的增量；Robot SDK 和 Ability 仍只看到绝对轨迹。
        """

        import numpy as np

        missing = [name for name in FRANKA_JOINT_NAMES if name not in target]
        extra = sorted(set(target).difference(FRANKA_JOINT_NAMES))
        if missing or extra:
            raise ValueError("Franka 关节目标不完整: missing=%s extra=%s" % (missing, extra))
        robot = self._env.env.robots[0]
        controller = robot.controller
        current = np.asarray(robot._joint_positions, dtype=np.float64)
        desired = np.asarray([float(target[name]) for name in FRANKA_JOINT_NAMES], dtype=np.float64)
        output_min = np.asarray(controller.output_min, dtype=np.float64)
        output_max = np.asarray(controller.output_max, dtype=np.float64)
        input_min = np.asarray(controller.input_min, dtype=np.float64)
        input_max = np.asarray(controller.input_max, dtype=np.float64)
        delta = np.clip(desired - current, output_min, output_max)
        span = output_max - output_min
        if bool((span <= 0).any()):
            raise RuntimeError("LIBERO JOINT_POSITION 控制器输出范围无效")
        normalized = input_min + (delta - output_min) * (input_max - input_min) / span
        action = np.concatenate([normalized, np.asarray([gripper_action])])
        low, high = self._env.env.action_spec
        if action.shape != low.shape or bool((action < low).any()) or bool((action > high).any()):
            raise ValueError("Franka 关节或夹爪目标超出 LIBERO 控制范围")
        return action

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        observation, reward, done, info = self._env.step(action)
        self._last_info = _json_scalars(info)
        self._raw_observation = observation
        self._last_observation = _normalize_observation(observation)
        return dict(self._last_observation), float(reward), bool(done), self._last_info

    def success(self) -> bool:
        return bool(self._env.check_success())

    def contact_state(self) -> Dict[str, Any]:
        """发布与具体对象关联的双指接触证据，不把全场接触数当成持物。

        采用上游夹指 pad 分组和原生对象 contact_geoms，公共接口只输出对象
        source_id。双指接触仅是候选持物证据，不代表箱体/碗已离开支撑面；
        上层 Skill 仍需结合物体运动、夹爪和连续观测进行抓取验收。
        """
        inner = self._env.env
        gripper = inner.robots[0].gripper
        contacts = []
        for name, obj in inner.objects_dict.items():
            sides = {
                side: bool(inner.check_contact(gripper.important_geoms[side + "_fingerpad"],
                                               obj.contact_geoms))
                for side in ("left", "right")
            }
            if any(sides.values()):
                contacts.append({"source_id": name, "left_contact": sides["left"],
                                 "right_contact": sides["right"],
                                 "bilateral_contact": all(sides.values())})
        count = int(inner.sim.data.ncon)
        return {
            "active": count > 0,
            "count": count,
            "contacts": contacts,
            # 保留旧字段，但不把一次接触报告升级为抓取成功。
            "holding": False,
        }

    def native_metrics(self) -> Dict[str, Any]:
        return {
            "package": self._package_name,
            "version": self._version,
            "suite": self._suite_name,
            "task_id": self._task_id,
            "task_name": self._task.name,
            "init_state_id": self._init_state_id,
            "bddl_file": str(self.bddl_path),
            "last_info": self._last_info,
        }

    def close(self) -> None:
        self._env.close()


def _activate_libero_source(source_root: Path, config_root: Path) -> Optional[str]:
    root = source_root.expanduser().resolve()
    if (root / "scene-content.json").is_file():
        # 发布场景只有数据，Python 实现来自已安装 Runtime 的 Wheel。
        # 上游以源码相对路径读取部分资产；只在适配层重定向其路径变量，
        # 不改写上游源码、BDDL 或初态，也不把场景目录加入 Python 搜索路径。
        metadata = json.loads((root / "scene-content.json").read_text())
        # source_revision 是来源记录，不是兼容性协议。兼容 Runtime Profile
        # 由场景清单声明；相同 commit 也不能代替加载接口和资产的实际检查。
        for directory in ("assets", "bddl_files", "init_files"):
            if not (root / directory).is_dir():
                raise FileNotFoundError("场景数据缺少 " + directory)
        _write_libero_config(root, config_root)
        from libero.libero.envs import bddl_base_domain
        from libero.libero.envs.objects import (
            articulated_objects,
            google_scanned_objects,
            hope_objects,
            turbosquid_objects,
        )
        for module in (
            articulated_objects, google_scanned_objects, hope_objects, turbosquid_objects,
        ):
            module.absolute_path = root
        bddl_base_domain.DIR_PATH = str(root / "envs")
        return metadata.get("source_revision")
    package = root / "libero" / "libero" / "__init__.py"
    if not package.is_file():
        raise FileNotFoundError("LIBERO 源码目录无效: " + str(root))
    commit = _git_commit(root)
    root_value = str(root)
    if root_value not in sys.path:
        sys.path.insert(0, root_value)

    benchmark_root = root / "libero" / "libero"
    _write_libero_config(benchmark_root, config_root)
    return commit


def _write_libero_config(benchmark_root: Path, config_root: Path) -> None:
    config_root = config_root.expanduser().resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(benchmark_root),
        "assets": str(benchmark_root / "assets"),
    }
    (config_root / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    # 上游在首次 import 缓存 config_file；切换场景包时更新到本次配置，
    # 只在旧场景关闭后执行，运行中 reset 继续使用同一份场景数据。
    package = sys.modules.get("libero.libero")
    if package is not None:
        package.config_file = str(config_root / "config.yaml")


def generate_perturbed_bddl(
    *,
    libero_pro_root: Path,
    original_bddl: Path,
    suite_name: str,
    perturbation_name: str,
    evaluation_config: Path,
    output_dir: Path,
    seed: int,
) -> Path:
    """调用 LIBERO-Pro 的 BDDL 扰动器；不创建第二套 Runtime。"""
    supported = {"environment", "spatial", "object", "language", "task"}
    if perturbation_name not in supported:
        raise ValueError("不支持的 LIBERO-Pro 扰动: " + perturbation_name)
    module = _load_perturbation_module(libero_pro_root)
    config = yaml.safe_load(evaluation_config.read_text(encoding="utf-8")) or {}
    raw_paths = config.get("ood_task_configs", {})
    path_keys = {
        "environment": "environment",
        "spatial": "swap",
        "object": "object",
        "language": "language",
        "task": "task",
    }
    configs = {}
    for key, value in raw_paths.items():
        path = Path(value)
        if not path.is_absolute():
            path = libero_pro_root / path
        configs[key] = str(path.resolve())
    flags = module.PerturbFlags(
        use_environment=perturbation_name == "environment",
        use_swap=perturbation_name == "spatial",
        use_object=perturbation_name == "object",
        use_language=perturbation_name == "language",
        use_task=perturbation_name == "task",
    )
    required_key = path_keys[perturbation_name]
    required_path = configs.get(required_key)
    if not required_path or not Path(required_path).is_file():
        raise FileNotFoundError("LIBERO-Pro 扰动配置不存在: " + str(required_path))
    content = original_bddl.read_text(encoding="utf-8")
    pipeline = module.BDDLCombinedPerturbator(configs=configs)
    perturbed = pipeline.perturb_content(
        content=content,
        task_suite_name=suite_name,
        task_name=original_bddl.stem,
        flags=flags,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / (original_bddl.stem + "-" + perturbation_name + ".bddl")
    target.write_text(perturbed, encoding="utf-8")
    manifest = {
        "source": str(original_bddl),
        "output": str(target),
        "suite": suite_name,
        "perturbation": perturbation_name,
        "seed": seed,
        "libero_pro_commit": _git_commit(libero_pro_root),
    }
    (output_dir / "perturbation.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def _load_perturbation_module(root: Path) -> Any:
    root = root.expanduser().resolve()
    path = root / "perturbation.py"
    if not path.is_file():
        raise FileNotFoundError("LIBERO-Pro perturbation.py 不存在: " + str(path))
    sys.path.insert(0, str(root))
    name = "semantic_libero_pro_perturbation"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 LIBERO-Pro perturbation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    source = "from __future__ import annotations\n" + path.read_text(encoding="utf-8")
    code = compile(source, str(path), "exec")
    exec(code, module.__dict__)
    return module


def _git_commit(root: Path) -> Optional[str]:
    # 源码压缩包可能没有 .git，此时版本未知，不能误读其父工作区的提交。
    # Git worktree 的 .git 是文件，同样保留支持。
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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
