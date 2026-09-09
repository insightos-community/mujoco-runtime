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
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import yaml

from semantic_sim_profiles.pose import read_root_body_pose
from semantic_sim_profiles.visual_source import public_source_for_body

LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_PRO_COMMIT = "0bcf73621c789ffd6ed8858467a89df9ca94fd6b"
FRANKA_JOINT_NAMES = tuple("panda_joint" + str(index) for index in range(1, 8))


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
    ) -> None:
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
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(self.bddl_path),
            controller="JOINT_POSITION",
            camera_names=list(camera_names),
            camera_heights=height,
            camera_widths=width,
            camera_depths=True,
            has_renderer=False,
            has_offscreen_renderer=True,
            control_freq=20,
            horizon=max(horizon, 1),
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
        self._last_observation = _normalize_observation(observation)
        return dict(self._last_observation)

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

    def visual_source_for_body(self, body_id: int, object_source_ids: Iterable[str]) -> str | None:
        return public_source_for_body(
            self._env.env.sim.model,
            body_id,
            robot_source_id="franka-0",
            object_source_ids=object_source_ids,
        )

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
        self._last_observation = _normalize_observation(observation)
        return dict(self._last_observation), float(reward), bool(done), self._last_info

    def success(self) -> bool:
        return bool(self._env.check_success())

    def contact_state(self) -> Dict[str, Any]:
        """只发布接触汇总，避免把 LIBERO / MuJoCo 内部 geom 名称变成公共接口。"""

        inner = getattr(self._env, "env", self._env)
        sim = getattr(inner, "sim", None)
        count = int(getattr(getattr(sim, "data", None), "ncon", 0))
        return {
            "active": count > 0,
            "count": count,
            "contacts": [],
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


def _activate_libero_source(source_root: Path, config_root: Path) -> str:
    root = source_root.expanduser().resolve()
    package = root / "libero" / "libero" / "__init__.py"
    if not package.is_file():
        raise FileNotFoundError("LIBERO 源码目录无效: " + str(root))
    commit = _git_commit(root)
    if commit != LIBERO_COMMIT:
        raise RuntimeError("LIBERO 源码提交不匹配，期望 " + LIBERO_COMMIT + "，实际 " + str(commit))
    root_value = str(root)
    if root_value not in sys.path:
        sys.path.insert(0, root_value)

    benchmark_root = root / "libero" / "libero"
    config_root = config_root.expanduser().resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(root / "libero" / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    (config_root / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    return commit


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
    commit = _git_commit(root)
    if commit != LIBERO_PRO_COMMIT:
        raise RuntimeError(
            "LIBERO-Pro 源码提交不匹配，期望 " + LIBERO_PRO_COMMIT + "，实际 " + str(commit)
        )
    sys.path.insert(0, str(root.resolve()))
    path = root / "perturbation.py"
    if not path.is_file():
        raise FileNotFoundError("LIBERO-Pro perturbation.py 不存在: " + str(path))
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
