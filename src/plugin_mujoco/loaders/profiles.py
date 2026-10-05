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

"""隔离 Runtime Profile 的环境探测与加载。"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from plugin_mujoco.assets import BackendSceneSnapshotProvider
from plugin_mujoco.errors import BackendFailureError, NotFoundError
from plugin_mujoco.models import RuntimeCapability, RuntimeProfile, SceneStartRequest
from plugin_mujoco.rendering import RenderExecutor
from plugin_mujoco.robots import BackendRobotDriver
from plugin_mujoco.runtime.interfaces import RuntimeComponents
from plugin_mujoco.runtime.physics import BackendPhysicsRuntime
from plugin_mujoco.scene import SceneDefinition
from plugin_mujoco.sensors import BackendSensorSource
from plugin_mujoco.settings import Settings
from plugin_mujoco.visuals import BackendSceneVisualProvider


class RuntimeProfileRegistry:
    """统一暴露 Profile；只有真正可导入的环境才报告 available。"""

    def __init__(self, settings: Settings, repository_root: Path) -> None:
        self.settings = settings
        self.repository_root = repository_root
        self._profiles = tuple(self._discover_profiles())

    def _discover_profiles(self) -> list[RuntimeProfile]:
        native_available = self.settings.backend == "fake" or _module_available("mujoco")
        return [
            RuntimeProfile(
                runtime_profile_id="native-mujoco",
                name="Native MuJoCo",
                engine="mujoco",
                loader="native",
                environment="project",
                environment_ready=native_available,
                available=native_available,
                unavailable_reason=None if native_available else "当前环境未安装 mujoco",
                scene_kinds=["scene_document", "asset_scene"],
                capabilities=RuntimeCapability(
                    editable_scene=True, robot_models=["r1_pro_chassis"]
                ),
            ),
            self._isolated(
                "robosuite-1.5",
                "robosuite 1.5",
                "robosuite",
                "profiles/robosuite",
                ["robosuite-task"],
                ["Panda", "Franka"],
                native_evaluator=True,
            ),
            self._isolated(
                "libero-robosuite-1.4",
                "LIBERO / LIBERO-Pro",
                "libero",
                "profiles/libero",
                ["libero-task", "libero-pro-evaluation"],
                ["Panda", "Franka"],
                native_evaluator=True,
            ),
        ]

    def list(self) -> list[RuntimeProfile]:
        """返回启动期探测结果的副本，禁止 HTTP 请求重复启动隔离解释器。

        Profile 环境只会随 Runtime 进程重启而改变。若在每个请求中导入
        robosuite/LIBERO，冷缓存会拖慢 native Profile 的健康检查，甚至让
        Supervisor 在 Runtime 已启动时仍持续超时。
        """
        return [profile.model_copy(deep=True) for profile in self._profiles]

    def get(self, profile_id: str) -> RuntimeProfile:
        profile = next((item for item in self.list() if item.profile_id == profile_id), None)
        if profile is None:
            raise NotFoundError(f"Runtime Profile 不存在: {profile_id}")
        return profile

    def load(
        self, profile_id: str, definition: SceneDefinition, request: SceneStartRequest
    ) -> RuntimeComponents:
        profile = self.get(profile_id)
        if not profile.available:
            raise BackendFailureError(
                "Runtime Profile 当前不可用",
                details={"profile_id": profile_id, "reason": profile.unavailable_reason},
            )
        if profile_id != "native-mujoco":
            # robosuite/LIBERO 需要由各自锁定环境中的进程入口启动。当前进程
            # 只负责准确探测，绝不退回 Fake Backend 冒充成功。
            raise BackendFailureError(
                "隔离 Profile 尚未连接到当前 Runtime 进程",
                details={"profile_id": profile_id, "environment": profile.environment},
            )
        if self.settings.backend == "fake":
            from plugin_mujoco.testing.fake_backend import FakeBackend

            backend = FakeBackend(definition, endpoint=self.settings.robot_endpoint)
            from plugin_mujoco.testing.fake_visuals import FakeSceneVisualProvider

            visuals = FakeSceneVisualProvider(backend)
        else:
            from plugin_mujoco.native import MujocoBackend

            backend = MujocoBackend(
                definition,
                render_backend=request.render_backend,
                seed=request.seed,
                endpoint=self.settings.robot_endpoint,
            )
            visuals = BackendSceneVisualProvider(backend)
        render_executor = RenderExecutor(definition.scene_key)

        def close_backend() -> None:
            # Renderer 必须在创建它的线程释放；即使 close 失败也要结束执行器。
            try:
                render_executor.call(backend.close)
            finally:
                render_executor.close()

        return RuntimeComponents(
            physics=BackendPhysicsRuntime(backend, close_callback=close_backend),
            robots=BackendRobotDriver(backend),
            sensors=BackendSensorSource(backend, render_executor),
            snapshot=BackendSceneSnapshotProvider(backend),
            visuals=visuals,
        )

    def _isolated(
        self,
        profile_id: str,
        name: str,
        module: str,
        environment: str,
        scene_types: list[str],
        robots: list[str],
        *,
        native_evaluator: bool,
    ) -> RuntimeProfile:
        env_path = self.repository_root / environment
        lock_exists = (env_path / "uv.lock").is_file()
        module_ready = _isolated_module_available(env_path, module)
        environment_ready = lock_exists and module_ready
        # 本仓已有独立 benchmark runner，但尚未实现 RuntimeComponents Loader。
        # 因此即使依赖可导入也不能对 Framework 宣称该 Profile 可启动。
        available = False
        reason = (
            f"隔离环境探测：锁文件={lock_exists}，可导入 {module}={module_ready}；"
            "Runtime Loader 尚未接入"
        )
        return RuntimeProfile(
            runtime_profile_id=profile_id,
            name=name,
            engine="mujoco",
            loader=module,
            environment=environment,
            environment_ready=environment_ready,
            available=available,
            unavailable_reason=reason,
            scene_kinds=scene_types,
            capabilities=RuntimeCapability(
                native_evaluator=native_evaluator,
                robot_models=robots,
                viewer_camera_modes=["fixed"],
            ),
        )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _isolated_module_available(environment: Path, module: str) -> bool:
    """使用隔离环境自己的解释器探测，不能拿主进程的 sys.path 代替。"""
    executable = (
        environment / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )
    if not executable.is_file():
        return False
    try:
        result = subprocess.run(
            [str(executable), "-c", f"import {module}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
