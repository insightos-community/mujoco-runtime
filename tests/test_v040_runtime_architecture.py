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
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from plugin_mujoco.application import RuntimeManager
from plugin_mujoco.compiler import RuntimeBundle
from plugin_mujoco.models import (
    RobotCommandRequest,
    RuntimeProfile,
    SceneEvaluation,
    SceneStartRequest,
)
from plugin_mujoco.settings import Settings


def test_public_command_only_accepts_low_level_trajectories():
    with pytest.raises(ValidationError):
        RobotCommandRequest(
            command_id="forbidden-eef",
            scene_generation=1,
            type="eef_pose",
        )
    with pytest.raises(ValidationError):
        RobotCommandRequest(
            command_id="forbidden-nav",
            scene_generation=1,
            type="base_pose",
        )


def test_runtime_has_no_proximity_attachment_grasp_strategy():
    """Runtime 只能报告真实接触，不能靠距离搬动物体或关闭碰撞。"""
    source = Path("src/plugin_mujoco/native/backend.py").read_text()

    forbidden = (
        "grasp_assist_mode",
        "_nearby_grasp_candidate",
        "_apply_attachments",
        "_disable_object_contacts",
    )
    for symbol in forbidden:
        assert symbol not in source


def test_holding_requires_bilateral_physical_contacts():
    """Holding 只反映实时双指接触；接触消失后不能保留旧结果。"""
    from plugin_mujoco.native.backend import MujocoBackend

    backend = MujocoBackend.__new__(MujocoBackend)
    backend._holding = {"r1": None}
    backend.build = SimpleNamespace(object_body_names={"box-1"})
    backend._profiles = {}
    names = {
        1: "box-1",
        2: "r1:left_gripper_finger_link1",
        3: "r1:left_gripper_finger_link2",
    }
    backend._public_body_for_geom = names.__getitem__

    backend.data = SimpleNamespace(
        ncon=1,
        contact=[SimpleNamespace(geom1=1, geom2=2)],
    )
    backend._detect_holding()
    assert backend._holding["r1"] is None

    backend.data = SimpleNamespace(
        ncon=2,
        contact=[SimpleNamespace(geom1=1, geom2=2), SimpleNamespace(geom1=1, geom2=3)],
    )
    backend._detect_holding()
    assert backend._holding["r1"] == "box-1"

    backend.data = SimpleNamespace(ncon=0, contact=[])
    backend._detect_holding()
    assert backend._holding["r1"] is None



def test_public_body_lookup_caches_static_model_identity():
    """接触采样可以缓存geom身份，但不能缓存动态接触数值。"""
    from plugin_mujoco.native.backend import MujocoBackend

    class FakeMJ:
        class mjtObj:
            mjOBJ_BODY = "body"

        def __init__(self):
            self.calls = 0

        def mj_id2name(self, _model, _kind, _body_id):
            self.calls += 1
            return "box-1"

    backend = MujocoBackend.__new__(MujocoBackend)
    backend._public_body_cache = {}
    backend.model = SimpleNamespace(geom_bodyid=[1], body_parentid=[0, 0])
    backend.build = SimpleNamespace(object_body_names={"box-1"}, robots=[])
    backend.mj = FakeMJ()

    assert backend._public_body_for_geom(0) == "box-1"
    assert backend._public_body_for_geom(0) == "box-1"
    assert backend.mj.calls == 1

def test_runtime_profiles_never_report_missing_environment_as_ready(asset_root):
    manager = RuntimeManager(Settings(asset_root=asset_root, backend="fake"))
    profiles = {item.profile_id: item for item in manager.runtime_profiles()}
    assert profiles["native-mujoco"].available is True
    for profile_id in ("robosuite-1.5", "libero-robosuite-1.4"):
        profile = profiles[profile_id]
        assert profile.available is False
        assert profile.unavailable_reason
        # 依赖缺失与依赖已安装但 Loader 未接入都是合法的明确不可用状态。


def test_runtime_profile_environment_probe_runs_once(asset_root, monkeypatch):
    """频繁健康检查不能重复启动 robosuite/LIBERO 的隔离解释器。"""
    from plugin_mujoco.loaders import RuntimeProfileRegistry

    probes: list[tuple[Path, str]] = []

    def record_probe(environment: Path, module: str) -> bool:
        probes.append((environment, module))
        return False

    monkeypatch.setattr("plugin_mujoco.loaders.profiles._isolated_module_available", record_probe)
    registry = RuntimeProfileRegistry(Settings(asset_root=asset_root, backend="fake"), Path.cwd())
    first = registry.list()
    first[0].capabilities.robot_models.append("mutated-by-client")
    assert registry.get("native-mujoco").capabilities.robot_models == ["r1_pro_chassis"]
    assert len(registry.list()) == 3
    assert [module for _, module in probes] == ["robosuite", "libero"]


def test_starting_can_be_cancelled_before_backend_creation(asset_root, monkeypatch):
    manager = RuntimeManager(Settings(asset_root=asset_root, backend="fake"))
    original = manager.catalog.load

    def slow_load(scene_key, layout):
        time.sleep(0.05)
        return original(scene_key, layout)

    monkeypatch.setattr(manager.catalog, "load", slow_load)
    started = manager.start(
        "palletizing_depalletizing_001",
        SceneStartRequest(request_id="cancel-start", layout="layout001"),
    )
    assert started.state.value == "starting"
    stopped = manager.stop(started.instance_id)
    assert stopped.state.value == "stopped"
    assert stopped.progress_message == "场景启动已取消"


def test_all_physics_mutations_run_on_single_thread(asset_root, monkeypatch):
    from plugin_mujoco.testing.fake_backend import FakeBackend

    mutation_threads: set[int] = set()

    class RecordingBackend(FakeBackend):
        def step(self):
            mutation_threads.add(threading.get_ident())
            super().step()

        def reset(self):
            mutation_threads.add(threading.get_ident())
            super().reset()

        def hold_robot(self, robot_id):
            mutation_threads.add(threading.get_ident())
            super().hold_robot(robot_id)

    monkeypatch.setattr("plugin_mujoco.testing.fake_backend.FakeBackend", RecordingBackend)
    manager = RuntimeManager(Settings(asset_root=asset_root, backend="fake", realtime=True))
    started = manager.start(
        "palletizing_depalletizing_001", SceneStartRequest(request_id="threads", layout="layout001")
    )
    assert manager.wait_ready(started.instance_id).state.value == "running"
    instance = manager.get(started.instance_id)
    instance.pause()
    instance.step(1)
    instance.reset()
    assert len(mutation_threads) == 1
    manager.shutdown()


def test_public_contract_examples_match_models(asset_root):
    """公共样例既要能解析，也要与 Runtime 当前实际标识完全一致。

    这里刻意固定 v1、scene kind 和 Robot model。若实现改名，必须同步更新
    Framework、Studio 和 Robot SDK 的 canonical fixture，不能依赖模型默认值
    让陈旧样例继续通过。
    """
    root = Path("examples/contracts")
    bundle = RuntimeBundle.model_validate(
        json.loads((root / "runtime-bundle.json").read_text(encoding="utf-8"))
    )
    start = SceneStartRequest.model_validate(
        json.loads((root / "scene-start-request.json").read_text(encoding="utf-8"))
    )
    profiles = [
        RuntimeProfile.model_validate(value)
        for value in json.loads((root / "runtime-profiles.json").read_text(encoding="utf-8"))
    ]
    commands = [
        RobotCommandRequest.model_validate(value)
        for value in json.loads((root / "robot-commands.json").read_text(encoding="utf-8"))
    ]
    evaluation = SceneEvaluation.model_validate(
        json.loads((root / "scene-evaluation.json").read_text(encoding="utf-8"))
    )

    native = next(item for item in profiles if item.profile_id == "native-mujoco")
    actual_native = next(
        item
        for item in RuntimeManager(
            Settings(asset_root=asset_root, backend="fake")
        ).runtime_profiles()
        if item.profile_id == "native-mujoco"
    )
    assert actual_native.model_dump(mode="json") == native.model_dump(mode="json")
    assert native.api_version == "v1"
    assert native.scene_kinds == ["scene_document", "asset_scene"]
    assert native.capabilities.robot_models == ["r1_pro_chassis"]
    assert native.environment_ready is True and native.available is True
    assert bundle.runtime_profile_id == native.profile_id
    assert bundle.document.nodes[0].properties["model"] == "r1_pro_chassis"
    assert start.runtime_profile_id == native.profile_id
    assert commands[1].joint_trajectory.points[-1].positions == {"left_arm_joint1": 0.3}
    assert evaluation.runtime_profile_id == "libero-robosuite-1.4"


def test_realtime_delay_uses_continuous_wall_clock_deadline():
    from plugin_mujoco.runtime.instance import _realtime_deadline_delay

    assert _realtime_deadline_delay(10.002, 10.0005) == pytest.approx(0.0015)
    assert _realtime_deadline_delay(10.002, 10.002) == 0.0
    assert _realtime_deadline_delay(10.002, 10.004) == 0.0
