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

import time

import pytest

from plugin_mujoco.application import RuntimeManager
from plugin_mujoco.errors import ConflictError
from plugin_mujoco.models import SceneStartRequest
from plugin_mujoco.settings import Settings


@pytest.fixture
def manager(asset_root):
    value = RuntimeManager(Settings(asset_root=asset_root, backend="fake", realtime=True))
    yield value
    value.shutdown()


def start(manager, request_id="start-1", layout="layout001"):
    return manager.start(
        "palletizing_depalletizing_001",
        SceneStartRequest(
            request_id=request_id,
            layout=layout,
            seed=7,
            headless=True,
            render_backend="auto",
        ),
    )


def test_start_is_idempotent_and_one_scene_is_active(manager):
    first = start(manager)
    second = start(manager)
    assert second.instance_id == first.instance_id

    with pytest.raises(ConflictError):
        start(manager, request_id="start-2", layout="layout002")
    with pytest.raises(ConflictError):
        manager.start(
            "palletizing_depalletizing_001",
            SceneStartRequest(
                request_id="start-1",
                layout="layout002",
                headless=True,
                render_backend="auto",
            ),
        )


def test_pause_uses_condition_wait_resume_reset_and_stop(manager):
    record = start(manager)
    assert manager.wait_ready(record.instance_id).state.value == "running"
    instance = manager.get(record.instance_id)
    time.sleep(0.02)
    before_pause = instance.pause()
    time.sleep(0.03)
    after_wait = instance.view()

    assert after_wait.state.value == "paused"
    assert after_wait.sim_time == before_pause.sim_time
    instance.resume()
    time.sleep(0.02)
    assert instance.view().sim_time > after_wait.sim_time

    generation = instance.view().generation
    reset = instance.reset()
    assert reset.generation == generation + 1
    assert reset.sim_time <= 0.01
    stopped = instance.stop()
    assert stopped.state.value == "stopped"
    assert manager.info().state == "ready"
    assert manager.info().active_instance_id is None


def test_shutdown_stops_active_scene_and_holds_robots(manager):
    record = start(manager)
    assert manager.wait_ready(record.instance_id).state.value == "running"
    instance = manager.get(record.instance_id)
    manager.shutdown()
    assert instance.view().state.value == "stopped"
    for robot_id in instance.robot_ids():
        assert instance.backend.robot_state(robot_id, 1).in_hold


def test_backend_failure_keeps_queryable_failed_instance(manager, monkeypatch):
    class BrokenBackend:
        def __init__(self, definition, *, endpoint=None):
            raise RuntimeError("模型损坏")

    monkeypatch.setattr("plugin_mujoco.testing.fake_backend.FakeBackend", BrokenBackend)
    request = SceneStartRequest(
        request_id="broken-scene", layout="layout001", render_backend="auto"
    )
    started = manager.start("palletizing_depalletizing_001", request)
    assert started.state.value == "starting"
    record = manager.wait_ready(started.instance_id)
    assert record.state.value == "failed"
    assert record.failure_reason == "模型损坏"
    assert manager.info().state == "failed"
    assert manager.info().active_instance_id == started.instance_id
    other = request.model_copy(update={"request_id": "replacement-before-stop"})
    with pytest.raises(ConflictError):
        manager.start("palletizing_depalletizing_001", other)
    assert manager.start("palletizing_depalletizing_001", request).state.value == "failed"
    stopped = manager.stop(started.instance_id)
    assert stopped.state.value == "stopped"
    assert manager.info().state == "ready"
    replacement = manager.start("palletizing_depalletizing_001", other)
    assert replacement.instance_id != started.instance_id


def test_runtime_rejects_render_backend_switch(manager):
    with pytest.raises(ConflictError) as captured:
        manager.start(
            "palletizing_depalletizing_001",
            SceneStartRequest(
                request_id="wrong-renderer",
                layout="layout001",
                render_backend="osmesa",
            ),
        )
    assert captured.value.details == {
        "configured": manager.settings.render_backend,
        "requested": "osmesa",
    }
