import json
import struct
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_runtime_service import FakeProfileAdapter, start

import semantic_sim_profiles.runtime_service as runtime_module
from semantic_sim_profiles.runtime_api import create_app
from semantic_sim_profiles.runtime_service import ProfileRuntimeError, ProfileRuntimeService


@pytest.fixture
def service(monkeypatch):
    adapter = FakeProfileAdapter()
    monkeypatch.setattr(runtime_module, "_adapter_factory", lambda *args: lambda: adapter)
    runtime = ProfileRuntimeService("robosuite-1.5")
    yield runtime, [adapter]
    runtime.shutdown()


def enable_sequence(adapter):
    adapter.controller_name = "OSC_POSE"
    adapter.consumed = []
    original_step = adapter.step

    def prepare(samples):
        return [
            {"positions": dict(adapter.positions), "gripper": 0.0, "sample": sample["index"]}
            for sample in samples
        ]

    def step(action):
        if "sample" in action:
            adapter.consumed.append(action["sample"])
        return original_step(action)

    adapter.prepare_control_sequence = prepare
    adapter.step = step


def command(instance, count=3, command_id="sequence-1", execution_id="skill-1"):
    return {
        "command_id": command_id, "execution_id": execution_id,
        "scene_generation": instance.generation, "type": "control_sequence",
        "control_sequence": {"control_period_s": 0.05,
                             "samples": [{"index": index} for index in range(count)]},
    }


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


def test_sequence_consumes_each_sample_once_and_keeps_physics_running(service):
    runtime, adapters = service
    instance = start(runtime)
    adapter = adapters[0]
    enable_sequence(adapter)
    instance.submit_command(instance.robot_id, command(instance))
    wait_until(lambda: instance.command(instance.robot_id, "sequence-1")["status"] == "succeeded")
    steps = adapter.steps
    wait_until(lambda: adapter.steps > steps + 2)
    assert adapter.consumed == [0, 1, 2]


@pytest.mark.parametrize("timeout,expected", [(30.0, "succeeded"), (0.05, "failed")])
def test_sequence_releases_old_target_on_worker_before_reporting_terminal(
    service, timeout, expected,
):
    runtime, adapters = service
    instance = start(runtime)
    adapter = adapters[0]
    enable_sequence(adapter)
    holds = []

    def hold_current():
        # 记录切换发生的线程、样本数与对外状态；不能先宣布完成再异步保持。
        holds.append((threading.get_ident(), len(adapter.consumed),
                      instance.command(instance.robot_id, "sequence-1")["status"]))

    adapter.hold_current = hold_current
    instance.submit_command(instance.robot_id, {**command(instance), "timeout_seconds": timeout})
    wait_until(lambda: instance.command(instance.robot_id, "sequence-1")["status"] == expected)
    steps = adapter.steps
    wait_until(lambda: adapter.steps > steps + 2)
    assert holds == [(instance._thread.ident, 3 if expected == "succeeded" else 1, "running")]


def test_cancel_rejects_late_inference_but_allows_a_new_execution(service):
    runtime, adapters = service
    instance = start(runtime)
    adapter = adapters[0]
    enable_sequence(adapter)
    instance.submit_command(instance.robot_id, command(instance, count=20))
    wait_until(lambda: len(adapter.consumed) > 0)
    instance.cancel_execution(instance.robot_id, "skill-1", instance.generation)
    consumed = list(adapter.consumed)
    steps = adapter.steps
    wait_until(lambda: adapter.steps > steps + 2)
    assert adapter.consumed == consumed
    with pytest.raises(ProfileRuntimeError, match="迟到"):
        instance.submit_command(instance.robot_id, command(instance, command_id="late"))
    instance.submit_command(instance.robot_id, command(
        instance, count=1, command_id="next", execution_id="skill-2",
    ))
    wait_until(lambda: instance.command(instance.robot_id, "next")["status"] == "succeeded")


def test_cancel_before_any_action_revokes_identity_and_reset_revokes_generation(service):
    runtime, adapters = service
    instance = start(runtime)
    enable_sequence(adapters[0])
    old = command(instance)
    instance.cancel_execution(instance.robot_id, "skill-1", instance.generation)
    with pytest.raises(ProfileRuntimeError, match="迟到"):
        instance.submit_command(instance.robot_id, old)
    instance.reset()
    with pytest.raises(ProfileRuntimeError, match="generation"):
        instance.submit_command(instance.robot_id, old)
    instance.submit_command(instance.robot_id, command(instance))


def test_sequence_timeout_clears_remaining_samples(service):
    runtime, adapters = service
    instance = start(runtime)
    adapter = adapters[0]
    enable_sequence(adapter)
    instance.submit_command(
        instance.robot_id, {**command(instance, count=100), "timeout_seconds": 0.1}
    )
    wait_until(lambda: instance.command(instance.robot_id, "sequence-1")["status"] == "failed")
    consumed = list(adapter.consumed)
    steps = adapter.steps
    wait_until(lambda: adapter.steps > steps + 2)
    assert adapter.consumed == consumed
    assert len(consumed) < 100


def test_synchronized_observation_keeps_original_pixels_and_same_sample(service):
    runtime, adapters = service
    start(runtime)
    adapter = adapters[0]
    # 行列梯度可暴露翻转、交换相机或 JPEG 转码，常量图像不能验证这些问题。
    original = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    adapter.policy_observation = lambda: {
        "agentview_image": original,
        "robot0_eye_in_hand_image": original[:, ::-1],
        "robot0_gripper_qpos": np.array([0.03, -0.02], dtype=np.float64),
    }
    with TestClient(create_app(runtime)) as client:
        response = client.get("/api/v1/robots/franka-0/observation")
        assert response.status_code == 200
        size = struct.unpack(">I", response.content[:4])[0]
        header = json.loads(response.content[4:4 + size])
        payload = response.content[4 + size:]
        assert header["generation"] == header["robot_state"]["generation"]
        assert header["gripper_joint_positions"]["panda_finger_joint2"] == -0.02
        assert payload[:original.nbytes] == original.tobytes()
        assert payload[original.nbytes:] == original[:, ::-1].tobytes()
