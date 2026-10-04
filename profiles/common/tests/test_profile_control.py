from __future__ import annotations

import numpy as np
import pytest

from semantic_sim_profiles.libero import FRANKA_JOINT_NAMES, LiberoAdapter
from semantic_sim_profiles.robosuite import RobosuiteAdapter


class _SimData:
    def __init__(self) -> None:
        self.position = np.asarray([-0.56, 0.1, 0.02], dtype=np.float64)
        self.rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def get_body_xpos(self, name: str):
        assert name == "panda_link0"
        return self.position

    def get_body_xmat(self, name: str):
        assert name == "panda_link0"
        return self.rotation.reshape(-1)


class _Robot:
    def __init__(self, positions, controller=None):
        self._joint_positions = np.asarray(positions, dtype=np.float64)
        self.controller = controller
        self._ref_joint_indexes = list(range(7))
        model = type("Model", (), {"jnt_range": np.asarray([[-3.0, 3.0]] * 7)})()
        self.robot_model = type("RobotModel", (), {"root_body": "panda_link0"})()
        self.sim = type("Sim", (), {"model": model, "data": _SimData()})()


class _Controller:
    input_min = np.full(7, -1.0)
    input_max = np.full(7, 1.0)
    output_min = np.full(7, -0.05)
    output_max = np.full(7, 0.05)


class _RobosuiteEnv:
    def __init__(self):
        self.robots = [_Robot([0.1 * index for index in range(7)])]
        self.action_spec = (np.asarray([-3.0] * 7 + [-1.0]), np.asarray([3.0] * 7 + [1.0]))


class _LiberoInnerEnv:
    def __init__(self):
        self.robots = [_Robot([0.0] * 7, _Controller())]
        self.action_spec = (np.asarray([-1.0] * 8), np.asarray([1.0] * 8))


class _LiberoWrapper:
    def __init__(self):
        self.env = _LiberoInnerEnv()


def test_profile_adapters_read_root_body_pose_as_world_xyzw() -> None:
    robosuite = RobosuiteAdapter.__new__(RobosuiteAdapter)
    robosuite._env = _RobosuiteEnv()
    libero = LiberoAdapter.__new__(LiberoAdapter)
    libero._env = _LiberoWrapper()

    for adapter in (robosuite, libero):
        pose = adapter.base_pose()
        adapter._last_observation = {"robot0_gripper_qpos": np.asarray([0.03, -0.02])}
        assert pose["position"] == pytest.approx([-0.56, 0.1, 0.02])
        assert pose["quaternion_xyzw"] == pytest.approx(
            [0.0, 0.0, 2**-0.5, 2**-0.5],
            abs=1e-7,
        )
        assert pose["frame_id"] == "world"
        assert np.linalg.norm(pose["quaternion_xyzw"]) == pytest.approx(1.0)


def test_robosuite_absolute_joint_action_preserves_current_pose() -> None:
    adapter = RobosuiteAdapter.__new__(RobosuiteAdapter)
    adapter._env = _RobosuiteEnv()

    joints = adapter.joint_positions()
    action = adapter.neutral_action()

    assert tuple(joints) == FRANKA_JOINT_NAMES
    assert action[:7] == pytest.approx(list(joints.values()))
    assert action[7] == 0.0

    target = dict(joints)
    target["panda_joint1"] += 0.2
    moved = adapter.joint_position_action(target, gripper_action=-0.5)
    assert moved[0] == pytest.approx(target["panda_joint1"])
    assert moved[7] == -0.5

    target.pop("panda_joint7")
    with pytest.raises(ValueError, match="目标不完整"):
        adapter.joint_position_action(target, gripper_action=0.0)


def test_libero_absolute_goal_is_converted_to_bounded_delta_action() -> None:
    adapter = LiberoAdapter.__new__(LiberoAdapter)
    adapter._env = _LiberoWrapper()
    target = {name: 0.025 for name in FRANKA_JOINT_NAMES}
    target["panda_joint7"] = 1.0

    action = adapter.joint_position_action(target, gripper_action=0.25)

    # 控制器输出范围为 ±0.05 rad，0.025 对应归一化 0.5；较远目标被限幅为 1。
    assert action[:6] == pytest.approx([0.5] * 6)
    assert action[6] == pytest.approx(1.0)
    assert action[7] == pytest.approx(0.25)
    assert tuple(adapter.joint_positions()) == FRANKA_JOINT_NAMES

    with pytest.raises(ValueError, match="控制范围"):
        adapter.joint_position_action(target, gripper_action=2.0)


def test_libero_osc_action_inverts_native_controller_scaling():
    adapter = LiberoAdapter.__new__(LiberoAdapter)
    adapter.controller_name = "OSC_POSE"
    adapter._env = _LiberoWrapper()
    controller = adapter._env.env.robots[0].controller
    controller.input_min = np.full(6, -1.0)
    controller.input_max = np.full(6, 1.0)
    controller.output_min = np.array([-0.05] * 3 + [-0.5] * 3)
    controller.output_max = -controller.output_min
    adapter._env.env.action_spec = (np.full(7, -1.0), np.full(7, 1.0))
    action = adapter.end_effector_delta_action(
        [0.025, 0.0, -0.05], [0.1, -0.5, 0.0], gripper_action=1,
    )
    assert action == pytest.approx([0.5, 0, -1, 0.2, -1, 0, 1])
    with pytest.raises(ValueError, match="控制范围"):
        adapter.end_effector_delta_action([0.06, 0, 0], [0, 0, 0], gripper_action=0)


def test_libero_policy_observation_does_not_use_display_flip():
    adapter = LiberoAdapter.__new__(LiberoAdapter)
    original = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    adapter._raw_observation = {
        "agentview_image": original, "robot0_eye_in_hand_image": original[:, ::-1],
        "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.array([0.03, -0.02]), "robot0_joint_pos": np.zeros(7),
    }
    result = adapter.policy_observation()
    assert np.array_equal(result["agentview_image"], original)
    result["agentview_image"][:] = 0
    assert original.sum() > 0
