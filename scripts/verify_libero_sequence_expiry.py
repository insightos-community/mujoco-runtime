"""真实 OSC 控制边界验证；独立实例，无模型请求，不作为任务成功验收。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from semantic_sim_profiles.libero import LiberoAdapter
from semantic_sim_profiles.runtime_service import ProfileRuntimeInstance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    captures = []

    def factory():
        adapter = LiberoAdapter(
            source_root=args.source, config_root=args.output / "config",
            suite_name="libero_spatial", task_id=0, init_state_id=0, seed=0,
            camera_names=("agentview", "robot0_eye_in_hand"),
            width=256, height=256, horizon=3000, controller="OSC_POSE",
        )
        hold = adapter.hold_current

        def observed_hold():
            controller = adapter._env.env.robots[0].controller
            controller.update(force=True)
            before = controller.goal_pos.copy()
            hold()
            captures.append({"old_goal": before.tolist(),
                             "new_goal": controller.goal_pos.tolist(),
                             "actual": controller.ee_pos.tolist()})

        adapter.hold_current = observed_hold
        return adapter

    runtime = ProfileRuntimeInstance(
        profile_id="libero-robosuite-1.4", scene_key="libero_spatial:0", layout="init-0",
        request={"request_id": "sequence-expiry-check", "seed": 0}, adapter_factory=factory,
    )
    report = {"agent_end_to_end": False, "passed": False}
    try:
        runtime.start()
        ready = runtime.wait_ready(60)
        assert ready["state"] == "running", ready
        # 一个非零增量只执行 50ms。检查“完成时捕获”的真实 OSC 目标，
        # 而不是以最终画面看起来静止、或模拟器暂停来推断控制边界正确。
        runtime.submit_command("franka-0", {
            "command_id": "expiry", "execution_id": "expiry-check", "scene_generation": 1,
            "type": "control_sequence", "control_sequence": {
                "control_period_s": .05, "samples": [{"controls": [
                    {"kind": "end_effector_delta", "group": "arm", "frame_id": "world",
                     "translation_m": [0, 0, .02], "rotation_axis_angle_rad": [0, 0, 0]},
                    {"kind": "gripper_direction", "group": "hand", "closing_direction": 0},
                ]}],
            },
        })
        deadline = time.monotonic() + 10
        while runtime.command("franka-0", "expiry")["status"] != "succeeded":
            if time.monotonic() > deadline:
                raise TimeoutError("控制序列未完成")
            time.sleep(.01)
        captured = captures[-1]
        np.testing.assert_allclose(captured["new_goal"], captured["actual"], atol=1e-10)
        assert np.linalg.norm(np.subtract(captured["old_goal"], captured["actual"])) > .001
        step = runtime.step_count
        time.sleep(.3)

        def read_goal(adapter):
            c = adapter._env.env.robots[0].controller
            return {"goal": c.goal_pos.tolist(), "actual": c.ee_pos.tolist()}

        after = runtime._worker_call(read_goal)
        np.testing.assert_allclose(after["goal"], captured["new_goal"], atol=1e-8)
        assert runtime.step_count > step
        report.update(passed=True, captured=captured, after_wait=after,
                      physics_steps_during_wait=runtime.step_count - step)
    finally:
        runtime.stop()
        (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
