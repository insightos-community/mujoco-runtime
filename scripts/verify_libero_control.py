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

"""真实 LIBERO 接口验收；不冒充模型任务或 Agent 端到端验收。

使用独立 Runtime 实例，不连接或重置主目录中正在运行的 R1 场景。
示例：PYTHONPATH=profiles/common/src profiles/libero/.venv/bin/python
scripts/verify_libero_control.py --source /path/to/LIBERO --output /path/to/evidence
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from semantic_sim_profiles.libero import LiberoAdapter, installed_task_catalog
from semantic_sim_profiles.runtime_service import ProfileRuntimeError, ProfileRuntimeInstance


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not predicate():
        raise RuntimeError("等待验证条件超时")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-state", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output / "libero-config"
    report = {"kind": "runtime_interface_verification", "native_task_success_required": False,
              "init_state": args.init_state}
    try:
        catalog = installed_task_catalog(args.source, config)
        report["catalog_tasks"] = len(catalog)
    except Exception as error:
        report.update(passed=False, phase="catalog", error=str(error))
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    instance = ProfileRuntimeInstance(
        profile_id="libero-robosuite-1.4", scene_key="libero_spatial:0",
        layout=f"init-{args.init_state}", request={"request_id": "interface-check"},
        adapter_factory=lambda: LiberoAdapter(
            source_root=args.source, config_root=config, suite_name="libero_spatial", task_id=0,
            init_state_id=args.init_state, seed=0, camera_names=("agentview", "robot0_eye_in_hand"),
            width=256, height=256, horizon=3, controller="OSC_POSE",
        ),
    )
    try:
        instance.start()
        ready = instance.wait_ready(120)
        if ready["state"] != "running":
            raise RuntimeError(ready["failure_reason"])
        wait_for(lambda: instance.step_count >= 12, timeout=60)
        snapshot = instance.snapshot()
        assert snapshot["objects"] and snapshot["regions"]
        assert all(obj["category"] != "object" for obj in snapshot["objects"])
        (args.output / "snapshot.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["map_objects"] = len(snapshot["objects"])
        report["map_regions"] = len(snapshot["regions"])
        metadata, pixels = instance.synchronized_observation(instance.robot_id)
        (args.output / "observation.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for frame in metadata["images"]:
            start = frame["offset"]
            image = np.frombuffer(pixels[start:start + frame["length"]], dtype=np.uint8)
            image = image.reshape(frame["height"], frame["width"], 3)
            Image.fromarray(image).save(args.output / (frame["sensor_id"] + "-raw.png"))
        sample = {"controls": [
            {"kind": "end_effector_delta", "group": "arm", "frame_id": "world",
             "translation_m": [0, 0, 0], "rotation_axis_angle_rad": [0, 0, 0]},
            {"kind": "gripper_direction", "group": "hand", "closing_direction": 0},
        ]}
        request = {
            "command_id": "zero-delta", "execution_id": "interface-check",
            "scene_generation": instance.generation, "type": "control_sequence",
            "control_sequence": {"control_period_s": 0.05, "samples": [sample, sample]},
        }
        instance.submit_command(instance.robot_id, request)
        wait_for(lambda: instance.command(instance.robot_id, "zero-delta")["status"] == "succeeded")
        report["command"] = instance.command(instance.robot_id, "zero-delta")
        instance.cancel_execution(instance.robot_id, "interface-check", instance.generation)
        try:
            instance.submit_command(instance.robot_id, {**request, "command_id": "late-result"})
        except ProfileRuntimeError:
            report["late_result_rejected"] = True
        else:
            raise AssertionError("已取消执行接受了迟到结果")
        instance.pause()
        paused = instance.step_count
        time.sleep(0.15)
        assert instance.step_count == paused
        instance.resume()
        wait_for(lambda: instance.step_count > paused + 2)
        report["evaluation"] = instance.evaluation()
        instance.reset()
        assert instance.generation == 2
        report["reset_generation"] = instance.generation
        report["passed"] = True
    except Exception as error:
        report.update(passed=False, error=str(error))
        raise
    finally:
        try:
            instance.stop()
        except Exception as error:
            report.update(passed=False, stop_error=str(error))
            raise
        finally:
            report["final_state"] = instance.view()
            (args.output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
