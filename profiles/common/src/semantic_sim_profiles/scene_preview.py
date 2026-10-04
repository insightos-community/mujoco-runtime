"""安装期原生初态预览。独立进程运行，不创建在线实例或加载策略模型。"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from semantic_sim_profiles.libero import LiberoAdapter


def settled_initial_state(adapter, index):
    """复用上游 render_single_task 的 5 个零动作步，再读取新渲染观测。

    原生初态保存的是物理状态，部分物体尚未落稳；直接截图会显示悬空。
    仅离线预览推进这 5 步，不改变在线 reset 或策略的控制时序。
    """
    observation = adapter.reset_initial_state(index)
    for _ in range(5):
        observation, _, _, _ = adapter.step(np.zeros(7))
    return observation


def state_summary(env):
    """从对象注册表和 MuJoCo 关节表取事实，覆盖任意 LIBERO 任务。"""
    model, data = env.sim.model, env.sim.data
    objects = {name: {"position": data.body_xpos[body].tolist(),
                      "rotation": data.body_xquat[body].tolist()}
               for name, body in env.obj_body_id.items()}
    joints = {model.joint_id2name(i): {"value": float(data.qpos[model.jnt_qposadr[i]]),
                                      "unit": "m" if model.jnt_type[i] == 2 else "rad"}
              for i in range(model.njnt) if model.jnt_type[i] in (2, 3)}
    return {"objects": objects, "joints": joints}


def describe_difference(baseline, current):
    changes = []
    for name, value in current["objects"].items():
        old = baseline["objects"].get(name)
        if old is None:
            continue
        distance = float(np.linalg.norm(np.array(value["position"]) - old["position"]))
        dot = float(abs(np.dot(value["rotation"], old["rotation"])))
        angle = math.degrees(2 * math.acos(min(1.0, dot)))
        parts = []
        if distance >= .001:
            parts.append("位置变化 %.1f cm" % (distance * 100))
        if angle >= 1:
            parts.append("朝向变化 %.1f°" % angle)
        if parts:
            changes.append(name + "：" + "、".join(parts))
    for name, value in current["joints"].items():
        old = baseline["joints"].get(name)
        if old and abs(value["value"] - old["value"]) >= (.001 if value["unit"] == "m" else .02):
            changes.append(
                "%s：变化 %.3f %s" % (name, value["value"] - old["value"], value["unit"])
            )
    return (
        "；".join(changes)
        if changes
        else "与默认初态相比，未见超过 1 mm / 1° 的物体位姿变化或明显关节变化"
    )


def prepare(request_path):
    request = json.loads(Path(request_path).read_text())
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    result_file = output / "result.json"
    result = json.loads(result_file.read_text()) if result_file.exists() else {"variants": {}}
    if result.get("description") and all(
            variant in result["variants"]
            and (output / result["variants"][variant]["preview"]).is_file()
            for variant in request["variants"]):
        print(
            "已生成 %d / %d 个初态预览（复用缓存）"
            % (len(request["variants"]), len(request["variants"])),
            flush=True,
        )
        return
    suite, task_id = request["scene_key"].rsplit(":", 1)
    adapter = LiberoAdapter(source_root=Path(request["content_root"]),
        config_root=output / ".config", suite_name=suite, task_id=int(task_id),
        init_state_id=0, seed=0, camera_names=("agentview",), width=384, height=384,
        horizon=10, controller="OSC_POSE")
    try:
        settled_initial_state(adapter, 0)
        env = adapter._env.env
        baseline = state_summary(env)
        problem = env.parsed_problem
        # 英文指令直接来自上游。目标表达式保持原意，不使用生成式模型猜测任务。
        result["description"] = "%s\n任务：%s / %s\n相关物体：%s\n原生目标：%s" % (
            adapter._task.language, suite, task_id, ", ".join(problem.get("obj_of_interest", [])),
            json.dumps(problem.get("goal_state", []), ensure_ascii=False))
        total = len(request["variants"])
        for index, variant in enumerate(request["variants"]):
            existing = result["variants"].get(variant)
            if not existing or not (output / existing["preview"]).is_file():
                if not variant.startswith("init-"):
                    raise ValueError("LIBERO 初态标识无效: " + variant)
                observation = settled_initial_state(adapter, int(variant[5:]))
                filename = "%d.jpg" % int(variant[5:])
                # reset 返回与 Web 相同方向的 RGB；不使用 VLA 专用原始图像。
                image = np.asarray(observation["agentview_image"], dtype=np.uint8)
                Image.fromarray(image).save(output / filename)
                snapshot = state_summary(env)
                result["variants"][variant] = {"preview": filename, "state": snapshot,
                    "description": "默认原生初态" if variant == "init-0" else
                        "相对 init-0：" + describe_difference(baseline, snapshot)}
            # 每个初态完成就发布，可取消并在下次只补全缺失项。
            temporary = result_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False))
            temporary.replace(result_file)
            print("已生成 %d / %d 个初态预览" % (index + 1, total), flush=True)
    finally:
        adapter.close()
