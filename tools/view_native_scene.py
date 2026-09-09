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

"""用 MuJoCo 官方 Viewer 打开 Runtime 实际拼装后的 native 场景。

资产仓中的场景 XML 只描述公共物理环境；Robot 和 Layout 对象由 Runtime
按目录配置拼装。本工具复用正式构建链，避免评审窗口与真正运行模型产生偏差。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from plugin_mujoco.native.backend import MujocoBackend
from plugin_mujoco.scene.catalog import SceneCatalog


def main() -> None:
    parser = argparse.ArgumentParser(description="打开完整 native MuJoCo 场景")
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--scene", default="palletizing_depalletizing_tote_v1")
    parser.add_argument("--layout", default="layout001")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只编译场景并输出摘要，不打开图形窗口",
    )
    args = parser.parse_args()

    asset_root = args.asset_root.expanduser().resolve()
    definition = SceneCatalog(asset_root).load(args.scene, args.layout)
    # 预览必须复用正式 Runtime Backend，而不能直接调用 mujoco.mj_step。
    # R1 Pro 的躯干与双臂采用力矩执行器；Backend.step() 会在每个物理周期
    # 执行关节 PD 保持并补偿重力/科氏力。绕过这条链路时 ctrl 默认为 0，
    # Robot 虽然模型加载正确，仍会在重力作用下立即瘫倒。
    backend = MujocoBackend(definition, render_backend="glfw", seed=0)
    try:
        model = backend.model
        data = backend.data
        print(
            f"已加载 {args.scene}/{args.layout}: "
            f"{len(definition.robots)} Robot, {len(definition.assets)} 个场景对象"
        )
        print(f"Runtime 临时场景：{backend.build.xml_path}")
        if args.check:
            return

        # 必须在 Backend 设置 MUJOCO_GL=glfw 并导入 mujoco 后再导入 viewer，
        # 避免同一进程先初始化成 EGL/OSMesa 后又尝试切换图形后端。
        import mujoco.viewer

        print("左键旋转；Shift+右键平移；滚轮缩放；关闭窗口即可退出。")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = model.stat.center
            viewer.cam.distance = max(float(model.stat.extent) * 1.5, 2.5)
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -22.0
            next_step_at = time.monotonic()
            while viewer.is_running():
                # 本调试入口和产品 Runtime 使用同一个单写者规则：只有这里的
                # 循环推进物理；Viewer 只同步画面，不直接修改 MjData。
                backend.step()
                viewer.sync()
                next_step_at += backend.timestep
                delay = next_step_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    # 窗口被拖动或系统调度造成落后时，从当前时刻重新计时，
                    # 不通过无休止追赶形成忙循环。
                    next_step_at = time.monotonic()
    finally:
        backend.close()


if __name__ == "__main__":
    main()
