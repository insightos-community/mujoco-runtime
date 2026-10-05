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

"""进程设置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    asset_root: Path
    host: str = "127.0.0.1"
    port: int = 8090
    render_backend: str = "egl"
    backend: str = "mujoco"
    realtime: bool = False
    authoring_root: Path | None = None
    public_endpoint: str | None = None

    @property
    def robot_endpoint(self) -> str:
        """返回 SDK/Pilot 可访问的 Runtime 地址。

        监听 0.0.0.0 只用于绑定 socket，不能作为消费者 Endpoint。部署时可
        用 PLUGIN_MUJOCO_PUBLIC_ENDPOINT 显式声明跨主机地址；本机默认回退
        到 127.0.0.1。
        """
        if self.public_endpoint:
            return self.public_endpoint.rstrip("/")
        host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{host}:{self.port}"

    @classmethod
    def from_env(cls) -> Settings:
        raw_root = os.getenv("MUJOCO_ASSET_ROOT", "").strip()
        asset_root = Path(raw_root).expanduser().resolve() if raw_root else Path.cwd()
        return cls(
            asset_root=asset_root,
            host=os.getenv("PLUGIN_MUJOCO_HOST", "127.0.0.1"),
            port=int(os.getenv("PLUGIN_MUJOCO_PORT", "8090")),
            render_backend=os.getenv("MUJOCO_GL", "egl").lower(),
            backend=os.getenv("PLUGIN_MUJOCO_BACKEND", "mujoco").lower(),
            # 进程入口服务于Web、Pilot和真机等价的实时控制，默认必须给HTTP、
            # 传感与命令线程留出调度窗口。离线批处理仍可显式设置为0，或在
            # 测试中直接构造Settings(realtime=False)。
            realtime=os.getenv("PLUGIN_MUJOCO_REALTIME", "1") == "1",
            authoring_root=(
                Path(os.getenv("PLUGIN_MUJOCO_AUTHORING_ROOT", "")).expanduser().resolve()
                if os.getenv("PLUGIN_MUJOCO_AUTHORING_ROOT", "").strip()
                else None
            ),
            public_endpoint=os.getenv("PLUGIN_MUJOCO_PUBLIC_ENDPOINT") or None,
        )
