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

"""进程入口。"""

from __future__ import annotations

import sys

import uvicorn

from plugin_mujoco.api import create_app
from plugin_mujoco.settings import Settings

settings = Settings.from_env()
app = create_app(settings)


def run() -> None:
    if sys.platform != "win32":
        uvicorn.run(app, host=settings.host, port=settings.port)
        return
    from plugin_mujoco.windows_stop import stop_endpoint

    server = uvicorn.Server(uvicorn.Config(app, host=settings.host, port=settings.port))
    with stop_endpoint(lambda: setattr(server, "should_exit", True)):
        server.run()


if __name__ == "__main__":
    run()
