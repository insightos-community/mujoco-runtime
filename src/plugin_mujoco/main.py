"""进程入口。"""

from __future__ import annotations

import uvicorn

from plugin_mujoco.api import create_app
from plugin_mujoco.settings import Settings

settings = Settings.from_env()
app = create_app(settings)


def run() -> None:
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
