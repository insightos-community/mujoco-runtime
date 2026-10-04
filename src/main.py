"""兼容旧启动方式；正式入口位于 plugin_mujoco.main。"""

from plugin_mujoco.main import app, run

__all__ = ["app", "run"]


if __name__ == "__main__":
    run()
