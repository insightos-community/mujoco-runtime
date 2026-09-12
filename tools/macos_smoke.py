"""Run native macOS physics and offscreen rendering; preserve failure evidence."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"system": platform.system(), "machine": platform.machine(), "python": sys.version,
              "physics": "not-run", "rendering": "not-run"}
    try:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("This validation requires native macOS arm64")
        os.environ["MUJOCO_GL"] = "cgl"
        import mujoco
        import numpy as np
        from plugin_mujoco.settings import Settings

        assert Settings.from_env().render_backend == "cgl"
        report.update(mujoco=mujoco.__version__, numpy=np.__version__, backend="cgl")
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <worldbody><light pos="0 0 3"/><geom type="plane" size="2 2 .1"/>
          <body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" rgba="1 0 0 1"/></body>
          </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        for _ in range(100):
            mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and data.qpos[2] < 1
        report["physics"] = "passed"
        with mujoco.Renderer(model, height=120, width=160) as renderer:
            renderer.update_scene(data)
            pixels = renderer.render()
            assert pixels.shape == (120, 160, 3) and pixels.max() > pixels.min()
            from OpenGL.GL import GL_RENDERER, GL_VENDOR, GL_VERSION, glGetString
            report["opengl"] = {
                key: (glGetString(value) or b"").decode("utf-8", errors="replace")
                for key, value in (("renderer", GL_RENDERER), ("vendor", GL_VENDOR), ("version", GL_VERSION))
            }
        report["rendering"] = "passed"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
