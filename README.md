# Semantic MuJoCo Runtime

[English](README.md) | [简体中文](README.zh-CN.md)

🌐 A physics and rendering service for Semantic simulation. It loads scene assets and exposes low-level robot/sensor operations. Planning, IK policy, task Skills, and robot orchestration belong to other components.

## Structure

- `src/plugin_mujoco/` — API, scene loading/compilation, physics, robots, sensors, and streaming.
- `packages/mujoco-visuals/` — shared visualization package.
- `profiles/` — native and optional environment profiles.
- `tools/` · `runtime-packs/` — runtime packaging.
- `tests/` — unit and integration tests.

## 🛠 Develop and build

The current package requires **Python >=3.10,<3.13**. Use an environment separate from Robot's Python 3.13. Linux rendering requires compatible EGL/Mesa drivers and separately obtained, licensed scene assets.

```bash
uv sync --python 3.10 --frozen --extra dev
export MUJOCO_ASSET_ROOT=/absolute/path/to/mujoco-asset
export MUJOCO_GL=egl
uv run plugin-mujoco
```

The native development service uses port `8090`. For managed use, quick-start registers it with Server instead of treating a manually started process as a registered Runtime.

```bash
make test
uv build
```

`dist/` contains Python distributions. Release maintainers can use `make runtime-pack-native` to produce a versioned Runtime Pack; inspect its version and dependency inputs before distributing it.

## Use the outputs

Runtime Packs provide the runtime and dependencies, **not** Robot models, scene assets, or Robot Bundles. Server assigns project/scene lifecycle; an idle runtime or absent scene is not necessarily an installation failure.

The current one-click product bundle is Linux x86_64, verified on Ubuntu 24.04. Source availability and cross-platform upstream Wheels do not establish validation of this complete runtime on every OS.

## Native macOS development

Apple Silicon development uses Python 3.13.15 and NumPy 2.3.5. Darwin defaults
to the native CGL backend; no Linux Mesa or EGL bundle is required.

```bash
uv sync --frozen --python 3.13.15 --extra dev
uv run --frozen python tools/macos_smoke.py --output .output/macos-report.json
MUJOCO_ASSET_ROOT=/absolute/path/to/mujoco-asset uv run --frozen pytest tests/test_native_smoke.py -vv
```

The native CI validates CPU physics, 62 API/logic tests, and package builds on
macOS arm64. Standard hosted macOS 15 runners returned `invalid pixel format`
when requesting CGL. Their CPU result does **not** qualify graphics. For graphics
validation, run the commands above on a physical Apple Silicon Mac, or manually
run `Native macOS` with `graphics=true` after registering a runner with labels
`self-hosted`, `macOS`, `ARM64`, `semantic-graphics` in an active graphical session.
That job requires rendered pixels and all three real pallet layouts to pass.
A complete macOS one-click installer has not yet been released.

## Troubleshooting

- Asset not found: check the asset root, catalogs, and completed LFS downloads.
- Render failure: check EGL/OSMesa configuration and driver availability.
- Robot offline: inspect Server/Pilot/Bundle/Skill state as well as the simulation service.
- Review asset provenance and [license scope](LICENSE_SCOPE.md) before redistribution.

[Detailed technical reference](README.reference.md) · [Packaging targets](Makefile)

## License

Copyright 2026 InsightOS. First-party code: [Apache-2.0](LICENSE). See [NOTICE](NOTICE) and [license scope](LICENSE_SCOPE.md) for third-party components and assets.
