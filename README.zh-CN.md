# Semantic MuJoCo Runtime

[English](README.md) | [简体中文](README.zh-CN.md)

🌐 Semantic 的物理仿真与渲染服务，负责加载场景资产并提供底层机器人 / 传感器操作。规划、IK 策略、任务 Skill 和机器人调度属于其他组件。

## 工程结构

- `src/plugin_mujoco/`：API、场景加载与编译、物理、机器人、传感器与流传输。
- `packages/mujoco-visuals/`：公共可视化包。
- `profiles/`：原生与可选环境配置。
- `tools/` · `runtime-packs/`：Runtime 打包。
- `tests/`：单元与集成测试。

## 🛠 开发与构建

当前包要求 **Python >=3.10,<3.13**，需要与 Robot 的 Python 3.13 隔离。Linux 渲染需要兼容的 EGL / Mesa 驱动及独立获取、具有许可的场景资产。

```bash
uv sync --python 3.10 --frozen --extra dev
export MUJOCO_ASSET_ROOT=/absolute/path/to/mujoco-asset
export MUJOCO_GL=egl
uv run plugin-mujoco
```

原生开发服务使用 `8090` 端口。托管使用时由 quick-start 向 Server 登记，手动启动进程不等于完成 Runtime 登记。

```bash
make test
uv build
```

`dist/` 包含 Python 分发产物。发布维护者可用 `make runtime-pack-native` 生成版本化 Runtime Pack，分发前检查版本与依赖输入。

## 产物使用

Runtime Pack 包含运行时及依赖，**不包含**机器人模型、场景资产或 Robot Bundle。Server 管理项目 / 场景生命周期；Runtime 空闲或尚无场景不一定表示安装失败。

当前一键部署包为 Linux x86_64，已在 Ubuntu 24.04 验证。源码可用或上游 Wheel 跨平台，不等于整套 Runtime 已验证所有操作系统。

## 常见问题

- 找不到资产：检查资产根目录、目录清单与 LFS 下载。
- 渲染失败：检查 EGL / OSMesa 配置和驱动。
- Robot 离线：除仿真服务外，同时检查 Server / Pilot / Bundle / Skill 状态。
- 再分发前检查资产来源和[许可范围](LICENSE_SCOPE.md)。

[详细技术参考](README.reference.md) · [打包目标](Makefile)

## 许可证

Copyright 2026 InsightOS。自有代码采用 [Apache-2.0](LICENSE)；第三方组件与资产请查看 [NOTICE](NOTICE) 和[许可范围](LICENSE_SCOPE.md)。
