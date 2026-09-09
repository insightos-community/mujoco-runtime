> Historical technical reference / 历史技术参考。For current build and usage instructions, see [English](README.md) / [中文](README.zh-CN.md). Version-specific examples below are not a current release manifest.

# Semantic MuJoCo Runtime

`plugin-mujoco` 是按场景启动的 MuJoCo Runtime。产品环境由 Semantic Framework 启动和停止；开发者也可以使用 `uv run` 独立运行。它负责加载已经构建的场景、推进物理仿真、执行低层轨迹、导出 GLB 场景与位姿流，并生成 Robot 传感器数据。

Plugin 不计算逆运动学（IK）、导航路径或抓取策略，也不包含 Ability、Robot Skill、Agent、MCP 或 Semantic Map。完整 Robot 执行链由 `semantic-robot-sdk` 规划轨迹，再由本 Runtime 的 RobotDriver 在物理线程中执行。

## 当前实现边界

- `native-mujoco`：已实现 Runtime 生命周期、异步模型加载、三套拆码垛布局、虚拟 R1 Pro、低层轨迹、GLB/位姿流、RGB/Depth/Contact/Holding、场景快照、stop/hold/reset 和 generation 隔离。
- `robosuite-1.5`：已实现 Lift/Stack 隔离 Runtime、Franka 绝对关节轨迹、夹爪、GLB/位姿流、RGB/Depth/Contact、生命周期、stop/hold/reset 和原生 reward/success 证据。
- `libero-robosuite-1.4`：已实现固定 suite/task/init-state 的隔离 Runtime，并通过同一 Franka、视觉导出、传感器、生命周期和 evaluator 接口运行。LIBERO-Pro 复用该环境，通过独立 runner 生成扰动与基础/扰动对比报告。
- Fake Backend：只用于单元测试和 Framework/Web 并行开发，不能作为 native 或 benchmark Profile 的验收证据。

Runtime 正式接收的 Robot 命令只有：

- `joint_trajectory`
- `base_trajectory`
- `gripper_command`

`stop` 和 `hold` 使用独立端点。末端位姿、导航目标、MuJoCo action array、body/geom ID 都不进入公共命令。

## 源码仓库与 Runtime Pack

本仓库是开发源码，不是正式机器上的安装目录。产品发布由固定 `v0.4.x`
Tag 构建三个离线 Runtime Pack：

- `native-mujoco`：Plugin Wheel、native 依赖、拆码垛公共目录与 Layout 模板；
- `robosuite-1.5`：Profile Wheel、robosuite 1.5 固定依赖和 Lift/Stack 索引；
- `libero-robosuite-1.4`：Profile Wheel、robosuite 1.4 固定依赖及 LIBERO 索引。

Pack 内包含固定 Python 版本、离线 Wheelhouse、依赖锁、场景索引、smoke
请求、许可提示和版本验证文件。它不包含 Project、Mesh、R1/Franka 模型、
LIBERO/LIBERO-Pro 源码或 benchmark 数据。大型和受限内容由管理员安装时提供
只读目录，Framework 把解析后的绝对路径写入 `RuntimeInstallation`。

~~~bash
# CI/发布负责人构建 Pack；普通用户不执行这些命令。
make runtime-pack-native
make runtime-pack-robosuite
make runtime-pack-libero
~~~

Builder 会把 `semantic-sim-profiles` 和没有上游二进制制品的依赖都在 CI 中预构建
为 Wheel；正式 Tag 构建时，Pack SemVer 必须与全部 Wheel 的 METADATA 版本一致，
因此 `0.4.0.dev0` 源码不能伪装成正式 `0.4.0` Pack。开发分支直接运行 `make
runtime-pack-native` 时默认生成 `0.4.0-dev.0`。正式 Pack 不包含 `pip install -e`、
源码相对路径或现场构建步骤。自有代码与第三方组件的许可范围见
[LICENSE_SCOPE.md](LICENSE_SCOPE.md)；资产仍需独立审查分发许可。

## 软件分层

```text
api             HTTP 与二进制流边界
application     用例入口
compiler        SceneDocument 到 MuJoCo 场景的构建入口
loaders         Runtime Profile 探测与组件装配
runtime         生命周期、物理线程和命令调度
robots          低层轨迹 RobotDriver 与 Robot 映射
sensors         传感器读取
rendering       RGB/Depth 传感器离屏渲染执行器
visuals         MjModel/MjData 到 GLB 与位姿流
assets          引擎内部对象到公共快照的转换
streaming       多客户端共享的 latest-frame 缓冲
native          原生 MuJoCo 具体实现
```

物理线程是唯一可以修改 MuJoCo `MjModel/MjData` 的线程。API、视觉导出、位姿流和传感器线程只提交操作或读取快照；慢速流客户端只会丢弃旧数据，不会阻塞物理步进。

## 源码开发启动

```bash
uv sync --frozen --extra dev
export MUJOCO_ASSET_ROOT=/absolute/path/to/mujoco_asset
export MUJOCO_GL=egl
uv run plugin-mujoco
```

服务默认监听 `127.0.0.1:8090`。可信开发网络需要远程访问时，可以设置 `PLUGIN_MUJOCO_HOST=0.0.0.0`。

```bash
curl http://127.0.0.1:8090/healthz
curl http://127.0.0.1:8090/api/v1/runtime
curl http://127.0.0.1:8090/api/v1/runtime-profiles
curl http://127.0.0.1:8090/api/v1/scenes
```

FastAPI `/docs` 展示全部接口。Framework 使用 `/api/v1/runtime-bundles` 注册已校验场景，通过场景实例端点启动、暂停、单步、继续、reset 和 stop。

robosuite 和 LIBERO 必须使用各自锁定的环境，不与 native Runtime 混装：

```bash
# robosuite Lift / Stack，默认 127.0.0.1:8091
MUJOCO_GL=egl \
SEMANTIC_SIM_PROFILE=robosuite-1.5 \
PLUGIN_MUJOCO_PORT=8091 \
uv run --project profiles/robosuite --frozen semantic-sim-runtime

# LIBERO 固定任务，默认 127.0.0.1:8092
MUJOCO_GL=egl \
SEMANTIC_SIM_PROFILE=libero-robosuite-1.4 \
SEMANTIC_LIBERO_ROOT=/absolute/path/to/LIBERO \
PLUGIN_MUJOCO_PORT=8092 \
uv run --project profiles/libero --frozen semantic-sim-runtime
```

这两个 Profile 是只读原生环境，因此拒绝 `RuntimeBundle` 和 SceneDocument 编辑。它们与 native Runtime 共享场景生命周期、Franka 描述、低层轨迹、GLB/位姿流及传感器协议。RGB 输出 JPEG；Profile 深度保持 `float32-le` 原始缓冲，未完成相机近远平面换算前不会冒充毫米深度。

## 独立验收

启动 native Runtime 后执行：

```bash
uv run plugin-mujoco-demo --layout layout001 --output .output/demo-layout001
```

该客户端验证低层底盘和关节轨迹、视觉场景/位姿流、传感器、pause/step/resume、reset 与 generation，并保存 JPEG、16 位 Depth PNG、场景快照和 JSON 报告。它不会伪造 IK、导航或完整拆码垛成功；这些能力由 `semantic-robot-sdk` 与 Ability 的组合测试验收。

`Holding` 只表示同一个物体同时与同侧夹爪的两根手指发生真实接触。Runtime 不会关闭物体碰撞、修改物体位姿或建立隐藏附着。当前拆码垛箱体最小边为 0.34 米，而单个 R1 Pro 夹爪的有效开口约为 0.1 米；因此“单夹爪搬运现有箱体”不是当前可通过的物理验收项。产品闭环必须先确定吸盘、双臂抱持或专用可夹持测试件，不能在 Runtime 内伪造抓取成功。

另外两套布局可运行生命周期与传感器 smoke：

```bash
uv run plugin-mujoco-demo --layout layout002 --scene-smoke --output .output/demo-layout002
uv run plugin-mujoco-demo --layout layout003 --scene-smoke --output .output/demo-layout003
```

## 测试

```bash
make lint
make test
MUJOCO_ASSET_ROOT=/absolute/path/to/mujoco_asset MUJOCO_GL=egl make test-native
make test-profiles
make build
```

- `make test` 使用 Fake Backend 验证领域模型、命令队列、生命周期、接口和流缓冲。
- `make test-native` 必须加载真实 MuJoCo 与资产，并单独输出 native 代码覆盖率；未提供资产时命令会明确失败，不能记为通过。
- `make test-profiles` 验证 robosuite/LIBERO runner、Profile Runtime 生命周期、低层轨迹、传感器和二进制流；真实环境演示仍需分别使用对应锁定环境启动。

## 资产边界

插件不复制场景、mesh 或材质。通过 `MUJOCO_ASSET_ROOT` 指向资产仓。当前拆码垛入口为：

```text
scene/palletizing_depalletizing_001/
├── scene_info.yaml
├── layout001.yaml
├── layout002.yaml
├── layout003.yaml
└── palletizing_depalletizing_001.xml
```

公共坐标使用米、弧度、秒和 `[x, y, z, w]` 四元数。引擎内部采用其他顺序时必须在边界处显式转换。

正式用户不需要导出这些变量或手工运行上述进程。管理员使用 Framework CLI
完成一次安装：

~~~bash
semantic runtime install --pack semantic-native-mujoco-0.4.0.runtime.tar.zst \
  --asset-root /data/semantic/mujoco-assets
semantic runtime doctor --all
semantic-server
~~~

Framework 在 Project 打开后按需启动 Runtime，Project 退出后回收受管进程。
源码模式仅供开发：`semantic runtime install --dev-source /path/to/plugin-mujoco
--profile native-mujoco ...`，installation 会明确标记 `development`，不能用于 RC。
