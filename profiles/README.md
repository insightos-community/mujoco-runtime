# MuJoCo 系隔离运行环境

这里保存 robosuite、LIBERO 和 LIBERO-Pro 的独立运行配置。它们共享测试报告格式，
但不会安装到原生 MuJoCo Runtime 的 Python 环境中。

## 环境边界

- 原生 Runtime 使用产品需要的 MuJoCo 与 R1 Pro 资产。
- robosuite 使用 Python 3.10、robosuite 1.5.2 和 MuJoCo 3.4.0。
- LIBERO 使用 Python 3.8、robosuite 1.4.0 和 NumPy 1.22.4。
- LIBERO-Pro 复用 LIBERO Runtime，只增加固定源码的 BDDL 扰动与对比报告。

env.step() 和 action array 只存在于 profile 内部，不会暴露给 Agent、Robot Skill 或 Studio。

## 公共测试

    make test-profiles

公共测试只验证运行器、报告、PNG 证据和错误处理，不能替代真实环境 smoke。

## robosuite

安装隔离环境：

    uv sync --project profiles/robosuite --python 3.10 --frozen

运行 Lift 和 Stack：

    MUJOCO_GL=egl uv run --project profiles/robosuite --frozen \
      semantic-sim-profile robosuite --environment Lift --seed 7 --steps 10 \
      --output-dir .output/profiles/robosuite-lift

    MUJOCO_GL=egl uv run --project profiles/robosuite --frozen \
      semantic-sim-profile robosuite --environment Stack --seed 7 --steps 10 \
      --output-dir .output/profiles/robosuite-stack

没有 EGL 时可把 MUJOCO_GL 改为 osmesa。报告包含 Observation 字段、reward、success、
RGB PNG 和 16 位 Depth PNG。中性控制 smoke 不代表任务已经完成。

## LIBERO 源码准备

上游 LIBERO 的普通 wheel 在固定提交上不会包含 libero 源码，因此本项目不安装该空 wheel。
请把 sources.lock.yaml 中的两个仓库检出到固定提交；运行时必须显式传入源码根目录。

    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git .output/sources/LIBERO
    git -C .output/sources/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
    git clone https://github.com/RLinf/LIBERO-PRO.git .output/sources/LIBERO-PRO
    git -C .output/sources/LIBERO-PRO checkout 0bcf73621c789ffd6ed8858467a89df9ca94fd6b

安装 Python 3.8 隔离环境：

    uv sync --project profiles/libero --python 3.8 --frozen

运行固定 LIBERO task/init-state：

    MUJOCO_GL=egl uv run --project profiles/libero --frozen \
      semantic-sim-profile libero --libero-root .output/sources/LIBERO \
      --suite libero_spatial --task-id 0 --init-state-id 0 --seed 7 --steps 10 \
      --output-dir .output/profiles/libero

运行 LIBERO-Pro 基础/扰动对比：

    MUJOCO_GL=egl uv run --project profiles/libero --frozen \
      semantic-sim-profile libero-pro \
      --libero-root .output/sources/LIBERO \
      --libero-pro-root .output/sources/LIBERO-PRO \
      --evaluation-config profiles/libero/libero-pro.example.yaml \
      --perturbation environment \
      --suite libero_spatial --task-id 0 --init-state-id 0 --seed 7 --steps 10 \
      --output-dir .output/profiles/libero-pro

启动时会校验两个源码提交。LIBERO 配置写入本次输出目录，不会修改用户主目录。
LIBERO-Pro 的 Python 3.10 注解通过加载兼容层延迟解析，上游源码本身保持不变。

## 输出与判定

- report.json：固定环境、seed、步数、语言目标、reward、success 和依赖版本。
- comparison.json：LIBERO-Pro 基础与扰动结果及差异。
- initial/final RGB PNG：验收画面。
- initial/final Depth PNG：归一化的 16 位深度证据。
- 运行错误时仍写 report.json，同时 CLI 返回非零退出码。

外部数据集不进入 Plugin 制品。Framework、Studio、Pilot、AbilityFramework 和
Semantic Map 的产品接线由各自版本分支完成。
