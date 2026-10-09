# Semantic MuJoCo Runtime

[English](README.md) | [简体中文](README.zh-CN.md)

`plugin-mujoco` is a MuJoCo Runtime started per Scene. In production it is started and stopped by Semantic Framework; developers can also run it standalone with `uv run`. It loads pre-built Scenes, advances physics simulation, executes low-level trajectories, exports GLB scenes and pose streams, and produces Robot sensor data.

The Plugin does not compute inverse kinematics (IK), navigation paths, or grasp policies, and it does not include Abilities, Robot Skills, Agents, MCP, or the Semantic Map. In the full Robot execution chain, `semantic-robot-sdk` plans trajectories and this Runtime's RobotDriver executes them on the physics thread.

## Project Structure

- `src/plugin_mujoco/`: API, Scene loading and compilation, physics, robots, sensors, and streaming.
- `packages/mujoco-visuals/`: shared visualization package.
- `profiles/`: native and optional environment profiles.
- `tools/` · `runtime-packs/`: Runtime packaging.
- `tests/`: unit and integration tests.

## Current Implementation Boundaries

- `native-mujoco`: Runtime lifecycle, asynchronous model loading, three depalletizing layouts, virtual R1 Pro, low-level trajectories, GLB/pose streams, RGB/Depth/Contact/Holding, Scene snapshots, stop/hold/reset, and generation isolation are implemented.
- `robosuite-1.5`: isolated Lift/Stack Runtime, Franka absolute joint trajectories, gripper, GLB/pose streams, RGB/Depth/Contact, lifecycle, stop/hold/reset, and native reward/success evidence are implemented.
- `libero-robosuite-1.4`: isolated Runtime with fixed suite/task/init-state, running through the same Franka, visual export, sensor, lifecycle, and evaluator interfaces. LIBERO-Pro reuses this environment and produces perturbation and baseline/perturbation comparison reports through a separate runner.
- Fake Backend: only for unit tests and parallel Framework/Web development; it is not acceptance evidence for native or benchmark Profiles.

The only Robot commands the Runtime officially accepts are:

- `joint_trajectory`
- `base_trajectory`
- `gripper_command`

`stop` and `hold` use separate endpoints. End-effector poses, navigation goals, MuJoCo action arrays, and body/geom IDs never enter the public commands.

## Source Repository and Runtime Packs

This repository is development source code, not the installation directory on a production machine. Product releases build three offline Runtime Packs from pinned `v0.4.x` tags:

- `native-mujoco`: Plugin Wheel, native dependencies, the shared depalletizing directory, and Layout templates;
- `robosuite-1.5`: Profile Wheel, pinned robosuite 1.5 dependencies, and the Lift/Stack index;
- `libero-robosuite-1.4`: Profile Wheel, pinned robosuite 1.4 dependencies, and the LIBERO index.

A Pack contains the pinned Python version, an offline wheelhouse, dependency locks, the Scene index, smoke requests, license notices, and version verification files. It does not contain Projects, Meshes, R1/Franka models, LIBERO/LIBERO-Pro source code, or benchmark data. Large or restricted content is provided by the administrator at install time as read-only directories, and Framework writes the resolved absolute paths into `RuntimeInstallation`.

~~~bash
# CI/release owners build Packs; regular users do not run these commands.
make runtime-pack-native PACK_VERSION=0.4.0
make runtime-pack-robosuite PACK_VERSION=0.4.0
make runtime-pack-libero PACK_VERSION=0.4.0
~~~

The Builder pre-builds `semantic-sim-profiles` and any dependency without an upstream binary artifact into Wheels in CI. For a release tag build, the Pack SemVer must match the METADATA version of every Wheel, so `0.4.0.dev0` source cannot masquerade as an official `0.4.0` Pack. Running `make runtime-pack-native` directly on a development branch produces `0.4.0-dev.0` by default. An official Pack contains no `pip install -e`, no source-relative paths, and no on-site build steps. This repository does not yet carry an approved public product license, so the generated artifacts are restricted to authorized internal deployments; do not mistake the notice files for an open-source license.

## Software Layering

```text
api             HTTP and binary stream boundary
application     use-case entry points
compiler        build entry from SceneDocument to a MuJoCo scene
loaders         Runtime Profile detection and component assembly
runtime         lifecycle, physics thread, and command dispatch
robots          low-level trajectory RobotDriver and Robot mapping
sensors         sensor reading
rendering       offscreen rendering executors for RGB/Depth sensors
visuals         MjModel/MjData to GLB and pose streams
assets          conversion of engine-internal objects to public snapshots
streaming       latest-frame buffer shared by multiple clients
native          concrete native MuJoCo implementation
```

The physics thread is the only thread allowed to modify MuJoCo `MjModel/MjData`. The API, visual export, pose stream, and sensor threads only submit operations or read snapshots; a slow stream client only drops stale data and never blocks physics stepping.

## Running from Source

This package requires **Python >=3.10,<3.13** and must be isolated from the Robot's Python 3.13. Rendering on Linux needs a compatible EGL / Mesa driver and separately obtained, licensed Scene assets.

```bash
uv sync --python 3.10 --frozen --extra dev
export MUJOCO_ASSET_ROOT=/absolute/path/to/mujoco_asset
export MUJOCO_GL=egl
uv run plugin-mujoco
```

The service listens on `127.0.0.1:8090` by default. To allow remote access on a trusted development network, set `PLUGIN_MUJOCO_HOST=0.0.0.0`. In managed use, quick-start registers the Runtime with the Server; starting the process manually does not complete Runtime registration.

```bash
curl http://127.0.0.1:8090/healthz
curl http://127.0.0.1:8090/api/v1/runtime
curl http://127.0.0.1:8090/api/v1/runtime-profiles
curl http://127.0.0.1:8090/api/v1/scenes
```

FastAPI `/docs` lists every endpoint. Framework registers validated Scenes via `/api/v1/runtime-bundles` and uses the Scene instance endpoints to start, pause, step, resume, reset, and stop.

robosuite and LIBERO must each run in their own locked environment and are never mixed with the native Runtime:

```bash
# robosuite Lift / Stack, default 127.0.0.1:8091
MUJOCO_GL=egl \
SEMANTIC_SIM_PROFILE=robosuite-1.5 \
PLUGIN_MUJOCO_PORT=8091 \
uv run --project profiles/robosuite --frozen semantic-sim-runtime

# LIBERO native task catalog, default 127.0.0.1:8092
MUJOCO_GL=egl \
SEMANTIC_SIM_PROFILE=libero-robosuite-1.4 \
SEMANTIC_LIBERO_ROOT=/absolute/path/to/LIBERO \
PLUGIN_MUJOCO_PORT=8092 \
uv run --project profiles/libero --frozen semantic-sim-runtime
```

For LIBERO, `GET /api/v1/scenes` reads all suites/tasks, language descriptions, and actually usable initial states from the pinned upstream installation. `scene_key` is `suite:task_id` and `layout` is `init-N`, numbered in the upstream benchmark's default order; the catalog is no longer restricted by `SEMANTIC_LIBERO_SUITE/TASK_ID`. When a BDDL file or initial state is missing, the entry returns `available=false` with the reason. A readable catalog only means the native task resources are installed; it does not mean a model can complete the task. The catalog cache refreshes when the Runtime process restarts.

### Optional LIBERO Policy Control Interface (in development)

The default remains the original `JOINT_POSITION` joint trajectory; a standalone policy deployment may set `SEMANTIC_LIBERO_CONTROLLER=OSC_POSE`. A Robot Profile with this configuration only declares `control_sequence` and does not pretend to be a joint-trajectory controller. End-effector inputs are metric displacements in world coordinates and axis-angle increments in radians; gripper direction is positive to close, negative to open, and zero to hold. Inside the Runtime they are mapped to upstream actions with the actual OSC scaling. Each sequence sample executes exactly one 0.05-second control period.

- `POST /api/v1/robots/{robot_id}/commands`: `type=control_sequence`, carrying `execution_id`, `scene_generation`, and `control_sequence.control_period_s/samples`.
- `POST /api/v1/robots/{robot_id}/executions/{execution_id}/cancel`: cancels the execution and rejects late sequences for that identity; the request body carries `scene_generation`. The existing command status endpoints are reused.
- `GET /api/v1/robots/{robot_id}/observation`: state, raw dual-camera RGB, and two-finger joint state at the same sampling instant. It keeps the four-byte big-endian JSON header length + JSON + binary buffer format, does not flip pixels, does not transcode through JPEG, and never passes a display image off as model input.

Completing a sequence does not mean the grasp or the native task completed. The physical world keeps advancing while the queue is empty, a command finishes, or inference is pending; the Scene does not auto-reset on the native horizon. `scripts/verify_libero_control.py` is the standalone real-environment interface acceptance, recorded separately from VLA/Agent task acceptance. Integration verification is still in progress; these endpoints or unit tests alone cannot prove SmolVLA works.

These two Profiles are read-only native environments, so they reject `RuntimeBundle` and SceneDocument edits. They share the Scene lifecycle, Franka description, low-level trajectories, GLB/pose streams, and sensor protocols with the native Runtime. RGB is exported as JPEG; Profile depth stays a raw `float32-le` buffer and is not passed off as millimeter depth before the camera near/far plane conversion is done.

## Standalone Acceptance

With the native Runtime running, execute:

```bash
uv run plugin-mujoco-demo --layout layout001 --output .output/demo-layout001
```

This client verifies low-level base and joint trajectories, visual scene/pose streams, sensors, pause/step/resume, reset, and generation, and saves JPEGs, 16-bit Depth PNGs, Scene snapshots, and a JSON report. It does not fake IK, navigation, or a complete depalletizing success; those capabilities are accepted by combined tests of `semantic-robot-sdk` and the Abilities.

`Holding` only means the same object is in real contact with both fingers of the same-side gripper at the same time. The Runtime never disables object collision, modifies object poses, or creates hidden attachments. The smallest edge of the current depalletizing totes is 0.34 m, while the effective opening of a single R1 Pro gripper is about 0.1 m; therefore "carrying the existing totes with a single gripper" is not a passable physical acceptance item today. The product closed loop must first settle on suction cups, dual-arm holding, or dedicated grippable test pieces — grasp success must not be faked inside the Runtime.

The other two layouts can run lifecycle and sensor smoke checks:

```bash
uv run plugin-mujoco-demo --layout layout002 --scene-smoke --output .output/demo-layout002
uv run plugin-mujoco-demo --layout layout003 --scene-smoke --output .output/demo-layout003
```

## Testing

```bash
make lint
make test
MUJOCO_ASSET_ROOT=/absolute/path/to/mujoco_asset MUJOCO_GL=egl make test-native
make test-profiles
make build
```

- `make test` uses the Fake Backend to verify domain models, command queues, lifecycle, interfaces, and stream buffers.
- `make test-native` must load real MuJoCo and assets and reports native code coverage separately; when assets are not provided the command fails explicitly and must not be recorded as passed.
- `make test-profiles` verifies the robosuite/LIBERO runners, Profile Runtime lifecycle, low-level trajectories, sensors, and binary streams; real-environment demos still need to be started separately in the corresponding locked environment.

`uv build` produces `dist/` with the Python distribution artifacts. Release maintainers can use `make runtime-pack-native` to produce a versioned Runtime Pack; check versions and dependency inputs before distribution.

## Asset Boundary

The plugin does not copy scenes, meshes, or materials. Point `MUJOCO_ASSET_ROOT` at the asset repository. The current depalletizing entry is:

```text
scene/palletizing_depalletizing_001/
├── scene_info.yaml
├── layout001.yaml
├── layout002.yaml
├── layout003.yaml
└── palletizing_depalletizing_001.xml
```

Public coordinates use meters, radians, seconds, and `[x, y, z, w]` quaternions. When an engine uses a different order internally, the conversion must be explicit at the boundary.

End users do not need to export these variables or run the above processes by hand. An administrator performs a one-time install with the Framework CLI:

~~~bash
semantic runtime install --pack semantic-native-mujoco-0.4.0.runtime.tar.zst \
  --asset-root /data/semantic/mujoco-assets
semantic runtime doctor --all
semantic-server
~~~

Framework starts the Runtime on demand after a Project is opened and reclaims the managed process when the Project exits. Source mode is for development only: `semantic runtime install --dev-source /path/to/plugin-mujoco --profile native-mujoco ...`; the installation is explicitly marked `development` and cannot be used for an RC.

## Using the Artifacts

A Runtime Pack contains the runtime and its dependencies, but **not** robot models, Scene assets, or Robot Bundles. The Server manages the project / scene lifecycle; an idle Runtime or one with no Scene yet does not necessarily indicate a failed installation.

The current one-click deployment package targets Linux x86_64 and has been verified on Ubuntu 24.04. Source availability or cross-platform upstream Wheels does not mean the full Runtime has been verified on every operating system.

## FAQ

- Assets not found: check the asset root, the directory manifest, and the LFS download.
- Rendering failure: check the EGL / OSMesa configuration and drivers.
- Robot offline: in addition to the simulation service, check Server / Pilot / Bundle / Skill status.
- Before redistributing, check asset provenance and the [license scope](LICENSE_SCOPE.md).

[Detailed technical reference](README.reference.md) · [Packaging targets](Makefile)

## License

Copyright 2026 InsightOS. First-party code is licensed under [Apache-2.0](LICENSE); for third-party components and assets see [NOTICE](NOTICE) and the [license scope](LICENSE_SCOPE.md).

## Build Reproduction on Three Platforms

See the [glibc, musl, and macOS build notes](README.build.md): pinned source versions, actual script entry points, tool requirements, local and CI instructions, artifact locations, and platform verification scope.

## Windows Native Validation

Native CI for Windows x64 has been added with CPython 3.13.15, NumPy 2.3.5, and MuJoCo 3.4.0, covering the API, scene validation, clean exit, and physics stepping. Build and reproduction commands are in the [Windows build notes](README.build.md#windows-x64-native-validation). Physics tests in regular CI do not constitute GPU rendering validation; real desktop/driver testing must be recorded separately. Integration progress of the full quick-start Windows installer is tracked in the quick-start repository.
