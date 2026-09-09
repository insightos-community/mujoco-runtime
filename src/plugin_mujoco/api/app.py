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

"""Runtime HTTP API。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from plugin_mujoco import __version__
from plugin_mujoco.application import RuntimeManager
from plugin_mujoco.compiler import RuntimeBundle, RuntimeBundleResult
from plugin_mujoco.errors import ConflictError, RuntimeErrorBase
from plugin_mujoco.models import (
    RobotCommand,
    RobotCommandRequest,
    RobotCommandStatus,
    RobotHoldRequest,
    RobotOperationResult,
    RobotProfile,
    RobotState,
    RuntimeInfo,
    RuntimeProfile,
    SceneDescriptor,
    SceneInstance,
    SceneSnapshot,
    SceneStartRequest,
    SceneStepRequest,
    SensorDescriptor,
    SensorFrame,
    utc_now,
)
from plugin_mujoco.settings import Settings
from plugin_mujoco.visuals import RuntimeVisualAssetStore, ViewerScene


def create_app(
    settings: Settings | None = None,
    manager: RuntimeManager | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    manager = manager or RuntimeManager(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(
        title="Semantic MuJoCo Runtime",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.runtime_manager = manager
    app.state.visual_assets = RuntimeVisualAssetStore(settings.asset_root)

    @app.exception_handler(RuntimeErrorBase)
    async def handle_runtime_error(
        _: Request,
        error: RuntimeErrorBase,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": error.details,
                }
            },
        )

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    runtime = APIRouter(prefix="/api/v1")

    @runtime.get("/runtime", response_model=RuntimeInfo)
    def runtime_info(request: Request) -> RuntimeInfo:
        return _manager(request).info()

    @runtime.get("/runtime-profiles", response_model=list[RuntimeProfile])
    def runtime_profiles(request: Request) -> list[RuntimeProfile]:
        return _manager(request).runtime_profiles()

    @runtime.get("/scenes", response_model=list[SceneDescriptor])
    def list_scenes(request: Request) -> list[SceneDescriptor]:
        return _manager(request).catalog.list()

    @runtime.get("/visual-assets/{visual_id}/{version}.glb")
    def visual_asset(visual_id: str, version: str, request: Request) -> Response:
        """返回 Runtime Pack 当前安装内容对应的浏览器模型。"""
        content = request.app.state.visual_assets.content(visual_id, version)
        return Response(
            content=content,
            media_type="model/gltf-binary",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "X-Semantic-Visual": f"{visual_id}@{version}",
            },
        )

    @runtime.post("/runtime-bundles", response_model=RuntimeBundleResult, status_code=201)
    def register_runtime_bundle(
        payload: RuntimeBundle,
        request: Request,
    ) -> RuntimeBundleResult:
        """编译 Framework 已校验的 RuntimeBundle，不保存 Studio 草稿状态。"""
        return _manager(request).register_runtime_bundle(payload)

    @runtime.post(
        "/scenes/{scene_key}/instances",
        response_model=SceneInstance,
        status_code=201,
    )
    def start_scene(
        scene_key: str,
        payload: SceneStartRequest,
        request: Request,
    ) -> SceneInstance:
        return _manager(request).start(scene_key, payload)

    @runtime.get("/scene-instances/{instance_id}", response_model=SceneInstance)
    def get_scene(instance_id: str, request: Request) -> SceneInstance:
        return _manager(request).view(instance_id)

    @runtime.post("/scene-instances/{instance_id}/pause", response_model=SceneInstance)
    def pause_scene(instance_id: str, request: Request) -> SceneInstance:
        return _manager(request).get(instance_id).pause()

    @runtime.post("/scene-instances/{instance_id}/resume", response_model=SceneInstance)
    def resume_scene(instance_id: str, request: Request) -> SceneInstance:
        return _manager(request).get(instance_id).resume()

    @runtime.post("/scene-instances/{instance_id}/reset", response_model=SceneInstance)
    def reset_scene(instance_id: str, request: Request) -> SceneInstance:
        return _manager(request).get(instance_id).reset()

    @runtime.post("/scene-instances/{instance_id}/stop", response_model=SceneInstance)
    def stop_scene(instance_id: str, request: Request) -> SceneInstance:
        return _manager(request).stop(instance_id)

    @runtime.get(
        "/scene-instances/{instance_id}/snapshot",
        response_model=SceneSnapshot,
    )
    def scene_snapshot(instance_id: str, request: Request) -> SceneSnapshot:
        return _manager(request).get(instance_id).snapshot()

    @runtime.get(
        "/scene-instances/{instance_id}/robots",
        response_model=list[RobotProfile],
    )
    def scene_robots(instance_id: str, request: Request) -> list[RobotProfile]:
        instance = _manager(request).get(instance_id)
        return [
            instance.components.robots.robot_profile(robot_id) for robot_id in instance.robot_ids()
        ]

    @runtime.post("/scene-instances/{instance_id}/step", response_model=SceneInstance)
    def step_scene(
        instance_id: str,
        payload: SceneStepRequest,
        request: Request,
    ) -> SceneInstance:
        return _manager(request).get(instance_id).step(payload.steps)

    @runtime.get(
        "/scene-instances/{instance_id}/viewer-scene",
        response_model=ViewerScene,
    )
    def viewer_scene(instance_id: str, request: Request) -> ViewerScene:
        return _manager(request).get(instance_id).viewer_scene()

    @runtime.get("/scene-instances/{instance_id}/viewer-scene/content")
    def viewer_scene_content(instance_id: str, request: Request) -> Response:
        instance = _manager(request).get(instance_id)
        return Response(
            content=instance.viewer_scene_content(),
            media_type="model/gltf-binary",
            headers={
                "Cache-Control": "private, max-age=31536000, immutable",
                "X-Semantic-Scene-Revision": instance.components.visuals.scene_revision,
            },
        )

    @runtime.websocket("/scene-instances/{instance_id}/pose-stream")
    async def scene_pose_stream(instance_id: str, socket: WebSocket) -> None:
        await socket.accept()
        sequence = 0
        try:
            instance = _socket_manager(socket).get(instance_id)
            generation = instance.view().generation
            while True:
                frame = await asyncio.to_thread(
                    instance.scene_pose_frame, after_sequence=sequence
                )
                if frame.metadata.generation != generation:
                    await socket.close(code=1000)
                    return
                sequence = frame.metadata.sequence
                await socket.send_bytes(frame.packet())
        except WebSocketDisconnect:
            return
        except (RuntimeErrorBase, TimeoutError):
            await socket.close(code=1011)


    app.include_router(runtime)

    robot = APIRouter()

    @robot.get("/robots/{robot_id}/profile", response_model=RobotProfile)
    def robot_profile(robot_id: str, request: Request) -> RobotProfile:
        return _active(request).components.robots.robot_profile(robot_id)

    @robot.get("/robots/{robot_id}/state", response_model=RobotState)
    def robot_state(robot_id: str, request: Request) -> RobotState:
        instance = _active(request)
        return instance.components.robots.robot_state(robot_id, instance.record.generation)

    @robot.websocket("/robots/{robot_id}/state/stream")
    async def robot_state_stream(robot_id: str, socket: WebSocket) -> None:
        """输出正式 Robot State 流；短路径和 /api/v1 路径共用此实现。"""

        await socket.accept()
        sequence = 0
        stream_generation: int | None = None
        try:
            while True:
                instance = _socket_manager(socket).active_instance()
                current_generation = instance.view().generation
                if stream_generation != current_generation:
                    # reset 后 LatestFrameHub 会创建新生产器，sequence 也从 1 重新开始。
                    sequence = 0
                frame = await asyncio.to_thread(
                    instance.encoded_robot_state,
                    robot_id,
                    after_sequence=sequence,
                )
                if frame.metadata.generation != instance.view().generation:
                    # reset 与等待帧并发时可能取到已失效缓存；该帧绝不能发送给 SDK。
                    sequence = 0
                    continue
                stream_generation = frame.metadata.generation
                sequence = frame.metadata.sequence
                await socket.send_bytes(frame.packet())
        except WebSocketDisconnect:
            return
        except RuntimeErrorBase:
            await socket.close(code=1011)

    @robot.post(
        "/robots/{robot_id}/commands",
        response_model=RobotCommandStatus,
        status_code=202,
    )
    def submit_command(
        robot_id: str,
        payload: RobotCommandRequest,
        request: Request,
    ) -> RobotCommand:
        return _active(request).submit_command(robot_id, payload)

    @robot.get(
        "/robots/{robot_id}/commands/{command_id}",
        response_model=RobotCommandStatus,
    )
    def get_command(robot_id: str, command_id: str, request: Request) -> RobotCommand:
        return _active(request).command(robot_id, command_id)

    @robot.post(
        "/robots/{robot_id}/commands/{command_id}/stop",
        response_model=RobotCommandStatus,
    )
    def stop_command(robot_id: str, command_id: str, request: Request) -> RobotCommand:
        return _active(request).stop_command(robot_id, command_id)

    @robot.post("/robots/{robot_id}/hold", response_model=RobotOperationResult)
    def hold_robot(
        robot_id: str,
        payload: RobotHoldRequest,
        request: Request,
    ) -> RobotOperationResult:
        instance = _active(request)
        if payload.scene_generation != instance.record.generation:
            raise ConflictError(
                "hold generation 已失效",
                details={
                    "requested": payload.scene_generation,
                    "current": instance.record.generation,
                },
            )
        instance.hold_robot(robot_id)
        return RobotOperationResult(
            command_id=f"hold-{robot_id}-{payload.scene_generation}",
            robot_id=robot_id,
            scene_generation=payload.scene_generation,
            type="hold",
            status="succeeded",
            message="Robot 已进入 hold",
            updated_at=utc_now(),
        )

    @robot.get("/robots/{robot_id}/sensors", response_model=list[SensorDescriptor])
    def robot_sensors(robot_id: str, request: Request) -> list[SensorDescriptor]:
        return _active(request).components.sensors.sensor_descriptors(robot_id)

    @robot.get(
        "/robots/{robot_id}/sensors/{sensor_id}/frames/latest",
        response_model=SensorFrame,
    )
    def latest_sensor_frame(
        robot_id: str,
        sensor_id: str,
        request: Request,
    ) -> SensorFrame:
        instance = _active(request)
        frame = instance.components.sensors.sensor_frame(robot_id, sensor_id)
        return frame.model_copy(update={"generation": instance.record.generation})

    @robot.get("/robots/{robot_id}/sensors/{sensor_id}/frames/latest/content")
    def latest_sensor_content(
        robot_id: str,
        sensor_id: str,
        request: Request,
    ) -> Response:
        frame = _active(request).encoded_sensor_frame(robot_id, sensor_id)
        metadata = frame.metadata
        return Response(
            content=frame.payload,
            media_type=metadata.media_type,
            headers={
                "X-Semantic-Sequence": str(metadata.sequence),
                "X-Semantic-Generation": str(metadata.generation),
                "X-Semantic-Encoding": metadata.encoding,
                "X-Semantic-Frame": metadata.frame,
                "X-Semantic-Width": str(metadata.width),
                "X-Semantic-Height": str(metadata.height),
                "X-Semantic-Observed-At": metadata.observed_at.isoformat(),
            },
        )

    @robot.websocket("/robots/{robot_id}/sensors/{sensor_id}/stream")
    async def sensor_stream(robot_id: str, sensor_id: str, socket: WebSocket) -> None:
        await socket.accept()
        sequence = 0
        try:
            while True:
                instance = _socket_manager(socket).active_instance()
                frame = await asyncio.to_thread(
                    instance.encoded_sensor_frame, robot_id, sensor_id, after_sequence=sequence
                )
                sequence = frame.metadata.sequence
                await socket.send_bytes(frame.packet())
        except WebSocketDisconnect:
            return
        except RuntimeErrorBase:
            await socket.close(code=1011)

    # Robot SDK 按计划保留短路径；同时提供 /api/v1 前缀，便于统一反向代理。
    app.include_router(robot)
    app.include_router(robot, prefix="/api/v1", include_in_schema=False)
    return app


def _manager(request: Request) -> RuntimeManager:
    return request.app.state.runtime_manager


def _socket_manager(socket: WebSocket) -> RuntimeManager:
    return socket.app.state.runtime_manager


def _active(request: Request):
    return _manager(request).active_instance()
