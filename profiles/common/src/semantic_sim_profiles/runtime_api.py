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

"""robosuite / LIBERO 隔离 Runtime 的 HTTP 与二进制流入口。

本模块与 native MuJoCo Runtime 使用同一组公开路径。Profile 内部 action array
只存在于 adapter worker 中；Framework 和 Robot SDK 只能提交已经规划好的低层轨迹。
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import struct
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import uvicorn
from fastapi import APIRouter, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from PIL import Image

from semantic_sim_profiles.runtime_service import (
    ProfileRuntimeError,
    ProfileRuntimeInstance,
    ProfileRuntimeService,
    utc_iso,
)


async def _run_in_thread(function: Any, *args: Any) -> Any:
    """兼容 Python 3.8：在线程池读取 Robot 状态，避免阻塞 WebSocket 事件循环。

    LIBERO 的锁定运行环境仍使用 Python 3.8，因此不能调用 Python 3.9 才新增的
    ``asyncio.to_thread``。阻塞函数仍由默认线程池执行，调用语义与新接口一致。
    """
    return await asyncio.get_running_loop().run_in_executor(None, function, *args)


def configured_profile() -> str:
    profile_id = os.getenv("SEMANTIC_SIM_PROFILE", "").strip()
    if profile_id not in {"robosuite-1.5", "libero-robosuite-1.4"}:
        raise RuntimeError("SEMANTIC_SIM_PROFILE 必须为 robosuite-1.5 或 libero-robosuite-1.4")
    return profile_id


def create_app(service: Optional[ProfileRuntimeService] = None) -> FastAPI:
    service = service or ProfileRuntimeService(configured_profile())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        service.shutdown()

    app = FastAPI(
        title="Semantic MuJoCo Profile Runtime",
        version="0.4.0-dev",
        lifespan=lifespan,
    )
    app.state.runtime_service = service

    @app.exception_handler(ProfileRuntimeError)
    async def handle_runtime_error(_: Request, error: ProfileRuntimeError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "code": "profile_runtime_error",
                    "message": error.message,
                    "details": {},
                }
            },
        )

    @app.get("/healthz")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": "0.4.0-dev",
            "runtime_profile_id": service.profile_id,
        }

    runtime = APIRouter(prefix="/api/v1")

    @runtime.get("/runtime")
    def runtime_info(request: Request) -> Dict[str, Any]:
        return _service(request).runtime_info()

    @runtime.get("/runtime-profiles")
    def runtime_profiles(request: Request) -> List[Dict[str, Any]]:
        current = _service(request)
        profile_id = current.profile_id
        return [
            {
                "runtime_profile_id": profile_id,
                "name": (
                    "robosuite 1.5" if profile_id == "robosuite-1.5" else "LIBERO / LIBERO-Pro"
                ),
                "engine": "mujoco",
                "loader": "robosuite" if profile_id == "robosuite-1.5" else "libero",
                "api_version": "v1",
                "scene_kinds": ["robosuite"]
                if profile_id == "robosuite-1.5"
                else ["libero", "libero_pro"],
                "capabilities": current.runtime_info()["capabilities"],
                "environment": profile_id,
                "environment_ready": True,
                "available": True,
                "unavailable_reason": None,
            }
        ]

    @runtime.get("/scenes")
    def scenes(request: Request) -> List[Dict[str, Any]]:
        return _service(request).scenes()

    @runtime.post("/runtime-bundles")
    def reject_runtime_bundle() -> None:
        raise ProfileRuntimeError(
            "robosuite 和 LIBERO 使用只读原生环境，不接受 SceneDocument RuntimeBundle",
            status_code=422,
        )

    @runtime.post("/scenes/{scene_key}/instances", status_code=201)
    def start_scene(scene_key: str, payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
        return _service(request).start_scene(scene_key, payload).view()

    @runtime.get("/scene-instances/{instance_id}")
    def get_scene(instance_id: str, request: Request) -> Dict[str, Any]:
        return _service(request).instance(instance_id).view()

    @runtime.post("/scene-instances/{instance_id}/pause")
    def pause_scene(instance_id: str, request: Request) -> Dict[str, Any]:
        return _service(request).instance(instance_id).pause()

    @runtime.post("/scene-instances/{instance_id}/resume")
    def resume_scene(instance_id: str, request: Request) -> Dict[str, Any]:
        return _service(request).instance(instance_id).resume()

    @runtime.post("/scene-instances/{instance_id}/step")
    def step_scene(instance_id: str, payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
        return _service(request).instance(instance_id).step_once(int(payload.get("steps", 1)))

    @runtime.post("/scene-instances/{instance_id}/reset")
    def reset_scene(instance_id: str, request: Request) -> Dict[str, Any]:
        return _service(request).instance(instance_id).reset()

    @runtime.post("/scene-instances/{instance_id}/stop")
    def stop_scene(instance_id: str, request: Request) -> Dict[str, Any]:
        return _service(request).instance(instance_id).stop()

    @runtime.get("/scene-instances/{instance_id}/snapshot")
    def scene_snapshot(instance_id: str, request: Request) -> Dict[str, Any]:
        return _service(request).instance(instance_id).snapshot()

    @runtime.get("/scene-instances/{instance_id}/evaluation")
    def scene_evaluation(instance_id: str, request: Request) -> Dict[str, Any]:
        return _service(request).instance(instance_id).evaluation()

    @runtime.get("/scene-instances/{instance_id}/robots")
    def scene_robots(instance_id: str, request: Request) -> List[Dict[str, Any]]:
        instance = _service(request).instance(instance_id)
        return [instance.robot_profile(instance.robot_id)]

    @runtime.get("/scene-instances/{instance_id}/viewer-scene")
    def viewer_scene(instance_id: str, request: Request) -> Dict[str, Any]:
        return _service(request).instance(instance_id).viewer_scene()

    @runtime.get("/scene-instances/{instance_id}/viewer-scene/content")
    def viewer_scene_content(instance_id: str, request: Request) -> Response:
        content, revision = _service(request).instance(instance_id).viewer_scene_content()
        return Response(
            content=content,
            media_type="model/gltf-binary",
            headers={
                "Cache-Control": "private, max-age=31536000, immutable",
                "X-Semantic-Scene-Revision": revision,
            },
        )

    @runtime.websocket("/scene-instances/{instance_id}/pose-stream")
    async def pose_stream(instance_id: str, socket: WebSocket) -> None:
        await socket.accept()
        sequence = 0
        try:
            instance = _socket_service(socket).instance(instance_id)
            generation = instance.generation
            while True:
                metadata, payload = await _run_in_thread(instance.scene_pose_frame, sequence)
                if metadata["generation"] != generation:
                    await socket.close(code=1000)
                    return
                sequence = int(metadata["sequence"])
                await socket.send_bytes(_packet(metadata, payload))
        except WebSocketDisconnect:
            return
        except ProfileRuntimeError:
            await socket.close(code=1011)


    app.include_router(runtime)

    robot = APIRouter()

    @robot.get("/robots/{robot_id}/profile")
    def robot_profile(robot_id: str, request: Request) -> Dict[str, Any]:
        return _service(request).active().robot_profile(robot_id)

    @robot.get("/robots/{robot_id}/state")
    def robot_state(robot_id: str, request: Request) -> Dict[str, Any]:
        return _service(request).active().robot_state(robot_id)

    @robot.websocket("/robots/{robot_id}/state/stream")
    async def robot_state_stream(robot_id: str, socket: WebSocket) -> None:
        """使用和 native Runtime 相同的二进制 packet 输出 Franka 状态。"""

        await socket.accept()
        sequence = 0
        try:
            while True:
                instance = _socket_service(socket).active()
                state = await _run_in_thread(instance.robot_state, robot_id)
                sequence += 1
                metadata = {
                    "stream": "robot_state",
                    "sequence": sequence,
                    "generation": state["generation"],
                    "sim_time": instance.view()["sim_time"],
                    "observed_at": state["observed_at"],
                    "frame_id": state["base_pose"]["frame_id"],
                    "robot_id": state["robot_id"],
                    "media_type": "application/json",
                    "encoding": "json",
                }
                payload = json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
                await socket.send_bytes(_packet(metadata, payload))
                await asyncio.sleep(instance.control_period_s)
        except WebSocketDisconnect:
            return
        except ProfileRuntimeError:
            await socket.close(code=1011)

    @robot.post("/robots/{robot_id}/commands", status_code=202)
    def submit_command(robot_id: str, payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
        return _service(request).active().submit_command(robot_id, payload)

    @robot.get("/robots/{robot_id}/commands/{command_id}")
    def command(robot_id: str, command_id: str, request: Request) -> Dict[str, Any]:
        return _service(request).active().command(robot_id, command_id)

    @robot.post("/robots/{robot_id}/commands/{command_id}/stop")
    def stop_command(robot_id: str, command_id: str, request: Request) -> Dict[str, Any]:
        return _service(request).active().stop_command(robot_id, command_id)

    @robot.post("/robots/{robot_id}/hold")
    def hold_robot(robot_id: str, payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
        return _service(request).active().hold(robot_id, int(payload.get("scene_generation", 0)))

    @robot.get("/robots/{robot_id}/sensors")
    def sensors(robot_id: str, request: Request) -> List[Dict[str, Any]]:
        return _service(request).active().sensor_descriptors(robot_id)

    @robot.get("/robots/{robot_id}/sensors/{sensor_id}/frames/latest")
    def latest_sensor(robot_id: str, sensor_id: str, request: Request) -> Dict[str, Any]:
        instance = _service(request).active()
        metadata, payload, structured = _encoded_sensor(instance, robot_id, sensor_id)
        return {
            "sensor_id": sensor_id,
            "kind": metadata["kind"],
            "observed_at": metadata["observed_at"],
            "sequence": metadata["sequence"],
            "generation": metadata["generation"],
            "frame": metadata["frame"],
            "media_type": metadata["media_type"],
            "encoding": metadata["encoding"],
            "width": metadata["width"],
            "height": metadata["height"],
            "payload_size": len(payload),
            "data": structured,
        }

    @robot.get("/robots/{robot_id}/sensors/{sensor_id}/frames/latest/content")
    def latest_sensor_content(robot_id: str, sensor_id: str, request: Request) -> Response:
        metadata, payload, _ = _encoded_sensor(_service(request).active(), robot_id, sensor_id)
        return Response(
            content=payload,
            media_type=metadata["media_type"],
            headers=_frame_headers(metadata),
        )

    @robot.websocket("/robots/{robot_id}/sensors/{sensor_id}/stream")
    async def sensor_stream(robot_id: str, sensor_id: str, socket: WebSocket) -> None:
        await socket.accept()
        try:
            while True:
                instance = _socket_service(socket).active()
                metadata, payload, _ = _encoded_sensor(instance, robot_id, sensor_id)
                packet_metadata = {
                    "stream": "sensor",
                    "sequence": metadata["sequence"],
                    "generation": metadata["generation"],
                    "observed_at": metadata["observed_at"],
                    "media_type": metadata["media_type"],
                    "encoding": metadata["encoding"],
                    "width": metadata["width"] or 0,
                    "height": metadata["height"] or 0,
                    "frame": metadata["frame"],
                    "sensor_id": sensor_id,
                    "viewer_session_id": None,
                }
                await socket.send_bytes(_packet(packet_metadata, payload))
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            return
        except ProfileRuntimeError:
            await socket.close(code=1011)

    app.include_router(robot)
    app.include_router(robot, prefix="/api/v1", include_in_schema=False)
    return app


def _service(request: Request) -> ProfileRuntimeService:
    return request.app.state.runtime_service


def _socket_service(socket: WebSocket) -> ProfileRuntimeService:
    return socket.app.state.runtime_service


def _encoded_sensor(
    instance: ProfileRuntimeInstance, robot_id: str, sensor_id: str
) -> Tuple[Dict[str, Any], bytes, Optional[Dict[str, Any]]]:
    kind, value, sequence = instance.sensor_payload(robot_id, sensor_id)
    descriptor = next(
        item for item in instance.sensor_descriptors(robot_id) if item["sensor_id"] == sensor_id
    )
    structured = None
    if kind == "rgb":
        payload = _encode_rgb(np.asarray(value), 85)
        media_type = "image/jpeg"
        encoding = "jpeg"
    elif kind == "depth":
        # robosuite / LIBERO 的深度缓冲在 profile 中保持 float32；
        # 未经过相机近远平面换算前不能冒充毫米深度 PNG。
        depth = np.ascontiguousarray(np.asarray(value, dtype="<f4").squeeze())
        payload = depth.tobytes(order="C")
        media_type = "application/octet-stream"
        encoding = "float32-le"
    else:
        structured = dict(value)
        payload = json.dumps(structured, ensure_ascii=False).encode("utf-8")
        media_type = "application/json"
        encoding = "json"
    array = np.asarray(value) if hasattr(value, "shape") else None
    return (
        {
            "sensor_id": sensor_id,
            "kind": kind,
            "sequence": sequence,
            "generation": instance.generation,
            "observed_at": utc_iso(),
            "frame": descriptor["frame_id"],
            "media_type": media_type,
            "encoding": encoding,
            "width": int(array.shape[1]) if array is not None and array.ndim >= 2 else None,
            "height": int(array.shape[0]) if array is not None and array.ndim >= 2 else None,
        },
        payload,
        structured,
    )


def _encode_rgb(array: np.ndarray, quality: int) -> bytes:
    value = np.asarray(array)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    value = np.clip(value[..., :3], 0, 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(value, mode="RGB").save(
        output, format="JPEG", quality=max(30, min(quality, 95))
    )
    return output.getvalue()


def _packet(metadata: Dict[str, Any], payload: bytes) -> bytes:
    header = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(header)) + header + payload


def _frame_headers(metadata: Dict[str, Any]) -> Dict[str, str]:
    return {
        "X-Semantic-Sequence": str(metadata["sequence"]),
        "X-Semantic-Generation": str(metadata["generation"]),
        "X-Semantic-Encoding": str(metadata["encoding"]),
        "X-Semantic-Frame": str(metadata["frame"]),
        "X-Semantic-Width": str(metadata["width"] or 0),
        "X-Semantic-Height": str(metadata["height"] or 0),
        "X-Semantic-Observed-At": str(metadata["observed_at"]),
    }


def main() -> None:
    host = os.getenv("PLUGIN_MUJOCO_HOST", "127.0.0.1")
    port = int(os.getenv("PLUGIN_MUJOCO_PORT", "8091"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
