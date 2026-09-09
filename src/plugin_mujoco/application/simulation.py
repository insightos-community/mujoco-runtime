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

"""单进程、单活动场景的 Runtime Manager。

模型加载在后台线程完成，因此 API 可以立即返回 starting 并持续查询进度。
物理循环由 RuntimeInstance 独占，Manager 只负责记录和生命周期编排。
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path

from plugin_mujoco.compiler import RuntimeBundle, RuntimeBundleResult, compile_scene_document
from plugin_mujoco.errors import ConflictError, NotFoundError, RuntimeErrorBase
from plugin_mujoco.loaders import RuntimeProfileRegistry
from plugin_mujoco.models import (
    RuntimeInfo,
    RuntimeProfile,
    SceneInstance,
    SceneStartRequest,
    SceneState,
    utc_now,
)
from plugin_mujoco.runtime.instance import RuntimeInstance
from plugin_mujoco.scene import SceneCatalog
from plugin_mujoco.settings import Settings


class RuntimeManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.catalog = SceneCatalog(settings.asset_root)
        repository_root = Path(__file__).resolve().parents[3]
        self.profiles = RuntimeProfileRegistry(settings, repository_root)
        self.authoring_root = (
            settings.authoring_root or Path(tempfile.gettempdir()) / "semantic-plugin-mujoco-builds"
        ).resolve()
        self._lock = threading.RLock()
        self._instances: dict[str, RuntimeInstance] = {}
        self._records: dict[str, SceneInstance] = {}
        self._active_instance_id: str | None = None
        self._requests: dict[str, tuple[str, str]] = {}
        self._build_requests: dict[str, tuple[str, RuntimeBundleResult]] = {}
        self._load_threads: dict[str, threading.Thread] = {}
        self._load_cancel: dict[str, threading.Event] = {}
        self._restore_scene_builds()

    def info(self) -> RuntimeInfo:
        with self._lock:
            active = (
                self._records.get(self._active_instance_id) if self._active_instance_id else None
            )
            if active and active.state == SceneState.FAILED:
                state = "failed"
            elif active and active.state != SceneState.STOPPED:
                state = "busy"
            else:
                state = "ready"
            native = self.profiles.get("native-mujoco")
            return RuntimeInfo(
                state=state,
                version="0.4.0-dev",
                active_instance_id=(
                    active.instance_id if active and active.state != SceneState.STOPPED else None
                ),
                capabilities=native.capabilities,
                asset_root=str(self.settings.asset_root),
            )

    def runtime_profiles(self) -> list[RuntimeProfile]:
        return self.profiles.list()

    def start(self, scene_key: str, request: SceneStartRequest) -> SceneInstance:
        """登记 starting 实例并后台加载，避免大型模型阻塞请求线程。"""
        effective_backend = (
            self.settings.render_backend
            if request.render_backend == "auto"
            else request.render_backend
        )
        if effective_backend != self.settings.render_backend:
            raise ConflictError(
                "请求的渲染后端与 Runtime 进程配置不一致",
                details={
                    "configured": self.settings.render_backend,
                    "requested": effective_backend,
                },
            )
        self.profiles.get(request.runtime_profile_id)
        if request.runtime_bundle_id:
            with self._lock:
                registered = self._build_requests.get(request.runtime_bundle_id)
            if registered is None:
                raise NotFoundError(f"RuntimeBundle 不存在: {request.runtime_bundle_id}")
            result = registered[1]
            if (
                not result.valid
                or result.scene_key != scene_key
                or result.runtime_profile_id != request.runtime_profile_id
            ):
                raise ConflictError(
                    "RuntimeBundle 与启动场景或 Profile 不一致",
                    details={"runtime_bundle_id": request.runtime_bundle_id},
                )
        request = request.model_copy(update={"render_backend": effective_backend})
        fingerprint = _fingerprint({"scene_key": scene_key, **request.model_dump(mode="json")})
        with self._lock:
            previous = self._requests.get(request.request_id)
            if previous:
                previous_fingerprint, instance_id = previous
                if previous_fingerprint != fingerprint:
                    raise ConflictError("相同 request_id 的启动输入不同")
                return self.view(instance_id)
            active = self._active_record()
            if active and active.state != SceneState.STOPPED:
                raise ConflictError(
                    "Runtime 已有活动场景",
                    details={"instance_id": active.instance_id, "state": active.state.value},
                )
            now = utc_now()
            record = SceneInstance(
                instance_id=str(uuid.uuid4()),
                scene_key=scene_key,
                layout=request.layout,
                seed=request.seed,
                headless=request.headless,
                render_backend=request.render_backend,
                generation=1,
                state=SceneState.STARTING,
                request_id=request.request_id,
                runtime_profile_id=request.runtime_profile_id,
                runtime_bundle_id=request.runtime_bundle_id,
                progress=0.05,
                progress_message="启动请求已登记",
                created_at=now,
                updated_at=now,
            )
            self._records[record.instance_id] = record
            self._active_instance_id = record.instance_id
            self._requests[request.request_id] = (fingerprint, record.instance_id)
            cancel = threading.Event()
            thread = threading.Thread(
                target=self._load_instance,
                args=(record.instance_id, scene_key, request, cancel),
                name=f"mujoco-loader-{record.instance_id}",
                daemon=True,
            )
            self._load_cancel[record.instance_id] = cancel
            self._load_threads[record.instance_id] = thread
            thread.start()
            return record.model_copy(deep=True)

    def _load_instance(
        self, instance_id: str, scene_key: str, request: SceneStartRequest, cancel: threading.Event
    ) -> None:
        components = None
        try:
            self._progress(instance_id, 0.2, "正在读取场景和资产")
            definition = self.catalog.load(scene_key, request.layout)
            if cancel.is_set():
                self._finish_cancelled_start(instance_id)
                return
            self._progress(instance_id, 0.45, "正在初始化物理模型")
            components = self.profiles.load(request.runtime_profile_id, definition, request)
            if cancel.is_set():
                components.physics.close()
                self._finish_cancelled_start(instance_id)
                return
            with self._lock:
                record = self._records[instance_id]
                record.progress = 1.0
                record.progress_message = "场景已加载"
                record.updated_at = utc_now()
                instance = RuntimeInstance(record, components, realtime=self.settings.realtime)
                self._instances[instance_id] = instance
            instance.start()
        except Exception as exc:
            if components is not None:
                with suppress(Exception):
                    components.physics.close()
            with self._lock:
                record = self._records[instance_id]
                record.state = SceneState.FAILED
                record.progress_message = "场景加载失败"
                if isinstance(exc, RuntimeErrorBase):
                    reason = str(exc.details.get("reason") or "").strip()
                    suffix = ": " + reason if reason else ""
                    record.failure_reason = exc.message + suffix
                else:
                    record.failure_reason = str(exc)
                record.updated_at = utc_now()
        finally:
            with self._lock:
                self._load_threads.pop(instance_id, None)
                self._load_cancel.pop(instance_id, None)

    def _progress(self, instance_id: str, value: float, message: str) -> None:
        with self._lock:
            record = self._records[instance_id]
            record.progress = value
            record.progress_message = message
            record.updated_at = utc_now()

    def _finish_cancelled_start(self, instance_id: str) -> None:
        with self._lock:
            record = self._records[instance_id]
            record.state = SceneState.STOPPED
            record.progress_message = "场景启动已取消"
            record.updated_at = utc_now()

    def wait_ready(self, instance_id: str, timeout: float = 10.0) -> SceneInstance:
        """测试和本地 CLI 使用的等待函数；HTTP 客户端应轮询实例状态。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.view(instance_id)
            if record.state == SceneState.RUNNING:
                return record
            if record.state in {SceneState.FAILED, SceneState.STOPPED}:
                return record
            time.sleep(0.005)
        raise ConflictError("等待场景启动超时", details={"instance_id": instance_id})

    def stop(self, instance_id: str, timeout: float = 10.0) -> SceneInstance:
        instance: RuntimeInstance | None = None
        with self._lock:
            record = self._records.get(instance_id)
            if record is None:
                raise NotFoundError(f"场景实例不存在: {instance_id}")
            if record.state == SceneState.STARTING:
                record.state = SceneState.STOPPING
                record.updated_at = utc_now()
                cancel = self._load_cancel.get(instance_id)
                thread = self._load_threads.get(instance_id)
                if cancel is not None:
                    cancel.set()
            else:
                cancel = None
                thread = None
                instance = self._instances.get(instance_id)
        if cancel is not None:
            if thread is not None:
                thread.join(timeout)
                if thread.is_alive():
                    raise ConflictError("场景加载线程未能停止")
            return self.view(instance_id)
        if instance is None:
            if record.state == SceneState.STOPPED:
                return record.model_copy(deep=True)
            if record.state == SceneState.FAILED:
                # 加载阶段失败时尚未创建 RuntimeInstance，但仍必须允许调用方通过
                # stop 明确结束该失败实例，之后才能启动另一场景。
                with self._lock:
                    record.state = SceneState.STOPPED
                    record.updated_at = utc_now()
                return record.model_copy(deep=True)
            raise ConflictError("场景实例不可操作", details={"state": record.state.value})
        return instance.stop(timeout=min(timeout, 5.0))

    def register_runtime_bundle(self, request: RuntimeBundle) -> RuntimeBundleResult:
        fingerprint = _fingerprint(request.model_dump(mode="json"))
        with self._lock:
            previous = self._build_requests.get(request.build_id)
            if previous:
                previous_fingerprint, result = previous
                if previous_fingerprint != fingerprint:
                    raise ConflictError("相同 runtime_bundle_id 的场景构建输入不同")
                return result.model_copy(deep=True)
        definition, result = compile_scene_document(
            self.settings.asset_root, self.authoring_root, request
        )
        with self._lock:
            if definition is not None:
                self.catalog.register(definition)
            self._build_requests[request.build_id] = (fingerprint, result)
        return result.model_copy(deep=True)

    def _restore_scene_builds(self) -> None:
        self.authoring_root.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.authoring_root.glob("*/build-request.json")):
            try:
                request = RuntimeBundle.model_validate_json(path.read_text(encoding="utf-8"))
                self.register_runtime_bundle(request)
            except (OSError, ValueError, RuntimeErrorBase):
                continue

    def active_instance(self) -> RuntimeInstance:
        with self._lock:
            record = self._active_record()
            if record is None or record.state in {SceneState.STOPPED, SceneState.FAILED}:
                raise NotFoundError("当前没有活动场景")
            if record.state in {SceneState.STARTING, SceneState.STOPPING}:
                raise ConflictError("活动场景尚不可操作", details={"state": record.state.value})
            return self._instances[record.instance_id]

    def view(self, instance_id: str) -> SceneInstance:
        with self._lock:
            instance = self._instances.get(instance_id)
            if instance is not None:
                return instance.view()
            record = self._records.get(instance_id)
            if record is None:
                raise NotFoundError(f"场景实例不存在: {instance_id}")
            return record.model_copy(deep=True)

    def get(self, instance_id: str) -> RuntimeInstance:
        with self._lock:
            instance = self._instances.get(instance_id)
            if instance is not None:
                return instance
            record = self._records.get(instance_id)
            if record is not None:
                raise ConflictError(
                    "场景实例不可操作",
                    details={
                        "instance_id": instance_id,
                        "state": record.state.value,
                        "reason": record.failure_reason,
                    },
                )
            raise NotFoundError(f"场景实例不存在: {instance_id}")

    def shutdown(self) -> None:
        with self._lock:
            loading = [
                (instance_id, cancel, self._load_threads.get(instance_id))
                for instance_id, cancel in self._load_cancel.items()
            ]
            instances = list(self._instances.values())
        for _, cancel, _ in loading:
            cancel.set()
        for _, _, thread in loading:
            if thread is not None:
                thread.join(timeout=10.0)
        for instance in instances:
            if instance.record.state != SceneState.STOPPED:
                try:
                    instance.stop()
                except ConflictError:
                    instance.fail("Runtime 进程退出时未能安全停止")

    def _active_record(self) -> SceneInstance | None:
        if self._active_instance_id is None:
            return None
        return self._records.get(self._active_instance_id)


def _fingerprint(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
