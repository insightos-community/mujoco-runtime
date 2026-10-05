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

"""所有 OpenGL/EGL 渲染调用共用的单线程执行器。

MuJoCo Renderer 内部持有线程相关的 OpenGL Context。互斥锁只能避免并发，不能
允许 Context 在多个线程间轮流使用，因此 Viewer 与各相机流必须统一到同一线程。
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _RenderCall:
    callback: Callable[[], Any]
    completed: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


class RenderExecutor:
    """串行执行渲染和 Renderer 释放，保证 Context 永不跨线程。"""

    def __init__(self, name: str) -> None:
        self._queue: queue.Queue[_RenderCall | None] = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name=f"mujoco-render-{name}", daemon=True
        )
        self._thread.start()

    def call(self, callback: Callable[[], Any], *, timeout: float = 10.0) -> Any:
        if threading.current_thread() is self._thread:
            return callback()
        if self._closed:
            raise RuntimeError("渲染执行器已关闭")
        call = _RenderCall(callback)
        self._queue.put(call)
        if not call.completed.wait(timeout):
            raise TimeoutError("渲染线程未在限定时间内响应")
        if call.error is not None:
            raise call.error
        return call.result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                raise TimeoutError("渲染线程未能停止")

    def _run(self) -> None:
        while True:
            call = self._queue.get()
            if call is None:
                return
            try:
                call.result = call.callback()
            except BaseException as exc:
                call.error = exc
            finally:
                call.completed.set()
