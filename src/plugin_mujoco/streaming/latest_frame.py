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

"""有界的共享最新帧生产器。

每个流只有一个后台生产线程。客户端消费慢时直接读取更新后的最新帧，不会
积压队列，也不会把 WebSocket 发送速度传导到物理循环。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from plugin_mujoco.streaming.protocol import EncodedFrame


@dataclass
class _Producer:
    stop: threading.Event
    condition: threading.Condition
    source: Callable[[int], EncodedFrame]
    interval: float
    frame: EncodedFrame | None = None
    error: Exception | None = None
    sequence: int = 0
    thread: threading.Thread | None = None


class LatestFrameHub:
    """按流键复用生产器；缓存始终只保留一帧。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._producers: dict[str, _Producer] = {}

    def ensure(self, key: str, fps: float, source: Callable[[int], EncodedFrame]) -> None:
        with self._lock:
            current = self._producers.get(key)
            if current is not None:
                thread_alive = current.thread is not None and current.thread.is_alive()
                if current.error is None and thread_alive and not current.stop.is_set():
                    return
                # 一次渲染异常不能永久毒化流键；重连时替换终止的生产器。
                self._producers.pop(key, None)
                current.stop.set()
                with current.condition:
                    current.condition.notify_all()
            producer = _Producer(threading.Event(), threading.Condition(), source, 1.0 / fps)
            producer.thread = threading.Thread(
                target=self._run, args=(producer,), name=f"frame-{key}", daemon=True
            )
            self._producers[key] = producer
            producer.thread.start()

    def latest(self, key: str, *, after_sequence: int = 0, timeout: float = 2.0) -> EncodedFrame:
        with self._lock:
            producer = self._producers[key]
        deadline = time.monotonic() + timeout
        with producer.condition:
            while producer.frame is None or producer.frame.metadata.sequence <= after_sequence:
                if producer.error is not None:
                    raise producer.error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if producer.frame is not None:
                        return producer.frame
                    raise TimeoutError(f"等待流首帧超时: {key}")
                producer.condition.wait(remaining)
            return producer.frame

    def remove(self, key: str) -> None:
        with self._lock:
            producer = self._producers.pop(key, None)
        if producer is not None:
            producer.stop.set()
            if producer.thread is not None and producer.thread is not threading.current_thread():
                # 原生 EGL Renderer 必须先等生产线程退出，再由 RenderExecutor
                # 所属线程释放。旧实现两秒后直接继续 close，慢帧尚在 OpenGL
                # 调用中时会形成释放竞态，进程可能在退出/重置阶段段错误。
                producer.thread.join(timeout=12.0)
                if producer.thread.is_alive():
                    raise TimeoutError(f"流生产器未能安全停止: {key}")

    def stop_all(self) -> None:
        with self._lock:
            keys = list(self._producers)
        for key in keys:
            self.remove(key)

    @staticmethod
    def _run(producer: _Producer) -> None:
        while not producer.stop.is_set():
            started = time.monotonic()
            try:
                next_sequence = producer.sequence + 1
                frame = producer.source(next_sequence)
                with producer.condition:
                    producer.sequence = next_sequence
                    producer.frame = frame
                    producer.condition.notify_all()
            except Exception as exc:
                with producer.condition:
                    producer.error = exc
                    producer.condition.notify_all()
                return
            producer.stop.wait(max(0.0, producer.interval - (time.monotonic() - started)))
