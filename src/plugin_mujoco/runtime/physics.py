"""物理生命周期适配器。"""

from __future__ import annotations

from typing import Any


class BackendPhysicsRuntime:
    """限制上层只能访问物理步进和资源生命周期。"""

    def __init__(self, backend: Any, *, close_callback: Any | None = None) -> None:
        self._backend = backend
        self._close_callback = close_callback or backend.close

    @property
    def sim_time(self) -> float:
        return self._backend.sim_time

    @property
    def timestep(self) -> float:
        return self._backend.timestep

    def step(self) -> None:
        self._backend.step()

    def reset(self) -> None:
        self._backend.reset()

    def close(self) -> None:
        self._close_callback()
