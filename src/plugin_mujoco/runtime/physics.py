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
