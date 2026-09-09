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

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from plugin_mujoco.streaming import EncodedFrame, SensorFrameMetadata
from plugin_mujoco.streaming.latest_frame import LatestFrameHub


def _frame(sequence: int) -> EncodedFrame:
    return EncodedFrame(
        SensorFrameMetadata(
            stream="sensor",
            sequence=sequence,
            generation=1,
            observed_at=datetime.now(timezone.utc),
            media_type="image/jpeg",
            encoding="jpeg",
            width=2,
            height=2,
            frame="world",
            sensor_id="camera.rgb",
        ),
        b"frame",
    )


def test_errored_producer_can_be_recreated_for_same_stream_key() -> None:
    """一次渲染异常后，重连必须得到新生产器，不能永久 1011 重连。"""
    hub = LatestFrameHub()
    failed = threading.Event()

    def broken(_sequence: int) -> EncodedFrame:
        failed.set()
        raise RuntimeError("render failed")

    try:
        hub.ensure("viewer:one", 30, broken)
        assert failed.wait(1)
        with pytest.raises(RuntimeError, match="render failed"):
            hub.latest("viewer:one", timeout=0.2)

        hub.ensure("viewer:one", 30, _frame)
        recovered = hub.latest("viewer:one", timeout=1)
        assert recovered.metadata.sequence == 1
        assert recovered.payload == b"frame"
    finally:
        hub.stop_all()
