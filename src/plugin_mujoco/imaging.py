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

"""图像编码只发生在采集线程，不阻塞 MuJoCo 物理循环。"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def encode_rgb_jpeg(array: np.ndarray, *, quality: int = 85) -> bytes:
    """把 RGB8 数组编码为浏览器和 Robot SDK 都能直接消费的 JPEG。"""
    value = np.asarray(array, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("RGB 图像必须是 HxWx3")
    output = io.BytesIO()
    Image.fromarray(value, mode="RGB").save(
        output,
        format="JPEG",
        quality=quality,
        optimize=False,
    )
    return output.getvalue()


def encode_depth_png16(array: np.ndarray) -> bytes:
    """把米制深度转为毫米制 16 位 PNG；0 表示无有效深度。"""
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("Depth 图像必须是 HxW")
    millimeters = np.where(
        np.isfinite(value) & (value > 0),
        np.clip(np.rint(value * 1000.0), 1, 65535),
        0,
    ).astype(np.uint16)
    output = io.BytesIO()
    Image.fromarray(millimeters, mode="I;16").save(output, format="PNG")
    return output.getvalue()
