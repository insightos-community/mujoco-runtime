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

"""浏览器可用的版本化视觉资产。"""

from .glb import VISUAL_CONTENT_VERSION, RuntimeVisualAssetStore, public_visual_id
from .scene import (
    BackendSceneVisualProvider,
    CapturedScenePose,
    SceneVisualProvider,
    ViewerCameraDescriptor,
    ViewerScene,
)

__all__ = [
    "VISUAL_CONTENT_VERSION",
    "BackendSceneVisualProvider",
    "CapturedScenePose",
    "RuntimeVisualAssetStore",
    "SceneVisualProvider",
    "ViewerCameraDescriptor",
    "ViewerScene",
    "public_visual_id",
]
