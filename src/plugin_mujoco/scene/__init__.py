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

"""场景目录与运行时 XML 拼装。"""

from .builder import RuntimeSceneBuild, build_runtime_scene
from .catalog import AssetSpec, RobotSpec, SceneCatalog, SceneDefinition, SensorSpec

__all__ = [
    "AssetSpec",
    "RobotSpec",
    "RuntimeSceneBuild",
    "SceneCatalog",
    "SceneDefinition",
    "SensorSpec",
    "build_runtime_scene",
]
