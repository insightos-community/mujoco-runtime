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

from pathlib import Path

from plugin_mujoco.settings import Settings


def test_process_settings_default_to_realtime(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MUJOCO_ASSET_ROOT", str(tmp_path))
    monkeypatch.delenv("PLUGIN_MUJOCO_REALTIME", raising=False)

    assert Settings.from_env().realtime is True

    monkeypatch.setenv("PLUGIN_MUJOCO_REALTIME", "0")
    assert Settings.from_env().realtime is False
